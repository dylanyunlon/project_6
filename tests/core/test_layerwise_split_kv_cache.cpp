/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Test suite for layerwise split KV cache sharding (PR #2260).
// Adapted for Iluvatar BI-V100 — verified hardware:
//   4× BI-V100, Bus-Id 4B-4E, NUMA 1, flat PIX topology (all pairs PIX).
//   32768 MiB HBM each, IX-ML 3.2.3, Driver 3.2.1, CUDA 10.2.
//
// Build: link against gtest, gflags, glog, torch, and the new source files.

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <unordered_set>
#include <vector>

#include "core/config/parallel_config_layerwise.h"
#include "core/distributed_runtime/layerwise_split_engine_ext.h"
#include "core/distributed_runtime/layerwise_split_master.h"
#include "core/framework/kv_cache/kv_cache_estimation_layerwise.h"
#include "core/framework/kv_cache/layerwise_split_layout.h"
#include "core/framework/parallel_state/mapping_ilu.h"

namespace xllm {
namespace {

// Matches verified hardware: 4 BI-V100 cards.
constexpr int32_t kBIV100WorldSize = 4;

// ---------------------------------------------------------------------------
// TC-01  Layerwise KV allocation correctness
// ---------------------------------------------------------------------------
// Precondition: 32-layer model, layers 0-15 dense (8 KV heads each),
//               layers 16-31 MoE (2 KV heads each), 4 TP ranks.
// Criteria: per-rank allocation matches layout; unassigned → zero KV;
//           total KV = sum of all per-layer allocations.
TEST(LayerwiseSplitKV, TC01_AllocationCorrectness) {
  const int64_t num_layers  = 32;
  const int64_t dense_heads = 8;   // layers 0-15: ≥ world_size → all ranks
  const int64_t moe_heads   = 2;   // layers 16-31: < world_size → subset

  std::vector<int64_t> per_layer_heads(num_layers);
  for (int64_t i = 0; i < 16; ++i) per_layer_heads[i] = dense_heads;
  for (int64_t i = 16; i < 32; ++i) per_layer_heads[i] = moe_heads;

  auto layout = compute_ilu_layerwise_layout(
      num_layers, per_layer_heads, kBIV100WorldSize);

  ASSERT_EQ(layout.num_layers(), num_layers);

  // Dense layers: all 4 ranks, each with 8/4 = 2 heads.
  for (int64_t lid = 0; lid < 16; ++lid) {
    const auto& spec = layout.layer_spec(lid);
    EXPECT_EQ(static_cast<int32_t>(spec.assigned_ranks.size()),
              kBIV100WorldSize);
    for (int32_t r = 0; r < kBIV100WorldSize; ++r) {
      EXPECT_EQ(layout.heads_for_rank(r, lid),
                dense_heads / kBIV100WorldSize);
    }
  }

  // MoE layers: 2 heads → exactly 2 ranks assigned per layer.
  for (int64_t lid = 16; lid < 32; ++lid) {
    const auto& spec = layout.layer_spec(lid);
    EXPECT_EQ(static_cast<int64_t>(spec.assigned_ranks.size()), moe_heads);
    EXPECT_EQ(spec.total_heads(), moe_heads);
    for (int32_t r = 0; r < kBIV100WorldSize; ++r) {
      if (!layout.rank_owns_layer(r, lid)) {
        EXPECT_EQ(layout.heads_for_rank(r, lid), 0);
      }
    }
  }

  // Total heads across all specs == original.
  int64_t total = 0;
  for (int64_t lid = 0; lid < num_layers; ++lid)
    total += layout.layer_spec(lid).total_heads();
  EXPECT_EQ(total, 16 * dense_heads + 16 * moe_heads);
}

// ---------------------------------------------------------------------------
// TC-02  Memory estimation accuracy
// ---------------------------------------------------------------------------
// Criteria: layerwise peak per-rank ≤ uniform; estimation > 0.
TEST(LayerwiseSplitKV, TC02_MemoryEstimation) {
  const int64_t num_layers  = 32;
  const int64_t dense_heads = 8;
  const int64_t moe_heads   = 2;
  const int64_t n_blocks    = 256;
  const int64_t block_size  = 16;
  const int64_t head_dim    = 128;
  const int64_t max_tokens  = 4096;
  const int dtype_enum      = 15;  // bfloat16

  std::vector<int64_t> per_layer_heads(num_layers);
  for (int64_t i = 0; i < 16; ++i) per_layer_heads[i] = dense_heads;
  for (int64_t i = 16; i < 32; ++i) per_layer_heads[i] = moe_heads;

  auto layout = compute_ilu_layerwise_layout(
      num_layers, per_layer_heads, kBIV100WorldSize);

  auto est = estimate_layerwise_kv_memory(
      layout, n_blocks, block_size, head_dim, max_tokens, dtype_enum,
      kBIV100WorldSize);

  EXPECT_LE(est.peak_per_rank_bytes, est.uniform_per_rank_bytes);
  EXPECT_GT(est.peak_per_rank_bytes, 0);
  EXPECT_GT(est.average_per_rank_bytes, 0);

  // Per-rank breakdown should have exactly 4 entries.
  EXPECT_EQ(static_cast<int32_t>(est.per_rank_bytes.size()),
            kBIV100WorldSize);
}

// ---------------------------------------------------------------------------
// TC-03  ILU topology-aware mapping (flat PIX)
// ---------------------------------------------------------------------------
// Verified precondition: 4 BI-V100 cards, all PIX (ixsmi topo -m).
// Criteria: all layers assigned; MoE layers round-robin across ranks
//           (no grouping since topology is flat); no rank oversubscribed.
TEST(LayerwiseSplitKV, TC03_IluTopologyMapping) {
  const int64_t num_layers = 32;
  std::vector<int64_t> per_layer_heads(num_layers);
  for (int64_t i = 0; i < 16; ++i) per_layer_heads[i] = 8;
  for (int64_t i = 16; i < 32; ++i) per_layer_heads[i] = 2;

  auto layout = compute_ilu_layerwise_layout(
      num_layers, per_layer_heads, kBIV100WorldSize,
      IluTopoKind::kFlatPIX);

  ASSERT_EQ(layout.num_layers(), num_layers);

  // All layers must have at least one assigned rank.
  for (int64_t lid = 0; lid < num_layers; ++lid) {
    EXPECT_FALSE(layout.layer_spec(lid).assigned_ranks.empty());
  }

  // Flat PIX: MoE layers round-robin, so across all 16 MoE layers
  // each rank should appear roughly equally (within ±1).
  std::vector<int32_t> moe_rank_count(kBIV100WorldSize, 0);
  for (int64_t lid = 16; lid < 32; ++lid) {
    for (auto r : layout.layer_spec(lid).assigned_ranks) {
      EXPECT_GE(r, 0);
      EXPECT_LT(r, kBIV100WorldSize);
      moe_rank_count[r]++;
    }
  }
  int32_t min_count = *std::min_element(moe_rank_count.begin(),
                                         moe_rank_count.end());
  int32_t max_count = *std::max_element(moe_rank_count.begin(),
                                         moe_rank_count.end());
  // With 16 MoE layers × 2 ranks each = 32 assignments over 4 ranks → ~8.
  // Round-robin should be exactly balanced or differ by at most 1.
  EXPECT_LE(max_count - min_count, 1)
      << "MoE layer assignments not balanced across flat PIX topology";
}

// ---------------------------------------------------------------------------
// TC-04  Distributed engine layout propagation
// ---------------------------------------------------------------------------
// Criteria: maybe_compute_layerwise_layout returns layout when enabled;
//           every rank is assigned at least one layer.
TEST(LayerwiseSplitKV, TC04_EnginePropagation) {
  FLAGS_enable_layerwise_split = true;

  std::vector<int64_t> heads(32);
  for (int64_t i = 0; i < 16; ++i) heads[i] = 8;
  for (int64_t i = 16; i < 32; ++i) heads[i] = 2;

  auto layout = maybe_compute_layerwise_layout(
      32, heads, kBIV100WorldSize);
  ASSERT_TRUE(layout.has_value());
  EXPECT_EQ(layout->num_layers(), 32);

  for (int32_t r = 0; r < kBIV100WorldSize; ++r) {
    EXPECT_GT(layout->layers_on_rank(r), 0);
  }

  FLAGS_enable_layerwise_split = false;
}

// ---------------------------------------------------------------------------
// TC-05  Worker KV shard application
// ---------------------------------------------------------------------------
// Criteria: assigned layers → heads > 0; unassigned → heads == 0.
TEST(LayerwiseSplitKV, TC05_WorkerShardApplication) {
  const int32_t test_rank = 2;
  const int64_t num_layers = 32;

  std::vector<int64_t> heads(num_layers);
  for (int64_t i = 0; i < 16; ++i) heads[i] = 8;
  for (int64_t i = 16; i < 32; ++i) heads[i] = 2;

  auto layout = compute_ilu_layerwise_layout(
      num_layers, heads, kBIV100WorldSize);

  // Dense layers: all ranks assigned, including rank 2.
  for (int64_t lid = 0; lid < 16; ++lid) {
    EXPECT_TRUE(layout.rank_owns_layer(test_rank, lid));
    EXPECT_GT(layout.heads_for_rank(test_rank, lid), 0);
  }

  // MoE layers: some assigned, some not.  Consistency check.
  for (int64_t lid = 16; lid < 32; ++lid) {
    int64_t h = layout.heads_for_rank(test_rank, lid);
    if (layout.rank_owns_layer(test_rank, lid)) {
      EXPECT_GT(h, 0);
    } else {
      EXPECT_EQ(h, 0);
    }
  }
}

// ---------------------------------------------------------------------------
// TC-06  Fallback to uniform when disabled
// ---------------------------------------------------------------------------
TEST(LayerwiseSplitKV, TC06_FallbackUniform) {
  FLAGS_enable_layerwise_split = false;

  std::vector<int64_t> heads(32, 8);
  auto layout = maybe_compute_layerwise_layout(
      32, heads, kBIV100WorldSize);
  EXPECT_FALSE(layout.has_value());
}

// ---------------------------------------------------------------------------
// TC-07  Speculative engine with layerwise KV
// ---------------------------------------------------------------------------
// Criteria: layout computed for both target (heterogeneous) and draft
//           (homogeneous) models without crash.
TEST(LayerwiseSplitKV, TC07_SpeculativeEngine) {
  FLAGS_enable_layerwise_split = true;

  // Target model: 60 layers, first 30 dense, rest MoE.
  std::vector<int64_t> target_heads(60);
  for (int64_t i = 0; i < 30; ++i) target_heads[i] = 16;
  for (int64_t i = 30; i < 60; ++i) target_heads[i] = 2;

  auto target = maybe_compute_layerwise_layout(
      60, target_heads, kBIV100WorldSize);
  ASSERT_TRUE(target.has_value());
  EXPECT_EQ(target->num_layers(), 60);

  // Draft model: 12 layers, all dense.
  std::vector<int64_t> draft_heads(12, 8);
  auto draft = maybe_compute_layerwise_layout(
      12, draft_heads, kBIV100WorldSize);
  ASSERT_TRUE(draft.has_value());
  EXPECT_EQ(draft->num_layers(), 12);

  FLAGS_enable_layerwise_split = false;
}

}  // namespace
}  // namespace xllm
