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

// Commit: 494f293b5629 · feat · PR #2260  (adapted for Iluvatar BI-V100)
// Memory estimation for layerwise-split KV cache.  Reports both peak
// (bottleneck) and average per-rank utilisation so that capacity planning
// on BI-V100 (32768 MiB HBM verified via ixsmi) can account for uneven
// sharding.

#include "framework/kv_cache/kv_cache_estimation_layerwise.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <unordered_set>
#include <vector>

#include "config/ilu_hw_constants.h"

namespace xllm {

namespace {

/// Bytes per KV element for a given dtype.
int64_t dtype_bytes(int dtype_enum) {
  // torch::kBFloat16 = 15, torch::kHalf = 5, torch::kFloat = 6
  switch (dtype_enum) {
    case 5:  return 2;   // float16
    case 15: return 2;   // bfloat16
    case 6:  return 4;   // float32
    case 2:  return 1;   // int8
    default: return 2;   // conservative
  }
}

/// Round up to next multiple of |align|.
inline int64_t align_up(int64_t val, int64_t align) {
  return ((val + align - 1) / align) * align;
}

}  // namespace

LayerwiseKVMemoryEstimate estimate_layerwise_kv_memory(
    const LayerwiseSplitLayout& layout,
    int64_t n_blocks,
    int64_t block_size,
    int64_t head_dim,
    int64_t max_tokens,
    int dtype_enum,
    int32_t world_size) {
  CHECK_GT(layout.num_layers(), 0);
  CHECK_GT(world_size, 0);

  const int64_t elem_bytes = dtype_bytes(dtype_enum);

  // BI-V100 warp = 64: the allocator pads head_dim to the next multiple
  // of 64.  The estimator must match, otherwise it under-reports.
#if defined(USE_ILU)
  const int64_t padded_head_dim = align_up(head_dim, ilu_hw::kWarpSize);
#else
  const int64_t padded_head_dim = head_dim;
#endif

  // Per-rank KV bytes: sum over layers of (2 * heads * n_blocks *
  // block_size * padded_head_dim * elem_bytes).  Factor 2 = K + V.
  std::vector<int64_t> per_rank_bytes(world_size, 0);
  for (int64_t lid = 0; lid < layout.num_layers(); ++lid) {
    const auto& spec = layout.layer_spec(lid);
    for (size_t i = 0; i < spec.assigned_ranks.size(); ++i) {
      int32_t rank = spec.assigned_ranks[i];
      int64_t heads = spec.heads_per_rank[i];
      int64_t layer_bytes = 2 * heads * n_blocks * block_size *
                            padded_head_dim * elem_bytes;
      CHECK_GE(rank, 0);
      CHECK_LT(rank, world_size);
      per_rank_bytes[rank] += layer_bytes;
    }
  }

  // Uniform baseline (also with padding for fair comparison).
  int64_t uniform_total = 0;
  for (const auto& s : layout.specs())
    uniform_total += s.total_heads();
  int64_t uniform_per_rank =
      2 * (uniform_total / world_size) * n_blocks * block_size *
      padded_head_dim * elem_bytes;

  int64_t peak = *std::max_element(per_rank_bytes.begin(),
                                    per_rank_bytes.end());
  int64_t sum  = std::accumulate(per_rank_bytes.begin(),
                                  per_rank_bytes.end(), int64_t{0});
  double average = static_cast<double>(sum) / world_size;

  LayerwiseKVMemoryEstimate est;
  est.peak_per_rank_bytes     = peak;
  est.average_per_rank_bytes  = static_cast<int64_t>(average);
  est.uniform_per_rank_bytes  = uniform_per_rank;
  est.per_rank_bytes          = std::move(per_rank_bytes);
  est.savings_vs_uniform_pct  =
      uniform_per_rank > 0
          ? 100.0 * (1.0 - static_cast<double>(peak) / uniform_per_rank)
          : 0.0;

  LOG(INFO) << "[LayerwiseSplit] KV memory estimate: peak="
            << (peak >> 20) << " MiB, avg="
            << (static_cast<int64_t>(average) >> 20) << " MiB, uniform="
            << (uniform_per_rank >> 20) << " MiB, saving="
            << est.savings_vs_uniform_pct << "%";

  return est;
}

}  // namespace xllm
