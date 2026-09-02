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
// ILU-specific device-to-layer mapping for layerwise split KV cache.
//
// Verified Iluvatar BI-V100 topology (ixsmi topo -m):
//   - 4 cards, Bus-Id 4B:00.0 – 4E:00.0, all on NUMA node 1
//   - All pairs connected via PIX (single PCIe bridge) — FLAT topology
//   - No switch hierarchy: all inter-card bandwidth is equal
//   - 32 GB HBM per card (32768 MiB), 1500 MHz SM, 1200 MHz mem
//   - Warp size: 64 (verified via CUDA kernel warpSize builtin)
//   - IX-ML 3.2.3, Driver 3.2.1, CUDA 10.2 (CoreX)
//   - CoreX SDK at /usr/local/corex/
//
// Strategy (flat PIX topology):
//   Dense attention layers (many KV heads) → shard across ALL TP ranks
//   MoE layers (few KV heads via GQA)      → round-robin across ranks to
//     balance HBM usage (no grouping benefit since all links are equal)

#include "framework/parallel_state/mapping_ilu.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include "framework/kv_cache/layerwise_split_layout.h"

namespace xllm {

LayerwiseSplitLayout compute_ilu_layerwise_layout(
    int64_t num_layers,
    const std::vector<int64_t>& per_layer_kv_heads,
    int32_t world_size,
    IluTopoKind topo_kind) {
  CHECK_EQ(static_cast<int64_t>(per_layer_kv_heads.size()), num_layers);
  CHECK_GT(world_size, 0);

  std::vector<LayerShardSpec> specs;
  specs.reserve(num_layers);

  // For MoE layers with fewer heads than ranks, we round-robin the starting
  // rank so that different layers land on different subsets, balancing HBM
  // pressure across the flat PIX topology.
  int32_t rr_offset = 0;

  for (int64_t lid = 0; lid < num_layers; ++lid) {
    LayerShardSpec spec;
    spec.layer_id = lid;
    const int64_t total_heads = per_layer_kv_heads[lid];

    if (total_heads >= world_size) {
      // Dense attention: shard across all ranks.
      for (int32_t r = 0; r < world_size; ++r)
        spec.assigned_ranks.push_back(r);
      int64_t base = total_heads / world_size;
      int64_t rem  = total_heads % world_size;
      for (int32_t r = 0; r < world_size; ++r)
        spec.heads_per_rank.push_back(base + (r < rem ? 1 : 0));
    } else {
      // MoE / GQA layer: heads < world_size.
      // Flat PIX topology — all links equal, so round-robin starting rank
      // to spread HBM load evenly.
      int32_t needed = static_cast<int32_t>(total_heads);
      for (int32_t j = 0; j < needed; ++j) {
        int32_t rank = (rr_offset + j) % world_size;
        spec.assigned_ranks.push_back(rank);
      }
      int64_t base = total_heads / needed;
      int64_t rem  = total_heads % needed;
      for (int32_t j = 0; j < needed; ++j)
        spec.heads_per_rank.push_back(base + (j < rem ? 1 : 0));
      rr_offset = (rr_offset + needed) % world_size;
    }
    specs.push_back(std::move(spec));
  }

  LOG(INFO) << "[LayerwiseSplit] ILU layout computed: " << num_layers
            << " layers, " << world_size << " ranks, topo="
            << (topo_kind == IluTopoKind::kFlatPIX ? "flat_PIX" : "grouped");

  return LayerwiseSplitLayout(std::move(specs));
}

}  // namespace xllm
