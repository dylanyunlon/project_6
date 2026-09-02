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
// Layerwise split KV cache sharding for heterogeneous layer structures
// (e.g. DeepSeek-V3: dense attention interleaved with MoE layers).

#pragma once

#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include <glog/logging.h>

namespace xllm {

/// Per-layer KV shard descriptor.
struct LayerShardSpec {
  int64_t layer_id = -1;
  std::vector<int32_t> assigned_ranks;   // TP ranks storing this layer's KV
  std::vector<int64_t> heads_per_rank;   // KV heads each rank holds

  int64_t total_heads() const {
    return std::accumulate(heads_per_rank.begin(), heads_per_rank.end(),
                           int64_t{0});
  }

  bool is_valid() const {
    if (layer_id < 0 || assigned_ranks.empty()) return false;
    if (assigned_ranks.size() != heads_per_rank.size()) return false;
    for (auto h : heads_per_rank) {
      if (h <= 0) return false;
    }
    return true;
  }
};

/// Full layout: one LayerShardSpec per model layer, computed at master
/// startup and broadcast to every worker.
class LayerwiseSplitLayout {
 public:
  LayerwiseSplitLayout() = default;
  explicit LayerwiseSplitLayout(std::vector<LayerShardSpec> specs)
      : specs_(std::move(specs)) { validate(); }

  int64_t num_layers() const { return static_cast<int64_t>(specs_.size()); }

  const LayerShardSpec& layer_spec(int64_t lid) const {
    CHECK_GE(lid, 0);
    CHECK_LT(lid, num_layers());
    return specs_[lid];
  }

  bool rank_owns_layer(int32_t rank, int64_t lid) const {
    for (auto r : specs_[lid].assigned_ranks)
      if (r == rank) return true;
    return false;
  }

  int64_t heads_for_rank(int32_t rank, int64_t lid) const {
    const auto& s = specs_[lid];
    for (size_t i = 0; i < s.assigned_ranks.size(); ++i)
      if (s.assigned_ranks[i] == rank) return s.heads_per_rank[i];
    return 0;
  }

  int64_t layers_on_rank(int32_t rank) const {
    int64_t n = 0;
    for (const auto& s : specs_)
      for (auto r : s.assigned_ranks)
        if (r == rank) { ++n; break; }
    return n;
  }

  void validate() const {
    for (int64_t i = 0; i < num_layers(); ++i) {
      CHECK(specs_[i].is_valid()) << "Invalid LayerShardSpec at " << i;
      CHECK_EQ(specs_[i].layer_id, i) << "Layer id mismatch at " << i;
    }
  }

  const std::vector<LayerShardSpec>& specs() const { return specs_; }

 private:
  std::vector<LayerShardSpec> specs_;
};

}  // namespace xllm
