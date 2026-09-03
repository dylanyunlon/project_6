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
// Layerwise split KV cache sharding for heterogeneous layer structures.
//
// Qwen3.5 (Qwen3.6-35B-A3B) has interleaved full_attention and
// linear_attention (GDN/DeltaNet) layers.  Only full_attention layers
// use KV cache; linear_attention layers use conv+temporal state.
// With layerwise split, each full_attention layer's KV cache ownership
// is assigned to a subset of TP ranks via round-robin (layer_id % group_size).
//
// Two API levels:
//   1. LayerwiseSplitLayout — upstream-compatible (enabled, group_size,
//      local_rank).  Determines ownership by layer_id % group_size.
//   2. LayerShardSpec / IluLayerwiseLayout — BI-V100-specific detailed
//      per-layer shard descriptors with explicit head-count assignments.

#pragma once

#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include <glog/logging.h>

namespace xllm {

// =========================================================================
// Upstream-compatible API (matches xllm layerwise_split_layout.h)
// =========================================================================

/// Check if the model type supports layerwise split.
/// Qwen3_5 (Qwen3.6-35B-A3B MoE) is supported on BI-V100.
[[nodiscard]] inline bool is_layerwise_split_supported_model(
    const std::string& model_type) noexcept {
  return model_type == "deepseek_v32" ||
         model_type == "glm_moe_dsa" ||
         model_type == "qwen3_5" ||
         model_type == "qwen3_5_moe_text";
}

inline void validate_layerwise_split_size_config(int32_t layerwise_split_size) {
  CHECK_GE(layerwise_split_size, 1)
      << "layerwise_split_size must be >= 1, got " << layerwise_split_size;
}

inline void validate_layerwise_split_enablement(int32_t layerwise_split_size,
                                                int32_t attn_tp_size,
                                                const std::string& model_type) {
  if (layerwise_split_size <= 1) {
    return;
  }
  CHECK(is_layerwise_split_supported_model(model_type))
      << "layerwise_split_size > 1 is only supported for deepseek_v32, "
         "glm_moe_dsa, and qwen3_5 variants, got "
      << model_type;
  CHECK_EQ(attn_tp_size % layerwise_split_size, 0)
      << "attention tp size (" << attn_tp_size
      << ") must be divisible by layerwise_split_size (" << layerwise_split_size
      << ").";
}

/// Upstream-compatible layout: round-robin ownership via layer_id % group_size.
/// Linear-attention layers (GDN) always return owns()=true because they have
/// no KV cache — ownership is only relevant for full-attention layers.
class LayerwiseSplitLayout {
 public:
  LayerwiseSplitLayout() = default;
  LayerwiseSplitLayout(bool enabled, int32_t group_size, int32_t local_rank)
      : enabled_(enabled), group_size_(group_size), local_rank_(local_rank) {
    CHECK_GT(group_size_, 0) << "Layerwise split group size must be positive.";
    CHECK(local_rank_ >= 0 && local_rank_ < group_size_)
        << "Layerwise split local rank must be in [0, group_size).";
  }

  [[nodiscard]] int32_t owner_rank(int64_t layer_id) const {
    return static_cast<int32_t>(layer_id % group_size_);
  }

  [[nodiscard]] bool owns(int64_t layer_id) const {
    return !enabled_ || owner_rank(layer_id) == local_rank_;
  }

  [[nodiscard]] bool enabled() const { return enabled_; }
  [[nodiscard]] int32_t group_size() const { return group_size_; }
  [[nodiscard]] int32_t local_rank() const { return local_rank_; }

 private:
  bool enabled_ = false;
  int32_t group_size_ = 1;
  int32_t local_rank_ = 0;
};

// =========================================================================
// BI-V100-specific detailed layout (per-layer shard descriptors)
// =========================================================================

/// Per-layer KV shard descriptor for BI-V100.
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

/// Full BI-V100-specific layout: one LayerShardSpec per model layer.
/// Computed at master startup and broadcast to every worker.
class IluLayerwiseLayout {
 public:
  IluLayerwiseLayout() = default;
  explicit IluLayerwiseLayout(std::vector<LayerShardSpec> specs)
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

// =========================================================================
// Layer-type helpers for Qwen3.5 (full_attention vs linear_attention)
// =========================================================================

/// Build a layer_cache_owned mask for Qwen3.5-like models.
/// Linear-attention layers always stay owned (they have no KV cache to split).
/// Full-attention layers follow LayerwiseSplitLayout.
///
/// |layer_types|: "full_attention" or "linear_attention" per layer.
///                If empty, all layers are treated as full_attention.
inline std::vector<bool> build_layer_cache_owned(
    const std::vector<std::string>& layer_types,
    const LayerwiseSplitLayout& layout,
    int64_t num_layers) {
  std::vector<bool> owned;
  owned.reserve(static_cast<size_t>(num_layers));
  int64_t full_attn_idx = 0;
  for (int64_t i = 0; i < num_layers; ++i) {
    const bool is_linear = !layer_types.empty() &&
        static_cast<size_t>(i) < layer_types.size() &&
        layer_types[static_cast<size_t>(i)] == "linear_attention";
    if (is_linear) {
      // Linear-attention layers have no KV cache — always "owned"
      // (nothing to split).
      owned.push_back(true);
    } else {
      // Full-attention layer: ownership follows layerwise split.
      owned.push_back(layout.owns(full_attn_idx));
      ++full_attn_idx;
    }
  }
  return owned;
}

}  // namespace xllm
