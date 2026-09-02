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

#pragma once

#include <cstdint>
#include <vector>

#include "framework/kv_cache/layerwise_split_layout.h"

namespace xllm {

struct LayerwiseKVMemoryEstimate {
  int64_t peak_per_rank_bytes    = 0;  // worst-case rank
  int64_t average_per_rank_bytes = 0;
  int64_t uniform_per_rank_bytes = 0;  // baseline (uniform sharding)
  std::vector<int64_t> per_rank_bytes;  // detailed per-rank breakdown
  double savings_vs_uniform_pct  = 0.0;
};

/// Estimate per-rank KV cache memory for a layerwise-split layout.
/// |dtype_enum| matches torch::ScalarType integer values.
LayerwiseKVMemoryEstimate estimate_layerwise_kv_memory(
    const LayerwiseSplitLayout& layout,
    int64_t n_blocks,
    int64_t block_size,
    int64_t head_dim,
    int64_t max_tokens,
    int dtype_enum,
    int32_t world_size);

}  // namespace xllm
