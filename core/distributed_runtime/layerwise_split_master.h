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
#include <optional>

#include "framework/kv_cache/layerwise_split_layout.h"

namespace xllm {

/// Master-side entry point: compute and log the layerwise layout.
/// |first_moe_layer|: index of the first MoE layer (layers before it are
///                     dense attention with |dense_kv_heads|).
std::optional<LayerwiseSplitLayout> master_compute_layerwise_layout(
    int64_t num_layers,
    int64_t dense_kv_heads,
    int64_t moe_kv_heads,
    int64_t first_moe_layer,
    int32_t world_size,
    int64_t n_blocks,
    int64_t block_size,
    int64_t head_dim,
    int64_t max_tokens,
    int dtype_enum);

}  // namespace xllm
