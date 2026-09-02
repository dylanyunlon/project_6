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
#include <vector>

#include "framework/kv_cache/layerwise_split_layout.h"

namespace xllm {

/// Called by llm_engine / speculative_engine at startup.
/// Returns a LayerwiseSplitLayout if the feature is enabled, otherwise
/// std::nullopt (fallback to uniform allocation).
std::optional<LayerwiseSplitLayout> maybe_compute_layerwise_layout(
    int64_t num_layers,
    const std::vector<int64_t>& per_layer_kv_heads,
    int32_t world_size);

}  // namespace xllm
