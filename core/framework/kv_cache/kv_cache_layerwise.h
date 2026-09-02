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

#include "framework/kv_cache/kv_cache.h"
#include "framework/kv_cache/kv_cache_shape.h"
#include "framework/kv_cache/kv_cache_utils.h"
#include "framework/kv_cache/layerwise_split_layout.h"

namespace xllm {

/// Allocate KV caches with per-layer head counts determined by |layout|.
/// Layers not assigned to |current_rank| receive an empty (default) KVCache.
void allocate_kv_caches_layerwise(
    std::vector<KVCache>& kv_caches,
    const KVCacheShape& base_shape,
    const KVCacheCreateOptions& create_options,
    const LayerwiseSplitLayout& layout,
    int32_t current_rank);

}  // namespace xllm
