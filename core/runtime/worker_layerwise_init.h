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

/// Worker-side entry: allocate KV caches per the received layout.
/// Returns true on success; false if any verification check fails.
bool worker_allocate_layerwise_kv_cache(
    std::vector<KVCache>& kv_caches,
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options,
    const LayerwiseSplitLayout& layout,
    int32_t rank);

}  // namespace xllm
