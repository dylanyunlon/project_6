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
// Worker-side helper: after the worker receives its LayerwiseSplitLayout
// from the master, it calls this to allocate per-layer KV caches with
// the correct shard sizes.

#include "runtime/worker_layerwise_init.h"

#include <glog/logging.h>

#include <vector>

#include "framework/kv_cache/kv_cache.h"
#include "framework/kv_cache/kv_cache_layerwise.h"
#include "framework/kv_cache/kv_cache_shape.h"
#include "framework/kv_cache/kv_cache_utils.h"
#include "framework/kv_cache/layerwise_split_layout.h"

namespace xllm {

bool worker_allocate_layerwise_kv_cache(
    std::vector<KVCache>& kv_caches,
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& create_options,
    const LayerwiseSplitLayout& layout,
    int32_t rank) {
  LOG(INFO) << "[Worker " << rank << "] Applying layerwise KV layout: "
            << layout.layers_on_rank(rank) << " layers assigned.";

  try {
    allocate_kv_caches_layerwise(
        kv_caches, kv_cache_shape, create_options, layout, rank);
  } catch (const std::exception& e) {
    LOG(ERROR) << "[Worker " << rank
               << "] Failed to allocate layerwise KV cache: " << e.what();
    return false;
  }

  // Verify: assigned layers should have non-empty caches.
  for (int64_t lid = 0; lid < layout.num_layers(); ++lid) {
    bool owns = layout.rank_owns_layer(rank, lid);
    bool empty = kv_caches[lid].empty();
    if (owns && empty) {
      LOG(ERROR) << "[Worker " << rank << "] Layer " << lid
                 << " is assigned but KV cache is empty.";
      return false;
    }
    if (!owns && !empty) {
      LOG(ERROR) << "[Worker " << rank << "] Layer " << lid
                 << " is NOT assigned but KV cache is non-empty.";
      return false;
    }
  }

  LOG(INFO) << "[Worker " << rank << "] Layerwise KV cache allocation OK.";
  return true;
}

}  // namespace xllm
