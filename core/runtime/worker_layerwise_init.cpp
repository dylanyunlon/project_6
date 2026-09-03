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
// Worker-side helper: after the worker receives its layerwise_split_size
// from the master, it computes the layer_cache_owned mask for KV cache
// allocation.
//
// On BI-V100, the actual KV cache tensor allocation is done by the vendor
// vLLM cache engine (CacheEngine._allocate_kv_cache), which uses the
// prebuilt corex_*.so kernel extensions.  This module only decides the
// ownership mask at the scheduling layer.

#include "runtime/worker_layerwise_init.h"

#include <glog/logging.h>

#include <string>
#include <vector>

#include "framework/kv_cache/kv_cache_layerwise.h"
#include "framework/kv_cache/layerwise_split_layout.h"

namespace xllm {

std::vector<bool> worker_compute_layer_cache_owned(
    const std::vector<std::string>& layer_types,
    int32_t layerwise_split_size,
    int32_t rank,
    int64_t num_layers) {
  CHECK_GE(layerwise_split_size, 1);
  CHECK_GE(rank, 0);
  CHECK_GT(num_layers, 0);

  if (layerwise_split_size <= 1) {
    // No layerwise split — all layers owned.
    LOG(INFO) << "[Worker " << rank << "] No layerwise split, all "
              << num_layers << " layers owned.";
    return std::vector<bool>(static_cast<size_t>(num_layers), true);
  }

  const LayerwiseSplitLayout layout(
      /*enabled=*/true,
      layerwise_split_size,
      rank % layerwise_split_size);

  auto owned = build_layer_cache_owned(layer_types, layout, num_layers);

  // Verify and log.
  int64_t owned_count = 0;
  int64_t linear_count = 0;
  for (int64_t i = 0; i < num_layers; ++i) {
    if (owned[static_cast<size_t>(i)]) {
      ++owned_count;
    }
    if (!layer_types.empty() &&
        static_cast<size_t>(i) < layer_types.size() &&
        layer_types[static_cast<size_t>(i)] == "linear_attention") {
      ++linear_count;
    }
  }

  LOG(INFO) << "[Worker " << rank << "] Layerwise split (size="
            << layerwise_split_size << "): " << owned_count << "/"
            << num_layers << " layers owned, " << linear_count
            << " linear-attention (always owned).";

  // Delegate to scheduling-layer allocation check.
  allocate_kv_caches_layerwise(owned, num_layers, rank);

  return owned;
}

}  // namespace xllm
