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
// allocate_kv_caches_layerwise: per-layer KV allocation using
// LayerwiseSplitLayout.  Each layer's shard size is determined by the number
// of heads assigned to the current rank instead of uniform division.
//
// On ILU (Iluvatar CoreX / BI-V100) the cache tensor layout is transposed:
//   [n_blocks, n_heads, block_size, head_dim]
// — the head dimension sits at axis 1, not axis 2 as on CUDA/NPU.
//
// BI-V100 warp size = 64.  head_dim (typically 128) is already a multiple
// of 64, so coalesced warp-wide loads across the head dimension are aligned.
// When local_heads * head_dim is not a multiple of 64, the last warp in
// a block will have idle lanes — we pad head_dim to the next multiple of
// 64 on ILU to avoid this.

#include "framework/kv_cache/kv_cache_layerwise.h"

#include "framework/kv_cache/kv_cache.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "config/ilu_hw_constants.h"
#include "framework/kv_cache/kv_cache_utils.h"
#include "framework/kv_cache/layerwise_split_layout.h"

namespace xllm {

namespace {

/// Round up |val| to the next multiple of |align|.
inline int64_t align_up(int64_t val, int64_t align) {
  return ((val + align - 1) / align) * align;
}

}  // namespace

void allocate_kv_caches_layerwise(
    std::vector<KVCache>& kv_caches,
    const KVCacheShape& base_shape,
    const KVCacheCreateOptions& create_options,
    const LayerwiseSplitLayout& layout,
    int32_t current_rank) {
  CHECK(kv_caches.empty()) << "KV caches already initialized.";

  const int64_t num_layers = create_options.num_layers();
  CHECK_EQ(num_layers, layout.num_layers())
      << "Layout/config layer count mismatch.";
  kv_caches.reserve(num_layers);

  for (int64_t i = 0; i < num_layers; ++i) {
    if (!layout.rank_owns_layer(current_rank, i)) {
      kv_caches.emplace_back();          // empty placeholder
      continue;
    }

    const int64_t local_heads = layout.heads_for_rank(current_rank, i);
    CHECK_GT(local_heads, 0);

    // ---------- key cache ----------
    CHECK(base_shape.has_key_cache_shape());
    std::vector<int64_t> k_shape = base_shape.key_cache_shape();
    CHECK_GE(k_shape.size(), 4u);

    // ILU/MLU transposed layout: [n_blocks, n_heads, block_size, head_dim]
    // CUDA/NPU default layout:   [n_blocks, block_size, n_heads, head_dim]
    //
    // BI-V100 warp = 64: pad head_dim to multiple of 64 so that each warp's
    // contiguous load spans an aligned region.  Standard head_dim (128) is
    // already aligned; non-standard sizes (e.g. 96) get padded.
#if defined(USE_ILU) || defined(USE_MLU)
    constexpr int64_t kHeadDimAlign = ilu_hw::kWarpSize;  // 64
    k_shape[1] = local_heads;            // axis 1 = n_heads (transposed)
    k_shape[3] = align_up(k_shape[3], kHeadDimAlign);  // pad head_dim
#else
    k_shape[2] = local_heads;            // axis 2 = n_heads (default)
#endif

    auto opts = torch::TensorOptions()
                    .dtype(create_options.dtype())
                    .device(create_options.device());
    torch::Tensor k_tensor = torch::zeros(k_shape, opts);

    // ---------- value cache ----------
    if (base_shape.has_value_cache_shape()) {
      std::vector<int64_t> v_shape = base_shape.value_cache_shape();
      CHECK_GE(v_shape.size(), 4u);
#if defined(USE_ILU) || defined(USE_MLU)
      v_shape[1] = local_heads;
      v_shape[3] = align_up(v_shape[3], kHeadDimAlign);
#else
      v_shape[2] = local_heads;
#endif
      torch::Tensor v_tensor = torch::zeros(v_shape, opts);
      kv_caches.emplace_back(KVCacheTensors{k_tensor, v_tensor});
    } else {
      kv_caches.emplace_back(KVCacheTensors{k_tensor, torch::Tensor{}});
    }
  }

  CHECK_EQ(static_cast<int64_t>(kv_caches.size()), num_layers);
  LOG(INFO) << "[LayerwiseSplit] rank " << current_rank << ": "
            << layout.layers_on_rank(current_rank) << "/" << num_layers
            << " layers assigned.";
}

}  // namespace xllm
