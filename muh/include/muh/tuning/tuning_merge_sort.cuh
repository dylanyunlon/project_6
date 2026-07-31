// muh/include/muh/tuning/tuning_merge_sort.cuh — BI-V100
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_merge_sort.cuh
// CCCL: block sort + merge phases with scale_mem_bound
//
// vllm relevance: large-scale token logits sorting
// SMEM risk: scale_mem_bound handles block sort; merge phase uses separate SMEM

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::merge_sort {

struct MergeSortPolicy {
  int block_sort_threads;
  int block_sort_items;
  BlockLoadAlgorithm block_sort_load;
  int merge_threads;
  int merge_items;
  CacheLoadModifier merge_load_modifier;
};

struct policy_selector {
  int key_size;
  int value_size;

  constexpr MergeSortPolicy operator()(const hardware_capability& /*hw*/) const {
    int pair_size = key_size + (value_size > 0 ? value_size : 0);
    
    // Block sort phase
    auto [bs_items, bs_threads] = scale_mem_bound(256, 11, pair_size);
    
    // Merge phase: fewer threads, more items
    auto [mg_items, mg_threads] = scale_mem_bound(256, 15, pair_size);

    return {bs_threads, bs_items, BLOCK_LOAD_WARP_TRANSPOSE,
            mg_threads, mg_items, LOAD_DEFAULT};
  }
};

} // namespace muh::tuning::merge_sort
