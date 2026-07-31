// muh/include/muh/tuning/tuning_segmented_sort.cuh — BI-V100
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_segmented_sort.cuh
// CCCL: chained_policy with per-segment-size dispatch (large/medium/small)
//
// vllm relevance: per-sequence token ranking

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::segmented_sort {

struct SegmentedSortPolicy {
  int large_threads;
  int large_items;
  int medium_threads;
  int medium_items;
  int small_threads;
  int small_items;
  BlockLoadAlgorithm load_algorithm;
};

struct policy_selector {
  int key_size;
  int value_size;
  bool keys_only;

  constexpr SegmentedSortPolicy operator()(const hardware_capability& hw) const {
    int pair_size = key_size + (keys_only ? 0 : value_size);
    
    // Large segments: full block sort
    auto [lg_items, lg_threads] = scale_mem_bound(256, 11, pair_size);
    // Medium: partial
    auto [md_items, md_threads] = scale_mem_bound(128, 15, pair_size);
    // Small: warp sort
    int sm_threads = 32;
    int sm_items = 4;

    return {lg_threads, lg_items, md_threads, md_items, 
            sm_threads, sm_items, BLOCK_LOAD_WARP_TRANSPOSE};
  }
};

} // namespace muh::tuning::segmented_sort
