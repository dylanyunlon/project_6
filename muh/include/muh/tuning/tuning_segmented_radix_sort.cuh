// muh/include/muh/tuning/tuning_segmented_radix_sort.cuh — BI-V100
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_segmented_radix_sort.cuh
// CCCL: segmented variant of radix sort with per-segment dispatch
//
// vllm relevance: multi-head independent top-k via segmented sort
// SMEM risk: inherits from radix_sort base policy

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::segmented_radix_sort {

struct SegmentedRadixSortPolicy {
  int threads_per_block;
  int items_per_thread;
  int radix_bits;
  bool is_descending;
  int large_segment_threshold;
  int medium_segment_threshold;
};

struct policy_selector {
  int key_size;
  int value_size;
  bool keys_only;

  constexpr SegmentedRadixSortPolicy operator()(const hardware_capability& hw) const {
    // CCCL: large segments use full radix sort, small segments use warp sort
    int bits = (key_size == 1) ? 5 : 6;
    int items = (4 * 4) / key_size;  // 16 bytes per thread
    if (items < 1) items = 1;
    
    // SMEM: threads * items * (key_size + value_size) for the sort buffer
    int pair_size = key_size + (keys_only ? 0 : value_size);
    int threads = 256;
    while (threads * items * pair_size > hw.max_shared_memory_per_block && items > 1)
      items--;

    return {threads, items, bits, false,
            threads * items,    // large threshold
            hw.warp_size * 4};  // medium threshold
  }
};

} // namespace muh::tuning::segmented_radix_sort
