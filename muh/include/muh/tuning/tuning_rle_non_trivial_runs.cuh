// muh/include/muh/tuning/tuning_rle_non_trivial_runs.cuh — BI-V100
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_rle_non_trivial_runs.cuh
// CCCL SM100: 14 type specializations, all tiles ≤ 36864 (safe for 48KB)
//
// vllm relevance: attention sparse pattern identification

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::rle_non_trivial_runs {

struct RleNonTrivialRunsPolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  BlockScanAlgorithm scan_algorithm;
  LookbackDelayPolicy delay;
};

struct policy_selector {
  int item_size;
  int offset_size;

  constexpr RleNonTrivialRunsPolicy operator()(const hardware_capability& hw) const {
    int threads = 320;
    int items = 10;
    
    if (item_size >= 8) items = 7;
    else if (item_size >= 4) items = 10;
    else items = 14;
    
    int pair_size = item_size + offset_size;
    while (threads * items * pair_size > hw.max_shared_memory_per_block && items > 1)
      items--;

    return {threads, items, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
            BLOCK_SCAN_WARP_SCANS,
            {LookbackDelayAlgorithm::fixed_delay, 350, 450}};
  }
};

} // namespace muh::tuning::rle_non_trivial_runs
