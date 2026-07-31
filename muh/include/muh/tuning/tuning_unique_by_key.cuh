// muh/include/muh/tuning/tuning_unique_by_key.cuh — BI-V100
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_unique_by_key.cuh
// CCCL SM100: 51 specializations, 7 SMEM overflow (max tile=57344)
//
// vllm relevance: deduplicated token sequences

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::unique_by_key {

struct UniqueByKeyPolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  BlockScanAlgorithm scan_algorithm;
  LookbackDelayPolicy delay;
};

struct policy_selector {
  int key_size;
  int value_size;
  int offset_size;

  constexpr UniqueByKeyPolicy operator()(const hardware_capability& hw) const {
    int pair_size = key_size + value_size;
    
    int threads = 256;
    int items = 12;
    
    if (pair_size <= 4) {
      threads = 320; items = 16;
    } else if (pair_size <= 8) {
      threads = 256; items = 12;
    } else {
      threads = 192; items = 8;
    }
    
    while (threads * items * pair_size > hw.max_shared_memory_per_block && items > 1)
      items--;

    return {threads, items, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
            BLOCK_SCAN_WARP_SCANS,
            {LookbackDelayAlgorithm::exponential_backon, 350, 450}};
  }
};

} // namespace muh::tuning::unique_by_key
