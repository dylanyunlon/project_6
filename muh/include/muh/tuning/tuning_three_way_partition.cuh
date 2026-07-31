// muh/include/muh/tuning/tuning_three_way_partition.cuh — BI-V100
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_three_way_partition.cuh
// CCCL SM100: 19 specializations, 5 SMEM overflow (threads=384-1024, items=20-22, type_size=8)
//
// vllm relevance: token classification (keep/reject/uncertain) in speculative decoding
// SMEM risk: HIGH. 1024*20*4=81920 > 49152.

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::three_way_partition {

struct ThreeWayPartitionPolicy {
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

  constexpr ThreeWayPartitionPolicy operator()(const hardware_capability& hw) const {
    int pair_size = key_size + value_size;
    
    // Start from SM100 defaults, then clamp for BI-V100 SMEM
    int threads = 256;
    int items = 14;
    
    if (pair_size <= 2) {
      threads = 384; items = 20;
    } else if (pair_size <= 4) {
      threads = 384; items = 18;
    } else if (pair_size <= 8) {
      threads = 256; items = 14;
    } else {
      threads = 192; items = 10;
    }
    
    // SMEM check: three output buffers → 3 × threads × items × pair_size
    // (worst case: all items go to one partition)
    while (threads * items * pair_size > hw.max_shared_memory_per_block && items > 1)
      items--;

    return {threads, items, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
            BLOCK_SCAN_WARP_SCANS,
            {LookbackDelayAlgorithm::fixed_delay, 350, 450}};
  }
};

} // namespace muh::tuning::three_way_partition
