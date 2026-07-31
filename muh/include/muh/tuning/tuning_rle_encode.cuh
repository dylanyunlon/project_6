// muh/include/muh/tuning/tuning_rle_encode.cuh — BI-V100
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_rle_encode.cuh
// CCCL SM100: 14 type specializations, all tiles ≤ 28672 (safe for 48KB)
//
// vllm relevance: attention mask compression via run-length encoding

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::rle_encode {

struct RleEncodePolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  BlockScanAlgorithm scan_algorithm;
  LookbackDelayPolicy delay;
};

struct policy_selector {
  int item_size;
  int length_size;

  constexpr RleEncodePolicy operator()(const hardware_capability& hw) const {
    // SM100 patterns: threads=192-448, items=7-15
    // All tiles ≤ 28672, no overflow risk on BI-V100
    int threads = 256;
    int items = 10;
    
    // Scale items by type size (larger types → fewer items)
    if (item_size >= 8) {
      items = 7;
    } else if (item_size >= 4) {
      items = 10;
    } else {
      items = 14;
    }
    
    // SMEM check: tile = threads * items * (item_size + length_size)
    int pair_size = item_size + length_size;
    while (threads * items * pair_size > hw.max_shared_memory_per_block && items > 1)
      items--;

    return {threads, items, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
            BLOCK_SCAN_WARP_SCANS,
            {LookbackDelayAlgorithm::fixed_delay, 350, 450}};
  }
};

} // namespace muh::tuning::rle_encode
