// muh/include/muh/tuning/tuning_scan_by_key.cuh — BI-V100
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_scan_by_key.cuh
// CCCL SM100: 66 type specializations, all tiles ≤ 47104 (fits 48KB)
//
// vllm relevance: key-segmented prefix sums (e.g. per-request cumulative attention)

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::scan_by_key {

struct ScanByKeyPolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  BlockScanAlgorithm scan_algorithm;
  BlockStoreAlgorithm store_algorithm;
  LookbackDelayPolicy delay;
};

struct policy_selector {
  int key_size;
  int accum_size;
  int offset_size;

  constexpr ScanByKeyPolicy operator()(const hardware_capability& hw) const {
    int pair_size = key_size + accum_size;
    
    int threads = 256;
    int items = 14;
    
    if (pair_size <= 4) {
      threads = 320; items = 18;
    } else if (pair_size <= 8) {
      threads = 256; items = 14;
    } else {
      threads = 192; items = 10;
    }
    
    while (threads * items * pair_size > hw.max_shared_memory_per_block && items > 1)
      items--;

    return {threads, items, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
            BLOCK_SCAN_WARP_SCANS, BLOCK_STORE_WARP_TRANSPOSE,
            {LookbackDelayAlgorithm::exponential_backon, 350, 450}};
  }
};

} // namespace muh::tuning::scan_by_key
