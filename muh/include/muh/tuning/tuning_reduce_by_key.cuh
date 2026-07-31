// muh/include/muh/tuning/tuning_reduce_by_key.cuh — BI-V100
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_reduce_by_key.cuh
// CCCL SM100: 67 type specializations, 2 SMEM overflow (8B types at high threads)
//
// vllm relevance: grouped token aggregation (e.g. expert routing in MoE)

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::reduce_by_key {

struct ReduceByKeyPolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  BlockScanAlgorithm scan_algorithm;
  LookbackDelayPolicy delay;
};

struct policy_selector {
  int key_size;
  int accum_size;
  int offset_size;
  type_t accum_type;

  constexpr ReduceByKeyPolicy operator()(const hardware_capability& hw) const {
    int pair_size = key_size + accum_size;
    
    // SM100 typical: threads=224-384, items=10-18
    int threads = 256;
    int items = 14;
    
    if (pair_size <= 4) {
      threads = 320; items = 16;
    } else if (pair_size <= 8) {
      threads = 256; items = 14;
    } else {
      threads = 192; items = 10;
    }
    
    // SMEM: tile = threads * items * pair_size (key + accum per element)
    while (threads * items * pair_size > hw.max_shared_memory_per_block && items > 1)
      items--;

    return {threads, items, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
            BLOCK_SCAN_WARP_SCANS,
            {LookbackDelayAlgorithm::exponential_backon, 350, 450}};
  }
};

} // namespace muh::tuning::reduce_by_key
