// muh/include/muh/tuning/tuning_select_if.cuh — BI-V100
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_select_if.cuh
// CCCL SM100: 77 specializations, 38 SMEM OVERFLOW (max tile=163840)
//   Most overflows from threads=896-1024 with items=20 at type_size=4-8
//
// vllm relevance: token filtering (select tokens matching criteria)
// SMEM risk: CRITICAL. Must clamp threads and items aggressively.
//   Strategy: cap tile = threads * items * max(key_size, value_size) ≤ 48KB

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::select_if {

struct SelectIfPolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  BlockScanAlgorithm scan_algorithm;
  LookbackDelayPolicy delay;
  bool may_alias;
};

struct policy_selector {
  int input_size;
  int flag_size;
  int output_size;
  int offset_size;
  bool may_alias;

  constexpr SelectIfPolicy operator()(const hardware_capability& hw) const {
    // Effective element size: max of input/output for SMEM tile
    int elem_size = input_size > output_size ? input_size : output_size;
    
    // Start from conservative values and validate
    int threads = 256;
    int items = 14;
    
    if (elem_size <= 2) {
      threads = 384; items = 20;
    } else if (elem_size <= 4) {
      threads = 320; items = 16;
    } else if (elem_size <= 8) {
      threads = 256; items = 12;
    } else {
      threads = 192; items = 8;
    }
    
    // SMEM: select_if needs input tile + output tile + flags
    // Conservative: 2 * threads * items * elem_size + threads * items * flag_size
    int smem_needed = threads * items * (2 * elem_size + flag_size);
    while (smem_needed > hw.max_shared_memory_per_block && items > 1) {
      items--;
      smem_needed = threads * items * (2 * elem_size + flag_size);
    }

    return {threads, items, BLOCK_LOAD_WARP_TRANSPOSE,
            may_alias ? LOAD_CA : LOAD_DEFAULT,
            BLOCK_SCAN_WARP_SCANS,
            {LookbackDelayAlgorithm::exponential_backon, 350, 450},
            may_alias};
  }
};

} // namespace muh::tuning::select_if
