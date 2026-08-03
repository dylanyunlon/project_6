// muh/include/muh/tuning/tuning_find.cuh — BI-V100
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_find.cuh
// CCCL source: 90 lines. Single policy for all CC:
//   {scale_mem_bound(128, 16, input_size), vec_size=4, LOAD_LDG}
//
// vllm relevance: DeviceFind::FindIf for locating EOS tokens, stop sequences,
// and special token positions in output sequences.
//
// No SMEM usage (vectorized global memory scan), BI-V100 safe.

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::find {

struct FindIfPolicy {
  int threads_per_block;
  int items_per_thread;
  int vec_size;
  CacheLoadModifier load_modifier;
};

// CCCL policy_selector (all CC):
//   scale_mem_bound(128, 16, input_type_size) → threads, items
//   vec_size = 4, load_modifier = LOAD_LDG

struct policy_selector {
  int input_type_size;

  constexpr FindIfPolicy operator()(const hardware_capability& hw) const {
    auto [items, threads] = scale_mem_bound(128, 16, input_type_size);
    return FindIfPolicy{threads, items, 4, LOAD_LDG};
  }
};

} // namespace muh::tuning::find
