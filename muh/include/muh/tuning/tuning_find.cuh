// muh/include/muh/tuning/tuning_find.cuh — BI-V100
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_find.cuh
// CCCL: single policy, uses scale_mem_bound(128, 16, input_type_size)
//
// vllm relevance: EOS token detection in decode
// SMEM risk: scale_mem_bound handles it

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

struct policy_selector {
  int input_type_size;

  constexpr FindIfPolicy operator()(const hardware_capability& /*hw*/) const {
    auto [items, threads] = scale_mem_bound(128, 16, input_type_size);
    return {threads, items, 4, LOAD_LDG};
  }
};

} // namespace muh::tuning::find
