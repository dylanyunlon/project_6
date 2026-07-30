// muh/include/muh/tuning/tuning_find.cuh — BI-V100 find tuning
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_find.cuh
// vllm impact: element search (minor)
// Competition weight: minimal

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::find {

struct FindPolicy {
  int threads_per_block;
  int items_per_thread;
  int vec_size;
  CacheLoadModifier load_modifier;
};

struct bi100_default {
  static constexpr int threads = 128;
  static constexpr int items = 16;
  static constexpr int vec_size = 4;
  static constexpr int load_mod = LOAD_LDG;
};

struct policy_selector {
  int input_size;

  constexpr FindPolicy operator()(const hardware_capability& hw) const {
    if (hw.at_least(hardware_capability::vendor_t::iluvatar, 100)) {
      return {bi100_default::threads, bi100_default::items, bi100_default::vec_size, LOAD_LDG};
    }
    // Fallback
    return {bi100_default::threads, bi100_default::items, bi100_default::vec_size, LOAD_LDG};
  }
};

} // namespace muh::tuning::find
