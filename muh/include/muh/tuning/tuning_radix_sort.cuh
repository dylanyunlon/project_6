// muh/include/muh/tuning/tuning_radix_sort.cuh — BI-V100 radix_sort tuning
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_radix_sort.cuh
// vllm impact: beam search token ranking, scheduler sorting
// Competition weight: Output TPS × 16.796

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::radix_sort {

struct RadixSortPolicy {
  int threads_per_block;
  int items_per_thread;
  int radix_bits;
};

struct bi100_default {
  static constexpr int threads = 384;
  static constexpr int items = 23;
  static constexpr int radix_bits = 8;
};

struct policy_selector {
  int key_size;
  int value_size;
  int offset_size;

  constexpr RadixSortPolicy operator()(const hardware_capability& hw) const {
    if (hw.at_least(hardware_capability::vendor_t::iluvatar, 100)) {
      return {bi100_default::threads, bi100_default::items, bi100_default::radix_bits};
    }
    // Fallback
    return {bi100_default::threads, bi100_default::items, bi100_default::radix_bits};
  }
};

} // namespace muh::tuning::radix_sort
