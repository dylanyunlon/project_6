// muh/include/muh/tuning/tuning_for.cuh — BI-V100 for-each tuning
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_for.cuh
//
// vllm impact: RoPE position encoding, simple elementwise kernels
// Competition weight: contributes to Output TPS

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::for_each {

/// For-each policy
struct ForPolicy {
  int threads_per_block;
  int items_per_thread;
};

// ============================================================
// BI-V100 tuning
//
// CCCL reference from tuning_for.cuh:
//   threads_per_block: 256 default, can be runtime-determined if set <1
//   items_per_thread: typically 1-4 for simple elementwise
//
//   The for_each kernel is extremely simple — it's a parallel_for
//   with no shared memory, no reduction, no scan. Tuning is purely
//   about occupancy (threads × items = tile_size).
// ============================================================

struct bi100_default {
  static constexpr int threads = 256;
  static constexpr int items   = 4;
};

// ============================================================
// policy_selector
// ============================================================

struct policy_selector {
  constexpr ForPolicy operator()(const hardware_capability& hw) const {
    if (hw.at_least(hardware_capability::vendor_t::iluvatar, 100)) {
      return {bi100_default::threads, bi100_default::items};
    }
    return {256, 4};
  }
};

} // namespace muh::tuning::for_each
