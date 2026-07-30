// muh/include/muh/tuning/tuning_transform_tile.cuh — BI-V100 transform_tile tuning
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_transform_tile.cuh
// vllm impact: tiled activation kernels (SiLU/GELU on tiles)
// Competition weight: Output TPS × 16.796

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::transform_tile {

struct TransformTilePolicy {
  int threads_per_block;
  int items_per_thread;
};

struct bi100_default {
  static constexpr int threads = 128;
  static constexpr int items = 8;
};

struct policy_selector {
  int min_elem_size;

  constexpr TransformTilePolicy operator()(const hardware_capability& hw) const {
    if (hw.at_least(hardware_capability::vendor_t::iluvatar, 100)) {
      return {bi100_default::threads, bi100_default::items};
    }
    // Fallback
    return {bi100_default::threads, bi100_default::items};
  }
};

} // namespace muh::tuning::transform_tile
