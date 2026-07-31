// muh/include/muh/tuning/tuning_transform_tile.cuh — BI-V100
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_transform_tile.cuh
// CCCL: tile-based transform, shares bytes-in-flight target with transform
//
// vllm relevance: fused activation+normalization tiles
// SMEM risk: tile size is compile-time shape, not threads*items

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::transform_tile {

struct TransformTilePolicy {
  int tile_size;
  int min_bytes_in_flight;
};

struct policy_selector {
  int elem_size;

  constexpr TransformTilePolicy operator()(const hardware_capability& hw) const {
    // BI-V100 per-SM BW ≈ A100 → bytes_in_flight = 16KB
    int bytes_in_flight = 16 * 1024;
    int tile = bytes_in_flight / elem_size;
    if (tile < 128) tile = 128;
    return {tile, bytes_in_flight};
  }
};

} // namespace muh::tuning::transform_tile
