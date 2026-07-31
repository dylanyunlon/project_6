// muh/include/muh/tuning/tuning_adjacent_difference.cuh — BI-V100
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_adjacent_difference.cuh
// CCCL: single policy for all architectures. No SM100 specialization.
//   threads=128, items=nominal_8B_items_to_items(7, value_size), WARP_TRANSPOSE, LDG/CA
//
// vllm relevance: low (KV cache delta encoding, not on hot path)
// SMEM risk: 128*items*value_size — at value_size=8, items=7, tile=7168 ≤ 49152 ✓

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::adjacent_difference {

struct AdjacentDifferencePolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  BlockStoreAlgorithm store_algorithm;
};

// CCCL's nominal_8B_items_to_items: scale items for non-8B types
// items = max(1, nominal_8B_items * 8 / value_type_size)
constexpr int nominal_8B_items(int nominal, int value_size) {
  int items = nominal * 8 / value_size;
  return items < 1 ? 1 : items;
}

struct policy_selector {
  int value_type_size;
  bool may_alias;

  constexpr AdjacentDifferencePolicy operator()(const hardware_capability& /*hw*/) const {
    // CCCL uses the same policy for all architectures
    return {128,
            nominal_8B_items(7, value_type_size),
            BLOCK_LOAD_WARP_TRANSPOSE,
            may_alias ? LOAD_CA : LOAD_LDG,
            BLOCK_STORE_WARP_TRANSPOSE};
  }
};

} // namespace muh::tuning::adjacent_difference
