// muh/include/muh/tuning/tuning_adjacent_difference.cuh — BI-V100
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_adjacent_difference.cuh
// CCCL source: 118 lines. Single policy for all compute capabilities.
//
// CCCL policy_selector (all CC):
//   {128, Nominal8BItems(7, value_size), WARP_TRANSPOSE, may_alias?LOAD_CA:LOAD_LDG, WARP_TRANSPOSE}
//
// Nominal8BItemsToItems(7, value_size) = max(1, 7 * 8 / value_size)
//   1B → 56, 2B → 28, 4B → 14, 8B → 7, 16B → 3
//
// vllm relevance: computing token-level delta logits for speculative decoding,
// detecting attention pattern changes between consecutive positions.
//
// SMEM: threads * items * value_size * 2 (BlockLoad + BlockStore, double buffer)
// BI-V100 48KB limit → cap items when value_size is large.

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::adjacent_difference {

// ============================================================================
// Policy struct — matches CCCL AdjacentDifferencePolicy exactly
// ============================================================================

struct AdjacentDifferencePolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  BlockStoreAlgorithm store_algorithm;
};

// ============================================================================
// Helper: nominal_8B_items_to_items (from CCCL util_device.cuh)
// Scales items from an 8-byte nominal to actual value_size
// ============================================================================

constexpr int nominal_8b_items(int nominal, int value_size) {
  int result = nominal * 8 / value_size;
  return result > 0 ? result : 1;
}

// ============================================================================
// policy_selector — single policy for all CC (matches CCCL)
// BI-V100 SMEM cap applied for large items
// ============================================================================

struct policy_selector {
  int value_type_size;
  bool may_alias;

  constexpr AdjacentDifferencePolicy operator()(const hardware_capability& hw) const {
    int items = nominal_8b_items(7, value_type_size);

    // SMEM check: BlockLoad + BlockStore share tile through union
    // tile = threads * items * value_type_size
    int threads = 128;
    int tile_smem = threads * items * value_type_size;
    while (tile_smem > hw.max_shared_memory_per_block - 2048 && items > 1) {
      items--;
      tile_smem = threads * items * value_type_size;
    }

    return AdjacentDifferencePolicy{
      threads, items,
      BLOCK_LOAD_WARP_TRANSPOSE,
      may_alias ? LOAD_CA : LOAD_LDG,
      BLOCK_STORE_WARP_TRANSPOSE
    };
  }
};

} // namespace muh::tuning::adjacent_difference
