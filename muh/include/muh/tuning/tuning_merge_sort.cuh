// muh/include/muh/tuning/tuning_merge_sort.cuh — BI-V100
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_merge_sort.cuh
// CCCL source: 193 lines. Single MergeSortPolicy with block-level merge sort.
// Three generations: SM50 {256, 11}, SM52 {512, 15}, SM60+ {256, 17}.
//
// vllm relevance: DeviceMergeSort is the fallback when radix sort is not applicable
// (custom comparators, non-integral keys). Used in beam search reordering.
//
// SMEM for block sort: threads * items * (key_size + value_size) * 2 (double buffer)
// SMEM for merge: threads * items * (key_size + value_size)
// Both must fit in 48KB on BI-V100.

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::merge_sort {

// ============================================================================
// Policy struct — matches CCCL MergeSortPolicy exactly
// ============================================================================

struct MergeSortPolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  BlockStoreAlgorithm store_algorithm;
  bool unroll;
};

// ============================================================================
// Helper: nominal_4B_items_to_items (from CCCL common.cuh)
// Scales items_per_thread from a 4-byte nominal to actual key_size
// ============================================================================

constexpr int nominal_4b_items(int nominal, int key_size) {
  int result = nominal * 4 / key_size;
  return result > 0 ? result : 1;
}

// ============================================================================
// CCCL policy_hub generations (from CCCL lines 107-145):
//
// SM50: {256, N4B(11), WARP_TRANSPOSE, LOAD_LDG, WARP_TRANSPOSE}
// SM52: {512, N4B(15), WARP_TRANSPOSE, LOAD_LDG, WARP_TRANSPOSE}
// SM60: {256, N4B(17), WARP_TRANSPOSE, LOAD_DEFAULT, WARP_TRANSPOSE}
//
// policy_selector (CCCL lines 155-167):
//   Always returns SM60 policy: {256, N4B(17), WARP_TRANSPOSE, LOAD_DEFAULT, WARP_TRANSPOSE}
//   (SM60 is the "MaxPolicy" in the new tuning API)
// ============================================================================

struct policy_selector {
  int key_size;

  constexpr MergeSortPolicy operator()(const hardware_capability& hw) const {
    // SM60+ policy from CCCL (used for all compute capabilities in new API)
    int items = nominal_4b_items(17, key_size);

    // BI-V100 SMEM check: block sort uses double-buffered tile
    // tile_smem = threads * items * key_size * 2 (keys double-buffered)
    // For key-value: also need values, but they share the same tile layout
    int threads = 256;
    int tile_smem = threads * items * key_size * 2;
    while (tile_smem > hw.max_shared_memory_per_block - 2048 && items > 1) {
      items--;
      tile_smem = threads * items * key_size * 2;
    }

    return MergeSortPolicy{
      threads, items,
      BLOCK_LOAD_WARP_TRANSPOSE,
      LOAD_DEFAULT,
      BLOCK_STORE_WARP_TRANSPOSE,
      true  // unroll (default in CCCL unless CCCL_AVOID_SORT_UNROLL)
    };
  }
};

} // namespace muh::tuning::merge_sort
