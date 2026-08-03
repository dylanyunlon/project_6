// muh/include/muh/tuning/tuning_merge.cuh — BI-V100
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_merge.cuh
// CCCL source: 180 lines. DeviceMerge tuning policy with bulk copy (cp.async.bulk)
// support on SM90+. Five generations: SM52/SM60/SM80/SM90/SM100.
//
// vllm relevance: DeviceMerge is used in beam search candidate merging,
// sorted KV cache compaction, and prefix-sharing merge operations.
//
// SMEM: threads * items * (key_size + value_size) for merge path tile.
// On SM90+, bulk copy (bl2sh) can bypass L1 for aligned trivially-relocatable types.
// BI-V100 does NOT have cp.async.bulk → use_bulk_copy = false always.

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::merge {

// ============================================================================
// Policy struct — matches CCCL MergePolicy exactly
// ============================================================================

struct MergePolicy {
  int threads_per_block;
  int items_per_thread;
  CacheLoadModifier load_modifier;
  BlockStoreAlgorithm store_algorithm;
  bool use_bulk_copy_for_keys;    // cp.async.bulk: SM90+ only, NOT available on BI-V100
  bool use_bulk_copy_for_values;  // cp.async.bulk: SM90+ only, NOT available on BI-V100
  bool unroll;
};

// ============================================================================
// CCCL policy_selector operator() — five CC tiers:
//
// SM100+: {512, N4B(15, key+val), LOAD_DEFAULT, WARP_TRANSPOSE, bulk_keys, bulk_vals}
// SM90:   {512, N4B(15, key+val), LOAD_DEFAULT, WARP_TRANSPOSE, conditional_bulk, conditional_bulk}
//   - bulk keys: key_size != 8 && aligned && trivially_relocatable && contiguous
//   - bulk pairs: complex conditions on key_size/value_size combinations
// SM80:   {512, N4B(15, key+val), LOAD_DEFAULT, WARP_TRANSPOSE, conditional_bulk, conditional_bulk}
//   - bulk keys: key_size < 4
//   - bulk pairs: key==1 || (key==2 && val<4) || (key==4 && val==1)
// SM60:   {512, N4B(15, key+val), LOAD_DEFAULT, WARP_TRANSPOSE, false, false}
// SM52:   {512, N4B(13, key+val), LOAD_LDG, WARP_TRANSPOSE, false, false}
//
// BI-V100: use SM80 items (N4B(15)), but bulk_copy=false (no cp.async.bulk).
// ============================================================================

constexpr int nominal_4b_items(int nominal, int combined_size) {
  int result = nominal * 4 / combined_size;
  return result > 0 ? result : 1;
}

struct policy_selector {
  int key_size;
  int value_size;  // 0 for keys-only
  int offset_size;

  constexpr MergePolicy operator()(const hardware_capability& hw) const {
    int combined = key_size + value_size;
    int items = nominal_4b_items(15, combined);

    // BI-V100 SMEM check: tile = threads * items * combined
    int threads = 512;
    int tile_smem = threads * items * combined;
    while (tile_smem > hw.max_shared_memory_per_block - 2048 && items > 1) {
      items--;
      tile_smem = threads * items * combined;
    }
    // If still too large, reduce threads
    while (tile_smem > hw.max_shared_memory_per_block - 2048 && threads > 128) {
      threads -= 128;
      tile_smem = threads * items * combined;
    }

    return MergePolicy{
      threads, items,
      LOAD_DEFAULT,
      BLOCK_STORE_WARP_TRANSPOSE,
      false,  // no cp.async.bulk on BI-V100
      false,  // no cp.async.bulk on BI-V100
      true    // unroll
    };
  }
};

} // namespace muh::tuning::merge
