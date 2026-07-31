// muh/include/muh/tuning/tuning_radix_sort.cuh — BI-V100
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_radix_sort.cuh
// CCCL: 2383 lines, chained_policy architecture (not sm100_tuning structs)
//   Uses ONESWEEP algorithm on SM90+, fallback multi-pass on older.
//
// vllm relevance: top-p (nucleus) sampling sorts the full vocab logits
// SMEM risk: radix sort SMEM = ONESWEEP_RADIX_BITS^2 bins × sizeof(int) per warp
//   bits=8 → 256 bins × 4B × (threads/32 warps) = 256*4*(512/32) = 16384 (safe)
//   bits=11 → 2048 bins × 4B × 16 = 131072 (OVERFLOW at default threads)
//
// Strategy: use ONESWEEP=true with bits=8 (safe), not 11

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::radix_sort {

struct RadixSortHistogramPolicy {
  int threads_per_block;
  int items_per_thread;
  int num_parts;
};

struct RadixSortOnesweepPolicy {
  int threads_per_block;
  int items_per_thread;
  int radix_bits;
  int rank_algorithm;  // 0=MATCH, 1=MATCH_EARLY_COUNTS_ANY
  BlockStoreAlgorithm store_algorithm;
  int portioned_smem_per_warp;
};

struct RadixSortExclusiveSumPolicy {
  int threads_per_block;
  int radix_bits;
};

struct RadixSortDownsweepPolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  int radix_bits;
  BlockScanAlgorithm scan_algorithm;
};

struct RadixSortPolicy {
  bool onesweep;
  int primary_radix_bits;
  int single_tile_radix_bits;
  int segmented_radix_bits;
  RadixSortHistogramPolicy histogram;
  RadixSortExclusiveSumPolicy exclusive_sum;
  RadixSortOnesweepPolicy onesweep_policy;
  RadixSortDownsweepPolicy downsweep;
};

struct policy_selector {
  int key_size;
  int value_size;
  bool keys_only;

  constexpr RadixSortPolicy operator()(const hardware_capability& hw) const {
    bool is_onesweep = true;  // SM90+ equivalent for BI-V100

    // SMEM-safe radix bits: 8 for all key sizes on BI-V100
    // bits=11 would need 2048 bins × warps × 4B → overflow
    int onesweep_bits = 8;
    int primary_bits = (key_size > 1) ? 7 : 5;
    int single_tile_bits = (key_size > 1) ? 6 : 5;
    int segmented_bits = (key_size > 1) ? 6 : 5;

    int hist_items = (4 * 4) / key_size;
    if (hist_items < 1) hist_items = 1;

    int sweep_items = hist_items;

    // Downsweep (fallback for non-onesweep path)
    int ds_items = (4 * 4) / key_size;
    if (ds_items < 1) ds_items = 1;

    return {
      is_onesweep,
      primary_bits,
      single_tile_bits,
      segmented_bits,
      // histogram
      {256, hist_items, 1},
      // exclusive_sum
      {256, onesweep_bits},
      // onesweep
      {256, sweep_items, onesweep_bits, 1, BLOCK_STORE_DIRECT,
       hw.max_shared_memory_per_block / (256 / hw.warp_size)},
      // downsweep
      {256, ds_items, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
       primary_bits, BLOCK_SCAN_WARP_SCANS},
    };
  }
};

} // namespace muh::tuning::radix_sort
