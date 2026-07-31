// muh/include/muh/tuning/tuning_radix_sort.cuh — BI-V100
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_radix_sort.cuh
// CCCL: 2383 lines, chained_policy architecture with ONESWEEP on SM90+.
//
// SMEM analysis for ONESWEEP:
//   TempStorage_ is a union of:
//     keys_out[TILE_ITEMS] = threads * items * sizeof(key_type)
//     values_out[TILE_ITEMS] = threads * items * sizeof(value_type)
//     rank_temp_storage (from BlockRadixRank)
//   PLUS global_offsets[RADIX_DIGITS] = (1 << bits) * sizeof(OffsetT)
//
//   For bits=8, threads=256, items=4, key=float32:
//     keys_out = 256*4*4 = 4096
//     offsets = 256*8 = 2048 (OffsetT=int64)
//     total ≈ 6144 (safe)
//
//   For bits=11: offsets = 2048*8 = 16384. rank_temp_storage with
//     MATCH_EARLY_COUNTS uses per-warp privatized bins: 2048*num_parts*4.
//     At num_parts=1, threads=256: just offsets + rank already ~32KB.
//     But TILE_ITEMS = 256*4*4 = 4096 in the union, so total ≈ 36KB.
//     Tight but might fit. However, rank_temp_storage for MATCH_EARLY_COUNTS
//     with num_parts>1 can push past 48KB. Use bits=8 to be safe.
//
// vllm relevance: top-p (nucleus) sampling sorts full vocab (152064 logits)

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::radix_sort {

// CCCL's actual RadixSortOnesweepPolicy fields (from tuning_radix_sort.cuh):
//   threads_per_block, items_per_thread, store_algorithm, rank_algorithm,
//   scan_algorithm, rank_private_partitions, radix_bits

enum class RadixSortStoreAlgo { DIRECT, ALIGNED };
enum class RadixRankAlgo { MATCH, MATCH_EARLY_COUNTS_ANY, MATCH_EARLY_COUNTS_ATOMIC_OR };

struct RadixSortOnesweepPolicy {
  int threads_per_block;
  int items_per_thread;
  RadixSortStoreAlgo store_algorithm;
  RadixRankAlgo rank_algorithm;
  BlockScanAlgorithm scan_algorithm;
  int rank_private_partitions;
  int radix_bits;
};

struct RadixSortHistogramPolicy {
  int threads_per_block;
  int items_per_thread;
  int num_parts;
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
    // ONESWEEP with bits=8 is the safe choice for BI-V100.
    // bits=11 risks SMEM overflow in rank_temp_storage with multiple partitions.
    constexpr int onesweep_bits = 8;
    int primary_bits = (key_size > 1) ? 7 : 5;
    int single_tile_bits = (key_size > 1) ? 6 : 5;
    int segmented_bits = (key_size > 1) ? 6 : 5;

    // items: 16 bytes per thread / key_size
    int items = 16 / key_size;
    if (items < 1) items = 1;

    // SMEM check for onesweep: max(keys_tile, values_tile) + offsets
    // keys_tile = threads * items * key_size
    // offsets = (1 << bits) * 8  (OffsetT = int64)
    int threads = 256;
    int keys_tile = threads * items * key_size;
    int offsets = (1 << onesweep_bits) * 8;
    // rank_temp_storage: approximately radix_digits * sizeof(int) * num_parts
    int rank_smem = (1 << onesweep_bits) * 4 * 1;  // num_parts=1
    int total_smem = keys_tile + offsets + rank_smem;  // union: max(keys,values) not sum
    // Actually it is a union, so: max(keys_tile, values_tile, rank_smem) + offsets
    int val_tile = keys_only ? 0 : threads * items * value_size;
    int main_union = keys_tile;
    if (val_tile > main_union) main_union = val_tile;
    if (rank_smem > main_union) main_union = rank_smem;
    total_smem = main_union + offsets;

    while (total_smem > hw.max_shared_memory_per_block - 2048 && items > 1) {
      // Leave 2KB headroom for kernel stack/locals
      items--;
      keys_tile = threads * items * key_size;
      val_tile = keys_only ? 0 : threads * items * value_size;
      main_union = keys_tile > val_tile ? keys_tile : val_tile;
      if (rank_smem > main_union) main_union = rank_smem;
      total_smem = main_union + offsets;
    }

    return {
      true,  // onesweep
      primary_bits,
      single_tile_bits,
      segmented_bits,
      // histogram: same threads/items as onesweep
      {threads, items, 1},
      // exclusive_sum
      {256, onesweep_bits},
      // onesweep
      {threads, items,
       RadixSortStoreAlgo::DIRECT,
       RadixRankAlgo::MATCH_EARLY_COUNTS_ANY,
       BLOCK_SCAN_WARP_SCANS,
       1,  // rank_private_partitions: 1 to minimize SMEM
       onesweep_bits},
      // downsweep (fallback for non-onesweep)
      {256, items, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
       primary_bits, BLOCK_SCAN_WARP_SCANS},
    };
  }
};

} // namespace muh::tuning::radix_sort
