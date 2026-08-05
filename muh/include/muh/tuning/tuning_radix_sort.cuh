// muh/include/muh/tuning/tuning_radix_sort.cuh — BI-V100
//
// Full port from: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_radix_sort.cuh (2381 lines)
//
// CCCL radix sort has 10 sub-policies in RadixSortPolicy:
//   algorithm, histogram, exclusive_sum, onesweep, scan,
//   downsweep, alt_downsweep, upsweep, alt_upsweep, single_tile
//
// Two algorithms:
//   onesweep: single-pass using decoupled lookback (SM60+, key≥4B)
//   multi_pass: traditional upsweep → scan → downsweep (key<4B or old SM)
//
// BI-V100 strategy:
//   Based on SM70 (V100) policy — closest hardware match:
//   - SM=80 (V100) vs SM=16 (BI-V100): both have HBM2, similar cache hierarchy
//   - onesweep for key≥4B, multi_pass for key<4B
//   - rank_private_partitions=4 (SM70 value, conservatively handles BI-V100 atomic perf)
//   - onesweep items scaled for 16 SMs (fewer CTAs → each processes more)
//
// Factory functions use scale_reg_bound (register-bound scaling, NOT SMEM-bound)
// because radix sort is register-intensive (keys/values in registers during ranking)
//
// SMEM in radix sort is used for:
//   - Histogram counting (histogram pass): num_bins * sizeof(int) per partition
//   - Rank arrays (onesweep/downsweep): threads * sizeof(int) for digit counts
//   - Key/value scatter staging
// These are all much smaller than scan's BlockLoad SMEM, so 48KB is not the binding constraint.

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::radix_sort {

// Re-export CCCL enum types needed for radix sort
enum class RadixSortAlgorithm { multi_pass, onesweep };
enum RadixSortStoreAlgorithm { RADIX_SORT_STORE_DIRECT, RADIX_SORT_STORE_STRIPED };
enum RadixRankAlgorithm {
  RADIX_RANK_BASIC,
  RADIX_RANK_MEMOIZE,
  RADIX_RANK_MATCH,
  RADIX_RANK_MATCH_EARLY_COUNTS_ANY
};

struct RadixSortHistogramPolicy {
  int threads_per_block;
  int items_per_thread;
  int private_partitions;
  int radix_bits;
};

struct RadixSortExclusiveSumPolicy {
  int threads_per_block;
  int radix_bits;
};

struct RadixSortOnesweepPolicy {
  int threads_per_block;
  int items_per_thread;
  RadixSortStoreAlgorithm store_algorithm;
  RadixRankAlgorithm rank_algorithm;
  BlockScanAlgorithm scan_algorithm;
  int rank_private_partitions;
  int radix_bits;
};

struct RadixSortDownsweepPolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  RadixRankAlgorithm rank_algorithm;
  BlockScanAlgorithm scan_algorithm;
  int radix_bits;
};

struct RadixSortUpsweepPolicy {
  int threads_per_block;
  int items_per_thread;
  CacheLoadModifier load_modifier;
  int radix_bits;
};

// Forward-declare ScanPolicy from tuning_scan.cuh
// (radix sort's multi_pass uses a scan sub-pass)
struct ScanPolicyForSort {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  BlockStoreAlgorithm store_algorithm;
  BlockScanAlgorithm scan_algorithm;
};

struct RadixSortPolicy {
  RadixSortAlgorithm algorithm;
  RadixSortHistogramPolicy histogram;
  RadixSortExclusiveSumPolicy exclusive_sum;
  RadixSortOnesweepPolicy onesweep;
  ScanPolicyForSort scan;
  RadixSortDownsweepPolicy downsweep;
  RadixSortDownsweepPolicy alt_downsweep;
  RadixSortUpsweepPolicy upsweep;
  RadixSortUpsweepPolicy alt_upsweep;
  RadixSortDownsweepPolicy single_tile;
};

// Register-bound scaling (from CCCL common.cuh scale_reg_bound)
// Unlike scale_mem_bound, this accounts for register file pressure
// items = nominal_4B_items * 4 / max(type_size, 4), clamped to [1, nominal]
struct reg_scaled {
  int threads_per_block;
  int items_per_thread;
};

constexpr reg_scaled scale_reg(int nominal_threads, int nominal_4b_items, int type_size) {
  int items = nominal_4b_items * 4 / (type_size > 4 ? type_size : 4);
  if (items < 1) items = 1;
  if (items > nominal_4b_items) items = nominal_4b_items;
  return {nominal_threads, items};
}

// Scale histogram private_partitions: more partitions for small types
constexpr int scale_num_parts(int nominal_4b_parts, int compute_size) {
  int p = nominal_4b_parts * 4 / (compute_size > 4 ? compute_size : 4);
  return p > 0 ? p : 1;
}

struct policy_selector {
  int key_size;
  int value_size;  // 0 = keys-only
  int offset_size;

  constexpr bool keys_only() const { return value_size == 0; }
  constexpr int dominant_size() const { return key_size > value_size ? key_size : value_size; }

  constexpr RadixSortPolicy operator()(const hardware_capability& hw) const {
    // BI-V100: use SM70 (V100) strategy
    // onesweep for key≥4B, multi_pass for key<4B
    int primary_radix_bits = (key_size > 1) ? 7 : 5;
    int single_tile_radix_bits = (key_size > 1) ? 6 : 5;
    auto algo = (key_size >= 4) ? RadixSortAlgorithm::onesweep : RadixSortAlgorithm::multi_pass;
    int onesweep_radix_bits = 8;
    bool offset_64bit = (offset_size == 8);
    int ds = dominant_size();

    // Histogram: SM70 style — 256 threads, 8 items, scale partitions
    auto histogram = RadixSortHistogramPolicy{
      256, 8, scale_num_parts(8, key_size), onesweep_radix_bits};

    auto exclusive_sum = RadixSortExclusiveSumPolicy{256, onesweep_radix_bits};

    // Onesweep: SM70 style — 256 threads, items depends on key/value sizes
    // SM70 special case: key=4B value=4B → items=46 (much higher than default 23)
    int onesweep_nominal = (key_size == 4 && value_size == 4) ? 46 : 23;
    auto [os_t, os_i] = scale_reg(256, onesweep_nominal, ds);
    auto onesweep_p = RadixSortOnesweepPolicy{
      os_t, os_i,
      RADIX_SORT_STORE_DIRECT,
      RADIX_RANK_MATCH_EARLY_COUNTS_ANY,
      BLOCK_SCAN_WARP_SCANS,
      4,  // SM70 rank_private_partitions (SM80+ uses 1, conservative for BI-V100)
      onesweep_radix_bits};

    // Scan: shared with scan tuning — 512 threads, 23 items (memory-bound scaled)
    auto [sc_i, sc_t] = scale_mem_bound(512, 23, offset_size);
    auto scan_p = ScanPolicyForSort{
      sc_t, sc_i,
      BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
      BLOCK_STORE_WARP_TRANSPOSE, BLOCK_SCAN_RAKING_MEMOIZE};

    // Downsweep: SM70 — 512 threads, 23 items, RADIX_RANK_MATCH
    auto [ds_t, ds_i] = scale_reg(512, 23, ds);
    auto downsweep_p = RadixSortDownsweepPolicy{
      ds_t, ds_i,
      BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
      RADIX_RANK_MATCH, BLOCK_SCAN_WARP_SCANS,
      primary_radix_bits};

    // Alt downsweep: fewer radix bits, more items for residual digits
    int alt_nominal = offset_64bit ? 46 : 47;
    int alt_threads = (key_size > 1) ? 256 : 128;
    auto [ad_t, ad_i] = scale_reg(alt_threads, alt_nominal, ds);
    auto alt_downsweep_p = RadixSortDownsweepPolicy{
      ad_t, ad_i,
      BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
      RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS,
      primary_radix_bits - 1};

    // Upsweep: mirrors downsweep params
    auto [us_t, us_i] = scale_reg(256, 23, ds);
    auto upsweep_p = RadixSortUpsweepPolicy{us_t, us_i, LOAD_DEFAULT, primary_radix_bits};

    auto [au_t, au_i] = scale_reg(256, alt_nominal, ds);
    auto alt_upsweep_p = RadixSortUpsweepPolicy{au_t, au_i, LOAD_DEFAULT, primary_radix_bits - 1};

    // Single tile: small inputs, one block
    auto [st_t, st_i] = scale_reg(256, 19, ds);
    auto single_tile_p = RadixSortDownsweepPolicy{
      st_t, st_i,
      BLOCK_LOAD_DIRECT, LOAD_LDG,
      RADIX_RANK_MEMOIZE, BLOCK_SCAN_WARP_SCANS,
      single_tile_radix_bits};

    return {algo, histogram, exclusive_sum, onesweep_p, scan_p,
            downsweep_p, alt_downsweep_p, upsweep_p, alt_upsweep_p, single_tile_p};
  }
};

} // namespace muh::tuning::radix_sort
