// muh/include/muh/tuning/tuning_radix_sort.cuh — BI-V100
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_radix_sort.cuh
// CCCL source: 2381 lines. This muh version ports the complete SM90/SM100
// tuning tables and policy_selector dispatch logic, with BI-V100 SMEM 48KB
// constraints applied.
//
// vllm relevance: top-p/top-k sampling sorts full vocab (152064 logits)
// every decode step. Output TPS weight = 83% of competition score.
//
// SMEM analysis for ONESWEEP on BI-V100:
//   TempStorage_ is a union of:
//     keys_out[TILE_ITEMS] = threads * items * sizeof(KeyT)
//     values_out[TILE_ITEMS] = threads * items * sizeof(ValueT)
//     rank_temp_storage (BlockRadixRank)
//   PLUS global_offsets[(1 << bits)] * sizeof(OffsetT)
//
//   For bits=8: offsets = 256*8 = 2048B
//   For bits=11: offsets = 2048*8 = 16384B → too expensive
//   → BI-V100 uses bits=8 for all key sizes

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::radix_sort {

// ============================================================================
// Policy structs (matching CCCL exactly)
// ============================================================================

enum class RadixSortAlgorithm { multi_pass, onesweep };
enum class RadixSortStoreAlgo { DIRECT, ALIGNED };
enum class RadixRankAlgo {
  BASIC, MEMOIZE, MATCH, MATCH_EARLY_COUNTS_ANY, MATCH_EARLY_COUNTS_ATOMIC_OR
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
  RadixSortStoreAlgo store_algorithm;
  RadixRankAlgo rank_algorithm;
  BlockScanAlgorithm scan_algorithm;
  int rank_private_partitions;
  int radix_bits;
};

struct RadixSortDownsweepPolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  RadixRankAlgo rank_algorithm;
  BlockScanAlgorithm scan_algorithm;
  int radix_bits;
};

struct RadixSortUpsweepPolicy {
  int threads_per_block;
  int items_per_thread;
  CacheLoadModifier load_modifier;
  int radix_bits;
};

struct RadixSortPolicy {
  RadixSortAlgorithm algorithm;
  RadixSortHistogramPolicy histogram;
  RadixSortExclusiveSumPolicy exclusive_sum;
  RadixSortOnesweepPolicy onesweep;
  ScanPolicy scan;
  RadixSortDownsweepPolicy downsweep;
  RadixSortDownsweepPolicy alt_downsweep;
  RadixSortUpsweepPolicy upsweep;
  RadixSortUpsweepPolicy alt_upsweep;
  RadixSortDownsweepPolicy single_tile;
};

struct small_key_tuning_values {
  int threads;
  int items;
};

// ============================================================================
// SM90 tuning table — complete from CCCL tuning_radix_sort.cuh:353-391
// ============================================================================

constexpr auto get_sm90_tuning(int key_size, int value_size, int offset_size)
  -> small_key_tuning_values
{
  // keys-only
  if (value_size == 0) {
    if (key_size == 1 && offset_size == 4) return {512,19};
    if (key_size == 1 && offset_size == 8) return {512,19};
    if (key_size == 2 && offset_size == 4) return {512,19};
    if (key_size == 2 && offset_size == 8) return {512,19};
  }

  // pairs 1-byte key
  if (key_size == 1) {
    if (value_size ==  1 && offset_size == 4) return {512, 15};
    if (value_size ==  1 && offset_size == 8) return {448, 16};
    if (value_size ==  2 && offset_size == 4) return {512, 17};
    if (value_size ==  2 && offset_size == 8) return {512, 14};
    if (value_size ==  4 && offset_size == 4) return {512, 17};
    if (value_size ==  4 && offset_size == 8) return {512, 14};
    if (value_size ==  8 && offset_size == 4) return {384, 23};
    if (value_size ==  8 && offset_size == 8) return {384, 18};
    if (value_size == 16 && offset_size == 4) return {512, 22};
    if (value_size == 16 && offset_size == 8) return {512, 22};
  }

  // pairs 2-byte key
  if (key_size == 2) {
    if (value_size ==  1 && offset_size == 4) return {384, 14};
    if (value_size ==  1 && offset_size == 8) return {384, 16};
    if (value_size ==  2 && offset_size == 4) return {384, 15};
    if (value_size ==  2 && offset_size == 8) return {448, 16};
    if (value_size ==  4 && offset_size == 4) return {512, 17};
    if (value_size ==  4 && offset_size == 8) return {512, 12};
    if (value_size ==  8 && offset_size == 4) return {384, 23};
    if (value_size ==  8 && offset_size == 8) return {512, 23};
    if (value_size == 16 && offset_size == 4) return {512, 21};
    if (value_size == 16 && offset_size == 8) return {576, 22};
  }

  // default fallback
  return {384, 23};
}

// ============================================================================
// SM100 tuning table — complete from CCCL tuning_radix_sort.cuh:395-850
// Falls back to SM90 for entries marked "same as previous tuning"
// Includes benchmark annotations: ipt_N.tpb_M speedup0 speedup1 speedup2 speedup3
// ============================================================================

constexpr auto get_sm100_tuning(int key_size, int value_size, int offset_size,
                                type_t key_type = type_t::unknown)
  -> small_key_tuning_values
{
  // keys-only
  if (value_size == 0) {
    if (offset_size == 4) {
      // key_size==1: same as SM90
      // ipt_20.tpb_512 1.013282 0.967525 1.015764 1.047982
      if (key_size == 2) return {512,20};
      // ipt_20.tpb_512 1.089698 0.979276 1.079822 1.199378
      if (key_size == 4 && key_type == type_t::float32) return {512,20};
      // ipt_21.tpb_512 1.002873 0.994608 1.004196 1.019301
      if (key_size == 4) return {512,21};
      // ipt_18.tpb_288 1.049258 0.985085 1.042400 1.107771
      if (key_size == 8 && key_type == type_t::float64) return {288,18};
      // ipt_14.tpb_320 1.256020 1.000000 1.228182 1.486711
      if (key_size == 8) return {320,14};
    } else if (offset_size == 8) {
      // key_size==1: same as SM90
      // ipt_20.tpb_384 1.038445 1.015608 1.037620 1.068105
      if (key_size == 2) return {384,20};
      // ipt_20.tpb_512 1.021557 0.981437 1.018920 1.039977
      if (key_size == 4 && key_type == type_t::float32) return {512,20};
      // key_size==4 default: same as SM90
      // ipt_21.tpb_256 1.068590 0.986635 1.059704 1.144921
      if (key_size == 8 && key_type == type_t::float64) return {256,21};
      // ipt_18.tpb_320 1.248354 1.000000 1.220666 1.446929
      if (key_size == 8) return {320,18};
    }
  }

  // pairs 1-byte key
  if (key_size == 1) {
    // offset_size == 4
    // value_size==1: same as SM90
    // ipt_18.tpb_512 1.011463 0.978807 1.010106 1.024056
    if (value_size == 2 && offset_size == 4) return {512,18};
    // ipt_18.tpb_512 1.008207 0.980377 1.007132 1.022155
    if (value_size == 4 && offset_size == 4) return {512,18};
    // value_size==8, offset_size==4: regresses for large problem sizes (commented in CCCL)
    // ipt_21.tpb_576 1.044274 0.979145 1.038723 1.072068
    if (value_size == 16 && offset_size == 4) return {576,21};

    // offset_size == 8
    // ipt_20.tpb_384 1.008881 0.968750 1.006846 1.026910
    if (value_size == 1 && offset_size == 8) return {384,20};
    // ipt_22.tpb_256 1.015597 0.966038 1.011167 1.045921
    if (value_size == 2 && offset_size == 8) return {256,22};
    // ipt_15.tpb_384 1.029730 0.972699 1.029066 1.067894
    if (value_size == 4 && offset_size == 8) return {384,15};
    // value_size==8, offset_size==8: regresses (commented in CCCL)
    // value_size==16, offset_size==8: same as SM90
  }

  // pairs 2-byte key
  if (key_size == 2) {
    // ipt_20.tpb_448 1.031929 0.936849 1.023411 1.075172
    if (value_size == 1 && offset_size == 4) return {448,20};
    // ipt_23.tpb_384 1.104683 0.939335 1.087342 1.234988
    if (value_size == 2 && offset_size == 4) return {384,23};
    // value_size==4, offset_size==4: same as SM90
    // value_size==8, offset_size==4: regresses (commented in CCCL)
    // value_size==16, offset_size==4: same as SM90
    // ipt_15.tpb_384 1.093598 1.000000 1.088111 1.183369
    if (value_size == 1 && offset_size == 8) return {384,15};
    // ipt_15.tpb_576 1.040476 1.000333 1.037060 1.084850
    if (value_size == 2 && offset_size == 8) return {576,15};
    // ipt_18.tpb_512 1.096819 0.953488 1.082026 1.209533
    if (value_size == 4 && offset_size == 8) return {512,18};
    // value_size==8, offset_size==8: regresses (commented in CCCL)
    // value_size==16, offset_size==8: same as SM90
  }

  // pairs 4-byte key (vllm hot path: float32 logits)
  if (key_size == 4) {
    // ipt_21.tpb_416 1.237956 1.001909 1.210882 1.469981
    if (value_size == 1 && offset_size == 4) return {416,21};
    // ipt_17.tpb_512 1.022121 1.012346 1.022439 1.038524
    if (value_size == 2 && offset_size == 4) return {512,17};
    // ipt_20.tpb_448 1.012688 0.999531 1.011865 1.028513
    if (value_size == 4 && offset_size == 4) return {448,20};
    // ipt_15.tpb_384 1.006872 0.998651 1.008374 1.026118
    if (value_size == 8 && offset_size == 4) return {384,15};
    // value_size==16, offset_size==4: same as SM90

    // ipt_17.tpb_512 1.080000 0.927362 1.066211 1.172959
    if (value_size == 1 && offset_size == 8) return {512,17};
    // ipt_15.tpb_384 1.068529 1.000000 1.062277 1.135281
    if (value_size == 2 && offset_size == 8) return {384,15};
    // ipt_21.tpb_448 1.080642 0.927713 1.064758 1.191177
    if (value_size == 4 && offset_size == 8) return {448,21};
    // ipt_13.tpb_448 1.019046 0.991228 1.016971 1.039712
    if (value_size == 8 && offset_size == 8) return {448,13};
    // value_size==16, offset_size==8: same as SM90
  }

  // pairs 8-byte key
  if (key_size == 8) {
    // ipt_17.tpb_256 1.276445 1.025562 1.248511 1.496947
    if (value_size == 1 && offset_size == 4) return {256,17};
    // ipt_12.tpb_352 1.128086 1.040000 1.117960 1.207254
    if (value_size == 2 && offset_size == 4) return {352,12};
    // ipt_12.tpb_352 1.132699 1.040000 1.122676 1.207716
    if (value_size == 4 && offset_size == 4) return {352,12};
    // ipt_18.tpb_256 1.266745 0.995432 1.237754 1.460538
    if (value_size == 8 && offset_size == 4) return {256,18};
    // value_size==16, offset_size==4: same as SM90

    // ipt_15.tpb_384 1.007343 0.997656 1.006929 1.047208
    if (value_size == 1 && offset_size == 8) return {384,15};
    // ipt_14.tpb_256 1.186477 1.012683 1.167150 1.332313
    if (value_size == 2 && offset_size == 8) return {256,14};
    // ipt_21.tpb_256 1.220607 1.000239 1.196400 1.390471
    if (value_size == 4 && offset_size == 8) return {256,21};
    // value_size==8, offset_size==8: same as SM90
    // value_size==16, offset_size==8: same as SM90
  }

  // fallback: delegate to SM90
  return get_sm90_tuning(key_size, value_size, offset_size);
}

// ============================================================================
// BI-V100 SMEM constraint: cap threads*items to fit in 48KB
// ONESWEEP SMEM = max(threads*items*key_size, threads*items*val_size,
//                     rank_temp_storage) + (1<<bits)*offset_size
// With bits=8, offset_size=8: offsets = 256*8 = 2048B
// With bits=8, offset_size=4: offsets = 256*4 = 1024B
// rank_temp_storage ≈ (1<<bits) * 4 * num_parts = 256*4*1 = 1024B
// Headroom: 2KB for kernel locals
// Effective limit for tile: 48KB - 2048 - 1024 - 2048 = 43008B
// ============================================================================

constexpr int BI100_SMEM_LIMIT = 49152;
constexpr int BI100_ONESWEEP_BITS = 8;
constexpr int BI100_HEADROOM = 2048;

constexpr auto bi100_smem_cap(small_key_tuning_values tuning,
                              int key_size, int value_size, int offset_size)
  -> small_key_tuning_values
{
  int offsets = (1 << BI100_ONESWEEP_BITS) * offset_size;
  int rank_smem = (1 << BI100_ONESWEEP_BITS) * 4; // num_parts=1
  int overhead = offsets + rank_smem + BI100_HEADROOM;
  int max_tile = BI100_SMEM_LIMIT - overhead;

  int dominant = key_size;
  if (value_size > dominant) dominant = value_size;

  int t = tuning.threads;
  int i = tuning.items;
  int tile = t * i * dominant;

  while (tile > max_tile && i > 1) {
    i--;
    tile = t * i * dominant;
  }
  while (tile > max_tile && t > 64) {
    t -= 32;
    tile = t * i * dominant;
  }

  return {t, i};
}

// ============================================================================
// BI-V100 tuning: start from SM100 values, apply SMEM cap
// SM100 tuning is the best available data point (SM100 ≈ B200, more
// recent than SM90). BI-V100 has 16 SMs (not 50), 48KB SMEM, 900 GB/s BW.
// We use SM100 as initial values and only reduce items when SMEM overflows.
// Actual BI-V100 benchmark data will replace these (project board: [muh-bench])
// ============================================================================

constexpr auto get_bi100_tuning(int key_size, int value_size, int offset_size,
                                type_t key_type = type_t::unknown)
  -> small_key_tuning_values
{
  auto sm100 = get_sm100_tuning(key_size, value_size, offset_size, key_type);
  return bi100_smem_cap(sm100, key_size, value_size, offset_size);
}

// ============================================================================
// policy_selector: matches CCCL's operator()(compute_capability) pattern
// ============================================================================

struct policy_selector {
  int key_size;
  int value_size;  // 0 for keys-only
  int offset_size;
  type_t key_type;

  constexpr bool keys_only() const { return value_size == 0; }
  constexpr int dominant_size() const {
    return value_size > key_size ? value_size : key_size;
  }

  // Scale onesweep items by register pressure (from CCCL make_reg_scaled_radix_sort_onesweep_policy)
  constexpr auto reg_scale_onesweep(int nominal_threads, int nominal_items,
                                    int dom_size) const
    -> small_key_tuning_values
  {
    // CCCL: items = clamp(nominal * 4 / dom_size, 1, nominal * 2)
    int items = nominal_items * 4 / (dom_size > 0 ? dom_size : 4);
    if (items < 1) items = 1;
    if (items > nominal_items * 2) items = nominal_items * 2;
    return {nominal_threads, items};
  }

  constexpr RadixSortPolicy operator()(const hardware_capability& hw) const {
    constexpr int onesweep_bits = BI100_ONESWEEP_BITS;
    int primary_bits = (key_size > 1) ? 7 : 5;
    int single_tile_bits = (key_size > 1) ? 6 : 5;
    int dom = dominant_size();

    // ---- Histogram policy ----
    int hist_num_parts = 4 / (key_size > 4 ? key_size : 4);
    if (hist_num_parts < 1) hist_num_parts = 1;
    auto histogram = RadixSortHistogramPolicy{128, 16, hist_num_parts, onesweep_bits};

    // ---- Exclusive sum policy ----
    auto exclusive_sum = RadixSortExclusiveSumPolicy{256, onesweep_bits};

    // ---- Onesweep policy ----
    // For small keys (<4B): use tuning table
    // For large keys (>=4B): use CCCL's formula-based approach
    RadixSortOnesweepPolicy onesweep;
    if (key_size < 4) {
      auto tuning = get_bi100_tuning(key_size, value_size, offset_size, key_type);
      onesweep = {tuning.threads, tuning.items,
                  RadixSortStoreAlgo::DIRECT,
                  RadixRankAlgo::MATCH_EARLY_COUNTS_ANY,
                  BLOCK_SCAN_RAKING_MEMOIZE,
                  1, onesweep_bits};
    } else if (key_size == 4) {
      // CCCL SM80 formula for 4B keys
      bool offset_64 = (offset_size == 8);
      bool is_float = (key_type == type_t::float32);
      int nom_items = keys_only()
        ? (20 - (int)offset_64 - (int)is_float)
        : (value_size < 8 ? (offset_64 ? 17 : 23) : (offset_64 ? 29 : 30));
      auto scaled = reg_scale_onesweep(384, nom_items, dom);
      auto capped = bi100_smem_cap(scaled, key_size, value_size, offset_size);
      onesweep = {capped.threads, capped.items,
                  RadixSortStoreAlgo::DIRECT,
                  RadixRankAlgo::MATCH_EARLY_COUNTS_ANY,
                  BLOCK_SCAN_RAKING_MEMOIZE,
                  1, onesweep_bits};
    } else {
      // 8B+ keys
      int nom_items = value_size < 8 ? 30 : 24;
      auto scaled = reg_scale_onesweep(384, nom_items, dom);
      auto capped = bi100_smem_cap(scaled, key_size, value_size, offset_size);
      onesweep = {capped.threads, capped.items,
                  RadixSortStoreAlgo::DIRECT,
                  RadixRankAlgo::MATCH_EARLY_COUNTS_ANY,
                  BLOCK_SCAN_RAKING_MEMOIZE,
                  1, onesweep_bits};
    }

    // ---- Scan policy (for onesweep internal scan) ----
    auto [scan_items, scan_threads] = scale_mem_bound(512, 23, offset_size);
    auto scan = ScanPolicy{
      ScanAlgorithm::lookback,
      ScanLookbackPolicy{
        scan_threads, scan_items,
        BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
        BLOCK_STORE_WARP_TRANSPOSE, BLOCK_SCAN_RAKING_MEMOIZE,
        {DelayAlgorithm::exponential_backon_jitter, 952, 498}  // SM100 * 0.5
      },
      {}
    };

    // ---- Downsweep (fallback for multi_pass) ----
    auto [ds_items, ds_threads] = scale_mem_bound(512, 23, dom);
    auto downsweep = RadixSortDownsweepPolicy{
      ds_threads, ds_items,
      BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT,
      RadixRankAlgo::MATCH, BLOCK_SCAN_WARP_SCANS,
      primary_bits};

    auto [alt_ds_items, alt_ds_threads] = scale_mem_bound(
      (key_size > 1) ? 256 : 128, 47, dom);
    auto alt_downsweep = RadixSortDownsweepPolicy{
      alt_ds_threads, alt_ds_items,
      BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT,
      RadixRankAlgo::MEMOIZE, BLOCK_SCAN_WARP_SCANS,
      primary_bits - 1};

    // ---- Upsweep ----
    auto [up_items, up_threads] = scale_mem_bound(256, 23, dom);
    auto upsweep = RadixSortUpsweepPolicy{up_threads, up_items, LOAD_DEFAULT, primary_bits};
    auto [alt_up_items, alt_up_threads] = scale_mem_bound(256, 47, dom);
    auto alt_upsweep = RadixSortUpsweepPolicy{alt_up_threads, alt_up_items, LOAD_DEFAULT, primary_bits - 1};

    // ---- Single tile ----
    auto [st_items, st_threads] = scale_mem_bound(256, 19, dom);
    auto single_tile = RadixSortDownsweepPolicy{
      st_threads, st_items,
      BLOCK_LOAD_DIRECT, LOAD_LDG,
      RadixRankAlgo::MEMOIZE, BLOCK_SCAN_WARP_SCANS,
      single_tile_bits};

    return RadixSortPolicy{
      // BI-V100: onesweep for key>=4B (matches SM80+), multi_pass for smaller
      key_size >= 4 ? RadixSortAlgorithm::onesweep : RadixSortAlgorithm::multi_pass,
      histogram, exclusive_sum, onesweep, scan,
      downsweep, alt_downsweep, upsweep, alt_upsweep, single_tile
    };
  }
};

} // namespace muh::tuning::radix_sort
