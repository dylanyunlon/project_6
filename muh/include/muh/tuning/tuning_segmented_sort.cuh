// muh/include/muh/tuning/tuning_segmented_sort.cuh — BI-V100
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_segmented_sort.cuh
// CCCL source: 640 lines. Three-tier policy: large (radix sort), medium (sub-warp
// merge sort 16 threads), small (sub-warp merge sort 2-8 threads).
// Six generations of tuning: SM50/SM60/SM61/SM62/SM70/SM80/SM86.
//
// vllm relevance: per-sequence token ranking in beam search, top-k per segment.
// Segments = sequences in a batch; each segment = vocab_size logits.
//
// BI-V100: 16 SMs, warp_size=32, 48KB SMEM.
// SMEM for large (radix sort): threads * items * dominant_size + rank_smem
// SMEM for medium/small (merge sort): segments_per_block * items_per_tile * dominant_size * 2
//   (double-buffered: keys + values or keys + indices)

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::segmented_sort {

// ============================================================================
// Policy structs (matching CCCL exactly)
// ============================================================================

enum class RadixRankAlgorithm { BASIC, MEMOIZE, MATCH };
enum class WarpLoadAlgorithm { DIRECT, TRANSPOSE };
enum class WarpStoreAlgorithm { DIRECT, TRANSPOSE };

struct SegmentedSortRadixSortPolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  RadixRankAlgorithm rank_algorithm;
  BlockScanAlgorithm scan_algorithm;
  int radix_bits;
};

struct SegmentedSortSubWarpMergeSortPolicy {
  int threads_per_block;
  int threads_per_warp;   // threads assigned to sort one segment
  int items_per_thread;
  WarpLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  WarpStoreAlgorithm store_algorithm;

  constexpr int segments_per_block() const { return threads_per_block / threads_per_warp; }
  constexpr int items_per_tile() const { return threads_per_warp * items_per_thread; }
};

struct SegmentedSortPolicy {
  SegmentedSortRadixSortPolicy large_segment;
  SegmentedSortSubWarpMergeSortPolicy medium_segment;
  SegmentedSortSubWarpMergeSortPolicy small_segment;
  int partitioning_threshold;
};

// ============================================================================
// Helper: scale items from nominal 4B type to actual dominant_size
// CCCL: Nominal4BItemsToItems<DominantT>(N) = max(1, N * 4 / sizeof(DominantT))
// ============================================================================

constexpr int nominal_4b_items_to_items(int nominal_items, int dominant_size) {
  int result = nominal_items * 4 / dominant_size;
  return result > 0 ? result : 1;
}

// ============================================================================
// Helper: scale radix sort items by register pressure (from CCCL scale_reg_bound)
// ============================================================================

struct scaled_policy { int threads_per_block; int items_per_thread; };

constexpr scaled_policy scale_reg_bound(int nom_threads, int nom_items, int dom_size) {
  int items = nom_items * 4 / (dom_size > 0 ? dom_size : 4);
  if (items < 1) items = 1;
  if (items > nom_items * 2) items = nom_items * 2;
  return {nom_threads, items};
}

// ============================================================================
// policy_selector — BI-V100 uses SM86 tuning (closest match to SM86/A40)
// with SMEM 48KB constraints applied
//
// CCCL SM86 policy (lines 205-223):
//   large: {256, 23, BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT, RADIX_RANK_MEMOIZE,
//           BLOCK_SCAN_WARP_SCANS, radix_bits=(key>1B ? 6 : 4)}
//   medium: {256, 16 threads_per_warp, medium_itp, WARP_LOAD_TRANSPOSE,
//            LOAD_LDG, WARP_STORE_DIRECT}
//   small:  {256, large_items?8:2 threads_per_warp, small_itp,
//            WARP_LOAD_TRANSPOSE, LOAD_LDG, WARP_STORE_DIRECT}
//   partitioning_threshold = 500
//
// CCCL SM80 policy (lines 186-205):
//   large: same as SM86
//   medium: {256, 32 threads_per_warp, medium_itp, WARP_LOAD_TRANSPOSE,
//            LOAD_DEFAULT, WARP_STORE_DIRECT}
//   small:  {256, keys_only?4:2, small_itp, WARP_LOAD_TRANSPOSE,
//            LOAD_DEFAULT, WARP_STORE_DIRECT}
//
// CCCL SM70 policy (lines 167-186):
//   large: {256, 19, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, RADIX_RANK_MEMOIZE,
//           BLOCK_SCAN_WARP_SCANS, radix_bits}
//   medium: {256, 32, medium_itp, WARP_LOAD_DIRECT, LOAD_DEFAULT}
//   small:  {256, keys_only?4:8, small_itp, WARP_LOAD_DIRECT, LOAD_DEFAULT}
// ============================================================================

struct policy_selector {
  int key_size;
  int value_size;
  bool keys_only;

  constexpr int dominant_size() const {
    return value_size > key_size ? value_size : key_size;
  }

  constexpr SegmentedSortPolicy operator()(const hardware_capability& hw) const {
    int dom = dominant_size();
    bool large_items = dom > 4;
    int radix_bits = key_size > 1 ? 6 : 4;

    // --- Large segment: radix sort ---
    // SM86 tuning: {256, 23} scaled by dominant size
    auto lg_scaled = scale_reg_bound(256, 23, dom);

    // SMEM check for large: threads * items * dom + rank tables
    // rank tables (MEMOIZE): (1 << radix_bits) * sizeof(int) * WARP_THREADS = 2^6 * 4 * 32 = 8192B
    // tile: lg_threads * lg_items * dom
    int lg_t = lg_scaled.threads_per_block;
    int lg_i = lg_scaled.items_per_thread;
    int rank_smem = (1 << radix_bits) * 4 * (lg_t / 32); // per-warp privatized
    int lg_tile = lg_t * lg_i * dom;
    while (lg_tile + rank_smem > hw.max_shared_memory_per_block - 2048 && lg_i > 1) {
      lg_i--;
      lg_tile = lg_t * lg_i * dom;
    }

    auto large = SegmentedSortRadixSortPolicy{
      lg_t, lg_i,
      BLOCK_LOAD_TRANSPOSE, LOAD_DEFAULT,
      RadixRankAlgorithm::MEMOIZE, BLOCK_SCAN_WARP_SCANS,
      radix_bits
    };

    // --- Medium segment: sub-warp merge sort, 16 threads per segment ---
    // SM86: threads_per_warp=16, medium_itp depends on large_items
    int medium_itp = nominal_4b_items_to_items(large_items ? 9 : 7, dom);
    // SMEM: segments_per_block * items_per_tile * dom * 2 (double buffer)
    // segments_per_block = 256/16 = 16
    // items_per_tile = 16 * medium_itp
    int med_segs = 256 / 16;
    int med_tile = 16 * medium_itp;
    int med_smem = med_segs * med_tile * dom * 2;
    while (med_smem > hw.max_shared_memory_per_block - 2048 && medium_itp > 1) {
      medium_itp--;
      med_tile = 16 * medium_itp;
      med_smem = med_segs * med_tile * dom * 2;
    }

    auto medium = SegmentedSortSubWarpMergeSortPolicy{
      256, 16, medium_itp,
      WarpLoadAlgorithm::TRANSPOSE, LOAD_LDG, WarpStoreAlgorithm::DIRECT
    };

    // --- Small segment: sub-warp merge sort, 2-8 threads per segment ---
    // SM86: threads_per_warp = large_items ? 8 : 2
    int small_tpw = large_items ? 8 : 2;
    int small_itp = nominal_4b_items_to_items(large_items ? 7 : 9, dom);
    int sm_segs = 256 / small_tpw;
    int sm_tile = small_tpw * small_itp;
    int sm_smem = sm_segs * sm_tile * dom * 2;
    while (sm_smem > hw.max_shared_memory_per_block - 2048 && small_itp > 1) {
      small_itp--;
      sm_tile = small_tpw * small_itp;
      sm_smem = sm_segs * sm_tile * dom * 2;
    }

    auto small = SegmentedSortSubWarpMergeSortPolicy{
      256, small_tpw, small_itp,
      WarpLoadAlgorithm::TRANSPOSE, LOAD_LDG, WarpStoreAlgorithm::DIRECT
    };

    return SegmentedSortPolicy{large, medium, small, 500};
  }
};

} // namespace muh::tuning::segmented_sort
