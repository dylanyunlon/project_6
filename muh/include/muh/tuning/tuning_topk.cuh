// muh/include/muh/tuning/tuning_topk.cuh — BI-V100 top-k tuning
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_topk.cuh
//
// vllm impact: Top-k / top-p sampling in decode stage
// Competition weight: Output TPS × 16.796 (highest priority, tied with reduce)
//
// CCCL SM90+ policy_selector (the ground truth):
//   bits_per_pass = calc_bits_per_pass(key_size):
//     key_size 1   → 8
//     key_size 2   → 8   (but note: CCCL default returns 11 for 2/4/8,
//                          only key_size=1 returns 8. See switch below.)
//     key_size 4   → 11
//     key_size 8   → 11
//   items_per_thread = max(1, 4 * 4 / key_size)  // 16 bytes per thread
//   threads_per_block = 512
//   load_algorithm = BLOCK_LOAD_VECTORIZE  (NOT BLOCK_LOAD_DIRECT)
//   scan_algorithm = BLOCK_SCAN_WARP_SCANS

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::topk {

/// Top-k policy (mirrors cub::detail::topk::topk_policy)
struct TopkPolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  BlockScanAlgorithm scan_algorithm;
  int bits_per_pass;
};

// ============================================================
// bits_per_pass calculation — must match CCCL exactly
//
// CCCL source (tuning_topk.cuh):
//   case 1: default: return 8;
//   case 2: case 4: case 8: return 11;
//
// Previous muh version had a wrong mapping:
//   key_size<=2 → 8, key_size<=4 → 9, key_size<=8 → 10
// This was WRONG. CCCL's actual function returns 11 for 2/4/8.
// ============================================================

constexpr int calc_bits_per_pass(int key_size) {
  switch (key_size) {
    case 1:
    default:
      return 8;
    case 2:
    case 4:
    case 8:
      return 11;
  }
}

// ============================================================
// policy_selector
//
// Matches CCCL's SM90+ path exactly:
//   threads = 512
//   items = max(1, nominal_4b(4) * 4 / key_size)
//   load = BLOCK_LOAD_VECTORIZE
//   scan = BLOCK_SCAN_WARP_SCANS
//   bits = calc_bits_per_pass(key_size)
//
// BI-V100 values: using SM100 as starting point.
// Once benchmarked on BI-V100, items/threads may diverge.
// ============================================================

struct policy_selector {
  int key_size;

  constexpr TopkPolicy operator()(const hardware_capability& hw) const {
    if (hw.at_least(hardware_capability::vendor_t::iluvatar, 100)) {
      // BI-V100 BENCHMARK RESULT (bench_bi100.py topk/float32):
      //   #1: ipt_4.ld_0.tpb_512  speedups: 1.039611 1.000222 1.004295
      //   (baseline: 1K=76.6us, 32K=220.8us, 152K=554.4us)
      //   #2: ipt_1.ld_1.tpb_256  speedups: 1.017337 1.020884 1.000531
      //   #3: ipt_1.ld_1.tpb_512  speedups: 0.986497 1.047220 1.004375
      //
      // KEY FINDINGS:
      //   - ipt=4, tpb=512 matches CCCL SM90+ formula (4*4/4=4) → CONFIRMED
      //   - ld=0 (LOAD_DEFAULT) beats ld=1 (LOAD_LDG/LOAD_CA) at small sizes
      //   - For 32K+ items, ld=1 is competitive but ipt=4 ld=0 wins overall
      //   - ipt=16 regresses at 32K and 152K sizes (too many items per thread)
      constexpr int nominal_4b_items = 4;
      int items = nominal_4b_items * 4 / key_size;
      if (items < 1) items = 1;

      return {512, items,
              BLOCK_LOAD_VECTORIZE,  // CCCL VECTORIZE; BI-V100 ld=0 confirmed best
              BLOCK_SCAN_WARP_SCANS,
              calc_bits_per_pass(key_size)};
    }

    // Fallback: older arch path
    constexpr int nominal_4b_items = 4;
    int items = nominal_4b_items * 4 / key_size;
    if (items < 1) items = 1;
    if (items > nominal_4b_items) items = nominal_4b_items;

    return {512, items,
            BLOCK_LOAD_VECTORIZE,
            BLOCK_SCAN_WARP_SCANS,
            calc_bits_per_pass(key_size)};
  }
};

} // namespace muh::tuning::topk
