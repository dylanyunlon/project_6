// muh/include/muh/tuning/tuning_reduce.cuh — BI-V100 reduce tuning
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_reduce.cuh
// vllm impact: Attention score reduction in multi-head attention
// Competition weight: Output TPS × 16.796 (highest priority)
//
// DERIVATION MODEL (not copy-paste from SM100):
//
// BI-V100 vs SM100 (B200):
//   SMEM:       48KB vs 48KB (default) — same
//   L2:         6MB vs 50MB — 8.3x smaller
//   BW:         900 GB/s vs 8000 GB/s — 8.9x lower
//   SM count:   50 vs 148 — 3x fewer
//   BW/SM:      18 GB/s vs 54 GB/s — 3x lower (≈ A100 level)
//
// Constraint: tile_size = threads * items * accum_size <= SMEM (48KB)
//
// SM100 reduce float64 uses threads=640, items=16 → tile = 81920 bytes.
// 81920 > 49152 (BI-V100 SMEM). This would CRASH on BI-V100.
// Similarly int64 uses threads=512, items=15 → tile = 61440 > 49152.
//
// Fix: derive threads/items from SMEM constraint, not copy from SM100.
//
// NOTE: scale_mem_bound returns {items, threads} (items-first), matching
// CCCL's scaling_result struct. Destructure as auto [i, t] = ...;
// NOT auto [t, i] which was the old (buggy) order.

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::reduce {

struct ReducePassPolicy {
  int threads_per_block;
  int items_per_thread;
  int vec_size;
  BlockReduceAlgorithm reduce_algorithm;
  CacheLoadModifier load_modifier;
};

struct ReducePolicy {
  ReducePassPolicy multi_tile;
  ReducePassPolicy single_tile;
};

enum class determinism_t {
  run_to_run,
  gpu_to_gpu,
  not_guaranteed,
};

// ============================================================
// BI-V100 tuning values — DERIVED from hardware constraints
//
// Key constraint: tile_bytes = threads * items * accum_size <= 48KB
// SM100 values that violate this are WRONG for BI-V100.
// ============================================================

struct bi100_float32_plus_o4 {
  // accum_size=4, tile = 512*16*4 = 32768 ≤ 49152 ✓
  // SM100 ref: ipt_16.tpb_512.ipv_2 1.061 1.000 1.065 1.167
  // Derivation: SMEM OK, threads=512. SM=16 (not 50 from spec sheet).
  // At 16 SMs, fewer concurrent CTAs → consider larger tiles. Pending benchmark.
  static constexpr int items              = 16;
  static constexpr int threads            = 512;
  static constexpr int items_per_vec_load = 2;
};

struct bi100_float64_plus_o4 {
  // SM100: threads=640, items=16 → tile = 640*16*8 = 81920 > 49152 ✗ OVERFLOW
  // Derivation: max items at 512 threads = 49152/(512*8) = 12
  // SM90 used threads=256, items=16 → tile = 32768 (conservative)
  // Choose: threads=512, items=12 → tile = 49152 (max utilization)
  static constexpr int items              = 12;
  static constexpr int threads            = 512;
  static constexpr int items_per_vec_load = 1;
};

struct bi100_int64_plus_o4 {
  // SM100: threads=512, items=15 → tile = 512*15*8 = 61440 > 49152 ✗ OVERFLOW
  // Derivation: max items at 384 threads = 49152/(384*8) = 16
  // Choose: threads=384, items=16 → tile = 49152 (max utilization)
  static constexpr int items              = 16;
  static constexpr int threads            = 384;
  static constexpr int items_per_vec_load = 2;
};

struct bi100_int64_plus_o8 {
  // SM100: threads=512, items=15 → same overflow
  // Derivation: same as o4 but vec=1 (8-byte offset reduces vectorization)
  static constexpr int items              = 16;
  static constexpr int threads            = 384;
  static constexpr int items_per_vec_load = 1;
};

// Deterministic tunings: BLOCK_REDUCE_RAKING, vec_size=1
struct bi100_det_float32 {
  // SM90 ref: ipt_13.tpb_224 1.107 1.010 1.097 1.317
  // tile = 224*13*4 = 11648 ≤ 49152 ✓ (safe, same as SM90)
  static constexpr int items   = 13;
  static constexpr int threads = 224;
};

struct bi100_det_float64 {
  // SM86 ref: ipt_11.tpb_128 1.232 1.002 1.245 1.582
  // tile = 128*11*8 = 11264 ≤ 49152 ✓
  static constexpr int items   = 11;
  static constexpr int threads = 128;
};

struct bi100_default {
  // SM60-equivalent fallback: tile = 256*16*accum_size
  // At accum_size=8: 256*16*8 = 32768 ≤ 49152 ✓
  static constexpr int items              = 16;
  static constexpr int threads            = 256;
  static constexpr int items_per_vec_load = 4;
};

// ============================================================
// policy_selector — three determinism modes matching CCCL
// ============================================================

struct policy_selector {
  type_t accum_t;
  op_kind_t operation_t;
  int offset_size;
  int accum_size;
  determinism_t determinism = determinism_t::run_to_run;

  constexpr ReducePolicy get_deterministic(const hardware_capability& hw) const {
    if (hw.at_least(hardware_capability::vendor_t::iluvatar, 100)) {
      if (accum_t == type_t::float32) {
        auto [i, t] = scale_mem_bound(bi100_det_float32::threads,
                                       bi100_det_float32::items, accum_size);
        ReducePassPolicy rp{t, i, 1, BLOCK_REDUCE_RAKING, LOAD_DEFAULT};
        return {rp, rp};
      }
      if (accum_t == type_t::float64) {
        auto [i, t] = scale_mem_bound(bi100_det_float64::threads,
                                       bi100_det_float64::items, accum_size);
        ReducePassPolicy rp{t, i, 1, BLOCK_REDUCE_RAKING, LOAD_DEFAULT};
        return {rp, rp};
      }
    }
    auto [i, t] = scale_mem_bound(256, 16, accum_size);
    ReducePassPolicy rp{t, i, 1, BLOCK_REDUCE_RAKING, LOAD_DEFAULT};
    return {rp, rp};
  }

  constexpr ReducePolicy get_two_phase(const hardware_capability& hw) const {
    if (operation_t == op_kind_t::plus &&
        hw.at_least(hardware_capability::vendor_t::iluvatar, 100)) {

      if (accum_t == type_t::float32 && offset_size == 4 && accum_size == 4) {
        auto [i, t] = scale_mem_bound(bi100_float32_plus_o4::threads,
                                       bi100_float32_plus_o4::items, accum_size);
        ReducePassPolicy rp{t, i, bi100_float32_plus_o4::items_per_vec_load,
                            BLOCK_REDUCE_WARP_REDUCTIONS, LOAD_LDG};
        return {rp, rp};
      }
      if (accum_t == type_t::float64 && offset_size == 4 && accum_size == 8) {
        auto [i, t] = scale_mem_bound(bi100_float64_plus_o4::threads,
                                       bi100_float64_plus_o4::items, accum_size);
        ReducePassPolicy rp{t, i, bi100_float64_plus_o4::items_per_vec_load,
                            BLOCK_REDUCE_WARP_REDUCTIONS, LOAD_LDG};
        return {rp, rp};
      }
      if (offset_size == 4 && accum_size == 8) {
        auto [i, t] = scale_mem_bound(bi100_int64_plus_o4::threads,
                                       bi100_int64_plus_o4::items, accum_size);
        ReducePassPolicy rp{t, i, bi100_int64_plus_o4::items_per_vec_load,
                            BLOCK_REDUCE_WARP_REDUCTIONS, LOAD_LDG};
        return {rp, rp};
      }
      if (offset_size == 8 && accum_size == 8) {
        auto [i, t] = scale_mem_bound(bi100_int64_plus_o8::threads,
                                       bi100_int64_plus_o8::items, accum_size);
        ReducePassPolicy rp{t, i, bi100_int64_plus_o8::items_per_vec_load,
                            BLOCK_REDUCE_WARP_REDUCTIONS, LOAD_LDG};
        return {rp, rp};
      }
    }
    auto [i, t] = scale_mem_bound(bi100_default::threads, bi100_default::items, accum_size);
    ReducePassPolicy rp{t, i, bi100_default::items_per_vec_load,
                        BLOCK_REDUCE_WARP_REDUCTIONS, LOAD_LDG};
    return {rp, rp};
  }

  constexpr ReducePolicy operator()(const hardware_capability& hw) const {
    if (determinism == determinism_t::gpu_to_gpu)
      return get_deterministic(hw);
    auto policy = get_two_phase(hw);
    if (determinism == determinism_t::not_guaranteed)
      policy.multi_tile.reduce_algorithm = BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC;
    return policy;
  }
};

} // namespace muh::tuning::reduce
