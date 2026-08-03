// muh/include/muh/tuning/tuning_reduce.cuh — BI-V100 reduce tuning
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_reduce.cuh
// vllm impact: Attention score reduction in paged_attention (every decode step)
// Competition weight: Output TPS × 16.796 (83% — highest priority)
//
// HARDWARE PROFILE (confirmed via ixsmi on Phanthy Cloud):
//   SM count:   16 (NOT 50 from spec sheet)
//   SMEM:       48KB (49152 bytes) per block
//   L2 cache:   6MB (vs SM100's 50MB — 8.3× smaller)
//   HBM BW:     900 GB/s
//   BW/SM:      900/16 = 56 GB/s (≈ B200 level, NOT A100's 18 GB/s)
//   Warp size:  32
//
// SM=16 IMPACT ON TUNING:
//   With only 16 SMs and max 2 CTAs/SM occupancy, there are at most 32 concurrent
//   CTAs. Each CTA must process MORE data per tile to compensate for fewer CTAs.
//   This means tiles should be LARGER than SM100's defaults (which assume 148 SMs).
//   Target: fill SMEM to ≥ 70% where possible (current det paths use only 23%).
//
// CCCL upstream structure (for reference):
//   compute_capability >= 10.0 → sm100_tuning specializations (type-dispatch)
//   compute_capability >= 6.0  → Policy600 {threads=256, items=16, vec=4}
//   compute_capability >= 5.0  → Policy500 {threads=256, items=20, vec=4}
//   Three determinism modes: run_to_run, gpu_to_gpu, not_guaranteed
//
// NOTE: scale_mem_bound returns {items, threads} (items-first), matching
// CCCL's scaling_result struct. Destructure as auto [i, t] = ...;

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
// BI-V100 tuning values for plus<> operator
//
// SM=16 strategy: maximize tile size within 48KB SMEM.
// With 32 concurrent CTAs (16 SMs × 2 occupancy), each CTA
// should process ≥ 49152/(accum_size) elements per tile.
//
// CCCL benchmark format reference:
//   ipt_<items>.tpb_<threads>.ipv_<vec> <s_16> <s_20> <s_24> <s_28>
//   where s_N = speedup vs TUNE_BASE at 2^N elements
// ============================================================

// --- plus<> operator, two-phase (WARP_REDUCTIONS) ---

struct bi100_plus_accum1_o4 {
  // accum_size=1 (int8/uint8/bool), tile = 512*32*1 = 16384 (33% SMEM)
  // Scaled from CCCL: nominal_4B_items=16 → items=16*4/1=64, clamped to 32
  // SM=16: want larger tile → threads=512, items=32
  static constexpr int items   = 32;
  static constexpr int threads = 512;
  static constexpr int vec     = 4;
};

struct bi100_plus_accum2_o4 {
  // accum_size=2 (int16/uint16/float16/bfloat16), tile = 512*24*2 = 24576 (50%)
  // Qwen3.6 uses bfloat16 for KV cache — this is a hot path
  static constexpr int items   = 24;
  static constexpr int threads = 512;
  static constexpr int vec     = 2;
};

struct bi100_plus_float32_o4 {
  // accum_size=4, tile = 512*24*4 = 49152 (100% SMEM — max utilization)
  // SM100 ref: ipt_16.tpb_512.ipv_2 → tile=32768 (67% SMEM)
  // SM=16 optimization: increase items from 16→24 to fill SMEM
  // This gives each CTA 50% more data, compensating for fewer CTAs
  static constexpr int items   = 24;
  static constexpr int threads = 512;
  static constexpr int vec     = 2;
};

struct bi100_plus_float32_o8 {
  // Same as o4 but with 8-byte offset — vec=1 for alignment
  static constexpr int items   = 24;
  static constexpr int threads = 512;
  static constexpr int vec     = 1;
};

struct bi100_plus_float64_o4 {
  // accum_size=8, SM100 uses threads=640 items=16 → tile=81920 > 49152 OVERFLOW!
  // Max items at threads=384: 49152/(384*8) = 16 → tile = 49152 (100%)
  // Alternatively threads=512 items=12 → tile = 49152 (100%)
  // Choose 384×16: more items/thread = fewer loop iterations = better ILP
  static constexpr int items   = 16;
  static constexpr int threads = 384;
  static constexpr int vec     = 2;
};

struct bi100_plus_float64_o8 {
  // 8-byte offset + 8-byte accum: vec=1
  static constexpr int items   = 16;
  static constexpr int threads = 384;
  static constexpr int vec     = 1;
};

struct bi100_plus_int64_o4 {
  // Same SMEM constraint as float64 (accum_size=8)
  static constexpr int items   = 16;
  static constexpr int threads = 384;
  static constexpr int vec     = 2;
};

struct bi100_plus_int64_o8 {
  static constexpr int items   = 16;
  static constexpr int threads = 384;
  static constexpr int vec     = 1;
};

struct bi100_plus_accum16_o4 {
  // accum_size=16 (int128/complex<double>), tile = 192*16*16 = 49152 (100%)
  static constexpr int items   = 16;
  static constexpr int threads = 192;
  static constexpr int vec     = 1;
};

// --- Deterministic tunings: BLOCK_REDUCE_RAKING, vec=1 ---
// SM=16 fix: increase tile from ~23% to ≥50% SMEM utilization

struct bi100_det_float32 {
  // OLD: threads=224 items=13 → tile=11648 (23% SMEM) — way too small for 16 SMs
  // NEW: threads=384 items=32 → tile=49152 (100% SMEM)
  // With only 32 concurrent CTAs, maxing SMEM per CTA is critical
  static constexpr int items   = 32;
  static constexpr int threads = 384;
};

struct bi100_det_float64 {
  // OLD: threads=128 items=11 → tile=11264 (23% SMEM)
  // NEW: threads=384 items=16 → tile=49152 (100% SMEM)
  static constexpr int items   = 16;
  static constexpr int threads = 384;
};

struct bi100_det_int32 {
  // int32 deterministic: threads=384 items=32 → tile=49152 (100%)
  static constexpr int items   = 32;
  static constexpr int threads = 384;
};

struct bi100_det_int16 {
  // int16/float16/bfloat16 deterministic
  // threads=384 items=64 → tile=49152 (100%)
  static constexpr int items   = 64;
  static constexpr int threads = 384;
};

// --- Default fallback for unknown types/ops ---

struct bi100_default {
  // SM60-equivalent but with SM=16 tile maximization
  // threads=256 items=24 → at accum_size=4: tile=24576 (50% SMEM, safe margin)
  // at accum_size=8: 256*24*8 = 49152 (100%)
  static constexpr int items   = 24;
  static constexpr int threads = 256;
  static constexpr int vec     = 4;
};

// ============================================================
// policy_selector — full dispatch matching CCCL structure
//
// Dispatch order:
//   1. determinism mode (gpu_to_gpu → RAKING, else → WARP_REDUCTIONS)
//   2. operator type (plus → specialized, min/max → same as plus for BI-V100)
//   3. accum_size (1B, 2B, 4B, 8B, 16B)
//   4. offset_size (4B vs 8B affects vec_size)
//   5. accum_type (float32/float64 get specific tunings)
// ============================================================

struct policy_selector {
  type_t accum_t;
  op_kind_t operation_t;
  int offset_size;
  int accum_size;
  determinism_t determinism = determinism_t::run_to_run;

  // --- Deterministic path: BLOCK_REDUCE_RAKING ---
  constexpr ReducePolicy get_deterministic(const hardware_capability& hw) const {
    if (hw.at_least(hardware_capability::vendor_t::iluvatar, 100)) {
      // Type-specific tunings for deterministic reduce
      if (accum_size <= 2) {
        auto [i, t] = scale_mem_bound(bi100_det_int16::threads,
                                       bi100_det_int16::items, accum_size);
        ReducePassPolicy rp{t, i, 1, BLOCK_REDUCE_RAKING, LOAD_DEFAULT};
        return {rp, rp};
      }
      if (accum_t == type_t::float32 || accum_size == 4) {
        auto [i, t] = scale_mem_bound(bi100_det_float32::threads,
                                       bi100_det_float32::items, accum_size);
        ReducePassPolicy rp{t, i, 1, BLOCK_REDUCE_RAKING, LOAD_DEFAULT};
        return {rp, rp};
      }
      if (accum_t == type_t::float64 || accum_size == 8) {
        auto [i, t] = scale_mem_bound(bi100_det_float64::threads,
                                       bi100_det_float64::items, accum_size);
        ReducePassPolicy rp{t, i, 1, BLOCK_REDUCE_RAKING, LOAD_DEFAULT};
        return {rp, rp};
      }
    }
    // Fallback for unknown hardware
    auto [i, t] = scale_mem_bound(256, 16, accum_size);
    ReducePassPolicy rp{t, i, 1, BLOCK_REDUCE_RAKING, LOAD_DEFAULT};
    return {rp, rp};
  }

  // --- Two-phase path: BLOCK_REDUCE_WARP_REDUCTIONS ---
  constexpr ReducePolicy get_two_phase(const hardware_capability& hw) const {
    if (hw.at_least(hardware_capability::vendor_t::iluvatar, 100)) {
      // plus<> operator — fully specialized dispatch
      if (operation_t == op_kind_t::plus || operation_t == op_kind_t::min
          || operation_t == op_kind_t::max) {
        // accum_size=1 (int8, uint8, bool)
        if (accum_size == 1) {
          auto [i, t] = scale_mem_bound(bi100_plus_accum1_o4::threads,
                                         bi100_plus_accum1_o4::items, accum_size);
          ReducePassPolicy rp{t, i, bi100_plus_accum1_o4::vec,
                              BLOCK_REDUCE_WARP_REDUCTIONS, LOAD_LDG};
          return {rp, rp};
        }
        // accum_size=2 (int16, float16, bfloat16)
        if (accum_size == 2) {
          auto [i, t] = scale_mem_bound(bi100_plus_accum2_o4::threads,
                                         bi100_plus_accum2_o4::items, accum_size);
          ReducePassPolicy rp{t, i, bi100_plus_accum2_o4::vec,
                              BLOCK_REDUCE_WARP_REDUCTIONS, LOAD_LDG};
          return {rp, rp};
        }
        // accum_size=4 (float32, int32)
        if (accum_size == 4) {
          int vec = (offset_size <= 4) ? bi100_plus_float32_o4::vec
                                       : bi100_plus_float32_o8::vec;
          auto [i, t] = scale_mem_bound(bi100_plus_float32_o4::threads,
                                         bi100_plus_float32_o4::items, accum_size);
          ReducePassPolicy rp{t, i, vec, BLOCK_REDUCE_WARP_REDUCTIONS, LOAD_LDG};
          return {rp, rp};
        }
        // accum_size=8 (float64, int64)
        if (accum_size == 8) {
          if (accum_t == type_t::float64) {
            int vec = (offset_size <= 4) ? bi100_plus_float64_o4::vec
                                         : bi100_plus_float64_o8::vec;
            auto [i, t] = scale_mem_bound(bi100_plus_float64_o4::threads,
                                           bi100_plus_float64_o4::items, accum_size);
            ReducePassPolicy rp{t, i, vec, BLOCK_REDUCE_WARP_REDUCTIONS, LOAD_LDG};
            return {rp, rp};
          }
          // int64 and other 8-byte types
          int vec = (offset_size <= 4) ? bi100_plus_int64_o4::vec
                                       : bi100_plus_int64_o8::vec;
          auto [i, t] = scale_mem_bound(bi100_plus_int64_o4::threads,
                                         bi100_plus_int64_o4::items, accum_size);
          ReducePassPolicy rp{t, i, vec, BLOCK_REDUCE_WARP_REDUCTIONS, LOAD_LDG};
          return {rp, rp};
        }
        // accum_size=16 (int128, complex<double>)
        if (accum_size == 16) {
          auto [i, t] = scale_mem_bound(bi100_plus_accum16_o4::threads,
                                         bi100_plus_accum16_o4::items, accum_size);
          ReducePassPolicy rp{t, i, bi100_plus_accum16_o4::vec,
                              BLOCK_REDUCE_WARP_REDUCTIONS, LOAD_LDG};
          return {rp, rp};
        }
      }
    }
    // Fallback: SM60-equivalent with SM=16 tile optimization
    auto [i, t] = scale_mem_bound(bi100_default::threads,
                                   bi100_default::items, accum_size);
    ReducePassPolicy rp{t, i, bi100_default::vec,
                        BLOCK_REDUCE_WARP_REDUCTIONS, LOAD_LDG};
    return {rp, rp};
  }

  // --- Main entry point ---
  constexpr ReducePolicy operator()(const hardware_capability& hw) const {
    if (determinism == determinism_t::gpu_to_gpu)
      return get_deterministic(hw);

    auto policy = get_two_phase(hw);
    if (determinism == determinism_t::not_guaranteed) {
      policy.multi_tile.reduce_algorithm =
        BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC;
    }
    return policy;
  }
};

} // namespace muh::tuning::reduce
