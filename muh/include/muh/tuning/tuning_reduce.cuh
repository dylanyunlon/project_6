// muh/include/muh/tuning/tuning_reduce.cuh — BI-V100 reduce tuning
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_reduce.cuh
// vllm impact: Attention score reduction in paged_attention (every decode step)
// Competition weight: Output TPS × 16.796 (83% — highest priority)
//
// HARDWARE PROFILE (confirmed via ixsmi on Phanthy Cloud):
//   SM count:   16 (NOT 50 from spec sheet)
//   SMEM:       48KB (49152 bytes) per block
//   L2 cache:   6MB
//   HBM BW:     900 GB/s
//   BW/SM:      900/16 = 56 GB/s
//   Warp size:  32
//
// CRITICAL ARCHITECTURE INSIGHT (from agent_reduce.cuh source):
//   Reduce does NOT use BlockLoad — data is loaded directly to registers
//   via striped access (ConsumeFullTile) or vectorized loads (VectorT).
//   The SMEM is ONLY used by BlockReduce (warp shuffles + small scratch).
//   BlockReduce<WARP_REDUCTIONS> SMEM ≈ threads * sizeof(AccumT) / 32 (one value per warp).
//
//   Previous muh tuning assumed "tile = tpb * ipt * accum_size ≤ 48KB SMEM"
//   — THIS WAS WRONG. That formula applies to scan (which uses BlockLoad staging).
//   For reduce, the real constraint is REGISTER PRESSURE:
//     Each thread holds AccumT items[ITEMS_PER_THREAD] in registers.
//     items=24 for float32 → 24 registers → acceptable (BI-V100 has 64K registers/SM)
//     items=24 also means fewer CTAs needed → good for 16 SMs
//
//   Vectorized load condition (agent_reduce.cuh line ~180):
//     ATTEMPT_VECTORIZATION requires: vec_size > 1, items % vec == 0,
//     is_pointer<InputT>, is_trivially_relocatable<InputT>, sizeof(InputT) <= 8
//
// CCCL kernel_reduce.cuh two execution paths:
//   1. multi-tile (DeviceReduceKernel): each block reduces a tile partition
//      - StableReductionOrder=false: atomicAdd aggregation (lowest latency)
//      - StableReductionOrder=true: write to d_out[blockIdx.x], then single-tile pass
//   2. single-tile (DeviceReduceSingleTileKernel): one block for all data
//
//   BI-V100: 16 SMs → max 32 CTAs → atomic contention is negligible
//   → not_guaranteed determinism (atomic path) is optimal for decode
//
// PENDING REAL BENCHMARK: LOAD_LDG vs LOAD_DEFAULT
//   topk bench showed LOAD_DEFAULT (ld=0) beat LOAD_LDG (ld=1).
//   reduce may be similar — BI-V100's L1/L2 behavior differs from NVIDIA.
//   Keep LOAD_LDG for now (CCCL default), switch if bench says otherwise.

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
// Tile sizing rationale (corrected):
//   NOT SMEM-limited — reduce loads to registers, not SMEM staging.
//   Constraint is register pressure + occupancy:
//     items=24, float32 → 24 regs for data + overhead → ~40 regs/thread
//     SM has 64K regs → 64K/40 = 1600 threads max → 3 CTAs @ 512 threads ✓
//   Also want large tiles (few CTAs) because 16 SMs can't launch many anyway.
//   CCCL SM100 uses items=16 for float32 (with 148 SMs, more CTAs = more parallelism)
//   BI-V100 uses items=24 (fewer CTAs, each processes more data)
// ============================================================

// --- plus<> operator, two-phase (WARP_REDUCTIONS) ---

struct bi100_plus_accum1_o4 {
  // accum_size=1, reg pressure: 32 regs × 1B = trivial
  static constexpr int items   = 32;
  static constexpr int threads = 512;
  static constexpr int vec     = 4;  // 4×1B = 4-byte vectorized load
};

struct bi100_plus_accum2_o4 {
  // accum_size=2 (float16/bfloat16 — Qwen3.6 KV cache hot path)
  // reg pressure: 24 regs × 2B accum overhead → moderate
  static constexpr int items   = 24;
  static constexpr int threads = 512;
  static constexpr int vec     = 2;  // 2×2B = 4-byte vectorized load
};

struct bi100_plus_float32_o4 {
  // accum_size=4 (float32 — paged_attention score reduction)
  // reg pressure: 24 × 4B = 96B per thread → ~24 regs just for data
  // CCCL SM100: items=16, threads=512, vec=2
  // BI-V100: items=24 (fewer CTAs = each does more, compensating for 16 SMs)
  static constexpr int items   = 24;
  static constexpr int threads = 512;
  static constexpr int vec     = 2;  // items%vec=0 ✓ → vectorized load enabled
};

struct bi100_plus_float32_o8 {
  static constexpr int items   = 24;
  static constexpr int threads = 512;
  static constexpr int vec     = 1;  // 8-byte offset → unaligned risk, disable vectorize
};

struct bi100_plus_float64_o4 {
  // accum_size=8, reg pressure: 16 × 2 regs (double) = 32 regs → fine
  // CCCL SM100: items=16, threads=640 → 640 NOT multiple of 32 on all warps
  // BI-V100: threads=384 (12 warps, clean), items=16
  static constexpr int items   = 16;
  static constexpr int threads = 384;
  static constexpr int vec     = 2;  // 2×8B = 16-byte vectorized load
};

struct bi100_plus_float64_o8 {
  static constexpr int items   = 16;
  static constexpr int threads = 384;
  static constexpr int vec     = 1;
};

struct bi100_plus_int64_o4 {
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
  // accum_size=16 (int128/complex<double>)
  // reg pressure: 16 × 4 regs = 64 regs → high, keep items=16
  // threads=192 (6 warps) to allow 2 CTAs per SM for occupancy
  static constexpr int items   = 16;
  static constexpr int threads = 192;
  static constexpr int vec     = 1;  // sizeof(InputT)=16 > 8 → ATTEMPT_VECTORIZATION=false
};

// --- Deterministic tunings: BLOCK_REDUCE_RAKING, vec=1 ---
// RAKING needs more SMEM than WARP_REDUCTIONS (scratch array per warp-step)
// but still far less than scan's BlockLoad staging

struct bi100_det_float32 {
  // Deterministic: RAKING with large tile
  // items=32, threads=384 → 32 regs for data → acceptable
  static constexpr int items   = 32;
  static constexpr int threads = 384;
};

struct bi100_det_float64 {
  static constexpr int items   = 16;
  static constexpr int threads = 384;
};

struct bi100_det_int32 {
  static constexpr int items   = 32;
  static constexpr int threads = 384;
};

struct bi100_det_int16 {
  // int16/float16/bfloat16: items=64 → 64 regs × 2B/accum → moderate
  static constexpr int items   = 64;
  static constexpr int threads = 384;
};

// --- Default fallback ---

struct bi100_default {
  static constexpr int items   = 24;
  static constexpr int threads = 256;
  static constexpr int vec     = 4;
};

// ============================================================
// policy_selector
// ============================================================

struct policy_selector {
  type_t accum_t;
  op_kind_t operation_t;
  int offset_size;
  int accum_size;
  determinism_t determinism = determinism_t::run_to_run;

  constexpr ReducePolicy get_deterministic(const hardware_capability& hw) const {
    if (hw.at_least(hardware_capability::vendor_t::iluvatar, 100)) {
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
    auto [i, t] = scale_mem_bound(256, 16, accum_size);
    ReducePassPolicy rp{t, i, 1, BLOCK_REDUCE_RAKING, LOAD_DEFAULT};
    return {rp, rp};
  }

  constexpr ReducePolicy get_two_phase(const hardware_capability& hw) const {
    if (hw.at_least(hardware_capability::vendor_t::iluvatar, 100)) {
      if (operation_t == op_kind_t::plus || operation_t == op_kind_t::min
          || operation_t == op_kind_t::max) {
        if (accum_size == 1) {
          auto [i, t] = scale_mem_bound(bi100_plus_accum1_o4::threads,
                                         bi100_plus_accum1_o4::items, accum_size);
          ReducePassPolicy rp{t, i, bi100_plus_accum1_o4::vec,
                              BLOCK_REDUCE_WARP_REDUCTIONS, LOAD_LDG};
          return {rp, rp};
        }
        if (accum_size == 2) {
          auto [i, t] = scale_mem_bound(bi100_plus_accum2_o4::threads,
                                         bi100_plus_accum2_o4::items, accum_size);
          ReducePassPolicy rp{t, i, bi100_plus_accum2_o4::vec,
                              BLOCK_REDUCE_WARP_REDUCTIONS, LOAD_LDG};
          return {rp, rp};
        }
        if (accum_size == 4) {
          int vec = (offset_size <= 4) ? bi100_plus_float32_o4::vec
                                       : bi100_plus_float32_o8::vec;
          auto [i, t] = scale_mem_bound(bi100_plus_float32_o4::threads,
                                         bi100_plus_float32_o4::items, accum_size);
          ReducePassPolicy rp{t, i, vec, BLOCK_REDUCE_WARP_REDUCTIONS, LOAD_LDG};
          return {rp, rp};
        }
        if (accum_size == 8) {
          if (accum_t == type_t::float64) {
            int vec = (offset_size <= 4) ? bi100_plus_float64_o4::vec
                                         : bi100_plus_float64_o8::vec;
            auto [i, t] = scale_mem_bound(bi100_plus_float64_o4::threads,
                                           bi100_plus_float64_o4::items, accum_size);
            ReducePassPolicy rp{t, i, vec, BLOCK_REDUCE_WARP_REDUCTIONS, LOAD_LDG};
            return {rp, rp};
          }
          int vec = (offset_size <= 4) ? bi100_plus_int64_o4::vec
                                       : bi100_plus_int64_o8::vec;
          auto [i, t] = scale_mem_bound(bi100_plus_int64_o4::threads,
                                         bi100_plus_int64_o4::items, accum_size);
          ReducePassPolicy rp{t, i, vec, BLOCK_REDUCE_WARP_REDUCTIONS, LOAD_LDG};
          return {rp, rp};
        }
        if (accum_size == 16) {
          auto [i, t] = scale_mem_bound(bi100_plus_accum16_o4::threads,
                                         bi100_plus_accum16_o4::items, accum_size);
          ReducePassPolicy rp{t, i, bi100_plus_accum16_o4::vec,
                              BLOCK_REDUCE_WARP_REDUCTIONS, LOAD_LDG};
          return {rp, rp};
        }
      }
    }
    auto [i, t] = scale_mem_bound(bi100_default::threads,
                                   bi100_default::items, accum_size);
    ReducePassPolicy rp{t, i, bi100_default::vec,
                        BLOCK_REDUCE_WARP_REDUCTIONS, LOAD_LDG};
    return {rp, rp};
  }

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
