// muh/include/muh/tuning/tuning_scan.cuh — BI-V100 scan tuning
//
// CRITICAL CCCL ARCHITECTURE FINDING (from single_pass_scan_operators.cuh):
//
//   template <int Delay, unsigned int GridThreshold = 500>
//   _CCCL_DEVICE _CCCL_FORCEINLINE void delay() {
//     if (gridDim.x < GridThreshold) {
//       __threadfence_block();    // ← ALL BI-V100 scans take this path
//     } else {
//       __nanosleep(Delay);       // ← only fires when grid > 500 CTAs
//     }
//   }
//
// BI-V100: 16 SMs × ~10 CTAs/SM max = ~160 CTAs. ALWAYS < 500.
// Therefore: ALL delay strategies (no_delay, fixed_delay, exponential_backon,
// exponential_backon_jitter, etc.) collapse to __threadfence_block() on BI-V100.
//
// This means:
//   1. The ns/dcid/l2w delay parameters are IRRELEVANT for BI-V100.
//   2. bench_bi100.py's finding that no_delay is optimal is CORRECT BY DESIGN.
//   3. The "ns×0.5, l2w×0.6" scaling heuristic was always a no-op on BI-V100.
//   4. Tuning effort should focus on threads/items/load_algo, NOT delay params.
//
// This architectural insight came from reading cub/agent/single_pass_scan_operators.cuh
// lines 136-148 (the delay() template function with GridThreshold=500 gate).
//
// CRITICAL INSIGHT FROM single_pass_scan_operators.cuh delay():
//   The CCCL delay function has a runtime branch:
//     if (gridDim.x < GridThreshold)     // GridThreshold = 500
//       __threadfence_block();            // lightweight, no nanosleep
//     else
//       __nanosleep(Delay);              // heavyweight
//
//   BI-V100: 16 SMs × subscription_factor(5) = max gridDim.x ≈ 80.
//   80 << 500, so ALL delay constructors (no_delay, fixed_delay,
//   exponential_backon_jitter, etc.) collapse to __threadfence_block().
//   This is why bench_bi100.py found no_delay optimal — because on BI-V100,
//   every delay policy IS effectively no_delay.
//
//   This also means the ns/dcid/l2w tuning dimensions from scan benchmark
//   (%RANGE% TUNE_MAGIC_NS, %RANGE% TUNE_DELAY_CONSTRUCTOR_ID, etc.)
//   are IRRELEVANT on BI-V100. The entire delay parameter space collapses
//   to a single point. Benchmarking should focus on ipt/tpb/trp/ld only.
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_scan.cuh
// This is the most complex tuning file in CCCL (900+ lines for NVIDIA).
//
// vllm impact: Prefix scan in paged attention block table lookup
// Competition weight: Input TPS × 2.799
//
// HARDWARE (confirmed via ixsmi):
//   SM count:   16 (NOT 50)
//   SMEM:       48KB (49152 bytes)
//   L2 cache:   6MB (vs SM100's 50MB — 8.3× smaller)
//   BW/SM:      900/16 = 56 GB/s
//
// SM=16 IMPACT ON SCAN:
//   1. SMEM constraint: tile = threads * items * value_size <= 48KB
//      SM100 8B tunings (416*23*8=76544, 320*22*8=56320) OVERFLOW on BI-V100
//   2. Delay parameters: SM100 L2=50MB, BI-V100 L2=6MB (8.3x smaller)
//      Smaller L2 → less inter-CTA contention on lookback status → shorter delays
//      With only 32 concurrent CTAs (16 SMs × 2), tile_status array fits in L2
//      Heuristic: ns *= 0.5, l2w *= 0.6 (PENDING BI-V100 BENCHMARK)
//   3. Tile maximization: fewer CTAs = each must process more data
//      Small tiles (e.g. 1B offset=4: tile=9216, 19% SMEM) waste capacity
//
// BI-V100 BENCHMARK VALIDATION (bench_bi100.py on iluvatar-bi-v100):
//   scan/float32 TOP 10 — all use ns=1904 (SM100 raw, NOT ×0.5!)
//   The ns×0.5 heuristic was WRONG. BI-V100 has 16 SMs = ~32 CTAs,
//   so lookback contention is minimal → large ns spacing is fine.
//   Best dcid=0 (no_delay), not dcid=6 (exponential_backon_jitter).
//   SMEM usage: 33792/49152 = 69% for ipt=22,tpb=384,value=4B.

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::scan {

/// Lookback scan policy (mirrors cub::ScanLookbackPolicy)
struct ScanLookbackPolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  BlockStoreAlgorithm store_algorithm;
  BlockScanAlgorithm scan_algorithm;
  LookbackDelayPolicy lookback_delay;
};

/// Lookahead scan policy (mirrors cub::ScanLookaheadPolicy)
struct ScanLookaheadPolicy {
  int reduce_and_scan_warps;
  int items_per_thread;
  int lookahead_items_per_thread;
  int lookahead_stages;
  int block_idx_stages;
};

/// Full scan policy
enum class ScanAlgorithm { lookback, lookahead };

struct ScanPolicy {
  ScanAlgorithm algorithm;
  ScanLookbackPolicy lookback;
  ScanLookaheadPolicy lookahead;
};

// ============================================================
// BI-V100 tuning values
//
// CCCL reference from tuning_scan.cuh policy_selector::operator():
//
// SM100 lookback (sum, primitive accum, offset_size=4):
//   value_size=1: tpb=512, ipt=18, delay=exponential_backon(768,820)  → 1.189x
//   value_size=2: tpb=512, ipt=13, delay=exponential_backon(1384,720) → 1.128x
//   value_size=4: tpb=384, ipt=22, delay=exponential_backon_jitter(1904,830) → 1.148x
//   value_size=8: tpb=416, ipt=23, delay=exponential_backon_jitter_window(772,710) → 1.089x
//
// SM100 lookahead:
//   value_size=1: warps=4, ipt=160-1, lai=8
//   value_size=2: warps=6, ipt=96-1, lai=2
//   value_size=4: float→warps=4,ipt=88-1,lai=3; int→warps=4,ipt=80-1,lai=3
//   value_size=8: warps=2, ipt=88-1, lai=5
//   value_size=16: warps=5, ipt=16-1, lai=8
// ============================================================

// --- Lookback tunings for BI-V100 ---

struct bi100_lookback_1B_o4 {
  // SM100 ref: ipt_18.tpb_512.ns_768.dcid_7.l2w_820 → 1.189x
  // SM=16 fix: tile = 512*18*1 = 9216 (19% SMEM — too small for 16 SMs)
  // Increase items: 512*32*1 = 16384 (33% SMEM, scan needs input+output buffer)
  // scan_tile = threads * items * accum_size * 2 (input+output) for SMEM
  // 512 * 32 * 1 * 2 = 32768 (67% SMEM) — good balance
  static constexpr int threads = 512;
  static constexpr int items   = 32;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::no_delay, 0, 492};
  static constexpr BlockLoadAlgorithm load_algo   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algo  = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_mod      = LOAD_DEFAULT;
};

struct bi100_lookback_2B_o4 {
  // SM100 ref: ipt_13.tpb_512.ns_1384.dcid_7.l2w_720 → 1.128x
  // SM=16 fix: tile = 512*13*2 = 13312 (27% SMEM)
  // Increase: 512*24*2 = 24576 → scan buffer = 24576*2 = 49152 (100% SMEM)
  static constexpr int threads = 512;
  static constexpr int items   = 24;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::no_delay, 0, 432};
  static constexpr BlockLoadAlgorithm load_algo   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algo  = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_mod      = LOAD_DEFAULT;
};

struct bi100_lookback_4B_o4 {
  // BI-V100 BENCHMARK RESULT (bench_bi100.py scan/float32):
  //   #1: dcid_0.ipt_22.l2w_500.ld_0.ns_1904.tpb_384.trp_1
  //   speedups: 1.038085 1.009473 1.007679 1.005803  SMEM=33792 (69%)
  //
  // KEY FINDING: ns=1904 (same as SM100 raw, NOT ×0.5!)
  //   The ns×0.5 heuristic was WRONG for BI-V100.
  //   dcid=0 (no_delay) beat dcid=6 (exponential_backon_jitter).
  //   With only 16 SMs → ~32 concurrent CTAs → minimal lookback contention
  //   → simple no_delay with ns=1904 spacing is optimal.
  static constexpr int threads = 384;
  static constexpr int items   = 22;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::no_delay, 1904, 500};
  static constexpr BlockLoadAlgorithm load_algo   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algo  = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_mod      = LOAD_DEFAULT;
};

struct bi100_lookback_4B_o8 {
  // SM100 ref: ipt_19.tpb_416.ns_956.dcid_7.l2w_550 → 1.146x
  static constexpr int threads = 416;
  static constexpr int items   = 19;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::no_delay, 0, 330};
  static constexpr BlockLoadAlgorithm load_algo   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algo  = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_mod      = LOAD_CA;
};

struct bi100_lookback_8B_o4 {
  // SM100 ref: ipt_23.tpb_416 → tile=76544 > 49152 SMEM OVERFLOW!
  // Fix: items = floor(49152/(384*8)) = 16 → tile = 384*16*8 = 49152 (100%)
  // Changed threads 416→384 (multiple of 32) for cleaner warp alignment
  static constexpr int threads = 384;
  static constexpr int items   = 16;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::no_delay, 0, 426};
  static constexpr BlockLoadAlgorithm load_algo   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algo  = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_mod      = LOAD_DEFAULT;
};

struct bi100_lookback_8B_o8 {
  // SM100 ref: ipt_22.tpb_320 → tile=56320 > 49152 SMEM OVERFLOW!
  // Fix: items = floor(49152/(320*8)) = 19 → tile = 320*19*8 = 48640 (99%)
  // 19 items confirmed safe, maximizes SMEM within constraint
  static constexpr int threads = 320;
  static constexpr int items   = 19;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::no_delay, 0, 579};
  static constexpr BlockLoadAlgorithm load_algo   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algo  = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_mod      = LOAD_DEFAULT;
};

struct bi100_lookback_1B_o8 {
  // CCCL SM100 ref: ipt_14.tpb_384.ns_228.dcid_7.l2w_775 → 1.107x
  // BI-V100 derived: delay halved (L2 6MB vs 50MB), LOAD_CA matches SM100
  // nominal_tile = 384*14*4 = 21504 ≤ 49152 ✓
  static constexpr int threads = 384;
  static constexpr int items   = 14;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::no_delay, 0, 465};
  static constexpr BlockLoadAlgorithm load_algo   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algo  = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_mod      = LOAD_CA;
};


// --- Lookahead tunings for BI-V100 ---

struct bi100_lookahead_1B {
  // SM100 ref: wrps_4.lbi_8.ipt_160 → 1.264x
  static constexpr int warps = 4;
  static constexpr int items = 159;   // 160-1
  static constexpr int lookahead_items = 8;
};

struct bi100_lookahead_2B {
  // SM100 ref: wrps_6.lbi_2.ipt_96 → 1.168x
  static constexpr int warps = 6;
  static constexpr int items = 95;    // 96-1
  static constexpr int lookahead_items = 2;
};

struct bi100_lookahead_4B {
  // SM100 ref (int): wrps_4.lbi_3.ipt_80 → 1.019x
  static constexpr int warps = 4;
  static constexpr int items = 79;    // 80-1
  static constexpr int lookahead_items = 3;
};

struct bi100_lookahead_4B_float {
  // SM100 ref (float32): wrps_4.lbi_3.ipt_88 → 1.047x
  static constexpr int warps = 4;
  static constexpr int items = 87;    // 88-1
  static constexpr int lookahead_items = 3;
};

struct bi100_lookahead_8B {
  // SM100 ref: wrps_2.lbi_5.ipt_88 → 1.086x
  static constexpr int warps = 2;
  static constexpr int items = 87;    // 88-1
  static constexpr int lookahead_items = 5;
};

struct bi100_lookahead_16B {
  // SM100 ref: wrps_5.lbi_8.ipt_16 → 1.160x
  static constexpr int warps = 5;
  static constexpr int items = 15;    // 16-1
  static constexpr int lookahead_items = 8;
};

// --- Lookback default fallback ---

struct bi100_lookback_default {
  static constexpr int threads = 128;
  static constexpr int items   = 15;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::no_delay, 0, 450};
  static constexpr BlockLoadAlgorithm load_algo   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algo  = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_mod      = LOAD_DEFAULT;
};

// ============================================================
// policy_selector
// ============================================================

struct policy_selector {
  int input_value_size;
  int accum_size;
  int offset_size;
  type_t input_type;
  type_t accum_type;
  op_kind_t operation_t;
  bool is_primitive_accum;

  /// Get the best lookback policy for BI-V100
  constexpr ScanLookbackPolicy get_lookback(const hardware_capability& hw) const {
    if (hw.at_least(hardware_capability::vendor_t::iluvatar, 100)
        && operation_t == op_kind_t::plus && is_primitive_accum) {
      if (offset_size == 4) {
        switch (input_value_size) {
          case 1: return {bi100_lookback_1B_o4::threads, bi100_lookback_1B_o4::items,
                          bi100_lookback_1B_o4::load_algo, bi100_lookback_1B_o4::load_mod,
                          bi100_lookback_1B_o4::store_algo, BLOCK_SCAN_WARP_SCANS,
                          bi100_lookback_1B_o4::delay};
          case 2: return {bi100_lookback_2B_o4::threads, bi100_lookback_2B_o4::items,
                          bi100_lookback_2B_o4::load_algo, bi100_lookback_2B_o4::load_mod,
                          bi100_lookback_2B_o4::store_algo, BLOCK_SCAN_WARP_SCANS,
                          bi100_lookback_2B_o4::delay};
          case 4: return {bi100_lookback_4B_o4::threads, bi100_lookback_4B_o4::items,
                          bi100_lookback_4B_o4::load_algo, bi100_lookback_4B_o4::load_mod,
                          bi100_lookback_4B_o4::store_algo, BLOCK_SCAN_WARP_SCANS,
                          bi100_lookback_4B_o4::delay};
          case 8: return {bi100_lookback_8B_o4::threads, bi100_lookback_8B_o4::items,
                          bi100_lookback_8B_o4::load_algo, bi100_lookback_8B_o4::load_mod,
                          bi100_lookback_8B_o4::store_algo, BLOCK_SCAN_WARP_SCANS,
                          bi100_lookback_8B_o4::delay};
          default: break;
        }
      } else if (offset_size == 8) {
        switch (input_value_size) {
          case 1: return {bi100_lookback_1B_o8::threads, bi100_lookback_1B_o8::items,
                          bi100_lookback_1B_o8::load_algo, bi100_lookback_1B_o8::load_mod,
                          bi100_lookback_1B_o8::store_algo, BLOCK_SCAN_WARP_SCANS,
                          bi100_lookback_1B_o8::delay};
          case 4: return {bi100_lookback_4B_o8::threads, bi100_lookback_4B_o8::items,
                          bi100_lookback_4B_o8::load_algo, bi100_lookback_4B_o8::load_mod,
                          bi100_lookback_4B_o8::store_algo, BLOCK_SCAN_WARP_SCANS,
                          bi100_lookback_4B_o8::delay};
          case 8: return {bi100_lookback_8B_o8::threads, bi100_lookback_8B_o8::items,
                          bi100_lookback_8B_o8::load_algo, bi100_lookback_8B_o8::load_mod,
                          bi100_lookback_8B_o8::store_algo, BLOCK_SCAN_WARP_SCANS,
                          bi100_lookback_8B_o8::delay};
          default: break;
        }
      }
    }

    // Fallback
    return {bi100_lookback_default::threads, bi100_lookback_default::items,
            bi100_lookback_default::load_algo, bi100_lookback_default::load_mod,
            bi100_lookback_default::store_algo, BLOCK_SCAN_WARP_SCANS,
            bi100_lookback_default::delay};
  }

  /// Get the best lookahead policy for BI-V100
  constexpr ScanLookaheadPolicy get_lookahead(const hardware_capability& hw) const {
    // Lookahead requires specific hardware features (pipeline stages, etc.)
    // BI-V100 support is TBD — if not available, caller falls back to lookback
    if (!hw.at_least(hardware_capability::vendor_t::iluvatar, 100))
      return {4, 63, 4, 2, -1}; // conservative default

    if (is_primitive_accum) {
      switch (input_value_size) {
        case 1:  return {bi100_lookahead_1B::warps, bi100_lookahead_1B::items,
                         bi100_lookahead_1B::lookahead_items, 2, -1};
        case 2:  return {bi100_lookahead_2B::warps, bi100_lookahead_2B::items,
                         bi100_lookahead_2B::lookahead_items, 2, -1};
        case 4:
          if (input_type == type_t::float32)
            return {bi100_lookahead_4B_float::warps, bi100_lookahead_4B_float::items,
                    bi100_lookahead_4B_float::lookahead_items, 2, -1};
          return {bi100_lookahead_4B::warps, bi100_lookahead_4B::items,
                  bi100_lookahead_4B::lookahead_items, 2, -1};
        case 8:  return {bi100_lookahead_8B::warps, bi100_lookahead_8B::items,
                         bi100_lookahead_8B::lookahead_items, 2, -1};
        case 16: return {bi100_lookahead_16B::warps, bi100_lookahead_16B::items,
                         bi100_lookahead_16B::lookahead_items, 2, -1};
      }
    }

    // Fallback lookahead
    int default_items = (256 / (input_value_size == 2 ? 2 : accum_size)) - 1;
    if (default_items < 1) default_items = 1;
    int lai = accum_size == 2 ? 3 : 4;
    return {4, default_items, lai, 2, -1};
  }

  /// Main dispatch — matches CCCL's operator()(cuda::compute_capability)
  constexpr ScanPolicy operator()(const hardware_capability& hw) const {
    // Try lookahead first (if hardware supports it)
    // TODO: add can_use_lookahead check once BI-V100 pipeline support is confirmed
    auto lookahead = get_lookahead(hw);

    // For now, default to lookback (safer, works on all hardware)
    auto lookback = get_lookback(hw);

    return {ScanAlgorithm::lookback, lookback, lookahead};
  }
};

} // namespace muh::tuning::scan
