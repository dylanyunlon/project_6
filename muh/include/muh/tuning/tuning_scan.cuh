// muh/include/muh/tuning/tuning_scan.cuh — BI-V100 scan tuning
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_scan.cuh
// This is the most complex tuning file in CCCL (900+ lines for NVIDIA).
//
// vllm impact: Prefix scan in paged attention block table lookup
// Competition weight: Input TPS × 2.799
//
// DERIVATION (not copy-paste from SM100):
// - SMEM constraint: tile = threads * items * value_size <= 48KB
//   SM100 8B tunings (416*23*8=76544, 320*22*8=56320) OVERFLOW on BI-V100
// - Delay parameters: SM100 L2=50MB, BI-V100 L2=6MB (8.3x smaller)
//   Smaller L2 → faster coherence → shorter delays
//   Heuristic: ns *= 0.5, l2w *= 0.6 (to be refined by benchmark)

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
  static constexpr int threads = 512;
  static constexpr int items   = 18;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::exponential_backon, 384, 492};
  static constexpr BlockLoadAlgorithm load_algo   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algo  = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_mod      = LOAD_DEFAULT;
};

struct bi100_lookback_2B_o4 {
  // SM100 ref: ipt_13.tpb_512.ns_1384.dcid_7.l2w_720 → 1.128x
  static constexpr int threads = 512;
  static constexpr int items   = 13;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::exponential_backon, 692, 432};
  static constexpr BlockLoadAlgorithm load_algo   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algo  = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_mod      = LOAD_DEFAULT;
};

struct bi100_lookback_4B_o4 {
  // SM100 ref: ipt_22.tpb_384.ns_1904.dcid_6.l2w_830 → 1.148x
  static constexpr int threads = 384;
  static constexpr int items   = 22;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::exponential_backon_jitter, 952, 498};
  static constexpr BlockLoadAlgorithm load_algo   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algo  = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_mod      = LOAD_DEFAULT;
};

struct bi100_lookback_4B_o8 {
  // SM100 ref: ipt_19.tpb_416.ns_956.dcid_7.l2w_550 → 1.146x
  static constexpr int threads = 416;
  static constexpr int items   = 19;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::exponential_backon, 478, 330};
  static constexpr BlockLoadAlgorithm load_algo   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algo  = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_mod      = LOAD_CA;
};

struct bi100_lookback_8B_o4 {
  // SM100 ref: ipt_23.tpb_416 → tile=76544 > 49152 SMEM OVERFLOW
  // Derived: items = 49152/(416*8) = 14. Delay halved (L2 6MB vs 50MB).
  static constexpr int threads = 416;
  static constexpr int items   = 14;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::exponential_backon_jitter_window, 386, 426};
  static constexpr BlockLoadAlgorithm load_algo   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algo  = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_mod      = LOAD_DEFAULT;
};

struct bi100_lookback_8B_o8 {
  // SM100 ref: ipt_22.tpb_320 → tile=56320 > 49152 SMEM OVERFLOW
  // Derived: items = 49152/(320*8) = 19. Delay: ns*0.5, l2w*0.6.
  static constexpr int threads = 320;
  static constexpr int items   = 19;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::exponential_backoff, 164, 579};
  static constexpr BlockLoadAlgorithm load_algo   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algo  = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_mod      = LOAD_DEFAULT;
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
    LookbackDelayAlgorithm::fixed_delay, 350, 450};
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
