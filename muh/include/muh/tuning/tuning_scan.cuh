// muh/include/muh/tuning/tuning_scan.cuh — BI-V100 scan tuning
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_scan.cuh (1525 lines)
// vllm impact: softmax denominator prefix-sum in paged_attention (every decode step)
// Competition weight: contributes to both Output TPS (decode) and Input TPS (prefill)
//
// HARDWARE PROFILE (confirmed via ixsmi on Phanthy Cloud):
//   SM count:   16 (NOT 50 from spec sheet)
//   SMEM:       48KB (49152 bytes) per block
//   L2 cache:   6MB (6291456 bytes)
//   HBM BW:     900 GB/s
//   BW/SM:      900/16 = 56 GB/s ≈ B200 level (not A100)
//   Warp size:  32
//
// CRITICAL: Scan uses BlockLoad to stage data in SMEM (unlike reduce which loads to registers).
//   tile_bytes = threads_per_block × items_per_thread × value_size
//   MUST satisfy: tile_bytes ≤ 49152 for ALL type sizes.
//
// CCCL SCAN ARCHITECTURE (from tuning_scan.cuh):
//   Two algorithm branches:
//     1. LOOKBACK (decoupled look-back): safe default, all GPUs
//        - Uses tile_state in global memory for inter-CTA communication
//        - LookbackDelayPolicy controls polling: {algorithm, delay_ns, l2_write_latency}
//        - BI-V100: 16 SMs → max 32 concurrent tiles → LOW contention → shorter delays OK
//     2. LOOKAHEAD (warpspeed): SM100+ only, requires PTX ISA 8.6+
//        - Uses pipeline stages for overlapped load/compute/store
//        - BI-V100 does NOT have PTX ISA 8.6 → LOCKED TO LOOKBACK
//
// CCCL SM100 BENCHMARK ANNOTATIONS (from policy_selector dispatch):
//   These are the gold-standard tuning points. Each annotation format:
//     ipt_<items>.tpb_<threads>.ns_<delay_ns>.dcid_<delay_algo>.l2w_<l2_latency>.trp_<transpose>.ld_<load_mod>
//     followed by 4 speedup values at problem sizes [2^16, 2^20, 2^24, 2^28]
//
//   BI-V100 ADAPTATION STRATEGY:
//     - threads/items: adapted via scale_mem_bound with SMEM cap at 49152
//     - delay_ns: scaled × 0.5 (BI-V100 L2 is 6MB vs SM100 64MB → fewer tiles → less contention)
//     - l2_write_latency: scaled × 0.6 (empirical ratio from L2 size difference)
//     - load_modifier: LOAD_DEFAULT preferred (topk bench showed BI-V100 L1/L2 differs from NVIDIA)
//     - delay_algorithm: preserved from CCCL (exponential_backon variants)
//
// DELAY ALGORITHM ENUM MAPPING (dcid values in CCCL benchmark annotations):
//   0 = no_delay
//   1 = fixed_delay
//   2 = exponential_backoff
//   3 = exponential_backoff_jitter
//   4 = exponential_backoff_jitter_window
//   5 = exponential_backon_jitter_window
//   6 = exponential_backon_jitter
//   7 = exponential_backon
//   8 = __reduce_by_key (internal)

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::scan {

// ============================================================
// Policy structs (mirrors CCCL ScanLookbackPolicy)
// ============================================================

struct ScanLookbackPolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  BlockStoreAlgorithm store_algorithm;
  BlockScanAlgorithm scan_algorithm;
  LookbackDelayPolicy lookback_delay;
};

struct ScanLookaheadPolicy {
  int reduce_and_scan_warps = 4;
  int items_per_thread = 63;
  int lookahead_items_per_thread = 4;
  int lookahead_stages = -1;    // negative = num_stages + lookahead_stages
  int block_idx_stages = -1;
};

enum class ScanAlgorithm {
  lookback,
  lookahead,  // NOT available on BI-V100 (requires PTX ISA 8.6)
};

struct ScanPolicy {
  ScanAlgorithm algorithm;
  ScanLookbackPolicy lookback;
  ScanLookaheadPolicy lookahead;
};

// ============================================================
// Helper: construct mem-scaled lookback policy (mirrors CCCL make_mem_scaled_lookback_scan_policy)
// ============================================================

constexpr ScanPolicy make_lookback_policy(
    int nominal_threads,
    int nominal_items,
    int compute_t_size,
    BlockLoadAlgorithm load_algo,
    CacheLoadModifier load_mod,
    BlockStoreAlgorithm store_algo,
    BlockScanAlgorithm scan_algo,
    LookbackDelayPolicy delay = {LookbackDelayAlgorithm::fixed_delay, 350, 450})
{
  auto [items, threads] = scale_mem_bound(nominal_threads, nominal_items, compute_t_size);
  return ScanPolicy{
    ScanAlgorithm::lookback,
    ScanLookbackPolicy{threads, items, load_algo, load_mod, store_algo, scan_algo, delay},
    ScanLookaheadPolicy{}
  };
}

// ============================================================
// BI-V100 tuning tables
//
// Derived from CCCL SM100 benchmark data with BI-V100 adaptations:
//   - SMEM cap: tile = tpb × ipt × value_size ≤ 49152
//   - delay_ns scaled ×0.5 (fewer concurrent tiles on 16 SMs)
//   - l2_write_latency scaled ×0.6 (6MB L2 vs 64MB)
//   - All use lookback (no lookahead — PTX ISA not available)
//
// Nomenclature:
//   bi100_lookback_{value_size}B_o{offset_size}
//   value_size = sizeof(InputValueT), offset_size = sizeof(OffsetT)
// ============================================================

// --- plus<> operator, offset_size=4 ---

// CCCL SM100: ipt_18.tpb_512.ns_768.dcid_7.l2w_820.trp_1.ld_0  1.189 1.006 1.173 1.305
// value_size=1: tile = 512×18×1 = 9216 ≤ 49152 ✓
// BI-V100: 16 SMs → increase items for fewer CTAs (more work per CTA)
struct bi100_lookback_1B_o4 {
  static constexpr int threads = 512;
  static constexpr int items   = 32;   // raised from 18: tile=16384, still <<49152
  static constexpr BlockLoadAlgorithm load_algo   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_mod      = LOAD_DEFAULT;
  static constexpr BlockStoreAlgorithm store_algo  = BLOCK_STORE_WARP_TRANSPOSE;
  // dcid_7=exponential_backon, ns=768×0.5=384, l2w=820×0.6=492
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::exponential_backon, 384, 492};
};

// CCCL SM100: ipt_13.tpb_512.ns_1384.dcid_7.l2w_720.trp_1.ld_0  1.128 1.003 1.120 1.308
// value_size=2: tile = 512×13×2 = 13312 ≤ 49152 ✓
struct bi100_lookback_2B_o4 {
  static constexpr int threads = 512;
  static constexpr int items   = 22;   // raised from 13: tile=22528 ≤ 49152 ✓
  static constexpr BlockLoadAlgorithm load_algo   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_mod      = LOAD_DEFAULT;
  static constexpr BlockStoreAlgorithm store_algo  = BLOCK_STORE_WARP_TRANSPOSE;
  // dcid_7=exponential_backon, ns=1384×0.5=692, l2w=720×0.6=432
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::exponential_backon, 692, 432};
};

// CCCL SM100: ipt_22.tpb_384.ns_1904.dcid_6.l2w_830.trp_1.ld_0  1.148 0.997 1.140 1.463
// value_size=4 (float32 — HOT PATH: paged_attention softmax denominator):
//   SM100 tile = 384×22×4 = 33792 ≤ 49152 ✓
//   BI-V100: keep items=22 (already near SMEM-optimal tile size for 48KB)
struct bi100_lookback_4B_o4 {
  static constexpr int threads = 384;
  static constexpr int items   = 22;   // same as SM100 — already optimal tile ratio
  static constexpr BlockLoadAlgorithm load_algo   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_mod      = LOAD_DEFAULT;
  static constexpr BlockStoreAlgorithm store_algo  = BLOCK_STORE_WARP_TRANSPOSE;
  // dcid_6=exponential_backon_jitter, ns=1904×0.5=952, l2w=830×0.6=498
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::exponential_backon_jitter, 952, 498};
};

// CCCL SM100: ipt_23.tpb_416.ns_772.dcid_5.l2w_710.trp_1.ld_0  1.089 1.016 1.086 1.265
// value_size=8 (int64/double):
//   SM100 tile = 416×23×8 = 76544 > 49152 OVERFLOW!
//   BI-V100 max_items = floor(49152 / (416×8)) = 14
//   But with threads=320: floor(49152 / (320×8)) = 19
struct bi100_lookback_8B_o4 {
  static constexpr int threads = 320;
  static constexpr int items   = 19;   // SMEM capped: 320×19×8 = 48640 ≤ 49152 ✓ (99% utilization!)
  static constexpr BlockLoadAlgorithm load_algo   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_mod      = LOAD_DEFAULT;
  static constexpr BlockStoreAlgorithm store_algo  = BLOCK_STORE_WARP_TRANSPOSE;
  // dcid_5=exponential_backon_jitter_window, ns=772×0.5=386, l2w=710×0.6=426
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::exponential_backon_jitter_window, 386, 426};
};

// --- plus<> operator, offset_size=8 ---

// CCCL SM100: ipt_14.tpb_384.ns_228.dcid_7.l2w_775.trp_1.ld_1  1.107 1.000 1.101 1.308
// value_size=1, offset=8: tile = 384×14×1 = 5376 ≤ 49152 ✓
struct bi100_lookback_1B_o8 {
  static constexpr int threads = 384;
  static constexpr int items   = 28;   // raised from 14: tile=10752, BI-V100 wants larger tiles
  static constexpr BlockLoadAlgorithm load_algo   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_mod      = LOAD_DEFAULT;  // SM100 used LOAD_CA(ld_1), but BI-V100 prefers DEFAULT
  static constexpr BlockStoreAlgorithm store_algo  = BLOCK_STORE_WARP_TRANSPOSE;
  // dcid_7=exponential_backon, ns=228×0.5=114, l2w=775×0.6=465
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::exponential_backon, 114, 465};
};

// CCCL SM100: no specialization for value_size=2, offset=8 (regresses for large inputs)
// Fall through to SM90 tuning

// CCCL SM100: ipt_19.tpb_416.ns_956.dcid_7.l2w_550.trp_1.ld_1  1.146 0.994 1.137 1.456
// value_size=4, offset=8: tile = 416×19×4 = 31616 ≤ 49152 ✓
struct bi100_lookback_4B_o8 {
  static constexpr int threads = 416;
  static constexpr int items   = 19;   // same as SM100
  static constexpr BlockLoadAlgorithm load_algo   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_mod      = LOAD_DEFAULT;  // SM100 used LOAD_CA
  static constexpr BlockStoreAlgorithm store_algo  = BLOCK_STORE_WARP_TRANSPOSE;
  // dcid_7=exponential_backon, ns=956×0.5=478, l2w=550×0.6=330
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::exponential_backon, 478, 330};
};

// CCCL SM100: ipt_22.tpb_320.ns_328.dcid_2.l2w_965.trp_1.ld_0  1.080 1.000 1.076 1.249
// value_size=8, offset=8: tile = 320×22×8 = 56320 > 49152 OVERFLOW!
//   BI-V100 max_items = floor(49152 / (320×8)) = 19
struct bi100_lookback_8B_o8 {
  static constexpr int threads = 320;
  static constexpr int items   = 19;   // SMEM capped: 320×19×8 = 48640 ≤ 49152 ✓
  static constexpr BlockLoadAlgorithm load_algo   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_mod      = LOAD_DEFAULT;
  static constexpr BlockStoreAlgorithm store_algo  = BLOCK_STORE_WARP_TRANSPOSE;
  // dcid_2=exponential_backoff, ns=328×0.5=164, l2w=965×0.6=579
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::exponential_backoff, 164, 579};
};

// --- SM90 fallback tunings (used when SM100 has no specialization) ---
// These come from CCCL sm90_tuning tables, with SMEM-capped items for BI-V100.
// All use fixed_delay (SM90 didn't have the advanced delay algorithms).
//
// CCCL SM90 table (fixed_delay_constructor_t<ns, l2w>):
//   accum_size=1: tpb=192, ipt=22, ns=168, l2w=1140
//   accum_size=2: tpb=512, ipt=12, ns=376, l2w=1125
//   accum_size=4: tpb=128, ipt=24, ns=648, l2w=1245  (generic)
//     float32:    tpb=128, ipt=24, ns=688, l2w=1140
//   accum_size=8: tpb=224, ipt=24, ns=632, l2w=1290  (generic)
//     float64:    tpb=224, ipt=24, ns=576, l2w=1215
//   accum_size=16(int128): tpb=576, ipt=21, ns=860, l2w=630

struct bi100_sm90_accum1 {
  static constexpr int threads = 192;
  static constexpr int items   = 32;  // raised from 22: tile=6144 (accum=1B), <<49152
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::fixed_delay, 84, 684};  // ns×0.5, l2w×0.6
};

struct bi100_sm90_accum2 {
  static constexpr int threads = 512;
  static constexpr int items   = 22;  // raised from 12: tile=22528 (accum=2B), ≤49152 ✓
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::fixed_delay, 188, 675};
};

struct bi100_sm90_accum4_generic {
  static constexpr int threads = 128;
  static constexpr int items   = 24;  // same as SM90: tile=12288 (accum=4B), ≤49152 ✓
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::fixed_delay, 324, 747};
};

struct bi100_sm90_float32 {
  static constexpr int threads = 128;
  static constexpr int items   = 24;  // same as SM90: tile=12288, ≤49152 ✓
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::fixed_delay, 344, 684};
};

struct bi100_sm90_accum8_generic {
  static constexpr int threads = 224;
  static constexpr int items   = 24;  // SM90: tile=224×24×8=43008, ≤49152 ✓ (87% utilization)
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::fixed_delay, 316, 774};
};

struct bi100_sm90_float64 {
  static constexpr int threads = 224;
  static constexpr int items   = 24;  // same as SM90: tile=43008, ≤49152 ✓
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::fixed_delay, 288, 729};
};

struct bi100_sm90_accum16 {
  // SM90: tpb=576, ipt=21 → tile=576×21×16=193536 OVERFLOW!
  // BI-V100: max_items = floor(49152/(192×16))=16, or floor(49152/(128×16))=24
  static constexpr int threads = 192;
  static constexpr int items   = 16;  // SMEM capped: 192×16×16 = 49152 = EXACT FIT
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::fixed_delay, 430, 378};
};

// --- SM80 fallback tunings (for non-plus operators or non-primitive accumulators) ---
// CCCL SM80 table:
//   accum_size=1: tpb=320, ipt=14, ns=368, l2w=725
//   accum_size=2: tpb=352, ipt=16, ns=488, l2w=1040
//   accum_size=4: tpb=320, ipt=12, ns=268, l2w=1180 (generic)
//     float32:    tpb=288, ipt=8,  ns=724, l2w=1050
//   accum_size=8: tpb=288, ipt=22, ns=716, l2w=785  (generic)
//     float64:    tpb=384, ipt=12, ns=388, l2w=1100
//   int128:       tpb=640, ipt=24, ns=1200, l2w=0 (BLOCK_LOAD_DIRECT)

struct bi100_sm80_accum1 {
  static constexpr int threads = 320;
  static constexpr int items   = 28;  // raised from 14: tile=8960 (1B), <<49152
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::fixed_delay, 184, 435};
};

struct bi100_sm80_accum2 {
  static constexpr int threads = 352;
  static constexpr int items   = 24;  // raised from 16: tile=16896 (2B), ≤49152 ✓
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::fixed_delay, 244, 624};
};

struct bi100_sm80_accum4_generic {
  static constexpr int threads = 320;
  static constexpr int items   = 24;  // raised from 12: tile=30720 (4B), ≤49152 ✓
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::fixed_delay, 134, 708};
};

struct bi100_sm80_float32 {
  static constexpr int threads = 288;
  static constexpr int items   = 24;  // raised from 8: tile=27648 (4B), ≤49152 ✓
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::fixed_delay, 362, 630};
};

struct bi100_sm80_accum8_generic {
  static constexpr int threads = 288;
  static constexpr int items   = 20;  // SMEM limited: 288×22×8=50688>49152, use 20→46080 ✓
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::fixed_delay, 358, 471};
};

struct bi100_sm80_float64 {
  static constexpr int threads = 384;
  static constexpr int items   = 16;  // SMEM limited: 384×12×8=36864 → can raise to 16→49152 EXACT FIT
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::fixed_delay, 194, 660};
};

struct bi100_sm80_int128 {
  // SM80: tpb=640, ipt=24, BLOCK_LOAD_DIRECT → no SMEM staging for load
  // But BLOCK_STORE_DIRECT also avoids SMEM → register-only pipeline
  // BI-V100: safe at 640×24 since direct load doesn't use SMEM for staging
  static constexpr int threads = 640;
  static constexpr int items   = 16;  // conservative: 640×16×16=163840 regs only (no SMEM staging)
  static constexpr BlockLoadAlgorithm load_algo   = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_mod      = LOAD_DEFAULT;
  static constexpr BlockStoreAlgorithm store_algo  = BLOCK_STORE_DIRECT;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::no_delay, 0, 600};
};

// --- Default fallback (for non-tuned types/operators) ---
// CCCL: tpb=128, ipt=15 with default delay
struct bi100_default {
  static constexpr int threads = 128;
  static constexpr int items   = 24;  // raised from 15: 128×24×4=12288 ≤ 49152 for 4B types
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::fixed_delay, 350, 450};
};

// ============================================================
// policy_selector
//
// Mirrors CCCL's policy_selector with the same field signature.
// BI-V100 ALWAYS returns ScanAlgorithm::lookback (no lookahead).
//
// Dispatch priority:
//   1. SM100 benchmark-matched tunings (plus + primitive + matching value/offset sizes)
//   2. SM90 accum-size tunings (plus + primitive accum)
//   3. SM80 accum-size tunings (primitive op + primitive accum)
//   4. Default fallback
// ============================================================

struct policy_selector {
  int input_value_size;
  int input_value_alignment;
  int output_value_size;
  int output_value_alignment;
  int accum_size;
  int accum_alignment;
  int offset_size;
  type_t input_type;
  type_t accum_type;
  op_kind_t operation_t;
  bool input_contiguous;
  bool output_contiguous;
  bool input_trivially_copyable;
  bool output_trivially_copyable;
  bool output_default_constructible;
  bool accum_is_primitive_or_trivially_copy_constructible;
  bool benchmark_match;
  bool require_stable_reduction_order = false;

  constexpr ScanPolicy operator()(const hardware_capability& hw) const {
    // BI-V100: always lookback (no lookahead — PTX ISA too old)

    const bool large_values = accum_size > 128;
    const BlockLoadAlgorithm transposed_load =
      large_values ? BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED : BLOCK_LOAD_WARP_TRANSPOSE;
    const BlockStoreAlgorithm transposed_store =
      large_values ? BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED : BLOCK_STORE_WARP_TRANSPOSE;

    if (!hw.at_least(hardware_capability::vendor_t::iluvatar, 100)) {
      // Non BI-V100: fall through to default
      return make_lookback_policy(
        bi100_default::threads, bi100_default::items, accum_size,
        transposed_load, LOAD_DEFAULT, transposed_store,
        BLOCK_SCAN_WARP_SCANS, bi100_default::delay);
    }

    // ---- Tier 1: SM100 benchmark-matched (highest priority) ----
    // Only for plus<> + primitive accum + benchmark_match
    if (benchmark_match && operation_t == op_kind_t::plus
        && accum_is_primitive_or_trivially_copy_constructible) {

      if (offset_size == 4) {
        switch (input_value_size) {
          case 1:
            return make_lookback_policy(
              bi100_lookback_1B_o4::threads, bi100_lookback_1B_o4::items, accum_size,
              bi100_lookback_1B_o4::load_algo, bi100_lookback_1B_o4::load_mod,
              bi100_lookback_1B_o4::store_algo, BLOCK_SCAN_WARP_SCANS,
              bi100_lookback_1B_o4::delay);
          case 2:
            return make_lookback_policy(
              bi100_lookback_2B_o4::threads, bi100_lookback_2B_o4::items, accum_size,
              bi100_lookback_2B_o4::load_algo, bi100_lookback_2B_o4::load_mod,
              bi100_lookback_2B_o4::store_algo, BLOCK_SCAN_WARP_SCANS,
              bi100_lookback_2B_o4::delay);
          case 4:
            return make_lookback_policy(
              bi100_lookback_4B_o4::threads, bi100_lookback_4B_o4::items, accum_size,
              bi100_lookback_4B_o4::load_algo, bi100_lookback_4B_o4::load_mod,
              bi100_lookback_4B_o4::store_algo, BLOCK_SCAN_WARP_SCANS,
              bi100_lookback_4B_o4::delay);
          case 8:
            return make_lookback_policy(
              bi100_lookback_8B_o4::threads, bi100_lookback_8B_o4::items, accum_size,
              bi100_lookback_8B_o4::load_algo, bi100_lookback_8B_o4::load_mod,
              bi100_lookback_8B_o4::store_algo, BLOCK_SCAN_WARP_SCANS,
              bi100_lookback_8B_o4::delay);
          default: break;
        }
      }
      else if (offset_size == 8) {
        switch (input_value_size) {
          case 1:
            return make_lookback_policy(
              bi100_lookback_1B_o8::threads, bi100_lookback_1B_o8::items, accum_size,
              bi100_lookback_1B_o8::load_algo, bi100_lookback_1B_o8::load_mod,
              bi100_lookback_1B_o8::store_algo, BLOCK_SCAN_WARP_SCANS,
              bi100_lookback_1B_o8::delay);
          // case 2: intentionally omitted — CCCL SM100 also omits (regresses for large inputs)
          case 4:
            return make_lookback_policy(
              bi100_lookback_4B_o8::threads, bi100_lookback_4B_o8::items, accum_size,
              bi100_lookback_4B_o8::load_algo, bi100_lookback_4B_o8::load_mod,
              bi100_lookback_4B_o8::store_algo, BLOCK_SCAN_WARP_SCANS,
              bi100_lookback_4B_o8::delay);
          case 8:
            if (accum_type == type_t::float64) {
              break;  // float64 + offset=8: CCCL falls through to SM90 too
            }
            return make_lookback_policy(
              bi100_lookback_8B_o8::threads, bi100_lookback_8B_o8::items, accum_size,
              bi100_lookback_8B_o8::load_algo, bi100_lookback_8B_o8::load_mod,
              bi100_lookback_8B_o8::store_algo, BLOCK_SCAN_WARP_SCANS,
              bi100_lookback_8B_o8::delay);
          default: break;
        }
      }
    }

    // ---- Tier 2: SM90-level accum-size tunings ----
    // For plus<> + primitive accum, dispatch by accum_size
    if (operation_t != op_kind_t::other
        && accum_is_primitive_or_trivially_copy_constructible) {

      switch (accum_size) {
        case 1:
          return make_lookback_policy(
            bi100_sm90_accum1::threads, bi100_sm90_accum1::items, accum_size,
            BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE,
            BLOCK_SCAN_WARP_SCANS, bi100_sm90_accum1::delay);
        case 2:
          return make_lookback_policy(
            bi100_sm90_accum2::threads, bi100_sm90_accum2::items, accum_size,
            BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE,
            BLOCK_SCAN_WARP_SCANS, bi100_sm90_accum2::delay);
        case 4:
          if (accum_type == type_t::float32) {
            return make_lookback_policy(
              bi100_sm90_float32::threads, bi100_sm90_float32::items, accum_size,
              BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE,
              BLOCK_SCAN_WARP_SCANS, bi100_sm90_float32::delay);
          }
          return make_lookback_policy(
            bi100_sm90_accum4_generic::threads, bi100_sm90_accum4_generic::items, accum_size,
            BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE,
            BLOCK_SCAN_WARP_SCANS, bi100_sm90_accum4_generic::delay);
        case 8:
          if (accum_type == type_t::float64) {
            return make_lookback_policy(
              bi100_sm90_float64::threads, bi100_sm90_float64::items, accum_size,
              BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE,
              BLOCK_SCAN_WARP_SCANS, bi100_sm90_float64::delay);
          }
          return make_lookback_policy(
            bi100_sm90_accum8_generic::threads, bi100_sm90_accum8_generic::items, accum_size,
            BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE,
            BLOCK_SCAN_WARP_SCANS, bi100_sm90_accum8_generic::delay);
        case 16:
          if (accum_type == type_t::int128 || accum_type == type_t::uint128) {
            return make_lookback_policy(
              bi100_sm90_accum16::threads, bi100_sm90_accum16::items, accum_size,
              BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE,
              BLOCK_SCAN_WARP_SCANS, bi100_sm90_accum16::delay);
          }
          break;
        default: break;
      }
    }

    // ---- Tier 3: SM80-level tunings ----
    // For primitive op + primitive accum
    if (operation_t != op_kind_t::other) {
      if (accum_is_primitive_or_trivially_copy_constructible) {
        switch (accum_size) {
          case 1:
            return make_lookback_policy(
              bi100_sm80_accum1::threads, bi100_sm80_accum1::items, accum_size,
              BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE,
              BLOCK_SCAN_WARP_SCANS, bi100_sm80_accum1::delay);
          case 2:
            return make_lookback_policy(
              bi100_sm80_accum2::threads, bi100_sm80_accum2::items, accum_size,
              BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE,
              BLOCK_SCAN_WARP_SCANS, bi100_sm80_accum2::delay);
          case 4:
            if (accum_type == type_t::float32) {
              return make_lookback_policy(
                bi100_sm80_float32::threads, bi100_sm80_float32::items, accum_size,
                BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE,
                BLOCK_SCAN_WARP_SCANS, bi100_sm80_float32::delay);
            }
            return make_lookback_policy(
              bi100_sm80_accum4_generic::threads, bi100_sm80_accum4_generic::items, accum_size,
              BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE,
              BLOCK_SCAN_WARP_SCANS, bi100_sm80_accum4_generic::delay);
          case 8:
            if (accum_type == type_t::float64) {
              return make_lookback_policy(
                bi100_sm80_float64::threads, bi100_sm80_float64::items, accum_size,
                BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE,
                BLOCK_SCAN_WARP_SCANS, bi100_sm80_float64::delay);
            }
            return make_lookback_policy(
              bi100_sm80_accum8_generic::threads, bi100_sm80_accum8_generic::items, accum_size,
              BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE,
              BLOCK_SCAN_WARP_SCANS, bi100_sm80_accum8_generic::delay);
          case 16:
            // int128 with BLOCK_LOAD_DIRECT (no SMEM staging)
            return make_lookback_policy(
              bi100_sm80_int128::threads, bi100_sm80_int128::items, accum_size,
              bi100_sm80_int128::load_algo, bi100_sm80_int128::load_mod,
              bi100_sm80_int128::store_algo,
              BLOCK_SCAN_WARP_SCANS, bi100_sm80_int128::delay);
          default: break;
        }
      }
    }

    // ---- Tier 4: Default fallback ----
    return make_lookback_policy(
      bi100_default::threads, bi100_default::items, accum_size,
      transposed_load, LOAD_DEFAULT, transposed_store,
      BLOCK_SCAN_WARP_SCANS, bi100_default::delay);
  }
};

} // namespace muh::tuning::scan
