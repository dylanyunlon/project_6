// muh/include/muh/tuning/tuning_scan_by_key.cuh — BI-V100
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_scan_by_key.cuh
// CCCL upstream: 85KB, 66 sm80_tuning type specializations (key_size × val_size)
//
// vllm relevance: per-sequence cumulative softmax denominator in paged_attention.
//   Each sequence is a key segment; within each segment, scan computes
//   cumsum(exp(score)). With max_num_seqs=8, up to 8 concurrent key segments.
//
// HARDWARE (confirmed):
//   SM=16, SMEM=48KB, L2=6MB
//   scan_by_key tile = threads * items * (key_size + accum_size)
//
// STRATEGY: Adapt CCCL sm80 tunings (they are the BI-V100-closest architecture
// due to similar SMEM constraints). The key difference is SM=16 means fewer
// concurrent CTAs, so tiles should be sized to FILL SMEM where sm80 left slack.
// Delay parameters halved for L2=6MB (less contention on lookback status).

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::scan_by_key {

enum class ScanByKeyAlgorithm { lookback };

struct ScanByKeyLookbackPolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  BlockStoreAlgorithm store_algorithm;
  BlockScanAlgorithm scan_algorithm;
  LookbackDelayPolicy lookback_delay;
};

struct ScanByKeyPolicy {
  ScanByKeyAlgorithm algorithm;
  ScanByKeyLookbackPolicy lookback;
};

// ============================================================
// BI-V100 tunings — adapted from CCCL sm80_tuning
//
// CCCL sm80 tiles all fit 48KB (max pair_tile = 192*10*10 = 19200).
// SM=16 strategy: increase items where sm80 left SMEM headroom.
//
// Tile constraint: threads * items * (key_size + val_size) <= 49152
//
// Delay: sm80 uses no_delay(ns) or fixed_delay(delay, l2w).
// For BI-V100 L2=6MB: halve fixed_delay values, keep no_delay
// (no_delay only uses a spin count, not L2-dependent).
// ============================================================

// key=1B, val=1B → pair=2B, sm80: tpb=128 ipt=12 → tile=3072 (6% SMEM)
// SM=16 fix: tpb=256 ipt=24 → tile=12288 (25% SMEM, 4x more data/CTA)
struct bi100_k1_v1 {
  static constexpr int threads = 256;
  static constexpr int items   = 24;
  static constexpr BlockLoadAlgorithm  load  = BLOCK_LOAD_DIRECT;
  static constexpr BlockStoreAlgorithm store = BLOCK_STORE_DIRECT;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::no_delay, 795, 0};
};

// key=1B, val=2B → pair=3B, sm80: tpb=288 ipt=12 → tile=10368 (21%)
// SM=16: tpb=288 ipt=18 → tile=15552 (32%)
struct bi100_k1_v2 {
  static constexpr int threads = 288;
  static constexpr int items   = 18;
  static constexpr BlockLoadAlgorithm  load  = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::no_delay, 825, 0};
};

// key=1B, val=4B → pair=5B, sm80: tpb=256 ipt=15 → tile=19200 (39%)
// SM=16: tpb=256 ipt=24 → tile=30720 (62%)
struct bi100_k1_v4 {
  static constexpr int threads = 256;
  static constexpr int items   = 24;
  static constexpr BlockLoadAlgorithm  load  = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::no_delay, 640, 0};
};

// key=1B, val=8B → pair=9B, sm80: tpb=192 ipt=10 → tile=17280 (35%)
// SM=16: tpb=192 ipt=16 → tile=27648 (56%)
struct bi100_k1_v8 {
  static constexpr int threads = 192;
  static constexpr int items   = 16;
  static constexpr BlockLoadAlgorithm  load  = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::fixed_delay, 62, 520};  // halved from sm80's 124,1040
};

// key=2B, val=1B → pair=3B, sm80: tpb=256 ipt=8 → tile=6144 (12%)
// SM=16: tpb=256 ipt=20 → tile=15360 (31%)
struct bi100_k2_v1 {
  static constexpr int threads = 256;
  static constexpr int items   = 20;
  static constexpr BlockLoadAlgorithm  load  = BLOCK_LOAD_DIRECT;
  static constexpr BlockStoreAlgorithm store = BLOCK_STORE_DIRECT;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::no_delay, 1070, 0};
};

// key=2B, val=2B → pair=4B, sm80: tpb=320 ipt=14 → tile=17920 (36%)
// SM=16: tpb=320 ipt=20 → tile=25600 (52%)
struct bi100_k2_v2 {
  static constexpr int threads = 320;
  static constexpr int items   = 20;
  static constexpr BlockLoadAlgorithm  load  = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::no_delay, 625, 0};
};

// key=2B, val=4B → pair=6B, sm80: tpb=256 ipt=15 → tile=23040 (47%)
// SM=16: tpb=256 ipt=20 → tile=30720 (62%)
struct bi100_k2_v4 {
  static constexpr int threads = 256;
  static constexpr int items   = 20;
  static constexpr BlockLoadAlgorithm  load  = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::no_delay, 1055, 0};
};

// key=2B, val=8B → pair=10B, sm80: tpb=160 ipt=17 → tile=27200 (55%)
// SM=16: keep — already at 55%, good balance
struct bi100_k2_v8 {
  static constexpr int threads = 160;
  static constexpr int items   = 17;
  static constexpr BlockLoadAlgorithm  load  = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::fixed_delay, 80, 348};  // halved from sm80's 160,695
};

// key=4B, val=1B → pair=5B, sm80: tpb=256 ipt=8 → tile=10240 (21%)
// SM=16: tpb=256 ipt=16 → tile=20480 (42%)
struct bi100_k4_v1 {
  static constexpr int threads = 256;
  static constexpr int items   = 16;
  static constexpr BlockLoadAlgorithm  load  = BLOCK_LOAD_DIRECT;
  static constexpr BlockStoreAlgorithm store = BLOCK_STORE_DIRECT;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::no_delay, 1070, 0};
};

// key=4B, val=2B → pair=6B, sm80: tpb=320 ipt=14 → tile=26880 (55%)
// SM=16: keep — already good
struct bi100_k4_v2 {
  static constexpr int threads = 320;
  static constexpr int items   = 14;
  static constexpr BlockLoadAlgorithm  load  = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::no_delay, 625, 0};
};

// key=4B, val=4B → pair=8B. THE HOT PATH: int32 key + float32 value = attention score
// sm80: tpb=256 ipt=15 → tile=30720 (62%)
// SM=16: tpb=256 ipt=24 → tile=49152 (100% SMEM — maximize for decode hot path)
struct bi100_k4_v4 {
  static constexpr int threads = 256;
  static constexpr int items   = 24;
  static constexpr BlockLoadAlgorithm  load  = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::no_delay, 1055, 0};
};

// key=4B, val=8B → pair=12B, sm80: tpb=160 ipt=17 → tile=32640 (66%)
// SM=16: tpb=192 ipt=21 → tile=48384 (98%)
struct bi100_k4_v8 {
  static constexpr int threads = 192;
  static constexpr int items   = 21;
  static constexpr BlockLoadAlgorithm  load  = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::fixed_delay, 80, 348};
};

// key=8B, val=4B → pair=12B (same as k4_v8)
struct bi100_k8_v4 {
  static constexpr int threads = 192;
  static constexpr int items   = 21;
  static constexpr BlockLoadAlgorithm  load  = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::fixed_delay, 80, 348};
};

// key=8B, val=8B → pair=16B, sm80: tpb=128 ipt=16 → tile=32768 (67%)
// SM=16: tpb=192 ipt=16 → tile=49152 (100%)
struct bi100_k8_v8 {
  static constexpr int threads = 192;
  static constexpr int items   = 16;
  static constexpr BlockLoadAlgorithm  load  = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::fixed_delay, 80, 348};
};

// ============================================================
// Default fallback for unknown key/val size combinations
// ============================================================
struct bi100_default {
  static constexpr int threads = 192;
  static constexpr int items   = 12;
  static constexpr BlockLoadAlgorithm  load  = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store = BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::fixed_delay, 175, 225};
};

// ============================================================
// policy_selector — key_size × val_size dispatch
// ============================================================

struct policy_selector {
  int key_size;
  int accum_size;
  int offset_size;

  constexpr ScanByKeyPolicy operator()(const hardware_capability& hw) const {
    if (!hw.at_least(hardware_capability::vendor_t::iluvatar, 100)) {
      // Non-BI-V100 fallback
      auto [i, t] = scale_mem_bound(192, 12, key_size + accum_size);
      return {ScanByKeyAlgorithm::lookback,
              {t, i, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
               BLOCK_STORE_WARP_TRANSPOSE, BLOCK_SCAN_WARP_SCANS,
               bi100_default::delay}};
    }

    // BI-V100 type dispatch — key_size × val_size
    // Select the best tuning for this combination

    #define MK_POLICY(T) ScanByKeyPolicy{ScanByKeyAlgorithm::lookback, \
      {T::threads, T::items, T::load, LOAD_DEFAULT, T::store, BLOCK_SCAN_WARP_SCANS, T::delay}}

    if (key_size == 1) {
      if (accum_size == 1) return MK_POLICY(bi100_k1_v1);
      if (accum_size == 2) return MK_POLICY(bi100_k1_v2);
      if (accum_size == 4) return MK_POLICY(bi100_k1_v4);
      if (accum_size == 8) return MK_POLICY(bi100_k1_v8);
    }
    if (key_size == 2) {
      if (accum_size == 1) return MK_POLICY(bi100_k2_v1);
      if (accum_size == 2) return MK_POLICY(bi100_k2_v2);
      if (accum_size == 4) return MK_POLICY(bi100_k2_v4);
      if (accum_size == 8) return MK_POLICY(bi100_k2_v8);
    }
    if (key_size == 4) {
      if (accum_size == 1) return MK_POLICY(bi100_k4_v1);
      if (accum_size == 2) return MK_POLICY(bi100_k4_v2);
      if (accum_size == 4) return MK_POLICY(bi100_k4_v4);  // HOT PATH
      if (accum_size == 8) return MK_POLICY(bi100_k4_v8);
    }
    if (key_size == 8) {
      if (accum_size == 4) return MK_POLICY(bi100_k8_v4);
      if (accum_size == 8) return MK_POLICY(bi100_k8_v8);
    }

    #undef MK_POLICY

    // Fallback: compute safe items from SMEM constraint
    int pair_size = key_size + accum_size;
    int max_items = hw.max_shared_memory_per_block / (bi100_default::threads * pair_size);
    if (max_items > 24) max_items = 24;
    if (max_items < 1) max_items = 1;

    return {ScanByKeyAlgorithm::lookback,
            {bi100_default::threads, max_items, bi100_default::load, LOAD_DEFAULT,
             bi100_default::store, BLOCK_SCAN_WARP_SCANS, bi100_default::delay}};
  }
};

} // namespace muh::tuning::scan_by_key
