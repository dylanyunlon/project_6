// muh/include/muh/tuning/tuning_reduce_by_key.cuh — BI-V100
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_reduce_by_key.cuh
// CCCL upstream: 70KB, 67 sm80_tuning specializations (key_size × accum_size)
//
// vllm relevance: KV cache eviction scoring (reduce per-key attention scores),
//   MoE expert routing (reduce per-expert token counts/scores).
//
// HARDWARE: SM=16, SMEM=48KB, L2=6MB
//   reduce_by_key tile = threads * items * (key_size + accum_size + output overhead)
//   With lookback: additional SMEM for scan status tile
//
// STRATEGY: Adapt CCCL sm80 tunings with SM=16 tile maximization.
// reduce_by_key has 7 %RANGE% parameters in CCCL benchmarks:
//   ipt, tpb, trp(transpose), ld(load), ns(delay), dcid(delay_type), l2w(L2_latency)
// We preserve the key dispatch dimensions: key_size × accum_size × delay

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::reduce_by_key {

enum class ReduceByKeyAlgorithm { lookback };

struct ReduceByKeyLookbackPolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  BlockScanAlgorithm scan_algorithm;
  LookbackDelayPolicy delay;
};

struct ReduceByKeyPolicy {
  ReduceByKeyAlgorithm algorithm;
  ReduceByKeyLookbackPolicy lookback;
};

// ============================================================
// BI-V100 tunings — key_size × accum_size dispatch
//
// SM=16 strategy: pair_tile = threads * items * (key_size + accum_size)
// Target ≥50% SMEM utilization where sm80 was at 20-40%.
// Delay halved for L2=6MB.
// ============================================================

// key=1B, accum=1B → pair=2B
struct bi100_k1_a1 {
  static constexpr int threads = 320;
  static constexpr int items   = 24;  // tile=320*24*2=15360 (31%)
  static constexpr BlockLoadAlgorithm load = BLOCK_LOAD_DIRECT;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::no_delay, 560, 0};
};

// key=1B, accum=2B → pair=3B
struct bi100_k1_a2 {
  static constexpr int threads = 288;
  static constexpr int items   = 20;  // tile=288*20*3=17280 (35%)
  static constexpr BlockLoadAlgorithm load = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::no_delay, 680, 0};
};

// key=1B, accum=4B → pair=5B
struct bi100_k1_a4 {
  static constexpr int threads = 256;
  static constexpr int items   = 20;  // tile=256*20*5=25600 (52%)
  static constexpr BlockLoadAlgorithm load = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::no_delay, 750, 0};
};

// key=1B, accum=8B → pair=9B
struct bi100_k1_a8 {
  static constexpr int threads = 192;
  static constexpr int items   = 16;  // tile=192*16*9=27648 (56%)
  static constexpr BlockLoadAlgorithm load = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::fixed_delay, 62, 520};
};

// key=2B, accum=2B → pair=4B
struct bi100_k2_a2 {
  static constexpr int threads = 320;
  static constexpr int items   = 20;  // tile=320*20*4=25600 (52%)
  static constexpr BlockLoadAlgorithm load = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::no_delay, 625, 0};
};

// key=2B, accum=4B → pair=6B
struct bi100_k2_a4 {
  static constexpr int threads = 256;
  static constexpr int items   = 20;  // tile=256*20*6=30720 (62%)
  static constexpr BlockLoadAlgorithm load = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::no_delay, 1055, 0};
};

// key=4B, accum=4B → pair=8B. HOT PATH: int32 key + float32 accum
struct bi100_k4_a4 {
  static constexpr int threads = 256;
  static constexpr int items   = 24;  // tile=256*24*8=49152 (100% SMEM!)
  static constexpr BlockLoadAlgorithm load = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::exponential_backon_jitter, 400, 300};
};

// key=4B, accum=8B → pair=12B
struct bi100_k4_a8 {
  static constexpr int threads = 192;
  static constexpr int items   = 21;  // tile=192*21*12=48384 (98%)
  static constexpr BlockLoadAlgorithm load = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::fixed_delay, 80, 348};
};

// key=8B, accum=4B → pair=12B
struct bi100_k8_a4 {
  static constexpr int threads = 192;
  static constexpr int items   = 21;  // tile=192*21*12=48384 (98%)
  static constexpr BlockLoadAlgorithm load = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::fixed_delay, 80, 348};
};

// key=8B, accum=8B → pair=16B
struct bi100_k8_a8 {
  static constexpr int threads = 192;
  static constexpr int items   = 16;  // tile=192*16*16=49152 (100%)
  static constexpr BlockLoadAlgorithm load = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::fixed_delay, 80, 348};
};

// Default fallback
struct bi100_default {
  static constexpr int threads = 192;
  static constexpr int items   = 12;
  static constexpr BlockLoadAlgorithm load = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr LookbackDelayPolicy delay = {
    LookbackDelayAlgorithm::fixed_delay, 175, 225};
};

// ============================================================
// policy_selector
// ============================================================

struct policy_selector {
  int key_size;
  int accum_size;
  int offset_size;
  type_t accum_type;

  constexpr ReduceByKeyPolicy operator()(const hardware_capability& hw) const {
    if (!hw.at_least(hardware_capability::vendor_t::iluvatar, 100)) {
      return {ReduceByKeyAlgorithm::lookback,
              {bi100_default::threads, bi100_default::items, bi100_default::load,
               LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS, bi100_default::delay}};
    }

    #define MK(T) ReduceByKeyPolicy{ReduceByKeyAlgorithm::lookback, \
      {T::threads, T::items, T::load, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS, T::delay}}

    if (key_size <= 1) {
      if (accum_size <= 1) return MK(bi100_k1_a1);
      if (accum_size <= 2) return MK(bi100_k1_a2);
      if (accum_size <= 4) return MK(bi100_k1_a4);
      if (accum_size <= 8) return MK(bi100_k1_a8);
    }
    if (key_size <= 2) {
      if (accum_size <= 2) return MK(bi100_k2_a2);
      if (accum_size <= 4) return MK(bi100_k2_a4);
      // key=2B + accum=8B → use k4_a8 (similar pair size)
      if (accum_size <= 8) return MK(bi100_k4_a8);
    }
    if (key_size <= 4) {
      if (accum_size <= 4) return MK(bi100_k4_a4);  // HOT PATH
      if (accum_size <= 8) return MK(bi100_k4_a8);
    }
    if (key_size <= 8) {
      if (accum_size <= 4) return MK(bi100_k8_a4);
      if (accum_size <= 8) return MK(bi100_k8_a8);
    }

    #undef MK

    // Fallback with dynamic SMEM check
    int pair_size = key_size + accum_size;
    int items = hw.max_shared_memory_per_block / (bi100_default::threads * pair_size);
    if (items > 24) items = 24;
    if (items < 1) items = 1;

    return {ReduceByKeyAlgorithm::lookback,
            {bi100_default::threads, items, bi100_default::load,
             LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS, bi100_default::delay}};
  }
};

} // namespace muh::tuning::reduce_by_key
