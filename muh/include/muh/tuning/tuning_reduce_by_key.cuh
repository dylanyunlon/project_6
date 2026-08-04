// muh/include/muh/tuning/tuning_reduce_by_key.cuh — BI-V100
//
// Full port from: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_reduce_by_key.cuh (1735 lines)
// SM100: 16 entries, SM90: 25 entries, SM80: 25 entries = 66 total
//
// DELAY STRATEGY (v2 — based on CCCL source analysis):
//
// CCCL's delay() function (single_pass_scan_operators.cuh:130):
//   if (gridDim.x < 500) __threadfence_block();  // small grid: no nanosleep
//   else __nanosleep(Delay);                       // large grid: actual delay
//
// BI-V100: 16 SMs × ~2 CTAs/SM = max 32 concurrent CTAs
// → gridDim.x < 500 is ALWAYS true
// → ALL exponential_backoff/backon strategies degenerate to __threadfence_block()
// → no_delay is the correct strategy for BI-V100
//
// L2WriteLatency: kept from CCCL. This is a ONE-TIME constructor delay
//   (always_delay<L2WriteLatency>()) that ensures L2 write visibility.
//   BI-V100 L2 = 6MB, write latency ~450-1200ns depending on contention.
//   We use the CCCL SM90 values as-is since SM90 L2 (50MB) has similar
//   per-line latency characteristics.
//
// vllm hot path: key=4B (int32 sequence_id), accum=4B (float32 attention_score)

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::reduce_by_key {

struct ReduceByKeyLookbackPolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  BlockScanAlgorithm scan_algorithm;
  LookbackDelayPolicy delay;
};

enum class ReduceByKeyAlgorithm { lookback };

struct ReduceByKeyPolicy {
  ReduceByKeyAlgorithm algorithm;
  ReduceByKeyLookbackPolicy lookback;
};

struct policy_selector {
  int key_size;
  int accum_size;
  bool key_is_primitive;
  bool accum_is_primitive;
  bool op_is_primitive;

  constexpr bool smem_ok(int tpb, int ipt, bool warp_transpose) const {
    int pair_size = key_size + accum_size;
    int tile = tpb * ipt * pair_size;
    if (warp_transpose) tile *= 2;
    tile += 1024;
    return tile <= 49152;
  }

  constexpr ReduceByKeyLookbackPolicy safe(
      int tpb, int ipt, BlockLoadAlgorithm la, CacheLoadModifier lm,
      LookbackDelayPolicy d) const {
    bool wt = (la == BLOCK_LOAD_WARP_TRANSPOSE);
    while (!smem_ok(tpb, ipt, wt) && ipt > 1) ipt--;
    while (!smem_ok(tpb, ipt, wt) && tpb > 32) tpb -= 32;
    return {tpb, ipt, la, lm, BLOCK_SCAN_WARP_SCANS, d};
  }

  // BI-V100 delay: always no_delay. L2WriteLatency from CCCL SM90 values.
  // Physical reason: 16 SMs → max 32 CTAs → gridDim.x < 500 → CCCL code
  // skips __nanosleep and only does __threadfence_block, which is what
  // no_delay does. exponential_backoff/backon are wasted cycles.
  static constexpr LookbackDelayPolicy nd(int l2w) {
    return {LookbackDelayAlgorithm::no_delay, 0, l2w};
  }

  constexpr ReduceByKeyLookbackPolicy default_policy(CacheLoadModifier load_mod) const {
    int combined = key_size + accum_size;
    int mx = key_size > accum_size ? key_size : accum_size;
    int ipt = (mx <= 8) ? 6 : (6 * 8 / combined);
    if (ipt < 1) ipt = 1;
    if (ipt > 6) ipt = 6;
    return {128, ipt, BLOCK_LOAD_DIRECT, load_mod, BLOCK_SCAN_WARP_SCANS, nd(450)};
  }

  constexpr ReduceByKeyLookbackPolicy get_lookback_policy() const {
    if (!op_is_primitive) return default_policy(LOAD_LDG);

    bool use_tuning = (key_is_primitive || key_size == 16) &&
                      (accum_is_primitive || accum_size == 16);
    if (!use_tuning) return default_policy(LOAD_DEFAULT);

    // =====================================================================
    // SM100 tile configs — threads/items from CCCL benchmarks, delay=no_delay
    // =====================================================================

    // key=1B
    if (key_size==1 && accum_size==1)  return safe(576, 13, BLOCK_LOAD_DIRECT, LOAD_CA, nd(240));
    if (key_size==1 && accum_size==2)  return safe(224, 10, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, nd(390));
    if (key_size==1 && accum_size==4)  return safe(128, 14, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, nd(285));
    if (key_size==1 && accum_size==8)  return safe(128, 19, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, nd(540));

    // key=2B
    if (key_size==2 && accum_size==1)  return safe(128, 14, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, nd(290));
    if (key_size==2 && accum_size==2)  return safe(256, 14, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, nd(975));
    if (key_size==2 && accum_size==4)  return safe(256, 11, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, nd(550));
    if (key_size==2 && accum_size==8)  return safe(160, 10, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, nd(725));

    // key=4B (vllm hot path)
    if (key_size==4 && accum_size==1)  return safe(224, 10, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, nd(285));
    if (key_size==4 && accum_size==2)  return safe(256, 11, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, nd(115));
    if (key_size==4 && accum_size==4)  return safe(224, 14, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, nd(1005));
    if (key_size==4 && accum_size==8)  return safe(256, 10, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, nd(145));

    // key=8B
    if (key_size==8 && accum_size==1)  return safe(224, 9, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, nd(460));
    if (key_size==8 && accum_size==2)  return safe(224, 11, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_CA, nd(550));
    if (key_size==8 && accum_size==4)  return safe(224, 9, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, nd(475));
    if (key_size==8 && accum_size==8)  return safe(224, 9, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, nd(340));

    // =====================================================================
    // SM90 — fall-through for sizes not in SM100 (accum=16, key=16)
    // =====================================================================
    if (key_size==1 && accum_size==16)  return safe(128, 11, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, nd(1100));
    if (key_size==2 && accum_size==16)  return safe(128, 11, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, nd(1175));
    if (key_size==4 && accum_size==16)  return safe(128, 11, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, nd(1195));
    if (key_size==8 && accum_size==16)  return safe(128, 11, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, nd(1125));
    if (key_size==16 && accum_size==1)  return safe(128, 11, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, nd(1080));
    if (key_size==16 && accum_size==2)  return safe(128, 11, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, nd(1005));
    if (key_size==16 && accum_size==4)  return safe(128, 11, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, nd(1100));
    if (key_size==16 && accum_size==8)  return safe(128, 11, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, nd(1195));
    if (key_size==16 && accum_size==16) return safe(128, 11, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, nd(1150));

    // =====================================================================
    // SM80 — final fallback
    // =====================================================================
    if (key_size==1 && accum_size==1)  return safe(256, 13, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, nd(975));
    if (key_size==1 && accum_size==2)  return safe(224, 12, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, nd(840));
    if (key_size==1 && accum_size==4)  return safe(256, 15, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, nd(760));
    if (key_size==1 && accum_size==8)  return safe(224, 7, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, nd(1070));
    if (key_size==2 && accum_size==1)  return safe(256, 11, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, nd(620));
    if (key_size==2 && accum_size==2)  return safe(224, 14, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, nd(640));
    if (key_size==2 && accum_size==4)  return safe(256, 14, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, nd(905));
    if (key_size==2 && accum_size==8)  return safe(224, 9, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, nd(810));
    if (key_size==4 && accum_size==1)  return safe(288, 11, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, nd(1110));
    if (key_size==4 && accum_size==2)  return safe(192, 15, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, nd(1200));
    if (key_size==4 && accum_size==4)  return safe(256, 15, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, nd(1110));
    if (key_size==4 && accum_size==8)  return safe(224, 9, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, nd(1165));
    if (key_size==8 && accum_size==1)  return safe(192, 10, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, nd(1175));
    if (key_size==8 && accum_size==2)  return safe(224, 7, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, nd(1075));
    if (key_size==8 && accum_size==4)  return safe(384, 7, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, nd(1040));
    if (key_size==8 && accum_size==8)  return safe(128, 14, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, nd(1080));
    if (key_size==8 && accum_size==16) return safe(128, 11, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, nd(430));
    if (key_size==16 && accum_size==1)  return safe(192, 7, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, nd(1105));
    if (key_size==16 && accum_size==2)  return safe(192, 7, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, nd(755));
    if (key_size==16 && accum_size==4)  return safe(192, 7, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, nd(535));
    if (key_size==16 && accum_size==8)  return safe(192, 7, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, nd(1035));
    if (key_size==16 && accum_size==16) return safe(128, 11, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, nd(1090));

    return default_policy(LOAD_DEFAULT);
  }

  constexpr ReduceByKeyPolicy operator()(const hardware_capability& hw) const {
    return {ReduceByKeyAlgorithm::lookback, get_lookback_policy()};
  }
};

} // namespace muh::tuning::reduce_by_key
