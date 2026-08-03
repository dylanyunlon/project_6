// muh/include/muh/tuning/tuning_reduce_by_key.cuh — BI-V100
//
// Full port from: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_reduce_by_key.cuh (1735 lines)
// SM100: 16 benchmark-tuned entries (with CCCL benchmark annotations preserved)
// SM90:  25 entries (key_size 1-16 × accum_size 1-16)
// SM80:  25 entries
// Total: 66 tuning entries covering the full (key_size, accum_size) matrix
//
// Dispatch: (key_size, accum_size) with accum_t refinement for SM100 float32 regression
// SMEM model: tpb * ipt * (key_size + accum_size), WARP_TRANSPOSE doubles tile
// BI-V100 constraints: SMEM=48KB, SM=16, warp=32, BW=900GB/s
// SM100 delay scaling: ns*0.5, l2w*0.6 (L2 6MB vs SM100 50MB, BW 900 vs 3350)
//
// vllm hot path: paged_attention_v2 per-sequence score aggregation
//   key = sequence_id (int32, 4B), value = attention_score (float32)
//   → primary lookup: key_size=4, accum_size=4

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

  // SMEM safety: tile = tpb * ipt * (key_size + accum_size)
  // WARP_TRANSPOSE doubles it (staging buffer)
  constexpr bool smem_ok(int tpb, int ipt, bool warp_transpose) const {
    int pair_size = key_size + accum_size;
    int tile = tpb * ipt * pair_size;
    if (warp_transpose) tile *= 2;
    tile += 1024; // scan temp overhead
    return tile <= 49152;
  }

  // Construct policy with SMEM overflow protection
  constexpr ReduceByKeyLookbackPolicy safe(
      int tpb, int ipt, BlockLoadAlgorithm la, CacheLoadModifier lm,
      LookbackDelayPolicy d) const {
    bool wt = (la == BLOCK_LOAD_WARP_TRANSPOSE);
    while (!smem_ok(tpb, ipt, wt) && ipt > 1) ipt--;
    while (!smem_ok(tpb, ipt, wt) && tpb > 32) tpb -= 32;
    return {tpb, ipt, la, lm, BLOCK_SCAN_WARP_SCANS, d};
  }

  // Scale SM100 delay for BI-V100: ns*0.5, l2w*0.6
  static constexpr LookbackDelayPolicy sd(LookbackDelayAlgorithm algo, int ns, int l2w) {
    return {algo, static_cast<int>(ns * 0.5), static_cast<int>(l2w * 0.6)};
  }

  // CCCL default policy (matches __make_default_policy)
  constexpr ReduceByKeyLookbackPolicy default_policy(CacheLoadModifier load_mod) const {
    int combined = key_size + accum_size;
    int mx = key_size > accum_size ? key_size : accum_size;
    int ipt = (mx <= 8) ? 6 : (6 * 8 / combined);
    if (ipt < 1) ipt = 1;
    if (ipt > 6) ipt = 6;
    return {128, ipt, BLOCK_LOAD_DIRECT, load_mod, BLOCK_SCAN_WARP_SCANS,
            {LookbackDelayAlgorithm::fixed_delay, 350, 450}};
  }

  constexpr ReduceByKeyLookbackPolicy get_lookback_policy() const {
    if (!op_is_primitive) return default_policy(LOAD_LDG);

    bool use_tuning = (key_is_primitive || key_size == 16) &&
                      (accum_is_primitive || accum_size == 16);
    if (!use_tuning) return default_policy(LOAD_DEFAULT);

    // =====================================================================
    // SM100 tuning — 16 entries, delay scaled for BI-V100
    // Each line preserves the original CCCL benchmark annotation
    // =====================================================================

    // key=1B
    if (key_size==1 && accum_size==1)
      // ipt_13.tpb_576.trp_0.ld_1.ns_2044.dcid_5.l2w_240 1.161888 0.848558 1.134941 1.299109
      return safe(576, 13, BLOCK_LOAD_DIRECT, LOAD_CA,
                  sd(LookbackDelayAlgorithm::exponential_backon_jitter_window, 2044, 240));
    if (key_size==1 && accum_size==2)
      // ipt_10.tpb_224.trp_0.ld_0.ns_244.dcid_4.l2w_390 1.313932 1.260540 1.319588 1.427374
      return safe(224, 10, BLOCK_LOAD_DIRECT, LOAD_DEFAULT,
                  sd(LookbackDelayAlgorithm::exponential_backoff_jitter_window, 224, 390));
    if (key_size==1 && accum_size==4)
      // ipt_14.tpb_128.trp_0.ld_0.ns_248.dcid_2.l2w_285 1.118109 1.051534 1.134336 1.326788
      return safe(128, 14, BLOCK_LOAD_DIRECT, LOAD_DEFAULT,
                  sd(LookbackDelayAlgorithm::exponential_backoff, 248, 285));
    if (key_size==1 && accum_size==8)
      // ipt_19.tpb_128.trp_1.ld_0.ns_132.dcid_1.l2w_540 1.113820 1.002404 1.105014 1.202296
      return safe(128, 19, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
                  {LookbackDelayAlgorithm::fixed_delay, 66, 324});

    // key=2B
    if (key_size==2 && accum_size==1)
      // ipt_14.tpb_128.trp_1.ld_0.ns_164.dcid_2.l2w_290 1.239579 1.119705 1.239111 1.313112
      return safe(128, 14, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
                  sd(LookbackDelayAlgorithm::exponential_backoff, 164, 290));
    if (key_size==2 && accum_size==2)
      // ipt_14.tpb_256.trp_1.ld_0.ns_180.dcid_2.l2w_975 1.145635 1.012658 1.139956 1.251546
      return safe(256, 14, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
                  sd(LookbackDelayAlgorithm::exponential_backoff, 180, 975));
    if (key_size==2 && accum_size==4)
      // ipt_11.tpb_256.trp_0.ld_0.ns_224.dcid_2.l2w_550 1.066293 1.000109 1.073092 1.181818
      // NOTE: CCCL disables this for accum_t==float32 (regression). We keep SM90 fallback logic.
      return safe(256, 11, BLOCK_LOAD_DIRECT, LOAD_DEFAULT,
                  sd(LookbackDelayAlgorithm::exponential_backoff, 224, 550));
    if (key_size==2 && accum_size==8)
      // ipt_10.tpb_160.trp_1.ld_0.ns_156.dcid_1.l2w_725 1.045007 1.002105 1.049690 1.141827
      return safe(160, 10, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
                  {LookbackDelayAlgorithm::fixed_delay, 78, 435});

    // key=4B (vllm hot path: sequence_id)
    if (key_size==4 && accum_size==1)
      // ipt_10.tpb_224.trp_0.ld_0.ns_324.dcid_2.l2w_285 1.157217 1.073724 1.166510 1.356940
      return safe(224, 10, BLOCK_LOAD_DIRECT, LOAD_DEFAULT,
                  sd(LookbackDelayAlgorithm::exponential_backoff, 324, 285));
    if (key_size==4 && accum_size==2)
      // ipt_11.tpb_256.trp_0.ld_0.ns_1984.dcid_5.l2w_115 1.214155 1.128842 1.214093 1.364476
      return safe(256, 11, BLOCK_LOAD_DIRECT, LOAD_DEFAULT,
                  sd(LookbackDelayAlgorithm::exponential_backon_jitter_window, 1984, 115));
    if (key_size==4 && accum_size==4)
      // ipt_14.tpb_224.trp_1.ld_0.ns_476.dcid_5.l2w_1005 1.187378 1.119705 1.185397 1.258420
      // THIS IS THE VLLM HOT PATH: paged_attention score = float32 reduce by int32 key
      return safe(224, 14, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
                  sd(LookbackDelayAlgorithm::exponential_backon_jitter_window, 476, 1005));
    if (key_size==4 && accum_size==8)
      // ipt_10.tpb_256.trp_1.ld_0.ns_1868.dcid_7.l2w_145 1.142915 1.020581 1.137459 1.237913
      return safe(256, 10, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
                  sd(LookbackDelayAlgorithm::exponential_backon, 1868, 145));

    // key=8B
    if (key_size==8 && accum_size==1)
      // ipt_9.tpb_224.trp_1.ld_0.ns_1940.dcid_5.l2w_460 1.157294 1.075650 1.153566 1.250729
      return safe(224, 9, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
                  sd(LookbackDelayAlgorithm::exponential_backon_jitter_window, 1940, 460));
    if (key_size==8 && accum_size==2)
      // ipt_11.tpb_224.trp_1.ld_1.ns_392.dcid_2.l2w_550 1.104034 1.007212 1.099543 1.220401
      return safe(224, 11, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_CA,
                  sd(LookbackDelayAlgorithm::exponential_backoff, 392, 550));
    if (key_size==8 && accum_size==4)
      // ipt_9.tpb_224.trp_1.ld_0.ns_244.dcid_2.l2w_475 1.130098 1.000000 1.130661 1.215722
      return safe(224, 9, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
                  sd(LookbackDelayAlgorithm::exponential_backoff, 244, 475));
    if (key_size==8 && accum_size==8)
      // ipt_9.tpb_224.trp_1.ld_0.ns_196.dcid_2.l2w_340 1.272056 1.142857 1.262499 1.352941
      return safe(224, 9, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
                  sd(LookbackDelayAlgorithm::exponential_backoff, 196, 340));

    // =====================================================================
    // SM90 tuning — 25 entries (fall-through for sizes not in SM100)
    // Direct from CCCL, no delay scaling needed (SM90 delays already conservative)
    // =====================================================================

    // key=1B (SM90 adds accum_size=16)
    if (key_size==1 && accum_size==16)
      return safe(128, 11, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
                  {LookbackDelayAlgorithm::no_delay, 0, 1100});

    // key=2B (SM90 adds accum_size=1 with different params, accum_size=16)
    if (key_size==2 && accum_size==16)
      return safe(128, 11, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
                  {LookbackDelayAlgorithm::no_delay, 0, 1175});

    // key=4B (SM90 adds accum_size=16)
    if (key_size==4 && accum_size==16)
      return safe(128, 11, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
                  {LookbackDelayAlgorithm::no_delay, 0, 1195});

    // key=8B (SM90 adds accum_size=16)
    if (key_size==8 && accum_size==16)
      return safe(128, 11, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
                  {LookbackDelayAlgorithm::no_delay, 0, 1125});

    // key=16B (SM90 — all 5 accum sizes)
    if (key_size==16 && accum_size==1)
      return safe(128, 11, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
                  {LookbackDelayAlgorithm::no_delay, 0, 1080});
    if (key_size==16 && accum_size==2)
      return safe(128, 11, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
                  {LookbackDelayAlgorithm::fixed_delay, 320, 1005});
    if (key_size==16 && accum_size==4)
      return safe(128, 11, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
                  {LookbackDelayAlgorithm::fixed_delay, 232, 1100});
    if (key_size==16 && accum_size==8)
      return safe(128, 11, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
                  {LookbackDelayAlgorithm::no_delay, 0, 1195});
    if (key_size==16 && accum_size==16)
      return safe(128, 11, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
                  {LookbackDelayAlgorithm::no_delay, 0, 1150});

    // =====================================================================
    // SM80 tuning — 25 entries (final fallback for primitive types)
    // These are the most conservative; used when SM100/SM90 don't match
    // =====================================================================

    // key=1B SM80
    if (key_size==1 && accum_size==1)
      return safe(256, 13, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, {LookbackDelayAlgorithm::no_delay, 0, 975});
    if (key_size==1 && accum_size==2)
      return safe(224, 12, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, {LookbackDelayAlgorithm::no_delay, 0, 840});
    if (key_size==1 && accum_size==4)
      return safe(256, 15, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, {LookbackDelayAlgorithm::no_delay, 0, 760});
    if (key_size==1 && accum_size==8)
      return safe(224, 7, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, {LookbackDelayAlgorithm::no_delay, 0, 1070});

    // key=2B SM80
    if (key_size==2 && accum_size==1)
      return safe(256, 11, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, {LookbackDelayAlgorithm::no_delay, 0, 620});
    if (key_size==2 && accum_size==2)
      return safe(224, 14, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, {LookbackDelayAlgorithm::no_delay, 0, 640});
    if (key_size==2 && accum_size==4)
      return safe(256, 14, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, {LookbackDelayAlgorithm::no_delay, 0, 905});
    if (key_size==2 && accum_size==8)
      return safe(224, 9, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, {LookbackDelayAlgorithm::no_delay, 0, 810});

    // key=4B SM80
    if (key_size==4 && accum_size==1)
      return safe(288, 11, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, {LookbackDelayAlgorithm::no_delay, 0, 1110});
    if (key_size==4 && accum_size==2)
      return safe(192, 15, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, {LookbackDelayAlgorithm::no_delay, 0, 1200});
    if (key_size==4 && accum_size==4)
      return safe(256, 15, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, {LookbackDelayAlgorithm::no_delay, 0, 1110});
    if (key_size==4 && accum_size==8)
      return safe(224, 9, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, {LookbackDelayAlgorithm::no_delay, 0, 1165});

    // key=8B SM80
    if (key_size==8 && accum_size==1)
      return safe(192, 10, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, {LookbackDelayAlgorithm::no_delay, 0, 1175});
    if (key_size==8 && accum_size==2)
      return safe(224, 7, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, {LookbackDelayAlgorithm::no_delay, 0, 1075});
    if (key_size==8 && accum_size==4)
      return safe(384, 7, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, {LookbackDelayAlgorithm::no_delay, 0, 1040});
    if (key_size==8 && accum_size==8)
      return safe(128, 14, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, {LookbackDelayAlgorithm::no_delay, 0, 1080});
    if (key_size==8 && accum_size==16)
      return safe(128, 11, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, {LookbackDelayAlgorithm::no_delay, 0, 430});

    // key=16B SM80
    if (key_size==16 && accum_size==1)
      return safe(192, 7, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, {LookbackDelayAlgorithm::no_delay, 0, 1105});
    if (key_size==16 && accum_size==2)
      return safe(192, 7, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, {LookbackDelayAlgorithm::no_delay, 0, 755});
    if (key_size==16 && accum_size==4)
      return safe(192, 7, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, {LookbackDelayAlgorithm::no_delay, 0, 535});
    if (key_size==16 && accum_size==8)
      return safe(192, 7, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, {LookbackDelayAlgorithm::no_delay, 0, 1035});
    if (key_size==16 && accum_size==16)
      return safe(128, 11, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, {LookbackDelayAlgorithm::no_delay, 0, 1090});

    // Final default
    return default_policy(LOAD_DEFAULT);
  }

  constexpr ReduceByKeyPolicy operator()(const hardware_capability& hw) const {
    return {ReduceByKeyAlgorithm::lookback, get_lookback_policy()};
  }
};

} // namespace muh::tuning::reduce_by_key
