// muh/include/muh/tuning/tuning_scan_by_key.cuh — BI-V100
//
// Full port from: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_scan_by_key.cuh (2008 lines)
// SM100: 16 benchmark-tuned entries (key 1-8B × value 1-8B, with CCCL annotations)
// SM90:  ~30 entries (key 1-16B × value 1-16B, incl int128)
// SM80:  ~30 entries (key 1-16B × value 1-16B, incl int128)
//
// scan_by_key adds store_algorithm vs reduce_by_key (7-field policy)
// SMEM model: tpb * ipt * (key_size + value_size) * 2 (load+store staging for WARP_TRANSPOSE)
// BI-V100: SMEM=48KB, SM=16, warp=32, BW=900GB/s
// SM100 delay scaling: ns*0.5, l2w*0.6
//
// vllm hot path: per-sequence prefix-sum in paged_attention
//   key = sequence_id (int32), value = attention_score (float32)
//   → key_size=4, value_size=4

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::scan_by_key {

struct ScanByKeyLookbackPolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  BlockStoreAlgorithm store_algorithm;
  BlockScanAlgorithm scan_algorithm;
  LookbackDelayPolicy delay;
};

enum class ScanByKeyAlgorithm { lookback };

struct ScanByKeyPolicy {
  ScanByKeyAlgorithm algorithm;
  ScanByKeyLookbackPolicy lookback;
};

struct policy_selector {
  int key_size;
  int value_size;
  bool value_is_primitive;
  bool accum_is_primitive;

  constexpr bool smem_ok(int tpb, int ipt, bool wt) const {
    int pair = key_size + value_size;
    int tile = tpb * ipt * pair;
    if (wt) tile *= 2; // load + store staging
    tile += 1024;
    return tile <= 49152;
  }

  constexpr ScanByKeyLookbackPolicy safe(
      int tpb, int ipt, BlockLoadAlgorithm la, CacheLoadModifier lm,
      BlockStoreAlgorithm sa, LookbackDelayPolicy d) const {
    bool wt = (la == BLOCK_LOAD_WARP_TRANSPOSE);
    while (!smem_ok(tpb, ipt, wt) && ipt > 1) ipt--;
    while (!smem_ok(tpb, ipt, wt) && tpb > 32) tpb -= 32;
    return {tpb, ipt, la, lm, sa, BLOCK_SCAN_WARP_SCANS, d};
  }

  // Shorthand: WARP_TRANSPOSE load+store pair
  constexpr ScanByKeyLookbackPolicy wt(int tpb, int ipt, CacheLoadModifier lm, LookbackDelayPolicy d) const {
    return safe(tpb, ipt, BLOCK_LOAD_WARP_TRANSPOSE, lm, BLOCK_STORE_WARP_TRANSPOSE, d);
  }
  // Shorthand: DIRECT load+store pair
  constexpr ScanByKeyLookbackPolicy dr(int tpb, int ipt, CacheLoadModifier lm, LookbackDelayPolicy d) const {
    return safe(tpb, ipt, BLOCK_LOAD_DIRECT, lm, BLOCK_STORE_DIRECT, d);
  }

  static constexpr LookbackDelayPolicy sd(LookbackDelayAlgorithm a, int ns, int l2w) {
    return {a, (int)(ns*0.5), (int)(l2w*0.6)};
  }
  static constexpr LookbackDelayPolicy nd(int l2w) {
    return {LookbackDelayAlgorithm::no_delay, 0, l2w};
  }
  static constexpr LookbackDelayPolicy fd(int ns, int l2w) {
    return {LookbackDelayAlgorithm::fixed_delay, ns, l2w};
  }

  constexpr ScanByKeyLookbackPolicy get_lookback_policy() const {
    bool pv = value_is_primitive;

    // =====================================================================
    // SM100 — 16 entries, delay scaled for BI-V100
    // All use WARP_TRANSPOSE load+store
    // =====================================================================
    if (pv) {
      // key=1B
      if (key_size==1 && value_size==1)
        // ipt_13.tpb_288.ns_420.dcid_0.l2w_745.trp_1.ld_0
        return wt(288, 13, LOAD_DEFAULT, nd(745));
      if (key_size==1 && value_size==2)
        // ipt_13.tpb_288.ns_388.dcid_1.l2w_570.trp_1.ld_0
        return wt(288, 13, LOAD_DEFAULT, fd(194, 342));
      if (key_size==1 && value_size==4)
        // ipt_19.tpb_224.ns_1028.dcid_5.l2w_910.trp_1.ld_1
        return wt(224, 19, LOAD_CA, sd(LookbackDelayAlgorithm::exponential_backon_jitter_window, 1028, 910));
      if (key_size==1 && value_size==8)
        // ipt_18.tpb_192.ns_432.dcid_1.l2w_1035.trp_1.ld_1
        return wt(192, 18, LOAD_CA, fd(216, 621));

      // key=2B
      if (key_size==2 && value_size==1)
        // ipt_12.tpb_384.ns_1900.dcid_0.l2w_840.trp_1.ld_0
        return wt(384, 12, LOAD_DEFAULT, nd(1900));
      if (key_size==2 && value_size==2)
        // ipt_14.tpb_160.ns_1736.dcid_7.l2w_170.trp_1.ld_0
        return wt(160, 14, LOAD_DEFAULT, sd(LookbackDelayAlgorithm::exponential_backon, 1736, 170));
      if (key_size==2 && value_size==4)
        // ipt_14.tpb_160.ns_336.dcid_1.l2w_805.trp_1.ld_0
        return wt(160, 14, LOAD_DEFAULT, fd(168, 483));
      if (key_size==2 && value_size==8)
        // ipt_13.tpb_224.trp_1.ld_2 (LOAD_CA)
        return wt(224, 13, LOAD_CA, sd(LookbackDelayAlgorithm::exponential_backoff, 348, 735));

      // key=4B (vllm hot path)
      if (key_size==4 && value_size==1)
        // ipt_20.tpb_224.ns_1436.dcid_7.l2w_155.trp_1.ld_1
        return wt(224, 20, LOAD_CA, sd(LookbackDelayAlgorithm::exponential_backon, 1436, 155));
      if (key_size==4 && value_size==2)
        // ipt_13.tpb_288.ns_620.dcid_7.l2w_925.trp_1.ld_2
        return wt(288, 13, LOAD_CA, sd(LookbackDelayAlgorithm::exponential_backon, 620, 925));
      if (key_size==4 && value_size==4)
        // ipt_20.tpb_224.ns_1856.dcid_5.l2w_280.trp_1.ld_1
        // THIS IS THE VLLM HOT PATH
        return wt(224, 20, LOAD_CA, sd(LookbackDelayAlgorithm::exponential_backon_jitter_window, 1856, 280));
      if (key_size==4 && value_size==8)
        // ipt_14.tpb_224.ns_464.dcid_2.l2w_680.trp_1.ld_1
        return wt(224, 14, LOAD_CA, sd(LookbackDelayAlgorithm::exponential_backoff, 464, 860));

      // key=8B
      if (key_size==8 && value_size==1)
        // ipt_12.tpb_160.ns_532.dcid_0.l2w_850.trp_1.ld_0
        return wt(160, 12, LOAD_DEFAULT, nd(532));
      if (key_size==8 && value_size==2)
        // ipt_15.tpb_288.ns_988.dcid_7.l2w_335.trp_1.ld_0
        return wt(288, 15, LOAD_DEFAULT, sd(LookbackDelayAlgorithm::exponential_backon, 988, 335));
      if (key_size==8 && value_size==4)
        // ipt_22.tpb_160.ns_1032.dcid_5.l2w_505.trp_1.ld_2
        return wt(160, 22, LOAD_CA, sd(LookbackDelayAlgorithm::exponential_backon_jitter_window, 1032, 505));
      if (key_size==8 && value_size==8)
        // ipt_23.tpb_256.ns_1232.dcid_0.l2w_810.trp_1.ld_0
        return wt(256, 23, LOAD_DEFAULT, nd(1232));
    }

    // =====================================================================
    // SM90 — key 1-8B × value 1-8B (primitive), no delay scaling
    // =====================================================================
    if (pv) {
      // key=1B SM90
      if (key_size==1 && value_size==1) return dr(128, 12, LOAD_DEFAULT, nd(650));
      if (key_size==1 && value_size==2) return wt(256, 16, LOAD_DEFAULT, fd(124, 995));
      if (key_size==1 && value_size==4) return wt(128, 15, LOAD_DEFAULT, fd(488, 545));
      if (key_size==1 && value_size==8) return wt(224, 10, LOAD_DEFAULT, fd(488, 1070));

      // key=2B SM90
      if (key_size==2 && value_size==1) return dr(128, 12, LOAD_DEFAULT, fd(136, 785));
      if (key_size==2 && value_size==2) return wt(128, 20, LOAD_DEFAULT, nd(445));
      if (key_size==2 && value_size==4) return wt(128, 22, LOAD_DEFAULT, fd(312, 865));
      if (key_size==2 && value_size==8) return wt(224, 10, LOAD_DEFAULT, fd(352, 1170));

      // key=4B SM90
      if (key_size==4 && value_size==1) return dr(128, 12, LOAD_DEFAULT, nd(850));
      if (key_size==4 && value_size==2) return wt(256, 14, LOAD_DEFAULT, fd(128, 965));
      if (key_size==4 && value_size==4) return wt(288, 14, LOAD_DEFAULT, fd(700, 1005));
      if (key_size==4 && value_size==8) return wt(224, 14, LOAD_DEFAULT, fd(556, 1195));

      // key=8B SM90
      if (key_size==8 && value_size==1) return dr(128, 12, LOAD_DEFAULT, fd(504, 1010));
      if (key_size==8 && value_size==2) return wt(224, 10, LOAD_DEFAULT, fd(420, 970));
      if (key_size==8 && value_size==4) return wt(192, 10, LOAD_DEFAULT, fd(500, 1125));
      if (key_size==8 && value_size==8) return wt(224, 11, LOAD_DEFAULT, fd(600, 930));
    }

    // SM90 key=16B (accum_is_primitive)
    if (key_size==16 && accum_is_primitive) {
      if (value_size==1) return wt(192, 7, LOAD_DEFAULT, fd(500, 975));
      if (value_size==2) return wt(224, 10, LOAD_DEFAULT, fd(164, 1075));
      if (value_size==4) return wt(256, 9, LOAD_DEFAULT, fd(268, 1120));
      if (value_size==8) return wt(192, 9, LOAD_DEFAULT, fd(320, 1200));
    }

    // SM90 int128 values (key 1-8B, value=16B)
    if (value_size==16) {
      if (key_size==1) return wt(128, 23, LOAD_DEFAULT, fd(936, 1105));
      if (key_size==2) return wt(128, 23, LOAD_DEFAULT, fd(504, 1190));
      if (key_size==4) return wt(128, 23, LOAD_DEFAULT, fd(512, 1030));
      if (key_size==8) return wt(192, 15, LOAD_DEFAULT, fd(364, 1085));
      if (key_size==16) return wt(128, 23, LOAD_DEFAULT, fd(364, 1050));
    }

    // =====================================================================
    // SM80 — key 1-8B × value 1-8B (primitive)
    // =====================================================================
    if (pv) {
      // key=1B SM80
      if (key_size==1 && value_size==1) return dr(128, 12, LOAD_DEFAULT, nd(795));
      if (key_size==1 && value_size==2) return wt(288, 12, LOAD_DEFAULT, nd(825));
      if (key_size==1 && value_size==4) return wt(256, 15, LOAD_DEFAULT, nd(640));
      if (key_size==1 && value_size==8) return wt(192, 10, LOAD_DEFAULT, fd(124, 1040));

      // key=2B SM80
      if (key_size==2 && value_size==1) return dr(256, 8, LOAD_DEFAULT, nd(1070));
      if (key_size==2 && value_size==2) return wt(320, 14, LOAD_DEFAULT, nd(625));
      if (key_size==2 && value_size==4) return wt(256, 15, LOAD_DEFAULT, nd(1055));
      if (key_size==2 && value_size==8) return wt(160, 17, LOAD_DEFAULT, fd(160, 695));

      // key=4B SM80
      if (key_size==4 && value_size==1) return dr(128, 12, LOAD_DEFAULT, nd(1130));
      if (key_size==4 && value_size==2) return wt(256, 12, LOAD_DEFAULT, nd(1130));
      if (key_size==4 && value_size==4) return wt(256, 15, LOAD_DEFAULT, nd(1140));
      if (key_size==4 && value_size==8) return wt(256, 9, LOAD_DEFAULT, fd(888, 635));

      // key=8B SM80
      if (key_size==8 && value_size==1) return wt(128, 11, LOAD_DEFAULT, nd(1120));
      if (key_size==8 && value_size==2) return wt(256, 10, LOAD_DEFAULT, nd(1115));
      if (key_size==8 && value_size==4) return wt(224, 13, LOAD_DEFAULT, fd(24, 1060));
      if (key_size==8 && value_size==8) return wt(224, 10, LOAD_DEFAULT, nd(1160));
    }

    // SM80 key=16B (accum_is_primitive)
    if (key_size==16 && accum_is_primitive) {
      if (value_size==1) return wt(192, 7, LOAD_DEFAULT, fd(144, 1120));
      if (value_size==2) return wt(192, 7, LOAD_DEFAULT, fd(364, 780));
      if (value_size==4) return wt(256, 7, LOAD_DEFAULT, nd(1170));
      if (value_size==8) return wt(128, 15, LOAD_DEFAULT, nd(1030));
    }

    // SM80 int128 values
    if (value_size==16) {
      if (key_size==1) return wt(128, 19, LOAD_DEFAULT, nd(1095));
      if (key_size==2) return wt(160, 14, LOAD_DEFAULT, nd(1105));
      if (key_size==4) return wt(128, 17, LOAD_DEFAULT, nd(1100));
      if (key_size==8) return wt(320, 8, LOAD_DEFAULT, nd(220));
      if (key_size==16) return wt(128, 15, LOAD_DEFAULT, nd(1160));
    }

    // =====================================================================
    // Default fallback
    // =====================================================================
    int mx = key_size > value_size ? key_size : value_size;
    int combined = key_size + value_size;
    int ipt = (mx <= 8) ? 9 : (9 * 8 / combined);
    if (ipt < 1) ipt = 1; if (ipt > 9) ipt = 9;
    return wt(256, ipt, LOAD_DEFAULT, fd(350, 450));
  }

  constexpr ScanByKeyPolicy operator()(const hardware_capability& hw) const {
    return {ScanByKeyAlgorithm::lookback, get_lookback_policy()};
  }
};

} // namespace muh::tuning::scan_by_key
