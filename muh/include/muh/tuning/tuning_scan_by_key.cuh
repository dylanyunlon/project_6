// muh/include/muh/tuning/tuning_scan_by_key.cuh — BI-V100
//
// Full port from CCCL (2008 lines). SM100: 16, SM90: ~30, SM80: ~30 entries.
// DELAY v2: all no_delay (16 SMs → gridDim.x < 500 → CCCL skips __nanosleep)
// L2WriteLatency preserved from CCCL (one-time constructor wait for L2 visibility)
#pragma once
#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::scan_by_key {

struct ScanByKeyLookbackPolicy {
  int threads_per_block; int items_per_thread;
  BlockLoadAlgorithm load_algorithm; CacheLoadModifier load_modifier;
  BlockStoreAlgorithm store_algorithm; BlockScanAlgorithm scan_algorithm;
  LookbackDelayPolicy delay;
};
enum class ScanByKeyAlgorithm { lookback };
struct ScanByKeyPolicy { ScanByKeyAlgorithm algorithm; ScanByKeyLookbackPolicy lookback; };

struct policy_selector {
  int key_size; int value_size;
  bool value_is_primitive; bool accum_is_primitive;

  constexpr bool smem_ok(int tpb, int ipt, bool wt) const {
    int pair = key_size + value_size;
    int tile = tpb * ipt * pair;
    if (wt) tile *= 2;
    return tile + 1024 <= 49152;
  }
  constexpr ScanByKeyLookbackPolicy safe(int tpb, int ipt, BlockLoadAlgorithm la,
      CacheLoadModifier lm, BlockStoreAlgorithm sa, LookbackDelayPolicy d) const {
    bool wt = (la == BLOCK_LOAD_WARP_TRANSPOSE);
    while (!smem_ok(tpb, ipt, wt) && ipt > 1) ipt--;
    while (!smem_ok(tpb, ipt, wt) && tpb > 32) tpb -= 32;
    return {tpb, ipt, la, lm, sa, BLOCK_SCAN_WARP_SCANS, d};
  }
  constexpr ScanByKeyLookbackPolicy wt(int tpb, int ipt, CacheLoadModifier lm, int l2w) const {
    return safe(tpb, ipt, BLOCK_LOAD_WARP_TRANSPOSE, lm, BLOCK_STORE_WARP_TRANSPOSE,
                {LookbackDelayAlgorithm::no_delay, 0, l2w});
  }
  constexpr ScanByKeyLookbackPolicy dr(int tpb, int ipt, CacheLoadModifier lm, int l2w) const {
    return safe(tpb, ipt, BLOCK_LOAD_DIRECT, lm, BLOCK_STORE_DIRECT,
                {LookbackDelayAlgorithm::no_delay, 0, l2w});
  }

  constexpr ScanByKeyLookbackPolicy get_lookback_policy() const {
    bool pv = value_is_primitive;

    // SM100 tile configs, all no_delay
    if (pv) {
      if (key_size==1 && value_size==1) return wt(288,13,LOAD_DEFAULT,745);
      if (key_size==1 && value_size==2) return wt(288,13,LOAD_DEFAULT,570);
      if (key_size==1 && value_size==4) return wt(224,19,LOAD_CA,910);
      if (key_size==1 && value_size==8) return wt(192,18,LOAD_CA,1035);
      if (key_size==2 && value_size==1) return wt(384,12,LOAD_DEFAULT,840);
      if (key_size==2 && value_size==2) return wt(160,14,LOAD_DEFAULT,170);
      if (key_size==2 && value_size==4) return wt(160,14,LOAD_DEFAULT,805);
      if (key_size==2 && value_size==8) return wt(224,13,LOAD_CA,735);
      if (key_size==4 && value_size==1) return wt(224,20,LOAD_CA,155);
      if (key_size==4 && value_size==2) return wt(288,13,LOAD_CA,925);
      if (key_size==4 && value_size==4) return wt(224,20,LOAD_CA,280); // VLLM HOT PATH
      if (key_size==4 && value_size==8) return wt(224,14,LOAD_CA,860);
      if (key_size==8 && value_size==1) return wt(160,12,LOAD_DEFAULT,850);
      if (key_size==8 && value_size==2) return wt(288,15,LOAD_DEFAULT,335);
      if (key_size==8 && value_size==4) return wt(160,22,LOAD_CA,505);
      if (key_size==8 && value_size==8) return wt(256,23,LOAD_DEFAULT,810);
    }
    // SM90
    if (pv) {
      if (key_size==1 && value_size==1) return dr(128,12,LOAD_DEFAULT,650);
      if (key_size==1 && value_size==2) return wt(256,16,LOAD_DEFAULT,995);
      if (key_size==1 && value_size==4) return wt(128,15,LOAD_DEFAULT,545);
      if (key_size==1 && value_size==8) return wt(224,10,LOAD_DEFAULT,1070);
      if (key_size==2 && value_size==1) return dr(128,12,LOAD_DEFAULT,785);
      if (key_size==2 && value_size==2) return wt(128,20,LOAD_DEFAULT,445);
      if (key_size==2 && value_size==4) return wt(128,22,LOAD_DEFAULT,865);
      if (key_size==2 && value_size==8) return wt(224,10,LOAD_DEFAULT,1170);
      if (key_size==4 && value_size==1) return dr(128,12,LOAD_DEFAULT,850);
      if (key_size==4 && value_size==2) return wt(256,14,LOAD_DEFAULT,965);
      if (key_size==4 && value_size==4) return wt(288,14,LOAD_DEFAULT,1005);
      if (key_size==4 && value_size==8) return wt(224,14,LOAD_DEFAULT,1195);
      if (key_size==8 && value_size==1) return dr(128,12,LOAD_DEFAULT,1010);
      if (key_size==8 && value_size==2) return wt(224,10,LOAD_DEFAULT,970);
      if (key_size==8 && value_size==4) return wt(192,10,LOAD_DEFAULT,1125);
      if (key_size==8 && value_size==8) return wt(224,11,LOAD_DEFAULT,930);
    }
    if (key_size==16 && accum_is_primitive) {
      if (value_size==1) return wt(192,7,LOAD_DEFAULT,975);
      if (value_size==2) return wt(224,10,LOAD_DEFAULT,1075);
      if (value_size==4) return wt(256,9,LOAD_DEFAULT,1120);
      if (value_size==8) return wt(192,9,LOAD_DEFAULT,1200);
    }
    if (value_size==16) {
      if (key_size==1) return wt(128,23,LOAD_DEFAULT,1105);
      if (key_size==2) return wt(128,23,LOAD_DEFAULT,1190);
      if (key_size==4) return wt(128,23,LOAD_DEFAULT,1030);
      if (key_size==8) return wt(192,15,LOAD_DEFAULT,1085);
      if (key_size==16) return wt(128,23,LOAD_DEFAULT,1050);
    }
    // SM80
    if (pv) {
      if (key_size==1 && value_size==1) return dr(128,12,LOAD_DEFAULT,795);
      if (key_size==1 && value_size==2) return wt(288,12,LOAD_DEFAULT,825);
      if (key_size==1 && value_size==4) return wt(256,15,LOAD_DEFAULT,640);
      if (key_size==1 && value_size==8) return wt(192,10,LOAD_DEFAULT,1040);
      if (key_size==2 && value_size==1) return dr(256,8,LOAD_DEFAULT,1070);
      if (key_size==2 && value_size==2) return wt(320,14,LOAD_DEFAULT,625);
      if (key_size==2 && value_size==4) return wt(256,15,LOAD_DEFAULT,1055);
      if (key_size==2 && value_size==8) return wt(160,17,LOAD_DEFAULT,695);
      if (key_size==4 && value_size==1) return dr(128,12,LOAD_DEFAULT,1130);
      if (key_size==4 && value_size==2) return wt(256,12,LOAD_DEFAULT,1130);
      if (key_size==4 && value_size==4) return wt(256,15,LOAD_DEFAULT,1140);
      if (key_size==4 && value_size==8) return wt(256,9,LOAD_DEFAULT,635);
      if (key_size==8 && value_size==1) return wt(128,11,LOAD_DEFAULT,1120);
      if (key_size==8 && value_size==2) return wt(256,10,LOAD_DEFAULT,1115);
      if (key_size==8 && value_size==4) return wt(224,13,LOAD_DEFAULT,1060);
      if (key_size==8 && value_size==8) return wt(224,10,LOAD_DEFAULT,1160);
    }
    if (key_size==16 && accum_is_primitive) {
      if (value_size==1) return wt(192,7,LOAD_DEFAULT,1120);
      if (value_size==2) return wt(192,7,LOAD_DEFAULT,780);
      if (value_size==4) return wt(256,7,LOAD_DEFAULT,1170);
      if (value_size==8) return wt(128,15,LOAD_DEFAULT,1030);
    }
    if (value_size==16) {
      if (key_size==1) return wt(128,19,LOAD_DEFAULT,1095);
      if (key_size==2) return wt(160,14,LOAD_DEFAULT,1105);
      if (key_size==4) return wt(128,17,LOAD_DEFAULT,1100);
      if (key_size==8) return wt(320,8,LOAD_DEFAULT,220);
      if (key_size==16) return wt(128,15,LOAD_DEFAULT,1160);
    }
    // Default
    int mx = key_size > value_size ? key_size : value_size;
    int ipt = (mx <= 8) ? 9 : (9*8/(key_size+value_size));
    if (ipt<1) ipt=1; if (ipt>9) ipt=9;
    return wt(256, ipt, LOAD_DEFAULT, 450);
  }

  constexpr ScanByKeyPolicy operator()(const hardware_capability& hw) const {
    return {ScanByKeyAlgorithm::lookback, get_lookback_policy()};
  }
};

} // namespace muh::tuning::scan_by_key
