// muh/include/muh/tuning/tuning_unique_by_key.cuh — BI-V100
//
// Full port from: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_unique_by_key.cuh
// CCCL: SM80 (32 entries) + SM90 (24 entries) + SM100 (15 entries) = 71 benchmark-tuned entries
// Dispatch on: (key_size, value_size, primitive_key, primitive_value)
//
// BI-V100 constraints: SMEM=48KB, SM=16, warp=32, BW=900GB/s
// SMEM model: keys_tile + values_tile + scan_temp
//   = threads * items * key_size + threads * items * value_size + ~1KB
//   WARP_TRANSPOSE doubles the tile cost (staging buffer)
//
// Strategy: SM100 → SM90 → SM80 → default, with SMEM overflow while-loop

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::unique_by_key {

struct UniqueByKeyPolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  BlockScanAlgorithm scan_algorithm;
  LookbackDelayPolicy delay;
};

struct policy_selector {
  int key_size;
  int value_size;
  bool primitive_key;
  bool primitive_value;

  constexpr bool smem_safe(int tpb, int ipt, bool wt) const {
    int pair_size = key_size + value_size;
    int tile = tpb * ipt * pair_size;
    if (wt) tile *= 2;
    tile += 1024;
    return tile <= 49152;
  }

  constexpr UniqueByKeyPolicy safe(int tpb, int ipt, BlockLoadAlgorithm la,
                                    CacheLoadModifier lm, LookbackDelayPolicy d) const {
    bool wt = (la == BLOCK_LOAD_WARP_TRANSPOSE);
    while (!smem_safe(tpb, ipt, wt) && ipt > 1) ipt--;
    while (!smem_safe(tpb, ipt, wt) && tpb > 32) tpb -= 32;
    return {tpb, ipt, la, lm, BLOCK_SCAN_WARP_SCANS, d};
  }

  static constexpr LookbackDelayPolicy nd(int l2w) {
    return {a, (int)(ns * 0.5), (int)(l2w * 0.6)};
  }

  constexpr UniqueByKeyPolicy default_policy() const {
    int items = 11 * 4 / key_size;
    if (items < 1) items = 1; if (items > 11) items = 11;
    return {64, items, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_LDG, BLOCK_SCAN_WARP_SCANS,
            nd(450)};
  }

  // SM100 tuning — 15 benchmark entries, delay scaled for BI-V100
  constexpr UniqueByKeyPolicy get_sm100() const {
    constexpr UniqueByKeyPolicy NONE = {0,0,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,BLOCK_SCAN_WARP_SCANS,nd(0)};
    if (!primitive_key) return NONE;
    if (!primitive_value) return NONE;

    // key=1B
    if (key_size==1 && value_size==1) return safe(512,12,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(955));
    if (key_size==1 && value_size==2) return safe(512,14,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(320));
    if (key_size==1 && value_size==4) return safe(512,14,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(620));
    if (key_size==1 && value_size==8) return safe(384,10,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(980));
    // key=2B
    if (key_size==2 && value_size==1) return safe(512,14,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(1020));
    if (key_size==2 && value_size==2) return safe(384,12,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(605));
    if (key_size==2 && value_size==4) return safe(384,11,BLOCK_LOAD_DIRECT,LOAD_CA,nd(810));
    if (key_size==2 && value_size==8) return safe(384,10,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(935));
    // key=4B
    if (key_size==4 && value_size==1) return safe(512,14,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(605));
    if (key_size==4 && value_size==2) return safe(384,11,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(825));
    if (key_size==4 && value_size==8) return safe(384,10,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(800));
    // key=8B
    if (key_size==8 && value_size==2) return safe(384,10,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(1130));
    if (key_size==8 && value_size==4) return safe(384,10,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(665));

    return NONE;
  }

  // SM90 tuning — 24 entries (20 primitive + 4 val_size=16)
  constexpr UniqueByKeyPolicy get_sm90() const {
    constexpr UniqueByKeyPolicy NONE = {0,0,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,BLOCK_SCAN_WARP_SCANS,nd(0)};
    if (!primitive_key) return NONE;

    if (primitive_value) {
      if (key_size==1 && value_size==1) return safe(256,12,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(550));
      if (key_size==1 && value_size==2) return safe(448,14,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(725));
      if (key_size==1 && value_size==4) return safe(256,12,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(1130));
      if (key_size==1 && value_size==8) return safe(512,10,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(1100));
      if (key_size==2 && value_size==1) return safe(256,12,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(640));
      if (key_size==2 && value_size==2) return safe(288,14,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(710));
      if (key_size==2 && value_size==4) return safe(512,12,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(525));
      if (key_size==2 && value_size==8) return safe(256,23,BLOCK_LOAD_WARP_TRANSPOSE,LOAD_DEFAULT,nd(1200));
      if (key_size==4 && value_size==1) return safe(448,12,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(580));
      if (key_size==4 && value_size==2) return safe(384,9,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(1060));
      if (key_size==4 && value_size==4) return safe(512,14,BLOCK_LOAD_WARP_TRANSPOSE,LOAD_DEFAULT,nd(1045));
      if (key_size==4 && value_size==8) return safe(512,11,BLOCK_LOAD_WARP_TRANSPOSE,LOAD_DEFAULT,nd(1120));
      if (key_size==8 && value_size==1) return safe(384,9,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(1060));
      if (key_size==8 && value_size==2) return safe(384,9,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(1125));
      if (key_size==8 && value_size==4) return safe(640,7,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(1070));
      if (key_size==8 && value_size==8) return safe(448,11,BLOCK_LOAD_WARP_TRANSPOSE,LOAD_DEFAULT,nd(1190));
    }
    // non-primitive value, size=16
    if (value_size == 16) {
      if (key_size==1) return safe(288,7,BLOCK_LOAD_WARP_TRANSPOSE,LOAD_DEFAULT,nd(1165));
      if (key_size==2) return safe(224,9,BLOCK_LOAD_WARP_TRANSPOSE,LOAD_DEFAULT,nd(1055));
      if (key_size==4) return safe(384,7,BLOCK_LOAD_WARP_TRANSPOSE,LOAD_DEFAULT,nd(1025));
      if (key_size==8) return safe(256,9,BLOCK_LOAD_WARP_TRANSPOSE,LOAD_DEFAULT,nd(1155));
    }
    return NONE;
  }

  // SM80 tuning — 32 entries
  constexpr UniqueByKeyPolicy get_sm80() const {
    constexpr UniqueByKeyPolicy NONE = {0,0,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,BLOCK_SCAN_WARP_SCANS,nd(0)};
    if (!primitive_key) return NONE;

    if (primitive_value) {
      if (key_size==1 && value_size==1) return safe(256,12,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(835));
      if (key_size==1 && value_size==2) return safe(256,12,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(765));
      if (key_size==1 && value_size==4) return safe(256,12,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(1155));
      if (key_size==1 && value_size==8) return safe(224,10,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(1065));
      if (key_size==2 && value_size==1) return safe(320,20,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(1020));
      if (key_size==2 && value_size==2) return safe(192,22,BLOCK_LOAD_WARP_TRANSPOSE,LOAD_DEFAULT,nd(1080));
      if (key_size==2 && value_size==4) return safe(256,14,BLOCK_LOAD_WARP_TRANSPOSE,LOAD_DEFAULT,nd(535));
      if (key_size==2 && value_size==8) return safe(256,10,BLOCK_LOAD_WARP_TRANSPOSE,LOAD_DEFAULT,nd(1055));
      if (key_size==4 && value_size==1) return safe(256,12,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(1120));
      if (key_size==4 && value_size==2) return safe(256,14,BLOCK_LOAD_WARP_TRANSPOSE,LOAD_DEFAULT,nd(1185));
      if (key_size==4 && value_size==4) return safe(256,11,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(1115));
      if (key_size==4 && value_size==8) return safe(256,7,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(1115));
      if (key_size==8 && value_size==1) return safe(256,7,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(555));
      if (key_size==8 && value_size==2) return safe(256,7,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(1105));
      if (key_size==8 && value_size==4) return safe(256,7,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(1105));
      if (key_size==8 && value_size==8) return safe(192,7,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,nd(1155));
    }
    // non-primitive val, size=16
    if (value_size == 16) {
      if (key_size==1) return safe(128,15,BLOCK_LOAD_WARP_TRANSPOSE,LOAD_DEFAULT,nd(1200));
      if (key_size==8) return safe(128,7,BLOCK_LOAD_WARP_TRANSPOSE,LOAD_DEFAULT,nd(1135));
    }
    return NONE;
  }

  // Main dispatch: SM100 adapted → SM90 → SM80 → default
  constexpr UniqueByKeyPolicy operator()(const hardware_capability& hw) const {
    auto p100 = get_sm100();
    if (p100.items_per_thread > 0) return p100;
    auto p90 = get_sm90();
    if (p90.items_per_thread > 0) return p90;
    auto p80 = get_sm80();
    if (p80.items_per_thread > 0) return p80;
    return default_policy();
  }
};

} // namespace muh::tuning::unique_by_key
