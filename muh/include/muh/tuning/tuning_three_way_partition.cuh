// muh/include/muh/tuning/tuning_three_way_partition.cuh — BI-V100
// Full port from CCCL: SM80 (4) + SM90 (10) + SM100 (5) = 19 benchmark-tuned entries
// Dispatch on: (offset_size, input_size)
// SMEM model: 3 * threads * items * input_size (three output partitions) + scan_temp
#pragma once
#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::three_way_partition {

struct ThreeWayPartitionLookbackPolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  BlockScanAlgorithm scan_algorithm;
  LookbackDelayPolicy delay;
};
enum class ThreeWayPartitionAlgorithm { lookback };
struct ThreeWayPartitionPolicy {
  ThreeWayPartitionAlgorithm algorithm;
  ThreeWayPartitionLookbackPolicy lookback;
};

struct policy_selector {
  int input_size;
  int offset_size;

  constexpr bool smem_ok(int tpb, int ipt, bool wt) const {
    int tile = 3 * tpb * ipt * input_size; // three partitions
    if (wt) tile += tpb * ipt * input_size; // staging
    return tile + 1024 <= 49152;
  }

  constexpr ThreeWayPartitionLookbackPolicy safe(int tpb, int ipt, BlockLoadAlgorithm la,
      CacheLoadModifier lm, LookbackDelayPolicy d) const {
    bool wt = (la == BLOCK_LOAD_WARP_TRANSPOSE);
    while (!smem_ok(tpb, ipt, wt) && ipt > 1) ipt--;
    return {tpb, ipt, la, lm, BLOCK_SCAN_WARP_SCANS, d};
  }

  static constexpr LookbackDelayPolicy sd(LookbackDelayAlgorithm a, int ns, int l2w) {
    return {a, (int)(ns*0.5), (int)(l2w*0.6)};
  }

  constexpr ThreeWayPartitionLookbackPolicy default_policy() const {
    int items = 9 * 4 / input_size;
    if (items < 1) items = 1; if (items > 9) items = 9;
    return {256, items, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS,
            {LookbackDelayAlgorithm::fixed_delay, 350, 450}};
  }

  constexpr ThreeWayPartitionLookbackPolicy dispatch() const {
    // === SM100 tunings (delay scaled for BI-V100) ===
    if (offset_size==4 && input_size==4)
      return safe(512,11,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,sd(LookbackDelayAlgorithm::exponential_backon_jitter,72,840));
    if (offset_size==4 && input_size==8)
      return safe(256,10,BLOCK_LOAD_WARP_TRANSPOSE,LOAD_DEFAULT,sd(LookbackDelayAlgorithm::exponential_backon_jitter,8,845));
    if (offset_size==8 && input_size==2)
      return safe(768,20,BLOCK_LOAD_WARP_TRANSPOSE,LOAD_DEFAULT,sd(LookbackDelayAlgorithm::exponential_backon_jitter_window,544,500));
    if (offset_size==8 && input_size==4)
      return safe(768,15,BLOCK_LOAD_WARP_TRANSPOSE,LOAD_DEFAULT,sd(LookbackDelayAlgorithm::exponential_backon_jitter,144,280));
    if (offset_size==8 && input_size==8)
      return safe(320,14,BLOCK_LOAD_WARP_TRANSPOSE,LOAD_DEFAULT,sd(LookbackDelayAlgorithm::exponential_backon,872,620));

    // === SM90 tunings ===
    if (offset_size==4 && input_size==1)
      return safe(256,12,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,{LookbackDelayAlgorithm::no_delay,0,445});
    if (offset_size==4 && input_size==2)
      return safe(256,12,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,{LookbackDelayAlgorithm::fixed_delay,104,512});
    // SM90 offset=4 input=4 already covered by SM100 above
    // SM90 offset=4 input=8 already covered
    if (offset_size==4 && input_size==16)
      return safe(128,7,BLOCK_LOAD_WARP_TRANSPOSE,LOAD_DEFAULT,{LookbackDelayAlgorithm::no_delay,0,1040});
    if (offset_size==8 && input_size==1)
      return safe(256,24,BLOCK_LOAD_DIRECT,LOAD_DEFAULT,{LookbackDelayAlgorithm::fixed_delay,4,285});
    // offset=8 input=2,4,8 covered by SM100
    if (offset_size==8 && input_size==16)
      return safe(256,11,BLOCK_LOAD_WARP_TRANSPOSE,LOAD_DEFAULT,{LookbackDelayAlgorithm::no_delay,0,1050});

    // === SM80 tunings ===
    if (offset_size==4 && input_size==2)
      return safe(256,12,BLOCK_LOAD_WARP_TRANSPOSE,LOAD_DEFAULT,{LookbackDelayAlgorithm::no_delay,0,910});
    if (offset_size==4 && input_size==4)
      return safe(256,11,BLOCK_LOAD_WARP_TRANSPOSE,LOAD_DEFAULT,{LookbackDelayAlgorithm::no_delay,0,1120});
    if (offset_size==4 && input_size==8)
      return safe(224,11,BLOCK_LOAD_WARP_TRANSPOSE,LOAD_DEFAULT,{LookbackDelayAlgorithm::fixed_delay,264,1080});
    if (offset_size==4 && input_size==16)
      return safe(128,10,BLOCK_LOAD_WARP_TRANSPOSE,LOAD_DEFAULT,{LookbackDelayAlgorithm::fixed_delay,672,1120});

    return default_policy();
  }

  constexpr ThreeWayPartitionPolicy operator()(const hardware_capability& hw) const {
    return {ThreeWayPartitionAlgorithm::lookback, dispatch()};
  }
};

} // namespace muh::tuning::three_way_partition
