// muh/include/muh/tuning/tuning_select_if.cuh — BI-V100
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_select_if.cuh
// CCCL SM100: 37 specializations across (flagged, keep_rejects, offset_size, input_size).
//   38 of them SMEM overflow on BI-V100 (max tile = 163840).
//
// Three dispatch dimensions preserved from CCCL (not collapsed):
//   1. may_alias → load_modifier: LOAD_CA (alias-safe) vs LOAD_LDG (no alias, faster)
//      CCCL: may_alias path uses LOAD_CA or LOAD_DEFAULT; no-alias uses LOAD_LDG
//      Impact: LOAD_LDG is ~5-10% faster for no-alias (the common case in vllm)
//   2. has_flags → items_per_thread: flagged path needs extra SMEM for flags array
//      CCCL: flagged=yes structs typically have 2-4 fewer items than flagged=no
//   3. delay → varies by type size, not fixed
//      CCCL SM100 delays range from backoff(0, 915) to backon_jitter_window(1508, 585)
//      BI-V100 heuristic: scale ns*0.5, l2w*0.6 (same as scan)
//
// vllm relevance: token filtering (e.g. select tokens above threshold in speculative decoding)
// SMEM risk: CRITICAL. select_if SMEM = input_tile + output_tile + scan_temp.
//   Conservative: 2 * threads * items * elem_size + scan overhead

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::select_if {

struct SelectLookbackPolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  BlockScanAlgorithm scan_algorithm;
  LookbackDelayPolicy delay;
};

enum class SelectAlgorithm { lookback };

struct SelectPolicy {
  SelectAlgorithm algorithm;
  SelectLookbackPolicy lookback;
};

struct policy_selector {
  int input_size;
  int flag_size;     // 0 if no flags (predicate-based select)
  int output_size;
  int offset_size;
  bool may_alias;    // SelectImpl::SelectPotentiallyInPlace

  constexpr SelectPolicy operator()(const hardware_capability& hw) const {
    bool has_flags = flag_size > 0;
    int elem_size = input_size > output_size ? input_size : output_size;

    // --- Dimension 1: may_alias → load config ---
    // CCCL: may_alias uses LOAD_CA (cache-all, alias-safe)
    //        no-alias uses LOAD_LDG (read-only texture cache, ~5-10% faster)
    //        no-alias + small type also allows BLOCK_LOAD_DIRECT (no smem shuffle)
    BlockLoadAlgorithm load_algo;
    CacheLoadModifier load_mod;

    if (may_alias) {
      load_algo = BLOCK_LOAD_WARP_TRANSPOSE;
      load_mod = LOAD_CA;
    } else {
      // No alias: can use faster load paths
      if (elem_size <= 4) {
        load_algo = BLOCK_LOAD_DIRECT;  // matches CCCL SM100 no-alias small-type
        load_mod = LOAD_LDG;
      } else {
        load_algo = BLOCK_LOAD_WARP_TRANSPOSE;
        load_mod = LOAD_LDG;
      }
    }

    // --- Dimension 2: has_flags → items adjustment ---
    // CCCL: flagged=yes structs have fewer items (flag array takes SMEM)
    // flag_tile = threads * items * sizeof(bool) = threads * items
    int threads, items;

    if (has_flags) {
      // Flagged path: fewer items due to flag SMEM
      if (elem_size <= 2)      { threads = 384; items = 18; }
      else if (elem_size <= 4) { threads = 320; items = 14; }
      else if (elem_size <= 8) { threads = 256; items = 10; }
      else                     { threads = 192; items = 7;  }
    } else {
      // No flags: more items available
      if (elem_size <= 2)      { threads = 384; items = 22; }
      else if (elem_size <= 4) { threads = 384; items = 18; }
      else if (elem_size <= 8) { threads = 256; items = 14; }
      else                     { threads = 192; items = 9;  }
    }

    // SMEM check: input_tile + output_scatter + scan_temp
    // Conservative: tile = threads * items * elem_size (input)
    //             + threads * items * elem_size (output scatter buffer)
    //             + threads * flag_size (if flagged)
    int smem_input = threads * items * elem_size;
    int smem_output = threads * items * elem_size;
    int smem_flags = has_flags ? threads * items : 0;
    int smem_total = smem_input + smem_output + smem_flags;

    while (smem_total > hw.max_shared_memory_per_block && items > 1) {
      items--;
      smem_input = threads * items * elem_size;
      smem_output = threads * items * elem_size;
      smem_flags = has_flags ? threads * items : 0;
      smem_total = smem_input + smem_output + smem_flags;
    }

    // --- Dimension 3: delay by type size ---
    // CCCL SM100 delay patterns (scaled for BI-V100: ns*0.5, l2w*0.6):
    // elem≤2: backon(~400, ~400) → bi100: backon(200, 240)
    // elem=4: backon_jitter(~800, ~500) → bi100: backon_jitter(400, 300)
    // elem=8: backoff(~300, ~600) → bi100: backoff(150, 360)
    // elem>8: fixed(350, 450) → bi100: fixed(350, 450) (no SM100 data)
    LookbackDelayPolicy delay;
    if (elem_size <= 2) {
      delay = {LookbackDelayAlgorithm::exponential_backon, 200, 240};
    } else if (elem_size <= 4) {
      delay = {LookbackDelayAlgorithm::exponential_backon_jitter, 400, 300};
    } else if (elem_size <= 8) {
      delay = {LookbackDelayAlgorithm::exponential_backoff, 150, 360};
    } else {
      delay = {LookbackDelayAlgorithm::fixed_delay, 350, 450};
    }

    return {SelectAlgorithm::lookback,
            {threads, items, load_algo, load_mod, BLOCK_SCAN_WARP_SCANS, delay}};
  }
};

} // namespace muh::tuning::select_if
