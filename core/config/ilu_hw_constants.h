/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Iluvatar BI-V100 hardware constants.
// ALL values verified by on-device probing — do NOT change without re-probing.
//
// Probing environment:
//   Machine:   cc-adc62d1c-476c-4ee4-9647-0c011c0b6d70-0
//   Cards:     4× Iluvatar BI-V100
//   Bus-Id:    4B:00.0, 4C:00.0, 4D:00.0, 4E:00.0
//   NUMA:      node 1, CPU affinity 16-31,80-95
//   Topology:  flat PIX (all pairs via single PCIe bridge, equal BW)
//   IX-ML:     3.2.3
//   Driver:    3.2.1
//   CUDA ver:  10.2 (CoreX compatibility layer)
//   SDK path:  /usr/local/corex/
//
// Probing commands used:
//   ixsmi -L                → card count, names, UUIDs
//   ixsmi topo -m           → PIX/PXB/PHB/SYS topology matrix
//   ixsmi -q -d MEMORY      → HBM capacity per card
//   ixsmi (default)         → SM clock, mem clock, TDP
//   debug_warpsize.py       → warp size via CUDA kernel (warpSize builtin)
//   torch.cuda.get_device_properties() → partial (warp_size N/A on CoreX)

#pragma once

#include <cstdint>

namespace xllm {
namespace ilu_hw {

// ---------- Core compute ----------

/// Warp size: 64 threads (NOT 32 like NVIDIA).
/// Verified via: CUDA kernel `warpSize` builtin → 64.
/// torch.cuda.get_device_properties(0).warp_size returns N/A on CoreX.
/// This affects all warp-level primitives: __shfl, __ballot, reductions, etc.
constexpr int32_t kWarpSize = 64;

/// SM clock: 1500 MHz (from ixsmi).
constexpr int32_t kSmClockMHz = 1500;

/// Memory clock: 1200 MHz (from ixsmi).
constexpr int32_t kMemClockMHz = 1200;

// ---------- Memory ----------

/// HBM per card: 32768 MiB (from ixsmi -q -d MEMORY).
constexpr int64_t kHbmPerCardMiB = 32768;
constexpr int64_t kHbmPerCardBytes = kHbmPerCardMiB * int64_t{1024} * 1024;

/// Baseline HBM usage (driver/runtime overhead): ~257 MiB observed idle.
constexpr int64_t kHbmBaselineUsageMiB = 257;

// ---------- Topology ----------

/// Number of cards in the verified configuration.
constexpr int32_t kVerifiedCardCount = 4;

/// Topology kind: all pairs are PIX (single PCIe bridge, equal bandwidth).
/// No NVLink, no HCCS mesh, no multi-switch hierarchy.
/// If deploying on a different BI-V100 server with PXB/PHB/SYS links,
/// use IluTopoKind::kGrouped instead.
constexpr bool kFlatTopology = true;

// ---------- TDP ----------

/// TDP per card: 250W (from ixsmi Pwr cap).
constexpr int32_t kTdpWatts = 250;

// ---------- Software ----------

/// CUDA compatibility version exposed by CoreX SDK.
constexpr int32_t kCudaMajor = 10;
constexpr int32_t kCudaMinor = 2;

}  // namespace ilu_hw
}  // namespace xllm
