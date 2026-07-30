// muh/include/muh/hardware.cuh — Iluvatar BI-V100 hardware descriptor
//
// This header replaces cuda::compute_capability as the dispatch key.
// CCCL's policy_selector uses operator()(cuda::compute_capability cc)
// to select tuning params. muh's policy_selector uses
// operator()(muh::hardware_capability hw) instead.

#pragma once

namespace muh {

/// Hardware capability descriptor for non-NVIDIA GPUs.
/// Replaces cuda::compute_capability {major, minor} with a richer
/// description that captures what actually matters for kernel tuning.
struct hardware_capability {
  int warp_size;                    // threads per warp (NVIDIA=32, BI-V100=TBD)
  int max_threads_per_block;        // max CTA size (typically 1024)
  int max_shared_memory_per_block;  // bytes of shared memory per block
  int max_registers_per_thread;     // max registers per thread
  int l2_cache_size_bytes;          // L2 cache size in bytes
  int memory_bandwidth_gbps;        // HBM bandwidth in GB/s
  int sm_count;                     // number of SMs / compute units

  // For dispatch: identifies which tuning table to use
  enum class vendor_t { nvidia, iluvatar, unknown };
  vendor_t vendor;
  int arch_version;                 // e.g. 100 for BI-V100

  // Convenience constructors
  constexpr static hardware_capability bi_v100() {
    return {
      .warp_size = 32,                     // TBD: confirm on actual hardware
      .max_threads_per_block = 1024,
      .max_shared_memory_per_block = 49152, // 48 KiB, TBD
      .max_registers_per_thread = 255,
      .l2_cache_size_bytes = 6 * 1024 * 1024, // 6 MiB, TBD
      .memory_bandwidth_gbps = 900,         // TBD
      .sm_count = 50,                       // 50c in the spec
      .vendor = vendor_t::iluvatar,
      .arch_version = 100,
    };
  }

  // Comparison for dispatch: exact match on vendor + arch
  constexpr bool operator==(const hardware_capability& o) const {
    return vendor == o.vendor && arch_version == o.arch_version;
  }
  constexpr bool operator!=(const hardware_capability& o) const {
    return !(*this == o);
  }

  // Check if this hardware is "at least" a given capability
  // For same vendor, compares arch_version
  constexpr bool at_least(vendor_t v, int min_arch) const {
    return vendor == v && arch_version >= min_arch;
  }
};

/// Global default target — set to BI-V100 for competition
inline constexpr auto target_hw = hardware_capability::bi_v100();

} // namespace muh
