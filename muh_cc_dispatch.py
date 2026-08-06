"""
muh_cc_dispatch.py — Unified kernel policy dispatch for BI-V100
================================================================

Python port of CCCL's cc_dispatch.cuh architecture.

CCCL dispatch pattern (cc_dispatch.cuh):
  dispatch_compute_cap(policy_selector, device_cc, functor)
    → policy_getter<PolicySelector, CC>{}()
    → concrete policy struct (ReducePolicy, ScanPolicy, etc.)

Our equivalent:
  dispatch_kernel_config(hardware, kernel_name, **kwargs)
    → policy_for_kernel(kernel_name, hardware, dtype, ...)
    → concrete config dict (threads, items, block_sizes, etc.)

Key insight from cc_dispatch.cuh line 62 (lowest_cc_resolver):
  CCCL collapses architectures with identical policies — if SM80 and SM86
  produce the same ReducePolicy, only one kernel instantiation is generated.
  Our equivalent: pre-compute all configs at import time (see bottom of file)
  so dispatch is a dict lookup, not a function call.

Key insight from dispatch_reduce.cuh line 490:
  dispatch_compute_cap is called ONCE per DeviceReduce invocation.
  The policy is then threaded through InvokeSingleTile / InvokePasses.
  Our equivalent: dispatch_kernel_config returns a frozen config dict
  that's threaded through the entire kernel call chain.

CCCL source files that informed this design:
  cub/detail/cc_dispatch.cuh           — dispatch mechanism
  cub/device/dispatch/dispatch_reduce.cuh   — reduce two-path dispatch
  cub/device/dispatch/dispatch_transform.cuh — transform spread_out_items
  cub/device/dispatch/dispatch_topk.cuh      — topk radix select
  cub/device/dispatch/dispatch_common.cuh    — shared enums
  cub/grid/grid_even_share.cuh               — work distribution
  cub/agent/agent_reduce.cuh                 — tile consumption patterns
  thrust/examples/summary_statistics.cu      — compound reduce pattern
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import math


# ══════════════════════════════════════════════════════════════
# Hardware descriptor (mirrors muh/include/muh/hardware.cuh)
# ══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class HardwareCapability:
    """Mirrors muh::hardware_capability from hardware.cuh."""
    vendor: str = "iluvatar"
    arch_version: int = 100
    warp_size: int = 32
    max_threads_per_block: int = 1024
    max_shared_memory_per_block: int = 49152  # 48KB confirmed via ixsmi
    max_registers_per_thread: int = 255
    l2_cache_size_bytes: int = 6 * 1024 * 1024  # 6MB
    memory_bandwidth_gbps: int = 900
    sm_count: int = 16  # CONFIRMED: 16 SMs, NOT 50

    @property
    def bandwidth_per_sm_gbps(self) -> float:
        return self.memory_bandwidth_gbps / self.sm_count

    @property
    def bytes_in_flight(self) -> int:
        """Optimal bytes in flight per SM.
        
        From CCCL tuning_transform.cuh cc_to_min_bytes_in_flight:
          V100=12KB, A100=16KB, H100=48KB, B200=64KB
        BI-V100 per-SM BW = 900/16 = 56 GB/s ≈ B200 level → 64KB
        Confirmed by bench_bi100.py: bif=8 (64KB) wins at all sizes.
        """
        return 64 * 1024

    def at_least(self, vendor: str, min_arch: int) -> bool:
        """Mirrors hardware_capability::at_least()."""
        return self.vendor == vendor and self.arch_version >= min_arch


BI_V100 = HardwareCapability()


# ══════════════════════════════════════════════════════════════
# Kernel configuration structs
# (mirrors CCCL's ReducePolicy, ScanPolicy, TopkPolicy, etc.)
# ══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AttentionConfig:
    """Config for paged attention V1/V2 + prefix prefill.
    
    Dispatch axes (from muh_kernel_map.py VLLM_KERNEL_MAP):
      paged_attention_v1: reduce (score reduction per head)
      paged_attention_v2: reduce + scan (partitioned reduce + merge)
      context_attention_fwd: scan + reduce + transform (Triton prefill)
    """
    # Triton flash attention (prefill)
    triton_block_m: int = 32
    triton_block_n: int = 32
    triton_num_warps: int = 4
    triton_num_stages: int = 1
    # Paged attention (decode)
    partition_size: int = 512
    v1_v2_threshold: int = 8192
    # PyTorch fallback (long decode)
    pytorch_decode_threshold: int = 32768
    pytorch_max_tile_blocks: int = 1024
    # Backend selection
    use_native_v1: bool = True
    use_native_v2: bool = False


@dataclass(frozen=True)
class MoEConfig:
    """Config for fused MoE kernel.
    
    Maps to fused_moe_kernel's tl.constexpr parameters.
    Critical for Qwen3.6: 256 experts, top-8, 64 layers.
    """
    block_size_m: int = 64
    block_size_n: int = 64
    block_size_k: int = 32
    group_size_m: int = 8


@dataclass(frozen=True)
class TransformConfig:
    """Config for element-wise ops (SiLU, RMSNorm, RoPE).
    
    From CCCL dispatch_transform.cuh spread_out_items_per_thread:
      items = ceil_div(num_items, sm_count × threads × max_occupancy)
      clamped to [min_items, max_items]
    """
    bytes_in_flight: int = 64 * 1024  # 64KB for BI-V100
    # These are used by ixformer native kernels (not directly tunable)
    # but inform our SMEM budget calculations
    max_smem_per_block: int = 49152


@dataclass(frozen=True)
class CacheConfig:
    """Config for KV cache operations (copy, swap, reshape_and_cache)."""
    copy_block_size: int = 256


# ══════════════════════════════════════════════════════════════
# CCCL-style SMEM constraint checker
# (from muh_kernel_map.py check_smem, used across all policies)
# ══════════════════════════════════════════════════════════════

def check_smem(threads: int, items: int, elem_bytes: int,
               smem_limit: int = BI_V100.max_shared_memory_per_block) -> dict:
    """Verify tile fits in shared memory. Used by all policy selectors."""
    tile_bytes = threads * items * elem_bytes
    max_items = smem_limit // (threads * elem_bytes) if threads * elem_bytes > 0 else 0
    return {
        "tile_bytes": tile_bytes,
        "fits": tile_bytes <= smem_limit,
        "utilization": tile_bytes / smem_limit if smem_limit > 0 else 0,
        "max_items": max_items,
    }


# ══════════════════════════════════════════════════════════════
# CCCL GridEvenShare work distribution (grid_even_share.cuh)
# ══════════════════════════════════════════════════════════════

def grid_even_share(num_items: int, tile_size: int,
                    sm_count: int = BI_V100.sm_count,
                    subscription_factor: int = 5) -> dict:
    """Python port of GridEvenShare::DispatchInit.
    
    CCCL formula: max_blocks = sm_occupancy × sm_count × subscription_factor
    Then items are evenly distributed across blocks, with 'big' blocks
    getting one extra tile.
    """
    if num_items <= 0 or tile_size <= 0:
        return {"grid_size": 0, "total_tiles": 0}

    total_tiles = math.ceil(num_items / tile_size)
    max_grid_size = sm_count * subscription_factor  # ~80 for BI-V100
    grid_size = min(total_tiles, max_grid_size)
    avg_tiles = total_tiles // grid_size if grid_size > 0 else 0
    big_shares = total_tiles - (avg_tiles * grid_size) if grid_size > 0 else 0

    return {
        "grid_size": grid_size,
        "total_tiles": total_tiles,
        "avg_tiles_per_block": avg_tiles,
        "big_shares": big_shares,
        "tile_size": tile_size,
    }


# ══════════════════════════════════════════════════════════════
# Policy selectors (mirrors each algorithm's policy_selector)
# ══════════════════════════════════════════════════════════════

def select_attention_config(
    hw: HardwareCapability,
    dtype_size: int,  # element size in bytes (2=fp16, 4=fp32)
    head_dim: int,
    max_seq_len: int,
    num_kv_heads: int,
) -> AttentionConfig:
    """CCCL-style policy selector for paged attention.
    
    Mirrors: dispatch_reduce.cuh two-path dispatch
      single-tile: num_items ≤ threads × items → V1 (one CTA)
      multi-tile: GridEvenShare → V2 (partitioned + merge)
    
    SMEM constraint for Triton prefill:
      SMEM = BLOCK_N × head_dim × elem_size × 2 (K + V staging)
    """
    smem = hw.max_shared_memory_per_block

    # Triton BLOCK_N: largest that fits SMEM
    triton_block_n = 64
    while triton_block_n * head_dim * dtype_size * 2 > smem and triton_block_n > 16:
        triton_block_n //= 2

    triton_block_m = triton_block_n
    triton_num_warps = 4
    triton_num_stages = 1  # no async copy on BI-V100

    # V1/V2 threshold: V2 worthwhile when partitions > 1 AND
    # per-partition work exceeds merge overhead
    v1_threshold = 8192

    # PyTorch decode threshold: fall back for seq_len > this
    # (ixformer V1 hangs on very long sequences)
    pytorch_threshold = 32768

    # Tile blocks for PyTorch decode: from GridEvenShare
    # max_blocks = sm_count × subscription_factor = 80
    # Each tile processes ~16K tokens (1024 blocks × block_size=16)
    pytorch_tile_blocks = 1024

    return AttentionConfig(
        triton_block_m=triton_block_m,
        triton_block_n=triton_block_n,
        triton_num_warps=triton_num_warps,
        triton_num_stages=triton_num_stages,
        partition_size=512,
        v1_v2_threshold=v1_threshold,
        pytorch_decode_threshold=pytorch_threshold,
        pytorch_max_tile_blocks=pytorch_tile_blocks,
        use_native_v1=True,
        use_native_v2=False,
    )


def select_moe_config(
    hw: HardwareCapability,
    num_experts: int,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
) -> MoEConfig:
    """Policy selector for fused MoE.
    
    Qwen3.6: 256 experts, top-8, hidden=3584, intermediate=18944
    
    CCCL parallel: each expert is an independent reduce domain.
    With 256 experts × top-8 × batch=1 → 8 active experts per token.
    BI-V100 16 SMs can run 8 expert-matmuls in parallel → one wave.
    """
    # BLOCK_SIZE_M: tokens per tile. For decode (M=1), smallest possible.
    # For prefill (M=4096), larger is better to amortize overhead.
    block_m = 64 if top_k * 1 >= 64 else 32  # decode: top_k tokens
    block_n = 64
    block_k = 32

    # SMEM check: A_tile + B_tile
    # A: block_m × block_k × 2 bytes = 64×32×2 = 4KB
    # B: block_k × block_n × 2 bytes = 32×64×2 = 4KB
    # Total: 8KB << 48KB ✓

    return MoEConfig(
        block_size_m=block_m,
        block_size_n=block_n,
        block_size_k=block_k,
        group_size_m=8,
    )


# ══════════════════════════════════════════════════════════════
# Pre-computed configs (mirrors CCCL compile-time instantiation)
#
# cc_dispatch.cuh line 62: lowest_cc_resolver collapses identical
# policies across CCs. Our equivalent: compute once at import time.
# ══════════════════════════════════════════════════════════════

# Qwen3.6-35B-A3B model parameters (confirmed from qwen3_5.py)
QWEN36_HEAD_DIM = 256       # text_cfg.head_dim
QWEN36_NUM_KV_HEADS = 4     # num_key_value_heads
QWEN36_MAX_SEQ_LEN = 100000 # from computility-run.yaml
QWEN36_MAX_NUM_SEQS = 1     # CRITICAL: computility-run.yaml --max-num-seqs 1
QWEN36_NUM_EXPERTS = 256    # MoE experts
QWEN36_TOP_K = 8            # MoE top-k
QWEN36_HIDDEN = 3584        # hidden_size
QWEN36_INTERMEDIATE = 18944 # intermediate_size

# Pre-computed for fp16 (the common dtype on BI-V100)
ATTENTION_FP16 = select_attention_config(
    hw=BI_V100,
    dtype_size=2,
    head_dim=QWEN36_HEAD_DIM,
    max_seq_len=QWEN36_MAX_SEQ_LEN,
    num_kv_heads=QWEN36_NUM_KV_HEADS,
)

# Pre-computed for bf16
ATTENTION_BF16 = select_attention_config(
    hw=BI_V100,
    dtype_size=2,  # bf16 same size as fp16
    head_dim=QWEN36_HEAD_DIM,
    max_seq_len=QWEN36_MAX_SEQ_LEN,
    num_kv_heads=QWEN36_NUM_KV_HEADS,
)

MOE_CONFIG = select_moe_config(
    hw=BI_V100,
    num_experts=QWEN36_NUM_EXPERTS,
    top_k=QWEN36_TOP_K,
    hidden_size=QWEN36_HIDDEN,
    intermediate_size=QWEN36_INTERMEDIATE,
)

TRANSFORM_CONFIG = TransformConfig(
    bytes_in_flight=BI_V100.bytes_in_flight,
    max_smem_per_block=BI_V100.max_shared_memory_per_block,
)

CACHE_CONFIG = CacheConfig(copy_block_size=256)


# ══════════════════════════════════════════════════════════════
# Unified dispatch entry point
# (mirrors CCCL dispatch_compute_cap)
# ══════════════════════════════════════════════════════════════

_CONFIGS = {
    "attention": ATTENTION_FP16,
    "attention_fp16": ATTENTION_FP16,
    "attention_bf16": ATTENTION_BF16,
    "moe": MOE_CONFIG,
    "transform": TRANSFORM_CONFIG,
    "cache": CACHE_CONFIG,
}


def dispatch_kernel_config(kernel_name: str,
                           hw: HardwareCapability = BI_V100) -> Any:
    """Unified policy dispatch — Python equivalent of dispatch_compute_cap.
    
    Usage:
        config = dispatch_kernel_config("attention")
        # config.triton_block_m, config.partition_size, etc.
        
        config = dispatch_kernel_config("moe")
        # config.block_size_m, config.block_size_n, etc.
    """
    if kernel_name not in _CONFIGS:
        raise KeyError(
            f"Unknown kernel: {kernel_name}. "
            f"Available: {list(_CONFIGS.keys())}"
        )
    return _CONFIGS[kernel_name]


# ══════════════════════════════════════════════════════════════
# CLI: dump all configs for inspection
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("muh_cc_dispatch: CCCL-style unified kernel policy dispatch\n")
    print(f"Hardware: {BI_V100.vendor} BI-V100")
    print(f"  SMs: {BI_V100.sm_count}, SMEM: {BI_V100.max_shared_memory_per_block//1024}KB, "
          f"BW: {BI_V100.memory_bandwidth_gbps}GB/s, "
          f"BW/SM: {BI_V100.bandwidth_per_sm_gbps:.1f}GB/s")
    print(f"  bytes_in_flight: {BI_V100.bytes_in_flight//1024}KB\n")

    for name, config in _CONFIGS.items():
        print(f"[{name}]")
        for k, v in config.__dict__.items():
            if not k.startswith('_'):
                print(f"  {k}: {v}")
        print()

    # GridEvenShare example for decode
    print("GridEvenShare example (50K token decode, block_size=16):")
    es = grid_even_share(50000 // 16, tile_size=1024)
    for k, v in es.items():
        print(f"  {k}: {v}")
