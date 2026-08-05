"""
muh_dispatch.py — CCCL-style type-dispatched kernel configuration for BI-V100
===============================================================================

Mirrors CCCL's cc_dispatch.cuh architecture:
  cc_dispatch: policy_selector(compute_capability) → policy struct
  muh_dispatch: select_attention_config(hw, dtype, head_dim, ...) → AttentionConfig

Key corrections from CCCL source reading (cc_dispatch.cuh, 150 lines):
  - CCCL collapses architectures with identical policies (lowest_cc_resolver)
  - CCCL dispatches at COMPILE TIME via policy_getter<PolicySelector, CC>
  - Python equivalent: precompute configs at import time, not per-call

Source: cccl_upstream/cub/cub/detail/cc_dispatch.cuh
        cccl_upstream/cub/cub/device/dispatch/dispatch_common.cuh
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HardwareCapability:
    """Mirrors muh/include/muh/hardware.cuh"""
    warp_size: int = 32
    max_threads_per_block: int = 1024
    max_shared_memory_per_block: int = 49152  # 48KB — confirmed via ixsmi
    sm_count: int = 16           # CONFIRMED: 16 SMs per BI-V100 (NOT 50)
    memory_bandwidth_gbps: int = 900
    l2_cache_size_bytes: int = 6 * 1024 * 1024  # 6MB

BI_V100 = HardwareCapability()


@dataclass
class AttentionConfig:
    """Complete kernel config — mirrors CCCL's ReducePolicy/ScanPolicy output."""
    # Triton flash attention (prefill)
    triton_block_m: int = 32
    triton_block_n: int = 32
    triton_num_warps: int = 4
    triton_num_stages: int = 1

    # Paged attention V1/V2 (decode)
    partition_size: int = 512
    v1_v2_threshold: int = 8192

    # Backend selection
    use_native_v1: bool = True
    use_native_v2: bool = False   # native V2 has correctness issues on BI-V100
    use_triton_prefill: bool = True


def select_attention_config(
    hw: HardwareCapability,
    dtype: torch.dtype,
    head_dim: int,
    max_seq_len: int,
    num_kv_heads: int,
) -> AttentionConfig:
    """CCCL-style policy selector for paged attention.

    CCCL dispatch axes: (compute_capability, type_t, op_kind_t, offset_size)
    Our dispatch axes: (hardware, dtype, head_dim, max_seq_len, num_kv_heads)
    """
    elem_size = dtype.itemsize if hasattr(dtype, 'itemsize') else torch.tensor([], dtype=dtype).element_size()
    smem = hw.max_shared_memory_per_block

    # --- Triton prefill config ---
    # SMEM = BLOCK_N × head_dim × elem_size × 2 (K + V staging)
    # Must fit in 48KB with margin for softmax accumulators
    # Qwen3.6: head_dim=256, bf16 → elem_size=2
    #   BLOCK_N=64: 64×256×2×2 = 64KB > 48KB → CRASH
    #   BLOCK_N=32: 32×256×2×2 = 32KB ≤ 48KB ✓
    #   BLOCK_N=64 only safe for head_dim≤128: 64×128×2×2 = 32KB

    triton_block_n = 64
    while triton_block_n * head_dim * elem_size * 2 > smem and triton_block_n > 16:
        triton_block_n //= 2

    # BLOCK_M: same as BLOCK_N for square tiles (simplifies causal mask)
    # BI-V100: 4 warps, not 8 (BLOCK=32 → 32 rows, 8 warps = 256 threads
    # means only 32/256=0.125 rows/thread — wasteful)
    triton_block_m = triton_block_n
    triton_num_warps = 4

    # fp32 halves the block (element size doubles → SMEM doubles)
    if dtype == torch.float32:
        triton_block_m //= 2
        triton_block_n //= 2

    # num_stages=1 on BI-V100: no async copy hardware (needs SM80+ cp.async)
    triton_num_stages = 1

    # --- Paged attention decode config ---
    # V1 threshold: for seq_len > threshold, V2 would be better IF V2 were native C++
    # Currently V2 is PyTorch → always slower than V1 ixformer
    # So threshold is effectively infinite (always V1)
    v1_threshold = max_seq_len + 1  # force V1

    return AttentionConfig(
        triton_block_m=triton_block_m,
        triton_block_n=triton_block_n,
        triton_num_warps=triton_num_warps,
        triton_num_stages=triton_num_stages,
        partition_size=512,
        v1_v2_threshold=v1_threshold,
        use_native_v1=True,
        use_native_v2=False,
        use_triton_prefill=True,
    )


# Pre-computed configs (mirrors CCCL's compile-time policy instantiation)
# CCCL does this via template instantiation; we do it at import time.

QWEN36_BF16 = select_attention_config(
    hw=BI_V100,
    dtype=torch.bfloat16,
    head_dim=256,        # CONFIRMED from qwen3_5.py: text_cfg.head_dim = 256
    max_seq_len=100000,  # from computility-run.yaml: --max-model-len 100000
    num_kv_heads=4,      # CONFIRMED: num_key_value_heads = 4
)

QWEN36_FP16 = select_attention_config(
    hw=BI_V100,
    dtype=torch.float16,
    head_dim=256,
    max_seq_len=100000,
    num_kv_heads=4,
)


if __name__ == "__main__":
    print("=== muh_dispatch: CCCL-style type-dispatched kernel config ===\n")
    print(f"Qwen3.6 bf16 (head_dim=256):")
    print(f"  triton: BLOCK_M={QWEN36_BF16.triton_block_m} BLOCK_N={QWEN36_BF16.triton_block_n}"
          f" warps={QWEN36_BF16.triton_num_warps} stages={QWEN36_BF16.triton_num_stages}")
    print(f"  decode: partition={QWEN36_BF16.partition_size} v1_thresh={QWEN36_BF16.v1_v2_threshold}")
    print(f"  SMEM: {QWEN36_BF16.triton_block_n}×256×2×2 = {QWEN36_BF16.triton_block_n*256*2*2} bytes"
          f" ({QWEN36_BF16.triton_block_n*256*2*2/1024:.0f}KB ≤ 48KB)")
    print(f"  V1 forced: {QWEN36_BF16.use_native_v1} (V2 native has correctness issues)")
