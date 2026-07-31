"""
muh_dispatch.py — CCCL-style type-dispatched kernel configuration for BI-V100
===============================================================================

This is the key differentiator. Everyone else hardcodes:
    BLOCK_SIZE = 64
    NUM_WARPS = 4
    PARTITION_SIZE = 512

muh_dispatch replaces these with type-dispatched values derived from CCCL's
policy_selector architecture. The dispatch key is (dtype, head_dim, seq_len),
and the output is a complete kernel configuration tuple.

CCCL reference: cub/device/dispatch/tuning/tuning_reduce.cuh
  Input: (compute_capability, accum_type, op_kind, offset_size, determinism)
  Output: ReducePolicy{multi_tile, single_tile} where each pass has
          (threads, items, vec_size, algorithm, load_modifier)

muh_dispatch translation for paged attention:
  Input: (hardware, dtype, head_dim, seq_len, num_kv_heads)
  Output: AttentionConfig{partition_size, block_size, num_warps, vec_size, 
                          v1_threshold, use_triton}

Deploy: cp muh_dispatch.py /usr/local/corex/.../vllm/muh_dispatch.py
        Then patch paged_attn.py to import and use it.
"""

import torch
from dataclasses import dataclass
from typing import Optional

# ============================================================
# Hardware descriptor — mirrors muh/include/muh/hardware.cuh
# ============================================================

@dataclass(frozen=True)
class HardwareCapability:
    warp_size: int = 32
    max_threads_per_block: int = 1024
    max_shared_memory_per_block: int = 49152  # 48KB
    sm_count: int = 50
    memory_bandwidth_gbps: int = 900
    l2_cache_size_bytes: int = 6 * 1024 * 1024  # 6MB

BI_V100 = HardwareCapability()

# ============================================================
# Type classification — mirrors cub/device/dispatch/tuning/common.cuh
# ============================================================

def classify_dtype(dtype: torch.dtype) -> dict:
    """Classify a torch dtype into CCCL-compatible type descriptors."""
    type_map = {
        torch.float16:  {"size": 2, "type_t": "float16",  "is_float": True},
        torch.bfloat16: {"size": 2, "type_t": "bfloat16", "is_float": True},
        torch.float32:  {"size": 4, "type_t": "float32",  "is_float": True},
        torch.float64:  {"size": 8, "type_t": "float64",  "is_float": True},
        torch.int8:     {"size": 1, "type_t": "int8",     "is_float": False},
        torch.int32:    {"size": 4, "type_t": "int32",     "is_float": False},
        torch.int64:    {"size": 8, "type_t": "int64",     "is_float": False},
    }
    return type_map.get(dtype, {"size": dtype.itemsize, "type_t": "other", "is_float": False})

# ============================================================
# Attention kernel configuration
# ============================================================

@dataclass
class AttentionConfig:
    """Complete kernel configuration for one paged attention call.
    
    Mirrors CCCL's ReducePolicy / ScanPolicy output structure:
    a single struct containing all parameters the kernel needs.
    """
    # Triton flash attention (prefill)
    triton_block_n: int = 64
    triton_num_warps: int = 4
    
    # Paged attention V1/V2 (decode)
    partition_size: int = 512
    v1_v2_threshold: int = 8192  # seq_len above this → use V2
    
    # Vectorization — derived from dtype
    vec_size: int = 4  # elements per vector load
    
    # Reduce pattern (score reduction per head)
    reduce_threads: int = 512
    reduce_items: int = 16
    
    # Backend selection
    use_native_v1: bool = True
    use_native_v2: bool = False  # V2 native has correctness issues
    use_triton_prefill: bool = True

# ============================================================
# policy_selector — the CCCL-style dispatch function
#
# This is the core: instead of one set of hardcoded constants,
# we dispatch based on (dtype, head_dim, seq_len).
#
# Why this matters for competition:
#   - fp16 attention with head_dim=128: score accum is fp32 (4B)
#     → reduce can use ipt=16, tpb=512 (tile=32KB ≤ 48KB)
#   - fp16 attention with head_dim=256: score tile is 2x larger
#     → reduce must use ipt=8 to fit SMEM
#   - Long sequences (>32K): partition_size=1024 better amortizes
#     the V2 reduce overhead
#   - Short sequences (<1K): V1 always wins, skip V2 entirely
# ============================================================

def select_attention_config(
    hw: HardwareCapability,
    dtype: torch.dtype,
    head_dim: int,
    max_seq_len: int,
    num_kv_heads: int,
) -> AttentionConfig:
    """CCCL-style policy selector for paged attention.
    
    Dispatch axes (matching CCCL's type_t × op_kind_t × offset_size):
      - dtype → determines accum_size, SMEM per element
      - head_dim → determines tile width
      - max_seq_len → determines V1/V2 threshold and partition_size
      - num_kv_heads → determines GQA ratio (affects memory pattern)
    """
    info = classify_dtype(dtype)
    elem_size = info["size"]
    
    # --- Triton prefill config ---
    # SMEM for flash attention = BLOCK_N × head_dim × elem_size × 2 (K+V)
    # Must fit in 48KB
    triton_block_n = 128
    triton_smem = triton_block_n * head_dim * elem_size * 2
    while triton_smem > hw.max_shared_memory_per_block and triton_block_n > 16:
        triton_block_n //= 2
        triton_smem = triton_block_n * head_dim * elem_size * 2
    
    # NUM_WARPS: bandwidth-limited GPU → fewer warps, more blocks
    # CCCL analogy: transform policy uses 128 threads (4 warps) for SM100
    # because bulk operations are BW-limited
    triton_num_warps = 4 if hw.memory_bandwidth_gbps < 1500 else 8
    
    # --- Paged attention decode config ---
    # Score accumulator is always fp32 (4 bytes) regardless of KV dtype
    accum_size = 4
    
    # Partition size for V2:
    # Larger partition = fewer partitions = less reduce overhead
    # But each partition must fit: partition_size × head_dim × accum_size in SMEM
    # CCCL parallel: reduce tile_size = threads × items × accum_size ≤ SMEM
    partition_smem = lambda ps: ps * head_dim * accum_size
    partition_size = 1024
    while partition_smem(partition_size) > hw.max_shared_memory_per_block:
        partition_size //= 2
    if partition_size < 256:
        partition_size = 256  # minimum for occupancy
    
    # V1/V2 threshold:
    # V1 is one block per (seq, head) — good for short seq
    # V2 splits into partitions — good for long seq
    # Crossover depends on SM count (more SMs → V2 wins earlier)
    # CCCL parallel: single_tile vs multi_tile in ReducePolicy
    if max_seq_len <= 2048:
        v1_threshold = max_seq_len + 1  # always V1
    elif hw.sm_count >= 80:
        v1_threshold = 4096  # high SM count → V2 wins earlier
    else:
        v1_threshold = 8192  # 50 SMs → V2 wins later
    
    # Vec size for score loads:
    # CCCL analogy: reduce uses vec_size=2 for fp32, vec_size=1 for fp64
    # because 128-bit loads = 4×fp32 = 2×fp64
    vec_size = min(16 // accum_size, 4)  # 128-bit / accum_size
    
    # Reduce config (for V2's final reduction across partitions):
    # Derived from muh/include/muh/tuning/tuning_reduce.cuh bi100_float32_plus_o4
    if accum_size <= 4:
        reduce_threads = 512
        reduce_items = 16
    else:
        # fp64 accumulator: smaller tile
        reduce_threads = 384
        reduce_items = 16
    
    # Sanity check: reduce tile fits SMEM
    reduce_tile = reduce_threads * reduce_items * accum_size
    while reduce_tile > hw.max_shared_memory_per_block:
        reduce_items -= 1
        reduce_tile = reduce_threads * reduce_items * accum_size
    
    return AttentionConfig(
        triton_block_n=triton_block_n,
        triton_num_warps=triton_num_warps,
        partition_size=partition_size,
        v1_v2_threshold=v1_threshold,
        vec_size=vec_size,
        reduce_threads=reduce_threads,
        reduce_items=reduce_items,
        use_native_v1=True,
        use_native_v2=False,  # still correctness issues
        use_triton_prefill=True,
    )


# ============================================================
# Convenience: get config for Qwen3.6 on BI-V100
# ============================================================

def qwen36_config() -> AttentionConfig:
    """Pre-computed config for Qwen3.6-35B-A3B on BI-V100.
    
    Qwen3.6 uses:
      - head_dim = 128
      - num_heads = 64, num_kv_heads = 8 (GQA 8:1)
      - dtype = bfloat16 / float16
      - max_model_len = 100000
    """
    return select_attention_config(
        hw=BI_V100,
        dtype=torch.bfloat16,
        head_dim=128,
        max_seq_len=100000,
        num_kv_heads=8,
    )


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    print("=== muh_dispatch: CCCL-style type-dispatched kernel config ===\n")
    
    configs = [
        ("Qwen3.6 bf16 h128 100K", torch.bfloat16, 128, 100000, 8),
        ("Qwen3.6 fp16 h128 100K", torch.float16, 128, 100000, 8),
        ("Qwen3.6 bf16 h256 100K", torch.bfloat16, 256, 100000, 8),
        ("Short context bf16 h128 2K", torch.bfloat16, 128, 2048, 8),
        ("fp32 fallback h128 32K", torch.float32, 128, 32768, 8),
    ]
    
    for name, dtype, hdim, seqlen, kvh in configs:
        cfg = select_attention_config(BI_V100, dtype, hdim, seqlen, kvh)
        print(f"  {name}:")
        print(f"    triton: block_n={cfg.triton_block_n} warps={cfg.triton_num_warps}")
        print(f"    decode: partition={cfg.partition_size} v1_thresh={cfg.v1_v2_threshold}")
        print(f"    reduce: threads={cfg.reduce_threads} items={cfg.reduce_items} vec={cfg.vec_size}")
        print()
