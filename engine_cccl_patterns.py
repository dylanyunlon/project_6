"""
engine_cccl_patterns.py — CCCL 系统级设计模式移植到 BI-V100 vllm 引擎
==========================================================================

从 CCCL 源码中提取的不是参数值，而是架构设计模式。
每个模式引用具体的 CCCL 源文件和行号。

核心发现（来自完整 CCCL 源码阅读）:

1. Reduce vs Scan 的 SMEM 差异:
   - agent_reduce.cuh: 数据直接 striped load 到寄存器，NOT SMEM staging
     → SMEM 只给 BlockReduce 的 warp shuffle scratch
     → items_per_thread 不受 SMEM 限制，只受 register pressure 限制
     → BI-V100 可以用 items=24 (CCCL SM100 只用 items=16)
   - agent_scan.cuh: 数据先 BlockLoad 到 SMEM staging buffer
     → SMEM = BlockLoad::TempStorage ∪ BlockStore::TempStorage ∪ (BlockScan + Prefix)
     → items_per_thread 严格受 tpb * ipt * type_size ≤ 48KB 约束
     → BI-V100 和 SM100 共享这个约束

2. scan delay 在 BI-V100 上完全无效:
   - single_pass_scan_operators.cuh 第 130 行:
     if (gridDim.x < GridThreshold=500) { __threadfence_block(); }
     else { __nanosleep(Delay); }
   - BI-V100: 16 SMs × 2 CTAs/SM = 32 blocks << 500
   - 结论: 所有 delay 策略退化为 __threadfence_block()
   - 意味着 dcid/ns/l2w 三个参数在 BI-V100 上无效，不需要调

3. dispatch_reduce.cuh 的 two-phase 模式 = paged_attention_v2:
   - Phase 1: DeviceReduceKernel → 每个 CTA 算一个 tile partition
     - GridEvenShare 均匀分配 → 对应 V2 的 partition 分配
     - StableReductionOrder=false → atomic 聚合 (BI-V100: 16 SM 低争用)
     - StableReductionOrder=true → write to d_out[blockIdx.x] + Phase 2
   - Phase 2: DeviceReduceSingleTileKernel → 一个 CTA 归约所有 partition 结果
     - 对应 V2 的 cross-partition log-sum-exp merge

4. agent_reduce.cuh 的向量化加载条件:
   ATTEMPT_VECTORIZATION = (vec_size > 1) && (items % vec == 0)
     && is_pointer<InputT> && is_trivially_relocatable<InputT>
     && sizeof(InputT) <= 8
   - PyTorch 等价: 用 .view().reshape() 做 contiguous 后 torch.bmm (已实现)
   - 不等价: scatter/gather 非连续内存 → 强制 scalar path

5. cc_dispatch.cuh 的 policy 折叠:
   - lowest_cc_resolver: 多个 CC 生成相同 policy → 共享 kernel 实例化
   - BI-V100 等价: 所有 Qwen3.6 配置 (bf16, head_dim=256, kv_heads=4)
     → 预计算一套配置，不做运行时 dispatch
"""

import torch
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# ═══════════════════════════════════════════════════════════════
# Hardware descriptor — mirrors muh/include/muh/hardware.cuh
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class HW:
    """BI-V100 hardware profile, confirmed via ixsmi on Phanthy Cloud."""
    sm_count: int = 16
    smem_per_block: int = 49152       # 48 KiB
    warp_size: int = 32
    max_threads: int = 1024
    hbm_bw_gbps: int = 900
    l2_bytes: int = 6 * 1024 * 1024   # 6 MiB
    bw_per_sm_gbps: float = 900 / 16  # 56.25 GB/s ≈ B200 level
    bytes_in_flight: int = 64 * 1024  # bench_bi100.py verified: bif=8 wins
    max_concurrent_ctas: int = 32     # 16 SM × ~2 occupancy

    # CCCL single_pass_scan_operators.cuh GridThreshold
    # All grids < 500 blocks → delay() becomes __threadfence_block()
    scan_delay_threshold: int = 500

BI100 = HW()


# ═══════════════════════════════════════════════════════════════
# Pattern 1: CCCL GridEvenShare work distribution
# Source: cub/grid/grid_even_share.cuh
# Used by: dispatch_reduce.cuh line ~200
# ═══════════════════════════════════════════════════════════════

def grid_even_share(
    num_items: int,
    sm_count: int = BI100.sm_count,
    sm_occupancy: int = 2,
    subscription_factor: int = 5,  # CCCL util_device.cuh default
    tile_size: int = 512 * 24,     # threads × items for reduce
) -> Dict:
    """
    CCCL's GridEvenShare maps work to CTAs.
    dispatch_reduce.cuh line 200:
      max_blocks = sm_occupancy * sm_count * subscription_factor
      even_share.DispatchInit(num_items, max_blocks, tile_size)

    Returns partition plan for paged_attention_v2.
    """
    max_blocks = sm_occupancy * sm_count * subscription_factor
    # GridEvenShare.DispatchInit: divide num_items into even tiles
    num_tiles = (num_items + tile_size - 1) // tile_size
    grid_size = min(num_tiles, max_blocks)

    # For BI-V100: max_blocks = 2 × 16 × 5 = 160
    # For 100K tokens with tile=12288: num_tiles=9, grid=9
    # For 100K tokens with partition=1024: num_tiles=98, grid=98

    return {
        "num_items": num_items,
        "tile_size": tile_size,
        "max_blocks": max_blocks,
        "grid_size": grid_size,
        "items_per_cta": (num_items + grid_size - 1) // grid_size if grid_size > 0 else num_items,
        "single_tile": num_tiles <= 1,
    }


# ═══════════════════════════════════════════════════════════════
# Pattern 2: CCCL AgentReduce tile consumption
# Source: agent_reduce.cuh ConsumeFullTile (two paths)
# Key insight: reduce does NOT use BlockLoad SMEM staging
# ═══════════════════════════════════════════════════════════════

def reduce_tile_config(
    accum_size: int,     # sizeof(AccumT) in bytes
    hw: HW = BI100,
) -> Dict:
    """
    Compute optimal reduce tile config for BI-V100.

    CCCL agent_reduce.cuh insight: data goes to REGISTERS not SMEM.
    The SMEM constraint that limits scan (tpb*ipt*type_size ≤ 48KB)
    does NOT apply to reduce. Instead, register pressure is the limit:
      - Each thread holds AccumT items[ITEMS_PER_THREAD] in registers
      - BI-V100 has 64K registers/SM (255 per thread max)
      - items=24 for float32 → 24 registers → acceptable
      - Larger items → fewer CTAs possible → but 16 SMs only need ~32 CTAs anyway

    Vectorized load condition (agent_reduce.cuh line ~243):
      ATTEMPT_VECTORIZATION = vec_size > 1 && items % vec == 0
        && is_pointer && is_trivially_relocatable && sizeof <= 8
    """
    # Register pressure limit
    regs_per_item = accum_size // 4  # 1 reg = 4 bytes for float32
    if regs_per_item < 1:
        regs_per_item = 1

    # Target: ~40 registers per thread total (data + overhead)
    # 255 max regs per thread, but high reg usage reduces occupancy
    max_items_by_regs = min(64, 40 // regs_per_item)

    # CCCL SM100 reference values
    cccl_items = {1: 32, 2: 24, 4: 16, 8: 16, 16: 16}
    reference = cccl_items.get(accum_size, 16)

    # BI-V100 adjustment: 16 SMs → larger tiles to compensate
    # Each CTA should process more data (fewer CTAs total)
    # Scale: items = reference × (SM100_count / BI100_count)^0.3
    #   = reference × (148/16)^0.3 ≈ reference × 2.2
    # But cap at register limit
    bi100_items = min(max_items_by_regs, int(reference * 2.0))

    # Threads: 512 for most types (CCCL SM100 default)
    # Except float64 where CCCL uses 640 → BI-V100 uses 384 (12 warps, clean)
    threads = 384 if accum_size >= 8 else 512

    # Vectorization
    if accum_size <= 8 and bi100_items % 2 == 0:
        vec_size = 2 if accum_size >= 4 else 4
    else:
        vec_size = 1

    return {
        "threads": threads,
        "items": bi100_items,
        "vec_size": vec_size,
        "tile_size": threads * bi100_items,
        "regs_per_thread": bi100_items * regs_per_item + 16,  # +16 for overhead
        "smem_limited": False,  # reduce is NOT SMEM limited
        "cccl_reference_items": reference,
    }


# ═══════════════════════════════════════════════════════════════
# Pattern 3: CCCL AgentScan tile with SMEM staging
# Source: agent_scan.cuh ConsumeTile
# Key insight: scan DOES use BlockLoad SMEM staging → strict SMEM limit
# ═══════════════════════════════════════════════════════════════

def scan_tile_config(
    accum_size: int,
    hw: HW = BI100,
) -> Dict:
    """
    Compute optimal scan tile config for BI-V100.

    CCCL agent_scan.cuh: uses BlockLoad → data goes through SMEM staging.
    _TempStorage is a union of:
      - BlockLoadT::TempStorage  (tpb * items * type_size)
      - BlockStoreT::TempStorage (tpb * items * type_size)
      - BlockScanT::TempStorage + TilePrefixCallbackOpT::TempStorage

    The BlockLoad/Store staging is the SMEM bottleneck:
      tpb * ipt * accum_size ≤ 48KB

    Additional constraint: WARP_TRANSPOSE load requires
      tpb * ipt * sizeof(AccumT) bytes of staging buffer.

    CCCL SM100 scan benchmark winners:
      float32/o4: ipt=22, tpb=384 (tile=33792 ≤ 48K) → speedup 1.148
      float64/o4: ipt=23, tpb=416 (tile=76544 > 48K!) → uses NoScaling
      int8/o4:    ipt=18, tpb=512 (tile=9216 ≤ 48K)

    But wait — CCCL SM100 float64 tile = 416*23*8 = 76544 > 49152!
    How does this work? Because SM100 can configure larger SMEM (228KB).
    BI-V100 is stuck at 48KB → must reduce items for large types.
    """
    max_smem = hw.smem_per_block

    # Start with CCCL SM100 winners, then constrain
    cccl_configs = {
        1: (512, 18),   # int8:    512*18*1 = 9216
        2: (512, 13),   # int16:   512*13*2 = 13312
        4: (384, 22),   # float32: 384*22*4 = 33792 ✓
        8: (384, 14),   # float64: 384*14*8 = 43008 ✓ (reduced from SM100's 23)
        16: (256, 12),  # int128:  256*12*16= 49152 = exactly 48KB
    }

    threads, items = cccl_configs.get(accum_size, (384, 16))

    # Verify SMEM constraint
    tile_bytes = threads * items * accum_size
    while tile_bytes > max_smem and items > 1:
        items -= 1
        tile_bytes = threads * items * accum_size

    # scan delay is IRRELEVANT on BI-V100
    # single_pass_scan_operators.cuh: gridDim.x < 500 → __threadfence_block()
    # BI-V100 max grid = ~160 << 500, so ALL delay strategies collapse
    delay_effective = "threadfence_block_only"

    return {
        "threads": threads,
        "items": items,
        "tile_size": threads * items,
        "tile_bytes": threads * items * accum_size,
        "smem_utilization": (threads * items * accum_size) / max_smem,
        "smem_limited": True,  # scan IS SMEM limited
        "delay_strategy": delay_effective,
        "load_algorithm": "WARP_TRANSPOSE" if accum_size >= 4 else "DIRECT",
    }


# ═══════════════════════════════════════════════════════════════
# Pattern 4: CCCL compound reduce (summary_statistics.cu Welford)
# Source: thrust/examples/summary_statistics.cu
# Maps to: paged_attention_v2 cross-partition merge
# ═══════════════════════════════════════════════════════════════

def compound_reduce_merge(
    max_a: torch.Tensor,    # [H, P_a] partition maxima from partition set A
    sum_a: torch.Tensor,    # [H, P_a] partition exp-sums
    out_a: torch.Tensor,    # [H, P_a, d] partition weighted outputs
    max_b: torch.Tensor,    # [H, P_b]
    sum_b: torch.Tensor,    # [H, P_b]
    out_b: torch.Tensor,    # [H, P_b, d]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Merge two sets of attention partition results.

    Direct translation of summary_statistics.cu binary_op,
    adapted for online softmax instead of Welford variance:

    CCCL summary_stats_binary_op (lines 97-125):
      n = x.n + y.n
      delta = y.mean - x.mean
      mean = x.mean + delta * y.n / n
      M2 = x.M2 + y.M2 + delta^2 * x.n * y.n / n

    Our attention equivalent:
      global_max = max(max_a, max_b)          // delta = max_b - max_a
      rescale_a = exp(max_a - global_max)     // similar to delta normalization
      rescale_b = exp(max_b - global_max)
      total_sum = sum_a * rescale_a + sum_b * rescale_b
      merged_out = (out_a * sum_a * rescale_a + out_b * sum_b * rescale_b) / total_sum

    The Welford parallel merge and log-sum-exp merge are structurally
    identical — both need to rescale accumulated statistics when combining
    partial results computed with different reference points (mean vs max).

    This function enables incremental/streaming V2: process new KV blocks
    without recomputing from scratch. CCCL's ConsumeTiles pattern:
      for each tile: ConsumeFullTile → ThreadReduce → update aggregate
    becomes:
      for each new KV block batch: compute partition → merge with running result
    """
    H = max_a.shape[0]
    device = max_a.device

    # Concatenate along partition dimension
    all_max = torch.cat([max_a, max_b], dim=1)  # [H, P_a + P_b]
    all_sum = torch.cat([sum_a, sum_b], dim=1)
    all_out = torch.cat([out_a, out_b], dim=1)  # [H, P_a + P_b, d]

    # Global max for numerical stability
    global_max = all_max.max(dim=1, keepdim=True).values  # [H, 1]

    # Rescale: exp(partition_max - global_max) * partition_sum
    rescale = torch.exp(all_max - global_max) * all_sum  # [H, P]
    total = rescale.sum(dim=1, keepdim=True)  # [H, 1]

    # Weighted merge: bmm(rescale, out) / total
    # CCCL norm.cu insight: fuse transform with reduce to minimize traversals
    result = torch.bmm(rescale.unsqueeze(1), all_out.float()).squeeze(1) / total  # [H, d]

    # Return merged statistics (for further merging if needed)
    merged_max = global_max.squeeze(1)         # [H]
    merged_sum = total.squeeze(1)              # [H]
    merged_out = result.unsqueeze(1)           # [H, 1, d]

    return merged_max, merged_sum, merged_out


# ═══════════════════════════════════════════════════════════════
# Pattern 5: CCCL dispatch_compute_cap policy precomputation
# Source: cc_dispatch.cuh lowest_cc_resolver
# BI-V100: all Qwen3.6 configs precomputed at import time
# ═══════════════════════════════════════════════════════════════

# Pre-computed tile configs for all Qwen3.6 data types
# (mirrors CCCL's compile-time policy instantiation)
REDUCE_CONFIGS = {
    "float16": reduce_tile_config(2),   # KV cache values
    "bfloat16": reduce_tile_config(2),
    "float32": reduce_tile_config(4),   # attention scores
    "float64": reduce_tile_config(8),   # (rarely used)
    "int32": reduce_tile_config(4),     # indices
}

SCAN_CONFIGS = {
    "float32": scan_tile_config(4),     # softmax denominator
    "float64": scan_tile_config(8),
    "int32": scan_tile_config(4),
}

# Qwen3.6 specific: paged attention V2 partition plan
QWEN36_V2_PLAN = grid_even_share(
    num_items=100000,   # max_model_len
    tile_size=1024,     # PARTITION_SIZE
)


# ═══════════════════════════════════════════════════════════════
# Pattern 6: CCCL single-tile fast path
# Source: kernel_reduce.cuh line ~270 (DeviceReduceSingleTileKernel)
# dispatch_reduce.cuh Invoke(): if small → InvokeSingleTile
# ═══════════════════════════════════════════════════════════════

def should_use_single_tile(
    seq_len: int,
    partition_size: int = 1024,
    reduce_config: Dict = None,
) -> bool:
    """
    CCCL dispatch_reduce.cuh decision logic:
      if (num_items <= threads * items_per_thread):
          InvokeSingleTile()  # one CTA, no Phase 2
      else:
          InvokePasses()      # multi-CTA + reduce

    For paged attention:
      - tokens ≤ partition_size → one partition → no Phase 2 merge needed
      - This is the common case during early decode (seq_len grows from 1 up)
      - Avoids partition overhead for the majority of decode steps
    """
    if reduce_config is None:
        reduce_config = REDUCE_CONFIGS["float32"]
    single_tile_capacity = reduce_config["tile_size"]  # e.g. 512 * 24 = 12288

    # Two conditions (from CCCL):
    # 1. Fits in one partition → skip partitioning entirely
    # 2. Fits in one CTA's tile → skip GridEvenShare overhead
    return seq_len <= partition_size or seq_len <= single_tile_capacity


if __name__ == "__main__":
    print("=== CCCL Pattern Analysis for BI-V100 ===\n")

    print("Reduce configs (NOT SMEM limited — register pressure only):")
    for dtype, cfg in REDUCE_CONFIGS.items():
        print(f"  {dtype}: threads={cfg['threads']}, items={cfg['items']}, "
              f"vec={cfg['vec_size']}, tile={cfg['tile_size']}, "
              f"regs/thread≈{cfg['regs_per_thread']}")

    print("\nScan configs (SMEM limited — strict 48KB constraint):")
    for dtype, cfg in SCAN_CONFIGS.items():
        print(f"  {dtype}: threads={cfg['threads']}, items={cfg['items']}, "
              f"tile_bytes={cfg['tile_bytes']}, "
              f"smem_util={cfg['smem_utilization']:.0%}, "
              f"delay={cfg['delay_strategy']}")

    print(f"\nQwen3.6 V2 partition plan (100K tokens):")
    plan = QWEN36_V2_PLAN
    print(f"  partitions={plan['grid_size']}, per_cta={plan['items_per_cta']}, "
          f"max_blocks={plan['max_blocks']}, single_tile={plan['single_tile']}")

    print(f"\nSingle-tile threshold examples:")
    for sl in [100, 500, 1024, 5000, 12288, 50000]:
        print(f"  seq_len={sl:>6d}: single_tile={should_use_single_tile(sl)}")


# ═══════════════════════════════════════════════════════════════
# Pattern 7: CCCL C API JIT Build-then-Run → Triton autotune
# Source: c/parallel.v2/src/reduce.cu, scan.cu
# EngineX's "algorithm factor substitution" = this pattern
# ═══════════════════════════════════════════════════════════════
TRITON_AUTOTUNE_CONFIGS = {
    "prefill_attention": [
        {"BLOCK_M": 32, "BLOCK_N": 32, "num_warps": 4, "num_stages": 1},
        {"BLOCK_M": 16, "BLOCK_N": 32, "num_warps": 4, "num_stages": 1},
    ],
    "decode_v1": [{"NUM_THREADS": 512, "items": 24, "vec": 2}],
    "decode_v2": [{"PARTITION_SIZE": 1024}],
    "topk": [{"threads": 512, "items": 4, "bits_per_pass": 11}],
}
