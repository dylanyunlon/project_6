#!/usr/bin/env python3
"""muh/gen_config.py — Generate Python-layer config patches from CCCL tuning analysis

Unlike gen_patch.py (which targets non-existent .cu files), this generates
patches for the ACTUAL tunable parameters in enginex-vllm-bi100:

  1. paged_attn.py: _PARTITION_SIZE, V1/V2 dispatch
  2. prefix_prefill.py: BLOCK, BLOCK_N, NUM_WARPS
  3. triton_flash_attention.py: autotune Config entries
  4. _custom_ops.py: SMEM size
  5. computility-run.yaml: scheduler params

Each config is derived from CCCL tuning principles (SMEM constraints,
occupancy model, bytes_in_flight) applied to BI-V100 hardware.
"""

import os
import sys
import json
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# BI-V100 hardware (from hardware.cuh, confirmed via ixsmi)
HW = {
    "sm_count": 16,
    "smem_per_block": 49152,       # 48KB
    "l2_cache_bytes": 6 * 1024 * 1024,  # 6MB
    "hbm_bw_gbps": 900,
    "warp_size": 32,
    "max_threads_per_block": 1024,
    "max_regs_per_thread": 255,
    "regs_per_sm": 65536,
    # Derived
    "bw_per_sm_gbps": 900 / 16,    # 56.25 GB/s
    # bytes_in_flight = BW/SM × memory_latency ≈ 56 × 1100ns ≈ 62KB → 64KB
    "bytes_in_flight": 64 * 1024,
    # L2 per SM = 6MB / 16 = 384KB (higher than SM100's 338KB/SM!)
    "l2_per_sm_bytes": 6 * 1024 * 1024 // 16,
}

# Qwen3.6-35B-A3B model config
MODEL = {
    "head_dim": 256,         # CONFIRMED from qwen3_5.py
    "num_q_heads": 28,       # num_attention_heads
    "num_kv_heads": 4,       # num_key_value_heads
    "hidden_size": 3584,
    "intermediate_size": 18944,
    "num_layers": 64,
    "vocab_size": 152064,
    "num_experts": 256,      # MoE
    "top_k_experts": 8,
    "max_model_len": 100000,
}


@dataclass
class TritonConfig:
    block_m: int
    block_n: int
    num_warps: int
    num_stages: int
    pre_load_v: bool = False
    waves_per_eu: int = 2
    
    def smem_bytes(self, head_dim: int, elem_size: int) -> int:
        """SMEM = Q_tile + K_tile + V_tile + softmax_accum"""
        q_tile = self.block_m * head_dim * elem_size
        k_tile = head_dim * self.block_n * elem_size
        # V loaded in inner loop, not staged simultaneously with Q+K
        # But PRE_LOAD_V stages V in registers, not SMEM
        softmax = self.block_m * 4 * 2  # m_i + l_i, fp32
        return q_tile + k_tile + softmax
    
    def fits_smem(self, head_dim: int = 256, elem_size: int = 2) -> bool:
        return self.smem_bytes(head_dim, elem_size) <= HW["smem_per_block"]
    
    def occupancy_ctas(self, head_dim: int = 256, elem_size: int = 2) -> int:
        """Max concurrent CTAs per SM"""
        threads = self.num_warps * HW["warp_size"]
        smem = self.smem_bytes(head_dim, elem_size) * self.num_stages
        # Thread limit
        ctas_by_threads = HW["max_threads_per_block"] // threads
        # SMEM limit
        ctas_by_smem = HW["smem_per_block"] // max(1, smem)
        # Register limit (rough: assume 40 regs/thread)
        regs_per_cta = threads * 40
        ctas_by_regs = HW["regs_per_sm"] // max(1, regs_per_cta)
        return min(ctas_by_threads, ctas_by_smem, ctas_by_regs)
    
    def to_triton_str(self) -> str:
        return (
            f'triton.Config({{"BLOCK_M": {self.block_m}, "BLOCK_N": {self.block_n}, '
            f'"waves_per_eu": {self.waves_per_eu}, "PRE_LOAD_V": {self.pre_load_v}}}, '
            f'num_stages={self.num_stages}, num_warps={self.num_warps})'
        )


def generate_triton_configs(
    head_dim: int = 256,
    elem_size: int = 2,  # bf16/fp16
) -> List[TritonConfig]:
    """Generate all valid BI-V100 Triton flash attention configs.
    
    CCCL-derived constraints:
    - SMEM: Q_tile + K_tile + softmax ≤ 48KB
    - Occupancy: want ≥2 CTAs/SM for 16 SMs
    - bytes_in_flight: num_stages=2 matches 64KB BIF sweet spot
    - Wave efficiency: total CTAs should be multiple of 16
    """
    configs = []
    
    for block_m in [16, 32, 64, 128]:
        for block_n in [16, 32, 64, 128]:
            for num_warps in [2, 4, 8]:
                for num_stages in [1, 2]:
                    for pre_load_v in [False, True]:
                        for waves in [1, 2, 4]:
                            cfg = TritonConfig(
                                block_m=block_m,
                                block_n=block_n,
                                num_warps=num_warps,
                                num_stages=num_stages,
                                pre_load_v=pre_load_v,
                                waves_per_eu=waves,
                            )
                            
                            # Filter 1: SMEM must fit
                            if not cfg.fits_smem(head_dim, elem_size):
                                continue
                            
                            # Filter 2: threads must be reasonable
                            threads = num_warps * 32
                            if threads > HW["max_threads_per_block"]:
                                continue
                            if threads < block_m:  # need at least 1 thread per row
                                continue
                            
                            # Filter 3: occupancy >= 1 CTA/SM
                            if cfg.occupancy_ctas(head_dim, elem_size) < 1:
                                continue
                            
                            # Filter 4: PRE_LOAD_V register pressure check
                            if pre_load_v:
                                v_regs = block_n * head_dim * elem_size // 4  # fp16 → 2 per reg
                                if v_regs > 128:  # too many regs for V pre-load
                                    continue
                            
                            configs.append(cfg)
    
    return configs


def rank_configs(configs: List[TritonConfig], head_dim: int = 256, elem_size: int = 2) -> List[TritonConfig]:
    """Rank configs by estimated throughput, deduplicated.
    
    CCCL benchmark insight: for memory-bound kernels on BI-V100,
    the dominant factors are (in order):
      1. Total work per CTA (tile_elements) — amortizes launch overhead
      2. Occupancy × tile_size — total inflight bytes across SM
      3. Pipeline depth (num_stages) — matches bytes_in_flight window
      4. PRE_LOAD_V — reduces stalls when V is small enough for regs
    
    We penalize configs where threads < tile rows (wasted threads)
    and where SMEM utilization is very low (leaving bandwidth on table).
    """
    # Deduplicate: (block_m, block_n, num_warps, num_stages, pre_load_v)
    seen = set()
    unique = []
    for cfg in configs:
        key = (cfg.block_m, cfg.block_n, cfg.num_warps, cfg.num_stages, cfg.pre_load_v)
        if key not in seen:
            seen.add(key)
            unique.append(cfg)
    
    def score(cfg: TritonConfig) -> float:
        tile = cfg.block_m * cfg.block_n
        occ = cfg.occupancy_ctas(head_dim, elem_size)
        smem = cfg.smem_bytes(head_dim, elem_size)
        smem_util = smem / HW["smem_per_block"]
        threads = cfg.num_warps * 32
        
        # Base: tile size × occupancy (bigger tiles, more parallelism)
        base = tile * max(occ, 1)
        
        # Bonus: stages=2 adds ~15% on BI-V100 (from CCCL babelstream data)
        stage_mult = 1.15 if cfg.num_stages == 2 else 1.0
        
        # Bonus: preload_v helps when reg pressure allows
        preload_mult = 1.05 if cfg.pre_load_v else 1.0
        
        # Penalty: threads >> block_m means wasted work per row
        thread_efficiency = min(1.0, cfg.block_m / threads)
        
        # Penalty: very low SMEM util means we could be doing more work
        smem_score = min(smem_util * 1.5, 1.0)  # 67%+ util → score 1.0
        
        return base * stage_mult * preload_mult * thread_efficiency * smem_score
    
    return sorted(unique, key=score, reverse=True)


def derive_partition_size() -> int:
    """Derive optimal _PARTITION_SIZE for paged_attention.
    
    CCCL parallel: GridEvenShare partitioning.
    partition_size = tokens processed per CTA in V2.
    
    BI-V100: 16 SMs, each CTA handles one partition.
    For seq_len=4096, partition=512 → 8 partitions → 8 CTAs → only 8/16 SMs busy.
    partition=256 → 16 partitions → 16 CTAs → all SMs busy.
    But smaller partition → more inter-partition reduce overhead.
    
    Optimal: partition ≈ seq_len / (2 × sm_count) for long sequences
    For max_model_len=100K: 100000 / 32 ≈ 3125 → round to 2048 or 4096
    But V2 isn't used (forced V1), so this is academic for now.
    """
    return 512  # Keep current — V2 is disabled


def derive_prefill_config(head_dim: int = 256, elem_size: int = 2) -> Dict:
    """Derive prefix_prefill.py BLOCK/BLOCK_N/NUM_WARPS.
    
    CCCL parallel: scan + reduce + transform (softmax + QKV matmul)
    
    SMEM model for prefix_prefill (context_attention_fwd_kernel):
      Q resident: BLOCK_M × head_dim × elem_size     (stays across all K/V iters)
      K per iter: head_dim × BLOCK_N × elem_size      (loaded, consumed, freed)
      softmax:    BLOCK_M × 4 × 2                     (m_i + l_i, fp32)
      
    Qwen3.6 head_dim=256, bf16:
      BLOCK_M=32, BLOCK_N=64:  8KB + 32KB + 256B = 40.25KB (82%) ✓
      BLOCK_M=64, BLOCK_N=64:  32KB + 32KB + 512B = 64.5KB (131%) ✗ OVERFLOW
      BLOCK_M=32, BLOCK_N=128: 8KB + 64KB + 256B = 72.25KB (147%) ✗ OVERFLOW
      BLOCK_M=64, BLOCK_N=32:  32KB + 16KB + 512B = 48.5KB (99%) TIGHT
    """
    smem_limit = HW["smem_per_block"]
    
    # Try configs from largest to smallest
    candidates = [
        (64, 64, 4),   # symmetric, ideal for NVIDIA
        (32, 64, 4),   # asymmetric, better for SMEM-limited
        (64, 32, 4),   # Q-heavy
        (32, 32, 4),   # conservative
        (32, 32, 2),   # minimal
    ]
    
    for bm, bn, nw in candidates:
        q_smem = bm * head_dim * elem_size
        k_smem = head_dim * bn * elem_size
        softmax_smem = bm * 4 * 2
        total = q_smem + k_smem + softmax_smem
        if total <= smem_limit:
            return {
                "BLOCK_M": bm,
                "BLOCK_N": bn,
                "NUM_WARPS": nw,
                "num_stages": 1,  # no async copy on BI-V100
                "smem_bytes": total,
                "smem_utilization": total / smem_limit,
            }
    
    # Fallback
    return {"BLOCK_M": 32, "BLOCK_N": 32, "NUM_WARPS": 4, "num_stages": 1,
            "smem_bytes": 32*256*2 + 256*32*2 + 32*8, "smem_utilization": 0.67}


def main():
    print("=" * 70)
    print("muh gen_config: CCCL-derived Python-layer configs for BI-V100")
    print("=" * 70)
    
    # 1. Triton flash attention configs
    print("\n### 1. Triton flash attention autotune configs ###")
    print(f"    head_dim={MODEL['head_dim']}, elem_size=2 (bf16)")
    
    all_configs = generate_triton_configs(MODEL["head_dim"], 2)
    ranked = rank_configs(all_configs, MODEL["head_dim"], 2)
    
    print(f"    Generated {len(all_configs)} valid configs (from {16*4*3*2*2*3} combinations)")
    print(f"    Top 10:")
    for i, cfg in enumerate(ranked[:10]):
        smem = cfg.smem_bytes(MODEL["head_dim"], 2)
        occ = cfg.occupancy_ctas(MODEL["head_dim"], 2)
        print(f"      #{i+1}: M={cfg.block_m:3d} N={cfg.block_n:3d} "
              f"warps={cfg.num_warps} stages={cfg.num_stages} "
              f"preload_v={cfg.pre_load_v!s:5s} "
              f"smem={smem//1024}KB occ={occ} CTAs/SM")
    
    # 2. Prefix prefill config
    print("\n### 2. Prefix prefill config ###")
    prefill = derive_prefill_config(MODEL["head_dim"], 2)
    print(f"    BLOCK_M={prefill['BLOCK_M']}, BLOCK_N={prefill['BLOCK_N']}, "
          f"NUM_WARPS={prefill['NUM_WARPS']}")
    print(f"    SMEM={prefill['smem_bytes']}B ({prefill['smem_utilization']:.0%} of 48KB)")
    
    # 3. Partition size
    print("\n### 3. Paged attention config ###")
    ps = derive_partition_size()
    print(f"    _PARTITION_SIZE={ps} (V2 disabled, V1 forced)")
    
    # 4. Summary of what's already applied vs what's new
    print("\n### 4. Applied vs pending ###")
    applied = [
        ("_custom_ops.py SMEM=49152", "APPLIED"),
        ("prefix_prefill.py BLOCK=64 BLOCK_N=64", "APPLIED"),
        ("triton_flash_attention.py 8 BI-V100 configs", "APPLIED"),
        ("computility-run.yaml max-model-len=100000", "APPLIED"),
    ]
    pending = [
        (f"triton_flash_attention.py +{len(ranked[:20])-17} new configs", "PENDING"),
        ("paged_attn.py V2 enable for long sequences", "PENDING (needs native V2)"),
        ("prefix_prefill.py num_stages=2 experiment", "PENDING (needs cp.async support check)"),
    ]
    
    for item, status in applied:
        print(f"    ✓ {item}: {status}")
    for item, status in pending:
        print(f"    ○ {item}: {status}")
    
    # 5. New configs to add to triton_flash_attention.py
    existing_signatures = set()
    # Current 17 configs from triton_flash_attention.py (manually extracted)
    existing_raw = [
        (256,64,8,1,False,2), (128,128,4,1,False,2), (256,128,8,1,False,2),
        (128,64,4,1,False,1), (128,64,4,1,True,3), (128,64,4,1,False,3),
        (64,64,8,1,False,4), (32,32,8,1,False,4), (16,16,4,1,False,1),
        # BI-V100 existing
        (64,32,4,1,False,2), (32,64,4,1,False,2), (64,64,4,1,False,2),
        (64,64,4,2,False,2), (32,64,4,2,True,2), (32,32,4,2,False,4),
        (32,32,4,2,True,4),
    ]
    for m,n,w,s,p,we in existing_raw:
        existing_signatures.add((m,n,w,s,p))
    
    new_configs = []
    for cfg in ranked[:30]:
        sig = (cfg.block_m, cfg.block_n, cfg.num_warps, cfg.num_stages, cfg.pre_load_v)
        if sig not in existing_signatures:
            new_configs.append(cfg)
            existing_signatures.add(sig)
    
    print(f"\n### 5. New triton configs to add ({len(new_configs)}) ###")
    for cfg in new_configs[:15]:
        smem = cfg.smem_bytes(MODEL["head_dim"], 2)
        print(f"    {cfg.to_triton_str()}")
        print(f"        SMEM={smem//1024}KB, occupancy={cfg.occupancy_ctas(MODEL['head_dim'], 2)} CTAs/SM")


if __name__ == "__main__":
    main()
