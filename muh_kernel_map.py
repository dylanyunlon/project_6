#!/usr/bin/env python3
"""muh/dispatch.py — Runtime policy dispatch for vllm kernel configuration

This is the core of the muh competitive moat.

CCCL's policy_selector is a compile-time C++ template that maps:
    (type_t, op_kind_t, accum_size, offset_size, compute_capability)
    → (threads_per_block, items_per_thread, vec_size, load_algorithm, ...)

vllm doesn't use CUB directly — it uses PyTorch/Triton/custom CUDA kernels.
But those kernels have the SAME tuning dimensions:
    - BLOCK_SIZE (= threads_per_block)
    - NUM_WARPS (= threads_per_block / 32)
    - PARTITION_SIZE (= threads_per_block * items_per_thread)
    - TILE_SIZE for shared memory

This module provides a Python-side policy_selector that:
1. Reads bi100_* values from C++ headers (via gen_patch.extract_bi100_structs)
2. Maps CCCL algorithm→vllm kernel paths (the INJECTION_POINTS)
3. Applies SMEM constraints for BI-V100 (48KB limit)
4. Outputs the concrete values to inject into vllm source

The moat is NOT the parameter values (anyone can benchmark those).
The moat is:
    a) Knowing WHICH 7 dimensions to search (from CCCL's policy structs)
    b) Knowing the CONSTRAINTS (SMEM ≤ 48KB, occupancy, L2 coherence delay)
    c) Knowing WHERE in vllm each algorithm appears (the injection mapping)
    d) Having the infrastructure to iterate: benchmark → update header → gen_patch → rebuild
"""

import os
import sys
import json

# Add parent dir for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_patch import extract_bi100_structs, algo_from_filename

# ──────────────────────────────────────────────────────────────
# BI-V100 hardware constraints (from hardware.cuh)
# These are the hard limits that make our tuning values different
# from every other GPU — and why copy-pasting SM100 values crashes.
# ──────────────────────────────────────────────────────────────

BI_V100 = {
    "warp_size": 32,
    "max_threads_per_block": 1024,
    "max_shared_memory_per_block": 49152,  # 48 KiB
    "max_registers_per_thread": 255,
    "l2_cache_size_bytes": 6 * 1024 * 1024,  # 6 MiB
    "memory_bandwidth_gbps": 900,
    "sm_count": 50,
    # Derived
    "bandwidth_per_sm_gbps": 900 / 50,  # 18 GB/s ≈ A100 level
}

SM100 = {
    "max_shared_memory_per_block": 49152,  # same default, but can configure higher
    "l2_cache_size_bytes": 50 * 1024 * 1024,  # 50 MiB
    "memory_bandwidth_gbps": 8000,
    "sm_count": 148,
    "bandwidth_per_sm_gbps": 8000 / 148,  # 54 GB/s
}

# ──────────────────────────────────────────────────────────────
# SMEM constraint checker
# This is the single most important function in muh.
# Every bi100_* struct MUST pass this check or the kernel will crash.
# ──────────────────────────────────────────────────────────────

def check_smem(threads: int, items: int, elem_bytes: int,
               smem_limit: int = BI_V100["max_shared_memory_per_block"]) -> dict:
    """Check if a tile fits in shared memory.
    
    Returns dict with:
        tile_bytes: actual shared memory usage
        fits: True if tile_bytes <= smem_limit  
        utilization: tile_bytes / smem_limit (higher = more efficient but riskier)
        max_items: maximum items_per_thread that fits
    """
    tile_bytes = threads * items * elem_bytes
    max_items = smem_limit // (threads * elem_bytes) if threads * elem_bytes > 0 else 0
    return {
        "tile_bytes": tile_bytes,
        "fits": tile_bytes <= smem_limit,
        "utilization": tile_bytes / smem_limit if smem_limit > 0 else 0,
        "max_items": max_items,
        "overflow_bytes": max(0, tile_bytes - smem_limit),
    }


def scale_mem_bound(nominal_4B_threads: int, nominal_4B_items: int,
                    type_size: int) -> tuple:
    """Scale items and threads for a given type size, matching CCCL exactly.
    
    Mirrors cub::detail::scale_mem_bound() from util_arch.cuh lines 153-161.
    Returns (items_per_thread, threads_per_block) — items-first, matching
    CCCL's scaling_result struct field order.
    
    Three differences from the old muh version (all were bugs):
    1. Return order: (items, threads) not (threads, items)
    2. Items clamp upper bound: nominal * 2, not nominal * 1
       (CCCL allows small types like char to double items_per_thread)
    3. Threads SMEM cap: min(nominal, round_up(max_smem/(type*items), 32))
       (prevents launching more threads than SMEM can feed)
    
    Verified against all 18 CCCL test cases in catch2_test_util_arch.cu.
    """
    MAX_SMEM = 48 * 1024  # 49152 bytes, hardcoded in CCCL as max_smem_per_block
    
    # Step 1: scale items inversely with type size
    items = nominal_4B_items * 4 // type_size
    items = max(1, min(items, nominal_4B_items * 2))  # clamp: [1, 2*nominal]
    
    # Step 2: cap threads by SMEM constraint
    # round_up(x, 32) aligns to warp boundary
    smem_per_item = type_size * items
    if smem_per_item > 0:
        max_threads_by_smem = ((MAX_SMEM // smem_per_item + 31) // 32) * 32
    else:
        max_threads_by_smem = nominal_4B_threads
    threads = min(nominal_4B_threads, max_threads_by_smem)
    
    return (items, threads)  # items-first, matching CCCL scaling_result


def scale_delay_for_l2(sm100_delay_ns: int, sm100_l2w: int) -> tuple:
    """Scale lookback delay parameters for BI-V100's smaller L2.
    
    SM100 L2 = 50MB, BI-V100 L2 = 6MB (8.3x smaller).
    Smaller L2 → faster coherence → shorter delays needed.
    Heuristic: ns *= 0.5, l2w *= 0.6 (to be refined by benchmark).
    """
    bi100_ns = int(sm100_delay_ns * 0.5)
    bi100_l2w = int(sm100_l2w * 0.6)
    return (bi100_ns, bi100_l2w)


# ──────────────────────────────────────────────────────────────
# vllm kernel → CCCL algorithm mapping
#
# This is the strategic knowledge that makes CCCL useful for vllm.
# Each entry maps a vllm kernel file to:
#   - The CCCL algorithm it implements (reduce, scan, sort, etc.)
#   - The data types it operates on (determines which bi100_* struct to use)
#   - The tuning dimensions that appear in the kernel code
#
# Built from reading:
#   - paged_attn.py (PagedAttention V1/V2 dispatch)
#   - prefix_prefill.py (Triton/PyTorch context attention)
#   - vllm/model_executor/layers/sampler.py (top-k/top-p)
#   - paged_attention_kernel_architecture.md (CCCL pattern mapping)
# ──────────────────────────────────────────────────────────────

VLLM_KERNEL_MAP = {
    # === DECODE HOT PATH (Output TPS × 16.796 = 83%) ===
    
    "paged_attention_v1": {
        "cccl_algorithms": ["reduce"],
        "description": "Single-pass decode attention for seq_len ≤ 8192",
        "data_types": {
            "query": "float16",   # Q: [num_seqs, num_heads, head_dim]
            "key_cache": "float16",  # K: [num_blocks, num_kv_heads, head_dim//x, block_size, x]
            "score": "float32",   # QK^T intermediate: always fp32 for precision
            "output": "float16",  # weighted V sum
        },
        "tuning_dimensions": {
            "NUM_THREADS": {"cccl_field": "threads_per_block", "range": [128, 256, 512]},
            "NUM_WARPS": {"derived_from": "NUM_THREADS / 32"},
            "_PARTITION_SIZE": {"value": 512, "note": "hardcoded in paged_attn.py, affects V2 threshold"},
        },
        "cccl_pattern": "compound reduce: summary_statistics.cu binary op pattern",
        "smem_formula": "NUM_THREADS * head_dim * sizeof(float) + head_dim * block_size * sizeof(half) * 2",
    },
    
    "paged_attention_v2": {
        "cccl_algorithms": ["reduce", "scan"],
        "description": "Two-pass partitioned attention for seq_len > 8192",
        "data_types": {
            "score": "float32",
            "exp_sum": "float32",
            "max_logits": "float32",
        },
        "tuning_dimensions": {
            "NUM_THREADS": {"cccl_field": "threads_per_block"},
            "PARTITION_SIZE": {"cccl_field": "threads_per_block * items_per_thread"},
        },
        "cccl_pattern": "reduce pass 1 (per-partition) + reduce pass 2 (cross-partition merge)",
    },
    
    "context_attention_fwd": {
        "cccl_algorithms": ["scan", "reduce", "transform"],
        "description": "Prefill attention (Triton kernel, bypassed on BI-V100)",
        "status": "BYPASSED — Triton hangs BI-V100, using _forward_prefix_pytorch",
        "tuning_dimensions": {
            "BLOCK_M": {"value": 64, "note": "query tile"},
            "BLOCK_N": {"value": 64, "note": "KV tile"},
            "BLOCK_DMODEL": {"value": 256, "note": "head_dim, must match model"},
        },
        "note": "PyTorch fallback has no tunable block sizes — optimization comes from algorithmic changes (K-tiling)",
    },
    
    "sampling_topk": {
        "cccl_algorithms": ["topk", "radix_sort"],
        "description": "Top-k token selection from logits",
        "data_types": {
            "logits": "float32",   # [batch, vocab_size=152064]
            "indices": "int32",
        },
        "tuning_dimensions": {
            "BLOCK_SIZE": {"cccl_field": "threads_per_block"},
            "RADIX_BITS": {"cccl_field": "bits_per_pass"},
        },
    },
    
    "activation_kernels": {
        "cccl_algorithms": ["transform"],
        "description": "SiLU, GELU, element-wise activations",
        "data_types": {"input": "float16", "output": "float16"},
        "tuning_dimensions": {
            "BLOCK_SIZE": {"cccl_field": "threads_per_block"},
            "VEC_SIZE": {"cccl_field": "vec_size"},
        },
    },
    
    "layernorm_kernels": {
        "cccl_algorithms": ["reduce", "transform"],
        "description": "RMSNorm / LayerNorm: reduce for variance, transform for normalize",
        "data_types": {"input": "float16", "accum": "float32"},
        "tuning_dimensions": {
            "BLOCK_SIZE": {"cccl_field": "threads_per_block"},
        },
    },
    
    "rotary_embedding": {
        "cccl_algorithms": ["for_each", "transform"],
        "description": "RoPE position encoding",
        "data_types": {"input": "float16"},
        "tuning_dimensions": {
            "BLOCK_SIZE": {"cccl_field": "threads_per_block"},
        },
    },
    
    # === CACHE PATH (Cache TPS × 0.56 = 3%) ===
    
    "cache_kernels": {
        "cccl_algorithms": ["batch_memcpy"],
        "description": "KV cache block copy/swap operations",
        "data_types": {"kv_cache": "float16"},
        "tuning_dimensions": {
            "BLOCK_SIZE": {"cccl_field": "threads_per_block"},
        },
    },
}


# ──────────────────────────────────────────────────────────────
# Policy dispatch: given a vllm kernel, return optimal BI-V100 config
# ──────────────────────────────────────────────────────────────

def dispatch_policy(kernel_name: str, tuning_headers_dir: str = "muh/include/muh/tuning") -> dict:
    """Given a vllm kernel name, return the optimal BI-V100 tuning parameters.
    
    This is the Python equivalent of CCCL's policy_selector::operator()().
    It reads the C++ headers, applies SMEM constraints, and returns
    the concrete values to inject into the vllm kernel.
    """
    if kernel_name not in VLLM_KERNEL_MAP:
        return {"error": f"Unknown kernel: {kernel_name}"}
    
    kernel_info = VLLM_KERNEL_MAP[kernel_name]
    cccl_algos = kernel_info["cccl_algorithms"]
    
    result = {
        "kernel": kernel_name,
        "description": kernel_info.get("description", ""),
        "policies": {},
        "smem_checks": [],
    }
    
    for algo in cccl_algos:
        header_path = os.path.join(tuning_headers_dir, f"tuning_{algo}.cuh")
        if algo == "for_each":
            header_path = os.path.join(tuning_headers_dir, "tuning_for.cuh")
            
        if not os.path.exists(header_path):
            result["policies"][algo] = {"status": "NO_HEADER", "fallback": "CCCL_DEFAULT"}
            continue
        
        structs = extract_bi100_structs(header_path)
        if not structs:
            result["policies"][algo] = {"status": "NO_BI100_STRUCTS"}
            continue
        
        # Select the most relevant struct for this kernel's data types
        algo_policies = {}
        for name, fields in structs:
            # Check SMEM constraint
            threads = fields.get("threads", fields.get("threads_per_block", 256))
            items = fields.get("items", fields.get("items_per_thread", 16))
            
            # Determine element size from kernel data types
            elem_bytes = 4  # default to float32
            if "float16" in str(kernel_info.get("data_types", {}).values()):
                elem_bytes = 2
            if "score" in kernel_info.get("data_types", {}):
                elem_bytes = 4  # scores are always fp32
            
            smem = check_smem(threads, items, elem_bytes)
            algo_policies[name] = {**fields, "_smem_check": smem}
            
            if not smem["fits"]:
                result["smem_checks"].append({
                    "struct": name,
                    "OVERFLOW": True,
                    "tile_bytes": smem["tile_bytes"],
                    "limit": BI_V100["max_shared_memory_per_block"],
                    "max_safe_items": smem["max_items"],
                })
        
        result["policies"][algo] = algo_policies
    
    return result


def dispatch_all(tuning_headers_dir: str = "muh/include/muh/tuning") -> dict:
    """Dispatch policies for ALL vllm kernels. Used by gen_patch."""
    results = {}
    for kernel_name in VLLM_KERNEL_MAP:
        results[kernel_name] = dispatch_policy(kernel_name, tuning_headers_dir)
    return results


# ──────────────────────────────────────────────────────────────
# CLI: dump all dispatch results for inspection
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="muh policy dispatch for vllm kernels")
    p.add_argument("--headers", default="muh/include/muh/tuning")
    p.add_argument("--kernel", default=None, help="Specific kernel to dispatch")
    p.add_argument("--json", action="store_true", help="JSON output")
    args = p.parse_args()
    
    if args.kernel:
        result = dispatch_policy(args.kernel, args.headers)
    else:
        result = dispatch_all(args.headers)
    
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        for kernel_name, policy in (result.items() if isinstance(result, dict) and "kernel" not in result else [(result.get("kernel","?"), result)]):
            if isinstance(policy, dict) and "kernel" in policy:
                kernel_name = policy["kernel"]
            print(f"\n{'='*60}")
            print(f"Kernel: {kernel_name}")
            if isinstance(policy, dict):
                print(f"  Description: {policy.get('description','')}")
                for algo, algo_policy in policy.get("policies", {}).items():
                    print(f"  [{algo}]:")
                    if isinstance(algo_policy, dict) and "status" in algo_policy:
                        print(f"    {algo_policy}")
                    elif isinstance(algo_policy, dict):
                        for struct_name, fields in algo_policy.items():
                            smem = fields.pop("_smem_check", {})
                            print(f"    {struct_name}: {fields}")
                            if smem:
                                status = "✓" if smem.get("fits") else "✗ OVERFLOW"
                                print(f"      SMEM: {smem.get('tile_bytes',0)} bytes ({status})")
                for check in policy.get("smem_checks", []):
                    print(f"  ⚠ SMEM OVERFLOW: {check}")
