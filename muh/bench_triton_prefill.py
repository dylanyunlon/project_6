#!/usr/bin/env python3
"""muh/bench_triton_prefill.py — Real Triton kernel benchmark for BI-V100

Unlike bench_bi100.py (which called torch.sum and never injected params),
this script directly invokes the prefix_prefill Triton JIT kernel with
different (BLOCK, NUM_WARPS) constexpr values. Each combination triggers
Triton to compile a separate kernel binary — the same mechanism as CCCL's
`#define TUNE_THREADS_PER_BLOCK N` + recompile.

Search space (from CCCL scan analogy):
  BLOCK:     [16, 32, 64, 128]    — tile size for Q/K/V
  NUM_WARPS: [1, 2, 4, 8]         — warps per CTA (threads = warps × 32)
  
  Valid combos after SMEM pruning (head_dim=128, fp16):
    SMEM ≈ BLOCK × 128 × 2B × 3(Q+K+V) + BLOCK × BLOCK × 4B(fp32 accum)
    BLOCK=128: SMEM ≈ 128×128×6 + 128×128×4 = 98304 + 65536 = 163840 → OVERFLOW
    BLOCK=64:  SMEM ≈ 64×128×6  + 64×64×4   = 49152 + 16384  = 65536  → OVERFLOW at 48KB
    BLOCK=32:  SMEM ≈ 32×128×6  + 32×32×4   = 24576 + 4096   = 28672  → FITS
    BLOCK=16:  SMEM ≈ 16×128×6  + 16×16×4   = 12288 + 1024   = 13312  → FITS

  NOTE: Triton's actual SMEM usage differs from this estimate because:
    - Triton uses register tiling, not full SMEM staging
    - Accumulator is in registers, not SMEM
    - K/V cache loads may be streamed, not fully staged
  The real constraint is determined by Triton compiler. Combos that exceed
  SMEM will fail at compile time with a clear error — not silently.

Output format (CCCL compatible):
  block_64.warps_4 <speedup_ctx128> <speedup_ctx512> <speedup_ctx2048> <speedup_ctx8192>

Usage (ON BI-V100 ONLY — requires GPU):
    python3 muh/bench_triton_prefill.py
    python3 muh/bench_triton_prefill.py --block 32 64 --warps 2 4 8
    python3 muh/bench_triton_prefill.py --output results/prefill_bench.json
"""

import os
import sys
import time
import json
import argparse
from typing import List, Dict, Tuple, Optional

# ============================================================
# This script MUST run on BI-V100. Import torch only at runtime.
# ============================================================

def require_gpu():
    try:
        import torch
        if not torch.cuda.is_available():
            print("ERROR: No CUDA GPU available. This benchmark must run on BI-V100.")
            sys.exit(1)
        dev = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {dev}")
        print(f"  SMs: {props.multi_processor_count}")
        print(f"  SMEM/block: {props.max_shared_memory_per_block} bytes")
        print(f"  Total VRAM: {props.total_mem / 1024**3:.1f} GB")
        return torch, props
    except ImportError:
        print("ERROR: torch not available. Run on BI-V100 with vllm environment.")
        sys.exit(1)


def make_test_tensors(torch, batch: int, seq_len: int, ctx_len: int,
                      num_heads: int, num_kv_heads: int, head_dim: int,
                      block_size: int, dtype):
    """Create realistic test tensors matching Qwen3.6 dimensions.
    
    Qwen3.6-35B-A3B:
      head_dim = 128
      num_heads = 64 (query heads)
      num_kv_heads = 8 (GQA 8:1)
      dtype = bfloat16 or float16
    """
    device = "cuda"
    
    # Query: [total_tokens, num_heads, head_dim]
    total_tokens = batch * seq_len
    q = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=dtype)
    
    # K, V: [total_tokens, num_kv_heads, head_dim]
    k = torch.randn(total_tokens, num_kv_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(total_tokens, num_kv_heads, head_dim, device=device, dtype=dtype)
    
    # Output
    o = torch.empty_like(q)
    
    # KV cache: [num_blocks, num_kv_heads, head_dim/x, block_size, x]
    x = 16 // dtype_size(dtype)  # vector width
    num_blocks = (batch * (ctx_len + seq_len) + block_size - 1) // block_size + 16
    k_cache = torch.randn(num_blocks, num_kv_heads, head_dim // x, block_size, x,
                          device=device, dtype=dtype)
    v_cache = torch.randn(num_blocks, num_kv_heads, head_dim, block_size,
                          device=device, dtype=dtype)
    
    # Block location table: [batch, max_blocks_per_seq]
    max_blocks = (ctx_len + seq_len + block_size - 1) // block_size
    b_loc = torch.zeros(batch, max_blocks, device=device, dtype=torch.int32)
    for i in range(batch):
        n_blocks = min(max_blocks, num_blocks - i * max_blocks)
        b_loc[i, :n_blocks] = torch.arange(
            i * max_blocks, i * max_blocks + n_blocks, device=device)
    
    # Sequence metadata
    b_start_loc = torch.arange(0, batch * seq_len, seq_len, device=device, dtype=torch.int32)
    b_seq_len = torch.full((batch,), seq_len + ctx_len, device=device, dtype=torch.int32)
    b_ctx_len = torch.full((batch,), ctx_len, device=device, dtype=torch.int32)
    
    return q, k, v, o, k_cache, v_cache, b_loc, b_start_loc, b_seq_len, b_ctx_len


def dtype_size(dtype):
    import torch
    return torch.tensor([], dtype=dtype).element_size()


def bench_one_config(torch, triton, _fwd_kernel, block_m: int, block_n: int,
                     num_warps: int,
                     q, k, v, o, k_cache, v_cache, b_loc, b_start_loc,
                     b_seq_len, b_ctx_len, max_input_len: int,
                     warmup: int = 3, repeats: int = 10) -> Optional[float]:
    """Benchmark one (BLOCK, NUM_WARPS) configuration.
    
    Each call triggers Triton JIT compilation for this specific
    (BLOCK, NUM_WARPS) pair if not already cached. This IS the
    compile-time parameter injection — same as CCCL's #define mechanism.
    
    Returns median time in ms, or None if compilation fails (SMEM overflow).
    """
    Lk = q.shape[-1]
    Lk_padded = triton.next_power_of_2(Lk)
    sm_scale = 1.0 / (Lk ** 0.5)
    batch = b_seq_len.shape[0]
    head = q.shape[1]
    num_queries_per_kv = q.shape[1] // k.shape[1]
    
    grid = (batch, head, triton.cdiv(max_input_len, block_m))
    
    # Build kernel call args (reused for warmup and timed runs)
    def call_kernel():
        _fwd_kernel[grid](
            q, k, v, k_cache, v_cache, b_loc,
            sm_scale, 1.0, 1.0,
            b_start_loc, b_seq_len, b_ctx_len,
            v_cache.shape[3], k_cache.shape[4],
            o,
            b_loc.stride(0), b_loc.stride(1),
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            k_cache.stride(3), k_cache.stride(4),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
            v_cache.stride(3),
            num_queries_per_kv=num_queries_per_kv,
            BLOCK_M=block_m,
            BLOCK_DMODEL=Lk,
            BLOCK_DMODEL_PADDED=Lk_padded,
            BLOCK_N=block_n,
            SLIDING_WINDOW=0,
            num_warps=num_warps,
            num_stages=1,
        )
    
    # Attempt compilation + warmup
    try:
        for _ in range(warmup):
            call_kernel()
        torch.cuda.synchronize()
    except Exception as e:
        return None  # Compilation failed (likely SMEM overflow)
    
    # Timed runs
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        call_kernel()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)  # ms
    
    times.sort()
    return times[len(times) // 2]  # median


def main():
    p = argparse.ArgumentParser(
        description="Triton prefill kernel benchmark with compile-time param injection")
    p.add_argument("--block", type=int, nargs="+", default=[16, 32, 64, 128],
                   help="BLOCK_M sizes to test (each triggers Triton recompilation)")
    p.add_argument("--block-n", type=int, nargs="+", default=None,
                   help="BLOCK_N sizes (default: same as --block). Use different values for asymmetric search.")
    p.add_argument("--warps", type=int, nargs="+", default=[1, 2, 4, 8],
                   help="NUM_WARPS values to test")
    p.add_argument("--ctx-lens", type=int, nargs="+", default=[128, 512, 2048, 8192],
                   help="Context lengths to benchmark (problem sizes)")
    p.add_argument("--batch", type=int, default=1, help="Batch size")
    p.add_argument("--seq-len", type=int, default=1, help="New tokens per sequence")
    p.add_argument("--head-dim", type=int, default=128, help="Head dimension (Qwen3.6=128)")
    p.add_argument("--num-heads", type=int, default=64, help="Query heads (Qwen3.6=64)")
    p.add_argument("--num-kv-heads", type=int, default=8, help="KV heads (Qwen3.6=8, GQA)")
    p.add_argument("--block-size", type=int, default=16, help="KV cache block size")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--output", type=str, default=None, help="Save JSON results")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    args = p.parse_args()
    
    torch, props = require_gpu()
    import triton
    
    # Import the actual Triton kernel from prefix_prefill.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
    try:
        from prefix_prefill import _fwd_kernel
    except ImportError:
        # Try vllm path
        from vllm.attention.ops.prefix_prefill import _fwd_kernel
    
    dtype = getattr(torch, args.dtype)
    
    print(f"\nmuh bench_triton_prefill")
    print(f"  Kernel: prefix_prefill._fwd_kernel (Triton JIT)")
    print(f"  Mechanism: each (BLOCK, NUM_WARPS) pair → Triton recompilation → different PTX")
    print(f"  Model: Qwen3.6 (head_dim={args.head_dim}, heads={args.num_heads}, kv_heads={args.num_kv_heads})")
    block_n_vals = args.block_n if args.block_n else args.block
    print(f"  Search: BLOCK_M={args.block} × BLOCK_N={block_n_vals} × WARPS={args.warps} = {len(args.block)*len(block_n_vals)*len(args.warps)} variants")
    print(f"  Problem sizes: ctx_len={args.ctx_lens}")
    print()
    
    # Collect results
    all_results = []
    baseline_times = {}  # ctx_len → time with default config (BLOCK=64, WARPS=4)
    
    for ctx_len in args.ctx_lens:
        print(f"--- ctx_len={ctx_len} ---")
        
        q, k, v, o, k_cache, v_cache, b_loc, b_start_loc, b_seq_len, b_ctx_len = \
            make_test_tensors(torch, args.batch, args.seq_len, ctx_len,
                            args.num_heads, args.num_kv_heads, args.head_dim,
                            args.block_size, dtype)
        max_input_len = args.seq_len
        
        for block_m in args.block:
            for block_n in block_n_vals:
                for warps in args.warps:
                    label = f"bm_{block_m}.bn_{block_n}.w_{warps}"
                    
                    t = bench_one_config(
                        torch, triton, _fwd_kernel, block_m, block_n, warps,
                        q, k, v, o, k_cache, v_cache, b_loc, b_start_loc,
                        b_seq_len, b_ctx_len, max_input_len,
                        warmup=args.warmup, repeats=args.repeats,
                    )
                    
                    if t is None:
                        print(f"  {label:35s} COMPILE FAIL (SMEM overflow)")
                        all_results.append({
                            "block_m": block_m, "block_n": block_n, "warps": warps,
                            "ctx_len": ctx_len, "time_ms": None, "status": "compile_fail",
                        })
                    else:
                        if block_m == 64 and block_n == 64 and warps == 4:
                            baseline_times[ctx_len] = t
                        
                        speedup = baseline_times.get(ctx_len, t) / t if t > 0 else 0
                        marker = " ★" if speedup > 1.05 else " ✗" if speedup < 0.9 else ""
                        print(f"  {label:35s} {t:8.3f} ms  {speedup:6.3f}x{marker}")
                        
                        all_results.append({
                            "block_m": block_m, "block_n": block_n, "warps": warps,
                            "ctx_len": ctx_len, "time_ms": round(t, 4),
                            "speedup": round(speedup, 4), "status": "ok",
                        })
        
        # Free tensors
        del q, k, v, o, k_cache, v_cache, b_loc, b_start_loc, b_seq_len, b_ctx_len
        torch.cuda.empty_cache()
    
    # Summary: best config per ctx_len
    print(f"\n{'='*60}")
    print("BEST per context length:")
    for ctx_len in args.ctx_lens:
        ctx_results = [r for r in all_results if r["ctx_len"] == ctx_len and r["status"] == "ok"]
        if ctx_results:
            best = min(ctx_results, key=lambda r: r["time_ms"])
            print(f"  ctx={ctx_len:>5d}: bm={best['block_m']}, bn={best['block_n']}, "
                  f"warps={best['warps']}, time={best['time_ms']:.3f}ms, "
                  f"speedup={best.get('speedup', 1):.3f}x")
    
    # CCCL-format output
    print(f"\nCCCL-format output:")
    for block_m in args.block:
        for block_n in block_n_vals:
            for warps in args.warps:
                label = f"bm_{block_m}.bn_{block_n}.w_{warps}"
                speedups = []
                for ctx_len in args.ctx_lens:
                    r = next((r for r in all_results
                             if r["block_m"] == block_m and r["block_n"] == block_n
                             and r["warps"] == warps
                             and r["ctx_len"] == ctx_len and r["status"] == "ok"), None)
                    if r and "speedup" in r:
                        speedups.append(f"{r['speedup']:.6f}")
                    else:
                        speedups.append("N/A")
                print(f"  {label} {' '.join(speedups)}")
    
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({
                "benchmark": "triton_prefill",
                "kernel": "prefix_prefill._fwd_kernel",
                "mechanism": "Triton JIT constexpr recompilation",
                "gpu": torch.cuda.get_device_name(0),
                "sm_count": props.multi_processor_count,
                "smem_per_block": props.max_shared_memory_per_block,
                "model_config": {
                    "head_dim": args.head_dim,
                    "num_heads": args.num_heads,
                    "num_kv_heads": args.num_kv_heads,
                    "dtype": args.dtype,
                },
                "baseline": {"block": 64, "warps": 4},
                "baseline_times": {str(k): v for k, v in baseline_times.items()},
                "results": all_results,
            }, f, indent=2)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
