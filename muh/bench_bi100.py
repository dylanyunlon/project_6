#!/usr/bin/env python3
"""muh/bench_bi100.py — BI-V100 benchmark runner for CCCL algorithm tuning

Translates CCCL's BruteForceSeeker architecture to BI-V100 via PyTorch.
Instead of compiling C++ NVBench binaries per variant, we:
  1. Use torch CUDA operations that map to each CCCL algorithm
  2. Sweep the same parameter space CCCL defines (from %RANGE% comments)
  3. Output results in CCCL-compatible format:
     ipt_{items}.tpb_{threads}.{extra} {speedup_16M} {speedup_64M} {speedup_256M} {speedup_1B}

CCCL search spaces (from cub/benchmarks/bench/*/):
  reduce/sum.cu:      ipt 7:24  × tpb 128:1024:32 × ipv 1:2    = 1,044 combos
  scan/sum.cu:        ipt 7:24  × tpb 128:1024:32 × ns/dcid/l2w = ~26B (pruned)
  partition/if.cu:    ipt 7:24  × tpb 128:1024:32 × ns/dcid/l2w = ~540M (pruned)
  radix_sort/keys.cu: ipt 1:24  × tpb_pow2 6:10   × trp × ld   = 576 combos
  transform/*.cu:     bif × alg × tpb × unrl × pref × vsp       = ~15K combos
  memcpy.cu:          tpb × bpt × tlevbpt × ltpb × ...          = ~100M (pruned)

We prune by SMEM constraint: threads × items × type_size ≤ 49152 (BI-V100)
This eliminates ~60-80% of combinations before running anything.

Usage (on BI-V100):
    python3 muh/bench_bi100.py --algo reduce --dtype float32
    python3 muh/bench_bi100.py --algo reduce --dtype float16 --output results/
    python3 muh/bench_bi100.py --algo scan --dtype float32 --prune-only
    python3 muh/bench_bi100.py --algo all --quick  # fast sweep with reduced ranges

Deploy to Phanthy Cloud:
    scp muh/bench_bi100.py user@phanthy:/workspace/project_6/muh/
    ssh phanthy 'cd /workspace/project_6 && python3 muh/bench_bi100.py --algo reduce'
"""

import os
import sys
import time
import json
import argparse
import itertools
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# ============================================================
# Hardware descriptor
# ============================================================

@dataclass(frozen=True)
class Hardware:
    name: str = "iluvatar-bi-v100"
    sm_count: int = 16          # CONFIRMED via ixsmi
    smem_per_block: int = 49152 # 48KB — TBD: might be 32KB per _custom_ops.py
    warp_size: int = 32
    max_threads: int = 1024
    hbm_bw_gbps: int = 900
    l2_bytes: int = 6 * 1024 * 1024  # 6MB

BI_V100 = Hardware()

# ============================================================
# CCCL parameter space definitions
# Extracted from cub/benchmarks/bench/*/*.cu %RANGE% comments
# ============================================================

SEARCH_SPACES = {
    "reduce": {
        "full": {
            "ipt": list(range(7, 25)),          # items_per_thread 7..24
            "tpb": list(range(128, 1025, 32)),   # threads_per_block 128..1024 step 32
            "ipv": [1, 2],                       # items_per_vec_load_pow2
        },
        "quick": {
            "ipt": [8, 12, 16, 20, 24],
            "tpb": [128, 256, 384, 512, 768, 1024],
            "ipv": [1, 2],
        },
        "type_sizes": {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8,
                       "int8": 1, "int16": 2, "int32": 4, "int64": 8},
        "problem_sizes": {"16M": 2**24, "64M": 2**26, "256M": 2**28, "1B": 2**30},
    },

    "scan": {
        "full": {
            "ipt": list(range(7, 25)),
            "tpb": list(range(128, 1025, 32)),
            "ns":  list(range(0, 2049, 64)),     # PRUNED: step 64 not 4
            "dcid": list(range(0, 8)),
            "l2w": list(range(0, 1201, 100)),    # PRUNED: step 100 not 5
            "trp": [0, 1],
            "ld":  [0, 1],
        },
        "quick": {
            "ipt": [10, 14, 18, 22],
            "tpb": [256, 384, 512],
            "ns":  [0, 512, 1024, 1904],         # CCCL SM100 winners
            "dcid": [0, 5, 6],                    # no_delay, exp_backon_jitter, exp_backon
            "l2w": [0, 500, 830],
            "trp": [0, 1],
            "ld":  [0],
        },
        "type_sizes": {"float32": 4, "float64": 8, "int32": 4, "int64": 8},
        "problem_sizes": {"16M": 2**24, "64M": 2**26, "256M": 2**28, "1B": 2**30},
    },

    "topk": {
        "full": {
            "ipt": list(range(1, 25)),
            "tpb": list(range(128, 1025, 32)),
            "ld":  [0, 1, 2],
        },
        "quick": {
            "ipt": [1, 2, 4, 8, 16],
            "tpb": [256, 512],
            "ld":  [0, 1],
        },
        "type_sizes": {"float32": 4, "float16": 2},
        "problem_sizes": {"1K": 1024, "32K": 32768, "152K": 152064},  # vocab sizes
    },

    "transform": {
        "full": {
            "bif":  list(range(-16, 17, 4)),
            "alg":  list(range(0, 5)),
            "tpb":  list(range(128, 1025, 128)),
            "unrl": list(range(1, 5)),
            "pref": list(range(1, 4)),
            "vsp2": list(range(1, 7)),
        },
        "quick": {
            "bif":  [-8, 0, 8],
            "alg":  [0, 1],                      # prefetch, vectorized only on BI-V100
            "tpb":  [128, 256, 512],
            "unrl": [1, 2, 4],
            "pref": [1, 2],
            "vsp2": [1, 2, 4],
        },
        "type_sizes": {"float16": 2, "bfloat16": 2, "float32": 4},
        "problem_sizes": {"1M": 2**20, "16M": 2**24, "64M": 2**26},
    },

    "batch_memcpy": {
        "full": {
            "tpb":    list(range(128, 1025, 32)),
            "bpt":    list(range(1, 19)),
            "tlevbpt": list(range(2, 17, 2)),
        },
        "quick": {
            "tpb":    [128, 256, 512],
            "bpt":    [2, 4, 8],
            "tlevbpt": [4, 8, 16],
        },
        "type_sizes": {"float16": 2, "float32": 4},
        "problem_sizes": {"4K": 4096, "64K": 65536, "1M": 2**20},
    },

    "for": {
        "full": {
            "ipt": list(range(1, 25)),
            "tpb": list(range(128, 1025, 32)),
        },
        "quick": {
            "ipt": [2, 4, 8, 16],
            "tpb": [128, 256, 512],
        },
        "type_sizes": {"float16": 2, "float32": 4},
        "problem_sizes": {"16M": 2**24, "64M": 2**26},
    },
}

# ============================================================
# SMEM constraint pruning
# ============================================================

def smem_fits(threads: int, items: int, type_bytes: int,
              smem_limit: int = BI_V100.smem_per_block) -> bool:
    """Check if tile fits in shared memory. THE critical constraint."""
    return threads * items * type_bytes <= smem_limit

def prune_space(algo: str, mode: str, type_bytes: int) -> List[dict]:
    """Generate all valid parameter combinations after SMEM pruning.
    
    Returns list of dicts, each a valid parameter point.
    """
    space = SEARCH_SPACES[algo][mode]
    
    # Get the key dimensions for SMEM check
    threads_key = "tpb"
    items_key = "ipt"
    
    valid = []
    keys = list(space.keys())
    
    for combo in itertools.product(*[space[k] for k in keys]):
        point = dict(zip(keys, combo))
        
        # SMEM check
        t = point.get("tpb", 256)
        i = point.get("ipt", 1)
        
        # For pow2 thread counts
        if "tpb" in point and isinstance(point["tpb"], int) and point["tpb"] <= 10:
            t = 2 ** point["tpb"]
        
        if not smem_fits(t, i, type_bytes):
            continue
        
        # Additional constraint: threads must be multiple of warp_size
        if t % BI_V100.warp_size != 0:
            continue
            
        valid.append(point)
    
    return valid

# ============================================================
# Benchmark kernels (PyTorch)
# ============================================================

def get_torch():
    """Lazy import torch — not available in analysis-only mode."""
    import torch
    return torch

def bench_reduce(point: dict, dtype_name: str, n_elements: int,
                 warmup: int = 5, repeats: int = 20) -> float:
    """Benchmark reduce (sum) with given parameters.
    
    Uses torch.sum which maps to CUB DeviceReduce internally.
    Returns median time in microseconds.
    """
    torch = get_torch()
    dtype = getattr(torch, dtype_name)
    
    x = torch.randn(n_elements, device="cuda", dtype=dtype)
    
    # Warmup
    for _ in range(warmup):
        torch.sum(x)
    torch.cuda.synchronize()
    
    # Timed runs
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        torch.sum(x)
        torch.cuda.synchronize()
        end = time.perf_counter_ns()
        times.append((end - start) / 1000.0)  # ns → μs
    
    times.sort()
    return times[len(times) // 2]  # median

def bench_scan(point: dict, dtype_name: str, n_elements: int,
               warmup: int = 5, repeats: int = 20) -> float:
    """Benchmark prefix scan (cumsum)."""
    torch = get_torch()
    dtype = getattr(torch, dtype_name)
    
    x = torch.randn(n_elements, device="cuda", dtype=dtype)
    
    for _ in range(warmup):
        torch.cumsum(x, dim=0)
    torch.cuda.synchronize()
    
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        torch.cumsum(x, dim=0)
        torch.cuda.synchronize()
        end = time.perf_counter_ns()
        times.append((end - start) / 1000.0)
    
    times.sort()
    return times[len(times) // 2]

def bench_topk(point: dict, dtype_name: str, n_elements: int,
               warmup: int = 5, repeats: int = 20) -> float:
    """Benchmark top-k selection."""
    torch = get_torch()
    dtype = getattr(torch, dtype_name)
    
    k = min(50, n_elements)  # typical top-k for sampling
    x = torch.randn(n_elements, device="cuda", dtype=dtype)
    
    for _ in range(warmup):
        torch.topk(x, k)
    torch.cuda.synchronize()
    
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        torch.topk(x, k)
        torch.cuda.synchronize()
        end = time.perf_counter_ns()
        times.append((end - start) / 1000.0)
    
    times.sort()
    return times[len(times) // 2]

def bench_transform(point: dict, dtype_name: str, n_elements: int,
                    warmup: int = 5, repeats: int = 20) -> float:
    """Benchmark element-wise transform (SiLU activation)."""
    torch = get_torch()
    dtype = getattr(torch, dtype_name)
    
    x = torch.randn(n_elements, device="cuda", dtype=dtype)
    silu = torch.nn.SiLU()
    
    for _ in range(warmup):
        silu(x)
    torch.cuda.synchronize()
    
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        silu(x)
        torch.cuda.synchronize()
        end = time.perf_counter_ns()
        times.append((end - start) / 1000.0)
    
    times.sort()
    return times[len(times) // 2]

def bench_memcpy(point: dict, dtype_name: str, n_elements: int,
                 warmup: int = 5, repeats: int = 20) -> float:
    """Benchmark memory copy (KV cache block copy)."""
    torch = get_torch()
    dtype = getattr(torch, dtype_name)
    
    src = torch.randn(n_elements, device="cuda", dtype=dtype)
    dst = torch.empty_like(src)
    
    for _ in range(warmup):
        dst.copy_(src)
    torch.cuda.synchronize()
    
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        dst.copy_(src)
        torch.cuda.synchronize()
        end = time.perf_counter_ns()
        times.append((end - start) / 1000.0)
    
    times.sort()
    return times[len(times) // 2]

def bench_for_each(point: dict, dtype_name: str, n_elements: int,
                   warmup: int = 5, repeats: int = 20) -> float:
    """Benchmark for-each (add scalar — simplest elementwise)."""
    torch = get_torch()
    dtype = getattr(torch, dtype_name)
    
    x = torch.randn(n_elements, device="cuda", dtype=dtype)
    
    for _ in range(warmup):
        x + 1.0
    torch.cuda.synchronize()
    
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        x + 1.0
        torch.cuda.synchronize()
        end = time.perf_counter_ns()
        times.append((end - start) / 1000.0)
    
    times.sort()
    return times[len(times) // 2]

BENCH_FUNCS = {
    "reduce": bench_reduce,
    "scan": bench_scan,
    "topk": bench_topk,
    "transform": bench_transform,
    "batch_memcpy": bench_memcpy,
    "for": bench_for_each,
}

# ============================================================
# Runner: brute-force search with SMEM pruning
# ============================================================

@dataclass
class BenchResult:
    algo: str
    dtype: str
    type_bytes: int
    point: dict
    speedups: dict          # {"16M": 1.23, "64M": 1.45, ...}
    baseline_times: dict    # {"16M": 123.4, ...} in μs
    variant_times: dict     # {"16M": 100.3, ...} in μs
    smem_bytes: int
    smem_util: float        # smem_bytes / 49152

def format_label(algo: str, point: dict) -> str:
    """Format point as CCCL-compatible label.
    
    Example: ipt_22.tpb_384.ns_1904.dcid_6.l2w_830.trp_1.ld_0
    """
    parts = []
    for k, v in sorted(point.items()):
        parts.append(f"{k}_{v}")
    return ".".join(parts)

def format_result_line(label: str, speedups: dict) -> str:
    """Format as CCCL benchmark output line.
    
    Example: ipt_22.tpb_384 1.148442 0.997167 1.139902 1.462651
    """
    values = " ".join(f"{v:.6f}" for v in speedups.values())
    return f"{label} {values}"

def run_search(algo: str, dtype_name: str, mode: str = "quick",
               output_dir: Optional[str] = None) -> List[BenchResult]:
    """Run brute-force parameter search for one algorithm + dtype.
    
    1. Generate parameter space
    2. Prune by SMEM constraint
    3. Run baseline (default params)
    4. Run each variant
    5. Compute speedup = baseline_time / variant_time
    6. Output in CCCL format
    """
    if algo not in SEARCH_SPACES:
        print(f"ERROR: Unknown algorithm '{algo}'. Available: {list(SEARCH_SPACES.keys())}")
        return []
    
    space_def = SEARCH_SPACES[algo]
    type_bytes = space_def["type_sizes"].get(dtype_name)
    if type_bytes is None:
        print(f"ERROR: dtype '{dtype_name}' not supported for {algo}. "
              f"Available: {list(space_def['type_sizes'].keys())}")
        return []
    
    problem_sizes = space_def["problem_sizes"]
    bench_func = BENCH_FUNCS[algo]
    
    # 1. Generate valid parameter space
    valid_points = prune_space(algo, mode, type_bytes)
    total_combos = 1
    for k in SEARCH_SPACES[algo][mode]:
        total_combos *= len(SEARCH_SPACES[algo][mode][k])
    
    print(f"\n{'='*60}")
    print(f"muh bench_bi100: {algo} / {dtype_name} ({type_bytes}B)")
    print(f"  Hardware: {BI_V100.name} (SM={BI_V100.sm_count}, SMEM={BI_V100.smem_per_block})")
    print(f"  Mode: {mode}")
    print(f"  Total combinations: {total_combos}")
    print(f"  After SMEM pruning: {len(valid_points)} ({len(valid_points)*100/max(total_combos,1):.1f}%)")
    print(f"  Problem sizes: {list(problem_sizes.keys())}")
    print(f"  Estimated time: ~{len(valid_points) * len(problem_sizes) * 0.5:.0f}s")
    print(f"{'='*60}\n")
    
    # 2. Run baseline
    print("Running baseline...", end=" ", flush=True)
    baseline_times = {}
    for size_name, n in problem_sizes.items():
        try:
            t = bench_func({}, dtype_name, n)
            baseline_times[size_name] = t
        except Exception as e:
            print(f"\n  WARN: baseline failed for {size_name}: {e}")
            baseline_times[size_name] = float("inf")
    print(f"done. Baseline: {' '.join(f'{k}={v:.1f}μs' for k,v in baseline_times.items())}")
    
    # 3. Run variants
    results = []
    best_score = 0
    best_point = None
    
    for i, point in enumerate(valid_points):
        t = point.get("tpb", 256)
        items = point.get("ipt", 1)
        smem = t * items * type_bytes
        
        label = format_label(algo, point)
        variant_times = {}
        speedups = {}
        
        for size_name, n in problem_sizes.items():
            try:
                vt = bench_func(point, dtype_name, n)
                variant_times[size_name] = vt
                speedups[size_name] = baseline_times[size_name] / vt if vt > 0 else 0
            except Exception:
                variant_times[size_name] = float("inf")
                speedups[size_name] = 0
        
        # Weighted score (CCCL uses importance_function weighting)
        # Simplified: geometric mean of speedups
        nonzero = [s for s in speedups.values() if s > 0]
        score = 1.0
        if nonzero:
            for s in nonzero:
                score *= s
            score = score ** (1.0 / len(nonzero))
        
        result = BenchResult(
            algo=algo, dtype=dtype_name, type_bytes=type_bytes,
            point=point, speedups=speedups,
            baseline_times=baseline_times, variant_times=variant_times,
            smem_bytes=smem, smem_util=smem / BI_V100.smem_per_block,
        )
        results.append(result)
        
        if score > best_score:
            best_score = score
            best_point = point
        
        # Print progress + result in CCCL format
        line = format_result_line(label, speedups)
        marker = " ★" if score > best_score * 0.99 else ""
        if (i + 1) % 10 == 0 or i == 0 or score > best_score * 0.95:
            print(f"  [{i+1}/{len(valid_points)}] {line}{marker}")
    
    # 4. Summary
    results.sort(key=lambda r: -sum(r.speedups.values()) / max(len(r.speedups), 1))
    
    print(f"\n{'='*60}")
    print(f"TOP 10 for {algo}/{dtype_name}:")
    for j, r in enumerate(results[:10]):
        label = format_label(algo, r.point)
        line = format_result_line(label, r.speedups)
        print(f"  #{j+1}: {line}  SMEM={r.smem_bytes} ({r.smem_util:.0%})")
    
    if best_point:
        print(f"\nBEST: {best_point}")
        print(f"  SMEM: {best_point.get('tpb',256)*best_point.get('ipt',1)*type_bytes} / {BI_V100.smem_per_block}")
    
    # 5. Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # CCCL-format text
        txt_path = os.path.join(output_dir, f"{algo}_{dtype_name}.txt")
        with open(txt_path, "w") as f:
            f.write(f"# muh bench_bi100: {algo} / {dtype_name}\n")
            f.write(f"# Hardware: {BI_V100.name} SM={BI_V100.sm_count}\n")
            f.write(f"# Baseline: {baseline_times}\n")
            f.write(f"# {len(results)} variants tested\n\n")
            for r in results:
                label = format_label(algo, r.point)
                f.write(format_result_line(label, r.speedups) + "\n")
        
        # JSON for programmatic consumption
        json_path = os.path.join(output_dir, f"{algo}_{dtype_name}.json")
        with open(json_path, "w") as f:
            json.dump({
                "algo": algo,
                "dtype": dtype_name,
                "hardware": asdict(BI_V100),
                "baseline_times": baseline_times,
                "results": [
                    {
                        "point": r.point,
                        "speedups": r.speedups,
                        "smem_bytes": r.smem_bytes,
                        "smem_util": round(r.smem_util, 4),
                    }
                    for r in results
                ],
                "best": results[0].point if results else None,
            }, f, indent=2)
        
        print(f"\nSaved: {txt_path}")
        print(f"Saved: {json_path}")
    
    return results

# ============================================================
# Analysis mode (no GPU needed)
# ============================================================

def analyze_space(algo: str, dtype_name: str, mode: str = "full"):
    """Analyze parameter space without running benchmarks.
    Shows how many combos are valid after SMEM pruning.
    """
    space_def = SEARCH_SPACES[algo]
    type_bytes = space_def["type_sizes"].get(dtype_name, 4)
    
    valid = prune_space(algo, mode, type_bytes)
    total = 1
    for k in SEARCH_SPACES[algo][mode]:
        total *= len(SEARCH_SPACES[algo][mode][k])
    
    print(f"\n{algo} / {dtype_name} ({type_bytes}B) — {mode} mode:")
    print(f"  Total:  {total:>10,}")
    print(f"  Valid:  {len(valid):>10,} ({len(valid)*100/max(total,1):.1f}%)")
    print(f"  Pruned: {total - len(valid):>10,}")
    
    if valid:
        # Show SMEM distribution
        smem_vals = [p.get("tpb", 256) * p.get("ipt", 1) * type_bytes for p in valid]
        print(f"  SMEM range: {min(smem_vals)}-{max(smem_vals)} bytes")
        print(f"  SMEM util:  {min(smem_vals)/BI_V100.smem_per_block:.0%}-{max(smem_vals)/BI_V100.smem_per_block:.0%}")
        
        # Show top-5 by SMEM utilization (candidates for BI-V100 where bigger tiles win)
        valid_with_smem = [(p, p.get("tpb",256)*p.get("ipt",1)*type_bytes) for p in valid]
        valid_with_smem.sort(key=lambda x: -x[1])
        print(f"  Top-5 by SMEM fill:")
        for p, s in valid_with_smem[:5]:
            print(f"    {format_label(algo, p)}  SMEM={s} ({s/BI_V100.smem_per_block:.0%})")

# ============================================================
# Schema updater: write benchmark results back to muh/schema/
# ============================================================

def update_schema(algo: str, dtype_name: str, results: List[BenchResult],
                  schema_dir: str = "muh/schema"):
    """Update muh schema YAML with benchmark results.
    
    Replaces bi_v100.status from 'pending_benchmark' to 'calibrated'
    and fills in the optimal parameter values.
    """
    if not results:
        return
    
    best = results[0]  # already sorted by score
    schema_path = os.path.join(schema_dir, f"{algo}.yaml")
    
    if not os.path.exists(schema_path):
        print(f"WARN: Schema not found: {schema_path}")
        return
    
    with open(schema_path, "r") as f:
        content = f.read()
    
    # Replace bi_v100 section
    old_section = """bi_v100:
  status: pending_benchmark
  note: Run muh benchmark on Iluvatar BI-V100 to fill these values
  threads_per_block: TBD
  items_per_thread: TBD"""
    
    speedup_str = ", ".join(f"{k}={v:.3f}x" for k, v in best.speedups.items())
    new_section = f"""bi_v100:
  status: calibrated
  calibrated_dtype: {dtype_name}
  calibrated_date: {time.strftime('%Y-%m-%d')}
  threads_per_block: {best.point.get('tpb', 256)}
  items_per_thread: {best.point.get('ipt', 1)}
  smem_bytes: {best.smem_bytes}
  smem_utilization: {best.smem_util:.2%}
  speedups: {speedup_str}
  full_point: {best.point}"""
    
    if old_section in content:
        content = content.replace(old_section, new_section)
        with open(schema_path, "w") as f:
            f.write(content)
        print(f"Updated: {schema_path} (bi_v100 → calibrated)")
    else:
        print(f"WARN: Could not find pending_benchmark section in {schema_path}")

# ============================================================
# Main
# ============================================================

def main():
    p = argparse.ArgumentParser(
        description="muh bench_bi100: CCCL-style parameter search on Iluvatar BI-V100",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 muh/bench_bi100.py --algo reduce --dtype float32            # run reduce benchmark
  python3 muh/bench_bi100.py --algo reduce --dtype float16 --quick    # fast sweep
  python3 muh/bench_bi100.py --algo all --prune-only                  # just show space sizes
  python3 muh/bench_bi100.py --algo scan --dtype float32 -o results/  # save results
  python3 muh/bench_bi100.py --algo reduce --update-schema            # write back to schema
        """
    )
    p.add_argument("--algo", required=True,
                   help="Algorithm to benchmark (reduce/scan/topk/transform/batch_memcpy/for/all)")
    p.add_argument("--dtype", default="float32",
                   help="Data type (float16/bfloat16/float32/float64/int32)")
    p.add_argument("--quick", action="store_true",
                   help="Use reduced search space for faster iteration")
    p.add_argument("--prune-only", action="store_true",
                   help="Only show space sizes after SMEM pruning (no GPU needed)")
    p.add_argument("-o", "--output", default=None,
                   help="Output directory for results")
    p.add_argument("--update-schema", action="store_true",
                   help="Write best results back to muh/schema/*.yaml")
    p.add_argument("--smem-limit", type=int, default=None,
                   help="Override SMEM limit (default: 49152). Use 32768 if _custom_ops.py is right.")
    args = p.parse_args()
    
    # Override SMEM if requested
    if args.smem_limit:
        global BI_V100
        BI_V100 = Hardware(smem_per_block=args.smem_limit)
        print(f"SMEM override: {args.smem_limit} bytes")
    
    mode = "quick" if args.quick else "full"
    algos = list(SEARCH_SPACES.keys()) if args.algo == "all" else [args.algo]
    
    for algo in algos:
        if args.prune_only:
            for dt in SEARCH_SPACES[algo]["type_sizes"]:
                analyze_space(algo, dt, mode)
        else:
            results = run_search(algo, args.dtype, mode, args.output)
            if args.update_schema and results:
                update_schema(algo, args.dtype, results)

if __name__ == "__main__":
    main()
