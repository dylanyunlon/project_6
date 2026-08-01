#!/usr/bin/env python3
"""
muh SMEM safety validator — tests all 26 tuning algorithms against 48KB limit.

Reimplements each policy_selector's SMEM calculation in Python and verifies
that no (algorithm, type_size) combination exceeds BI-V100's 49152-byte limit.

Usage:
    python3 test_smem_safety.py [--verbose]

Exit code:
    0 = all safe
    1 = at least one overflow detected
"""

import sys
import csv
import io

MAX_SMEM = 49152  # BI-V100: 48 KiB
WARP_SIZE = 32

def scale_mem_bound(nom_threads, nom_items, type_size):
    """Python mirror of muh::tuning::scale_mem_bound (common.cuh)"""
    items = nom_items * 4 // type_size
    items = max(1, min(items, nom_items * 2))
    smem_per_thread = type_size * items
    if smem_per_thread > 0:
        raw = MAX_SMEM // smem_per_thread
        max_threads = ((raw + 31) // 32) * 32
    else:
        max_threads = nom_threads
    threads = min(nom_threads, max_threads)
    if threads < 32:
        threads = 32
    return items, threads

def clamp_items(threads, items, elem_size, limit=MAX_SMEM, multiplier=1):
    """Reduce items until tile fits SMEM"""
    while threads * items * elem_size * multiplier > limit and items > 1:
        items -= 1
    return items


# ============================================================
# Policy selectors for all 26 algorithms
# ============================================================

def reduce_policy(type_size):
    items, threads = scale_mem_bound(512, 16, type_size)
    return threads, items, threads * items * type_size

def scan_policy(type_size):
    items, threads = scale_mem_bound(512, 22, type_size)
    return threads, items, threads * items * type_size

def topk_policy(key_size):
    bits = 8  # BI-V100: always 8
    threads = 512
    items = 4
    return threads, items, threads * items * key_size  # tile only, histogram is separate

def transform_policy(type_size):
    items, threads = scale_mem_bound(256, 16, type_size)
    return threads, items, threads * items * type_size

def batch_memcpy_policy(type_size):
    return 256, 4, 256 * 4 * type_size

def for_policy(type_size):
    return 256, 4, 256 * 4 * type_size

def adjacent_difference_policy(type_size):
    items = max(1, 7 * 8 // type_size)
    return 128, items, 128 * items * type_size

def find_policy(type_size):
    items, threads = scale_mem_bound(128, 16, type_size)
    return threads, items, threads * items * type_size

def find_bound_policy(type_size):
    return 256, 8, 256 * 8 * type_size

def segmented_reduce_policy(type_size):
    # Delegates to reduce
    return reduce_policy(type_size)

def segmented_scan_policy(type_size):
    tuple_size = type_size + max(type_size, 4)  # conservative: tuple<AccumT, bool>
    items, threads = scale_mem_bound(128, 9, tuple_size)
    return threads, items, threads * items * tuple_size

def merge_policy(type_size):
    items = max(1, 15 * 4 // type_size)
    items = clamp_items(256, items, type_size)
    return 256, items, 256 * items * type_size

def merge_sort_policy(type_size):
    items, threads = scale_mem_bound(256, 11, type_size)
    return threads, items, threads * items * type_size

def transform_tile_policy(type_size):
    tile = max(128, 16384 // type_size)
    return 256, tile // 256, tile * type_size  # approximate

def batched_topk_policy(key_size):
    bits = 8  # BI-V100: always 8
    buckets = 1 << bits
    hist_smem = buckets * 4
    max_batches = min(32, MAX_SMEM // hist_smem)
    return 512, 4, 512 * 4 * key_size + hist_smem * max_batches

def segmented_radix_sort_policy(key_size, value_size=0):
    pair_size = key_size + value_size
    items = max(1, 16 // key_size)
    items = clamp_items(256, items, pair_size)
    return 256, items, 256 * items * pair_size

def histogram_policy(type_size, max_bins=256):
    priv_bins = min(max_bins, MAX_SMEM // (4 * 8))
    return 384, 12, priv_bins * 4 * 8  # privatized bins SMEM

def rle_encode_policy(item_size, length_size=4):
    items = 14 if item_size < 4 else (10 if item_size < 8 else 7)
    pair_size = item_size + length_size
    items = clamp_items(256, items, pair_size)
    return 256, items, 256 * items * pair_size

def rle_non_trivial_runs_policy(item_size, offset_size=4):
    items = 14 if item_size < 4 else (10 if item_size < 8 else 7)
    pair_size = item_size + offset_size
    items = clamp_items(320, items, pair_size)
    return 320, items, 320 * items * pair_size

def three_way_partition_policy(key_size, value_size=0):
    pair_size = key_size + value_size
    if pair_size <= 2:    threads, items = 384, 20
    elif pair_size <= 4:  threads, items = 384, 18
    elif pair_size <= 8:  threads, items = 256, 14
    else:                 threads, items = 192, 10
    items = clamp_items(threads, items, pair_size)
    return threads, items, threads * items * pair_size

def segmented_sort_policy(key_size, value_size=0):
    pair_size = key_size + value_size
    items, threads = scale_mem_bound(256, 11, pair_size)
    return threads, items, threads * items * pair_size

def reduce_by_key_policy(key_size, accum_size):
    pair_size = key_size + accum_size
    if pair_size <= 4:    threads, items = 320, 16
    elif pair_size <= 8:  threads, items = 256, 14
    else:                 threads, items = 192, 10
    items = clamp_items(threads, items, pair_size)
    return threads, items, threads * items * pair_size

def scan_by_key_policy(key_size, accum_size):
    pair_size = key_size + accum_size
    if pair_size <= 4:    threads, items = 320, 18
    elif pair_size <= 8:  threads, items = 256, 14
    else:                 threads, items = 192, 10
    items = clamp_items(threads, items, pair_size)
    return threads, items, threads * items * pair_size

def select_if_policy(input_size, flag_size=0, may_alias=False):
    elem_size = input_size
    has_flags = flag_size > 0
    if has_flags:
        if elem_size <= 2:    threads, items = 384, 18
        elif elem_size <= 4:  threads, items = 320, 14
        elif elem_size <= 8:  threads, items = 256, 10
        else:                 threads, items = 192, 7
    else:
        if elem_size <= 2:    threads, items = 384, 22
        elif elem_size <= 4:  threads, items = 384, 18
        elif elem_size <= 8:  threads, items = 256, 14
        else:                 threads, items = 192, 9
    
    smem_in = threads * items * elem_size
    smem_out = threads * items * elem_size
    smem_flags = threads * items if has_flags else 0
    total = smem_in + smem_out + smem_flags
    while total > MAX_SMEM and items > 1:
        items -= 1
        total = threads * items * elem_size * 2 + (threads * items if has_flags else 0)
    return threads, items, total

def unique_by_key_policy(key_size, value_size):
    pair_size = key_size + value_size
    if pair_size <= 4:    threads, items = 320, 16
    elif pair_size <= 8:  threads, items = 256, 12
    else:                 threads, items = 192, 8
    items = clamp_items(threads, items, pair_size)
    return threads, items, threads * items * pair_size

def radix_sort_policy(key_size, value_size=0, keys_only=True):
    bits = 8
    items = max(1, 16 // key_size)
    threads = 256
    keys_tile = threads * items * key_size
    val_tile = 0 if keys_only else threads * items * value_size
    offsets = (1 << bits) * 8
    rank_smem = (1 << bits) * 4
    main_union = max(keys_tile, val_tile, rank_smem)
    total = main_union + offsets
    headroom = 2048
    while total > MAX_SMEM - headroom and items > 1:
        items -= 1
        keys_tile = threads * items * key_size
        val_tile = 0 if keys_only else threads * items * value_size
        main_union = max(keys_tile, val_tile, rank_smem)
        total = main_union + offsets
    return threads, items, total


# ============================================================
# Test runner
# ============================================================

def run_tests(verbose=False):
    type_sizes = [1, 2, 4, 8, 16]
    results = []
    failures = 0
    
    # Simple algorithms: (algo_name, policy_fn, uses_type_size)
    simple_algos = [
        ("reduce", reduce_policy),
        ("scan", scan_policy),
        ("transform", transform_policy),
        ("batch_memcpy", batch_memcpy_policy),
        ("for", for_policy),
        ("adjacent_difference", adjacent_difference_policy),
        ("find", find_policy),
        ("find_bound", find_bound_policy),
        ("segmented_reduce", segmented_reduce_policy),
        ("segmented_scan", segmented_scan_policy),
        ("merge", merge_policy),
        ("merge_sort", merge_sort_policy),
        ("transform_tile", transform_tile_policy),
    ]
    
    for algo, fn in simple_algos:
        for ts in type_sizes:
            threads, items, smem = fn(ts)
            safe = smem <= MAX_SMEM
            if not safe: failures += 1
            results.append((algo, ts, 0, threads, items, smem, safe))
            if verbose or not safe:
                status = "✓" if safe else "✗ OVERFLOW"
                print(f"  {status} {algo:30s} ts={ts:2d}  t={threads:4d} i={items:3d} smem={smem:6d}")
    
    # topk + batched_topk (key_size only)
    for ts in [1, 2, 4, 8]:
        threads, items, smem = topk_policy(ts)
        safe = smem <= MAX_SMEM
        if not safe: failures += 1
        results.append(("topk", ts, 0, threads, items, smem, safe))
        
        threads, items, smem = batched_topk_policy(ts)
        safe = smem <= MAX_SMEM
        if not safe: failures += 1
        results.append(("batched_topk", ts, 0, threads, items, smem, safe))
    
    # Pair algorithms: (algo, fn, key_sizes, value_sizes)
    pair_algos = [
        ("segmented_radix_sort", segmented_radix_sort_policy, [1,2,4,8], [0,4,8]),
        ("three_way_partition", three_way_partition_policy, [1,2,4,8], [0,4,8]),
        ("segmented_sort", segmented_sort_policy, [1,2,4,8], [0,4,8]),
        ("reduce_by_key", reduce_by_key_policy, [2,4,8], [2,4,8]),
        ("scan_by_key", scan_by_key_policy, [2,4,8], [2,4,8]),
        ("unique_by_key", unique_by_key_policy, [2,4,8], [2,4,8]),
    ]
    
    for algo, fn, ks, vs in pair_algos:
        for k in ks:
            for v in vs:
                threads, items, smem = fn(k, v)
                safe = smem <= MAX_SMEM
                if not safe: failures += 1
                results.append((algo, k, v, threads, items, smem, safe))
                if verbose or not safe:
                    status = "✓" if safe else "✗ OVERFLOW"
                    print(f"  {status} {algo:30s} k={k:2d} v={v:2d}  t={threads:4d} i={items:3d} smem={smem:6d}")
    
    # histogram
    for ts in type_sizes:
        for bins in [256, 1024, 4096]:
            threads, items, smem = histogram_policy(ts, bins)
            safe = smem <= MAX_SMEM
            if not safe: failures += 1
            results.append((f"histogram(bins={bins})", ts, 0, threads, items, smem, safe))
    
    # rle
    for ts in [1, 2, 4, 8]:
        threads, items, smem = rle_encode_policy(ts)
        safe = smem <= MAX_SMEM
        if not safe: failures += 1
        results.append(("rle_encode", ts, 4, threads, items, smem, safe))
        
        threads, items, smem = rle_non_trivial_runs_policy(ts)
        safe = smem <= MAX_SMEM
        if not safe: failures += 1
        results.append(("rle_non_trivial_runs", ts, 4, threads, items, smem, safe))
    
    # select_if: 3 dimensions
    for ts in [1, 2, 4, 8, 16]:
        for may_alias in [False, True]:
            for flag_size in [0, 1]:
                threads, items, smem = select_if_policy(ts, flag_size, may_alias)
                safe = smem <= MAX_SMEM
                if not safe: failures += 1
                label = f"select_if(alias={'Y' if may_alias else 'N'},flags={'Y' if flag_size else 'N'})"
                results.append((label, ts, flag_size, threads, items, smem, safe))
                if verbose or not safe:
                    status = "✓" if safe else "✗ OVERFLOW"
                    print(f"  {status} {label:45s} ts={ts:2d}  t={threads:4d} i={items:3d} smem={smem:6d}")
    
    # radix_sort
    for k in [1, 2, 4, 8]:
        for v in [0, 4, 8]:
            threads, items, smem = radix_sort_policy(k, v, v == 0)
            safe = smem <= MAX_SMEM - 2048
            if not safe: failures += 1
            results.append(("radix_sort", k, v, threads, items, smem, safe))
            if verbose or not safe:
                status = "✓" if safe else "✗ OVERFLOW"
                print(f"  {status} radix_sort k={k:2d} v={v:2d}  t={threads:4d} i={items:3d} smem={smem:6d} (limit={MAX_SMEM-2048})")
    
    # Summary
    total = len(results)
    safe_count = sum(1 for r in results if r[6])
    print(f"\n{'='*60}")
    print(f"SMEM Safety: {safe_count}/{total} combinations safe")
    if failures:
        print(f"FAILURES: {failures} combinations exceed {MAX_SMEM} bytes")
    else:
        print(f"✓ ALL SAFE — no combination exceeds {MAX_SMEM} bytes")
    
    return failures == 0


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    ok = run_tests(verbose=verbose)
    sys.exit(0 if ok else 1)
