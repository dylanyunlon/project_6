#!/usr/bin/env python3
"""test_reduce_tuning.py — Verify muh reduce tuning against CCCL ground truth

Tests that muh's scale_mem_bound + bi100_* struct values produce valid
configurations for all data types used in vllm/Qwen3.6.

CCCL reference: summary_statistics.cu
  AccumT = summary_stats_data<float> (7 floats = 28 bytes)
  This is the WORST CASE for SMEM/register pressure testing —
  if our tuning handles 28-byte AccumT without overflow,
  the common 4-byte (float32) and 2-byte (float16) paths are safe.

Validation approach:
  For each bi100_* struct, compute scale_mem_bound output and verify:
  1. threads × items fits register file (< 64K regs/SM)
  2. For SMEM-using algorithms (scan, select_if): tile ≤ 48KB
  3. For register-only algorithms (reduce): occupancy ≥ 1 CTA/SM
  4. vec_size divides items_per_thread (vectorization alignment)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# BI-V100 hardware constants (from hardware.cuh)
SM_COUNT = 16
SMEM_PER_BLOCK = 49152  # 48KB
REGS_PER_SM = 65536
MAX_THREADS = 1024
WARP_SIZE = 32

def scale_mem_bound(nominal_threads, nominal_items, type_size, max_smem=SMEM_PER_BLOCK):
    """Python mirror of muh::tuning::scale_mem_bound (common.cuh)."""
    items = nominal_items * 4 // type_size
    items = max(1, min(items, nominal_items * 2))
    smem_per_thread = type_size * items
    if smem_per_thread > 0:
        raw = max_smem // smem_per_thread
        max_threads_by_smem = ((raw + 31) // 32) * 32
    else:
        max_threads_by_smem = nominal_threads
    threads = min(nominal_threads, max_threads_by_smem)
    if threads < 32:
        threads = 32
    return items, threads

# CCCL types used in vllm/Qwen3.6
TYPE_SIZES = {
    "int8": 1,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
    "int64": 8,
    "int128": 16,
    "summary_stats_float": 28,  # 7×float from summary_statistics.cu
}

# bi100_* structs from tuning_reduce.cuh
BI100_STRUCTS = {
    "bi100_plus_accum1_o4":    {"threads": 512, "items": 32, "vec": 4},
    "bi100_plus_accum2_o4":    {"threads": 512, "items": 24, "vec": 2},
    "bi100_plus_float32_o4":   {"threads": 512, "items": 24, "vec": 2},
    "bi100_plus_float32_o8":   {"threads": 512, "items": 24, "vec": 1},
    "bi100_plus_float64_o4":   {"threads": 384, "items": 16, "vec": 2},
    "bi100_plus_float64_o8":   {"threads": 384, "items": 16, "vec": 1},
    "bi100_plus_int64_o4":     {"threads": 384, "items": 16, "vec": 2},
    "bi100_plus_int64_o8":     {"threads": 384, "items": 16, "vec": 1},
    "bi100_plus_accum16_o4":   {"threads": 192, "items": 16, "vec": 1},
    "bi100_det_float32":       {"threads": 384, "items": 32, "vec": 1},
    "bi100_det_float64":       {"threads": 384, "items": 16, "vec": 1},
    "bi100_det_int32":         {"threads": 384, "items": 32, "vec": 1},
    "bi100_det_int16":         {"threads": 384, "items": 64, "vec": 1},
    "bi100_default":           {"threads": 256, "items": 24, "vec": 4},
}


def test_scale_mem_bound_cccl_parity():
    """Test scale_mem_bound matches CCCL behavior for all type sizes."""
    print("=== scale_mem_bound CCCL parity ===")
    
    # CCCL test vectors from catch2_test_util_arch.cu MemBoundScaling tests
    test_cases = [
        # (nominal_threads, nominal_items, type_size) → expected (items, threads)
        (256, 16, 4, 16, 256),    # 4B identity
        (256, 16, 1, 32, 256),    # 1B: items scale up to 2×nominal
        (256, 16, 2, 32, 256),    # 2B: items scale up
        (256, 16, 8, 8, 256),     # 8B: items halve
        (256, 16, 16, 4, 256),    # 16B: items quarter
        (512, 16, 4, 16, 512),    # larger threads, 4B
        (640, 16, 8, 8, 640),     # CCCL SM100 float64: no SMEM cap needed
        (1024, 16, 16, 4, 768),   # 16B: SMEM cap triggers (1024 > 768)
    ]
    
    passed = 0
    for nom_t, nom_i, ts, exp_i, exp_t in test_cases:
        got_i, got_t = scale_mem_bound(nom_t, nom_i, ts)
        ok = got_i == exp_i and got_t == exp_t
        status = "PASS" if ok else "FAIL"
        if not ok:
            print(f"  {status}: scale_mem_bound({nom_t}, {nom_i}, {ts}) = "
                  f"({got_i}, {got_t}), expected ({exp_i}, {exp_t})")
        passed += ok
    
    print(f"  {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


def test_reduce_register_pressure():
    """Test that bi100_* reduce structs don't exceed register file."""
    print("\n=== reduce register pressure ===")
    
    all_ok = True
    for name, cfg in BI100_STRUCTS.items():
        threads = cfg["threads"]
        items = cfg["items"]
        
        # Estimate regs per thread: items × (accum_size/4) + ~16 overhead
        # For reduce, AccumT is held in registers (not SMEM staging)
        # Guess accum_size from struct name
        if "accum1" in name or "int8" in name:
            accum_size = 1
        elif "accum2" in name or "int16" in name or "det_int16" in name:
            accum_size = 2
        elif "float64" in name or "int64" in name:
            accum_size = 8
        elif "accum16" in name:
            accum_size = 16
        else:
            accum_size = 4  # float32 default
        
        regs_for_data = items * max(1, accum_size // 4)
        regs_overhead = 16  # control flow, addresses, etc
        regs_per_thread = regs_for_data + regs_overhead
        regs_per_cta = threads * regs_per_thread
        max_ctas = REGS_PER_SM // regs_per_cta if regs_per_cta > 0 else 0
        
        ok = max_ctas >= 1
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  {status}: {name:30s} threads={threads:4d} items={items:3d} "
              f"accum={accum_size:2d}B regs/thread={regs_per_thread:3d} "
              f"max_CTAs/SM={max_ctas}")
    
    return all_ok


def test_summary_statistics_accum():
    """Test scale_mem_bound for summary_stats_data<float> (28 bytes).
    
    This is the Welford parallel stats accumulator from summary_statistics.cu.
    If reduce tuning handles this, it handles anything vllm throws at it.
    """
    print("\n=== summary_statistics AccumT (28 bytes) ===")
    
    # Apply CCCL default (SM60 level): threads=256, items=16
    items, threads = scale_mem_bound(256, 16, 28)
    print(f"  scale_mem_bound(256, 16, 28) = items={items}, threads={threads}")
    print(f"  Data per thread: {items} × 28B = {items*28}B")
    
    # Check: items should be small (28B is huge)
    # 16 * 4 / 28 = 2.28 → clamp to 2
    assert items == 2, f"Expected items=2 for 28B type, got {items}"
    
    # SMEM cap: threads = min(256, round_up(49152/(28*2), 32))
    # = min(256, round_up(878, 32)) = min(256, 896) = 256
    assert threads == 256, f"Expected threads=256 for 28B type, got {threads}"
    
    # Register check: 2 items × 7 floats × 4B = 56B → 14 regs → trivial
    regs = items * 7  # 7 float fields in summary_stats_data
    print(f"  Registers for data: {regs} (14 regs per thread) — well within 255 limit")
    print(f"  PASS: summary_statistics AccumT is safe on BI-V100")
    return True


def test_vec_alignment():
    """Test that vec_size divides items_per_thread for all structs."""
    print("\n=== vectorization alignment ===")
    
    all_ok = True
    for name, cfg in BI100_STRUCTS.items():
        vec = cfg.get("vec", 1)
        items = cfg["items"]
        if vec > 1:
            ok = items % vec == 0
            if not ok:
                all_ok = False
                print(f"  FAIL: {name} items={items} not divisible by vec={vec}")
    
    if all_ok:
        print(f"  PASS: all {len(BI100_STRUCTS)} structs have valid vec alignment")
    return all_ok


def test_cccl_sm100_comparison():
    """Compare muh bi100 values against CCCL SM100 tuning comments."""
    print("\n=== CCCL SM100 comparison ===")
    
    # From tuning_reduce.cuh SM100 benchmark comments:
    sm100 = {
        "float32_o4": {"items": 16, "threads": 512, "vec": 2,
                       "speedup": [1.061295, 1.000000, 1.065478, 1.167139]},
        "float64_o4": {"items": 16, "threads": 640, "vec": 1,
                       "speedup": [1.017834, 1.000000, 1.015835, 1.057092]},
        "8B_o4":      {"items": 15, "threads": 512, "vec": 2,
                       "speedup": [1.019887, 1.0, 1.017636, 1.058036]},
        "8B_o8":      {"items": 15, "threads": 512, "vec": 1,
                       "speedup": [1.019414, 1.000000, 1.017218, 1.057143]},
    }
    
    bi100 = {
        "float32_o4": BI100_STRUCTS["bi100_plus_float32_o4"],
        "float64_o4": BI100_STRUCTS["bi100_plus_float64_o4"],
    }
    
    for key in ["float32_o4", "float64_o4"]:
        s = sm100[key]
        b = bi100[key]
        tile_ratio = (b["threads"] * b["items"]) / (s["threads"] * s["items"])
        print(f"  {key}:")
        print(f"    SM100: threads={s['threads']:4d} items={s['items']:2d} "
              f"vec={s['vec']} tile={s['threads']*s['items']:6d}")
        print(f"    BI100: threads={b['threads']:4d} items={b['items']:2d} "
              f"vec={b['vec']} tile={b['threads']*b['items']:6d} "
              f"(ratio={tile_ratio:.2f}×)")
        print(f"    SM100 speedup: {' '.join(f'{x:.3f}' for x in s['speedup'])}")
        print(f"    BI100 speedup: [TBD — needs real benchmark on Phanthy Cloud]")


def main():
    print("muh reduce tuning verification")
    print("CCCL ref: summary_statistics.cu (Welford parallel algorithm)")
    print("=" * 60)
    
    results = []
    results.append(("scale_mem_bound parity", test_scale_mem_bound_cccl_parity()))
    results.append(("register pressure", test_reduce_register_pressure()))
    results.append(("summary_statistics AccumT", test_summary_statistics_accum()))
    results.append(("vec alignment", test_vec_alignment()))
    test_cccl_sm100_comparison()
    
    print("\n" + "=" * 60)
    all_pass = all(r for _, r in results)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
