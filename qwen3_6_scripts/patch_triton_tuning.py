"""
patch_triton_tuning.py — BI-V100 Triton kernel parameter optimization
======================================================================

Patches prefix_prefill.py to use BI-V100-optimal BLOCK and NUM_WARPS values.

Hardware derivation:
  BI-V100 SMEM = 48KB. Triton Flash Attention needs K+V tiles in SMEM:
    SMEM = BLOCK_N × head_dim × sizeof(fp16) × 2
    BLOCK_N=64, head_dim=128 → 32KB ≤ 48KB ✓ (current, correct)
    BLOCK_N=128, head_dim=128 → 64KB > 48KB ✗ (would crash)
  → BLOCK must stay at 64.

  NUM_WARPS derivation:
    At BLOCK=64, each block does 64 query positions.
    8 warps = 256 threads → each thread handles 32 elements from Q tile.
    4 warps = 128 threads → each thread handles 64 elements.
    
    With 16 SMs (confirmed) and typical grid of 37K+ blocks:
      At 8 warps + 32KB SMEM: 1 block per SM (SMEM-limited)
      At 4 warps + 32KB SMEM: potentially 2 blocks per SM
    
    BI-V100 is bandwidth-limited (900 GB/s), not latency-limited.
    Fewer warps hiding latency matters less; more blocks = better.
    → NUM_WARPS = 4

Deploy: python3 qwen3_6_scripts/patch_triton_tuning.py
"""

import os

PREFIX_PREFILL_PATHS = [
    "/usr/local/corex/lib/python3/dist-packages/vllm/attention/ops/prefix_prefill.py",
    "/usr/local/corex/lib64/python3/dist-packages/vllm/attention/ops/prefix_prefill.py",
]

# Original line (baseline):
OLD_BLOCK = "        BLOCK = 128 if current_platform.has_device_capability(80) else 64\n        NUM_WARPS = 8"

# Optimized for BI-V100:
NEW_BLOCK = """\
        # BI-V100 optimization (patch_triton_tuning.py):
        # BLOCK=64: SMEM constraint — BLOCK_N=128 overflows 48KB at head_dim=128
        # NUM_WARPS=4: bandwidth-limited GPU benefits from more blocks/SM over more warps
        # Derivation: 4 warps at BLOCK=64 allows 2 concurrent blocks per SM,
        # doubling occupancy vs 8 warps (which is SMEM-limited to 1 block/SM).
        BLOCK = 64
        NUM_WARPS = 4"""


def patch():
    for path in PREFIX_PREFILL_PATHS:
        if not os.path.exists(path):
            continue
        
        with open(path, "r") as f:
            content = f.read()
        
        if "NUM_WARPS = 4" in content:
            print(f"  [skip] {path}: already patched")
            return
        
        if OLD_BLOCK not in content:
            # Try the alternative: maybe it's already using BLOCK=64 hardcoded
            alt_old = "        BLOCK = 64\n        NUM_WARPS = 8"
            if alt_old in content:
                content = content.replace(alt_old, NEW_BLOCK, 1)
                with open(path, "w") as f:
                    f.write(content)
                print(f"  [ok] {path}: patched NUM_WARPS 8→4 (BLOCK was already 64)")
                return
            print(f"  [warn] {path}: original block not found, manual check needed")
            return
        
        content = content.replace(OLD_BLOCK, NEW_BLOCK, 1)
        with open(path, "w") as f:
            f.write(content)
        print(f"  [ok] {path}: patched BLOCK=64, NUM_WARPS=4")
        return
    
    print("  [error] prefix_prefill.py not found at any expected path")


def main():
    print("=== patch_triton_tuning: BI-V100 Triton kernel optimization ===")
    patch()
    print("Done.")


if __name__ == "__main__":
    main()
