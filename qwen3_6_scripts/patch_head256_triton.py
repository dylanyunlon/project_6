"""
patch_head256_triton.py — Enable Triton prefill for head_dim=256
=================================================================

Qwen3.6-35B-A3B uses head_dim=256 (confirmed: text_cfg.head_dim=256,
num_heads=24, num_kv_heads=4, GQA ratio=6).

Current problem:
  prefix_prefill.py: BLOCK = 64 (set by patch_triton_tuning.py)
  SMEM needed: BLOCK_N × head_dim × sizeof(fp16) × 2 (K+V tiles)
  = 64 × 256 × 2 × 2 = 64KB > 48KB (BI-V100 SMEM limit)
  → Triton kernel CANNOT launch at BLOCK=64 for head_dim=256.

  xformers.py: _run_sdpa_fallback triggers for head_size > 128.
  This is a Python for-loop with Q-tiling — orders of magnitude slower.

Fix:
  1. In prefix_prefill.py launcher, use BLOCK based on head_dim:
     head_dim ≤ 128: BLOCK = 64, NUM_WARPS = 4  (32KB SMEM, fits)
     head_dim = 256: BLOCK = 32, NUM_WARPS = 4  (32KB SMEM, fits)
     head_dim > 256: BLOCK = 16, NUM_WARPS = 2  (16KB SMEM, fits)
  
  2. Triton BLOCK=32 means more kernel launches per sequence but
     each launch uses only 32KB SMEM — well within 48KB limit.
     32×256×2×2 = 32KB ≤ 48KB ✓

  3. Also optimize _run_sdpa_fallback _Q_CHUNK:
     Current: 256 (same for all head dims)
     For head_dim=256: memory = _Q_CHUNK × seq_len × H × 4 bytes (float32)
     At Q_CHUNK=256, seq_len=100K, H=24: 256×100K×24×4 ≈ 2.3GB
     Better: _Q_CHUNK=128 for head_dim=256 → 1.2GB (safer for OOM)

Deploy: python3 qwen3_6_scripts/patch_head256_triton.py
  Must run AFTER patch_ops.sh and patch_triton_tuning.py
"""

import os

PREFIX_PREFILL_PATHS = [
    "/usr/local/corex/lib/python3/dist-packages/vllm/attention/ops/prefix_prefill.py",
    "/usr/local/corex/lib64/python3/dist-packages/vllm/attention/ops/prefix_prefill.py",
]

XFORMERS_PATHS = [
    "/usr/local/corex/lib/python3/dist-packages/vllm/attention/backends/xformers.py",
    "/usr/local/corex/lib64/python3/dist-packages/vllm/attention/backends/xformers.py",
]

# --- Patch 1: prefix_prefill.py BLOCK selection based on head_dim ---

# The current patch_triton_tuning.py sets:
#   BLOCK = 64
#   NUM_WARPS = 4
# We need to make BLOCK depend on head_dim:

OLD_BLOCK_SETTING = """\
        # BI-V100 optimization (patch_triton_tuning.py):
        # BLOCK=64: SMEM constrains BLOCK_N≤64 for head_dim=128
        # NUM_WARPS=4: fewer warps → more blocks/SM → better occupancy
        BLOCK = 64
        NUM_WARPS = 4"""

NEW_BLOCK_SETTING = """\
        # BI-V100: BLOCK must fit in 48KB SMEM.
        # SMEM = BLOCK_N × head_dim × sizeof(fp16) × 2 (K+V tiles)
        # head_dim=128: BLOCK=64 → 64×128×2×2 = 32KB ✓
        # head_dim=256: BLOCK=32 → 32×256×2×2 = 32KB ✓ (BLOCK=64 → 64KB overflow!)
        # head_dim>256: BLOCK=16 → fallback
        Lk = q.shape[-1]
        if Lk <= 128:
            BLOCK = 64
            NUM_WARPS = 4
        elif Lk <= 256:
            BLOCK = 32
            NUM_WARPS = 4
        else:
            BLOCK = 16
            NUM_WARPS = 2"""

# Alternative: if patch_triton_tuning.py hasn't run yet, patch the original
OLD_BLOCK_ORIGINAL = """\
        BLOCK = 128 if current_platform.has_device_capability(80) else 64
        NUM_WARPS = 8"""

NEW_BLOCK_FROM_ORIGINAL = NEW_BLOCK_SETTING


# --- Patch 2: _run_sdpa_fallback Q_CHUNK for head_dim=256 ---

OLD_Q_CHUNK = "        _Q_CHUNK = 256"
NEW_Q_CHUNK = """\
        # Adapt Q chunk size to head_dim to control memory:
        # head_dim=128: 256 × seq_len × H × 4B → manageable
        # head_dim=256: halve to 128 to avoid OOM on long sequences
        _Q_CHUNK = 128 if self.head_size > 128 else 256"""


def patch_prefix_prefill():
    for path in PREFIX_PREFILL_PATHS:
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            content = f.read()
        
        changed = False
        if "head_dim=256: BLOCK=32" in content:
            print(f"  [skip] {path}: head_dim-aware BLOCK already present")
            return True
        
        if OLD_BLOCK_SETTING in content:
            content = content.replace(OLD_BLOCK_SETTING, NEW_BLOCK_SETTING, 1)
            changed = True
            print(f"  [ok] Replaced fixed BLOCK=64 with head_dim-dependent selection")
        elif OLD_BLOCK_ORIGINAL in content:
            content = content.replace(OLD_BLOCK_ORIGINAL, NEW_BLOCK_FROM_ORIGINAL, 1)
            changed = True
            print(f"  [ok] Replaced original BLOCK selection with head_dim-dependent version")
        else:
            print(f"  [warn] Neither BLOCK anchor found in {path}")
        
        # Also need to move Lk computation before BLOCK selection
        # Currently Lk is computed AFTER BLOCK is set (line ~750)
        # We need it before. Check if Lk is already available:
        if "Lk = q.shape[-1]" in content and "Lk, Lk, Lv" in content:
            # Lk is computed later — we duplicate the computation for BLOCK selection
            # This is safe because q.shape[-1] doesn't change
            print(f"  [note] Lk computed early for BLOCK selection + later for kernel args")
        
        if changed:
            with open(path, "w") as f:
                f.write(content)
            print(f"  Written: {path}")
        return changed
    
    print("  [error] prefix_prefill.py not found")
    return False


def patch_xformers_sdpa():
    for path in XFORMERS_PATHS:
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            content = f.read()
        
        if "head_size > 128 else 256" in content:
            print(f"  [skip] {path}: adaptive Q_CHUNK already present")
            return True
        
        if OLD_Q_CHUNK in content:
            content = content.replace(OLD_Q_CHUNK, NEW_Q_CHUNK, 1)
            with open(path, "w") as f:
                f.write(content)
            print(f"  [ok] {path}: _Q_CHUNK now adapts to head_dim")
            return True
        else:
            print(f"  [warn] _Q_CHUNK anchor not found in {path}")
    
    print("  [error] xformers.py not found")
    return False


def main():
    print("=== patch_head256_triton: Enable Triton for head_dim=256 ===")
    print(f"  Qwen3.6: head_dim=256, num_heads=24, num_kv_heads=4, GQA=6")
    print()
    
    print("--- Patch 1: prefix_prefill.py BLOCK selection ---")
    patch_prefix_prefill()
    
    print("\n--- Patch 2: xformers Q_CHUNK for head_dim=256 ---")
    patch_xformers_sdpa()
    
    print("\nSMEM budget at BLOCK=32, head_dim=256:")
    print(f"  K tile: 32 × 256 × 2B = 16KB")
    print(f"  V tile: 32 × 256 × 2B = 16KB")
    print(f"  Total: 32KB ≤ 48KB ✓")
    print("\nDone.")


if __name__ == "__main__":
    main()
