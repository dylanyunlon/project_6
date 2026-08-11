#!/usr/bin/env python3
"""Verify ix_unified_bridge.so with ixformer symbols pre-loaded."""
import ctypes, glob, importlib.util, os, sys, torch

# Step 1: find and pre-load ixformer .so to resolve symbols
ixf_paths = [
    "/usr/local/corex/lib64/python3/dist-packages/ixformer",
    "/usr/local/corex/lib/python3/dist-packages/ixformer",
]
loaded = False
for base in ixf_paths:
    for so in glob.glob(os.path.join(base, "**/*.so"), recursive=True):
        try:
            ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
        except:
            pass
    # Try importing ixformer to trigger all symbol loads
    try:
        import ixformer.functions
        loaded = True
        print(f"✓ ixformer.functions loaded")
        break
    except:
        pass

if not loaded:
    print("✗ ixformer not found, bridge will have unresolved symbols")
    sys.exit(1)

# Step 2: load our bridge
so_files = glob.glob("ex_engine/build/ix_unified_bridge*.so")
if not so_files:
    print("✗ bridge .so not built")
    sys.exit(1)

spec = importlib.util.spec_from_file_location("ix_unified_bridge", so_files[0])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
funcs = [x for x in dir(mod) if not x.startswith('_')]
print(f"✓ bridge loaded: {len(funcs)} functions: {funcs}")

# Step 3: smoke test on GPU
x = torch.randn(4, 512, device="cuda", dtype=torch.float16)
out = mod.silu_and_mul(x)
print(f"✓ silu_and_mul via bridge: {x.shape} → {out.shape}")

inp = torch.randn(2, 2048, device="cuda", dtype=torch.float16)
outp = torch.empty_like(inp)
w = torch.ones(2048, device="cuda", dtype=torch.float16)
mod.rms_norm(outp, inp, w, 1e-6)
print(f"✓ rms_norm via bridge: {inp.shape}")

gate = torch.randn(4, 64, device="cuda", dtype=torch.float16)
weights, indices = mod.moe_topk_softmax(gate, 8, True)
print(f"✓ moe_topk_softmax via bridge: weights={weights.shape}")

print("\nALL BRIDGE TESTS PASSED — Tier 0 active")
