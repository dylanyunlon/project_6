#!/usr/bin/env python3
"""Find which .so files export ixformer::infer symbols."""
import subprocess, glob, os

targets = ["silu_and_mul", "rms_norm", "ixformer_linear", "topk_softmax",
           "xllm_paged_attention", "xllm_reshape_and_cache",
           "moe_w16a16_group_gemm", "residual_rms_norm"]

search_dirs = [
    "/usr/local/corex/lib64",
    "/usr/local/corex/lib",
    "/usr/local/corex-3.2.3/lib64",
    "/usr/local/corex-3.2.3/lib",
    "/usr/local/lib",
]

so_files = []
for d in search_dirs:
    so_files.extend(glob.glob(os.path.join(d, "**/*.so*"), recursive=True))

print(f"Scanning {len(so_files)} .so files...")

for target in targets:
    found = False
    for so in so_files:
        try:
            out = subprocess.run(["nm", "-D", so], capture_output=True, text=True, timeout=5)
            if target in out.stdout:
                # Get the full symbol name
                for line in out.stdout.split('\n'):
                    if target in line and ' T ' in line:
                        sym = line.split()[-1]
                        print(f"✓ {target}: {os.path.basename(so)}  [{sym[:80]}]")
                        found = True
                        break
                if found:
                    break
        except:
            pass
    if not found:
        # Try with grep on all lines (U = undefined, T = defined)
        for so in so_files:
            try:
                out = subprocess.run(["nm", "-D", so], capture_output=True, text=True, timeout=5)
                for line in out.stdout.split('\n'):
                    if target in line:
                        print(f"? {target}: {os.path.basename(so)}  [{line.strip()[:100]}]")
                        found = True
                        break
                if found:
                    break
            except:
                pass
    if not found:
        print(f"✗ {target}: NOT FOUND in any .so")
