#!/bin/bash
set -euo pipefail
cat << 'PYEOF' | CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 -u -
"""Audit: which .so functions are actually called in the hot path vs available but unused."""
import importlib.util, os, sys

SO_DIR = "qwen3_6_scripts/prebuilt/corex-3.2.3-ivcore10"
QWEN = "qwen3_6_scripts/qwen3_5.py"
PATCH = "qwen3_6_scripts/ex_engine/python/patch_vllm_hot_path.py"
XLLM_OPS = "qwen3_6_scripts/ex_engine/python/xllm_ops.py"

# 1. Collect all exported functions from all .so
print("=" * 70)
print("  AUDIT: .so function usage")
print("=" * 70)

so_exports = {}
for f in sorted(os.listdir(SO_DIR)):
    if not f.endswith(".so"):
        continue
    name = f[:-3]
    path = os.path.join(SO_DIR, f)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        fns = [x for x in dir(m) if not x.startswith("_")]
        so_exports[name] = fns
    except Exception as e:
        so_exports[name] = [f"LOAD_ERROR: {e}"]

# 2. Search for usage in qwen3_5.py, patch_vllm_hot_path.py, xllm_ops.py
code_files = {}
for label, path in [("qwen3_5.py", QWEN), ("patch_hot_path.py", PATCH), ("xllm_ops.py", XLLM_OPS)]:
    try:
        with open(path) as f:
            code_files[label] = f.read()
    except:
        code_files[label] = ""

# Also scan all ex_engine python files
for f in os.listdir("qwen3_6_scripts/ex_engine/python"):
    if f.endswith(".py"):
        path = os.path.join("qwen3_6_scripts/ex_engine/python", f)
        try:
            with open(path) as fh:
                code_files[f"ex_engine/{f}"] = fh.read()
        except:
            pass

all_code = "\n".join(code_files.values())

# 3. For each .so and function, check if it's referenced
print(f"\n{'SO Module':<35} {'Function':<30} {'Used?':<6} {'Where'}")
print("-" * 110)

total_fns = 0
used_fns = 0
unused = []

for so_name in sorted(so_exports.keys()):
    fns = so_exports[so_name]
    for fn in fns:
        if "LOAD_ERROR" in fn:
            print(f"{so_name:<35} {fn}")
            continue
        total_fns += 1
        
        # Search patterns: module.fn, .fn(, "fn"
        found_in = []
        for label, code in code_files.items():
            if f".{fn}" in code or f'"{fn}"' in code or f"'{fn}'" in code:
                found_in.append(label)
        
        is_used = len(found_in) > 0
        if is_used:
            used_fns += 1
        else:
            unused.append((so_name, fn))
        
        where = ", ".join(found_in[:3]) if found_in else ""
        marker = "  ✓" if is_used else "  ✗"
        print(f"{so_name:<35} {fn:<30} {marker:<6} {where}")

print(f"\n{'=' * 70}")
print(f"  TOTAL: {used_fns}/{total_fns} functions used")
print(f"  UNUSED: {total_fns - used_fns} functions")
print(f"{'=' * 70}")

if unused:
    print(f"\n  === UNUSED FUNCTIONS ===")
    for so_name, fn in unused:
        print(f"    {so_name}.{fn}")

# 4. Check which ixformer_torch_ext functions exist but aren't wrapped
print(f"\n  === ixformer_torch_ext available but not in any bridge .so ===")
ix_fns = [
    "ixformer_linear", "ixformer_linear_ex", "ixformer_linear_allreduce",
    "linear_i8w8o32", "quantized_linear_awq", "quantized_linear_gptq",
    "quantized_linear_int8", "quantized_linear_float4", "ixformer_quantized_linear",
    "silu_and_mul_forward", "rms_norm_forward", "fused_add_rms_norm_forward",
]
for fn in ix_fns:
    in_bridge = fn in all_code
    print(f"    {fn:<40} {'✓ wrapped' if in_bridge else '✗ NOT wrapped'}")
PYEOF