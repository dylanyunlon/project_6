#!/usr/bin/env python3
"""probe_moe_detail.py — Find exactly how to make MoE work on BI-V100"""
import os, sys, traceback

# 1. Check if vllm_moe_topk_softmax exists anywhere
print("=== 1. Search for vllm_moe_topk_softmax ===")
try:
    import ixformer.functions as ixf_F
    if hasattr(ixf_F, 'vllm_moe_topk_softmax'):
        print("  FOUND in ixf_F!")
    else:
        print("  NOT in ixf_F")
        # Check submodules
        for attr in dir(ixf_F):
            mod = getattr(ixf_F, attr)
            if hasattr(mod, 'vllm_moe_topk_softmax'):
                print(f"  FOUND in ixf_F.{attr}")
except Exception as e:
    print(f"  {e}")

# 2. Read the actual _custom_ops.py from base image (not our copy)
print("\n=== 2. Base image _custom_ops.py topk_softmax ===")
for p in ["/usr/local/corex/lib64/python3/dist-packages/vllm/_custom_ops.py",
          "/usr/local/corex/lib/python3/dist-packages/vllm/_custom_ops.py"]:
    if os.path.exists(p):
        print(f"  File: {p}")
        with open(p) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if 'topk_softmax' in line or 'moe_topk' in line or 'invoke_fused_moe' in line:
                # Print context
                start = max(0, i-2)
                end = min(len(lines), i+5)
                for j in range(start, end):
                    marker = ">>>" if j == i else "   "
                    print(f"  {marker} {j+1}: {lines[j].rstrip()}")
                print()
        break

# 3. Read base image fused_moe.py — the actual kernel dispatch
print("\n=== 3. Base image fused_moe.py kernel dispatch ===")
for p in ["/usr/local/corex/lib64/python3/dist-packages/vllm/model_executor/layers/fused_moe/fused_moe.py",
          "/usr/local/corex/lib/python3/dist-packages/vllm/model_executor/layers/fused_moe/fused_moe.py"]:
    if os.path.exists(p):
        print(f"  File: {p}")
        with open(p) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if 'invoke_fused_moe' in line or 'triton' in line.lower() or 'kernel' in line.lower() or 'ixf' in line.lower():
                start = max(0, i-1)
                end = min(len(lines), i+3)
                for j in range(start, end):
                    marker = ">>>" if j == i else "   "
                    print(f"  {marker} {j+1}: {lines[j].rstrip()}")
                print()
        break

# 4. Check _ixformer_torch for topk
print("\n=== 4. _ixformer_torch Python bindings ===")
try:
    import ixformer._ixformer_torch as ixt
    print(f"  Module: {ixt}")
    for attr in sorted(dir(ixt)):
        if not attr.startswith('__'):
            print(f"    {attr}")
except Exception as e:
    print(f"  {e}")

# 5. Check ixformer.functions.vllm source
print("\n=== 5. ixformer.functions.vllm source (for vllm_moe references) ===")
try:
    import ixformer.functions.vllm as ixf_vllm
    import inspect
    src = inspect.getsource(ixf_vllm)
    for i, line in enumerate(src.split('\n')):
        if 'moe' in line.lower() or 'topk' in line.lower() or 'expert' in line.lower() or 'mlp' in line.lower():
            print(f"  {i+1}: {line}")
except Exception as e:
    print(f"  {e}")

# 6. What does _custom_ops invoke_fused_moe_kernel look like?
print("\n=== 6. invoke_fused_moe_kernel in _custom_ops ===")
for p in ["/usr/local/corex/lib64/python3/dist-packages/vllm/_custom_ops.py"]:
    if os.path.exists(p):
        with open(p) as f:
            content = f.read()
        if 'invoke_fused_moe' in content:
            idx = content.index('invoke_fused_moe')
            start = max(0, content.rfind('\n', 0, idx-100))
            end = content.find('\n\n', idx+100)
            print(content[start:end])
        else:
            print("  invoke_fused_moe NOT in _custom_ops.py")
            # What IS there for MoE?
            for line in content.split('\n'):
                if 'moe' in line.lower() or 'expert' in line.lower():
                    print(f"  {line.strip()}")
