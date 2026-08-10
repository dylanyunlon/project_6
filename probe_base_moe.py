#!/usr/bin/env python3
"""
probe_base_moe.py — Find how base image vllm's FusedMoE actually works

The key question: when vllm calls FusedMoE on BI-V100, what kernel does it use?
comp 168 log shows "expert-grouped-wmma" — this is a WMMA (tensor core) kernel.
"""
import sys, os, traceback

print("=" * 60)
print("PROBE: Base image vllm FusedMoE dispatch chain")
print("=" * 60)

# 1. Check what _custom_ops.py does for topk_softmax
print("\n--- 1. vllm._custom_ops topk_softmax ---")
try:
    from vllm._custom_ops import topk_softmax
    print(f"  topk_softmax: {topk_softmax}")
    import inspect
    src = inspect.getsource(topk_softmax)
    # Print first 20 lines
    for i, line in enumerate(src.split('\n')[:20]):
        print(f"    {line}")
except Exception as e:
    print(f"  {e}")

# 2. Check FusedMoE layer
print("\n--- 2. vllm FusedMoE layer ---")
try:
    from vllm.model_executor.layers.fused_moe import FusedMoE
    print(f"  FusedMoE: {FusedMoE}")
    import inspect
    src_file = inspect.getfile(FusedMoE)
    print(f"  File: {src_file}")
    # Check forward method
    if hasattr(FusedMoE, 'forward'):
        src = inspect.getsource(FusedMoE.forward)
        for i, line in enumerate(src.split('\n')[:30]):
            print(f"    {line}")
except Exception as e:
    print(f"  {e}")

# 3. Check fused_moe function (the one that actually runs)
print("\n--- 3. vllm fused_moe function ---")
try:
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_moe
    import inspect
    src = inspect.getsource(fused_moe)
    for i, line in enumerate(src.split('\n')[:40]):
        print(f"    {line}")
except Exception as e:
    try:
        from vllm.model_executor.layers.fused_moe import fused_moe
        import inspect
        src = inspect.getsource(fused_moe)
        for i, line in enumerate(src.split('\n')[:40]):
            print(f"    {line}")
    except Exception as e2:
        print(f"  {e2}")

# 4. Check what torch.ops.vllm has
print("\n--- 4. torch.ops.vllm MoE ops ---")
try:
    import torch
    vllm_ops = torch.ops.vllm
    for name in dir(vllm_ops):
        if 'moe' in name.lower() or 'topk' in name.lower() or 'expert' in name.lower():
            print(f"  torch.ops.vllm.{name}")
except Exception as e:
    print(f"  {e}")

# 5. Check ixformer_torch_ext for any MoE-related ops
print("\n--- 5. _ixformer_torch MoE symbols (demangled) ---")
os.system("nm -D /usr/local/corex/lib64/python3/dist-packages/ixformer/_ixformer_torch.cpython-310-x86_64-linux-gnu.so 2>/dev/null | grep -i 'moe\\|expert\\|topk\\|gemm' | c++filt | head -20")

# 6. Check if there's a Triton-based MoE
print("\n--- 6. Triton MoE kernels ---")
try:
    from vllm.model_executor.layers.fused_moe import fused_moe as fm_module
    import inspect
    src_file = inspect.getfile(fm_module)
    print(f"  Module file: {src_file}")
except:
    pass

# Check for any .so with group_gemm
print("\n--- 7. group_gemm in any system .so ---")
os.system("find /usr/local/corex -name '*.so*' -exec sh -c 'nm -D \"$1\" 2>/dev/null | grep -q group_gemm && echo \"  $1\"' _ {} \\;")

# 8. Check the actual _custom_ops topk_softmax implementation
print("\n--- 8. _custom_ops.py full topk_softmax chain ---")
try:
    custom_ops_path = None
    for p in ["/usr/local/corex/lib64/python3/dist-packages/vllm/_custom_ops.py",
              "/usr/local/corex/lib/python3/dist-packages/vllm/_custom_ops.py"]:
        if os.path.exists(p):
            custom_ops_path = p
            break
    if custom_ops_path:
        with open(custom_ops_path) as f:
            content = f.read()
        # Find topk_softmax function
        lines = content.split('\n')
        in_func = False
        for i, line in enumerate(lines):
            if 'def topk_softmax' in line or 'topk_softmax' in line:
                in_func = True
            if in_func:
                print(f"  {i+1}: {line}")
                if line.strip() == '' and in_func:
                    in_func = False
            if i > 0 and in_func and not line.startswith(' ') and not line.startswith('\t') and line.strip():
                in_func = False
except Exception as e:
    print(f"  {e}")

# 9. What does ixformer.functions.vllm do?
print("\n--- 9. ixformer.functions.vllm module ---")
try:
    import ixformer.functions.vllm as ixf_vllm
    print(f"  Module: {ixf_vllm}")
    for attr in dir(ixf_vllm):
        if not attr.startswith('_'):
            print(f"    {attr}")
except Exception as e:
    print(f"  {e}")
