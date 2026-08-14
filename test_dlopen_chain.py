#!/usr/bin/env python3
"""Test dlopen chain on BI-V100. Run on real machine."""
import sys, os, importlib, ctypes

def test_prebuilt_so():
    """Test all 14 prebuilt corex .so can import from vllm package."""
    try:
        import vllm
        vllm_root = os.path.dirname(vllm.__file__)
    except ImportError:
        print("[SKIP] vllm not installed, testing .so ELF headers only")
        vllm_root = None
    
    modules = [
        "corex_attn_head_rms_norm", "corex_block_major_kv_transfer",
        "corex_fused_paged_prefill", "corex_gdn_beta_decay",
        "corex_gdn_causal_conv", "corex_gdn_chunk_recurrent",
        "corex_gdn_gated_norm", "corex_gdn_packed_decode",
        "corex_gdn_qk_map", "corex_moe_direct_routed",
        "corex_moe_exact_reduce", "corex_moe_topk_softmax",
        "corex_moe_weight_gather", "corex_paged_kv_gather",
    ]
    
    ok = fail = 0
    for name in modules:
        if vllm_root:
            so_path = os.path.join(vllm_root, f"{name}.so")
            if os.path.exists(so_path):
                try:
                    mod = importlib.import_module(f"vllm.{name}")
                    funcs = [x for x in dir(mod) if not x.startswith('_')]
                    print(f"  [OK] {name}: {funcs[:3]}")
                    ok += 1
                except Exception as e:
                    print(f"  [FAIL] {name}: {e}")
                    fail += 1
            else:
                print(f"  [MISS] {name}: not installed at {so_path}")
                fail += 1
        else:
            # Just check prebuilt exists
            prebuilt = f"qwen3_6_scripts/prebuilt/corex-3.2.3-ivcore10/{name}.so"
            if os.path.exists(prebuilt):
                print(f"  [FILE] {name}: {os.path.getsize(prebuilt)} bytes")
                ok += 1
            else:
                print(f"  [MISS] {name}")
                fail += 1
    return ok, fail

def test_ixformer():
    """Test ixformer dispatch chain."""
    checks = [
        ("ixformer", None),
        ("ixformer.functions", "vllm_single_query_cached_kv_attention"),
        ("ixformer.contrib.vllm_flash_attn", "flash_attn_varlen_func"),
    ]
    ok = fail = 0
    for mod_name, func_name in checks:
        try:
            mod = importlib.import_module(mod_name)
            if func_name:
                fn = getattr(mod, func_name, None)
                if fn:
                    print(f"  [OK] {mod_name}.{func_name}")
                    ok += 1
                else:
                    avail = [x for x in dir(mod) if not x.startswith('_')]
                    print(f"  [MISS] {mod_name}.{func_name} — available: {avail[:5]}")
                    fail += 1
            else:
                print(f"  [OK] {mod_name} v{getattr(mod, '__version__', '?')}")
                ok += 1
        except ImportError as e:
            print(f"  [FAIL] {mod_name}: {e}")
            fail += 1
    return ok, fail

def test_base_so():
    """Test base image .so availability."""
    paths = [
        "/usr/local/corex/lib64/libcorex_gdn.so",
        "/usr/local/corex/lib64/libixattn.so",
    ]
    ok = fail = 0
    for p in paths:
        if os.path.exists(p):
            try:
                ctypes.CDLL(p, mode=ctypes.RTLD_LAZY)
                print(f"  [OK] {p}")
                ok += 1
            except Exception as e:
                print(f"  [FAIL] {p}: {e}")
                fail += 1
        else:
            print(f"  [MISS] {p}")
            fail += 1
    return ok, fail

def test_torch_cuda():
    """Test basic CUDA/torch."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            print(f"  [OK] {name}, {mem:.1f}GB")
            t = torch.zeros(1024, device='cuda')
            del t
            print(f"  [OK] CUDA alloc/free works")
            return 2, 0
        else:
            print(f"  [FAIL] CUDA not available")
            return 0, 1
    except Exception as e:
        print(f"  [FAIL] {e}")
        return 0, 1

if __name__ == "__main__":
    total_ok = total_fail = 0
    
    print("=== 1. Torch/CUDA ===")
    ok, fail = test_torch_cuda()
    total_ok += ok; total_fail += fail
    
    print("\n=== 2. Prebuilt CoreX .so (14 modules) ===")
    ok, fail = test_prebuilt_so()
    total_ok += ok; total_fail += fail
    
    print("\n=== 3. ixformer dispatch chain ===")
    ok, fail = test_ixformer()
    total_ok += ok; total_fail += fail
    
    print("\n=== 4. Base image .so ===")
    ok, fail = test_base_so()
    total_ok += ok; total_fail += fail
    
    print(f"\n{'='*50}")
    print(f"OK: {total_ok}  FAIL: {total_fail}")
    if total_fail == 0:
        print("All dlopen chains verified.")
    else:
        print(f"WARNING: {total_fail} checks failed!")
