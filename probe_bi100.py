#!/usr/bin/env python3
"""probe_bi100.py — Run on BI-V100 real machine, paste output back.
Usage: python3 probe_bi100.py
"""
import os, sys, importlib, struct, pathlib, traceback

def section(t):
    print(f"\n{'='*60}\n  {t}\n{'='*60}")

# 1. ixformer.functions 完整 API 清单
section("1. ixformer.functions API surface")
try:
    import ixformer.functions as ixf_F
    names = sorted([n for n in dir(ixf_F) if not n.startswith('_')])
    print(f"total: {len(names)}")
    for n in names:
        print(f"  {n}")
except Exception as e:
    print(f"IMPORT FAILED: {e}")

# 2. ixformer._C.infer API
section("2. ixformer._C.infer API surface")
try:
    import ixformer._C as ops
    if hasattr(ops, 'infer'):
        names = sorted([n for n in dir(ops.infer) if not n.startswith('_')])
        print(f"total: {len(names)}")
        for n in names:
            print(f"  {n}")
    else:
        print("ops.infer not found")
        print(f"ops attrs: {[n for n in dir(ops) if not n.startswith('_')]}")
except Exception as e:
    print(f"IMPORT FAILED: {e}")

# 3. 关键函数存在性
section("3. Critical function checks")
checks = [
    ("ixformer.functions", "vllm_moe_topk_softmax"),
    ("ixformer.functions", "moe_topk_softmax"),
    ("ixformer.functions", "moe_compute_token_index"),
    ("ixformer.functions", "moe_expand_input"),
    ("ixformer.functions", "moe_output_reduce_sum"),
    ("ixformer.functions", "moe_w8a8_group_gemm"),
    ("ixformer.functions", "silu_and_mul"),
    ("ixformer.functions", "rms_norm"),
    ("ixformer.functions", "fused_add_rms_norm"),
    ("ixformer.functions", "vllm_rotary_embedding_neox"),
    ("ixformer.functions", "vllm_paged_attention"),
    ("ixformer.functions", "vllm_reshape_and_cache"),
    ("ixformer.functions", "flash_attn_varlen_func"),
    ("ixformer.functions", "vllm_single_query_cached_kv_attention"),
]
for mod_name, func_name in checks:
    try:
        mod = importlib.import_module(mod_name)
        has = hasattr(mod, func_name)
        print(f"  {'OK' if has else 'MISSING':7s} {mod_name}.{func_name}")
    except Exception as e:
        print(f"  ERROR   {mod_name}.{func_name} — {e}")

# 4. vllm 路径和已安装的 .so
section("4. vllm install paths + installed .so")
try:
    import vllm
    vroot = pathlib.Path(vllm.__path__[0])
    print(f"vllm root: {vroot}")
    sos = sorted(vroot.glob("*.so"))
    print(f".so count: {len(sos)}")
    for s in sos:
        print(f"  {s.name:45s} {s.stat().st_size:>10d} bytes")
except Exception as e:
    print(f"ERROR: {e}")

# 5. _custom_ops.py 实际位置
section("5. _custom_ops.py location + topk_softmax test")
try:
    import vllm._custom_ops as ops
    print(f"_custom_ops: {ops.__file__}")
    # Try calling topk_softmax
    import torch
    if torch.cuda.is_available():
        g = torch.randn(4, 8, device='cuda', dtype=torch.float32)
        tw = torch.empty(4, 2, device='cuda', dtype=torch.float32)
        ti = torch.empty(4, 2, device='cuda', dtype=torch.int32)
        tei = torch.empty(4, 2, device='cuda', dtype=torch.int32)
        try:
            ops.topk_softmax(tw, ti, tei, g)
            print("  topk_softmax: OK")
        except Exception as e:
            print(f"  topk_softmax: FAILED — {e}")
    else:
        print("  no CUDA device")
except Exception as e:
    print(f"ERROR: {e}")

# 6. protocol.py 检查
section("6. protocol.py max_completion_tokens")
try:
    from vllm.entrypoints.openai.protocol import ChatCompletionRequest, OpenAIBaseModel
    print(f"protocol: {ChatCompletionRequest.__module__}")
    print(f"extra config: {OpenAIBaseModel.model_config.get('extra', 'NOT SET')}")
    has_mct = 'max_completion_tokens' in ChatCompletionRequest.model_fields
    print(f"max_completion_tokens field: {'YES' if has_mct else 'NO'}")
    # Try validation
    req = ChatCompletionRequest(
        model="llm",
        messages=[{"role":"user","content":"test"}],
        max_completion_tokens=8192,
    )
    print(f"  validation OK — max_tokens={req.max_tokens}")
except Exception as e:
    print(f"FAILED: {e}")

# 7. CoreX compiler
section("7. CoreX compiler availability")
for p in ["/usr/local/corex-3.2.3/bin/clang++", "/usr/local/corex/bin/clang++",
          "/opt/corex/bin/clang++"]:
    exists = os.path.isfile(p)
    print(f"  {'OK' if exists else '--':2s} {p}")

# 8. corex prebuilt .so import test
section("8. corex prebuilt .so import test")
try:
    import vllm
    vroot = pathlib.Path(vllm.__path__[0])
    for name in ["corex_moe_topk_softmax", "corex_gdn_causal_conv",
                 "corex_moe_direct_routed", "corex_moe_index_combine",
                 "corex_attn_head_rms_norm", "corex_fused_paged_prefill",
                 "corex_paged_kv_gather", "ix_full_bridge"]:
        so = vroot / f"{name}.so"
        if so.exists():
            try:
                mod = importlib.import_module(f"vllm.{name}")
                funcs = [f for f in dir(mod) if not f.startswith('_')]
                print(f"  OK    {name} — {funcs}")
            except Exception as e:
                print(f"  LOAD_FAIL {name} — {e}")
        else:
            print(f"  MISSING   {name}.so")
except Exception as e:
    print(f"ERROR: {e}")

# 9. torch/CUDA info
section("9. torch/CUDA environment")
try:
    import torch
    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")
        print(f"memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
except Exception as e:
    print(f"ERROR: {e}")

# 10. CUB header availability (for building .cu)
section("10. CUB headers for CoreX build")
cub_paths = [
    "/usr/local/corex-3.2.3/include/cub/block/block_scan.cuh",
    "/usr/local/corex/include/cub/block/block_scan.cuh",
    "/usr/include/cub/block/block_scan.cuh",
]
for p in cub_paths:
    print(f"  {'OK' if os.path.isfile(p) else '--':2s} {p}")

print(f"\n{'='*60}")
print("  DONE — paste this entire output back")
print(f"{'='*60}")
