"""在真机上运行：python3 probe_all_so.py
输出每个.so的全部导出Python方法"""
import importlib.util, os, sys

SO_DIR = None
for d in [
    "/usr/local/corex/lib64/python3/dist-packages/vllm",
    "/usr/local/corex/lib/python3/dist-packages/vllm",
]:
    if os.path.isfile(os.path.join(d, "corex_gdn_causal_conv.so")):
        SO_DIR = d
        break

if not SO_DIR:
    # try prebuilt
    SO_DIR = os.path.join(os.path.dirname(__file__), 
                          "qwen3_6_scripts/prebuilt/corex-3.2.3-ivcore10")

ALL = [
    "corex_gdn_causal_conv",
    "corex_gdn_packed_decode",
    "corex_gdn_beta_decay",
    "corex_gdn_qk_map",
    "corex_gdn_gated_norm",
    "corex_attn_head_rms_norm",
    "corex_paged_kv_gather",
    "corex_fused_paged_prefill",
    "corex_block_major_kv_transfer",
    "corex_moe_direct_routed",
    "corex_moe_exact_reduce",
    "corex_moe_weight_gather",
]

for name in ALL:
    so = os.path.join(SO_DIR, f"{name}.so")
    if not os.path.isfile(so):
        print(f"✗ {name}: NOT FOUND at {so}")
        continue
    try:
        spec = importlib.util.spec_from_file_location(name, so)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        funcs = [x for x in dir(mod) if not x.startswith('_')]
        print(f"✓ {name}: {funcs}")
        # Try to get docstrings/signatures
        for f in funcs:
            obj = getattr(mod, f)
            doc = getattr(obj, '__doc__', '')
            if doc:
                print(f"    {f}: {doc.strip()[:200]}")
    except Exception as e:
        print(f"✗ {name}: {e}")
