#!/bin/bash
set -e

echo "=== 1. .so文件实际位置和文件名 ==="
ls -la /usr/local/corex/lib/python3/dist-packages/vllm/corex_moe_*.so 2>/dev/null
ls -la /usr/local/corex/lib/python3/dist-packages/vllm/ix_*.so 2>/dev/null
echo ""

echo "=== 2. Python import路径 ==="
python3 -c "
import vllm, os
vllm_dir = os.path.dirname(vllm.__file__)
print('vllm.__file__:', vllm.__file__)
print('vllm dir:', vllm_dir)
# 列出vllm目录下所有.so
for f in sorted(os.listdir(vllm_dir)):
    if f.endswith('.so'):
        print(f'  {f}')
"

echo ""
echo "=== 3. 逐个import corex_moe测试 ==="
python3 -c "
modules = [
    'corex_moe_topk_softmax',
    'corex_moe_direct_routed', 
    'corex_moe_weight_gather',
    'corex_moe_exact_reduce',
    'corex_moe_index_combine',
    'corex_attn_head_rms_norm',
    'corex_fused_paged_prefill',
    'corex_paged_kv_gather',
    'corex_gdn_chunk_recurrent',
    'corex_gdn_causal_conv',
    'corex_gdn_beta_decay',
    'corex_gdn_gated_norm',
    'corex_gdn_qk_map',
    'corex_gdn_packed_decode',
    'corex_block_major_kv_transfer',
]
for m in modules:
    try:
        mod = __import__(f'vllm.{m}', fromlist=[m])
        fns = [x for x in dir(mod) if not x.startswith('_')]
        print(f'  ✓ from vllm import {m} → {fns}')
    except ImportError as e:
        print(f'  ✗ from vllm import {m} → {e}')
"

echo ""
echo "=== 4. ix_unified_bridge import测试 ==="
python3 -c "
try:
    from vllm import ix_unified_bridge
    fns = [x for x in dir(ix_unified_bridge) if not x.startswith('_')]
    print(f'  ✓ ix_unified_bridge: {fns}')
except ImportError as e:
    print(f'  ✗ ix_unified_bridge: {e}')
"

echo ""
echo "=== 5. 我们的qwen3_5.py里各flag的实际值 ==="
python3 -c "
import sys, os
# 模拟qwen3_5.py的import环境
sys.path.insert(0, '/usr/local/corex/lib/python3/dist-packages')
os.environ.setdefault('BI100_MOE_COREX_TOPK_SOFTMAX', '1')
os.environ.setdefault('BI100_MOE_COREX_WEIGHT_GATHER', '1')
os.environ.setdefault('BI100_MOE_COREX_DIRECT_ROUTED', '0')
os.environ.setdefault('BI100_MOE_COREX_EXACT_REDUCE', '1')

def env_bool(key, default):
    v = os.environ.get(key, str(default))
    return v.lower() in ('1', 'true', 'yes')

flags = {}

# corex_moe_topk_softmax
try:
    from vllm import corex_moe_topk_softmax as _m
    flags['_USE_COREX_MOE_TOPK_SOFTMAX'] = _m is not None and env_bool('BI100_MOE_COREX_TOPK_SOFTMAX', True)
except:
    flags['_USE_COREX_MOE_TOPK_SOFTMAX'] = False

# corex_moe_direct_routed
try:
    from vllm import corex_moe_direct_routed as _m
    flags['_USE_COREX_MOE_DIRECT_ROUTED'] = _m is not None and env_bool('BI100_MOE_COREX_DIRECT_ROUTED', False)
except:
    flags['_USE_COREX_MOE_DIRECT_ROUTED'] = False

# corex_moe_weight_gather
try:
    from vllm import corex_moe_weight_gather as _m
    flags['_USE_COREX_MOE_WEIGHT_GATHER'] = _m is not None and env_bool('BI100_MOE_COREX_WEIGHT_GATHER', True)
except:
    flags['_USE_COREX_MOE_WEIGHT_GATHER'] = False

# corex_moe_exact_reduce
try:
    from vllm import corex_moe_exact_reduce as _m
    flags['_USE_COREX_MOE_EXACT_REDUCE'] = _m is not None and env_bool('BI100_MOE_COREX_EXACT_REDUCE', True)
except:
    flags['_USE_COREX_MOE_EXACT_REDUCE'] = False

# corex_moe_index_combine
try:
    from vllm import corex_moe_index_combine as _m
    flags['_USE_COREX_MOE_INDEX_COMBINE'] = _m is not None and env_bool('BI100_MOE_COREX_INDEX_COMBINE', True)
except:
    flags['_USE_COREX_MOE_INDEX_COMBINE'] = False

# ix_fused_moe
try:
    from vllm.model_executor.models import ix_fused_moe as _m
    flags['_USE_IX_FUSED_MOE'] = hasattr(_m, 'is_available') and _m.is_available()
except:
    flags['_USE_IX_FUSED_MOE'] = False

# naive_batched
try:
    from ex_engine.moe.naive_batched_experts import naive_batched_moe_forward
    flags['_USE_NAIVE_BATCHED_MOE'] = True
except:
    flags['_USE_NAIVE_BATCHED_MOE'] = False

# corex_batched_gemm
try:
    from vllm import corex_batched_gemm as _m
    flags['_USE_COREX_BATCHED_GEMM'] = _m is not None
except:
    try:
        from qwen3_6_scripts.prebuilt import corex_batched_gemm as _m
        flags['_USE_COREX_BATCHED_GEMM'] = _m is not None
    except:
        flags['_USE_COREX_BATCHED_GEMM'] = False

for k, v in sorted(flags.items()):
    status = '✓' if v else '✗'
    print(f'  {status} {k} = {v}')
"

echo ""
echo "=== 6. 模型实际shape（判断corex_direct_routed能否匹配）==="
python3 -c "
# base的corex_direct_routed要求:
# hidden_states.shape == (1, 2048)
# w13.shape == (256, 256, 2048)
# w2.shape == (256, 2048, 128)
# eids.shape == (8,) ws.shape == (8,)
# 
# Qwen3.5-27B的实际shape是什么?
print('Qwen3.5-27B MoE config (from config.json):')
print('  num_experts = 128 (per TP shard: 128/4=32? or 128?)')
print('  top_k = 8')
print('  hidden_size = 3584 (per TP shard: 3584/4=896? or 3584?)')
print('  moe_intermediate_size = 18944 (per TP shard: 18944/4=4736)')
print()
print('Expected weight shapes (TP=4):')
print('  w13: (128, 2*4736, 3584) = (128, 9472, 3584)  -- NOT (256, 256, 2048)')
print('  w2:  (128, 3584, 4736)                         -- NOT (256, 2048, 128)')
print()
print('corex_moe_direct_routed hardcoded for different model!')
print('We need corex_moe_weight_gather + F.linear path instead.')
" 2>&1
