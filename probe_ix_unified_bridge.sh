#!/bin/bash
# probe_ix_unified_bridge.sh — cat ix_unified_bridge的完整接口
set -e

echo "=== ix_unified_bridge.so 函数列表 ==="
python3 -c "
import importlib.util, sys

# 方法1: 直接import
for path in [
    '/usr/local/corex/lib/python3/dist-packages/vllm/ix_unified_bridge.cpython-310-x86_64-linux-gnu.so',
    '/usr/local/corex/lib/python3/dist-packages/vllm/ix_unified_bridge.so',
]:
    try:
        spec = importlib.util.spec_from_file_location('ix_unified_bridge', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fns = [x for x in dir(mod) if not x.startswith('_')]
        print(f'loaded from: {path}')
        print(f'functions ({len(fns)}):')
        for f in sorted(fns):
            print(f'  {f}')
        break
    except Exception as e:
        print(f'  {path}: {e}')
"

echo ""
echo "=== ixformer 里跟vllm相关的函数签名 ==="
python3 -c "
import ixformer
import inspect

# 列出所有vllm_开头的函数
for name in sorted(dir(ixformer)):
    if 'vllm' in name.lower() or name in ['silu_and_mul', 'fused_add_rms_norm', 'rms_norm', 'flash_attn_func', 'linear', 'matmul', 'gemv', 'rotary_embedding']:
        obj = getattr(ixformer, name)
        if callable(obj):
            try:
                sig = inspect.signature(obj)
                print(f'{name}{sig}')
            except:
                print(f'{name}(...)')
"

echo ""
echo "=== corex_moe_topk_softmax.so 函数列表 ==="
python3 -c "
import importlib.util
path = '/usr/local/corex/lib/python3/dist-packages/vllm/corex_moe_topk_softmax.so'
try:
    spec = importlib.util.spec_from_file_location('corex_moe_topk_softmax', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fns = [x for x in dir(mod) if not x.startswith('_')]
    print(f'functions ({len(fns)}):')
    for f in sorted(fns):
        print(f'  {f}')
except Exception as e:
    print(f'FAIL: {e}')
"

echo ""
echo "=== 所有corex_*.so的函数列表 ==="
python3 -c "
import importlib.util, os, glob
for so in sorted(glob.glob('/usr/local/corex/lib/python3/dist-packages/vllm/corex_*.so')):
    name = os.path.basename(so).replace('.so','')
    try:
        spec = importlib.util.spec_from_file_location(name, so)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fns = [x for x in dir(mod) if not x.startswith('_')]
        print(f'{name}: {fns}')
    except Exception as e:
        print(f'{name}: FAIL {e}')
"

echo ""
echo "=== base镜像 _custom_ops.py 完整内容 ==="
VLLM_BASE="/usr/local/corex/lib/python3/dist-packages/vllm"
if [ -f "$VLLM_BASE/_custom_ops.py" ]; then
    cat "$VLLM_BASE/_custom_ops.py"
else
    echo "NOT FOUND at $VLLM_BASE/_custom_ops.py"
    # 搜索
    find /usr/local/corex -name "_custom_ops.py" -path "*/vllm/*" 2>/dev/null | head -5
fi

echo ""
echo "=== base镜像 qwen3_5.py MoE forward ==="
BASE_QWEN="$VLLM_BASE/model_executor/models/qwen3_5.py"
if [ -f "$BASE_QWEN" ]; then
    grep -n "topk_softmax\|FusedMoE\|fused_moe\|_pure_pytorch\|corex_moe\|ix_unified" "$BASE_QWEN" | head -30
else
    echo "NOT FOUND"
    find /usr/local/corex -name "qwen3_5.py" -path "*/models/*" 2>/dev/null | head -5
fi
