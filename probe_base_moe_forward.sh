#!/bin/bash
set -e

BASE="/usr/local/corex/lib/python3/dist-packages/vllm/model_executor/models/qwen3_5.py"

echo "=== base qwen3_5.py line count ==="
wc -l "$BASE"

echo ""
echo "=== _pure_pytorch_experts 完整函数 ==="
sed -n '/def _pure_pytorch_experts/,/^    def [a-z]/p' "$BASE" | head -200

echo ""
echo "=== forward 中调用 _pure_pytorch_experts 的上下文 ==="
grep -n -B5 -A5 "_pure_pytorch_experts\|corex_moe_direct\|corex_moe_weight\|corex_moe_exact\|corex_moe_topk" "$BASE" | head -100

echo ""
echo "=== corex_moe_direct_routed.w13 签名 ==="
python3 -c "
from vllm import corex_moe_direct_routed as m
import inspect
for name in dir(m):
    if not name.startswith('_'):
        obj = getattr(m, name)
        try:
            sig = inspect.signature(obj)
            print(f'{name}{sig}')
        except:
            print(f'{name}: {type(obj)}')
" 2>&1

echo ""
echo "=== corex_moe_topk_softmax.moe_topk_softmax 签名 ==="
python3 -c "
from vllm import corex_moe_topk_softmax as m
import inspect
for name in dir(m):
    if not name.startswith('_'):
        obj = getattr(m, name)
        try:
            sig = inspect.signature(obj)
            print(f'{name}{sig}')
        except:
            print(f'{name}: {type(obj)}')
" 2>&1

echo ""
echo "=== corex_moe_exact_reduce 签名 ==="
python3 -c "
from vllm import corex_moe_exact_reduce as m
import inspect
for name in dir(m):
    if not name.startswith('_'):
        obj = getattr(m, name)
        try:
            sig = inspect.signature(obj)
            print(f'{name}{sig}')
        except:
            print(f'{name}: {type(obj)}')
" 2>&1

echo ""
echo "=== corex_moe_weight_gather 签名 ==="
python3 -c "
from vllm import corex_moe_weight_gather as m
import inspect
for name in dir(m):
    if not name.startswith('_'):
        obj = getattr(m, name)
        try:
            sig = inspect.signature(obj)
            print(f'{name}{sig}')
        except:
            print(f'{name}: {type(obj)}')
" 2>&1

echo ""
echo "=== corex_moe_index_combine 签名 ==="
python3 -c "
from vllm import corex_moe_index_combine as m
import inspect
for name in dir(m):
    if not name.startswith('_'):
        obj = getattr(m, name)
        try:
            sig = inspect.signature(obj)
            print(f'{name}{sig}')
        except:
            print(f'{name}: {type(obj)}')
" 2>&1
