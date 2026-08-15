#!/bin/bash
# probe_real_machine.sh — 在真机上执行，cat所有关键数据
# 用法: bash probe_real_machine.sh | tee probe_output.txt
set -e

echo "========================================"
echo "  probe_real_machine.sh"
echo "  $(date)"
echo "========================================"

echo ""
echo "=== 1. ixformer Python包结构 ==="
python3 -c "
import ixformer
print('ixformer.__file__:', ixformer.__file__)
print('dir(ixformer):', [x for x in dir(ixformer) if not x.startswith('__')])
" 2>&1 || echo "FAIL: import ixformer"

echo ""
echo "=== 2. ixformer.functions ==="
python3 -c "
try:
    import ixformer.functions as F
    print('dir(ixformer.functions):', [x for x in dir(F) if not x.startswith('__')])
except Exception as e:
    print('FAIL:', e)
" 2>&1

echo ""
echo "=== 3. ixformer._C ==="
python3 -c "
try:
    import ixformer._C as C
    print('dir(ixformer._C):', [x for x in dir(C) if not x.startswith('__')])
except Exception as e:
    print('FAIL:', e)
" 2>&1

echo ""
echo "=== 4. 找topk_softmax在哪 ==="
python3 -c "
import ixformer
import os, importlib, pkgutil
root = os.path.dirname(ixformer.__file__)
for loader, name, ispkg in pkgutil.walk_packages([root], prefix='ixformer.'):
    try:
        mod = importlib.import_module(name)
        attrs = [a for a in dir(mod) if 'topk' in a.lower() or 'softmax' in a.lower()]
        if attrs:
            print(f'{name}: {attrs}')
    except:
        pass
" 2>&1 || echo "walk failed"

echo ""
echo "=== 5. grep topk in ixformer ==="
IXDIR=$(python3 -c "import ixformer; import os; print(os.path.dirname(ixformer.__file__))" 2>/dev/null)
if [ -n "$IXDIR" ]; then
    echo "ixformer dir: $IXDIR"
    grep -r "topk_softmax\|topk_soft\|moe_topk" "$IXDIR" --include="*.py" -l 2>/dev/null | head -10
    echo "---"
    grep -r "topk_softmax\|topk_soft\|moe_topk" "$IXDIR" --include="*.py" 2>/dev/null | head -20
fi

echo ""
echo "=== 6. base镜像 _custom_ops.py topk调用 ==="
VLLMDIR=$(python3 -c "import vllm; import os; print(os.path.dirname(vllm.__file__))" 2>/dev/null)
if [ -n "$VLLMDIR" ]; then
    echo "vllm dir: $VLLMDIR"
    grep -n "topk_softmax\|topk_soft" "$VLLMDIR/_custom_ops.py" 2>/dev/null | head -10
    echo "---"
    # cat完整的topk_softmax函数
    sed -n '/def topk_softmax/,/^def /p' "$VLLMDIR/_custom_ops.py" 2>/dev/null | head -30
fi

echo ""
echo "=== 7. base镜像的fused_moe调用 ==="
if [ -n "$VLLMDIR" ]; then
    grep -rn "topk_softmax\|FusedMoE\|fused_moe" "$VLLMDIR/model_executor/layers/fused_moe/" --include="*.py" 2>/dev/null | grep -v __pycache__ | head -20
fi

echo ""
echo "=== 8. .so文件在base镜像里的位置 ==="
find /usr/local/corex/lib/python3/dist-packages -name "*.so" -path "*/ixformer/*" 2>/dev/null | head -20
find /usr/local/corex/lib/python3/dist-packages -name "*.so" -path "*/vllm/*" 2>/dev/null | head -20

echo ""
echo "=== 9. libixinfer / libixattn ==="
find /usr/local/corex -name "libixinfer*" -o -name "libixattn*" -o -name "libixformer*" 2>/dev/null | head -10
ls -la /usr/local/corex/lib64/libix* 2>/dev/null | head -10

echo ""
echo "=== 10. torch CUDA能力 ==="
python3 -c "
import torch
print('torch.cuda.is_available():', torch.cuda.is_available())
print('torch.version.cuda:', torch.version.cuda)
if torch.cuda.is_available():
    print('device:', torch.cuda.get_device_name(0))
    print('capability:', torch.cuda.get_device_capability(0))
" 2>&1

echo ""
echo "=== 11. cublas batched gemm验证 ==="
python3 -c "
import torch
torch.cuda.set_device(0)
E, T, H, I = 8, 1, 4096, 11264
w = torch.randn(E, 2*I, H, device='cuda', dtype=torch.float16)
x = torch.randn(E, T, H, device='cuda', dtype=torch.float16)

# 方法1: torch.bmm (cublas batchedGemm)
import time
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(10):
    out = torch.bmm(x, w.transpose(1,2))
torch.cuda.synchronize()
t1 = time.perf_counter()
print(f'torch.bmm: {(t1-t0)/10*1000:.3f} ms, shape: {out.shape}')

# 方法2: 循环F.linear
t0 = time.perf_counter()
for _ in range(10):
    outs = []
    for e in range(E):
        outs.append(x[e] @ w[e].transpose(0,1))
    out2 = torch.stack(outs)
torch.cuda.synchronize()
t1 = time.perf_counter()
print(f'loop matmul: {(t1-t0)/10*1000:.3f} ms, shape: {out2.shape}')
" 2>&1

echo ""
echo "========================================"
echo "  probe complete"
echo "========================================"
