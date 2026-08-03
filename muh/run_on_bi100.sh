#!/bin/bash
# muh/run_on_bi100.sh — Run on Phanthy Cloud BI-V100
# Paste this entire script into the terminal on the BI-V100 machine.
# It will: diagnose hardware → confirm SMEM → run Triton benchmark → output results
set -e

echo "=========================================="
echo "muh BI-V100 diagnostic + benchmark"
echo "=========================================="

cd ~/project_6

# --- 1. Hardware confirmation ---
echo ""
echo "=== STEP 1: Hardware diagnostics ==="
python3 -c "
import torch
print(f'torch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'GPU {i}: {p.name}')
    print(f'  SMs: {p.multi_processor_count}')
    print(f'  SMEM/block: {p.max_shared_memory_per_block}')
    print(f'  Total VRAM: {p.total_mem // 1024**2} MiB')
    print(f'  Major.Minor: {p.major}.{p.minor}')
"

# --- 2. SMEM 32KB vs 48KB test ---
echo ""
echo "=== STEP 2: SMEM actual limit test ==="
python3 -c "
import torch
# The critical question: is SMEM 32KB or 48KB?
# torch.cuda.get_device_properties tells us the hardware max.
# But ixformer _custom_ops.py hardcodes 32KB.
# Let's check what the driver reports.
p = torch.cuda.get_device_properties(0)
print(f'Hardware SMEM/block: {p.max_shared_memory_per_block} bytes')
print(f'  = {p.max_shared_memory_per_block / 1024} KB')
if p.max_shared_memory_per_block >= 49152:
    print('  → 48KB confirmed. _custom_ops.py 32KB is WRONG/conservative.')
elif p.max_shared_memory_per_block >= 32768:
    print('  → 32KB confirmed. _custom_ops.py 32KB is CORRECT.')
else:
    print(f'  → Unexpected value: {p.max_shared_memory_per_block}')
"

# --- 3. Check Triton availability ---
echo ""
echo "=== STEP 3: Triton availability ==="
python3 -c "
try:
    import triton
    print(f'triton: {triton.__version__}')
    print('Triton JIT: available')
except ImportError as e:
    print(f'Triton NOT available: {e}')
    print('Cannot run Triton kernel benchmarks.')
"

# --- 4. Check if prefix_prefill kernel can be imported ---
echo ""
echo "=== STEP 4: Kernel import test ==="
python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from prefix_prefill import _fwd_kernel
    print('prefix_prefill._fwd_kernel: imported OK')
except Exception as e:
    print(f'prefix_prefill import failed: {e}')
    # Try vllm path
    try:
        from vllm.attention.ops.prefix_prefill import _fwd_kernel
        print('vllm.attention.ops.prefix_prefill._fwd_kernel: imported OK')
    except Exception as e2:
        print(f'vllm path also failed: {e2}')
"

# --- 5. Quick functional test: can Triton compile a kernel on this GPU? ---
echo ""
echo "=== STEP 5: Triton compilation test ==="
python3 -c "
import torch
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _test_kernel(X, Y, N: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * N + tl.arange(0, N)
        x = tl.load(X + offs)
        tl.store(Y + offs, x + 1.0)

    x = torch.randn(1024, device='cuda')
    y = torch.empty_like(x)
    _test_kernel[(4,)](x, y, N=256)
    torch.cuda.synchronize()
    diff = (y - (x + 1.0)).abs().max().item()
    print(f'Triton compile+run: OK (max error={diff:.2e})')
except Exception as e:
    print(f'Triton compile FAILED: {e}')
"

# --- 6. Run the actual benchmark (if all above passed) ---
echo ""
echo "=== STEP 6: Triton prefill kernel benchmark ==="
echo "(Each BLOCK×WARPS combo triggers Triton recompilation into different PTX)"
echo ""

python3 muh/bench_triton_prefill.py \
    --block 16 32 64 --block-n 16 32 64 \
    --warps 1 2 4 8 \
    --ctx-lens 128 512 2048 8192 \
    --batch 1 --seq-len 1 \
    --head-dim 128 --num-heads 64 --num-kv-heads 8 \
    --dtype float16 \
    --warmup 3 --repeats 10 \
    --output results/prefill_bench.json \
    2>&1

echo ""
echo "=== STEP 7: Show vllm launch config ==="
cat computility-run.yaml

echo ""
echo "=== STEP 8: Check fused_moe BLOCK_SIZE_M path ==="
python3 -c "
# Check what BLOCK_SIZE_M values ixformer actually receives
import sys
sys.path.insert(0, '.')
from vllm.model_executor.layers.fused_moe.fused_moe import get_default_config
for M in [1, 8, 16, 64, 128, 256, 1024]:
    cfg = get_default_config(M, 128, 5504, 2048, 8, 'float16', False)
    print(f'  M={M:>5d} → BLOCK_SIZE_M={cfg[\"BLOCK_SIZE_M\"]}')
"

echo ""
echo "=========================================="
echo "DONE. Paste all output back to Claude."
echo "=========================================="
