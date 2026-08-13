#!/usr/bin/env bash
# Quick test for CCCL preload allocator on BI-V100
# Usage: cd /home/dylan/project_6/qwen3_6_scripts && bash test_cccl_preload.sh
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SO="${SCRIPT_DIR}/cccl_preload_allocator.so"

echo "=== Step 1: Build ==="
bash "${SCRIPT_DIR}/build_cccl_preload_allocator.sh" "${SCRIPT_DIR}"
echo ""

if [[ ! -f "$SO" ]]; then
    echo "BUILD FAILED: $SO not found"
    exit 1
fi

echo "=== Step 2: Basic smoke test (torch.zeros on GPU) ==="
echo "Without preload:"
python3 -c "
import time, torch
t0=time.time()
for i in range(100):
    x=torch.zeros(1024*1024, device='cuda')
    del x
    torch.cuda.synchronize()
print(f'100 alloc+free cycles: {time.time()-t0:.3f}s')
"

echo ""
echo "With CCCL preload:"
CCCL_ALLOC_DEBUG=0 LD_PRELOAD="$SO" python3 -c "
import time, torch
t0=time.time()
for i in range(100):
    x=torch.zeros(1024*1024, device='cuda')
    del x
    torch.cuda.synchronize()
print(f'100 alloc+free cycles: {time.time()-t0:.3f}s')
" 2>&1

echo ""
echo "=== Step 3: Varied sizes (simulating model inference allocations) ==="
CCCL_ALLOC_DEBUG=0 LD_PRELOAD="$SO" python3 -c "
import time, torch

# Simulate inference: repeated allocs of same sizes (should hit cache)
sizes = [512, 4096, 32768, 262144, 1048576, 4194304, 16777216]  # 512B to 16MB
tensors = []

print('First pass (cold cache):')
t0 = time.time()
for s in sizes:
    x = torch.empty(s // 2, dtype=torch.float16, device='cuda')  # s bytes
    tensors.append(x)
t1 = time.time()
print(f'  {len(sizes)} allocs: {(t1-t0)*1000:.1f}ms')

print('Free all:')
del tensors
torch.cuda.synchronize()
t2 = time.time()
print(f'  {len(sizes)} frees: {(t2-t1)*1000:.1f}ms')

print('Second pass (warm cache - should be faster):')
tensors2 = []
for s in sizes:
    x = torch.empty(s // 2, dtype=torch.float16, device='cuda')
    tensors2.append(x)
t3 = time.time()
print(f'  {len(sizes)} allocs: {(t3-t2)*1000:.1f}ms')

print('Third pass (reuse same sizes 100x):')
for _ in range(100):
    for s in sizes:
        x = torch.empty(s // 2, dtype=torch.float16, device='cuda')
        del x
t4 = time.time()
print(f'  700 alloc+free: {(t4-t3)*1000:.1f}ms ({(t4-t3)/700*1000000:.0f}μs/op)')
" 2>&1

echo ""
echo "=== Step 4: Large allocation test (model weights sized) ==="
CCCL_ALLOC_DEBUG=0 LD_PRELOAD="$SO" python3 -c "
import torch
# Simulate KV cache blocks (typical: 256KB-2MB each)
blocks = []
for i in range(100):
    b = torch.empty(256*1024 // 2, dtype=torch.float16, device='cuda')
    blocks.append(b)
print(f'Allocated 100 x 256KB blocks = {100*256/1024:.0f}MB')
del blocks
torch.cuda.synchronize()
print('Freed all blocks')
# Reallocate (should hit cache)
blocks2 = []
for i in range(100):
    b = torch.empty(256*1024 // 2, dtype=torch.float16, device='cuda')
    blocks2.append(b)
print('Re-allocated 100 blocks (from cache)')
print('OK: large allocation test passed')
" 2>&1

echo ""
echo "=== Step 5: Stats output ==="
CCCL_ALLOC_DEBUG=0 LD_PRELOAD="$SO" python3 -c "
import torch
for _ in range(50):
    x = torch.zeros(1024*1024, device='cuda')
    del x
# Stats print on process exit
" 2>&1

echo ""
echo "=== DONE ==="
echo "If all tests passed, add to your launch command:"
echo "  LD_PRELOAD=$SO python3 -m vllm.entrypoints.openai.api_server ..."
echo ""
echo "Or set in computility-run.yaml env:"
echo "  - name: LD_PRELOAD"
echo "    value: /workspace/qwen3_6_scripts/cccl_preload_allocator.so"
