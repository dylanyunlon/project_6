#!/bin/bash
# probe_torch_matmul_backend.sh — Find out what torch.matmul actually calls on BI-V100

echo "=== 1. torch.matmul profiling — which kernel launches ==="
python3 << 'PY'
import torch
import os

M, N, K = 256, 11008, 4096
A = torch.randn(M, K, dtype=torch.float16, device='cuda')
B = torch.randn(K, N, dtype=torch.float16, device='cuda')

# Warmup
for _ in range(5):
    torch.matmul(A, B)
torch.cuda.synchronize()

# Profile with torch profiler
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    for _ in range(3):
        torch.matmul(A, B)
    torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
PY

echo ""
echo "=== 2. CUDA_LAUNCH_BLOCKING trace ==="
CUDA_LAUNCH_BLOCKING=1 python3 << 'PY'
import torch
import os

M, N, K = 256, 11008, 4096
A = torch.randn(M, K, dtype=torch.float16, device='cuda')
B = torch.randn(K, N, dtype=torch.float16, device='cuda')

# Warmup
torch.matmul(A, B)
torch.cuda.synchronize()

# Single call with sync
torch.cuda.synchronize()
C = torch.matmul(A, B)
torch.cuda.synchronize()
print("torch.matmul completed")
print(f"Result shape: {C.shape}, dtype: {C.dtype}")
PY

echo ""
echo "=== 3. Check if ixformer.matmul is faster or same as torch.matmul ==="
python3 << 'PY'
import torch
import time

M, N, K = 256, 11008, 4096
A = torch.randn(M, K, dtype=torch.float16, device='cuda')
B = torch.randn(K, N, dtype=torch.float16, device='cuda')

# torch.matmul
for _ in range(10):
    torch.matmul(A, B)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    torch.matmul(A, B)
torch.cuda.synchronize()
ms_torch = (time.time() - t0) / 100 * 1000

# torch.mm (should be same)
for _ in range(10):
    torch.mm(A, B)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    torch.mm(A, B)
torch.cuda.synchronize()
ms_mm = (time.time() - t0) / 100 * 1000

# ixformer.matmul
import ixformer
A_ix = ixformer.Tensor(A)
B_ix = ixformer.Tensor(B)
for _ in range(10):
    ixformer.matmul(A_ix, B_ix)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    ixformer.matmul(A_ix, B_ix)
torch.cuda.synchronize()
ms_ix = (time.time() - t0) / 100 * 1000

# ixformer.linear
A2 = torch.randn(M, K, dtype=torch.float16, device='cuda')
W = torch.randn(N, K, dtype=torch.float16, device='cuda')  # (out, in)
A2_ix = ixformer.Tensor(A2)
W_ix = ixformer.Tensor(W)
for _ in range(10):
    ixformer.linear(A2_ix, W_ix)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    ixformer.linear(A2_ix, W_ix)
torch.cuda.synchronize()
ms_linear = (time.time() - t0) / 100 * 1000

# F.linear (torch)
for _ in range(10):
    torch.nn.functional.linear(A2, W)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(100):
    torch.nn.functional.linear(A2, W)
torch.cuda.synchronize()
ms_flinear = (time.time() - t0) / 100 * 1000

# cublas via direct cublasHgemm (through torch C++ extension would need compile)
# Skip for now

print(f"torch.matmul:       {ms_torch:.3f} ms")
print(f"torch.mm:           {ms_mm:.3f} ms")
print(f"ixformer.matmul:    {ms_ix:.3f} ms")
print(f"ixformer.linear:    {ms_linear:.3f} ms")
print(f"F.linear:           {ms_flinear:.3f} ms")
PY

echo ""
echo "=== 4. MoE hot loop: Python for-loop overhead measurement ==="
python3 << 'PY'
import torch
import time

num_experts = 8
top_k = 8
K, N = 4096, 11008

# Simulate decode: 1 token, 8 experts
hidden = torch.randn(1, K, dtype=torch.float16, device='cuda')
weights = [torch.randn(N, K, dtype=torch.float16, device='cuda') for _ in range(num_experts)]

# Warmup
for w in weights:
    torch.nn.functional.linear(hidden, w)
torch.cuda.synchronize()

# Measure: 8 F.linear calls in Python loop
t0 = time.time()
for _ in range(1000):
    results = []
    for i in range(top_k):
        results.append(torch.nn.functional.linear(hidden, weights[i]))
    out = sum(results)
torch.cuda.synchronize()
ms_loop = (time.time() - t0) / 1000 * 1000

# Measure: single F.linear with same total FLOPS
big_w = torch.randn(N * top_k, K, dtype=torch.float16, device='cuda')
big_hidden = hidden.expand(top_k, K).contiguous().view(top_k, K)
t0 = time.time()
for _ in range(1000):
    out = torch.mm(big_hidden, big_w.t())
torch.cuda.synchronize()
ms_single = (time.time() - t0) / 1000 * 1000

print(f"8x F.linear loop (decode): {ms_loop:.3f} ms")
print(f"1x torch.mm equivalent:    {ms_single:.3f} ms")
print(f"Python loop overhead:      {ms_loop - ms_single:.3f} ms")
print(f"Per-expert overhead:       {(ms_loop - ms_single)/top_k:.3f} ms")
PY
