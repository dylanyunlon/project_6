#!/bin/bash
# probe_moe_fused_breakdown.sh — Time each step of MoE decode

python3 << 'PY'
import torch
import time
import importlib.util

spec = importlib.util.spec_from_file_location('m',
    'qwen3_6_scripts/prebuilt/corex-3.2.3-ivcore10/corex_batched_gemm.so')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

K = 8
H = 4096
I = 2752

x = torch.randn(1, H, dtype=torch.float16, device='cuda')
w13 = torch.randn(K, 2*I, H, dtype=torch.float16, device='cuda')
w2 = torch.randn(K, H, I, dtype=torch.float16, device='cuda')
ws = torch.softmax(torch.randn(K, device='cuda'), 0).half()

def time_fn(fn, name, iters=50):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / iters * 1000
    print(f"  {name}: {ms:.3f} ms")
    return ms

print("=== Step-by-step breakdown ===")

# Step 0: expand+contiguous
time_fn(lambda: x.expand(K, 1, H).contiguous(), "expand+contiguous (1,H)->(K,1,H)")

# Step 1: batched GEMM FC1 only
x_exp = x.expand(K, 1, H).contiguous()
time_fn(lambda: mod.batched_gemm_fp16(x_exp, w13), "batched_gemm FC1 (K,1,H)x(K,2I,H)")

# Step 2: silu * mul
gate_up = mod.batched_gemm_fp16(x_exp, w13).squeeze(1)
chunks = gate_up.chunk(2, dim=1)
time_fn(lambda: torch.sigmoid(chunks[0]) * chunks[0] * chunks[1], "silu*mul (K,I)")

# Step 3: batched GEMM FC2 only
act = (torch.sigmoid(chunks[0]) * chunks[0] * chunks[1]).unsqueeze(1)
time_fn(lambda: mod.batched_gemm_fp16(act, w2), "batched_gemm FC2 (K,1,I)x(K,H,I)")

# Step 4: weighted reduction
eo = mod.batched_gemm_fp16(act, w2).squeeze(1)
time_fn(lambda: (eo * ws.unsqueeze(1)).sum(0, True), "weighted_sum")

# Full fused
time_fn(lambda: mod.moe_decode_fused(x, w13, w2, ws), "moe_decode_fused (full)")

# Comparison: 8x F.linear loop
import torch.nn.functional as F
def flinear_loop():
    outs = []
    for i in range(K):
        gu = F.linear(x, w13[i])
        g, u = gu.chunk(2, dim=-1)
        a = F.silu(g) * u
        outs.append(F.linear(a, w2[i]))
    return sum(outs[i] * ws[i] for i in range(K))
time_fn(flinear_loop, "F.linear loop (baseline)")
PY
