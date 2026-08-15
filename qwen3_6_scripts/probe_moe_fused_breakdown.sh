#!/bin/bash
# probe_moe_fused_breakdown.sh — Time each step of MoE decode

python3 << 'PY'
import torch
import torch.nn.functional as F
import time

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

print("=== Individual operation timings ===")

# Single F.linear (one expert FC1)
time_fn(lambda: F.linear(x, w13[0]), "F.linear FC1 1 expert (1,H)x(2I,H)")

# Single F.linear (one expert FC2)
gu = F.linear(x, w13[0])
g, u = gu.chunk(2, dim=-1)
a = F.silu(g) * u
time_fn(lambda: F.linear(a, w2[0]), "F.linear FC2 1 expert (1,I)x(H,I)")

# silu * mul
time_fn(lambda: F.silu(gu[:,:I]) * gu[:,I:], "silu*mul (1, I)")

# 8x F.linear loop (baseline)
def flinear_loop():
    outs = []
    for i in range(K):
        gu = F.linear(x, w13[i])
        g, u = gu.chunk(2, dim=-1)
        a = F.silu(g) * u
        outs.append(F.linear(a, w2[i]))
    return sum(outs[i] * ws[i] for i in range(K))
time_fn(flinear_loop, "F.linear loop 8 experts (FULL)")

# torch.mm loop (no F.linear overhead)
def mm_loop():
    outs = []
    for i in range(K):
        gu = torch.mm(x, w13[i].t())
        g, u = gu.chunk(2, dim=-1)
        a = F.silu(g) * u
        outs.append(torch.mm(a, w2[i].t()))
    return sum(outs[i] * ws[i] for i in range(K))
time_fn(mm_loop, "torch.mm loop 8 experts (FULL)")

# Batched via cublasHgemmStridedBatched (pre-gathered weights)
# First gather weights contiguously
print("\n=== Batched approaches ===")

# Measure gather cost
time_fn(lambda: w13.reshape(K, 2*I*H), "w13 reshape (view, should be free)")

# cublasHgemmStridedBatched via torch.bmm
x_exp = x.expand(K, 1, H).contiguous()
w13_t = w13.transpose(1, 2).contiguous()  # (K, H, 2I)
time_fn(lambda: w13.transpose(1, 2).contiguous(), "w13 transpose+contiguous (K,2I,H)->(K,H,2I)")
time_fn(lambda: torch.bmm(x_exp, w13_t), "torch.bmm FC1 (K,1,H)x(K,H,2I)")

# What if weights are pre-transposed?
print("\n=== Pre-transposed weights (no runtime copy) ===")
w13_pre = w13.transpose(1, 2).contiguous()  # (K, H, 2I) — do this once at model load
w2_pre = w2.transpose(1, 2).contiguous()    # (K, I, H)
time_fn(lambda: torch.bmm(x_exp, w13_pre), "torch.bmm FC1 pre-transposed")

gu = torch.bmm(x_exp, w13_pre).squeeze(1)
g, u = gu.chunk(2, dim=-1)
act = F.silu(g) * u
act_3d = act.unsqueeze(1)
time_fn(lambda: torch.bmm(act_3d, w2_pre), "torch.bmm FC2 pre-transposed")

def bmm_fused_pretransposed():
    gu = torch.bmm(x_exp, w13_pre).squeeze(1)
    g, u = gu.chunk(2, dim=-1)
    a = F.silu(g) * u
    eo = torch.bmm(a.unsqueeze(1), w2_pre).squeeze(1)
    return (eo * ws.unsqueeze(1)).sum(0, True)
time_fn(bmm_fused_pretransposed, "bmm full MoE (pre-transposed)")

print("\n=== Summary ===")
PY
