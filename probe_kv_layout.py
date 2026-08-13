#!/usr/bin/env python3
"""Probe ixformer paged attention KV cache shape requirements."""
import torch
import ixformer

num_heads = 4
num_kv_heads = 1
head_dim = 256
block_size = 16
num_blocks = 4
context_len = num_blocks * block_size
head_mapping = torch.zeros(num_heads, dtype=torch.int32, device="cuda")
scale = head_dim ** -0.5
query = torch.randn(1, num_heads, head_dim, device="cuda", dtype=torch.float16)
context_lens = torch.tensor([context_len], device="cuda", dtype=torch.int32)
block_tables = torch.arange(num_blocks, device="cuda", dtype=torch.int32).unsqueeze(0)

# Read the ixformer vllm source for the correct layout
import inspect
src_file = "/usr/local/corex/lib64/python3/dist-packages/ixformer/functions/vllm.py"
try:
    with open(src_file) as f:
        print(f"=== {src_file} ===")
        print(f.read())
except:
    print(f"Cannot read {src_file}")

# Try different 5D layouts
print("\n=== Testing 5D KV cache layouts ===")
for x in [1, 2, 4, 8, 16]:
    if head_dim % x != 0:
        continue
    # Layout: (num_blocks, num_kv_heads, head_dim//x, block_size, x)
    kc = torch.randn(num_blocks, num_kv_heads, head_dim // x, block_size, x,
                       device="cuda", dtype=torch.float16)
    vc = torch.randn(num_blocks, num_kv_heads, head_dim // x, block_size, x,
                       device="cuda", dtype=torch.float16)
    out = torch.empty(1, num_heads, head_dim, device="cuda", dtype=torch.float16)
    try:
        ixformer.vllm_single_query_cached_kv_attention(
            out, query, kc, vc, head_mapping, scale,
            block_tables, context_lens, block_size, context_len)
        print(f"  x={x:2d} shape={kc.shape}: OK nan={out.isnan().any().item()}")
    except Exception as e:
        err = str(e)[:80]
        print(f"  x={x:2d} shape={kc.shape}: {err}")
