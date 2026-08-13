#!/usr/bin/env python3
"""Find KV cache layout from vllm + test paged attn with correct shapes."""
import torch
import ixformer

# Read vllm's _custom_ops to find the x value
try:
    from vllm._custom_ops import get_cache_block_size
    print("Has get_cache_block_size")
except:
    pass

# Check vllm worker for cache layout
import vllm.worker.cache_engine as ce
import inspect
src = inspect.getsource(ce)
# Find references to key_cache shape
for line in src.split('\n'):
    if 'x' in line.lower() and ('cache' in line.lower() or 'block' in line.lower()):
        if 'shape' in line.lower() or 'size' in line.lower() or 'dim' in line.lower():
            print(f"  {line.strip()}")

# Also check _custom_ops for reshape_and_cache
try:
    from vllm import _custom_ops
    src2 = inspect.getsource(_custom_ops)
    for line in src2.split('\n'):
        if 'reshape_and_cache' in line or 'key_cache' in line:
            print(f"  {line.strip()}")
except:
    pass

# Direct approach: check what vllm uses for x
# In vllm 0.6.3, x = 16 // dtype_size (for fp16: x = 16/2 = 8)
print("\n=== Testing with vllm standard layout ===")
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

for x in [1, 2, 4, 8, 16]:
    if head_dim % x != 0:
        continue
    # key_cache: 5D (num_blocks, num_kv_heads, head_dim//x, block_size, x)
    # value_cache: 4D (num_blocks, num_kv_heads, head_dim, block_size)
    kc = torch.randn(num_blocks, num_kv_heads, head_dim // x, block_size, x,
                       device="cuda", dtype=torch.float16)
    vc = torch.randn(num_blocks, num_kv_heads, head_dim, block_size,
                       device="cuda", dtype=torch.float16)
    out = torch.empty(1, num_heads, head_dim, device="cuda", dtype=torch.float16)
    try:
        ixformer.vllm_single_query_cached_kv_attention(
            out, query, kc, vc, head_mapping, scale,
            block_tables, context_lens, block_size, context_len)
        nan = out.isnan().any().item()
        print(f"  x={x:2d} key={kc.shape} val={vc.shape}: OK nan={nan}")
    except Exception as e:
        err = str(e)[:100]
        print(f"  x={x:2d} key={kc.shape} val={vc.shape}: {err}")
