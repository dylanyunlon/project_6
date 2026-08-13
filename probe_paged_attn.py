#!/usr/bin/env python3
"""Probe ixformer.vllm_single_query_cached_kv_attention signature and test."""
import inspect
import torch
import ixformer

# Print signature
fn = ixformer.vllm_single_query_cached_kv_attention
print(f"Signature: {inspect.signature(fn)}")

# Also check v2
if hasattr(ixformer, 'vllm_single_query_cached_kv_attention_v2'):
    fn2 = ixformer.vllm_single_query_cached_kv_attention_v2
    print(f"V2 Signature: {inspect.signature(fn2)}")

# Check contrib.vllm_flash_attn if available
try:
    from ixformer.contrib import vllm_flash_attn
    print(f"\nvllm_flash_attn dir: {[x for x in dir(vllm_flash_attn) if not x.startswith('_')]}")
except Exception as e:
    print(f"\nvllm_flash_attn: {e}")

# Check ixformer.vllm submodule
try:
    import ixformer.vllm as ixv
    print(f"\nixformer.vllm dir: {[x for x in dir(ixv) if not x.startswith('_')]}")
except Exception as e:
    print(f"\nixformer.vllm: {e}")
