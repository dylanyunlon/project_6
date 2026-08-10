"""
EngineX cache operators.

KV cache management for paged attention.
CCCL parallel: dispatch_batch_memcpy (block copies between cache slots).
"""

from typing import Dict, List

import torch


def reshape_and_cache_pytorch(
    key: torch.Tensor,            # [num_tokens, num_kv_heads, head_size]
    value: torch.Tensor,          # [num_tokens, num_kv_heads, head_size]
    key_cache: torch.Tensor,      # [num_blocks, num_kv_heads, block_size, head_size]
    value_cache: torch.Tensor,    # [num_blocks, num_kv_heads, block_size, head_size]
    slot_mapping: torch.Tensor,   # [num_tokens] — maps token → (block, offset)
    kv_cache_dtype: str = "auto",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> None:
    """Write new K,V into their assigned cache slots."""
    num_tokens = key.shape[0]
    block_size = key_cache.shape[2]

    for i in range(num_tokens):
        slot = slot_mapping[i].item()
        if slot < 0:
            continue
        block_idx = slot // block_size
        block_offset = slot % block_size
        key_cache[block_idx, :, block_offset, :] = key[i] * k_scale
        value_cache[block_idx, :, block_offset, :] = value[i] * v_scale


def copy_blocks_pytorch(
    key_caches: List[torch.Tensor],
    value_caches: List[torch.Tensor],
    block_mapping: torch.Tensor,    # [num_pairs, 2] src→dst
) -> None:
    """Copy cache blocks (used for fork/copy-on-write)."""
    num_pairs = block_mapping.shape[0]
    num_layers = len(key_caches)

    for i in range(num_pairs):
        src = block_mapping[i, 0].item()
        dst = block_mapping[i, 1].item()
        for layer in range(num_layers):
            key_caches[layer][dst].copy_(key_caches[layer][src])
            value_caches[layer][dst].copy_(value_caches[layer][src])


def swap_blocks_pytorch(
    src: torch.Tensor,
    dst: torch.Tensor,
    block_mapping: torch.Tensor,
) -> None:
    """Swap cache blocks between GPU and CPU."""
    for i in range(block_mapping.shape[0]):
        src_idx = block_mapping[i, 0].item()
        dst_idx = block_mapping[i, 1].item()
        dst[dst_idx].copy_(src[src_idx])
