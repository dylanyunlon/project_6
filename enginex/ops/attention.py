"""
EngineX Attention operators.

Sub168 log shows three attention paths:
  1. CoreX FA2 packed prefill: B=2 Hq=4 Hkv=1 D=256 (full attention layers)
  2. CoreX paged FA2 chunked prefill: B=1 Hq=4 Hkv=1 D=256 cache_blocks=2
  3. CoreX GDN (handled in gdn.py, 4 of 36 layers)

Our image has:
  - libixattn.so (present but not wired)
  - ixformer.flash_attn_varlen_func (available)
  - xformers SDPA (current fallback, patched for head_dim=256)

CCCL parallel:
  paged_attention_v1 = dispatch_reduce (reduce over KV blocks)
  paged_attention_v2 = dispatch_reduce two-pass (partition-level reduce + final reduce)
"""

import math
from typing import List, Optional

import torch
import torch.nn.functional as F


def fa2_xformers_fallback(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
) -> torch.Tensor:
    """
    xformers SDPA fallback for FA2.
    This is what we currently use — works but slower than native FA2.
    Head_dim=256 bypass already applied in patch_xformers_sdpa_*.py.
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(query.shape[-1])

    # Standard scaled dot product attention
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * softmax_scale

    if causal and attn_weights.shape[-2] > 1:
        L = attn_weights.shape[-2]
        S = attn_weights.shape[-1]
        mask = torch.triu(
            torch.full((L, S), float('-inf'), device=query.device),
            diagonal=S - L + 1
        )
        attn_weights = attn_weights + mask

    attn_weights = F.softmax(attn_weights, dim=-1)
    output = torch.matmul(attn_weights, value)
    return output


def paged_attention_v1_pytorch(
    output: torch.Tensor,           # [num_seqs, num_heads, head_size]
    query: torch.Tensor,            # [num_seqs, num_heads, head_size]
    key_cache: torch.Tensor,        # [num_blocks, num_kv_heads, block_size, head_size]
    value_cache: torch.Tensor,      # [num_blocks, num_kv_heads, block_size, head_size]
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,     # [num_seqs, max_blocks_per_seq]
    seq_lens: torch.Tensor,         # [num_seqs]
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor] = None,
    kv_cache_dtype: str = "auto",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
) -> None:
    """
    Paged attention v1 — single-pass reduce over all KV blocks.

    CCCL parallel: dispatch_reduce single-tile kernel.
    For short sequences (< 2 × sm_count × partition_size), v1 is faster
    because it avoids the two-pass overhead.

    BI-V100 with 16 SMs: threshold ≈ 16 × 2 × 512 = 16384 tokens.
    """
    num_seqs = query.shape[0]
    num_heads = query.shape[1]
    head_size = query.shape[2]
    num_queries_per_kv = num_heads // num_kv_heads

    for seq_idx in range(num_seqs):
        seq_len = seq_lens[seq_idx].item()
        if seq_len == 0:
            continue

        q = query[seq_idx]  # [num_heads, head_size]

        num_blocks = (seq_len + block_size - 1) // block_size
        keys_list = []
        values_list = []

        for block_idx in range(num_blocks):
            physical_block = block_tables[seq_idx, block_idx].item()
            if block_idx == num_blocks - 1:
                # Last block may be partial
                tokens_in_block = seq_len - block_idx * block_size
            else:
                tokens_in_block = block_size

            k_block = key_cache[physical_block, :, :tokens_in_block, :]
            v_block = value_cache[physical_block, :, :tokens_in_block, :]
            keys_list.append(k_block)
            values_list.append(v_block)

        # Concatenate all KV
        all_keys = torch.cat(keys_list, dim=1)    # [num_kv_heads, seq_len, head_size]
        all_values = torch.cat(values_list, dim=1)

        # GQA: repeat KV heads
        if num_queries_per_kv > 1:
            all_keys = all_keys.repeat_interleave(num_queries_per_kv, dim=0)
            all_values = all_values.repeat_interleave(num_queries_per_kv, dim=0)

        # Attention: q @ k^T → softmax → @ v
        attn = torch.einsum('hd,hsd->hs', q, all_keys) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('hs,hsd->hd', attn, all_values)

        output[seq_idx].copy_(out)


def paged_attention_v2_pytorch(
    output: torch.Tensor,
    exp_sums: torch.Tensor,         # [num_seqs, num_heads, max_partitions]
    max_logits: torch.Tensor,       # [num_seqs, num_heads, max_partitions]
    tmp_output: torch.Tensor,       # [num_seqs, num_heads, max_partitions, head_size]
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor] = None,
    kv_cache_dtype: str = "auto",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
) -> None:
    """
    Paged attention v2 — two-pass reduce with partitioning.

    CCCL parallel: dispatch_reduce two-pass pattern.
    Pass 1: per-partition reduce (each partition = PARTITION_SIZE KV tokens)
    Pass 2: reduce across partitions (log-sum-exp correction)

    For BI-V100 with 16 SMs, v2 is better when seq_len > 8192 (multiple
    waves of partitions keep all SMs busy).
    """
    # For correctness, delegate to v1 — the two-pass optimization
    # only matters for perf on long sequences
    paged_attention_v1_pytorch(
        output, query, key_cache, value_cache,
        num_kv_heads, scale, block_tables, seq_lens,
        block_size, max_seq_len, alibi_slopes, kv_cache_dtype,
        k_scale, v_scale, tp_rank,
        blocksparse_local_blocks, blocksparse_vert_stride,
        blocksparse_block_size, blocksparse_head_sliding_step,
    )
