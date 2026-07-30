"""
paged_attention_v2_triton.py — Triton PagedAttention V2 for BI-V100
=====================================================================

Two-kernel V2 implementation using Triton:
  Phase 1: _paged_attn_v2_partition — per-partition attention (paged K/V gather)
  Phase 2: _paged_attn_v2_reduce — cross-partition log-sum-exp reduction

The K/V gather pattern is adapted from prefix_prefill.py (lines 100-170):
  bn = tl.load(block_tables + seq * stride + (token // block_size) * stride)
  off_k = bn * stride_kc_b + kv_head * stride_kc_h + (d // x) * stride_kc_dx + ...
  k = tl.load(key_cache + off_k, mask=...)

For decode (BLOCK_M=1), the Q tile is just one vector [HEAD_DIM].
The inner loop iterates over BLOCK_N KV tokens per step.
Online softmax accumulates (max, sum, weighted_V) across steps.

After all steps in a partition, we have:
  max_logits[seq, head, part]: running max
  exp_sums[seq, head, part]: running exp sum
  tmp_output[seq, head, part, :]: unnormalized weighted V

Phase 2 combines partitions using the CCCL summary_statistics pattern:
  global_max = max(part_maxes)
  rescaled_sum = sum(exp(part_max - global_max) * part_sum)
  output = sum(weight[p] * part_output[p])

SMEM analysis:
  Phase 1: K tile [BLOCK_N, HEAD_DIM] loaded via gather (no explicit SMEM tile)
           Triton manages register allocation for tl.load + tl.dot
           At BLOCK_N=32, HEAD_DIM=256: 32×256 fp16 values in registers = 16KB
  Phase 2: No SMEM needed (partitions ≈ 200, all in registers)
"""

import torch
import triton
import triton.language as tl
from typing import Optional


@triton.jit
def _paged_attn_v2_partition_kernel(
    # Outputs
    tmp_output_ptr,     # [num_seqs, num_heads, max_num_parts, head_size]
    exp_sums_ptr,       # [num_seqs, num_heads, max_num_parts]
    max_logits_ptr,     # [num_seqs, num_heads, max_num_parts]
    # Inputs
    query_ptr,          # [num_seqs, num_heads, head_size]
    key_cache_ptr,      # [num_blocks, num_kv_heads, head_size/x, block_size, x]
    value_cache_ptr,    # [num_blocks, num_kv_heads, head_size, block_size]
    block_tables_ptr,   # [num_seqs, max_blocks_per_seq]
    seq_lens_ptr,       # [num_seqs]
    # Scalars
    scale: tl.float32,
    num_queries_per_kv: tl.int32,
    block_size: tl.int32,
    x_pack: tl.int32,             # key_cache packing factor: 16 // sizeof(dtype)
    # Strides: query [S, H, D]
    stride_qs: tl.int32, stride_qh: tl.int32, stride_qd: tl.int32,
    # Strides: key_cache [B, KH, D/X, BS, X]
    stride_kc_b: tl.int32, stride_kc_h: tl.int32,
    stride_kc_dx: tl.int32, stride_kc_bs: tl.int32, stride_kc_x: tl.int32,
    # Strides: value_cache [B, KH, D, BS]
    stride_vc_b: tl.int32, stride_vc_h: tl.int32,
    stride_vc_d: tl.int32, stride_vc_bs: tl.int32,
    # Strides: block_tables [S, MAX_BLOCKS]
    stride_bt_s: tl.int32, stride_bt_b: tl.int32,
    # Strides: tmp_output [S, H, P, D]
    stride_to_s: tl.int32, stride_to_h: tl.int32,
    stride_to_p: tl.int32, stride_to_d: tl.int32,
    # Strides: exp_sums / max_logits [S, H, P]
    stride_es_s: tl.int32, stride_es_h: tl.int32, stride_es_p: tl.int32,
    # Compile-time constants
    PARTITION_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Phase 1: Per-partition paged attention for decode (BLOCK_M=1).

    Grid: (num_seqs, num_heads, max_num_partitions)
    Each program instance processes one (seq, head, partition) triple.

    Adapted from prefix_prefill.py's paged K/V gather pattern.
    Key difference: BLOCK_M=1 (decode has 1 query token per head).
    """
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    part_idx = tl.program_id(2)

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    part_start = part_idx * PARTITION_SIZE
    part_end = tl.minimum(part_start + PARTITION_SIZE, seq_len)

    if part_start >= seq_len:
        # Unused partition — write sentinel values
        tl.store(max_logits_ptr + seq_idx * stride_es_s + head_idx * stride_es_h + part_idx * stride_es_p,
                 float('-inf'))
        tl.store(exp_sums_ptr + seq_idx * stride_es_s + head_idx * stride_es_h + part_idx * stride_es_p,
                 0.0)
        return

    # GQA: map query head → KV head
    kv_head_idx = head_idx // num_queries_per_kv

    # Load query vector: [HEAD_DIM]
    offs_d = tl.arange(0, HEAD_DIM)
    q = tl.load(query_ptr + seq_idx * stride_qs + head_idx * stride_qh + offs_d * stride_qd).to(tl.float32)

    # Online softmax state
    m_i = float('-inf')       # running max
    l_i = 0.0                 # running exp sum
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)  # weighted V accumulator

    # KV token offsets within each BLOCK_N chunk
    offs_n = tl.arange(0, BLOCK_N)

    # Iterate over BLOCK_N KV tokens at a time
    for start_n in range(part_start, part_end, BLOCK_N):
        # Token positions in the sequence
        token_ids = start_n + offs_n
        valid_mask = token_ids < part_end

        # === Paged K gather (from prefix_prefill.py pattern) ===
        # Look up physical block numbers from block_tables
        block_indices = token_ids // block_size
        within_block = token_ids % block_size

        # bn: physical block ids [BLOCK_N]
        bn = tl.load(
            block_tables_ptr + seq_idx * stride_bt_s + block_indices * stride_bt_b,
            mask=valid_mask, other=0)

        # K offsets: key_cache[bn, kv_head, d//x, within_block, d%x]
        # Layout: [num_blocks, num_kv_heads, head_size/x, block_size, x]
        # off_k: [HEAD_DIM, BLOCK_N] — each column is one token's K vector
        off_k = (bn[None, :] * stride_kc_b +
                 kv_head_idx * stride_kc_h +
                 (offs_d[:, None] // x_pack) * stride_kc_dx +
                 within_block[None, :] * stride_kc_bs +
                 (offs_d[:, None] % x_pack) * stride_kc_x)

        k = tl.load(key_cache_ptr + off_k, mask=valid_mask[None, :], other=0.0)  # [D, N]

        # Scores: q @ k = [1, D] @ [D, N] → [N]
        # For BLOCK_M=1: this is a dot product per KV token
        scores = tl.sum(q[:, None] * k, axis=0) * scale  # [BLOCK_N]
        scores = tl.where(valid_mask, scores, float('-inf'))

        # Online softmax (adapted from prefix_prefill.py — proven correct)
        m_ij = tl.max(scores, axis=0)  # scalar: max of this chunk
        p = tl.exp(scores - m_ij)      # [BLOCK_N] — unnormalized probs
        l_ij = tl.sum(p, axis=0)       # scalar: sum of exp for this chunk

        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)  # rescale factor for old accumulator
        beta = tl.exp(m_ij - m_i_new)  # rescale factor for new chunk
        l_i_new = alpha * l_i + beta * l_ij

        # === Paged V gather ===
        off_v = (bn[:, None] * stride_vc_b +
                 kv_head_idx * stride_vc_h +
                 offs_d[None, :] * stride_vc_d +
                 within_block[:, None] * stride_vc_bs)
        v = tl.load(value_cache_ptr + off_v, mask=valid_mask[:, None], other=0.0)  # [N, D]

        # Update accumulator (Flash Attention online softmax pattern):
        #   acc = acc * (alpha * l_i / l_i_new) + (p * beta / l_i_new) @ V
        # Safe division: if l_i_new == 0, this is the first chunk
        acc_scale = alpha * l_i / tl.maximum(l_i_new, 1e-6)
        acc = acc * acc_scale
        p_scale = beta / tl.maximum(l_i_new, 1e-6)
        p_scaled = p * p_scale  # [BLOCK_N]
        acc += tl.sum(p_scaled[:, None] * v, axis=0)  # [HEAD_DIM]

        l_i = l_i_new
        m_i = m_i_new

    # Store partition results
    tl.store(max_logits_ptr + seq_idx * stride_es_s + head_idx * stride_es_h + part_idx * stride_es_p,
             m_i)
    tl.store(exp_sums_ptr + seq_idx * stride_es_s + head_idx * stride_es_h + part_idx * stride_es_p,
             l_i)

    # Store accumulated output: [HEAD_DIM]
    out_base = seq_idx * stride_to_s + head_idx * stride_to_h + part_idx * stride_to_p
    tl.store(tmp_output_ptr + out_base + offs_d * stride_to_d, acc.to(tmp_output_ptr.dtype.element_ty))


@triton.jit
def _paged_attn_v2_reduce_kernel(
    # Output
    output_ptr,        # [num_seqs, num_heads, head_size]
    # Inputs
    tmp_output_ptr,    # [num_seqs, num_heads, max_num_parts, head_size]
    exp_sums_ptr,      # [num_seqs, num_heads, max_num_parts]
    max_logits_ptr,    # [num_seqs, num_heads, max_num_parts]
    seq_lens_ptr,      # [num_seqs]
    # Scalars
    max_num_parts: tl.int32,
    # Strides
    stride_out_s: tl.int32, stride_out_h: tl.int32, stride_out_d: tl.int32,
    stride_to_s: tl.int32, stride_to_h: tl.int32,
    stride_to_p: tl.int32, stride_to_d: tl.int32,
    stride_es_s: tl.int32, stride_es_h: tl.int32, stride_es_p: tl.int32,
    # Constants
    PARTITION_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    MAX_NUM_PARTS: tl.constexpr,
):
    """Phase 2: Cross-partition log-sum-exp reduction.

    Grid: (num_seqs, num_heads)
    Combines partition results using CCCL summary_statistics pattern.
    """
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    num_parts = (seq_len + PARTITION_SIZE - 1) // PARTITION_SIZE

    # Load partition statistics
    part_offsets = tl.arange(0, MAX_NUM_PARTS)
    valid_mask = part_offsets < num_parts

    es_base = seq_idx * stride_es_s + head_idx * stride_es_h
    part_max = tl.load(max_logits_ptr + es_base + part_offsets * stride_es_p,
                       mask=valid_mask, other=float('-inf'))
    part_sum = tl.load(exp_sums_ptr + es_base + part_offsets * stride_es_p,
                       mask=valid_mask, other=0.0)

    # Global max
    global_max = tl.max(part_max, axis=0)

    # Rescale and normalize
    rescale = tl.exp(part_max - global_max) * part_sum
    total = tl.sum(rescale, axis=0)
    weights = rescale / total  # [MAX_NUM_PARTS]

    # Weighted sum of partition outputs
    offs_d = tl.arange(0, HEAD_DIM)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for p in range(MAX_NUM_PARTS):
        if p < num_parts:
            w = tl.load(max_logits_ptr + es_base + p * stride_es_p)  # reload for weight
            w_rescaled = tl.exp(w - global_max) * tl.load(exp_sums_ptr + es_base + p * stride_es_p) / total

            to_base = seq_idx * stride_to_s + head_idx * stride_to_h + p * stride_to_p
            part_out = tl.load(tmp_output_ptr + to_base + offs_d * stride_to_d)
            acc += w_rescaled * part_out.to(tl.float32)

    # Store final output
    out_base = seq_idx * stride_out_s + head_idx * stride_out_h
    tl.store(output_ptr + out_base + offs_d * stride_out_d, acc.to(output_ptr.dtype.element_ty))


def paged_attention_v2_triton(
    output: torch.Tensor,
    exp_sums: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str = "auto",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    **kwargs,
) -> None:
    """Launch Triton V2 kernels."""
    num_seqs, num_heads, head_size = query.shape
    num_queries_per_kv = num_heads // num_kv_heads
    max_num_parts = tmp_output.shape[2]
    x_pack = key_cache.shape[-1]  # packing factor

    PARTITION_SIZE = 512
    # BLOCK_N: must fit in SMEM. For decode (BLOCK_M=1), SMEM is dominated by K/V gather.
    # head_dim=256: BLOCK_N=32 → 32×256×2 = 16KB per tile (K or V)
    # head_dim=128: BLOCK_N=64 → 64×128×2 = 16KB per tile
    BLOCK_N = 32 if head_size > 128 else 64

    # Phase 1: partition attention
    num_partitions = (max_seq_len + PARTITION_SIZE - 1) // PARTITION_SIZE
    grid_phase1 = (num_seqs, num_heads, num_partitions)

    _paged_attn_v2_partition_kernel[grid_phase1](
        tmp_output, exp_sums, max_logits,
        query, key_cache, value_cache, block_tables, seq_lens,
        scale, num_queries_per_kv, block_size, x_pack,
        # query strides
        query.stride(0), query.stride(1), query.stride(2),
        # key_cache strides
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
        key_cache.stride(3), key_cache.stride(4),
        # value_cache strides
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
        value_cache.stride(3),
        # block_tables strides
        block_tables.stride(0), block_tables.stride(1),
        # tmp_output strides
        tmp_output.stride(0), tmp_output.stride(1), tmp_output.stride(2), tmp_output.stride(3),
        # exp_sums strides
        exp_sums.stride(0), exp_sums.stride(1), exp_sums.stride(2),
        # Constants
        PARTITION_SIZE=PARTITION_SIZE,
        HEAD_DIM=head_size,
        BLOCK_N=BLOCK_N,
    )

    # Phase 2: cross-partition reduction
    MAX_NUM_PARTS_CONST = triton.next_power_of_2(max_num_parts)
    if MAX_NUM_PARTS_CONST > 1024:
        MAX_NUM_PARTS_CONST = 1024

    grid_phase2 = (num_seqs, num_heads)
    _paged_attn_v2_reduce_kernel[grid_phase2](
        output,
        tmp_output, exp_sums, max_logits, seq_lens,
        max_num_parts,
        output.stride(0), output.stride(1), output.stride(2),
        tmp_output.stride(0), tmp_output.stride(1), tmp_output.stride(2), tmp_output.stride(3),
        exp_sums.stride(0), exp_sums.stride(1), exp_sums.stride(2),
        PARTITION_SIZE=PARTITION_SIZE,
        HEAD_DIM=head_size,
        MAX_NUM_PARTS=MAX_NUM_PARTS_CONST,
    )
