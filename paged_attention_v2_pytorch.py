"""
paged_attention_v2_pytorch.py — BI-V100 PagedAttention V2 (CCCL-informed)
===========================================================================

Fills the `raise NotImplementedError()` hole in vllm/_custom_ops.py.

Algorithm: Partitioned attention with log-sum-exp reduction.
Architecture informed by CCCL patterns:
  - summary_statistics.cu: fuse multiple statistics in a single reduction pass
  - warp_reduce_shfl.cuh: accumulate (max, sum, weighted_output) as one compound type
  - block_reduce_warp_reductions.cuh: reduce across partitions via shared accumulators

Key optimization: Batched partition attention via reshaped 3D bmm.
  Instead of looping over P partitions with P × torch.bmm calls,
  reshape KV into [H, P*part_len, d] and Q into [H, 1, d], then
  slice scores into [H, P, part_len] for partition-wise softmax.
  This gives ONE bmm launch for all partitions.

  For seq_len=100K, PARTITION_SIZE=512:
    Before: 195 × bmm([H,1,d] @ [H,d,512]) = 195 kernel launches
    After:  1 × bmm([H,1,d] @ [H,d,100K]) + reshape = 1 kernel launch

  The partition-wise softmax is then a reshape + per-chunk operation:
    scores: [H, 100K] → [H, P, 512] → max/exp/sum per partition

Phase 2 reduction (cross-partition combine) follows CCCL's summary_statistics
binary_op pattern: combine (max_a, sum_a, out_a) with (max_b, sum_b, out_b)
using the numerically stable log-sum-exp rescaling.
"""

import torch
from typing import Optional

_PARTITION_SIZE = 512


def paged_attention_v2_pytorch(
    output: torch.Tensor,          # [num_seqs, num_heads, head_size]
    exp_sums: torch.Tensor,        # [num_seqs, num_heads, max_num_partitions]
    max_logits: torch.Tensor,      # [num_seqs, num_heads, max_num_partitions]
    tmp_output: torch.Tensor,      # [num_seqs, num_heads, max_num_partitions, head_size]
    query: torch.Tensor,           # [num_seqs, num_heads, head_size]
    key_cache: torch.Tensor,       # [num_blocks, num_kv_heads, head_size/x, block_size, x]
    value_cache: torch.Tensor,     # [num_blocks, num_kv_heads, head_size, block_size]
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,    # [num_seqs, max_blocks_per_seq]
    seq_lens: torch.Tensor,        # [num_seqs]
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str = "auto",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
) -> None:
    num_seqs, num_heads, head_size = query.shape
    gqa_ratio = num_heads // num_kv_heads
    max_num_partitions = tmp_output.shape[2]

    # Initialize unused slots
    max_logits.fill_(float('-inf'))
    exp_sums.zero_()
    tmp_output.zero_()

    for seq_idx in range(num_seqs):
        seq_len = int(seq_lens[seq_idx].item())
        if seq_len == 0:
            output[seq_idx].zero_()
            continue

        num_blocks_seq = (seq_len + block_size - 1) // block_size
        num_partitions = (seq_len + _PARTITION_SIZE - 1) // _PARTITION_SIZE

        # =============================================================
        # Batched KV gather: ONE index_select, ONE reshape
        # Pattern: avoid per-block Python loop (CCCL does this via
        # block-cooperative load, we do it via batched indexing)
        # =============================================================
        blk_ids = block_tables[seq_idx, :num_blocks_seq]

        # Key: [nblk, kv_h, d/x, blk_sz, x] → [nblk*blk_sz, kv_h, d]
        k_gathered = key_cache[blk_ids]
        k_flat = (k_gathered
                  .permute(0, 3, 1, 2, 4)
                  .reshape(-1, num_kv_heads, head_size))[:seq_len]

        # Value: [nblk, kv_h, d, blk_sz] → [nblk*blk_sz, kv_h, d]
        v_flat = (value_cache[blk_ids]
                  .permute(0, 3, 1, 2)
                  .reshape(-1, num_kv_heads, head_size))[:seq_len]

        if k_scale != 1.0:
            k_flat = k_flat.float().mul_(k_scale)
        if v_scale != 1.0:
            v_flat = v_flat.float().mul_(v_scale)

        # GQA: expand (zero-copy view) then reshape to contiguous for bmm
        # [seq_len, kv_h, d] → [seq_len, H, d]
        if gqa_ratio > 1:
            k_all = (k_flat.unsqueeze(2)
                     .expand(-1, -1, gqa_ratio, -1)
                     .reshape(seq_len, num_heads, head_size))
            v_all = (v_flat.unsqueeze(2)
                     .expand(-1, -1, gqa_ratio, -1)
                     .reshape(seq_len, num_heads, head_size))
        else:
            k_all = k_flat
            v_all = v_flat

        # =============================================================
        # Phase 1: ALL partitions in ONE bmm (CCCL transform_reduce pattern)
        #
        # Instead of: for p in range(195): bmm(Q, K_p)
        # We do:      scores = Q @ K_all^T  → [H, seq_len]
        #             reshape to [H, P, part_sz] → partition-wise softmax
        #
        # This is one kernel launch vs 195.
        # =============================================================
        q = query[seq_idx].float()  # [H, d]

        # Q @ K^T: [H, 1, d] @ [H, d, seq_len] → [H, 1, seq_len] → [H, seq_len]
        k_t = k_all.permute(1, 2, 0).float().contiguous()  # [H, d, seq_len]
        scores_all = torch.bmm(q.unsqueeze(1), k_t).squeeze(1) * scale  # [H, seq_len]

        # Alibi bias (if needed)
        if alibi_slopes is not None:
            positions = torch.arange(seq_len, device=query.device, dtype=torch.float32)
            scores_all = scores_all + alibi_slopes.unsqueeze(1) * positions.unsqueeze(0)

        # Pad to exact multiple of _PARTITION_SIZE for clean reshape
        padded_len = num_partitions * _PARTITION_SIZE
        if padded_len > seq_len:
            pad_size = padded_len - seq_len
            scores_padded = torch.full(
                (num_heads, padded_len), float('-inf'),
                dtype=scores_all.dtype, device=scores_all.device)
            scores_padded[:, :seq_len] = scores_all
        else:
            scores_padded = scores_all

        # Reshape: [H, padded_len] → [H, P, part_sz]
        scores_parts = scores_padded.view(num_heads, num_partitions, _PARTITION_SIZE)

        # Per-partition online softmax (vectorized over H and P simultaneously)
        # Pattern from CCCL summary_statistics: compute (max, sum) in one pass
        part_max = scores_parts.max(dim=-1).values          # [H, P]
        scores_exp = torch.exp(scores_parts - part_max.unsqueeze(-1))  # [H, P, part_sz]
        part_sum = scores_exp.sum(dim=-1)                    # [H, P]

        # Weighted values per partition: need V reshaped the same way
        # V: [seq_len, H, d] → pad → [padded_len, H, d] → [H, P, part_sz, d]
        v_perm = v_all.permute(1, 0, 2).float().contiguous()  # [H, seq_len, d]
        if padded_len > seq_len:
            v_padded = torch.zeros(
                (num_heads, padded_len, head_size),
                dtype=v_perm.dtype, device=v_perm.device)
            v_padded[:, :seq_len, :] = v_perm
        else:
            v_padded = v_perm
        v_parts = v_padded.view(num_heads, num_partitions, _PARTITION_SIZE, head_size)

        # Weighted sum: [H, P, 1, part_sz] @ [H, P, part_sz, d] → [H, P, 1, d] → [H, P, d]
        # Reshape for batched bmm: [H*P, 1, part_sz] @ [H*P, part_sz, d] → [H*P, 1, d]
        HP = num_heads * num_partitions
        scores_exp_flat = scores_exp.reshape(HP, 1, _PARTITION_SIZE)
        v_parts_flat = v_parts.reshape(HP, _PARTITION_SIZE, head_size)
        part_out_flat = torch.bmm(scores_exp_flat, v_parts_flat)  # [HP, 1, d]
        part_out = part_out_flat.view(num_heads, num_partitions, head_size)  # [H, P, d]

        # Store partition results
        max_logits[seq_idx, :, :num_partitions] = part_max
        exp_sums[seq_idx, :, :num_partitions] = part_sum
        tmp_output[seq_idx, :, :num_partitions, :] = part_out.to(tmp_output.dtype)

        # =============================================================
        # Phase 2: Cross-partition reduction (CCCL binary_op pattern)
        #
        # This is the summary_statistics.binary_op pattern:
        #   Combine (max_a, sum_a, out_a) ⊕ (max_b, sum_b, out_b)
        #   using numerically stable log-sum-exp rescaling.
        #
        # Fully vectorized — no loop over partitions.
        # =============================================================
        pm = max_logits[seq_idx, :, :num_partitions]     # [H, P]
        ps = exp_sums[seq_idx, :, :num_partitions]       # [H, P]
        po = tmp_output[seq_idx, :, :num_partitions, :]  # [H, P, d]

        global_max = pm.max(dim=-1).values               # [H]
        rescale = torch.exp(pm - global_max.unsqueeze(-1)) * ps  # [H, P]
        total = rescale.sum(dim=-1, keepdim=True)         # [H, 1]
        weights = rescale / total                          # [H, P]

        # [H, 1, P] @ [H, P, d] → [H, 1, d] → [H, d]
        final = torch.bmm(weights.unsqueeze(1), po.float()).squeeze(1)  # [H, d]
        output[seq_idx] = final.to(output.dtype)
