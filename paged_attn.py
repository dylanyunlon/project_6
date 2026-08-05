from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from vllm import _custom_ops as ops

from vllm.attention.ops.prefix_prefill import context_attention_fwd

# Should be the same as PARTITION_SIZE in `paged_attention_v2_launcher`.
# BI-V100 (16 SMs): 1024 tokens/partition → fewer partitions → fewer CTAs
# → less inter-CTA sync overhead in the V2 reduce pass.
# CCCL insight: GridEvenShare distributes tiles as
#   num_tiles = ceil(N / tile_size), CTAs_per_SM = ceil(num_tiles / sm_count).
# With 16 SMs and PARTITION_SIZE=512, a 100K-token sequence produces 196
# partitions → 12.3 CTAs/SM. With 1024, only 98 → 6.1 CTAs/SM, which
# matches the occupancy sweet spot observed in reduce benchmarks.
_PARTITION_SIZE = 1024

# Pre-allocated tensors for V2 reduce intermediates, following the same
# pattern as _moe_intermediate_cache in fused_moe.py.
# Eliminates 3 torch.empty (CUDA malloc) calls per decode step when V2 is active.
# Design source: CCCL dispatch_reduce.cuh alias_temporaries pattern.
_v2_cache = {}


@dataclass
class PagedAttentionMetadata:
    """Metadata for PagedAttention."""
    # (batch_size,). The length of sequences (entire tokens seen so far) per
    # sequence.
    seq_lens_tensor: Optional[torch.Tensor]
    # Maximum sequence length in the batch. 0 if it is prefill-only batch.
    max_decode_seq_len: int
    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    # E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    # in the kv cache. Each block can contain up to block_size tokens.
    # 2nd dimensions are padded up to max_blocks_per_seq if it is cuda-graph
    # captured.
    block_tables: Optional[torch.Tensor]


class PagedAttention:

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [64, 80, 96, 112, 120, 128, 192, 256]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (2, num_blocks, block_size * num_kv_heads * head_size)

    @staticmethod
    def split_kv_cache(
        kv_cache: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = 16 // kv_cache.element_size()
        num_blocks = kv_cache.shape[1]

        key_cache = kv_cache[0]
        key_cache = key_cache.view(num_blocks, num_kv_heads, head_size // x,
                                   -1, x)
        value_cache = kv_cache[1]
        value_cache = value_cache.view(num_blocks, num_kv_heads, head_size, -1)
        return key_cache, value_cache

    @staticmethod
    def write_to_paged_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: float,
        v_scale: float,
    ) -> None:
        ops.reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping.flatten(),
            kv_cache_dtype,
            k_scale,
            v_scale,
        )

    @staticmethod
    def forward_decode(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        kv_cache_dtype: str,
        num_kv_heads: int,
        scale: float,
        alibi_slopes: Optional[torch.Tensor],
        k_scale: float,
        v_scale: float,
        tp_rank: int = 0,
        blocksparse_local_blocks: int = 0,
        blocksparse_vert_stride: int = 0,
        blocksparse_block_size: int = 64,
        blocksparse_head_sliding_step: int = 0,
    ) -> torch.Tensor:
        if blocksparse_vert_stride is not None and blocksparse_vert_stride > 1:
            # use blocksparse paged attention
            block_size = value_cache.size(-1)
            assert (blocksparse_block_size > 0 and
                    blocksparse_block_size % block_size == 0), \
                (f"{blocksparse_block_size=} needs to be a multiple of"
                 f"{block_size=} used in block_tables.")

        output = torch.empty_like(query)
        block_size = value_cache.shape[3]
        num_seqs, num_heads, head_size = query.shape
        max_num_partitions = ((max_seq_len + _PARTITION_SIZE - 1) //
                              _PARTITION_SIZE)
        # NOTE(woosuk): We use a simple heuristic to decide whether to use
        # PagedAttention V1 or V2. If the number of partitions is 1, we use
        # V1 to avoid the overhead of reduction. Also, if the number of
        # sequences or heads is large, we use V1 since there is enough work
        # to parallelize.
        # TODO(woosuk): Tune this heuristic.
        # For context len > 8192, use V2 kernel to avoid shared memory shortage.
        use_v1 = (max_seq_len <= 8192
                  and (max_num_partitions == 1 or num_seqs * num_heads > 512))
        # V1/V2 adaptive dispatch (restored from original vllm logic):
        #
        # V1: single fused C++ kernel, iterates ALL KV blocks in one CTA.
        #   Best when: short sequences (≤8192), or enough seqs×heads for parallelism.
        #
        # V2: partitioned attention with cross-partition reduce.
        #   ops.paged_attention_v2 IS a C++ kernel (not pure PyTorch).
        #   Best when: long sequences where V1's single CTA cannot saturate 16 SMs.
        #
        # CCCL parallel: V2 reduce pass = DeviceReduce over compound accumulator
        # (max_logits, exp_sums, partial_output). The accumulator merge uses the
        # same online softmax pattern as thrust/examples/summary_statistics.cu.
        #
        # With PARTITION_SIZE=1024 and 16 SMs, V2 is beneficial for sequences
        # longer than 1024 * 16 * 2 = 32768 tokens (where V1 would have a single
        # CTA iterating for too long while other SMs sit idle).
        if use_v1:
            # Run PagedAttention V1.
            ops.paged_attention_v1(
                output,
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                scale,
                block_tables,
                seq_lens,
                block_size,
                max_seq_len,
                alibi_slopes,
            )
        else:
            # Run PagedAttention V2.
            assert _PARTITION_SIZE % block_size == 0

            # Pre-allocate V2 intermediate tensors (same pattern as MoE cache).
            # These shapes depend on (num_seqs, num_heads, max_num_partitions, head_size)
            # which are stable across decode steps within a batch.
            tmp_shape = (num_seqs, num_heads, max_num_partitions, head_size)
            sum_shape = (num_seqs, num_heads, max_num_partitions)
            cache_key = (tmp_shape, sum_shape, output.dtype, output.device)

            cached = _v2_cache.get("v2_tensors")
            if (cached is not None
                    and cached[0].shape == tmp_shape
                    and cached[0].dtype == output.dtype):
                tmp_output, exp_sums, max_logits = cached
            else:
                tmp_output = torch.empty(
                    size=tmp_shape,
                    dtype=output.dtype,
                    device=output.device,
                )
                exp_sums = torch.empty(
                    size=sum_shape,
                    dtype=torch.float32,
                    device=output.device,
                )
                max_logits = torch.empty_like(exp_sums)
                _v2_cache["v2_tensors"] = (tmp_output, exp_sums, max_logits)
            ops.paged_attention_v2(
                output,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                scale,
                block_tables,
                seq_lens,
                block_size,
                max_seq_len,
                alibi_slopes,
                kv_cache_dtype,
                k_scale,
                v_scale,
                tp_rank,
                blocksparse_local_blocks,
                blocksparse_vert_stride,
                blocksparse_block_size,
                blocksparse_head_sliding_step,
            )
        return output

    @staticmethod
    def forward_prefix(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache_dtype: str,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens_tensor: torch.Tensor,
        context_lens: torch.Tensor,
        max_query_len: int,
        alibi_slopes: Optional[torch.Tensor],
        sliding_window: Optional[int],
        k_scale: float,
        v_scale: float,
    ) -> torch.Tensor:
        output = torch.empty_like(query)
        context_attention_fwd(
            query,
            key,
            value,
            output,
            kv_cache_dtype,
            key_cache,
            value_cache,
            block_tables,
            # query_start_loc is (batch_size + 1,)
            query_start_loc[:-1],
            seq_lens_tensor,
            context_lens,
            max_query_len,
            k_scale,
            v_scale,
            alibi_slopes,
            sliding_window,
        )
        return output

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache = src_kv_cache[0]
        dst_key_cache = dst_kv_cache[0]
        ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)

        src_value_cache = src_kv_cache[1]
        dst_value_cache = dst_kv_cache[1]
        ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        key_caches = [kv_cache[0] for kv_cache in kv_caches]
        value_caches = [kv_cache[1] for kv_cache in kv_caches]
        ops.copy_blocks(key_caches, value_caches, src_to_dists)
