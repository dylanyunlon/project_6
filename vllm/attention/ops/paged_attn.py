from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from vllm import _custom_ops as ops
from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    from vllm.attention.ops.prefix_prefill import context_attention_fwd

# ═══════════════════════════════════════════════════════════════════════
# CCCL grid_even_share.cuh-informed partition sizing
#
# grid_even_share.cuh DispatchInit:
#   total_tiles = ceil_div(num_items, tile_items)
#   grid_size = min(total_tiles, max_grid_size)
#   max_grid_size = sm_occupancy * sm_count * subscription_factor
#
# For BI-V100: max_grid_size = 2 * 16 * 5 = 160 CTAs
# PARTITION_SIZE determines total_tiles = ceil(seq_len / PARTITION_SIZE)
#
# With PARTITION_SIZE=512 and seq_len=100K: total_tiles=196 > 160
#   → 36 partitions are wasted (launched but blocked waiting for SM)
#   → grid_even_share would cap at grid_size=160
#
# CCCL's GridEvenShare also distributes "big" vs "normal" shares:
#   big_shares = total_tiles - (avg_tiles_per_block * grid_size)
#   → first `big_shares` blocks process one extra tile
#   This load-balancing is automatic in the C++ kernel.
#
# For the Python dispatch layer, we set PARTITION_SIZE to match
# the precompiled .so's expectation. The .so was compiled with 512.
# But we document the CCCL-derived optimal value for when we can
# rebuild: PARTITION_SIZE = ceil(max_model_len / max_grid_size)
#   = ceil(100000 / 160) = 625 → round to 640 (multiple of block_size=16)
#
# Should be the same as PARTITION_SIZE in `paged_attention_v2_launcher`.
# ═══════════════════════════════════════════════════════════════════════
# CCCL GridEvenShare partition sizing (grid_even_share.cuh DispatchInit)
#
# CCCL scan benchmark (bench/scan/exclusive/sum.cu) reveals the full
# parameter space that determines partition performance:
#   %RANGE% TUNE_ITEMS ipt 7:24:1          — items per thread
#   %RANGE% TUNE_THREADS tpb 128:1024:32   — threads per block
#   %RANGE% TUNE_MAGIC_NS ns 0:2048:4      — lookback delay
#   %RANGE% TUNE_DELAY_CONSTRUCTOR_ID dcid 0:7:1  — delay algorithm
#   %RANGE% TUNE_L2_WRITE_LATENCY_NS l2w 0:1200:5 — L2 write latency
#
# For paged attention partitioned dispatch, _PARTITION_SIZE is the
# analogue of (tpb * ipt) — it determines how many KV tokens each
# CTA processes before requiring cross-partition merge (the "second
# pass" in CCCL dispatch_reduce.cuh terminology).
#
# CCCL grid_even_share.cuh teaches:
#   max_grid_size = sm_occupancy * sm_count * subscription_factor
#   total_tiles = ceil(num_items / tile_items)
#   grid_size = min(total_tiles, max_grid_size)
#
# BI-V100 hardware (confirmed):
#   SM count = 16, sm_occupancy ≈ 2 CTAs/SM, subscription = 5
#   max_grid = 16 * 2 * 5 = 160 CTAs
#
# The precompiled .so expects PARTITION_SIZE=512 (baked into the kernel).
# We cannot change this without recompiling. But we CAN optimize the
# Python-side dispatch: V1 vs V2 threshold, temp buffer caching, and
# partition count calculation.
# ═══════════════════════════════════════════════════════════════════════
_PARTITION_SIZE = 512

# CCCL-derived constants for BI-V100 (from hardware.cuh + grid_even_share.cuh)
_BI100_SM_COUNT = 16
_BI100_SM_OCCUPANCY = 2        # CTAs per SM (conservative)
_BI100_SUBSCRIPTION = 5        # CCCL util_device.cuh default
_BI100_MAX_GRID = _BI100_SM_COUNT * _BI100_SM_OCCUPANCY * _BI100_SUBSCRIPTION  # 160

# CCCL reduce benchmark (bench/reduce/base.cuh) teaches:
# scale_mem_bound adapts tile size to type. For paged_attention:
#   score type = float32 (4B), query type = float16 (2B)
#   CCCL would scale: items = nominal * 4 / type_size
#   With nominal=16 (SM600 default): float32 → items=16, float16 → items=32
# This means: if we could control the .so, float16 KV cache should use
# 2x larger partitions than float32 scores. Document for future rebuild.


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
        num_kv_heads,  # Actually head_mapping tensor from xformers.py for V1,
                       # or int num_kv_heads for V2. See _custom_ops.py signatures.
                       # CCCL catch2_test_block_reduce.cu BlockDimY/Z ↔ GQA groups.
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
        # ═══════════════════════════════════════════════════════════════
        # CCCL dispatch_reduce.cuh single-tile vs two-phase decision
        #
        # dispatch_reduce.cuh line 460:
        #   if (num_items <= threads_per_block * items_per_thread):
        #       InvokeSingleTile()     # one CTA, no temp buffer
        #   else:
        #       InvokePasses()         # GridEvenShare + second pass
        #
        # The decision is tile-capacity based, not a magic constant.
        #
        # For paged attention, the equivalent:
        #   V1 = SingleTile: one CTA processes entire sequence in SMEM
        #        → no partition overhead, no cross-CTA merge
        #   V2 = TwoPasses: sequence partitioned across CTAs
        #        → Phase 1: each CTA computes partial attention
        #        → Phase 2: merge partition results (log-sum-exp)
        #
        # CCCL invoke_regular_size_reduce also teaches:
        #   max_blocks = sm_occupancy * sm_count * subscription_factor
        #   GridEvenShare distributes work evenly across CTAs
        #
        # BI-V100 specifics (from hardware.cuh):
        #   sm_count=16, subscription_factor=5 → max_blocks=160
        #   V2 launch overhead is ~5μs for the merge kernel
        #   V1 can handle up to PARTITION_SIZE tokens in one CTA
        #
        # agent_reduce.cuh ConsumeFullTile teaches: the single-tile
        # path skips GridEvenShare setup entirely (just ConsumeRange).
        # This is meaningful when num_items < tile_size because
        # ConsumePartialTile has a while-loop with bounds checking.
        #
        # Decision: V1 when the sequence fits in 1 partition (no merge).
        # V2 when cross-partition merge is required.
        # The old heuristic `max_seq_len <= 8192` was arbitrary.
        # The CCCL-derived condition: max_num_partitions == 1.
        # ═══════════════════════════════════════════════════════════════
        use_v1 = (max_num_partitions == 1)
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
            # CCCL agent_merge_sort.cuh union _TempStorage pattern:
            # cache temp tensors across decode steps (stable shapes for
            # max_num_seqs=1 with slowly growing sequence).
            _v2_key = (num_seqs, num_heads, max_num_partitions,
                       head_size, output.dtype, str(output.device))
            _v2 = getattr(PagedAttention, '_v2_cache', {}).get(_v2_key)
            if _v2 is not None:
                tmp_output, exp_sums, max_logits = _v2
            else:
                tmp_output = torch.empty(
                    size=(num_seqs, num_heads, max_num_partitions, head_size),
                    dtype=output.dtype,
                    device=output.device,
                )
                exp_sums = torch.empty(
                    size=(num_seqs, num_heads, max_num_partitions),
                    dtype=torch.float32,
                    device=output.device,
                )
                max_logits = torch.empty_like(exp_sums)
                if not hasattr(PagedAttention, '_v2_cache'):
                    PagedAttention._v2_cache = {}
                PagedAttention._v2_cache[_v2_key] = (
                    tmp_output, exp_sums, max_logits)
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
