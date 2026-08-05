from dataclasses import dataclass
from typing import List, Optional, Tuple
import sys
import torch
import traceback
from vllm import _custom_ops as ops

# from vllm.attention.ops.prefix_prefill import context_attention_fwd
# NOTE: context_attention_fwd (Triton kernel from prefix_prefill.py) is NOT
# imported here.  On Iluvatar BI-V100 that kernel hangs the GPU card
# permanently.  Chunked-prefill / prefix-caching attention is handled by
# _forward_prefix_pytorch below (pure PyTorch, no Triton dependency).

# Should be the same as PARTITION_SIZE in `paged_attention_v2_launcher`.
_PARTITION_SIZE = 512


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
    def _forward_decode_pytorch(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Pure-PyTorch decode attention for long contexts (no hardware kernel).

        Architecture mirrors CCCL's three-layer reduce:
          dispatch_reduce.cuh  →  kernel_reduce.cuh  →  agent_reduce.cuh
          (work distribution)     (kernel entry)         (tile consumption)

        CCCL agent_reduce.cuh has two key patterns we translate here:

        1. ConsumeFullTile vectorized path: data loaded as VectorT in striped
           access (no BlockLoad staging → no SMEM for data, only for BlockReduce
           scratch). PyTorch equivalent: single reshape+view without .contiguous()
           when possible; fall back to one .contiguous() per K/V gather.

        2. ConsumeTiles with GridEvenShare STRIP_MINE: each CTA strides across
           the input with stride = grid_size * tile_items. For decode (q_len=1),
           we tile over KV blocks with adaptive tile_sz per the same
           GridEvenShare formula: max_tiles = sm_count * subscription_factor.

        3. summary_statistics.cu compound reduce: accumulator = {m, l, o}.
           unary_op: score_tile → (max, sum_exp, weighted_V).
           binary_op: online softmax merge with correction factor.
           This is the Flash Attention online softmax — identical structure.

        For decode, q_len=1 per sequence. The attention weight is [H, 1, seq_len]
        which is small (~5 MB at 50K tokens). We tile over KV blocks to control
        peak memory and apply online softmax (Flash Attention Algorithm 1) per tile.

        Shapes
        ------
        query       : [num_seqs, num_heads, head_dim]
        key_cache   : [num_blocks, num_kv_heads, head_dim//x, block_size, x]
        value_cache : [num_blocks, num_kv_heads, head_dim,    block_size]
        block_tables: [num_seqs, max_blocks_per_seq]
        seq_lens    : [num_seqs]
        """
        num_seqs, num_heads, head_dim = query.shape
        num_kv_heads = key_cache.shape[1]
        block_size = value_cache.shape[3]
        gqa_ratio = num_heads // num_kv_heads
        orig_dtype = query.dtype
        dev = query.device

        output = torch.empty_like(query)

        # ================================================================
        # CCCL spread_out_items_per_thread adaptive tile sizing for decode
        #
        # Ported from dispatch_transform.cuh::spread_out_items_per_thread
        # and dispatch_reduce.cuh::InvokePasses GridEvenShare.
        #
        # CCCL formula (dispatch_transform.cuh line 183):
        #   items = min(max_items,
        #               ceil_div(num_items, sm_count * threads * max_occupancy))
        #   items = clamp(items, min_items, max_items)
        #
        # Our translation for PyTorch decode:
        #   "items" = KV blocks per tile (how much work per matmul call)
        #   "num_items" = total KV blocks in the sequence
        #   "sm_count * max_occupancy" = target number of tiles (~4-8)
        #   Fewer tiles = fewer Python loop iterations = less launch overhead
        #
        # For decode (q_len=1), score tensor per tile is tiny:
        #   kv_h × gqa × 1 × (tile_blocks × block_size) × 4 bytes
        #   = 4 × 6 × 1 × 16384 × 4 = 1.5 MB  (even at kv_h=4, safe)
        # So the constraint is NOT memory — it's minimizing loop iterations.
        #
        # CCCL grid_even_share.cuh DispatchInit logic:
        #   total_tiles = ceil_div(num_items, tile_size)
        #   grid_size = min(total_tiles, max_grid_size)
        #   big_shares = total_tiles - (avg_tiles * grid_size)
        # Our target: ~4 tiles max (Python overhead >> kernel launch overhead)
        # ================================================================
        _BI100_TARGET_TILES = 4   # minimize Python loop iterations
        _MIN_TILE_BLOCKS = 64     # floor: avoid tiny matmuls
        _MAX_TILE_BLOCKS = 4096   # ceiling: avoid single huge allocation

        try:
            for i in range(num_seqs):
                seq_len = int(seq_lens[i].item())
                if seq_len == 0:
                    output[i].zero_()
                    continue

                num_blocks_i = (seq_len + block_size - 1) // block_size
                blk_ids = block_tables[i, :num_blocks_i]

                # Q reshaped once: [kv_h, gqa, 1, d] fp32 — tiny for decode
                q_grouped = (query[i].float()
                             .view(num_kv_heads, gqa_ratio, head_dim)
                             .unsqueeze(2)
                             .mul_(scale))

                # Online softmax accumulators (CCCL summary_stats_data pattern)
                # accumulator = {m (running max), l (running sum_exp), o (running output)}
                m = torch.full((num_kv_heads, gqa_ratio, 1),
                               float('-inf'), dtype=torch.float32, device=dev)
                l = torch.zeros_like(m)
                o = torch.zeros((num_kv_heads, gqa_ratio, 1, head_dim),
                                dtype=torch.float32, device=dev)

                # Tile over KV blocks — CCCL spread_out_items_per_thread pattern
                # Adaptive: tile_blocks = ceil(num_blocks / target_tiles)
                # clamped to [_MIN_TILE_BLOCKS, _MAX_TILE_BLOCKS]
                tile_blocks = max(_MIN_TILE_BLOCKS,
                                  min(_MAX_TILE_BLOCKS,
                                      (num_blocks_i + _BI100_TARGET_TILES - 1)
                                      // _BI100_TARGET_TILES))
                for tile_start in range(0, num_blocks_i, tile_blocks):
                    tile_end = min(tile_start + tile_blocks, num_blocks_i)
                    tile_blk_ids = blk_ids[tile_start:tile_end]

                    # Valid tokens in this tile
                    tile_token_start = tile_start * block_size
                    tile_token_end = min(tile_end * block_size, seq_len)
                    valid_tokens = tile_token_end - tile_token_start

                    # --------------------------------------------------------
                    # KV gather — agent_reduce.cuh ConsumeFullTile pattern
                    #
                    # agent_reduce loads VectorT in striped access when possible.
                    # PyTorch equivalent: reshape the 5D cache layout to 3D in
                    # one permute+contiguous, avoiding the double-contiguous
                    # pattern of the old code.
                    #
                    # key_cache shape: [num_blocks, kv_h, d//x, blk_sz, x]
                    # Target: [kv_h, d, valid_tokens] for Q@K^T
                    #
                    # Optimized path: permute(1,2,4,0,3) → [kv_h, d//x, x, n_blk, blk_sz]
                    # → reshape to [kv_h, d, n_blk*blk_sz] → slice [:valid_tokens]
                    # This is ONE contiguous() call instead of TWO.
                    # --------------------------------------------------------
                    k_gathered = key_cache[tile_blk_ids]  # [n, kv_h, d//x, blk_sz, x]
                    k_t = (k_gathered
                           .permute(1, 2, 4, 0, 3)       # [kv_h, d//x, x, n, blk_sz]
                           .contiguous()
                           .view(num_kv_heads, head_dim, -1)  # [kv_h, d, n*blk_sz]
                           [:, :, :valid_tokens]
                           .unsqueeze(1)                  # [kv_h, 1, d, valid]
                           .float())
                    del k_gathered

                    v_gathered = value_cache[tile_blk_ids]  # [n, kv_h, d, blk_sz]
                    v_t = (v_gathered
                           .permute(1, 2, 0, 3)           # [kv_h, d, n, blk_sz]
                           .contiguous()
                           .view(num_kv_heads, head_dim, -1)  # [kv_h, d, n*blk_sz]
                           [:, :, :valid_tokens]
                           .transpose(1, 2)               # [kv_h, valid, d]
                           .unsqueeze(1)                  # [kv_h, 1, valid, d]
                           .float())
                    del v_gathered

                    # --------------------------------------------------------
                    # Scores + online softmax — summary_statistics.cu pattern
                    #
                    # unary_op: score_tile → (max, sum_exp, weighted_V)
                    # binary_op: merge with correction factor
                    #
                    # CCCL summary_stats_binary_op merges:
                    #   result.mean = x.mean + delta * y.n / n
                    #   result.M2  = x.M2 + y.M2 + delta² * x.n * y.n / n
                    #
                    # Online softmax merge:
                    #   m_new  = max(m_old, m_tile)
                    #   corr   = exp(m_old - m_new)        ← rescale factor
                    #   l_new  = l_old * corr + l_tile
                    #   o_new  = o_old * corr + tile_exp @ V
                    #
                    # Structurally identical: m↔max, l↔n, o↔mean×n.
                    # --------------------------------------------------------

                    # [kv_h, gqa, 1, valid_tokens]
                    s = torch.matmul(q_grouped, k_t)
                    del k_t

                    # Online softmax update (Flash Attention Algorithm 1)
                    m_tile = s.amax(dim=-1, keepdim=True)    # [kv_h, gqa, 1, 1]
                    m_new = torch.maximum(m, m_tile.squeeze(-1))
                    corr = torch.exp(m - m_new)              # rescale old accum

                    exp_s = torch.exp(s - m_new.unsqueeze(-1))
                    del s

                    m.copy_(m_new)
                    l.mul_(corr).add_(exp_s.sum(dim=-1))
                    o.mul_(corr.unsqueeze(-1)).add_(torch.matmul(exp_s, v_t))
                    del exp_s, v_t, corr, m_new, m_tile

                # Finalize: normalize
                o.div_(l.unsqueeze(-1))
                output[i] = (o.view(num_heads, head_dim)
                              .to(orig_dtype))

        except Exception as e:
            print(f"[decode_pytorch ERROR] {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            raise

        return output

    # ================================================================
    # CCCL Design Pattern: summary_statistics.cu transform_reduce
    #
    # CCCL packs {n, min, max, mean, M2, M3, M4} into one struct and
    # computes ALL statistics in a single pass via transform_reduce.
    # The binary_op merges two partial results (Welford parallel algo).
    #
    # Our online softmax is the same pattern:
    #   accumulator = {m (running max), l (running sum_exp), o (running output)}
    #   unary_op: score_tile → {max(tile), sum(exp(tile-max)), exp(tile-max) @ V}
    #   binary_op: merge two accumulators with correction factor
    #
    # Key insight: kv_heads are INDEPENDENT — no cross-head dependency.
    # Current code already batches via [kv_h, gqa, q_len, tile_sz] tensor ops.
    # The CCCL pattern validates this is optimal: one matmul per tile across
    # all heads simultaneously, not per-head iteration.
    #
    # Future optimization: if we ever get Triton/CUDA access, the binary_op
    # merge step ({m,l,o} update) could be fused with the matmul via a
    # custom epilogue — this is what FlashAttention-2/3 does at the CUDA level.
    # ================================================================

    # paged_attention_v1 on BI-V100 fails for long contexts.
    # Route on actual sequence length (seq_lens.max()), not the max_seq_len
    # parameter which is inflated to max_model_len in CUDA graph mode.
    _PYTORCH_DECODE_THRESHOLD = 32768

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
        actual_max = int(seq_lens.max().item()) if seq_lens.numel() > 0 else max_seq_len
        if actual_max > PagedAttention._PYTORCH_DECODE_THRESHOLD:
            return PagedAttention._forward_decode_pytorch(
                query, key_cache, value_cache, block_tables, seq_lens, scale)

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
        # CCCL dispatch_reduce.cuh two-path dispatch architecture:
        #   single-tile:  num_items ≤ threads × items  →  one CTA, zero temp buffer
        #   multi-tile:   GridEvenShare partitions across sm_count × occupancy CTAs
        #
        # Paged attention equivalent:
        #   V1 = single-pass: one CTA iterates ALL KV blocks (like DeviceReduceSingleTileKernel)
        #   V2 = partitioned: KV blocks split into PARTITION_SIZE chunks across CTAs,
        #        then a second kernel merges partition results (like InvokePasses two-phase)
        #
        # V1 is optimal when seq_len fits in one CTA's tile (small context).
        # V2 is optimal when seq_len >> PARTITION_SIZE (long context) — parallelism
        # across partitions compensates for the merge overhead.
        #
        # CCCL's GridEvenShare formula:
        #   max_blocks = sm_occupancy × sm_count × subscription_factor
        #   BI-V100: ~1 × 16 × 5 = 80 max blocks
        #   V2 becomes worthwhile when max_num_partitions > 1 AND the partition
        #   parallelism exceeds the sequence×head parallelism.
        #
        # Original heuristic (before hardcode): V1 when max_seq_len ≤ 8192 OR
        # when batch×heads already saturates the GPU (num_seqs*num_heads > 512).
        # Restored with BI-V100 SM count awareness.
        bi100_sm_count = 16
        bi100_saturation = bi100_sm_count * 32  # ~512 concurrent warps
        use_v1 = (max_num_partitions == 1
                  or max_seq_len <= 8192
                  or num_seqs * num_heads > bi100_saturation)
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
        # NOTE: The Triton context_attention_fwd kernel hangs on Iluvatar
        # BI-V100 hardware (same class of issue as cudnnFlashAttnForward).
        # Use a pure-PyTorch fallback that reads the paged KV cache directly.
        return PagedAttention._forward_prefix_pytorch(
            query, key, value,
            key_cache, value_cache,
            block_tables, query_start_loc,
            seq_lens_tensor, context_lens,
        )

    @staticmethod
    def _forward_prefix_pytorch(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens_tensor: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Pure-PyTorch prefix-attention with K-tiling (Flash-Attention online softmax).

        Memory complexity: O(q_len), independent of kv_len.
        With chunked prefill (q_len ≤ max_num_batched_tokens = 4096) peak
        per layer ≈ 96 MB regardless of context length.

        Algorithm: Flash Attention online softmax.
        Q is reshaped once to [kv_h, gqa, q_len, d] (24 MB) and held for all
        K-tiles.  For each tile a running (m, l, o) accumulator is updated —
        the [q_len × kv_len] attention matrix is NEVER materialised in full.

        Tile budget (kv_h=1, gqa=6, q_len=4096, tile=256 tokens):
            q_seq   [1, 6, 4096, 256] fp32  24 MB  (held all tiles)
            o_acc   same shape               24 MB  (held all tiles)
            s       same shape               24 MB  (per tile, freed before exp_s)
            exp_s   same shape               24 MB  (per tile, brief overlap with s)
            Peak ≈ 96 MB  (s and exp_s briefly coexist during update).

        Shapes
        ------
        query          : [total_q_tokens, num_q_heads,  head_dim]
        key            : [total_q_tokens, num_kv_heads, head_dim]
        value          : [total_q_tokens, num_kv_heads, head_dim]
        key_cache      : [num_blocks, num_kv_heads, head_dim//x, block_size, x]
        value_cache    : [num_blocks, num_kv_heads, head_dim,    block_size]
        block_tables   : [batch_size, max_blocks_per_seq]
        query_start_loc: [batch_size + 1]
        seq_lens_tensor: [batch_size]  total length (context + query)
        context_lens   : [batch_size]  tokens already in KV cache
        """
        try:
            # ================================================================
            # Tile sizing strategy — ported from CCCL dispatch_reduce.cuh
            #
            # CCCL's GridEvenShare computes:
            #   max_blocks = sm_occupancy × sm_count × subscription_factor
            #   tile_size  = num_items / max_blocks  (evenly distributed)
            #
            # For BI-V100 (16 SMs), fixed _BLOCKS_PER_TILE=32 wastes memory
            # on short contexts and underutilizes on long ones.
            #
            # Key insight from kernel_reduce.cuh:
            #   StableReductionOrder=false uses atomicAdd → single kernel pass.
            #   For online softmax (our case), we accumulate (m, l, o) per tile
            #   then merge — this IS a multi-pass reduce. Larger tiles = fewer
            #   merge steps = less numerical drift + less Python loop overhead.
            #
            # CCCL subscription_factor = CUB_SUBSCRIPTION_FACTOR(0) = 5
            # Effective: 16 SM × 1 CTA/SM × 5 = 80 concurrent tiles max.
            # But Python loop overhead dominates, so we want FEWER, LARGER tiles.
            #
            # Strategy: target ~4-8 tiles per context phase.
            #   Fewer tiles → fewer matmul calls → less launch overhead.
            #   SMEM constraint: score tensor [kv_h, gqa, q_len, tile_sz] fp32
            #     must not cause OOM. With q_len=4096, kv_h=1, gqa=6:
            #     tile_sz=1024 → 1×6×4096×1024×4 = 96 MB (too much)
            #     tile_sz=512  → 48 MB (borderline)
            #     tile_sz=256  → 24 MB (safe)
            #   For decode (q_len=1): tile_sz=4096 → only 96 KB (always safe)
            # ================================================================
            _SMEM_BUDGET_BYTES = 96 * 1024 * 1024  # 96 MB score tensor budget

            batch_size   = seq_lens_tensor.shape[0]
            num_q_heads  = query.shape[1]
            num_kv_heads = key_cache.shape[1]
            head_dim     = query.shape[2]
            gqa_ratio    = num_q_heads // num_kv_heads
            block_size   = value_cache.shape[3]
            scale        = head_dim ** -0.5
            orig_dtype   = query.dtype
            output       = torch.empty_like(query)
            dev          = query.device

            for i in range(batch_size):
                ctx_len = int(context_lens[i].item())
                q_start = int(query_start_loc[i].item())
                q_end   = int(query_start_loc[i + 1].item())
                q_len   = q_end - q_start

                q_i = query[q_start:q_end]   # [q_len, q_h,  d]
                k_i = key  [q_start:q_end]   # [q_len, kv_h, d]
                v_i = value[q_start:q_end]

                # CCCL spread_out_items_per_thread adaptive tile sizing.
                #
                # Two constraints compete:
                # 1. Memory: score tensor [kv_h, gqa, q_len, tile_sz] × 4 ≤ budget
                # 2. Iteration count: want ~4-8 tiles to minimize Python overhead
                #
                # CCCL dispatch_transform.cuh::spread_out_items_per_thread:
                #   items = ceil_div(num_items, sm_count * threads * occupancy)
                #   items = clamp(items, min_items, max_items)
                #
                # Our translation: tile_sz = max context tokens / target_tiles,
                # then clamp by memory budget.
                score_row_bytes = num_kv_heads * gqa_ratio * q_len * 4
                if score_row_bytes > 0:
                    mem_max_tokens = _SMEM_BUDGET_BYTES // score_row_bytes
                    mem_max_tokens = (mem_max_tokens // block_size) * block_size
                else:
                    mem_max_tokens = block_size * 256

                total_kv_tokens = ctx_len + q_len
                # spread_out: target 4 tiles for context, 4 for current chunk
                spread_tile = max(block_size,
                                  (total_kv_tokens + 3) // 4)
                # Round to block_size
                spread_tile = (spread_tile // block_size) * block_size
                spread_tile = max(spread_tile, block_size)
                # Clamp by memory budget
                tile_sz = min(spread_tile, mem_max_tokens)
                tile_sz = max(tile_sz, block_size)  # floor

                # Q reshaped and scaled once; held for all K-tiles.
                # [kv_h, gqa, q_len, d] fp32  —  24 MB for q_len=4096, d=256
                q_seq = (q_i.permute(1, 0, 2)
                           .float()
                           .view(num_kv_heads, gqa_ratio, q_len, head_dim)
                           .mul_(scale))

                # Flash-Attention online-softmax accumulators.
                # m, l : [kv_h, gqa, q_len]     fp32  — <0.1 MB
                # o    : [kv_h, gqa, q_len, d]  fp32  — 24 MB
                m = torch.full((num_kv_heads, gqa_ratio, q_len),
                               float('-inf'), dtype=torch.float32, device=dev)
                l = torch.zeros_like(m)
                o = torch.zeros((num_kv_heads, gqa_ratio, q_len, head_dim),
                                dtype=torch.float32, device=dev)

                # --------------------------------------------------------------
                # Phase 1 — context tokens (positions 0 … ctx_len-1).
                #
                # Every context key has absolute position < ctx_len; every
                # query has position ≥ ctx_len.  k_pos < q_pos is always True
                # → no causal mask needed for pure context tiles.
                # --------------------------------------------------------------
                # Convert token-based tile_sz to block count for iteration
                blocks_per_tile = tile_sz // block_size

                if ctx_len > 0:
                    num_ctx_blocks = (ctx_len + block_size - 1) // block_size
                    if num_ctx_blocks > block_tables.shape[1]:
                        print(
                            f"[paged_attn WARNING] seq {i}: num_ctx_blocks={num_ctx_blocks} "
                            f"> block_tables.shape[1]={block_tables.shape[1]}, ctx_len={ctx_len}. "
                            "Block table is undersized (prefix_cache_hit bug). "
                            "Capping context to available blocks — attention may be incorrect.",
                            file=sys.stderr, flush=True)
                        num_ctx_blocks = block_tables.shape[1]
                    for tile_blk in range(0, num_ctx_blocks, blocks_per_tile):
                        blk_end = min(tile_blk + blocks_per_tile, num_ctx_blocks)
                        blk_ids = block_tables[i, tile_blk:blk_end]

                        # Gather K/V for this tile.
                        # key_cache  [blk_ids]: [n, kv_h, d//x, blk_sz, x]
                        # value_cache[blk_ids]: [n, kv_h, d,    blk_sz]
                        k_tile = (key_cache[blk_ids]
                                  .permute(0, 3, 1, 2, 4)
                                  .contiguous()
                                  .view(-1, num_kv_heads, head_dim))
                        v_tile = (value_cache[blk_ids]
                                  .permute(0, 3, 1, 2)
                                  .contiguous()
                                  .view(-1, num_kv_heads, head_dim))

                        # Trim padding in the last block of the tile.
                        valid = (min(blk_end * block_size, ctx_len)
                                 - tile_blk * block_size)
                        k_tile = k_tile[:valid]   # [valid, kv_h, d]
                        v_tile = v_tile[:valid]

                        # k_t: [kv_h, 1, d, valid]   (broadcast over gqa_ratio)
                        # v_t: [kv_h, 1, valid, d]
                        k_t = (k_tile.permute(1, 0, 2)
                                     .unsqueeze(1)
                                     .transpose(-1, -2)
                                     .float())
                        v_t = (v_tile.permute(1, 0, 2)
                                     .unsqueeze(1)
                                     .float())
                        del k_tile, v_tile

                        # Scores: [kv_h, gqa, q_len, valid]
                        s = torch.matmul(q_seq, k_t)
                        del k_t
                        # No causal mask: all context keys precede all queries.

                        # Online softmax update — Flash-Attention Algorithm 1.
                        # exp_s = s - new_max  (in-place exp after del s)
                        m_blk = s.amax(dim=-1)
                        m_new = torch.maximum(m, m_blk)
                        exp_s = s - m_new.unsqueeze(-1)
                        del s
                        exp_s.exp_()
                        corr  = torch.exp(m - m_new)
                        m.copy_(m_new)
                        del m_blk, m_new
                        l.mul_(corr).add_(exp_s.sum(dim=-1))
                        o.mul_(corr.unsqueeze(-1)).add_(
                            torch.matmul(exp_s, v_t))
                        del exp_s, v_t, corr

                # --------------------------------------------------------------
                # Phase 2 — current-chunk tokens (positions ctx_len … ctx_len+q_len-1).
                #
                # Causal mask: query at relative position j sees key at relative
                # position k only when k ≤ j.  Tiles of tile_sz tokens each.
                # --------------------------------------------------------------
                for kc_start in range(0, q_len, tile_sz):
                    kc_end = min(kc_start + tile_sz, q_len)
                    kc_len = kc_end - kc_start

                    k_blk = k_i[kc_start:kc_end]   # [kc_len, kv_h, d]
                    v_blk = v_i[kc_start:kc_end]

                    k_t = (k_blk.permute(1, 0, 2)
                                 .unsqueeze(1)
                                 .transpose(-1, -2)
                                 .float())   # [kv_h, 1, d, kc_len]
                    v_t = (v_blk.permute(1, 0, 2)
                                 .unsqueeze(1)
                                 .float())   # [kv_h, 1, kc_len, d]

                    s = torch.matmul(q_seq, k_t)   # [kv_h, gqa, q_len, kc_len]
                    del k_t

                    # Causal mask: key at (kc_start+k) must not exceed query j.
                    k_rel = torch.arange(kc_start, kc_end, device=dev)
                    q_rel = torch.arange(q_len, device=dev)
                    mask  = k_rel.unsqueeze(0) > q_rel.unsqueeze(1)  # [q_len, kc_len]
                    s.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                    del mask, k_rel, q_rel

                    # Online softmax update (identical to context phase).
                    m_blk = s.amax(dim=-1)
                    m_new = torch.maximum(m, m_blk)
                    exp_s = s - m_new.unsqueeze(-1)
                    del s
                    exp_s.exp_()
                    corr  = torch.exp(m - m_new)
                    m.copy_(m_new)
                    del m_blk, m_new
                    l.mul_(corr).add_(exp_s.sum(dim=-1))
                    o.mul_(corr.unsqueeze(-1)).add_(
                        torch.matmul(exp_s, v_t))
                    del exp_s, v_t, corr

                # --------------------------------------------------------------
                # Finalize: normalize running output by normalization factor.
                # o: [kv_h, gqa, q_len, d]  →  [q_len, q_h, d]
                # --------------------------------------------------------------
                o.div_(l.unsqueeze(-1))
                output[q_start:q_end] = (
                    o.view(num_q_heads, q_len, head_dim)
                     .permute(1, 0, 2)
                     .to(orig_dtype)
                )

        except Exception as e:
            print(f"[paged_attn ERROR] {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            raise
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
