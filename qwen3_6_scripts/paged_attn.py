from dataclasses import dataclass
from typing import List, Optional, Tuple
import sys
import torch
import traceback
from vllm import _custom_ops as ops

# Should be the same as PARTITION_SIZE in `paged_attention_v2_launcher`.
_PARTITION_SIZE = 512


@dataclass
class PagedAttentionMetadata:
    """Metadata for PagedAttention."""
    seq_lens_tensor: Optional[torch.Tensor]
    max_decode_seq_len: int
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
        key_cache = key_cache.view(num_blocks, num_kv_heads, head_size // x, -1, x)
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
            slot_mapping,
            kv_cache_dtype,
            k_scale,
            v_scale,
        )

    @staticmethod
    def _forward_decode_pytorch(
        query, key_cache, value_cache, block_tables, seq_lens, scale
    ):
        """Pure-PyTorch decode fallback for seq_len > threshold.

        Used when ixf_F.paged_attention_v1 cannot handle the sequence length.
        Optimized with batched KV gather (no per-block Python loop).
        """
        num_seqs, num_heads, head_dim = query.shape
        num_kv_heads = key_cache.shape[1]
        block_size = value_cache.shape[3]
        gqa_ratio = num_heads // num_kv_heads
        orig_dtype = query.dtype
        output = torch.empty_like(query)

        try:
            for i in range(num_seqs):
                seq_len = int(seq_lens[i].item())
                num_blocks = (seq_len + block_size - 1) // block_size
                blk_ids = block_tables[i, :num_blocks]

                # Batched gather: one index_select for all blocks
                k_t = (key_cache[blk_ids]
                       .permute(0, 3, 1, 2, 4)
                       .contiguous()
                       .view(-1, num_kv_heads, head_dim))[:seq_len] \
                      .permute(1, 2, 0).contiguous().float()

                v_t = (value_cache[blk_ids]
                       .permute(0, 3, 1, 2)
                       .contiguous()
                       .view(-1, num_kv_heads, head_dim))[:seq_len] \
                      .permute(1, 0, 2).contiguous().float()

                q_grouped = (query[i].float()
                             .view(num_kv_heads, gqa_ratio, head_dim)
                             .unsqueeze(2))

                attn_w = torch.matmul(q_grouped * scale, k_t.unsqueeze(1))
                attn_w = torch.softmax(attn_w, dim=-1)
                out_i = torch.matmul(attn_w, v_t.unsqueeze(1))
                output[i] = out_i.view(num_heads, head_dim).to(orig_dtype)

        except Exception as e:
            print(f"[decode_pytorch ERROR] {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            raise

        return output

    # BI-V100: Try higher threshold for compiled v1 kernel.
    # Compiled kernel is ~100x faster than Python fallback.
    _PYTORCH_DECODE_THRESHOLD = 65536

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

        output = torch.empty_like(query)
        block_size = value_cache.shape[3]
        num_seqs, num_heads, head_size = query.shape
        max_num_partitions = ((max_seq_len + _PARTITION_SIZE - 1) //
                              _PARTITION_SIZE)

        use_v1 = (max_seq_len <= 8192
                  and (max_num_partitions == 1 or num_seqs * num_heads > 512))
        # V2 now works (paged_attention_v2_pytorch), so use the original heuristic
        # instead of hardcoding use_v1=True
        if use_v1:
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

    # Triton prefill: try once, fall back permanently if it fails
    _triton_prefill_ok = None

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
        # Try Triton kernel if available and not known to fail
        if PagedAttention._triton_prefill_ok is not False:
            try:
                from vllm.triton_utils import HAS_TRITON
                if HAS_TRITON:
                    from vllm.attention.ops.prefix_prefill import context_attention_fwd
                    output = torch.empty_like(query)
                    context_attention_fwd(
                        query, key, value, output, kv_cache_dtype,
                        key_cache, value_cache, block_tables,
                        query_start_loc[:-1], seq_lens_tensor, context_lens,
                        max_query_len, k_scale, v_scale,
                        alibi_slopes, sliding_window,
                    )
                    if PagedAttention._triton_prefill_ok is None:
                        print("[paged_attn] Triton prefill kernel: SUCCESS", flush=True)
                        PagedAttention._triton_prefill_ok = True
                    return output
            except Exception as e:
                print(f"[paged_attn] Triton prefill failed: {type(e).__name__}: {e}",
                      flush=True)
                print("[paged_attn] Falling back to PyTorch prefill permanently", flush=True)
                PagedAttention._triton_prefill_ok = False

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
        """Pure-PyTorch prefix-attention with pre-gathered KV and Flash-Attention online softmax.

        Optimization over baseline:
          - Context KV is gathered ONCE outside the tile loop (one index_select + reshape)
          - Tile loop just slices views from the pre-gathered tensor (no per-tile gather)
          - Eliminates 194 redundant permute+contiguous calls for 100K context

        Memory: O(q_len × tile_sz) per tile — same as baseline.
        The full context K/V tensor is ~50MB for 100K tokens, fits in GPU memory.
        """
        try:
            _BLOCKS_PER_TILE = 32

            batch_size = seq_lens_tensor.shape[0]
            num_q_heads = query.shape[1]
            num_kv_heads = key_cache.shape[1]
            head_dim = query.shape[2]
            gqa_ratio = num_q_heads // num_kv_heads
            block_size = value_cache.shape[3]
            tile_sz = _BLOCKS_PER_TILE * block_size
            scale = head_dim ** -0.5
            orig_dtype = query.dtype
            output = torch.empty_like(query)
            dev = query.device

            for i in range(batch_size):
                ctx_len = int(context_lens[i].item())
                q_start = int(query_start_loc[i].item())
                q_end = int(query_start_loc[i + 1].item())
                q_len = q_end - q_start

                q_i = query[q_start:q_end]
                k_i = key[q_start:q_end]
                v_i = value[q_start:q_end]

                q_seq = (q_i.permute(1, 0, 2)
                           .float()
                           .view(num_kv_heads, gqa_ratio, q_len, head_dim)
                           .mul_(scale))

                m = torch.full((num_kv_heads, gqa_ratio, q_len),
                               float('-inf'), dtype=torch.float32, device=dev)
                l = torch.zeros_like(m)
                o = torch.zeros((num_kv_heads, gqa_ratio, q_len, head_dim),
                                dtype=torch.float32, device=dev)

                # ===========================================================
                # Phase 1: Context tokens — PRE-GATHER optimization
                # Gather ALL context K/V in ONE shot, then tile via slicing
                # ===========================================================
                if ctx_len > 0:
                    num_ctx_blocks = (ctx_len + block_size - 1) // block_size
                    if num_ctx_blocks > block_tables.shape[1]:
                        print(
                            f"[paged_attn WARNING] seq {i}: num_ctx_blocks={num_ctx_blocks} "
                            f"> block_tables.shape[1]={block_tables.shape[1]}. "
                            "Capping context to available blocks.",
                            file=sys.stderr, flush=True)
                        num_ctx_blocks = block_tables.shape[1]

                    # ONE gather for ALL context blocks
                    ctx_blk_ids = block_tables[i, :num_ctx_blocks]

                    # [num_ctx_blocks, kv_h, d/x, blk_sz, x] → [ctx_tokens, kv_h, d]
                    ctx_k_all = (key_cache[ctx_blk_ids]
                                 .permute(0, 3, 1, 2, 4)
                                 .contiguous()
                                 .view(-1, num_kv_heads, head_dim))[:ctx_len]

                    ctx_v_all = (value_cache[ctx_blk_ids]
                                 .permute(0, 3, 1, 2)
                                 .contiguous()
                                 .view(-1, num_kv_heads, head_dim))[:ctx_len]

                    # Pre-transpose for matmul: [kv_h, d, ctx_len] and [kv_h, ctx_len, d]
                    ctx_k_t = ctx_k_all.permute(1, 2, 0).contiguous().float()
                    ctx_v_t = ctx_v_all.permute(1, 0, 2).contiguous().float()

                    # Tile loop: just SLICE from pre-gathered tensors
                    for tile_start in range(0, ctx_len, tile_sz):
                        tile_end = min(tile_start + tile_sz, ctx_len)

                        # Slice (view, no copy)
                        k_t = ctx_k_t[:, :, tile_start:tile_end].unsqueeze(1)
                        v_t = ctx_v_t[:, tile_start:tile_end, :].unsqueeze(1)

                        s = torch.matmul(q_seq, k_t)
                        del k_t

                        m_blk = s.amax(dim=-1)
                        m_new = torch.maximum(m, m_blk)
                        exp_s = s - m_new.unsqueeze(-1)
                        del s
                        exp_s.exp_()
                        corr = torch.exp(m - m_new)
                        m.copy_(m_new)
                        del m_blk, m_new
                        l.mul_(corr).add_(exp_s.sum(dim=-1))
                        o.mul_(corr.unsqueeze(-1)).add_(
                            torch.matmul(exp_s, v_t))
                        del exp_s, v_t, corr

                    del ctx_k_t, ctx_v_t, ctx_k_all, ctx_v_all

                # ===========================================================
                # Phase 2: Current-chunk tokens (with causal mask)
                # ===========================================================
                for kc_start in range(0, q_len, tile_sz):
                    kc_end = min(kc_start + tile_sz, q_len)

                    k_blk = k_i[kc_start:kc_end]
                    v_blk = v_i[kc_start:kc_end]

                    k_t = (k_blk.permute(1, 0, 2)
                                 .unsqueeze(1)
                                 .transpose(-1, -2)
                                 .float())
                    v_t = (v_blk.permute(1, 0, 2)
                                 .unsqueeze(1)
                                 .float())

                    s = torch.matmul(q_seq, k_t)
                    del k_t

                    k_rel = torch.arange(kc_start, kc_end, device=dev)
                    q_rel = torch.arange(q_len, device=dev)
                    mask = k_rel.unsqueeze(0) > q_rel.unsqueeze(1)
                    s.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                    del mask, k_rel, q_rel

                    m_blk = s.amax(dim=-1)
                    m_new = torch.maximum(m, m_blk)
                    exp_s = s - m_new.unsqueeze(-1)
                    del s
                    exp_s.exp_()
                    corr = torch.exp(m - m_new)
                    m.copy_(m_new)
                    del m_blk, m_new
                    l.mul_(corr).add_(exp_s.sum(dim=-1))
                    o.mul_(corr.unsqueeze(-1)).add_(
                        torch.matmul(exp_s, v_t))
                    del exp_s, v_t, corr

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
