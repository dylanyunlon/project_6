"""
ex_engine/factors/attn_dispatch.py

Layer 3: Attention prefill/decode dispatch

Upstream parallel: xllm_layers/ilu/attention.h + attention.cpp (82 + ~200 lines)
  → batch_prefill() dispatches to ixinfer_flash_attn_unpad_with_block_tables
  → batch_decode() dispatches to xllm_paged_attention

The key dispatch decision:
  prefill (num_prefill_tokens > 0):
    → flash attention with block tables (unpadded, variable-length)
    → supports cu_seqlens for multi-request batching
    → window attention via window_size_left/right

  decode (pure autoregressive, 1 token per sequence):
    → paged attention v1/v2
    → v1 vs v2 decision: total_tiles vs 2 × sm_count
    → BI-V100: 16 SMs → V2 beneficial when seq_len > 1024

GDN (GatedDeltaNet) layers [1,7,13,19] bypass this entirely —
they use the gdn_dispatch module instead.

Call chain:
  qwen3_5.py Qwen3_5DecoderLayer.forward()
    → (full attention layers): attn_dispatch.dispatch_attention()
      → prefill: flash_attn_with_block_tables
      → decode:  paged_attention_v1 or v2
    → (GDN layers): gdn_dispatch.dispatch_gdn()
"""

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger("ex_engine.attn_dispatch")

# BI-V100 dispatch thresholds
# Source: SYSTEM_DESIGN.md, sub694 TPS analysis
_SM_COUNT = 16
_V2_THRESHOLD_FACTOR = 2  # V2 when total_tiles > 2 × SM_COUNT
_PAGE_BLOCK_SIZE = 16     # default paged attention block size


class AttnDispatchConfig:
    """
    Attention dispatch configuration.

    Parallels xllm_layers/ilu/attention.h struct members:
      scale, is_causal, window_size_left, window_size_right, softcap
    """
    __slots__ = (
        'num_heads', 'num_kv_heads', 'head_dim', 'scale',
        'is_causal', 'window_left', 'window_right',
        'block_size', 'max_context_len', 'softcap',
    )

    def __init__(
        self,
        num_heads: int = 28,
        num_kv_heads: int = 4,
        head_dim: int = 128,
        scale: Optional[float] = None,
        is_causal: bool = True,
        window_left: int = -1,
        window_right: int = -1,
        block_size: int = 16,
        max_context_len: int = 131072,
        softcap: float = 0.0,
    ):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale or (head_dim ** -0.5)
        self.is_causal = is_causal
        self.window_left = window_left
        self.window_right = window_right
        self.block_size = block_size
        self.max_context_len = max_context_len
        self.softcap = softcap


def should_use_paged_v2(
    seq_len: int,
    num_kv_heads: int,
    block_size: int = 16,
    partition_size: int = 512,
) -> bool:
    """
    V1 vs V2 decision for paged attention.

    Upstream: vllm/attention/ops/paged_attn.py PagedAttention._use_v2()
    Rule: total_tiles = num_kv_heads × ceil(seq_len / partition_size)
          use V2 when total_tiles > 2 × SM_COUNT (BI-V100: 32)

    On BI-V100 with 16 SMs and 4 KV heads:
      V2 when seq_len > 512 × (2 × 16 / 4) = 4096
      In practice, V2 is better for seq_len > 1024 due to latency hiding.
    """
    num_tiles = num_kv_heads * ((seq_len + partition_size - 1) // partition_size)
    return num_tiles > _V2_THRESHOLD_FACTOR * _SM_COUNT


def dispatch_prefill(
    config: AttnDispatchConfig,
    query: torch.Tensor,          # (total_q_tokens, num_heads, head_dim)
    key_cache: torch.Tensor,      # (num_blocks, num_kv_heads, block_size, head_dim)
    value_cache: torch.Tensor,    # (num_blocks, num_kv_heads, block_size, head_dim)
    block_tables: torch.Tensor,   # (batch_size, max_blocks_per_seq)
    cu_seq_q: torch.Tensor,       # (batch_size + 1,) int32 — query cumulative lengths
    cu_seq_k: torch.Tensor,       # (batch_size + 1,) int32 — key cumulative lengths
    max_seq_q: int,
    max_seq_k: int,
) -> torch.Tensor:
    """
    Prefill attention via flash attention with block tables.

    Upstream: xllm::kernel::ilu::batch_prefill
      → ixformer::infer::ixinfer_flash_attn_unpad_with_block_tables

    The BI-V100 ixformer implements this as a modified flash attention
    that reads KV from paged cache (block_tables → physical blocks).
    """
    # Try ix_ops_dispatch first
    try:
        from ex_engine.python import ix_ops_dispatch
        output = ix_ops_dispatch.flash_attn_with_block_tables(
            query, key_cache, value_cache, block_tables,
            cu_seq_q, cu_seq_k, max_seq_q, max_seq_k,
            config.scale,
            is_causal=config.is_causal,
            window_left=config.window_left,
            window_right=config.window_right,
            softcap=config.softcap,
        )
        return output
    except (ImportError, RuntimeError, AttributeError) as e:
        logger.debug("flash_attn dispatch failed, using fallback: %s", e)

    # Try direct ixformer
    try:
        import ixformer.functions as ixf_F
        output = torch.empty_like(query)
        ixf_F.ixinfer_flash_attn_unpad_with_block_tables(
            query, key_cache, value_cache, output, block_tables,
            cu_seq_q, cu_seq_k, max_seq_q, max_seq_k,
            config.is_causal, config.window_left, config.window_right,
            config.scale, config.softcap, False, None, None, None)
        return output
    except (ImportError, AttributeError):
        pass

    # PyTorch fallback — SDPA (no block table support, for testing only)
    logger.warning("prefill: using PyTorch SDPA fallback (no block tables)")
    output = torch.nn.functional.scaled_dot_product_attention(
        query.unsqueeze(0), query.unsqueeze(0), query.unsqueeze(0),
        scale=config.scale, is_causal=config.is_causal)
    return output.squeeze(0)


def dispatch_decode(
    config: AttnDispatchConfig,
    output: torch.Tensor,         # (batch_size, num_heads, head_dim) preallocated
    query: torch.Tensor,          # (batch_size, num_heads, head_dim)
    key_cache: torch.Tensor,      # (num_blocks, num_kv_heads, block_size, head_dim)
    value_cache: torch.Tensor,    # (num_blocks, num_kv_heads, block_size, head_dim)
    block_tables: torch.Tensor,   # (batch_size, max_blocks_per_seq)
    context_lens: torch.Tensor,   # (batch_size,) int32
    max_context_len: int,
) -> None:
    """
    Decode attention via paged attention v1/v2.

    Upstream: xllm::kernel::ilu::batch_decode
      → ixformer::infer::xllm_paged_attention
    """
    # Try ix_ops_dispatch
    try:
        from ex_engine.python import ix_ops_dispatch
        ix_ops_dispatch.paged_attention_v1(
            output, query, key_cache, value_cache,
            config.num_kv_heads, config.scale,
            block_tables, context_lens,
            config.block_size, max_context_len,
            window_left=config.window_left,
            window_right=config.window_right,
            softcap=config.softcap,
        )
        return
    except (ImportError, RuntimeError, AttributeError) as e:
        logger.debug("paged_attention dispatch failed: %s", e)

    # Try direct ixformer
    try:
        import ixformer.functions as ixf_F
        ixf_F.vllm_single_query_cached_kv_attention(
            output, query, key_cache, value_cache,
            config.num_kv_heads, config.scale,
            block_tables, context_lens,
            config.block_size, max_context_len, None)
        return
    except (ImportError, AttributeError):
        pass

    # PyTorch fallback — extremely slow, decode-only test path
    logger.warning("decode: using PyTorch fallback (very slow)")
    batch_size = query.shape[0]
    for b in range(batch_size):
        ctx_len = context_lens[b].item()
        q = query[b]  # (num_heads, head_dim)
        # Reconstruct KV from cache
        blocks = block_tables[b]
        num_blocks_used = (ctx_len + config.block_size - 1) // config.block_size
        k_list, v_list = [], []
        for bi in range(num_blocks_used):
            block_idx = blocks[bi].item()
            tokens_in_block = min(config.block_size,
                                  ctx_len - bi * config.block_size)
            k_list.append(key_cache[block_idx, :, :tokens_in_block])
            v_list.append(value_cache[block_idx, :, :tokens_in_block])
        k = torch.cat(k_list, dim=1)  # (kv_heads, ctx_len, head_dim)
        v = torch.cat(v_list, dim=1)

        # GQA: expand kv heads
        num_q_per_kv = config.num_heads // config.num_kv_heads
        k = k.repeat_interleave(num_q_per_kv, dim=0)
        v = v.repeat_interleave(num_q_per_kv, dim=0)

        # Standard attention
        scores = torch.einsum('hd,hsd->hs', q.float(), k.float())
        scores = scores * config.scale
        scores = torch.softmax(scores, dim=-1)
        out = torch.einsum('hs,hsd->hd', scores, v.float())
        output[b] = out.to(output.dtype)


def dispatch_attention(
    config: AttnDispatchConfig,
    is_prefill: bool,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    # Prefill-specific
    cu_seq_q: Optional[torch.Tensor] = None,
    cu_seq_k: Optional[torch.Tensor] = None,
    max_seq_q: int = 0,
    max_seq_k: int = 0,
    # Decode-specific
    context_lens: Optional[torch.Tensor] = None,
    max_context_len: int = 0,
) -> torch.Tensor:
    """
    Top-level attention dispatcher.

    Mirrors xllm's split between batch_prefill and batch_decode,
    routing to the correct kernel based on is_prefill flag.
    """
    if is_prefill:
        return dispatch_prefill(
            config, query, key_cache, value_cache,
            block_tables, cu_seq_q, cu_seq_k,
            max_seq_q, max_seq_k)
    else:
        output = torch.empty_like(query)
        dispatch_decode(
            config, output, query, key_cache, value_cache,
            block_tables, context_lens, max_context_len)
        return output
