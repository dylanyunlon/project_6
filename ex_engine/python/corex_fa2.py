"""
corex_fa2.py — Flash Attention 2 dispatch for BI-V100 via ixformer

Sub168 log reference:
  corex_fa2.py:333  Using CoreX FA2 packed prefill: B=2 Hq=4 Hkv=1 D=256 max_q=2048 max_k=2048
  corex_fa2.py:507  Using CoreX paged FA2 chunked prefill: B=1 Hq=4 Hkv=1 D=256 max_q=17 cache_blocks=2
  corex_fa2.py:225  Using CoreX paged decode: B=1 Hq=4 Hkv=1 D=256 max_k=45455 partition=256

Call chain:
  qwen3_5.py → Attention.forward() → corex_fa2.forward()
    → ixformer.functions.ixinfer_flash_attn_unpad()       (packed prefill)
    → ixformer.functions.vllm_single_query_cached_kv_attention_v2()  (paged decode)
    → ixformer.functions.ixdnn_flash_attn_unpad()          (paged chunked prefill)

Source: upstream_ref/xllm/xllm/core/kernels/ilu/attention.cpp
        upstream_ref/xllm/xllm/core/layers/ilu/attention.cpp
"""

import logging
import math
import torch
from typing import Optional

logger = logging.getLogger(__name__)

# ============================================================================
# Load ixformer.functions — these ARE in the base image Python binding
# ============================================================================
_ixf_F = None
try:
    import ixformer.functions as _ixf_F
except ImportError:
    logger.warning("ixformer.functions not available — FA2 will use xformers fallback")


class CoreXFA2:
    """
    Flash Attention 2 operator for BI-V100.

    Three modes matching Sub168 log:
    1. Packed prefill (non-paged, full sequence)
    2. Paged chunked prefill (paged KV cache, chunked prefill)
    3. Paged decode (single token decode with KV cache)
    """

    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: Optional[float] = None,
        block_size: int = 16,
    ):
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale or (1.0 / math.sqrt(head_dim))
        self.block_size = block_size
        self._prefill_logged = False
        self._chunked_logged = False
        self._decode_logged = False

    def forward_packed_prefill(
        self,
        query: torch.Tensor,        # (total_q, num_q_heads, head_dim)
        key: torch.Tensor,           # (total_k, num_kv_heads, head_dim)
        value: torch.Tensor,         # (total_k, num_kv_heads, head_dim)
        cu_seqlens_q: torch.Tensor,  # (batch+1,)
        cu_seqlens_k: torch.Tensor,  # (batch+1,)
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        """Packed variable-length prefill using ixinfer flash attn."""
        if _ixf_F is None:
            raise RuntimeError("ixformer not available for FA2 prefill")

        batch_size = cu_seqlens_q.size(0) - 1
        if not self._prefill_logged:
            logger.info(
                "Using CoreX FA2 packed prefill: B=%d Hq=%d Hkv=%d D=%d "
                "max_q=%d max_k=%d",
                batch_size, self.num_q_heads, self.num_kv_heads,
                self.head_dim, max_seqlen_q, max_seqlen_k)
            self._prefill_logged = True

        out = torch.empty_like(query)
        _ixf_F.ixinfer_flash_attn_unpad(
            query, key, value, out,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k,
            self.scale, True,  # is_causal
        )
        return out

    def forward_paged_decode(
        self,
        query: torch.Tensor,         # (batch, 1, num_q_heads, head_dim)
        key_cache: torch.Tensor,     # (num_blocks, block_size, num_kv_heads, head_dim)
        value_cache: torch.Tensor,   # (num_blocks, block_size, num_kv_heads, head_dim)
        block_tables: torch.Tensor,  # (batch, max_blocks_per_seq)
        context_lens: torch.Tensor,  # (batch,)
    ) -> torch.Tensor:
        """Single-token paged decode using vllm paged attention v2."""
        if _ixf_F is None:
            raise RuntimeError("ixformer not available for paged decode")

        batch_size = query.size(0)
        max_context_len = int(context_lens.max().item())

        if not self._decode_logged:
            partition_size = 256
            logger.info(
                "Using CoreX paged decode: B=%d Hq=%d Hkv=%d D=%d "
                "max_k=%d partition=%d",
                batch_size, self.num_q_heads, self.num_kv_heads,
                self.head_dim, max_context_len, partition_size)
            self._decode_logged = True

        out = query.new_empty(batch_size, self.num_q_heads, self.head_dim)
        q_flat = query.squeeze(1)  # (batch, num_q_heads, head_dim)

        _ixf_F.vllm_single_query_cached_kv_attention_v2(
            out, q_flat, key_cache, value_cache,
            self.scale, block_tables, context_lens,
            self.block_size, max_context_len,
        )
        return out.unsqueeze(1)

    def forward_paged_chunked_prefill(
        self,
        query: torch.Tensor,         # (total_q, num_q_heads, head_dim)
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
    ) -> torch.Tensor:
        """Paged chunked prefill using ixdnn flash attn with block tables."""
        if _ixf_F is None:
            raise RuntimeError("ixformer not available for chunked prefill")

        batch_size = cu_seqlens_q.size(0) - 1
        num_cache_blocks = block_tables.size(1) if block_tables.dim() > 1 else 0

        if not self._chunked_logged:
            logger.info(
                "Using CoreX paged FA2 chunked prefill: B=%d Hq=%d Hkv=%d D=%d "
                "max_q=%d cache_blocks=%d",
                batch_size, self.num_q_heads, self.num_kv_heads,
                self.head_dim, max_seqlen_q, num_cache_blocks)
            self._chunked_logged = True

        out = torch.empty_like(query)

        # Use ixdnn flash attn with block tables for paged chunked prefill
        if hasattr(_ixf_F, 'ixdnn_flash_attn_unpad'):
            _ixf_F.ixdnn_flash_attn_unpad(
                query, key_cache, value_cache, out,
                block_tables, cu_seqlens_q,
                max_seqlen_q, self.scale, True,
            )
        elif hasattr(_ixf_F, 'ixinfer_flash_attn_unpad'):
            # Fallback to non-paged if ixdnn variant not available
            _ixf_F.ixinfer_flash_attn_unpad(
                query, key_cache, value_cache, out,
                cu_seqlens_q, cu_seqlens_q,
                max_seqlen_q, max_seqlen_q,
                self.scale, True,
            )
        else:
            raise RuntimeError("No flash attn variant available for chunked prefill")

        return out
