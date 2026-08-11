"""ix_unified.py — Unified Python interface to all ixformer::infer APIs.

Dispatch hierarchy (CCCL policy_selector pattern):
  Tier 0: ix_unified_bridge.so  (C++ direct call to ixformer::infer)
  Tier 1: ixformer.functions.*  (base image Python bindings, partial)
  Tier 2: PyTorch fallback      (always works, slowest)

Usage:
    from ex_engine.python.ix_unified import ix
    out = ix.silu_and_mul(input)
    ix.rms_norm(output, input, weight, eps)
    weights, indices = ix.moe_topk_softmax(gating, topk, renorm)
"""

import os
import sys
import importlib
import importlib.util
import torch
import logging

logger = logging.getLogger("ix_unified")

_bridge = None


def _load_bridge():
    """Load ix_unified_bridge.so from known locations."""
    global _bridge
    if _bridge is not None:
        return _bridge

    search_paths = []

    # 1. Same directory as this file
    here = os.path.dirname(os.path.abspath(__file__))
    search_paths.append(os.path.join(here, "..", "build"))
    search_paths.append(here)

    # 2. vllm install root (where prebuilt .so are deployed)
    for p in sys.path:
        if "vllm" in p or "dist-packages" in p:
            search_paths.append(p)

    # 3. Explicit env var
    env_path = os.getenv("IX_BRIDGE_PATH")
    if env_path:
        search_paths.insert(0, env_path)

    for search_dir in search_paths:
        for name in ["ix_unified_bridge.so",
                     "ix_unified_bridge.cpython-310-x86_64-linux-gnu.so"]:
            so_path = os.path.join(search_dir, name)
            if os.path.isfile(so_path):
                try:
                    spec = importlib.util.spec_from_file_location(
                        "ix_unified_bridge", so_path)
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    _bridge = mod
                    logger.info("ix_unified_bridge loaded from %s", so_path)
                    return _bridge
                except Exception as e:
                    logger.warning("Failed to load %s: %s", so_path, e)

    logger.info("ix_unified_bridge.so not found, using fallback dispatch")
    return None


def _try_ixformer_functions():
    """Try importing ixformer.functions from base image."""
    try:
        import ixformer.functions as ixf
        return ixf
    except (ImportError, AttributeError):
        return None


# ============================================================================
# Dispatch class
# ============================================================================

class IXDispatch:
    """Three-tier dispatch for all ixformer ops."""

    def __init__(self):
        self._bridge = _load_bridge()
        self._ixf = _try_ixformer_functions()
        tier = ("Tier0:bridge" if self._bridge else
                "Tier1:ixformer" if self._ixf else "Tier2:pytorch")
        logger.info("IXDispatch initialized: %s", tier)

    # --- Activation -----------------------------------------------------------
    def silu_and_mul(self, input: torch.Tensor) -> torch.Tensor:
        if self._bridge:
            return self._bridge.silu_and_mul(input)
        if self._ixf and hasattr(self._ixf, 'silu_and_mul'):
            d = input.size(-1) // 2
            out = input.new_empty([input.size(0), d])
            self._ixf.silu_and_mul(input, out)
            return out
        # PyTorch fallback
        d = input.size(-1) // 2
        x, gate = input[..., :d], input[..., d:]
        return x * torch.sigmoid(gate)

    # --- Norm -----------------------------------------------------------------
    def rms_norm(self, output: torch.Tensor, input: torch.Tensor,
                 weight: torch.Tensor, eps: float):
        if self._bridge:
            self._bridge.rms_norm(output, input, weight, eps)
            return
        if self._ixf and hasattr(self._ixf, 'rms_norm'):
            self._ixf.rms_norm(input, weight, output, eps)
            return
        # PyTorch fallback
        variance = input.float().pow(2).mean(-1, keepdim=True)
        normed = input * torch.rsqrt(variance + eps)
        output.copy_(normed * weight)

    def fused_add_rms_norm(self, input: torch.Tensor,
                           residual: torch.Tensor,
                           weight: torch.Tensor, eps: float):
        if self._bridge:
            self._bridge.fused_add_rms_norm(input, residual, weight, eps)
            return
        if self._ixf and hasattr(self._ixf, 'fused_add_rms_norm'):
            self._ixf.fused_add_rms_norm(input, residual, weight, eps, 1.0)
            return
        # PyTorch fallback
        hidden = input + residual
        residual.copy_(hidden)
        variance = hidden.float().pow(2).mean(-1, keepdim=True)
        normed = hidden * torch.rsqrt(variance + eps)
        input.copy_(normed * weight)

    # --- Linear ---------------------------------------------------------------
    def linear(self, input: torch.Tensor, weight: torch.Tensor,
               bias=None) -> torch.Tensor:
        if self._bridge:
            return self._bridge.linear(input, weight, bias)
        # PyTorch fallback
        out = torch.nn.functional.linear(input, weight, bias)
        return out

    # --- RoPE -----------------------------------------------------------------
    def rotary_embedding(self, positions, query, key, head_size,
                         cos_sin_cache, is_neox=True):
        if self._bridge:
            self._bridge.rotary_embedding(positions, query, key, head_size,
                                          cos_sin_cache, is_neox)
            return
        if self._ixf and hasattr(self._ixf, 'vllm_rotary_embedding_neox'):
            self._ixf.vllm_rotary_embedding_neox(
                positions, query, key, head_size, cos_sin_cache, is_neox)
            return
        # No PyTorch fallback — this is handled by vllm's own rope

    # --- KV Cache -------------------------------------------------------------
    def reshape_and_cache(self, key, value, key_cache, value_cache,
                          slot_mapping):
        if self._bridge:
            self._bridge.reshape_and_cache(key, value, key_cache, value_cache,
                                           slot_mapping)
            return
        if self._ixf and hasattr(self._ixf, 'vllm_cache_ops_reshape_and_cache'):
            self._ixf.vllm_cache_ops_reshape_and_cache(
                key, value, key_cache, value_cache, slot_mapping)
            return
        # PyTorch fallback — slot-by-slot copy
        for i, slot in enumerate(slot_mapping):
            if slot < 0:
                continue
            block_idx = slot // key_cache.size(2)
            block_off = slot % key_cache.size(2)
            key_cache[block_idx, :, block_off, :] = key[i]
            value_cache[block_idx, :, block_off, :] = value[i]

    # --- Attention: prefill ---------------------------------------------------
    def flash_attn_prefill(self, query, key_cache, value_cache, output,
                           block_tables, cu_seq_q, cu_seq_k,
                           max_seq_q, max_seq_k, is_causal, scale):
        if self._bridge:
            return self._bridge.flash_attn_prefill(
                query, key_cache, value_cache, output, block_tables,
                cu_seq_q, cu_seq_k, max_seq_q, max_seq_k, is_causal, scale)
        if self._ixf and hasattr(self._ixf, 'ixinfer_flash_attn_unpad'):
            return self._ixf.ixinfer_flash_attn_unpad(
                query, key_cache, value_cache, output, block_tables,
                cu_seq_q, cu_seq_k, max_seq_q, max_seq_k,
                is_causal, -1, -1, scale, 0.0, False, None, None, None)
        raise RuntimeError("flash_attn_prefill: no backend available")

    # --- Attention: decode (paged) -------------------------------------------
    def paged_attention(self, output, query, key_cache, value_cache,
                        num_kv_heads, scale, block_tables, context_lens,
                        block_size, max_context_len):
        if self._bridge:
            return self._bridge.paged_attention(
                output, query, key_cache, value_cache,
                num_kv_heads, scale, block_tables, context_lens,
                block_size, max_context_len)
        if self._ixf and hasattr(self._ixf,
                                 'vllm_single_query_cached_kv_attention_v2'):
            return self._ixf.vllm_single_query_cached_kv_attention_v2(
                output, query, key_cache, value_cache,
                num_kv_heads, scale, block_tables, context_lens,
                block_size, max_context_len, None)
        raise RuntimeError("paged_attention: no backend available")

    # --- MoE: topk_softmax ---------------------------------------------------
    def moe_topk_softmax(self, gating_output: torch.Tensor,
                         topk: int, renormalize: bool = True):
        if self._bridge:
            return self._bridge.moe_topk_softmax(
                gating_output, topk, renormalize)
        # PyTorch fallback
        scores = torch.softmax(gating_output.float(), dim=-1)
        topk_weights, topk_indices = torch.topk(scores, k=topk, dim=-1)
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1,
                                                            keepdim=True)
        return topk_weights, topk_indices.to(torch.int32)

    # --- MoE: gen_idx ---------------------------------------------------------
    def moe_gen_idx(self, expert_ids: torch.Tensor, num_experts: int):
        if self._bridge:
            return self._bridge.moe_gen_idx(expert_ids, num_experts)
        # PyTorch fallback: compute scatter/gather indices
        flat = expert_ids.view(-1)
        n = flat.numel()
        src_dst = torch.empty(n, dtype=flat.dtype, device=flat.device)
        dst_src = torch.empty(n, dtype=flat.dtype, device=flat.device)
        expert_sizes = torch.zeros(num_experts, dtype=flat.dtype,
                                   device=flat.device)
        # Simple counting sort
        for i in range(n):
            expert_sizes[flat[i].item()] += 1
        cumsum = expert_sizes.cumsum(-1)
        offsets = torch.zeros_like(expert_sizes)
        offsets[1:] = cumsum[:-1]
        counts = torch.zeros_like(expert_sizes)
        for i in range(n):
            e = flat[i].item()
            pos = (offsets[e] + counts[e]).item()
            src_dst[i] = pos
            dst_src[pos] = i
            counts[e] += 1
        return [src_dst, dst_src, expert_sizes, cumsum]

    # --- MoE: expand_input ----------------------------------------------------
    def moe_expand_input(self, input: torch.Tensor,
                         gather_index: torch.Tensor,
                         combine_idx: torch.Tensor, topk: int):
        if self._bridge:
            return self._bridge.moe_expand_input(
                input, gather_index, combine_idx, topk)
        # PyTorch fallback
        return input.index_select(0, combine_idx.view(-1).long())

    # --- MoE: group_gemm -----------------------------------------------------
    def moe_group_gemm(self, input: torch.Tensor, weight: torch.Tensor,
                       tokens_per_experts: torch.Tensor):
        if self._bridge:
            return self._bridge.moe_group_gemm(
                input, weight, tokens_per_experts)
        # PyTorch fallback: sequential per-expert GEMM
        outputs = []
        offset = 0
        for e in range(tokens_per_experts.size(0)):
            count = tokens_per_experts[e].item()
            if count == 0:
                continue
            inp_e = input[offset:offset + count]
            w_e = weight[e]  # [out_features, in_features]
            outputs.append(inp_e @ w_e.t())
            offset += count
        if outputs:
            return torch.cat(outputs, dim=0)
        return input.new_empty(0, weight.size(-2))

    # --- MoE: combine_result -------------------------------------------------
    def moe_combine_result(self, expert_output: torch.Tensor,
                           weights: torch.Tensor):
        if self._bridge:
            return self._bridge.moe_combine_result(expert_output, weights)
        # PyTorch fallback: weighted sum
        # expert_output: [n_tokens, topk, hidden]
        # weights: [n_tokens, topk]
        return (expert_output * weights.unsqueeze(-1)).sum(dim=1)


# Singleton
ix = IXDispatch()
