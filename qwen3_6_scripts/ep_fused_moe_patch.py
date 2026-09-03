"""EP-aware FusedMoE override for vllm 0.6.3.

PR #2269 Sub-task 2: Expert Parallelism weight sharding.

Problem:
  TP=2 on 4× BI-V100 (32 GiB each) OOMs during model loading because
  FusedMoE.create_weights allocates ALL experts on every card, splitting
  only the intermediate_size dimension. For Qwen3.6-35B-A3B (256 experts),
  MoE weights alone consume ~22.5 GiB/card under TP=2.

Solution:
  When enable_expert_parallel=True, each card holds only
  num_experts/ep_size experts with FULL intermediate_size (no TP split
  on MoE). Forward uses Option B (All-Gather + Local Compute +
  Reduce-Scatter) following Google's tpu-inference implementation
  (PR #2577 / Qwen3.5-397B blog post):

  1. All-Gather: replicate all tokens on every rank
  2. Local compute: each rank sorts tokens by expert, computes only
     its local expert range, zeros the rest
  3. Reduce-Scatter: sum partial results across ranks, each rank
     gets back its local token slice

  This replaces the previous Option A (3× All-to-All) approach which
  had unpredictable communication patterns and was harder to debug.

  EP=4: each card holds 64/256 experts → MoE weights ~5.6 GiB/card
  (savings: 16.9 GiB/card vs TP=2)

Memory layout comparison (per card, 30 MoE layers, fp16):
  TP=2:  256 experts × (2×256×2048 + 2048×256) × 2B × 30 = 22.5 GiB
  EP=4:   64 experts × (2×512×2048 + 2048×512) × 2B × 30 =  5.6 GiB

Reference implementation:
  Google tpu-inference: tpu_inference/layers/common/fused_moe_gmm.py
    - fused_moe_func() → expert_parallel_gmm() → moe_gmm_local()
    - PR #2577 (Attention DP + EP sharding)
    - PR #2836 (3-to-2 All-Gather packing optimization)
    - PR #2679 (Hierarchical Reduce-Scatter)

xllm equivalent: core/layers/cuda/fused_moe.cpp
  ep_size_ = parallel_args.ep_size();
  num_experts_per_rank_ = num_experts / ep_size;
  start_expert_id_ = ep_rank * num_experts_per_rank_;
  output = cutlass_fused_moe(..., ep_size_, ep_rank_, ...);
  output = parallel_state::reduce(output, ep_pg_);
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
from typing import Callable, List, Optional, Tuple

from vllm.config import ParallelConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# BI-V100 kernel availability (same probing logic as qwen3_5.py)
# ---------------------------------------------------------------------------
def _probe_bi100_kernels():
    """Probe available BI-V100 MoE kernels at import time.

    Returns a dict of (module_or_None, enabled_bool) for each kernel tier.
    This mirrors the probing in qwen3_5.py but is self-contained so
    ep_fused_moe_patch.py doesn't depend on qwen3_5 module-level globals.
    """
    import os

    def env_bool(name, default=True):
        val = os.environ.get(name, "")
        if val == "":
            return default
        return val.strip().lower() in ("1", "true", "yes")

    # --- grouped GEMM (CUTLASS Cu10) ---
    try:
        from vllm import gemm_grouped
    except ImportError:
        gemm_grouped = None
    use_gemm_grouped = (gemm_grouped is not None
                        and env_bool("BI100_MOE_GEMM_GROUPED", True))

    # --- corex_moe_index_combine (fused histogram+prefix_sum+place) ---
    try:
        from vllm import corex_moe_index_combine
    except ImportError:
        corex_moe_index_combine = None
    use_corex_index = (corex_moe_index_combine is not None
                       and env_bool("BI100_MOE_COREX_INDEX_COMBINE", True))

    # --- xllm_moe (fused topk + compute_index) ---
    try:
        from vllm import xllm_moe
    except ImportError:
        try:
            import xllm_moe
        except ImportError:
            xllm_moe = None
    use_xllm_moe = (xllm_moe is not None
                     and env_bool("BI100_MOE_XLLM", True))

    return {
        "gemm_grouped": gemm_grouped,
        "use_gemm_grouped": use_gemm_grouped,
        "corex_moe_index_combine": corex_moe_index_combine,
        "use_corex_index": use_corex_index,
        "xllm_moe": xllm_moe,
        "use_xllm_moe": use_xllm_moe,
    }


def _is_ep_requested():
    """Check if EP is requested via environment variable.

    This is safe to call at any time (no dist dependency).
    """
    import os
    return bool(int(os.environ.get("VLLM_ENABLE_EXPERT_PARALLEL", "0")))


def _get_ep_runtime_config():
    """Get EP size and rank at runtime when dist is initialized.

    Must only be called from within worker processes (during __init__,
    weight_loader, or forward) where torch.distributed is guaranteed
    to be initialized.

    Returns (ep_size, ep_rank).
    """
    if dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    # Fallback — should not happen during normal model loading
    logger.warning("[PR #2269] dist not initialized when querying EP config, "
                   "falling back to ep_size=1")
    return 1, 0


def _bi100_local_expert_compute(
    recv_hidden: torch.Tensor,     # (N, H) — received tokens
    local_ids: torch.Tensor,       # (N,)   — local expert ids (0..num_local-1)
    recv_weights: torch.Tensor,    # (N,)   — topk weights per token
    w13: torch.Tensor,             # (num_local_experts, 2*I, H)
    w2: torch.Tensor,              # (num_local_experts, H, I)
    num_local_experts: int,
    kernels: dict,
) -> torch.Tensor:
    """Compute local expert outputs using BI-V100 native kernels.

    This replaces the vllm fused_experts() call which depends on
    ixformer.functions.vllm_moe_align_block_size (unavailable on BI-V100).

    Uses the same kernel stack as Qwen3_5MoeSparseBlock._pure_pytorch_experts:
    - Tier 1: CUTLASS grouped GEMM (gemm_grouped) for batches
    - Tier 2: per-expert F.linear loop as fallback

    Each recv item is a (token, expert) pair with top_k=1, already weighted.
    """
    N, H = recv_hidden.shape
    device = recv_hidden.device

    _gemm_grouped = kernels["gemm_grouped"]
    _use_gemm_grouped = kernels["use_gemm_grouped"]
    _corex_moe_index_combine = kernels["corex_moe_index_combine"]
    _use_corex_index = kernels["use_corex_index"]
    _xllm_moe = kernels["xllm_moe"]
    _use_xllm_moe = kernels["use_xllm_moe"]

    # Each received item is already a single (token, expert) pair.
    # local_ids is (N,) with values in 0..num_local_experts-1.
    # We need to group tokens by expert, run expert computation, then
    # re-weight and return in the original order.

    flat_eids = local_ids.to(torch.int64)

    # --- Sort tokens by expert id ---
    if _use_corex_index:
        src_dst, dst_src, expert_sizes = \
            _corex_moe_index_combine.moe_compute_index(
                flat_eids, num_local_experts)
        sorted_tok_ids = dst_src.long()
        expert_counts = expert_sizes.tolist()
    elif _use_xllm_moe:
        src_dst, dst_src, expert_sizes = \
            _xllm_moe.moe_compute_index(flat_eids, num_local_experts)
        sorted_tok_ids = dst_src.long()
        expert_counts = expert_sizes.tolist()
    else:
        order = torch.argsort(flat_eids, stable=True)
        sorted_tok_ids = order
        expert_counts = torch.bincount(
            flat_eids, minlength=num_local_experts).tolist()

    sorted_hidden = recv_hidden[sorted_tok_ids]    # (N, H)
    sorted_weights = recv_weights[sorted_tok_ids]  # (N,)

    # --- Expert computation ---
    if _use_gemm_grouped and recv_hidden.dtype == torch.float16:
        # CUTLASS grouped GEMM path
        expert_counts_t = torch.tensor(
            expert_counts, dtype=torch.int32, device=device) \
            if not isinstance(expert_counts, torch.Tensor) \
            else expert_counts.to(dtype=torch.int32, device=device)

        # FC1: w13 (gate_proj + up_proj)  → (N, 2*I)
        gemm1_out = _gemm_grouped.moe_group_gemm(
            sorted_hidden, w13, expert_counts_t)
        gate, up = gemm1_out.chunk(2, dim=-1)
        act_out = F.silu(gate) * up  # (N, I)

        # FC2: w2 (down_proj) → (N, H)
        gemm2_out = _gemm_grouped.moe_group_gemm(
            act_out, w2, expert_counts_t)

        # Apply topk weights and unsort back to original order
        weighted = (gemm2_out * sorted_weights.unsqueeze(-1)).to(recv_hidden.dtype)
        output = torch.empty_like(recv_hidden)
        output[sorted_tok_ids] = weighted
    else:
        # Per-expert F.linear loop (always works, no ixformer dependency)
        output = torch.zeros(N, H, dtype=recv_hidden.dtype, device=device)
        start = 0
        for eid, count in enumerate(expert_counts):
            end = start + count
            if count == 0:
                start = end
                continue
            tok_ids = sorted_tok_ids[start:end]
            tokens = recv_hidden[tok_ids]                  # (n, H)
            w = sorted_weights[start:end].unsqueeze(-1)    # (n, 1)

            gate_up = F.linear(tokens, w13[eid])           # (n, 2*I)
            gate, up_val = gate_up.chunk(2, dim=-1)
            act = F.silu(gate) * up_val                    # (n, I)
            down = F.linear(act, w2[eid])                  # (n, H)

            output[tok_ids] = (down * w).to(recv_hidden.dtype)
            start = end

    return output


def patch_fused_moe_for_ep():
    """Monkey-patch FusedMoE.__init__, weight_loader, and forward to support EP.

    Call this during model loading (e.g. in qwen3_5.py module init or
    patch_ops.sh) BEFORE any FusedMoE layers are constructed.

    Only checks the env var at patch time. Actual ep_size/ep_rank are
    queried at runtime (inside __init__, weight_loader, forward) when
    torch.distributed is guaranteed to be initialized.
    """
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    if not _is_ep_requested():
        logger.info("[PR #2269] EP not enabled (VLLM_ENABLE_EXPERT_PARALLEL!=1), "
                    "skipping FusedMoE EP patch")
        return

    logger.info("[PR #2269] EP requested via env var, installing FusedMoE monkey-patches "
                "(ep_size/ep_rank will be resolved at runtime)")

    # Probe BI-V100 kernels once at patch time
    _kernels = _probe_bi100_kernels()
    logger.info("[PR #2269] BI-V100 kernel probe: gemm_grouped=%s, "
                "corex_index=%s, xllm_moe=%s",
                _kernels["use_gemm_grouped"],
                _kernels["use_corex_index"],
                _kernels["use_xllm_moe"])

    _orig_init = FusedMoE.__init__

    def _ep_init(self, num_experts, top_k, hidden_size, intermediate_size,
                 params_dtype=None, reduce_results=False, renormalize=True,
                 use_grouped_topk=False, num_expert_group=None,
                 topk_group=None, quant_config=None, tp_size=None,
                 prefix="", custom_routing_function=None):
        """EP-aware __init__: allocate only local experts, full intermediate."""

        # Query ep_size/ep_rank NOW — dist is initialized in worker processes
        ep_size, ep_rank = _get_ep_runtime_config()

        # Store global expert info before modifying
        self._ep_enabled = True
        self._ep_size = ep_size
        self._ep_rank = ep_rank
        self._global_num_experts = num_experts
        assert num_experts % ep_size == 0, (
            f"num_experts ({num_experts}) must be divisible by "
            f"ep_size ({ep_size})")
        self._num_experts_per_rank = num_experts // ep_size
        self._start_expert_id = ep_rank * self._num_experts_per_rank

        logger.info(
            "[PR #2269] FusedMoE EP init: global_experts=%d, ep_size=%d, "
            "ep_rank=%d, local_experts=%d (ids %d..%d), "
            "intermediate_size=%d (full, no TP split)",
            num_experts, ep_size, ep_rank,
            self._num_experts_per_rank,
            self._start_expert_id,
            self._start_expert_id + self._num_experts_per_rank - 1,
            intermediate_size)

        # Call original __init__ with:
        # - num_experts = local experts only (saves memory)
        # - tp_size = 1 (no TP split on intermediate — EP replaces TP for MoE)
        # - reduce_results = False (we do EP reduce ourselves)
        _orig_init(
            self,
            num_experts=self._num_experts_per_rank,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            params_dtype=params_dtype,
            reduce_results=False,  # EP does its own reduce
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            quant_config=quant_config,
            tp_size=1,  # no TP split — EP splits by expert count instead
            prefix=prefix,
            custom_routing_function=custom_routing_function,
        )

        # Override num_experts to global for routing (select_experts sees all)
        self.num_experts = num_experts
        # But weights are sized for local experts only
        self._reduce_results_ep = reduce_results

    _orig_weight_loader = FusedMoE.weight_loader

    def _ep_weight_loader(self, param, loaded_weight, weight_name,
                          shard_id, expert_id):
        """EP-aware weight_loader: skip experts not owned by this rank.

        Critical fix: the original weight_loader calls
        get_tensor_model_parallel_rank() internally to compute tp_rank,
        then uses it to narrow() the loaded weight for TP sharding.
        Under EP we pass tp_size=1 (no TP split on MoE), so weights are
        full-sized, but tp_rank still returns the real rank (0-3).
        rank >= 1 causes narrow(dim, 512*rank, 512) to exceed dim size 512.

        Solution: temporarily monkey-patch get_tensor_model_parallel_rank
        to return 0 during EP weight loading, so narrow() always starts
        at offset 0 and loads the full (unsplit) weight.
        """
        if not getattr(self, '_ep_enabled', False):
            return _orig_weight_loader(self, param, loaded_weight,
                                       weight_name, shard_id, expert_id)

        start = self._start_expert_id
        end = start + self._num_experts_per_rank

        # Skip experts not owned by this EP rank
        if expert_id < start or expert_id >= end:
            return

        # Remap global expert_id to local index
        local_expert_id = expert_id - start

        # Monkey-patch tp rank to 0 during load so narrow() doesn't OOB.
        # The weight is already full-sized (tp_size=1), no split needed.
        import vllm.model_executor.layers.fused_moe.layer as _fused_moe_mod
        _real_get_tp_rank = _fused_moe_mod.get_tensor_model_parallel_rank
        _fused_moe_mod.get_tensor_model_parallel_rank = lambda: 0
        try:
            _orig_weight_loader(self, param, loaded_weight, weight_name,
                                shard_id, local_expert_id)
        finally:
            _fused_moe_mod.get_tensor_model_parallel_rank = _real_get_tp_rank

    _orig_forward = FusedMoE.forward

    def _ep_forward(self, hidden_states, router_logits):
        """EP-aware forward: All-Gather + Local Compute + Reduce-Scatter.

        Migrated from Google's tpu-inference/layers/common/fused_moe_gmm.py
        (Option B in Google's Qwen3.5-397B blog post), adapted for PyTorch
        on BI-V100.

        Flow (Option B — no All-to-All):
        1. Route all tokens globally (all ranks see all expert logits)
        2. All-Gather hidden states so every rank has all tokens
        3. Each rank permutes tokens by expert, computes ONLY its local
           experts (start_expert..end_expert), zeros the rest
        4. Reduce-Scatter (or All-Reduce) to sum partial results across ranks

        Why Option B over Option A (All-to-All):
        - Deterministic communication pattern (no variable-length splits)
        - Simpler logic: no sort/unsort, no send/recv count exchange
        - Google validated this at scale on Qwen3.5-397B (PR #2577)
        """
        if not getattr(self, '_ep_enabled', False):
            return _orig_forward(self, hidden_states, router_logits)

        ep_size = self._ep_size
        ep_rank = self._ep_rank
        start_expert = self._start_expert_id
        num_local_experts = self._num_experts_per_rank
        T_local, H = hidden_states.shape
        K = self.top_k
        device = hidden_states.device

        # ----- Step 1: Global routing -----
        # Every rank computes topk over ALL experts.
        #
        # Qwen3.5 uses grouped_topk (128 experts / 8 groups, topk_group=2).
        # grouped_topk is pure PyTorch — no ixformer dependency — so we call
        # it directly.  For non-grouped models, fall back to kernel-accelerated
        # topk (xllm_moe → corex → PyTorch).
        if self.use_grouped_topk:
            from vllm.model_executor.layers.fused_moe.fused_moe import (
                grouped_topk)
            logger.info("[EP] routing: grouped_topk (num_expert_group=%s, "
                        "topk_group=%s, num_experts=%d, top_k=%d)",
                        self.num_expert_group, self.topk_group,
                        self.num_experts, self.top_k)
            topk_weights, topk_ids = grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=self.top_k,
                renormalize=self.renormalize,
                num_expert_group=self.num_expert_group,
                topk_group=self.topk_group)
            topk_ids = topk_ids.to(torch.int64)
            topk_weights = topk_weights.to(hidden_states.dtype)
        elif self.custom_routing_function is not None:
            topk_weights, topk_ids = self.custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=self.top_k,
                renormalize=self.renormalize)
            topk_ids = topk_ids.to(torch.int64)
            topk_weights = topk_weights.to(hidden_states.dtype)
        else:
            # Non-grouped: use BI-V100 kernel-accelerated topk if available
            try:
                from vllm import xllm_moe as _xllm
            except ImportError:
                try:
                    import xllm_moe as _xllm
                except ImportError:
                    _xllm = None
            try:
                from vllm import corex_moe_topk_softmax as _corex_topk
            except ImportError:
                _corex_topk = None

            if _xllm is not None:
                topk_weights, topk_ids = _xllm.moe_fused_topk(
                    router_logits, self.top_k, True, None, "softmax")
                topk_ids = topk_ids.to(torch.int64)
                topk_weights = topk_weights.to(hidden_states.dtype)
            elif _corex_topk is not None:
                topk_weights, topk_ids = _corex_topk.moe_topk_softmax(
                    router_logits.float(), self.top_k, True)
                topk_ids = topk_ids.to(torch.int64)
                topk_weights = topk_weights.to(hidden_states.dtype)
            else:
                topk_logits, topk_ids = torch.topk(
                    router_logits.float(), self.top_k, dim=-1)
                topk_weights = torch.softmax(topk_logits, dim=-1)
                topk_weights = topk_weights.to(hidden_states.dtype)
        # topk_ids: (T_local, K) with values in [0, num_experts)
        # topk_weights: (T_local, K)

        # ----- Step 2: Token replication -----
        # Under TP=4 + EP=4 (same process group), every rank already sees
        # ALL tokens (TP shards hidden dimensions, not tokens).  No need
        # to all_gather — each rank can directly compute its local experts
        # on hidden_states, then all_reduce the partial results.
        #
        # all_gather + reduce_scatter is only needed when EP spans a
        # DIFFERENT group than TP (e.g. DP>1 where different DP ranks
        # have different tokens).  For now TP==EP, so skip.
        all_hidden = hidden_states
        all_topk_ids = topk_ids
        all_topk_weights = topk_weights

        T_global = all_hidden.shape[0]

        # ----- Step 3: Local expert computation -----
        # Sort all tokens by expert id, then compute only our local experts.
        # This mirrors Google's _process_tokens_locally + moe_gmm_local.
        flat_ids = all_topk_ids.view(-1)            # (T_global * K,)
        flat_weights = all_topk_weights.view(-1)    # (T_global * K,)
        flat_token_idx = torch.arange(
            T_global, device=device
        ).unsqueeze(1).expand(T_global, K).reshape(-1)  # (T_global * K,)

        # Sort by expert id (same as Google's argsort)
        sorted_order = flat_ids.argsort(stable=True)
        sorted_ids = flat_ids[sorted_order]
        sorted_token_idx = flat_token_idx[sorted_order]
        sorted_weights = flat_weights[sorted_order]

        # Group sizes: how many tokens assigned to each global expert
        num_experts_global = self.num_experts
        expert_counts = torch.bincount(
            sorted_ids, minlength=num_experts_global).tolist()

        # Gather sorted hidden states
        sorted_hidden = all_hidden[sorted_token_idx]  # (T_global * K, H)

        # Compute only local experts [start_expert, start_expert + num_local)
        # For expert ids outside our range, output is zero (same as Google's
        # valid_rows_mask approach).
        end_expert = start_expert + num_local_experts

        # Find the range in the sorted array that belongs to our experts
        cumsum = 0
        local_start_idx = 0
        local_end_idx = 0
        for eid in range(num_experts_global):
            if eid == start_expert:
                local_start_idx = cumsum
            if eid == end_expert:
                local_end_idx = cumsum
                break
            cumsum += expert_counts[eid]
        else:
            # end_expert == num_experts_global
            local_end_idx = cumsum + (expert_counts[end_expert - 1]
                                       if end_expert > start_expert else 0)
            # Recalculate properly
            local_end_idx = sum(expert_counts[:end_expert])

        # Recompute cleanly
        local_start_idx = sum(expert_counts[:start_expert])
        local_end_idx = sum(expert_counts[:end_expert])
        local_count = local_end_idx - local_start_idx

        # Expert computation on our local slice
        expert_output = torch.zeros(
            T_global * K, H, dtype=hidden_states.dtype, device=device)

        if local_count > 0:
            local_hidden = sorted_hidden[local_start_idx:local_end_idx]
            local_expert_ids = sorted_ids[local_start_idx:local_end_idx] \
                - start_expert  # remap to 0..num_local-1
            local_weights_slice = sorted_weights[local_start_idx:local_end_idx]

            local_result = _bi100_local_expert_compute(
                local_hidden, local_expert_ids, local_weights_slice,
                self.w13_weight, self.w2_weight,
                num_local_experts, _kernels)

            expert_output[local_start_idx:local_end_idx] = local_result

        # ----- Unpermute: scatter results back to token positions -----
        # Reverse the argsort to get results back in original (T_global*K) order
        unsorted_output = torch.zeros_like(expert_output)
        unsorted_output[sorted_order] = expert_output

        # Reduce topk dimension: sum over K experts per token
        # unsorted_output is (T_global * K, H), reshape to (T_global, K, H)
        token_output = unsorted_output.view(T_global, K, H).sum(dim=1)

        # ----- Step 4: All-Reduce partial results -----
        # Each rank computed a partial result (only its local experts contribute
        # non-zero values). Sum across all EP ranks to get the full output.
        # Since all ranks have the SAME token set (TP=EP), use all_reduce.
        if ep_size > 1:
            dist.all_reduce(token_output)

        return token_output

    # Apply patches
    FusedMoE.__init__ = _ep_init
    FusedMoE.weight_loader = _ep_weight_loader
    FusedMoE.forward = _ep_forward

    logger.info(
        "[PR #2269] FusedMoE patched for EP. "
        "ep_size/ep_rank will be resolved per-layer at runtime.")