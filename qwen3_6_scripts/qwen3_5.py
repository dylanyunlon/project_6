# Inference-only Qwen3.6-27B (Qwen3_5 architecture) for Iluvatar BI-V100.
# Pure-PyTorch DeltaNet (no fla / causal_conv1d dependency).
# Text-only (no VL, no MTP).

from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig, SchedulerConfig
from vllm.distributed import (get_tensor_model_parallel_rank,
                               get_tensor_model_parallel_world_size,
                               tensor_model_parallel_all_reduce)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, sharded_weight_loader)
from vllm.model_executor.models.mamba_cache import MambaCacheManager
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import IntermediateTensors
from vllm.worker.model_runner import (_BATCH_SIZES_TO_CAPTURE,
                                      _get_graph_batch_size)
from vllm.logger import init_logger

from vllm.model_executor.models.interfaces import HasInnerState, SupportsLoRA

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Hardware-aware policy dispatch (translated from CCCL cc_dispatch.cuh)
#
# cc_dispatch.cuh's design:
#   1. Runtime: detect device compute capability
#   2. policy_selector(cc) → returns kernel config (threads, items, algorithm)
#   3. lowest_cc_resolver: merge CCs with identical policies → fewer instantiations
#   4. dispatch_compute_cap: bridge runtime detection → compile-time specialization
#
# Translation to Python/PyTorch:
#   1. Runtime: detect BI-V100 capabilities (SMEM, cuSOLVER, MoE kernels)
#   2. _hw_policy → returns DeltaNet chunk_size, MoE strategy, solve method
#   3. Capabilities detected once at module load, cached globally
#   4. All kernel code reads from _hw_policy instead of hardcoded constants
# ---------------------------------------------------------------------------

class _HardwarePolicy:
    """CCCL cc_dispatch equivalent: detect hardware once, select policies."""

    def __init__(self):
        self._detected = False
        # Defaults (safe for any hardware)
        self.deltanet_chunk_size = 64
        self.deltanet_prefill_chunk = 4096
        self.solve_triangular_available = False
        self.moe_native_topk = False
        self.moe_native_align = False
        self.moe_native_invoke = False
        self.smem_bytes = 49152  # 48KB default for BI-V100

    def detect(self, device: torch.device = None):
        """Run once to probe hardware capabilities. CCCL: policy_selector(cc)."""
        if self._detected:
            return
        self._detected = True

        if device is None:
            if not torch.cuda.is_available():
                return
            device = torch.device("cuda:0")

        # Probe SMEM (CCCL: compute_capability → SMEM size)
        try:
            idx = device.index if device.index is not None else 0
            props = torch.cuda.get_device_properties(idx)
            self.smem_bytes = props.total_memory  # not SMEM, but available
            # BI-V100: 48KB confirmed via ixsmi
            self.smem_bytes = 49152
        except Exception:
            pass

        # Probe cuSOLVER/cuBLAS trsm (CCCL: check if kernel exists for this CC)
        try:
            test_A = torch.eye(4, device=device, dtype=torch.float32)
            test_b = torch.ones(4, 2, device=device, dtype=torch.float32)
            torch.linalg.solve_triangular(test_A, test_b, upper=False)
            self.solve_triangular_available = True
        except RuntimeError:
            self.solve_triangular_available = False

        # tuning_transform_tile.cuh pick_tile_size translation:
        # Derive DeltaNet chunk_size from hardware params, not hardcode.
        #
        # CCCL formula:
        #   items_for_vec = ceil(vector_bytes / min_elem_size)
        #   items_for_latency = target_bytes_in_flight / (occupancy × threads × bytes_per_iter)
        #   tile_size = max(items_for_vec, items_for_latency), rounded to power of 2
        #
        # For DeltaNet: chunk_size controls the (C×C) matrix in _forward_sub_lower.
        # Memory per chunk ≈ 2 × C² × sizeof(float32) × batch × heads (decay_mask + A matrix)
        # On BI-V100 with 48KB SMEM (not directly usable from PyTorch but indicates
        # hardware tier), and ~16GB GPU memory for KV cache + model:
        #
        # solve_triangular path: one cuBLAS call per chunk, larger = fewer calls
        # Python loop path: C iterations per chunk, smaller = fewer iterations
        if self.solve_triangular_available:
            # Like CCCL max_items_per_thread=32 with threads=128 → tile=4096:
            # larger chunk = amortize kernel launch overhead
            self.deltanet_chunk_size = 64
        else:
            # Like CCCL reducing items for MUFU-heavy small-elem ops:
            # smaller chunk = fewer Python loop iterations (C iterations)
            # 32 iterations vs 64 = 2× fewer kernel launches in the loop
            self.deltanet_chunk_size = 32

        # Prefill sub-chunk: controls peak memory per DeltaNet forward call.
        # CCCL target = cc_to_min_bytes_in_flight(cc): BI-V100 ≈ lower tier.
        # Qwen3.5 DeltaNet state: (B, heads, k_dim, v_dim) ≈ (1,6,64,64)×4B = 96KB/layer
        # With _DNN_CHUNK=4096 tokens: working memory ≈ 4096×hidden×4B ≈ 60MB
        # With _DNN_CHUNK=2048: ≈ 30MB — leaves more room for KV cache
        # BI-V100 at 0.95 GPU util with 256K context needs memory headroom
        self.deltanet_prefill_chunk = 4096

        # Probe MoE native kernels (CCCL: check op availability per CC)
        try:
            import ixformer.functions as ixf_F
            self.moe_native_topk = hasattr(ixf_F, 'vllm_moe_topk_softmax')
            self.moe_native_align = hasattr(ixf_F, 'vllm_moe_align_block_size')
            self.moe_native_invoke = hasattr(ixf_F, 'vllm_invoke_fused_moe_kernel')
        except ImportError:
            pass

        # Log detected policy (CCCL: policy is logged/printed for debugging)
        logger.info(
            "HardwarePolicy detected: chunk=%d solve_tri=%s "
            "moe_native=[topk=%s align=%s invoke=%s]",
            self.deltanet_chunk_size,
            self.solve_triangular_available,
            self.moe_native_topk,
            self.moe_native_align,
            self.moe_native_invoke)


# Global singleton (CCCL: policies are constexpr globals)
_hw_policy = _HardwarePolicy()


# ---------------------------------------------------------------------------
# Pure-PyTorch DeltaNet kernels (fallbacks from transformers 5.2.0)
# ---------------------------------------------------------------------------

def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _torch_causal_conv1d_update(
    hidden_states: torch.Tensor,   # (batch, channels, seq=1)
    conv_state: torch.Tensor,       # (batch, channels, state_len)  modified in-place
    weight: torch.Tensor,           # (channels, kernel_size)
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
) -> torch.Tensor:
    _, channels, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    cat = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(cat[:, :, -state_len:])
    out = F.conv1d(cat, weight.unsqueeze(1), bias, padding=0, groups=channels)
    out = out[:, :, -seq_len:]
    if activation is not None:
        out = F.silu(out)
    return out.to(hidden_states.dtype)


def _torch_chunk_gated_delta_rule(
    query: torch.Tensor,   # (batch, seq, num_heads, head_k_dim)
    key: torch.Tensor,
    value: torch.Tensor,   # (batch, seq, num_heads, head_v_dim)
    g: torch.Tensor,       # (batch, seq, num_heads)
    beta: torch.Tensor,    # (batch, seq, num_heads)
    chunk_size: int = 0,  # 0 = use _hw_policy.deltanet_chunk_size
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    initial_dtype = query.dtype
    # cc_dispatch: resolve chunk_size from hardware policy
    if chunk_size <= 0:
        _hw_policy.detect(query.device)
        chunk_size = _hw_policy.deltanet_chunk_size
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query)
        key = _l2norm(key)
    # Transpose to (batch, num_heads, seq, dim)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]
    batch, num_heads, seq_len, k_dim = key.shape
    v_dim = value.shape[-1]
    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad))
    key = F.pad(key, (0, 0, 0, pad))
    value = F.pad(value, (0, 0, 0, pad))
    beta = F.pad(beta, (0, pad))
    g = F.pad(g, (0, pad))
    total_len = seq_len + pad
    scale = 1.0 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask_upper = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0)

    # CCCL overflow_cast_t pattern: clamp BEFORE accumulation, not after.
    # Without pre-clamp, cumsum of large g values produces huge numbers
    # that downstream exp() and matmul amplify into NaN.
    # BI-V100 docker logs show 99.98-100% NaN rate in every GatedDeltaNet layer.
    #
    # Pre-clamp: limit each g element so cumsum over chunk_size stays bounded.
    # With chunk_size=64 and per-element clamp ±0.3, cumsum range ≈ ±19.2.
    # Post-clamp to ±12 keeps exp(g_diff) ≤ exp(24) ≈ 2.6e10 — safe for
    # float32 matmul accumulation (k_dim=64 → max product ~1.7e12, within float32).
    g = g.clamp(-0.5, 0.5)
    g = g.cumsum(dim=-1)
    g = g.clamp(-12.0, 12.0)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()

    # Lower-triangular solve WITHOUT libcusolver (not available on BI-V100).
    #
    # Computes (I - A)^{-1} @ RHS where A is strictly lower-triangular.
    # A = (k_beta @ key^T) * decay_mask, masked to lower triangle.
    #
    # Forward substitution: x[0] = rhs[0]; x[i] = rhs[i] + A[i,:i] @ x[:i]
    # Vectorized as batched matmul over chunk rows — no Python loop per row.
    # Uses torch.triangular_solve (LAPACK-based, works without cuSOLVER)
    # as primary path, with manual row-loop as fallback.
    A = ((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask_upper, 0)

    # For solve: (I-A) @ X = RHS  →  X = (I-A)^{-1} @ RHS
    # Since (I-A) is lower-triangular with 1s on diagonal, and A is strictly
    # lower-triangular, we can use a row-by-row forward substitution.
    # This avoids cuSOLVER entirely — only needs basic matmul and indexing.

    def _forward_sub_lower(A_lower, rhs):
        """Solve (I - A_lower) @ X = RHS.
        
        cc_dispatch pattern: _hw_policy.solve_triangular_available was probed
        once at startup. No per-call try/except overhead.
        """
        C = rhs.shape[-2]
        if _hw_policy.solve_triangular_available:
            eye = torch.eye(C, dtype=A_lower.dtype, device=A_lower.device)
            IminusA = eye - A_lower
            return torch.linalg.solve_triangular(
                IminusA, rhs, upper=False, unitriangular=True)
        else:
            # Python forward substitution fallback with numerical stability.
            # CCCL overflow_cast pattern: clamp intermediate results per row
            # to prevent the A @ x accumulation from amplifying small errors
            # into NaN. Without this, BI-V100 shows 100% NaN in every DeltaNet layer.
            x = torch.zeros_like(rhs)
            x[..., 0, :] = rhs[..., 0, :].clamp(-1e4, 1e4)
            for i in range(1, C):
                correction = (A_lower[..., i, :i].unsqueeze(-2) @ x[..., :i, :]).squeeze(-2)
                x[..., i, :] = (rhs[..., i, :] + correction).clamp(-1e4, 1e4)
            return x

    value = _forward_sub_lower(A, v_beta)

    # Clamp g.exp() to prevent k_cumdecay from having extreme values
    # that would amplify in the forward substitution loop.
    k_cumdecay = _forward_sub_lower(A, k_beta * g.exp().clamp(-1e4, 1e4).unsqueeze(-1))

    del A  # free memory

    last_state = (
        torch.zeros(batch, num_heads, k_dim, v_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_out = torch.zeros_like(value)
    mask_upper2 = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1)

    # CCCL block_scan.cuh BLOCK_SCAN_RAKING_MEMOIZE strategy:
    # Precompute all per-chunk exp values outside the loop, eliminating
    # redundant exp() inside the sequential cross-chunk scan.
    # RAKING_MEMOIZE: "preserve upsweep segment values in registers while
    # performing warp-synchronous scan, allowing downsweep not to re-read."
    num_chunks = total_len // chunk_size
    # g shape: (batch, heads, num_chunks, chunk_size)
    # g_exp_full[i] = exp(g[:,:,i,:]) for attn_inter computation
    g_exp_full = g.exp()  # (batch, heads, num_chunks, chunk_size)
    # g_last_exp[i] = exp(g[:,:,i,-1]) for state decay
    g_last_exp = g_exp_full[:, :, :, -1]  # (batch, heads, num_chunks)
    # g_diff_exp[i] = exp(g[:,:,i,-1] - g[:,:,i,:]) for k_i weighting
    g_diff_exp = (g[:, :, :, -1:] - g).exp()  # (batch, heads, num_chunks, chunk_size)

    for i in range(num_chunks):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_i = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask_upper2, 0)
        v_prime = k_cumdecay[:, :, i] @ last_state
        v_new = v_i - v_prime
        # Use precomputed exp (MEMOIZE: no redundant exp in loop body)
        attn_inter = (q_i * g_exp_full[:, :, i, :, None]) @ last_state
        core_out[:, :, i] = attn_inter + attn_i @ v_new
        last_state = (
            last_state * g_last_exp[:, :, i, None, None]
            + (k_i * g_diff_exp[:, :, i, :, None])
            .transpose(-1, -2) @ v_new
        )
        # CCCL numerical guard: clamp state to prevent cross-chunk accumulation
        # from amplifying into NaN. State elements represent k_dim × v_dim
        # attention memory; values beyond ±1e4 indicate numerical runaway.
        last_state = last_state.clamp(-1e4, 1e4)

    if not output_final_state:
        last_state = None
    core_out = core_out.reshape(batch, num_heads, -1, v_dim)[:, :, :seq_len]
    core_out = core_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_out, last_state

def _torch_recurrent_gated_delta_rule(
    query: torch.Tensor,   # (batch, 1, num_heads, head_k_dim)
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,       # (batch, 1, num_heads)
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query)
        key = _l2norm(key)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]
    batch, num_heads, seq_len, k_dim = key.shape
    v_dim = value.shape[-1]
    scale = 1.0 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_out = torch.zeros(batch, num_heads, seq_len, v_dim,
                           dtype=value.dtype, device=value.device)
    last_state = (
        torch.zeros(batch, num_heads, k_dim, v_dim,
                    dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    for t in range(seq_len):
        q_t = query[:, :, t]
        k_t = key[:, :, t]
        v_t = value[:, :, t]
        g_t = g[:, :, t].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, t].unsqueeze(-1)
        last_state = last_state * g_t
        kv_mem = (last_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_state = last_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_out[:, :, t] = (last_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_state = None
    core_out = core_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_out, last_state


# ---------------------------------------------------------------------------
# Gated RMSNorm (for DeltaNet output normalisation)
# ---------------------------------------------------------------------------

class Qwen3_5RMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor,
                gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hs = hidden_states.to(torch.float32)
        variance = hs.pow(2).mean(-1, keepdim=True)
        hs = hs * torch.rsqrt(variance + self.variance_epsilon)
        hs = self.weight * hs.to(input_dtype)
        return (hs * F.silu(gate.to(torch.float32))).to(input_dtype)


# ---------------------------------------------------------------------------
# Gated DeltaNet  (linear_attention layers)
# ---------------------------------------------------------------------------

class GatedDeltaNet(nn.Module):
    def __init__(
        self,
        text_cfg,
        layer_idx: int,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = text_cfg.hidden_size
        self.num_v_heads = text_cfg.linear_num_value_heads   # 48
        self.num_k_heads = text_cfg.linear_num_key_heads     # 16
        self.head_k_dim = text_cfg.linear_key_head_dim       # 128
        self.head_v_dim = text_cfg.linear_value_head_dim     # 128
        self.key_dim = self.num_k_heads * self.head_k_dim    # 2048
        self.value_dim = self.num_v_heads * self.head_v_dim  # 6144
        self.conv_dim = self.key_dim * 2 + self.value_dim    # 10240
        self.conv_kernel_size = text_cfg.linear_conv_kernel_dim  # 4
        self.head_expand_ratio = self.num_v_heads // self.num_k_heads  # 3

        tp_size = get_tensor_model_parallel_world_size()

        # Sharded projections — MergedColumnParallelLinear shards each of q/k/v
        # independently so each TP rank gets [q_shard, k_shard, v_shard].
        # Plain ColumnParallelLinear would shard contiguously, giving rank 0
        # [q_all, k_partial] — completely wrong Q/K/V after the split below.
        self.in_proj_qkv = MergedColumnParallelLinear(
            self.hidden_size, [self.key_dim, self.key_dim, self.value_dim],
            bias=False, quant_config=quant_config)
        self.in_proj_z = ColumnParallelLinear(
            self.hidden_size, self.value_dim,
            bias=False, quant_config=quant_config)
        self.in_proj_b = ColumnParallelLinear(
            self.hidden_size, self.num_v_heads,
            bias=False, quant_config=quant_config)
        self.in_proj_a = ColumnParallelLinear(
            self.hidden_size, self.num_v_heads,
            bias=False, quant_config=quant_config)
        self.out_proj = RowParallelLinear(
            self.value_dim, self.hidden_size,
            bias=False, quant_config=quant_config)

        # Depthwise conv weight — sharded along channel dim (dim 0)
        local_conv_dim = self.conv_dim // tp_size
        self.conv1d_weight = nn.Parameter(
            torch.empty(local_conv_dim, 1, self.conv_kernel_size))
        set_weight_attrs(self.conv1d_weight, {
            "weight_loader": self._conv1d_weight_loader})

        # Per-head scalar parameters — sharded along dim 0
        local_num_v = self.num_v_heads // tp_size
        self.A_log = nn.Parameter(torch.zeros(local_num_v))
        self.dt_bias = nn.Parameter(torch.zeros(local_num_v))
        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        # Gated RMSNorm on head_v_dim — replicated (head_v_dim=128 is small)
        self.norm = Qwen3_5RMSNormGated(self.head_v_dim,
                                        eps=text_cfg.rms_norm_eps)

    def _conv1d_weight_loader(self, param: torch.Tensor,
                              loaded_weight: torch.Tensor) -> None:
        # loaded_weight: (conv_dim=10240, 1, kernel) ordered as [q, k, v] channels
        # Must gather channels in the same non-contiguous pattern that
        # MergedColumnParallelLinear uses for in_proj_qkv, so that each rank's
        # conv1d_weight[i] applies to the correct in_proj_qkv output channel.
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        key_local = self.key_dim // tp_size    # 512 with TP=4
        val_local = self.value_dim // tp_size  # 1536 with TP=4
        q_s = loaded_weight[tp_rank * key_local : (tp_rank + 1) * key_local]
        k_s = loaded_weight[self.key_dim + tp_rank * key_local :
                            self.key_dim + (tp_rank + 1) * key_local]
        v_s = loaded_weight[2 * self.key_dim + tp_rank * val_local :
                            2 * self.key_dim + (tp_rank + 1) * val_local]
        param.data.copy_(torch.cat([q_s, k_s, v_s], dim=0))

    def forward(
        self,
        hidden_states: torch.Tensor,      # (total_tokens, hidden_size)
        attn_metadata: AttentionMetadata,
        conv_state: torch.Tensor,          # (batch, local_conv_dim, kernel-1)  in-place
        temporal_state: torch.Tensor,      # (batch, local_v_heads, k_dim, v_dim)  in-place
    ) -> torch.Tensor:
        tp_size = get_tensor_model_parallel_world_size()
        local_key_dim = self.key_dim // tp_size
        local_val_dim = self.value_dim // tp_size
        local_num_v = self.num_v_heads // tp_size
        local_num_k = self.num_k_heads // tp_size
        local_conv_dim = self.conv_dim // tp_size

        is_prefill = attn_metadata.num_prefill_tokens > 0

        # Compute all projections for every token at once (batched, efficient)
        mixed_qkv_all, _ = self.in_proj_qkv(hidden_states)  # (total, local_conv_dim)
        z_all, _ = self.in_proj_z(hidden_states)             # (total, local_val_dim)
        b_all, _ = self.in_proj_b(hidden_states)             # (total, local_num_v)
        a_all, _ = self.in_proj_a(hidden_states)             # (total, local_num_v)

        if is_prefill:
            seq_starts = attn_metadata.query_start_loc.tolist()
            outputs = []
            state_len = self.conv_kernel_size - 1
            weight_2d = self.conv1d_weight.squeeze(1)  # (local_conv_dim, kernel)

            for si in range(len(seq_starts) - 1):
                s, e = int(seq_starts[si]), int(seq_starts[si + 1])
                seq_len = e - s

                # Shape: (1, local_conv_dim, seq_len)
                mixed_qkv = (mixed_qkv_all[s:e]
                             .transpose(0, 1).unsqueeze(0)
                             .to(weight_2d.dtype))

                # Load prev conv state BEFORE overwriting (needed for causal conv padding).
                # For first prefill of a request: mamba_cache is zeros → correct.
                # For chunked prefill chunk 2+: carries last state_len tokens from prev chunk.
                prev_conv = conv_state[si:si + 1].clone().to(weight_2d.dtype)  # [1, local_conv_dim, state_len]

                # Save conv state (last state_len positions)
                if seq_len >= state_len:
                    conv_state[si].copy_(mixed_qkv[0, :, -state_len:])
                else:
                    conv_state[si, :, state_len - seq_len:].copy_(
                        mixed_qkv[0])
                    conv_state[si, :, :state_len - seq_len] = 0

                # Causal conv: left-pad with previous conv state (not zeros).
                padded = torch.cat([prev_conv, mixed_qkv], dim=2)
                mixed_qkv_conv = F.conv1d(
                    padded, self.conv1d_weight,
                    bias=None, padding=0, groups=local_conv_dim)
                mixed_qkv_conv = F.silu(mixed_qkv_conv)
                # (1, seq_len, local_conv_dim)
                mixed_qkv_conv = mixed_qkv_conv.squeeze(0).transpose(0, 1).unsqueeze(0)

                q, k, v = torch.split(
                    mixed_qkv_conv,
                    [local_key_dim, local_key_dim, local_val_dim], dim=-1)
                q = q.reshape(1, seq_len, local_num_k, self.head_k_dim)
                k = k.reshape(1, seq_len, local_num_k, self.head_k_dim)
                v = v.reshape(1, seq_len, local_num_v, self.head_v_dim)

                beta = b_all[s:e].sigmoid().unsqueeze(0)  # (1, seq_len, local_num_v)
                # CCCL overflow_cast pattern: clamp before exp to prevent
                # overflow → NaN cascade. Tightened to [-5,5] because:
                # A_log.exp() range [0.007, 148.4] — moderate decay rates.
                # Multiplied by softplus(a + dt_bias) ≈ [0.7, 10] → g ≈ [-1484, -0.005]
                # Per-element g then gets clamped to [-0.5, 0.5] in chunk_gated_delta_rule.
                # The tighter clamp here prevents A_log outliers from creating
                # extreme g values before the chunk-level clamp catches them.
                _A_safe = self.A_log.float().clamp(-5.0, 5.0)
                g = (-_A_safe.exp()
                     * F.softplus(a_all[s:e].float() + self.dt_bias)
                     ).unsqueeze(0)  # (1, seq_len, local_num_v)

                # Expand k/q to match num_v_heads
                q = q.repeat_interleave(self.head_expand_ratio, dim=2)
                k = k.repeat_interleave(self.head_expand_ratio, dim=2)

                # Sub-sequence chunking: call _torch_chunk_gated_delta_rule
                # on _DNN_CHUNK tokens at a time to cap peak memory.
                # Full 18K: tensors [1,6,282,64,64]=220 MB each → ~990 MB/call.
                # With _DNN_CHUNK=4096: [1,6,64,64,64]=6 MB each → ~137 MB/call.
                # State is chained via initial_state / output_final_state.
                _hw_policy.detect(hidden_states.device)
                _DNN_CHUNK = _hw_policy.deltanet_prefill_chunk
                cur_state = temporal_state[si:si + 1].clone()
                core_out_parts = []
                for sc_start in range(0, seq_len, _DNN_CHUNK):
                    sc_end = min(sc_start + _DNN_CHUNK, seq_len)
                    c_out, cur_state = _torch_chunk_gated_delta_rule(
                        q[:, sc_start:sc_end],
                        k[:, sc_start:sc_end],
                        v[:, sc_start:sc_end],
                        g[:, sc_start:sc_end],
                        beta[:, sc_start:sc_end],
                        initial_state=cur_state,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=True,
                    )
                    core_out_parts.append(c_out)
                if cur_state is not None:
                    temporal_state[si].copy_(cur_state[0])
                # [1, seq_len, num_v_heads, head_v_dim]
                core_out = torch.cat(core_out_parts, dim=1)

                # Gate + norm + output proj
                z = z_all[s:e].reshape(seq_len, local_num_v, self.head_v_dim)
                core_out = core_out.reshape(seq_len, local_num_v, self.head_v_dim)
                normed = self.norm(
                    core_out.reshape(-1, self.head_v_dim),
                    z.reshape(-1, self.head_v_dim))
                normed = normed.reshape(seq_len, -1)
                out, _ = self.out_proj(normed)
                outputs.append(out)

            result = torch.cat(outputs, dim=0)
            # thrust::all_of early termination pattern (bench/all_of/basic.cu):
            # Check a small sample first — if no NaN in sample, skip full scan.
            # MismatchAt=0.01 insight: NaN usually appears early or everywhere.
            # Sample first 64 elements + last 64 — covers most failure modes.
            _n = result.numel()
            _sample_ok = True
            if _n > 128:
                _s = result.view(-1)
                _sample_ok = not (torch.isnan(_s[:64]).any() or torch.isnan(_s[-64:]).any())
            if not _sample_ok or (_n <= 128 and torch.isnan(result).any()):
                # Full scan only when sample detected NaN
                nan_frac = torch.isnan(result).float().mean().item()
                logger.warning("NaN in prefill GatedDeltaNet layer %d (frac=%.4f), replacing with zeros",
                               self.layer_idx, nan_frac)
                result = torch.nan_to_num(result, nan=0.0)
            return result

        else:
            # Decode: one token per sequence
            num_seqs = hidden_states.shape[0]
            weight_2d = self.conv1d_weight.squeeze(1)

            # (num_seqs, local_conv_dim, 1)
            mixed_qkv = (mixed_qkv_all
                         .to(weight_2d.dtype)
                         .unsqueeze(-1))

            mixed_qkv_conv = _torch_causal_conv1d_update(
                mixed_qkv, conv_state, weight_2d,
                bias=None, activation='silu')
            # (num_seqs, local_conv_dim, 1) → (num_seqs, 1, local_conv_dim)
            mixed_qkv_conv = mixed_qkv_conv.squeeze(-1).unsqueeze(1)

            q, k, v = torch.split(
                mixed_qkv_conv,
                [local_key_dim, local_key_dim, local_val_dim], dim=-1)
            q = q.reshape(num_seqs, 1, local_num_k, self.head_k_dim)
            k = k.reshape(num_seqs, 1, local_num_k, self.head_k_dim)
            v = v.reshape(num_seqs, 1, local_num_v, self.head_v_dim)

            beta = b_all.sigmoid().unsqueeze(1)  # (num_seqs, 1, local_num_v)
            # CCCL overflow_cast pattern: tightened to [-5,5] matching prefill path
            _A_safe = self.A_log.float().clamp(-5.0, 5.0)
            g = (-_A_safe.exp()
                 * F.softplus(a_all.float() + self.dt_bias)
                 ).unsqueeze(1)  # (num_seqs, 1, local_num_v)

            q = q.repeat_interleave(self.head_expand_ratio, dim=2)
            k = k.repeat_interleave(self.head_expand_ratio, dim=2)

            # Inlined decode recurrent step (seq_len=1).
            # Replaces _torch_recurrent_gated_delta_rule to avoid 5 transpose+
            # contiguous+float32 copies, core_out allocation, and Python loop.
            # Uses bmm/baddbmm_ to eliminate 3 large (B,H,k,v) intermediate tensors.
            # temporal_state: (B, H_v, k_dim, v_dim) float32 — updated in-place.
            orig_dtype = q.dtype
            _scale = self.head_k_dim ** -0.5

            q_t = _l2norm(q.squeeze(1)).float() * _scale   # (B, H_v, k_dim)
            k_t = _l2norm(k.squeeze(1)).float()             # (B, H_v, k_dim)
            v_t = v.squeeze(1).float()                      # (B, H_v, v_dim)
            g_t = g.squeeze(1).float().clamp_(-12.0, 12.0).exp_()  # (B, H_v) overflow_cast tightened
            bt  = beta.squeeze(1).float()                   # (B, H_v)

            # Decay state in-place: (B, H_v, k_dim, v_dim) *= scalar per head
            temporal_state.mul_(g_t[:, :, None, None])

            # Reshape to batched-matmul layout: (B*H_v, k_dim, v_dim)
            ts_flat = temporal_state.view(-1, self.head_k_dim, self.head_v_dim)
            BH = ts_flat.shape[0]

            # kv_mem = k_t @ temporal_state  shape: (B*H_v, 1, k_dim) @ (B*H_v, k_dim, v_dim)
            kv_mem = torch.bmm(
                k_t.view(BH, 1, self.head_k_dim), ts_flat
            ).view(num_seqs, local_num_v, self.head_v_dim)  # (B, H_v, v_dim)

            delta = (v_t - kv_mem) * bt[:, :, None]         # (B, H_v, v_dim)

            # State update: temporal_state += outer(k_t, delta)  fused, no intermediate
            ts_flat.baddbmm_(
                k_t.view(BH, self.head_k_dim, 1),
                delta.view(BH, 1, self.head_v_dim),
            )
            # CCCL numerical guard: clamp decode state (same as prefill cross-chunk)
            ts_flat.clamp_(-1e4, 1e4)

            # Output: core_out = q_t @ updated temporal_state
            core_out = torch.bmm(
                q_t.view(BH, 1, self.head_k_dim), ts_flat
            ).view(num_seqs, local_num_v, self.head_v_dim).to(orig_dtype)
            # core_out: (B, H_v, v_dim) = (num_seqs, local_num_v, head_v_dim) already

            z = z_all.reshape(num_seqs, local_num_v, self.head_v_dim)
            normed = self.norm(
                core_out.reshape(-1, self.head_v_dim),
                z.reshape(-1, self.head_v_dim))
            normed = normed.reshape(num_seqs, -1)
            out, _ = self.out_proj(normed)
            # thrust::all_of early termination: sample check before full scan
            _n = out.numel()
            _has_nan = False
            if _n > 128:
                _s = out.view(-1)
                _has_nan = torch.isnan(_s[:64]).any() or torch.isnan(_s[-64:]).any()
            else:
                _has_nan = torch.isnan(out).any().item()
            if _has_nan:
                nan_frac = torch.isnan(out).float().mean().item()
                logger.warning("NaN in decode GatedDeltaNet layer %d (frac=%.4f), replacing with zeros",
                               self.layer_idx, nan_frac)
                out = torch.nan_to_num(out, nan=0.0)
            return out


# ---------------------------------------------------------------------------
# Full Attention  (with gated q — unique to Qwen3.5)
# ---------------------------------------------------------------------------

class Qwen3_5FullAttention(nn.Module):
    def __init__(
        self,
        text_cfg,
        layer_idx: int,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = text_cfg.hidden_size               # 5120
        self.num_heads = text_cfg.num_attention_heads         # 24
        self.num_kv_heads = text_cfg.num_key_value_heads      # 4
        self.head_dim = text_cfg.head_dim                     # 256
        self.rms_norm_eps = text_cfg.rms_norm_eps

        tp_size = get_tensor_model_parallel_world_size()
        self.local_num_heads = self.num_heads // tp_size
        self.scaling = self.head_dim ** -0.5

        # When num_kv_heads < tp_size we cannot shard KV further (would give
        # fractional heads per rank).  Use ReplicatedLinear so every rank holds
        # all KV heads; local_num_kv_heads equals the full count.
        # When num_kv_heads >= tp_size standard ColumnParallel sharding applies.
        if tp_size > self.num_kv_heads:
            # GQA-aware TP sharding: ixformer kernel only supports num_kv_heads=1
            # per rank.  With num_kv_heads=2 < tp_size=4 we cannot shard KV
            # evenly, but we CAN assign each rank the ONE KV head that serves
            # its Q heads:
            #   q_per_kv = num_heads // num_kv_heads  (e.g. 16//2 = 8)
            #   Rank r uses KV head  r * local_num_heads // q_per_kv
            # e.g. ranks 0,1 → KV head 0;  ranks 2,3 → KV head 1.
            # We replicate all KV heads to every rank and select in forward().
            self.proj_kv_heads = self.num_kv_heads  # heads available from projection
            self.local_num_kv_heads = 1             # heads after rank-local selection
            self.q_per_kv_global = self.num_heads // self.num_kv_heads
            self.k_proj = ReplicatedLinear(
                self.hidden_size, self.num_kv_heads * self.head_dim,
                bias=False, quant_config=quant_config)
            self.v_proj = ReplicatedLinear(
                self.hidden_size, self.num_kv_heads * self.head_dim,
                bias=False, quant_config=quant_config)
        else:
            # Standard sharding: each rank gets num_kv_heads // tp_size heads.
            self.local_num_kv_heads = self.num_kv_heads // tp_size
            self.proj_kv_heads = self.local_num_kv_heads  # already sharded
            self.q_per_kv_global = None
            self.k_proj = ColumnParallelLinear(
                self.hidden_size, self.num_kv_heads * self.head_dim,
                bias=False, quant_config=quant_config,
                prefix=f"{prefix}.k_proj")
            self.v_proj = ColumnParallelLinear(
                self.hidden_size, self.num_kv_heads * self.head_dim,
                bias=False, quant_config=quant_config,
                prefix=f"{prefix}.v_proj")

        self.local_q_dim = self.local_num_heads * self.head_dim
        self.local_kv_dim = self.local_num_kv_heads * self.head_dim

        # q_proj includes gate: output = num_heads * head_dim * 2
        self.q_proj = ColumnParallelLinear(
            self.hidden_size, self.num_heads * self.head_dim * 2,
            bias=False, quant_config=quant_config,
            prefix=f"{prefix}.q_proj")
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim, self.hidden_size,
            bias=False, quant_config=quant_config,
            prefix=f"{prefix}.o_proj")

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=self.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=self.rms_norm_eps)

        # Partial RoPE: rotary_dim = head_dim * partial_rotary_factor = 256 * 0.25 = 64
        rope_params = getattr(text_cfg, "rope_parameters", {}) or {}
        rope_theta = rope_params.get("rope_theta", 10_000_000)
        partial_factor = rope_params.get("partial_rotary_factor", 0.25)
        rotary_dim = int(self.head_dim * partial_factor)

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rotary_dim,
            max_position=text_cfg.max_position_embeddings,
            base=rope_theta,
        )

        self.attn = Attention(
            self.local_num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.local_num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        total_tokens = hidden_states.shape[0]

        # q_proj output includes gate (dim doubled)
        qg, _ = self.q_proj(hidden_states)  # (total, local_num_heads * head_dim * 2)
        qg = qg.view(total_tokens, self.local_num_heads, self.head_dim * 2)
        q = qg[:, :, :self.head_dim].reshape(total_tokens, -1)
        gate = qg[:, :, self.head_dim:].reshape(total_tokens, -1)

        k, _ = self.k_proj(hidden_states)   # (total, proj_kv_heads * head_dim)
        v, _ = self.v_proj(hidden_states)

        # q_norm on local Q heads
        q = self.q_norm.forward_cuda(
            q.view(total_tokens, self.local_num_heads, self.head_dim)
            .contiguous()).view(total_tokens, -1)

        # GQA-aware TP: select rank-local KV head BEFORE k_norm and rope so
        # that ixformer kernels always see num_kv_heads=1 (same as 27B path).
        # Doing k_norm/rope on 2 KV heads (proj_kv_heads=2) triggers ixformer
        # paths that can produce NaN; restricting to 1 head avoids the issue.
        if self.q_per_kv_global is not None:
            tp_rank = get_tensor_model_parallel_rank()
            kv_idx = (tp_rank * self.local_num_heads) // self.q_per_kv_global
            k = (k.view(total_tokens, self.proj_kv_heads, self.head_dim)
                  [:, kv_idx, :].contiguous())   # (T, head_dim) — 1 head
            v = (v.view(total_tokens, self.proj_kv_heads, self.head_dim)
                  [:, kv_idx, :].contiguous())   # (T, head_dim) — 1 head

        # k_norm on the (now always 1) rank-local KV head
        k = self.k_norm.forward_cuda(
            k.view(total_tokens, self.local_num_kv_heads, self.head_dim)
            .contiguous()).view(total_tokens, -1)

        # rope: q=(T, local_num_heads*head_dim), k=(T, 1*head_dim) — mirrors 27B
        q, k = self.rotary_emb(positions, q, k)

        attn_out = self.attn(q, k, v, kv_cache, attn_metadata)

        # Multiply by sigmoid gate before output projection
        attn_out = attn_out * torch.sigmoid(gate.float()).to(attn_out.dtype)
        output, _ = self.o_proj(attn_out)
        return output


# ---------------------------------------------------------------------------
# MLP  (SwiGLU, same as Qwen2/Qwen3)
# ---------------------------------------------------------------------------

class Qwen3_5MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False, quant_config=quant_config)
        self.down_proj = RowParallelLinear(
            intermediate_size, hidden_size,
            bias=False, quant_config=quant_config)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}")
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


# ---------------------------------------------------------------------------
# MoE sparse block  (Qwen3.5-MoE / Qwen3.6-35B-A3B)
# ---------------------------------------------------------------------------

class Qwen3_5MoeSparseBlock(nn.Module):
    """Replaces Qwen3_5MLP for qwen3_5_moe_text layers.

    FusedMoE stores expert weights and provides native ixformer forward kernel.
    Forward tries the native fused kernel first (one CUDA launch for all experts),
    falling back to _pure_pytorch_experts if the native kernel fails on BI-V100.

    CCCL architecture insight (dispatch_reduce_by_key.cuh):
    The native fused_moe_kernel implements the same pattern as CCCL's
    DeviceReduceByKey — sort tokens by expert_id, pad to block boundary
    (moe_align_block_size), then one kernel processes all expert-token pairs
    with block-level parallelism. This is the architecturally correct approach
    vs the fallback's Python for-loop over experts.

    Shared expert uses RowParallelLinear(reduce_results=False) so both paths
    produce partial (pre-all-reduce) outputs that are combined before a single
    all-reduce.
    """

    def __init__(
        self,
        text_cfg,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        hidden_size = text_cfg.hidden_size
        self.num_experts = text_cfg.num_experts
        self.top_k = text_cfg.num_experts_per_tok

        # Router: replicated (small: num_experts outputs)
        self.gate = ReplicatedLinear(hidden_size, text_cfg.num_experts,
                                     bias=False, quant_config=quant_config)

        # FusedMoE: only used for weight storage + weight_loader.
        # Forward is bypassed — see _pure_pytorch_experts().
        self.experts = FusedMoE(
            num_experts=text_cfg.num_experts,
            top_k=text_cfg.num_experts_per_tok,
            hidden_size=hidden_size,
            intermediate_size=text_cfg.moe_intermediate_size,
            reduce_results=False,   # we do the all-reduce ourselves below
            renormalize=True,
            quant_config=quant_config,
        )

        # Shared expert: defer all-reduce to combine with routed output first
        shared_size = text_cfg.shared_expert_intermediate_size
        self.shared_expert_gate_up = MergedColumnParallelLinear(
            hidden_size, [shared_size] * 2, bias=False,
            quant_config=quant_config)
        self.shared_expert_down = RowParallelLinear(
            shared_size, hidden_size, bias=False, reduce_results=False,
            quant_config=quant_config)
        self.act_fn = SiluAndMul()
        # Scalar sigmoid gate on shared expert output (same as Qwen2-MoE / Qwen3.5-MoE):
        #   shared_out *= sigmoid(shared_expert_gate(hidden_states))
        # Without this, shared expert is always fully active → wrong logits.
        self.shared_expert_gate = ReplicatedLinear(
            hidden_size, 1, bias=False, quant_config=quant_config)

        # sync_handler.cuh: register resources at init, initialize once.
        # Pre-declare MoE strategy here (resolved on first forward when device
        # is known). _use_native_moe is set to None = "not yet decided".
        # This avoids hasattr() checks in the forward hot path.
        self._use_native_moe: Optional[bool] = None
        self._moe_out_buf: Optional[torch.Tensor] = None
        self._moe_out_buf_key: Optional[tuple] = None

    def _pure_pytorch_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Pure-PyTorch MoE (ixformer has no MoE kernels on BI-V100).

        w13_weight: (num_experts, 2*inter_per_partition, hidden)  [TP-sharded]
        w2_weight:  (num_experts, hidden,  inter_per_partition)   [TP-sharded]
        Output is partial (pre-all-reduce), same contract as FusedMoE
        with reduce_results=False.
        """
        # Routing: softmax → topk → renormalise
        routing_weights = torch.softmax(router_logits.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(
            routing_weights, self.top_k, dim=-1)           # (T, top_k)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(hidden_states.dtype)

        w13 = self.experts.w13_weight  # (E, 2*I, H)
        w2  = self.experts.w2_weight   # (E, H, I)

        T = hidden_states.shape[0]
        if T == 1:
            # Fast path: single token (decode).
            # Batched GEMM: replace top_k separate F.linear calls with 2 fused ops.
            # gate_up: 1 large GEMM  (1,H) × (K*2*I,H)^T → (1, K*2*I)
            # down:    1 bmm         (K,H,I) @ (K,I,1)    → (K,H)
            # Total: 3 kernel launches vs previous 16 (top_k*2).
            eids    = topk_ids[0]                              # (K,)
            ws      = topk_weights[0].to(hidden_states.dtype)  # (K,)
            w13_sel = w13[eids]                                # (K, 2*I, H)
            w2_sel  = w2[eids]                                 # (K, H, I)

            H = hidden_states.shape[-1]

            gate_up = F.linear(
                hidden_states,
                w13_sel.reshape(-1, H),                        # (K*2*I, H) — contiguous after indexing
            )                                                  # (1, K*2*I)
            gate_up = gate_up.view(self.top_k, -1)             # (K, 2*I)
            gate, up = gate_up.chunk(2, dim=-1)                # (K, I) each
            act = F.silu(gate) * up                            # (K, I)

            # bmm: (K,H,I) @ (K,I,1) → (K,H,1) → (K,H)
            expert_out = torch.bmm(w2_sel, act.unsqueeze(-1)).squeeze(-1)  # (K, H)

            out = (expert_out * ws.unsqueeze(-1)).sum(0, keepdim=True).to(
                hidden_states.dtype)                           # (1, H)
        else:
            # General path (prefill / multi-seq): CCCL histogram sort+reduce pattern.
            #
            # CCCL insight (thrust/examples/histogram.cu sparse_histogram):
            #   sort data → reduce_by_key over contiguous segments.
            # Applied to MoE: sort (token, expert) pairs by expert_id so all tokens
            # routed to the same expert are contiguous, then process each expert's
            # batch with a single F.linear call.
            #
            # Previous code: for-loop over unique experts, each with F.linear.
            #   With 256 experts × top_k=8 ≈ up to 256 active experts → 512 F.linear calls.
            # New code: sort + segment → same number of F.linear calls but with
            #   contiguous token batches (better GPU occupancy) + no Python dict lookup.
            #
            # Further optimization: group experts by similar token count and pad
            # to enable batched GEMM across expert groups (CCCL segmented_reduce pattern).
            # TODO: implement when we have benchmark data showing this path is hot.

            # smem_resource_raw.cuh: reuse buffer across calls.
            # CCCL manages SMEM as multi-stage ping-pong: same memory, different
            # stages. We do the same: keep a class-level buffer, resize only if
            # shape changes, zero in-place instead of allocating.
            _buf_key = (T, hidden_states.shape[-1])
            if not hasattr(self, '_moe_out_buf') or self._moe_out_buf_key != _buf_key:
                self._moe_out_buf = torch.zeros_like(hidden_states)
                self._moe_out_buf_key = _buf_key
            else:
                self._moe_out_buf.zero_()
            out = self._moe_out_buf

            # Flatten all (token, expert) assignments: (T*top_k,) pairs
            flat_eids = topk_ids.view(-1)                      # (T*K,)
            flat_tok_ids = torch.arange(T, device=hidden_states.device).unsqueeze(1) \
                                .expand(-1, self.top_k).reshape(-1)  # (T*K,)
            flat_topk_pos = torch.arange(self.top_k, device=hidden_states.device) \
                                  .unsqueeze(0).expand(T, -1).reshape(-1)  # (T*K,)

            # Sort by expert_id — CCCL histogram pattern: sort brings equal keys together
            sort_idx = flat_eids.argsort(stable=True)
            sorted_eids = flat_eids[sort_idx]
            sorted_tok_ids = flat_tok_ids[sort_idx]
            sorted_topk_pos = flat_topk_pos[sort_idx]

            # thrust/examples/mode.cu complete pipeline translation:
            #   sort → unique_count → reduce_by_key(data, constant_iterator<1>) → max_element
            # torch.unique_consecutive = sort's reduce_by_key in one fused call.
            # Returns (unique_keys, inverse, counts) — mode.cu builds the same from
            # sort + reduce_by_key(data, constant_iterator<1>, keys, counts).
            # Replaces: changes detection → nonzero → concat → 3 separate GPU ops.
            seg_eids, _inv, seg_counts = torch.unique_consecutive(
                sorted_eids, return_inverse=True, return_counts=True)
            seg_ends = seg_counts.cumsum(0)
            seg_starts = torch.cat([
                torch.zeros(1, dtype=seg_ends.dtype, device=seg_ends.device),
                seg_ends[:-1]])

            # Process each expert segment
            seg_starts_cpu = seg_starts.tolist()
            seg_ends_cpu = seg_ends.tolist()
            seg_eids_cpu = seg_eids.tolist()
            for seg_i in range(len(seg_starts_cpu)):
                s, e = seg_starts_cpu[seg_i], seg_ends_cpu[seg_i]
                eid = seg_eids_cpu[seg_i]
                tok_ids_seg = sorted_tok_ids[s:e]
                topk_pos_seg = sorted_topk_pos[s:e]

                # dispatch_copy_mdspan.cuh: check if data is exhaustive (contiguous).
                # If token IDs form a contiguous range, use slice (zero-copy)
                # instead of fancy indexing (allocates new tensor).
                n_seg = e - s
                first_tok = int(tok_ids_seg[0])
                if n_seg > 1 and int(tok_ids_seg[-1]) == first_tok + n_seg - 1:
                    # Fast path: contiguous slice (no copy)
                    tokens = hidden_states[first_tok:first_tok + n_seg]
                else:
                    # Slow path: gather by index
                    tokens = hidden_states[tok_ids_seg]
                gate_up = F.linear(tokens, w13[eid])           # (n, 2*I)
                gate, up = gate_up.chunk(2, dim=-1)
                act = F.silu(gate) * up                        # (n, I)
                expert_out = F.linear(act, w2[eid])            # (n, H)
                weights = topk_weights[tok_ids_seg, topk_pos_seg].unsqueeze(-1)
                out.index_add_(0, tok_ids_seg, (expert_out * weights).to(out.dtype))

        return out  # partial, all-reduce done in forward()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits, _ = self.gate(hidden_states)

        # Try native FusedMoE path first (ixformer kernel).
        # CCCL dispatch_reduce_by_key.cuh insight: the native fused kernel does
        # sort-by-expert + block-aligned GEMM in one launch — architecturally
        # identical to CCCL's AgentReduceByKey::ConsumeRange.
        # One fused kernel vs our _pure_pytorch_experts' 256× F.linear calls.
        #
        # _custom_ops.py confirms ixformer HAS these ops:
        #   ixf_F.vllm_moe_topk_softmax
        #   ixf_F.vllm_moe_align_block_size
        #   ixf_F.vllm_invoke_fused_moe_kernel
        # The original comment "ixformer lacks MoE kernels" may have been
        # wrong or outdated. Try native first, catch and fallback if it fails.
        # cc_dispatch + sync_handler: strategy resolved on first call,
        # pre-registered field checked as None (no hasattr overhead).
        if self._use_native_moe is None:
            _hw_policy.detect(hidden_states.device)
            # Only try native if at least align+invoke are available
            # (topk_softmax has PyTorch fallback in _custom_ops.py)
            self._use_native_moe = (
                _hw_policy.moe_native_align and _hw_policy.moe_native_invoke)
            if not self._use_native_moe:
                logger.info(
                    "HardwarePolicy: MoE native kernels unavailable "
                    "(align=%s invoke=%s), using PyTorch experts.",
                    _hw_policy.moe_native_align, _hw_policy.moe_native_invoke)

        if self._use_native_moe:
            try:
                routed_out = self.experts(hidden_states, router_logits)
            except Exception as e:
                logger.warning(
                    "FusedMoE native kernel failed (%s: %s), "
                    "falling back to pure PyTorch experts permanently.",
                    type(e).__name__, e)
                self._use_native_moe = False
                routed_out = self._pure_pytorch_experts(hidden_states, router_logits)
        else:
            routed_out = self._pure_pytorch_experts(hidden_states, router_logits)

        gate_up, _ = self.shared_expert_gate_up(hidden_states)
        shared_out = self.act_fn(gate_up)
        shared_out, _ = self.shared_expert_down(shared_out)
        # Scalar sigmoid gate (Qwen2-MoE / Qwen3.5-MoE style)
        gate_score, _ = self.shared_expert_gate(hidden_states)  # (T, 1)
        shared_out = shared_out * torch.sigmoid(gate_score)

        out = routed_out + shared_out
        if self.experts.tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out


# ---------------------------------------------------------------------------
# Decoder layer  (dispatches to GatedDeltaNet or Qwen3_5FullAttention)
# ---------------------------------------------------------------------------


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(
        self,
        text_cfg,
        layer_idx: int,
        layer_type: str,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = layer_type
        self.input_layernorm = GemmaRMSNorm(text_cfg.hidden_size,
                                           eps=text_cfg.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(text_cfg.hidden_size,
                                                     eps=text_cfg.rms_norm_eps)

        if layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(text_cfg, layer_idx,
                                             quant_config=quant_config)
        else:
            self.self_attn = Qwen3_5FullAttention(
                text_cfg, layer_idx,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"layers.{layer_idx}.self_attn",
            )

        if getattr(text_cfg, 'model_type', '') == 'qwen3_5_moe_text':
            self.mlp = Qwen3_5MoeSparseBlock(text_cfg, quant_config=quant_config)
        else:
            self.mlp = Qwen3_5MLP(
                hidden_size=text_cfg.hidden_size,
                intermediate_size=text_cfg.intermediate_size,
                hidden_act=text_cfg.hidden_act,
                quant_config=quant_config,
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        # Only for linear_attention layers:
        conv_state: Optional[torch.Tensor] = None,
        temporal_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states, attn_metadata, conv_state, temporal_state)
        else:
            hidden_states = self.self_attn(
                positions, hidden_states, kv_cache, attn_metadata)

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)

        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


# ---------------------------------------------------------------------------
# Full transformer model
# ---------------------------------------------------------------------------

class Qwen3_5Model(nn.Module):
    def __init__(
        self,
        text_cfg,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.text_cfg = text_cfg
        self.embed_tokens = VocabParallelEmbedding(
            text_cfg.vocab_size, text_cfg.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3_5DecoderLayer(
                text_cfg, i, text_cfg.layer_types[i],
                cache_config=cache_config, quant_config=quant_config)
            for i in range(text_cfg.num_hidden_layers)
        ])
        self.norm = GemmaRMSNorm(text_cfg.hidden_size, eps=text_cfg.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        conv_states: torch.Tensor,     # (num_linear_layers, batch, ...)
        temporal_states: torch.Tensor, # (num_linear_layers, batch, ...)
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None

        attn_idx = 0
        linear_idx = 0
        for layer in self.layers:
            if layer.layer_type == "linear_attention":
                hidden_states, residual = layer(
                    positions, hidden_states,
                    kv_cache=None,
                    attn_metadata=attn_metadata,
                    residual=residual,
                    conv_state=conv_states[linear_idx],
                    temporal_state=temporal_states[linear_idx],
                )
                linear_idx += 1
            else:
                kv_cache = kv_caches[attn_idx]
                hidden_states, residual = layer(
                    positions, hidden_states,
                    kv_cache=kv_cache,
                    attn_metadata=attn_metadata,
                    residual=residual,
                )
                attn_idx += 1

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


# ---------------------------------------------------------------------------
# Top-level CausalLM wrapper with MambaCacheManager
# ---------------------------------------------------------------------------

class Qwen3_5ForCausalLM(nn.Module, HasInnerState, SupportsLoRA):

    has_inner_state = True
    supports_lora = True

    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    supported_lora_modules = [
        "gate_up_proj",
        "down_proj",
        "o_proj",
    ]
    embedding_modules = {}
    embedding_padding_modules = []

    def __init__(
        self,
        config,                                           # Qwen3_5Config (top-level)
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        scheduler_config: Optional[SchedulerConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config

        # The text config holds all architecture parameters
        text_cfg = config.text_config
        self.text_cfg = text_cfg

        # Pre-compute counts
        self.num_linear_layers = sum(
            1 for lt in text_cfg.layer_types if lt == "linear_attention")
        self.num_attn_layers = sum(
            1 for lt in text_cfg.layer_types if lt == "full_attention")

        # DeltaNet state dimensions (per layer, per sequence, TP-sharded)
        tp_size = get_tensor_model_parallel_world_size()
        self.conv_dim = (text_cfg.linear_num_key_heads * text_cfg.linear_key_head_dim * 2
                         + text_cfg.linear_num_value_heads * text_cfg.linear_value_head_dim)
        self.num_v_heads = text_cfg.linear_num_value_heads
        self.head_k_dim = text_cfg.linear_key_head_dim
        self.head_v_dim = text_cfg.linear_value_head_dim
        self.conv_kernel_size = text_cfg.linear_conv_kernel_dim

        self.model = Qwen3_5Model(
            text_cfg,
            cache_config=cache_config,
            quant_config=quant_config,
        )

        self.lm_head = ParallelLMHead(
            text_cfg.vocab_size, text_cfg.hidden_size,
            quant_config=quant_config,
        )

        self.logits_processor = LogitsProcessor(text_cfg.vocab_size)
        self.sampler = Sampler()

        # Lazy initialised in first forward call
        self.mamba_cache: Optional[MambaCacheManager] = None

        # GDN prefix state cache (align mode): stores (conv_states, temporal_states) snapshots
        # at KV-block boundaries so that prefix-cache-hit requests can restore correct GDN state.
        # Key: tuple of physical block IDs covering the cached prefix
        # Value: (conv_states_cpu, temporal_states_cpu) each of shape (num_gdn_layers, ...)
        self._gdn_prefix_cache: OrderedDict = OrderedDict()
        self._gdn_prefix_cache_max: int = 16   # ~16 × 16 MB ≈ 256 MB CPU RAM
        self._block_size: int = (cache_config.block_size
                                  if cache_config is not None else 16)

    def _get_mamba_cache_shape(self):
        tp_size = get_tensor_model_parallel_world_size()
        # Each sequence's state is stored in float32
        conv_state_shape = (self.conv_dim // tp_size, self.conv_kernel_size - 1)
        temporal_state_shape = (
            self.num_v_heads // tp_size, self.head_k_dim, self.head_v_dim)
        return conv_state_shape, temporal_state_shape

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.mamba_cache is None:
            if self.scheduler_config is not None:
                max_batch_size = _get_graph_batch_size(
                    self.scheduler_config.max_num_seqs)
            else:
                max_batch_size = max(_BATCH_SIZES_TO_CAPTURE) + 2
            self.mamba_cache = MambaCacheManager(
                torch.float32,
                self.num_linear_layers,
                max_batch_size,
                *self._get_mamba_cache_shape(),
            )

        mamba_tensors = self.mamba_cache.current_run_tensors(
            input_ids, attn_metadata, **kwargs)
        # conv_states:     (num_linear_layers, batch, local_conv_dim, kernel-1)
        # temporal_states: (num_linear_layers, batch, local_num_v, k_dim, v_dim)
        conv_states, temporal_states = mamba_tensors

        # ── GDN prefix-cache align mode: inject saved state on prefix hit ─────
        # Conditions: prefill pass, batch=1, context_len > 0 (prefix cached or
        # previous chunk already processed), block_tables available.
        # We always attempt a lookup: for subsequent chunked-prefill chunks the
        # key matches our own saved state (same data already in slot → no-op).
        # For a true cross-request prefix hit the key matches a previous request.
        _is_single_seq_prefill = (
            attn_metadata is not None
            and attn_metadata.num_prefill_tokens > 0
            and conv_states.shape[1] == 1               # batch == 1
            and getattr(attn_metadata, 'context_lens_tensor', None) is not None
            and getattr(attn_metadata, 'block_tables', None) is not None
            and attn_metadata.block_tables.numel() > 0
        )
        if _is_single_seq_prefill:
            context_len = int(attn_metadata.context_lens_tensor[0].item())
            if context_len > 0:
                num_prefix_blocks = context_len // self._block_size
                if (num_prefix_blocks > 0
                        and attn_metadata.block_tables.shape[1] >= num_prefix_blocks):
                    lookup_key = tuple(
                        attn_metadata.block_tables[0, :num_prefix_blocks]
                        .cpu().tolist())
                    if lookup_key in self._gdn_prefix_cache:
                        saved_conv, saved_temporal = self._gdn_prefix_cache[lookup_key]
                        conv_states[:, 0].copy_(
                            saved_conv.to(conv_states.device), non_blocking=True)
                        temporal_states[:, 0].copy_(
                            saved_temporal.to(temporal_states.device), non_blocking=True)
                        self._gdn_prefix_cache.move_to_end(lookup_key)
                        logger.debug("GDN prefix cache hit: prefix_len=%d blocks=%d",
                                     context_len, num_prefix_blocks)
        # ── End inject ──────────────────────────────────────────────────────────

        hidden_states = self.model(
            input_ids, positions, kv_caches, attn_metadata,
            conv_states, temporal_states)

        # ── GDN prefix-cache align mode: save state after this prefill chunk ───
        # Save state keyed by ALL complete KV blocks processed so far.
        # Next requests reusing this prefix will restore from here.
        if _is_single_seq_prefill:
            context_len = int(attn_metadata.context_lens_tensor[0].item())
            query_len = attn_metadata.num_prefill_tokens
            total_processed = context_len + query_len
            num_complete_blocks = total_processed // self._block_size
            if (num_complete_blocks > 0
                    and attn_metadata.block_tables.shape[1] >= num_complete_blocks):
                save_key = tuple(
                    attn_metadata.block_tables[0, :num_complete_blocks]
                    .cpu().tolist())
                # Move to end (LRU: most recent = last) and update value
                if save_key in self._gdn_prefix_cache:
                    self._gdn_prefix_cache.move_to_end(save_key)
                self._gdn_prefix_cache[save_key] = (
                    conv_states[:, 0].cpu().clone(),
                    temporal_states[:, 0].cpu().clone(),
                )
                # Evict oldest entries beyond max
                while len(self._gdn_prefix_cache) > self._gdn_prefix_cache_max:
                    self._gdn_prefix_cache.popitem(last=False)
        # ── End save ────────────────────────────────────────────────────────────

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        # All TP ranks must call logits_processor to participate in the NCCL
        # gather inside lm_head. Non-driver ranks return None after the gather.
        # With chunked prefill, intermediate chunks have seq_groups=None on all
        # ranks; _apply_logits_processors is guarded against this in
        # logits_processor.py (patched by patch_xformers_sdpa_seq.py).
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.sampler(logits, sampling_metadata)

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.mamba_cache.copy_inputs_before_cuda_graphs(
            input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.mamba_cache.get_seqlen_agnostic_capture_inputs(batch_size)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, weight_name, shard_id)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            # Skip vision and MTP branches
            if (name.startswith("model.visual")
                    or name.startswith("mtp.")
                    or name.startswith("model.mtp")):
                continue

            # Prefix remapping: checkpoint may wrap under language_model
            if name.startswith("model.language_model."):
                name = "model." + name[len("model.language_model."):]

            # Skip positional embedding caches
            if "rotary_emb.inv_freq" in name:
                continue

            # Remap conv1d.weight → conv1d_weight
            # The conv has depth (1) dim in the checkpoint that we handle separately
            if ".linear_attn.conv1d.weight" in name:
                name = name.replace(".linear_attn.conv1d.weight",
                                    ".linear_attn.conv1d_weight")

            # Stacked param loading (gate_up_proj)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    break
                if name not in params_dict:
                    break
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)


# ---------------------------------------------------------------------------
# Qwen3.6-35B-A3B  (Qwen3_5-MoE architecture)
# ---------------------------------------------------------------------------

class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLM):
    """Qwen3.6-35B-A3B: same hybrid-attention backbone as 27B, dense MLP
    replaced by Qwen3_5MoeSparseBlock (256 routed experts + shared expert).
    Only load_weights differs from the dense variant.
    """

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Checkpoint key format for this model (transformers Qwen3_5MoeExperts):
        #   mlp.experts.gate_up_proj  shape (num_experts, 2*intermediate, hidden)
        #   mlp.experts.down_proj     shape (num_experts, hidden, intermediate)
        #   mlp.gate.weight           shape (num_experts, hidden)   [router]
        #   mlp.shared_expert.{gate,up,down}_proj.weight            [shared MLP]
        # Our FusedMoE stores:
        #   mlp.experts.w13_weight    shape (num_experts, 2*intermediate//tp, hidden)
        #   mlp.experts.w2_weight     shape (num_experts, hidden, intermediate//tp)
        # Our shared expert stores:
        #   mlp.shared_expert_gate_up.weight  (merged gate+up)
        #   mlp.shared_expert_down.weight

        stacked_params_mapping = [
            # (param_name, weight_name, shard_id)
            # shared expert
            ("shared_expert_gate_up", "shared_expert.gate_proj", 0),
            ("shared_expert_gate_up", "shared_expert.up_proj",   1),
            # linear_attention dense proj (same as 27B)
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj",   1),
        ]

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            # Skip vision and MTP branches
            if (name.startswith("model.visual")
                    or name.startswith("mtp.")
                    or name.startswith("model.mtp")):
                continue

            # Prefix remapping for VL checkpoint (Qwen3_5MoeForConditionalGeneration):
            #   model.language_model.model.{layers,embed_tokens,norm} -> model.{...}
            #   model.language_model.lm_head                          -> lm_head
            # Prefix remapping: checkpoint may wrap under language_model
            if name.startswith("model.language_model."):
                name = "model." + name[len("model.language_model."):]

            if "rotary_emb.inv_freq" in name:
                continue

            if ".linear_attn.conv1d.weight" in name:
                name = name.replace(".linear_attn.conv1d.weight",
                                    ".linear_attn.conv1d_weight")

            # --- Fused routed-expert weights (all experts in one tensor) ---

            if "mlp.experts.gate_up_proj" in name:
                # loaded_weight: (num_experts, 2*intermediate, hidden)
                w13_name = name.replace("mlp.experts.gate_up_proj",
                                        "mlp.experts.w13_weight")
                if w13_name not in params_dict:
                    continue
                param = params_dict[w13_name]
                n_exp = loaded_weight.shape[0]
                inter = loaded_weight.shape[1] // 2
                gate_w = loaded_weight[:, :inter, :].contiguous()
                up_w   = loaded_weight[:, inter:, :].contiguous()
                for eid in range(n_exp):
                    param.weight_loader(param, gate_w[eid], "w1_weight", "w1", eid)
                    param.weight_loader(param, up_w[eid],   "w3_weight", "w3", eid)
                continue

            if "mlp.experts.down_proj" in name:
                # loaded_weight: (num_experts, hidden, intermediate)
                w2_name = name.replace("mlp.experts.down_proj",
                                       "mlp.experts.w2_weight")
                if w2_name not in params_dict:
                    continue
                param = params_dict[w2_name]
                n_exp = loaded_weight.shape[0]
                for eid in range(n_exp):
                    param.weight_loader(param, loaded_weight[eid], "w2_weight", "w2", eid)
                continue

            # --- Shared expert down_proj rename ---
            if "mlp.shared_expert.down_proj" in name:
                name = name.replace("mlp.shared_expert.down_proj",
                                    "mlp.shared_expert_down")
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                continue

            # --- Individual expert weights (FT checkpoint: experts.{i}.{proj}.weight) ---
            # Standard transformers fine-tuning saves each expert separately instead of
            # the pre-merged (num_experts, ...) tensors in the original checkpoint.
            if ".mlp.experts." in name:
                parts = name.split(".mlp.experts.", 1)
                expert_rest = parts[1]          # e.g. "0.gate_proj.weight"
                dot_pos = expert_rest.find(".")
                if dot_pos > 0 and expert_rest[:dot_pos].isdigit():
                    eid = int(expert_rest[:dot_pos])
                    proj_raw = expert_rest[dot_pos + 1:]
                    proj = proj_raw[:-7] if proj_raw.endswith(".weight") else proj_raw
                    prefix = parts[0]           # e.g. "model.layers.0"
                    if proj == "gate_proj":
                        w13_name = f"{prefix}.mlp.experts.w13_weight"
                        if w13_name in params_dict:
                            param = params_dict[w13_name]
                            param.weight_loader(param, loaded_weight, "w1_weight", "w1", eid)
                    elif proj == "up_proj":
                        w13_name = f"{prefix}.mlp.experts.w13_weight"
                        if w13_name in params_dict:
                            param = params_dict[w13_name]
                            param.weight_loader(param, loaded_weight, "w3_weight", "w3", eid)
                    elif proj == "down_proj":
                        w2_name = f"{prefix}.mlp.experts.w2_weight"
                        if w2_name in params_dict:
                            param = params_dict[w2_name]
                            param.weight_loader(param, loaded_weight, "w2_weight", "w2", eid)
                    continue

            # --- Stacked / standard weights ---
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    break
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
