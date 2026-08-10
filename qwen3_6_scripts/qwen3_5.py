# Inference-only Qwen3.6-27B (Qwen3_5 architecture) for Iluvatar BI-V100.
# CoreX dispatch: try native fused kernels first, fallback to PyTorch.
# CCCL env_dispatch pattern: query capability → try native → fallback.
# Text-only (no VL, no MTP).

from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple

import os
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
# ixformer hardware acceleration (BI-V100 native ops)
#
# Confirmed available on BI-V100 via SSH probe (Aug 8 2026):
#   ixformer.matmul(input, other, out=None, transa=False, transb=False, alpha=1.0, beta=0.0)
#   ixformer.softmax(input, dim=None)
#   ixformer.rms_norm(input, weight, output=None, eps=1e-6)
#   ixformer.fused_add_rms_norm(input, residual, weight, eps=1e-5, scale=1.0)
#   ixformer.silu_and_mul(input, output=None)
#   ixformer.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
#   ixformer.flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)
#   ixformer.gemv(x, A)
#
# No topk/moe/expert/gate ops available — MoE stays pure PyTorch.
# No fused GDN scan kernel — GDN loop stays, but individual ops inside are accelerated.
# ---------------------------------------------------------------------------
_ix = None
_ix_available = False

try:
    import ixformer as _ix
    _ix_available = True
    logger.info("ixformer loaded — BI-V100 hardware acceleration available")
except ImportError:
    logger.warning("ixformer not found — using pure PyTorch (no hardware acceleration)")

# corex_gdn/corex_moe: these are custom modules that teams package into their
# Docker image. If present, they provide fused GDN/MoE kernels.
# ix_bridge: C++ bridge to ixformer::infer (full MoE pipeline)
_ix_bridge_available = False
_ix_topk_softmax = None
_ix_fused_moe_forward = None
try:
    from ex_engine.python.ix_bridge import (
        topk_softmax as _ix_topk_softmax,
        fused_moe_forward as _ix_fused_moe_forward,
        is_available as _ix_bridge_check,
    )
    _ix_bridge_available = True
    logger.info("ix_bridge: full ixformer MoE pipeline available (topk + fused_moe)")
except ImportError:
    try:
        import sys
        _ex_dir = os.path.join(os.path.dirname(__file__), "ex_engine")
        if os.path.isdir(_ex_dir) and _ex_dir not in sys.path:
            sys.path.insert(0, os.path.dirname(_ex_dir))
        from ex_engine.python.ix_bridge import (
            topk_softmax as _ix_topk_softmax,
            fused_moe_forward as _ix_fused_moe_forward,
            is_available as _ix_bridge_check,
        )
        _ix_bridge_available = True
        logger.info("ix_bridge: full ixformer MoE pipeline available (deployed path)")
    except ImportError as e:
        logger.warning(
            "ix_bridge: IMPORT FAILED (%s). MoE will use PyTorch fallback. "
            "This is 3-10x slower.", e)
_corex_gdn_available = False
_corex_moe_available = False

# SM70 FlashQLA GDN kernel (from 1Cat-vLLM, MIT license)
# Fused CUDA kernel for GatedDeltaNet on SM70/SM75 (V100/BI-V100)
# JIT compiled via torch.utils.cpp_extension.load() on first call
_flash_qla_sm70 = None
_flash_qla_available = False

try:
    from vllm.model_executor.models.flash_qla_sm70 import (
        chunk_gated_delta_rule_fwd_sm70,
        chunk_gated_delta_rule_fwd_sm70_vlk_varlen,
    )
    _flash_qla_available = True
    logger.info("FlashQLA SM70 GDN module found — fused CUDA kernel available (JIT on first call)")
except ImportError as e:
    logger.warning("FlashQLA SM70 GDN not found (%s) — using PyTorch GDN", e)

try:
    from vllm.model_executor.models import corex_gdn as _corex_gdn_module
    _corex_gdn_available = True
    logger.info("CoreX GDN module found — fused GDN kernels available")
except ImportError as e:
    logger.warning("corex_gdn import failed: %s", e)

try:
    from vllm.model_executor.models import corex_moe as _corex_moe_module
    _corex_moe_available = True
    logger.info("CoreX MoE module found — fused MoE kernels available")
except ImportError as e:
    logger.warning("corex_moe import failed: %s — MoE uses PyTorch loop (SLOW)", e)

_corex_fa2_available = False
_corex_fa2_module = None
try:
    from vllm.model_executor.models import corex_fa2 as _corex_fa2_module
    _corex_fa2_available = True
    logger.info("CoreX FA2 module found — fused attention kernels available")
except ImportError as e:
    logger.warning("corex_fa2 import failed: %s", e)

# EX Engine: fused MoE topk_softmax CUDA kernel (xllm CUB-based)
_ex_moe_topk_softmax = None
_ex_moe_topk_available = False
try:
    from ex_engine.python.moe_topk import moe_topk_softmax as _ex_moe_topk_softmax
    _ex_moe_topk_available = True
    logger.info("EX Engine MoE topk_softmax kernel available")
except ImportError:
    try:
        from vllm.model_executor.models.ex_engine.moe_topk import moe_topk_softmax as _ex_moe_topk_softmax
        _ex_moe_topk_available = True
        logger.info("EX Engine MoE topk_softmax kernel available (vllm path)")
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# ixformer-accelerated ops (drop-in replacements for torch ops)
# ---------------------------------------------------------------------------

def _ix_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """BI-V100 accelerated matmul via ixformer. Only for half — ixformer rejects float32."""
    if _ix_available and a.dtype == torch.float16:
        try:
            return _ix.matmul(a, b)
        except Exception:
            pass
    return torch.matmul(a, b)

def _ix_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched matmul — ixformer.matmul handles batched half inputs."""
    if _ix_available and a.dtype == torch.float16:
        try:
            return _ix.matmul(a, b)
        except Exception:
            pass
    return torch.matmul(a, b)

def _ix_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """BI-V100 accelerated softmax via ixformer. Only for half."""
    if _ix_available and x.dtype == torch.float16:
        try:
            return _ix.softmax(x, dim=dim)
        except Exception:
            pass
    return torch.softmax(x, dim=dim)


# ---------------------------------------------------------------------------
# Pure-PyTorch DeltaNet kernels (with ixformer acceleration where possible)
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
    # CCCL agent_radix_sort_upsweep overflow pattern: UNROLL_COUNT = min(64, 255/KEYS_PER_THREAD)
    # prevents counter overflow by limiting accumulation steps.
    # Same principle: chunk_size limits cumsum steps. With pre-clamp [-5,2]:
    #   chunk=64: worst cumsum = 64*2 = 128 → exp(128) = inf
    #   chunk=16: worst cumsum = 16*2 = 32  → clamp(-20,20) catches it
    chunk_size: int = 16,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    initial_dtype = query.dtype
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

    # CCCL accumulator_t pattern: clamp BEFORE cumsum to prevent overflow
    # at the source. Without this, individual g values of ±10 accumulate
    # over 64 positions to ±640 — far beyond float32 exp() safe range (~88).
    g = g.clamp(-5.0, 2.0)
    g = g.cumsum(dim=-1)
    g = g.clamp(-20.0, 20.0)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((_ix_matmul(k_beta, key.transpose(-1, -2))) * decay_mask).masked_fill(mask_upper, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = _ix_matmul(attn, v_beta)
    k_cumdecay = _ix_matmul(attn, k_beta * g.clamp(-20, 20).exp().unsqueeze(-1))

    last_state = (
        torch.zeros(batch, num_heads, k_dim, v_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_out = torch.zeros_like(value)
    mask_upper2 = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1)

    # dispatch_scan.cuh Phase 1: pre-compute ALL chunk-local attention matrices
    # outside the state loop. attn_i[c] only depends on q, k, decay_mask — NOT state.
    # This is the CCCL "init kernel" pattern: compute everything possible
    # before the sequential scan kernel that needs tile_state propagation.
    num_chunks = total_len // chunk_size
    attn_i_all = torch.empty(
        batch, num_heads, num_chunks, chunk_size, chunk_size,
        dtype=value.dtype, device=value.device)
    for i in range(num_chunks):
        attn_i_all[:, :, i] = (
            _ix_matmul(query[:, :, i], key[:, :, i].transpose(-1, -2))
            * decay_mask[:, :, i]
        ).masked_fill_(mask_upper2, 0)

    # dispatch_scan.cuh Phase 2: sequential state propagation (scan kernel).
    # Only state-dependent ops remain in this loop.
    g_exp_cache = g.clamp(-20, 20).exp()  # pre-compute once
    g_clamped = g.clamp(-20, 20)  # keep raw clamped g for difference computation
    for i in range(num_chunks):
        v_prime = _ix_matmul(k_cumdecay[:, :, i], last_state)
        v_new = value[:, :, i] - v_prime
        attn_inter = _ix_matmul(query[:, :, i] * g_exp_cache[:, :, i, :, None], last_state)
        core_out[:, :, i] = attn_inter + _ix_matmul(attn_i_all[:, :, i], v_new)
        # State update uses difference form: exp(g[-1] - g[:]) to avoid division
        last_state = (
            last_state * g_exp_cache[:, :, i, -1, None, None]
            + _ix_matmul(
                (key[:, :, i] * (g_clamped[:, :, i, -1, None] - g_clamped[:, :, i]).exp()[..., None])
                .transpose(-1, -2), v_new)
        )

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

        # CoreX dispatch: try to create fused GDN operator from base image
        self._use_corex_gdn = False
        if _corex_gdn_available and _corex_gdn_module is not None:
            try:
                self._corex_gdn_obj = _corex_gdn_module.CoreXGDN(
                    num_v_heads=self.num_v_heads // tp_size,
                    num_k_heads=self.num_k_heads // tp_size,
                    head_k_dim=self.head_k_dim,
                    head_v_dim=self.head_v_dim,
                    conv_kernel_size=self.conv_kernel_size,
                    layer_idx=layer_idx,
                )
                self._use_corex_gdn = True
                logger.info("GatedDeltaNet layer %d: CoreX fused GDN enabled", layer_idx)
            except Exception as e:
                logger.warning(
                    "GatedDeltaNet layer %d: CoreX GDN init failed (%s), using PyTorch",
                    layer_idx, e)

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
        # CoreX dispatch: try fused GDN kernel first (CCCL env_dispatch pattern)
        if self._use_corex_gdn:
            try:
                return self._corex_gdn_obj.forward(
                    hidden_states, attn_metadata,
                    conv_state, temporal_state,
                    self.in_proj_qkv, self.in_proj_z,
                    self.in_proj_b, self.in_proj_a,
                    self.conv1d_weight, self.A_log, self.dt_bias,
                    self.norm, self.out_proj,
                )
            except Exception as e:
                if self.layer_idx == 0:
                    logger.warning(
                        "CoreX GDN forward failed (%s), falling back", e)
                self._use_corex_gdn = False  # permanent fallback

        # flash_qla SM70 DISABLED: produces inf on BI-V100 (abs mean=inf from real test)
        # xllm uses equivalent PyTorch chunked path (qwen3_gated_delta_net_base.cpp)
        # which works correctly in fp32. Keeping PyTorch path only.
        #
        # if _flash_qla_available and attn_metadata.num_prefill_tokens > 0:
        #     try:
        #         return self._flash_qla_prefill(...)

        return self._pytorch_forward(
            hidden_states, attn_metadata, conv_state, temporal_state)

    def _flash_qla_prefill(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        conv_state: torch.Tensor,
        temporal_state: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill using FlashQLA SM70 fused CUDA kernel."""
        tp_size = get_tensor_model_parallel_world_size()
        local_key_dim = self.key_dim // tp_size
        local_val_dim = self.value_dim // tp_size
        local_num_v = self.num_v_heads // tp_size
        local_num_k = self.num_k_heads // tp_size
        local_conv_dim = self.conv_dim // tp_size

        # Project all tokens
        mixed_qkv_all, _ = self.in_proj_qkv(hidden_states)
        z_all, _ = self.in_proj_z(hidden_states)
        b_all, _ = self.in_proj_b(hidden_states)
        a_all, _ = self.in_proj_a(hidden_states)

        seq_starts = attn_metadata.query_start_loc.tolist()
        outputs = []

        for i in range(len(seq_starts) - 1):
            s, e = seq_starts[i], seq_starts[i + 1]
            L = e - s
            if L == 0:
                continue

            mixed = mixed_qkv_all[s:e]  # (L, local_conv_dim)
            z_seq = z_all[s:e]
            b_seq = torch.sigmoid(b_all[s:e])  # (L, local_num_v)
            dt = F.softplus(a_all[s:e] + self.dt_bias)  # (L, local_num_v)
            gate = -dt * self.A_log.exp()  # (L, local_num_v) — decay

            # Conv1d
            conv_out = F.conv1d(
                F.pad(mixed.unsqueeze(0).transpose(1, 2),
                      (self.conv_kernel_size - 1, 0)),
                self.conv1d_weight, groups=local_conv_dim
            ).transpose(1, 2).squeeze(0)

            # Split into q, k, v
            qkv = conv_out.view(L, local_num_k + local_num_k + local_num_v,
                                self.head_k_dim)
            q_raw = qkv[:, :local_num_k, :]
            k_raw = qkv[:, local_num_k:2*local_num_k, :]
            v_raw = qkv[:, 2*local_num_k:, :local_val_dim // local_num_v]

            # L2 normalize q, k
            q = _l2norm(q_raw)
            k = _l2norm(k_raw)

            # Reshape to [1, L, H, D] for SM70 kernel
            q_4d = q.unsqueeze(0)            # (1, L, Hk, K)
            k_4d = k.unsqueeze(0)            # (1, L, Hk, K)
            v_4d = v_raw.unsqueeze(0)        # (1, L, Hv, V)
            g_3d = gate.unsqueeze(0)         # (1, L, Hv)
            # Clamp gate to prevent exp() overflow in CUDA kernel.
            # gate = -dt * A_log.exp(), typically negative (decay).
            # But pathological weights can produce positive values → exp > 1
            # → state grows exponentially over L tokens → inf.
            # PyTorch ref clamps g ∈ [-5, 2] before cumsum.
            # For recurrent kernel: clamp raw gate so exp(gate) ∈ [exp(-5), exp(2)]
            g_3d = g_3d.clamp(-5.0, 2.0)
            beta_3d = b_seq.unsqueeze(0)     # (1, L, Hv)

            # Initial state from temporal_state
            init_state = temporal_state[i:i+1]  # (1, Hv, K, V)

            # Call SM70 fused kernel
            output_4d, final_state = chunk_gated_delta_rule_fwd_sm70(
                q_4d, k_4d, v_4d, g_3d, beta_3d,
                scale=1.0,  # q already normalized
                initial_state=init_state,
                output_final_state=True,
                gate_is_exp=False,
            )

            # Update temporal state
            if final_state is not None:
                temporal_state[i] = final_state[0]

            # output_4d: (1, L, Hv, V) → (L, local_val_dim)
            out_seq = output_4d.squeeze(0).reshape(L, local_val_dim)

            # Apply gated RMSNorm + z gate
            z_seq_heads = z_seq.view(L, local_num_v, self.head_v_dim)
            out_heads = out_seq.view(L, local_num_v, self.head_v_dim)
            normed = self.norm(out_heads, z_seq_heads)
            normed_flat = normed.reshape(L, local_val_dim)

            proj_out, _ = self.out_proj(normed_flat)
            outputs.append(proj_out)

        return torch.cat(outputs, dim=0)

    def _pytorch_forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        conv_state: torch.Tensor,
        temporal_state: torch.Tensor,
    ) -> torch.Tensor:
        """Pure-PyTorch GatedDeltaNet forward (fallback path)."""
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
                # CCCL overflow guard: clamp A_log before exp to prevent
                # extreme decay rates that cause cumsum → exp → NaN chain
                _A_safe = self.A_log.float().clamp(-8.0, 4.0)
                g = (-_A_safe.exp()
                     * F.softplus(a_all[s:e].float() + self.dt_bias).clamp(max=10.0)
                     ).unsqueeze(0)  # (1, seq_len, local_num_v)

                # Expand k/q to match num_v_heads
                q = q.repeat_interleave(self.head_expand_ratio, dim=2)
                k = k.repeat_interleave(self.head_expand_ratio, dim=2)

                # Sub-sequence chunking: call _torch_chunk_gated_delta_rule
                # on _DNN_CHUNK tokens at a time to cap peak memory.
                # Full 18K: tensors [1,6,282,64,64]=220 MB each → ~990 MB/call.
                # With _DNN_CHUNK=4096: [1,6,64,64,64]=6 MB each → ~137 MB/call.
                # State is chained via initial_state / output_final_state.
                _DNN_CHUNK = 2048
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
                # Force fp16 — ixformer matmul requires kHalf
                core_out = core_out.to(torch.float16)
                z = z.to(torch.float16)
                normed = self.norm(
                    core_out.reshape(-1, self.head_v_dim),
                    z.reshape(-1, self.head_v_dim))
                normed = normed.reshape(seq_len, -1)
                out, _ = self.out_proj(normed)
                outputs.append(out)

            result = torch.cat(outputs, dim=0)
            if torch.isnan(result).any():
                logger.warning("NaN in prefill GatedDeltaNet layer %d (frac=%.4f), replacing with zeros",
                               self.layer_idx, torch.isnan(result).float().mean().item())
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
            _A_safe = self.A_log.float().clamp(-8.0, 4.0)
            g = (-_A_safe.exp()
                 * F.softplus(a_all.float() + self.dt_bias).clamp(max=10.0)
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
            g_t = g.squeeze(1).float().clamp_(-20.0, 2.0).exp_()  # (B, H_v) — clamp before exp
            bt  = beta.squeeze(1).float()                   # (B, H_v)

            # Decay state in-place: (B, H_v, k_dim, v_dim) *= scalar per head
            temporal_state.mul_(g_t[:, :, None, None])

            # Reshape to batched-matmul layout: (B*H_v, k_dim, v_dim)
            ts_flat = temporal_state.view(-1, self.head_k_dim, self.head_v_dim)
            BH = ts_flat.shape[0]

            # kv_mem = k_t @ temporal_state  shape: (B*H_v, 1, k_dim) @ (B*H_v, k_dim, v_dim)
            kv_mem = _ix_bmm(
                k_t.view(BH, 1, self.head_k_dim), ts_flat
            ).view(num_seqs, local_num_v, self.head_v_dim)  # (B, H_v, v_dim)

            delta = (v_t - kv_mem) * bt[:, :, None]         # (B, H_v, v_dim)

            # State update: temporal_state += outer(k_t, delta)  fused, no intermediate
            ts_flat.baddbmm_(
                k_t.view(BH, self.head_k_dim, 1),
                delta.view(BH, 1, self.head_v_dim),
            )
            # Clamp state to prevent gradual drift → NaN over long sequences
            temporal_state.clamp_(-65504.0, 65504.0)

            # Output: core_out = q_t @ updated temporal_state
            core_out = _ix_bmm(
                q_t.view(BH, 1, self.head_k_dim), ts_flat
            ).view(num_seqs, local_num_v, self.head_v_dim).to(orig_dtype)
            # core_out: (B, H_v, v_dim) = (num_seqs, local_num_v, head_v_dim) already

            z = z_all.reshape(num_seqs, local_num_v, self.head_v_dim)
            normed = self.norm(
                core_out.reshape(-1, self.head_v_dim),
                z.reshape(-1, self.head_v_dim))
            normed = normed.reshape(num_seqs, -1)
            out, _ = self.out_proj(normed)
            if torch.isnan(out).any():
                logger.warning("NaN in decode GatedDeltaNet layer %d (frac=%.4f), replacing with zeros",
                               self.layer_idx, torch.isnan(out).float().mean().item())
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

    FusedMoE is used ONLY for weight storage and loading (create_weights /
    weight_loader are pure PyTorch).  Its forward kernel is bypassed because
    ixformer on BI-V100 lacks vllm_moe_topk_softmax / vllm_invoke_fused_moe_kernel.
    Routing and expert computation use a pure-PyTorch loop instead.

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

        # CoreX dispatch: try to use fused MoE kernels from base image
        self._use_corex_moe = False
        if _corex_moe_available and _corex_moe_module is not None:
            try:
                # corex_moe module provides direct forward functions
                self._corex_moe_forward = getattr(
                    _corex_moe_module, 'moe_forward', None)
                if self._corex_moe_forward is not None:
                    self._use_corex_moe = True
                    logger.info("MoE: CoreX fused MoE forward available")
                else:
                    logger.warning("MoE: corex_moe has no moe_forward, using PyTorch")
            except Exception as e:
                logger.warning("MoE: CoreX MoE init failed (%s), using PyTorch", e)

    def _pure_pytorch_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """MoE expert computation with tiered dispatch.

        Dispatch order:
          Tier 0: ix_fused_moe_forward — full C++ pipeline (7 kernel launches)
          Tier 1: EX Engine CUB topk kernel + PyTorch GEMM
          Tier 2: ix_bridge topk_softmax + PyTorch GEMM
          Tier 3: Pure PyTorch (torch.softmax + torch.topk + for-loop)

        w13_weight: (num_experts, 2*inter_per_partition, hidden)  [TP-sharded]
        w2_weight:  (num_experts, hidden,  inter_per_partition)   [TP-sharded]
        Output is partial (pre-all-reduce), same contract as FusedMoE.
        """
        w13 = self.experts.w13_weight  # (E, 2*I, H)
        w2  = self.experts.w2_weight   # (E, H, I)

        # Tier 0: Full fused MoE pipeline via ixformer C++
        # 7 kernel launches vs 3*E in Python loop
        if _ix_fused_moe_forward is not None and _ix_bridge_available:
            try:
                return _ix_fused_moe_forward(
                    hidden_states, router_logits,
                    w13, w2,
                    self.top_k, self.num_experts,
                    renormalize=True,
                )
            except Exception as e:
                if not getattr(self, '_ix_fused_warned', False):
                    logger.warning("ix_fused_moe_forward failed (%s), falling back to tiered dispatch", e)
                    self._ix_fused_warned = True

        # Routing: fused topk+softmax dispatch chain
        # Tier 1: EX Engine CUB kernel → Tier 2: ix_bridge → Tier 3: PyTorch
        if _ex_moe_topk_available:
            T_tok = router_logits.shape[0]
            topk_weights = torch.empty(T_tok, self.top_k, dtype=torch.float32,
                                       device=router_logits.device)
            topk_ids = torch.empty(T_tok, self.top_k, dtype=torch.int32,
                                   device=router_logits.device)
            token_expert_indices = torch.empty(T_tok, self.top_k, dtype=torch.int32,
                                               device=router_logits.device)
            _ex_moe_topk_softmax(topk_weights, topk_ids, token_expert_indices,
                                 router_logits.float(), True)
            topk_ids = topk_ids.to(torch.long)
            topk_weights = topk_weights.to(hidden_states.dtype)
        elif _ix_bridge_available:
            topk_weights, topk_ids = _ix_topk_softmax(
                router_logits, self.top_k, renormalize=True)
            topk_weights = topk_weights.to(hidden_states.dtype)
        else:
            routing_weights = _ix_softmax(router_logits.float(), dim=-1)
            topk_weights, topk_ids = torch.topk(
                routing_weights, self.top_k, dim=-1)           # (T, top_k)
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
            topk_weights = topk_weights.to(hidden_states.dtype)

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
            expert_out = _ix_bmm(w2_sel, act.unsqueeze(-1)).squeeze(-1)  # (K, H)

            out = (expert_out * ws.unsqueeze(-1)).sum(0, keepdim=True).to(
                hidden_states.dtype)                           # (1, H)
        else:
            # General path (prefill / multi-seq): loop over unique active experts.
            # At most T*top_k unique experts, always <= num_experts.
            out = torch.zeros_like(hidden_states)
            unique_eids = topk_ids.view(-1).unique().tolist()
            for eid in unique_eids:
                eid = int(eid)
                mask = (topk_ids == eid)                       # (T, top_k)
                tok_ids, topk_pos = mask.nonzero(as_tuple=True)
                tokens = hidden_states[tok_ids]                # (n, H)
                gate_up = F.linear(tokens, w13[eid])           # (n, 2*I)
                gate, up = gate_up.chunk(2, dim=-1)
                act = F.silu(gate) * up                        # (n, I)
                expert_out = F.linear(act, w2[eid])            # (n, H)
                weights = topk_weights[tok_ids, topk_pos].unsqueeze(-1)
                out.index_add_(0, tok_ids, (expert_out * weights).to(out.dtype))

        return out  # partial, all-reduce done in forward()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits, _ = self.gate(hidden_states)

        # CoreX dispatch: try fused MoE kernel first
        if self._use_corex_moe:
            try:
                routed_out = self._corex_moe_forward(
                    hidden_states, router_logits,
                    self.experts.w13_weight, self.experts.w2_weight,
                    w3=None, topk=self.top_k,
                )
            except Exception as e:
                # NO FALLBACK — crash with error log so we can diagnose
                logger.error("CoreX MoE forward FAILED: %s", e)
                raise RuntimeError(
                    f"corex_moe.moe_forward failed: {e}. "
                    f"Shapes: hidden={hidden_states.shape}, router={router_logits.shape}, "
                    f"w13={self.experts.w13_weight.shape}, w2={self.experts.w2_weight.shape}"
                ) from e
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
