import contextlib
import functools
from typing import TYPE_CHECKING, List, Optional, Tuple, Union, Dict, Any

import torch
import torch.library

import vllm.envs as envs
from vllm._core_ext import ScalarType
from vllm.logger import init_logger
from vllm.platforms import current_platform
# import ixformer.inference.functions as ops
import ixformer.functions as ixf_F
from ixformer.distributed import _distributed as cdist
import torch.nn.functional as F

logger = init_logger(__name__)

supports_moe_ops = True

# ============================================================================
# MoE CUDA kernels — JIT-compiled from vllm v0.5.5 (torch::Tensor API)
# topk_softmax + moe_align_block_size compiled as moe_kernels.so
# ============================================================================
_moe_kernels = None
_moe_kernels_loaded = False

def _load_moe_kernels():
    """Load pre-compiled moe_kernels.so or JIT compile on demand."""
    global _moe_kernels, _moe_kernels_loaded
    if _moe_kernels_loaded:
        return _moe_kernels
    _moe_kernels_loaded = True
    
    import os, glob, importlib.util
    
    # Try pre-compiled .so from torch extensions cache
    try:
        import moe_kernels
        _moe_kernels = moe_kernels
        logger.info("[EX] moe_kernels loaded from cache")
        return _moe_kernels
    except ImportError:
        pass
    
    # Try to find .so in known locations
    search_paths = [
        os.path.expanduser('~/.cache/torch_extensions'),
        '/root/.cache/torch_extensions',
        '/workspace/ex_engine/build',
    ]
    for sp in search_paths:
        for so in glob.glob(os.path.join(sp, '**/moe_kernels*.so'), recursive=True):
            try:
                spec = importlib.util.spec_from_file_location('moe_kernels', so)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                _moe_kernels = mod
                logger.info(f"[EX] moe_kernels loaded from {so}")
                return _moe_kernels
            except Exception:
                continue
    
    # JIT compile as last resort
    moe_dir = None
    for candidate in [
        '/workspace/ex_engine/csrc/moe_v055',
        os.path.join(os.path.dirname(__file__), '..', 'model_executor', 'models', 
                     'ex_engine', 'csrc', 'moe_v055'),
    ]:
        if os.path.isdir(candidate):
            moe_dir = candidate
            break
    
    if moe_dir and os.path.isfile(os.path.join(moe_dir, 'moe_pybind.cpp')):
        try:
            from torch.utils.cpp_extension import load
            _moe_kernels = load(
                name='moe_kernels',
                sources=[
                    os.path.join(moe_dir, 'moe_pybind.cpp'),
                    os.path.join(moe_dir, 'topk_softmax_kernels.cu'),
                    os.path.join(moe_dir, 'moe_align_block_size_kernels.cu'),
                ],
                extra_include_paths=[moe_dir],
                extra_cflags=['-O2', '-std=c++17'],
                extra_cuda_cflags=['-O2', '--expt-relaxed-constexpr'],
                verbose=False,
            )
            logger.info(f"[EX] moe_kernels JIT compiled from {moe_dir}")
            return _moe_kernels
        except Exception as e:
            logger.warning(f"[EX] moe_kernels JIT compile failed: {e}")
    
    logger.warning("[EX] moe_kernels NOT available — MoE will use PyTorch path")
    return None

if TYPE_CHECKING:

    def register_fake(fn):
        return lambda name: fn
else:
    try:
        from torch.library import register_fake
    except ImportError:
        try:
            from torch.library import impl_abstract as register_fake
        except:
            def register_fake(fn):
                return lambda name: fn


def hint_on_error(fn):

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)

        except NotImplementedError as e:
            msg = (
                "Error in calling custom op %s: %s\n"
                "Not implemented or built, mostly likely because the current current device "
                "does not support this kernel (less likely TORCH_CUDA_ARCH_LIST was set "
                "incorrectly while building)")
            logger.error(msg, fn.__name__, e)
            raise NotImplementedError(msg % (fn.__name__, e)) from e
        except AttributeError as e:
            msg = (
                "Error in calling custom op %s: %s\n"
                "Possibly you have built or installed an obsolete version of vllm.\n"
                "Please try a clean build and install of vllm,"
                "or remove old built files such as vllm/*cpython*.so and build/ ."
            )
            logger.error(msg, fn.__name__, e)
            raise e

    return wrapper


# activation ops
def silu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    ixf_F.silu_and_mul(x, out)


def gelu_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    ixf_F.gelu_and_mul(x, out)


def gelu_tanh_and_mul(out: torch.Tensor, x: torch.Tensor) -> None:
    ixf_F.gelu_tanh_and_mul(x, out)


def gelu_fast(out: torch.Tensor, x: torch.Tensor) -> None:
    out.copy_(F.gelu(x,approximate="tanh"))
    return out


def gelu_new(out: torch.Tensor, x: torch.Tensor) -> None:
    out.copy_(F.gelu(x,approximate="tanh"))
    return out


def gelu_quick(out: torch.Tensor, x: torch.Tensor) -> None:
    out.copy_(F.gelu(x,approximate="tanh"))
    return out



def paged_attention_v1(
        output,
        query,
        key_cache,
        value_cache,
        head_mapping,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        alibi_slopes=None,
        kv_cache_dtype=None,
):
    return ixf_F.vllm_single_query_cached_kv_attention(
            output,
            query,
            key_cache,
            value_cache,
            head_mapping,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            alibi_slopes,
        )



def paged_attention_v2(
    out: torch.Tensor,
    exp_sum: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
    tp_rank: int = 0,
    blocksparse_local_blocks: int = 0,
    blocksparse_vert_stride: int = 0,
    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
) -> None:
    # CCCL two-pass dispatch pattern (dispatch_reduce.cuh):
    #   Pass 1: N CTAs each reduce their tile → d_block_reductions[N]
    #   Pass 2: 1 CTA reduces d_block_reductions[N] → d_out
    # Our PyTorch V2 implementation follows the same pattern:
    #   Phase 1: partition attention (each partition = one tile)
    #   Phase 2: cross-partition log-sum-exp reduction (summary_statistics binary_op)
    # paged_attention_v2_pytorch.py — try multiple import locations
    # In docker: may be at /workspace/, next to vllm package, or in vllm/ itself
    import sys, os
    _pav2 = None
    # Try 1: same package (patch_ops copies it next to _custom_ops.py)
    try:
        from vllm.paged_attention_v2_pytorch import paged_attention_v2_pytorch
        _pav2 = paged_attention_v2_pytorch
    except ImportError:
        pass
    # Try 2: /workspace/ (Dockerfile WORKDIR)
    if _pav2 is None:
        try:
            _ws = '/workspace'
            if _ws not in sys.path:
                sys.path.insert(0, _ws)
            from paged_attention_v2_pytorch import paged_attention_v2_pytorch
            _pav2 = paged_attention_v2_pytorch
        except ImportError:
            pass
    # Try 3: repo root relative to this file
    if _pav2 is None:
        _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _repo_root not in sys.path:
            sys.path.insert(0, _repo_root)
        from paged_attention_v2_pytorch import paged_attention_v2_pytorch
        _pav2 = paged_attention_v2_pytorch
    _pav2(
        out, exp_sum, max_logits, tmp_out,
        query, key_cache, value_cache,
        num_kv_heads, scale, block_tables, seq_lens,
        block_size, max_seq_len, alibi_slopes,
        kv_cache_dtype, k_scale, v_scale, tp_rank,
        blocksparse_local_blocks, blocksparse_vert_stride,
        blocksparse_block_size, blocksparse_head_sliding_step,
    )


def paged_attention_rocm(
    out: torch.Tensor,
    exp_sum: torch.Tensor,
    max_logits: torch.Tensor,
    tmp_out: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_kv_heads: int,
    scale: float,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_seq_len: int,
    alibi_slopes: Optional[torch.Tensor],
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
) -> None:
    raise NotImplementedError()


# pos encoding ops
def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    ixf_F.vllm_rotary_embedding_neox(positions, query, key, head_size,
                                  cos_sin_cache, is_neox)


def batched_rotary_embedding(positions: torch.Tensor, query: torch.Tensor,
                             key: torch.Tensor, head_size: int,
                             cos_sin_cache: torch.Tensor, is_neox: bool,
                             rot_dim: int,
                             cos_sin_cache_offsets: torch.Tensor) -> None:
    ixf_F.vllm_batched_rotary_embedding(positions, query, key, head_size,
                                          cos_sin_cache, is_neox, rot_dim,
                                          cos_sin_cache_offsets)


# layer norm ops
def rms_norm(out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor,
             epsilon: float) -> None:
    ixf_F.rms_norm(input, weight, out, epsilon)


def fused_add_rms_norm(input: torch.Tensor, residual: torch.Tensor,
                       weight: torch.Tensor, epsilon: float,
                       residual_alpha: Optional[float] = 1) -> None:
    ixf_F.fused_add_rms_norm(input, residual, weight, epsilon)


def advance_step_flashattn(num_seqs: int, num_queries: int, block_size: int,
                           input_tokens: torch.Tensor,
                           sampled_token_ids: torch.Tensor,
                           input_positions: torch.Tensor,
                           seq_lens: torch.Tensor, slot_mapping: torch.Tensor,
                           block_tables: torch.Tensor) -> None:
    """Advance a step on GPU for existing inputs for a multi-step runner"""
    return ixf_F.advance_step_flashattn(num_seqs, num_queries, block_size,
                                      input_tokens,
                                      sampled_token_ids,
                                      input_positions, 
                                      seq_lens, slot_mapping, 
                                      block_tables)


def advance_step_flashinfer(num_seqs: int, num_queries: int, block_size: int,
                            input_tokens: torch.Tensor,
                            sampled_token_ids: torch.Tensor,
                            input_positions: torch.Tensor,
                            seq_lens: torch.Tensor, slot_mapping: torch.Tensor,
                            block_tables: torch.Tensor,
                            paged_kv_indices: torch.Tensor,
                            paged_kv_indptr: torch.Tensor,
                            paged_kv_last_page_len: torch.Tensor,
                            block_table_bound: torch.Tensor) -> None:
    raise NotImplementedError("FIX SOON")


# quantization ops
# awq
def awq_dequantize(qweight: torch.Tensor, scales: torch.Tensor,
                   zeros: torch.Tensor, split_k_iters: int, thx: int,
                   thy: int) -> torch.Tensor:
    raise NotImplementedError()


def awq_gemm(input: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor,
             pack_factor, group_size: int = 128) -> torch.Tensor:
        return ixf_F.quantized_linear(input, qweight, scales,"awq",32 // pack_factor,qzeros=qzeros,group_size=group_size)


# gptq
def gptq_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
              b_gptq_qzeros: torch.Tensor, b_gptq_scales: torch.Tensor,
              b_g_idx: torch.Tensor, use_exllama: bool,
              bit: int) -> torch.Tensor:
    batch = a.shape[0]
    if batch <= 8:
        return ixf_F.quantized_linear(a,b_q_weight,b_gptq_scales,"gptq",4,b_gptq_qzeros,None,group_size=128)
    o_dtype_str = "fp16" if a.dtype == torch.half else "bf16"
    deq_w = ixf_F.quantized_weight_dequant(b_q_weight,b_gptq_scales,"gptq",o_dtype_str,4,b_gptq_qzeros,group_size=128)
    return torch.matmul(a,deq_w)


if hasattr(torch.ops._C, "gptq_gemm"):

    @register_fake("_C::gptq_gemm")
    def _gptq_gemm_fake(a: torch.Tensor, b_q_weight: torch.Tensor,
                        b_gptq_qzeros: torch.Tensor,
                        b_gptq_scales: torch.Tensor, b_g_idx: torch.Tensor,
                        use_exllama: bool, bit: int) -> torch.Tensor:
        return torch.empty((a.size(0), b_q_weight.size(1)),
                           dtype=a.dtype,
                           device=a.device)


def gptq_shuffle(q_weight: torch.Tensor, q_perm: torch.Tensor,
                 bit: int) -> None:
    return ixf_F.vllm_gptq_shuffle(q_weight,q_perm)


# marlin
def marlin_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
                b_scales: torch.Tensor, workspace: torch.Tensor, size_m: int,
                size_n: int, size_k: int) -> torch.Tensor:
    raise NotImplementedError()


# marlin_24
def gptq_marlin_24_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
                        b_meta: torch.Tensor, b_scales: torch.Tensor,
                        workspace: torch.Tensor, b_q_type: ScalarType,
                        size_m: int, size_n: int, size_k: int) -> torch.Tensor:
    raise NotImplementedError()


if hasattr(torch.ops._C, "gptq_marlin_24_gemm"):

    @register_fake("_C::gptq_marlin_24_gemm")
    def _gptq_marlin_24_gemm_fake(a: torch.Tensor, b_q_weight: torch.Tensor,
                                  b_meta: torch.Tensor, b_scales: torch.Tensor,
                                  workspace: torch.Tensor,
                                  b_q_type: ScalarType, size_m: int,
                                  size_n: int, size_k: int) -> torch.Tensor:
        return torch.empty((size_m, size_n), device=a.device, dtype=a.dtype)

    @register_fake("_C::gptq_marlin_gemm")
    def _gptq_marlin_gemm_fake(a: torch.Tensor,
                               b_q_weight: torch.Tensor,
                               b_scales: torch.Tensor,
                               b_zeros: torch.Tensor,
                               g_idx: torch.Tensor,
                               perm: torch.Tensor,
                               workspace: torch.Tensor,
                               b_q_type: ScalarType,
                               size_m: int,
                               size_n: int,
                               size_k: int,
                               is_k_full: bool,
                               has_zp: bool = False,
                               use_fp32_reduce: bool = False) -> torch.Tensor:
        return torch.empty((size_m, size_n), device=a.device, dtype=a.dtype)

    @register_fake("_C::ggml_dequantize")
    def _ggml_dequantize_fake(W: torch.Tensor, quant_type: int, m: int,
                              n: int) -> torch.Tensor:
        return torch.empty((m, n), dtype=torch.float16, device=W.device)

    @register_fake("_C::ggml_mul_mat_vec_a8")
    def _ggml_mul_mat_vec_a8_fake(
        W: torch.Tensor,
        X: torch.Tensor,
        quant_type: int,
        row: int,
    ) -> torch.Tensor:
        return torch.empty((1, row), dtype=torch.float16, device=W.device)

    @register_fake("_C::ggml_mul_mat_a8")
    def _ggml_mul_mat_a8_fake(
        W: torch.Tensor,
        X: torch.Tensor,
        quant_type: int,
        row: int,
    ) -> torch.Tensor:
        batch = X.size(0)
        return torch.empty((batch, row), dtype=torch.float16, device=W.device)

    @register_fake("_C::marlin_qqq_gemm")
    def _marlin_qqq_gemm_fake(a: torch.Tensor, b_q_weight: torch.Tensor,
                              s_tok: torch.Tensor, s_ch: torch.Tensor,
                              s_group: torch.Tensor, workspace: torch.Tensor,
                              size_m: int, size_n: int,
                              size_k: int) -> torch.Tensor:
        return torch.empty((size_m, size_n),
                           dtype=torch.float16,
                           device=a.device)

    @register_fake("_C::marlin_gemm")
    def _marlin_gemm_fake(a: torch.Tensor, b_q_weight: torch.Tensor,
                          b_scales: torch.Tensor, workspace: torch.Tensor,
                          size_m: int, size_n: int,
                          size_k: int) -> torch.Tensor:
        return torch.empty((size_m, size_n),
                           dtype=torch.float16,
                           device=a.device)

    @register_fake("_C::awq_dequantize")
    def _awq_dequantize_fake(qweight: torch.Tensor, scales: torch.Tensor,
                             zeros: torch.Tensor, split_k_iters: int, thx: int,
                             thy: int) -> torch.Tensor:
        in_c = qweight.size(0)
        qout_c = qweight.size(1)
        out_c = qout_c * 8
        return torch.empty((in_c, out_c),
                           dtype=scales.dtype,
                           device=scales.device)

    @register_fake("_C::awq_gemm")
    def _awq_gemm_fake(input: torch.Tensor, qweight: torch.Tensor,
                       qzeros: torch.Tensor, scales: torch.Tensor,
                       split_k_iters: int) -> torch.Tensor:
        num_in_feats = input.size(0)
        return torch.empty((split_k_iters, num_in_feats, qweight.size(1) * 8),
                           dtype=input.dtype,
                           device=input.device).sum(0)

    @register_fake("_C::aqlm_gemm")
    def _aqlm_gemm_fake(input: torch.Tensor, codes: torch.Tensor,
                        codebooks: torch.Tensor, scales: torch.Tensor,
                        codebook_partition_sizes: List[int],
                        bias: Optional[torch.Tensor]) -> torch.Tensor:
        out_features = codes.size(0) * codebooks.size(2)
        flat_input = input.reshape((-1, input.size(-1)))
        flat_output = torch.empty((flat_input.size(0), out_features),
                                  dtype=input.dtype,
                                  device=input.device)

        output_sizes = list(input.shape)
        output_sizes.pop()
        output_sizes.append(-1)
        return flat_output.reshape(tuple(output_sizes))

    @register_fake("_C::aqlm_dequant")
    def _aqlm_dequant_fake(
            codes: torch.Tensor, codebooks: torch.Tensor,
            codebook_partition_sizes: List[int]) -> torch.Tensor:
        in_features = codes.size(1) * 8
        out_features = codes.size(0)
        return torch.empty((out_features, in_features),
                           dtype=codebooks.dtype,
                           device=codebooks.device)

    @register_fake("_C::fp8_marlin_gemm")
    def _fp8_marlin_gemm_fake(a: torch.Tensor, b_q_weight: torch.Tensor,
                              b_scales: torch.Tensor, workspace: torch.Tensor,
                              num_bits: int, size_m: int, size_n: int,
                              size_k: int) -> torch.Tensor:
        return torch.empty((size_m, size_n), dtype=a.dtype, device=a.device)

    @register_fake("_C::machete_gemm")
    def machete_gemm_fake(
        a: torch.Tensor,
        # Should be the tensor returned by machete_prepack_B
        b_q: torch.Tensor,
        b_type: ScalarType,
        b_scales: Optional[torch.Tensor] = None,
        b_zeros: Optional[torch.Tensor] = None,
        b_group_size: Optional[int] = None,
        c: Optional[torch.Tensor] = None,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        schedule: Optional[str] = None,
    ) -> torch.Tensor:
        m = a.size(0)
        n = b_q.size(1)
        return torch.empty((m, n), device=a.device, dtype=a.dtype)

    @register_fake("_C::machete_prepack_B")
    def machete_prepack_B_fake(b_q_weight: torch.Tensor,
                               b_type: ScalarType) -> torch.Tensor:
        return torch.empty_like(b_q_weight,
                                memory_format=torch.contiguous_format)

    @register_fake("_C::causal_conv1d_fwd")
    def causal_conv1d_fwd_fake(x: torch.Tensor, weight: torch.Tensor,
                               bias_: Optional[torch.Tensor],
                               conv_states: Optional[torch.Tensor],
                               cu_seq_len: Optional[torch.Tensor],
                               cache_indices: Optional[torch.Tensor],
                               has_initial_state: Optional[torch.Tensor],
                               silu_activation: bool) -> torch.Tensor:
        return torch.empty_like(x)

    @register_fake("_C::causal_conv1d_update")
    def causal_conv1d_update_fake(
            x: torch.Tensor, conv_state: torch.Tensor, weight: torch.Tensor,
            bias_: Optional[torch.Tensor], silu_activation: bool,
            cache_seqlens: Optional[torch.Tensor],
            conv_state_indices: Optional[torch.Tensor]) -> torch.Tensor:
        return torch.empty_like(x)

    @register_fake("_C::selective_scan_fwd")
    def selective_scan_fwd_fake(u: torch.Tensor, delta: torch.Tensor,
                                A: torch.Tensor, B: torch.Tensor,
                                C: torch.Tensor, D_: Optional[torch.Tensor],
                                z_: Optional[torch.Tensor],
                                delta_bias_: Optional[torch.Tensor],
                                delta_softplus: bool,
                                cu_seq_len: Optional[torch.Tensor],
                                cache_indices: Optional[torch.Tensor],
                                has_initial_state: Optional[torch.Tensor],
                                ssm_states: Optional[torch.Tensor]) -> None:
        return None


# cutlass
def cutlass_scaled_mm_supports_fp8(cuda_device_capability: int) -> bool:
    return True


def cutlass_scaled_mm(a: torch.Tensor,
                      b: torch.Tensor,
                      scale_a: torch.Tensor,
                      scale_b: torch.Tensor,
                      out_dtype: torch.dtype,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:

    m = a.shape[0]
    n = b.shape[1]
    out = torch.empty((m, n), dtype=out_dtype, device=a.device)
    ixf_F.w8a8(a, b.transpose(0,1), scale_a, scale_b, bias, output=out, out_dtype=out_dtype)
        
    return out


def cutlass_scaled_mm_azp(a: torch.Tensor,
                          b: torch.Tensor,
                          scale_a: torch.Tensor,
                          scale_b: torch.Tensor,
                          out_dtype: torch.dtype,
                          azp_adj: torch.Tensor,
                          azp: Optional[torch.Tensor] = None,
                          bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    raise NotImplementedError()


# aqlm
def aqlm_gemm(input: torch.Tensor, codes: torch.Tensor,
              codebooks: torch.Tensor, scales: torch.Tensor,
              codebook_partition_sizes: List[int],
              bias: Optional[torch.Tensor]) -> torch.Tensor:
    raise NotImplementedError()


def aqlm_dequant(codes: torch.Tensor, codebooks: torch.Tensor,
                 codebook_partition_sizes: List[int]) -> torch.Tensor:
    raise NotImplementedError()


# gptq_marlin
def gptq_marlin_repack(b_q_weight: torch.Tensor, perm: torch.Tensor,
                       size_k: int, size_n: int,
                       num_bits: int) -> torch.Tensor:
    raise NotImplementedError()


# gptq_marlin
def awq_marlin_repack(b_q_weight: torch.Tensor, size_k: int, size_n: int,
                      num_bits: int) -> torch.Tensor:
    raise NotImplementedError()


def gptq_marlin_moe_repack(b_q_weight: torch.Tensor, perm: torch.Tensor,
                           size_k: int, size_n: int,
                           num_bits: int) -> torch.Tensor:
    raise NotImplementedError()


def awq_marlin_moe_repack(b_q_weight: torch.Tensor, perm: torch.Tensor,
                          size_k: int, size_n: int,
                          num_bits: int) -> torch.Tensor:
    num_experts = b_q_weight.shape[0]
    assert size_k % 16 == 0
    output = torch.empty((num_experts, size_k // 16, size_n * (num_bits // 2)),
                         device=b_q_weight.device,
                         dtype=b_q_weight.dtype)
    for e in range(num_experts):
        output[e] = torch.ops._C.awq_marlin_repack(b_q_weight[e], size_k,
                                                   size_n, num_bits)
    return output


def gptq_marlin_gemm(a: torch.Tensor,
                     b_q_weight: torch.Tensor,
                     b_scales: torch.Tensor,
                     b_zeros: torch.Tensor,
                     g_idx: torch.Tensor,
                     perm: torch.Tensor,
                     workspace: torch.Tensor,
                     b_q_type: ScalarType,
                     size_m: int,
                     size_n: int,
                     size_k: int,
                     is_k_full: bool,
                     has_zp: bool = False,
                     use_fp32_reduce: bool = False) -> torch.Tensor:
    raise NotImplementedError()


# fp8 marlin
def fp8_marlin_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
                    b_scales: torch.Tensor, workspace: torch.Tensor,
                    num_bits: int, size_m: int, size_n: int,
                    size_k: int) -> torch.Tensor:
    raise NotImplementedError()


# machete
def machete_supported_schedules(b_type: ScalarType) -> List[str]:
    raise NotImplementedError()


def machete_gemm(
    a: torch.Tensor,
    b_q: torch.Tensor,  # Should be the tensor returned by machete_prepack_B
    b_type: ScalarType,
    b_scales: Optional[torch.Tensor] = None,
    b_zeros: Optional[torch.Tensor] = None,
    b_group_size: Optional[int] = None,
    c: Optional[torch.Tensor] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    schedule: Optional[str] = None,
) -> torch.Tensor:
    raise NotImplementedError()


def machete_prepack_B(b_q_weight: torch.Tensor,
                      b_type: ScalarType) -> torch.Tensor:
    raise NotImplementedError()


if hasattr(torch.ops._C, "permute_cols"):

    @register_fake("_C::permute_cols")
    def _permute_cols_fake(a: torch.Tensor,
                           perm: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(a)


def permute_cols(a: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError()


# fp8
def scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    num_token_padding: Optional[int] = None,
    scale_ub: Optional[torch.Tensor] = None,
    use_per_token_if_dynamic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor to FP8 and return quantized tensor and scale.

    This function supports both static and dynamic quantization: If you
    provide the scale, it will use static scaling and if you omit it,
    the scale will be determined dynamically. The function also allows
    optional padding of the output tensors for downstream kernels that
    will benefit from padding.

    Args:
        input: The input tensor to be quantized to FP8
        scale: Optional scaling factor for the FP8 quantization
        scale_ub: Optional upper bound for scaling factor in dynamic 
            per token case
        num_token_padding: If specified, pad the first dimension
            of the output to at least this value.
        use_per_token_if_dynamic: Whether to do per_tensor or per_token 
            in the dynamic quantization case.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The output tensor in FP8 and
            scaling factor.
    """
    raise NotImplementedError()


# int8
def scaled_int8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    azp: Optional[torch.Tensor] = None,
    symmetric: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Quantize the input tensor to int8 and return the quantized tensor and scale, and maybe azp.

    Args:
        input: The input tensor to be quantized to int8.
        scale: Optional scaling factor for the int8 quantization.
            When not provided, we invoke dynamic-per-token quantization.
        azp: Optional zero-point for the int8 quantization.
            Must be provided for asymmetric quantization if `scale` is provided.
        symmetric: Whether to use symmetric quantization (scale only, azp ignored).

    Returns:
      Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]] : Output int8 tensor, scales, and optionally azp.
    """
    output = torch.empty_like(input, dtype=torch.int8)
    if scale is not None:
        # static-per-tensor quantization.
        assert symmetric == (
            azp is
            None), "azp must only be provided for asymmetric quantization."
        ixf_F.static_scaled_int8_quant(output, input, scale)
        return output, scale, None

    # dynamic-per-token quantization.
    input_scales = torch.empty((input.numel() // input.shape[-1], 1),
                               device=input.device,
                               dtype=torch.float32)
    input_azp = None if symmetric else torch.empty_like(input_scales,
                                                        dtype=torch.int32)
    ixf_F.dynamic_scaled_int8_quant(output, input, input_scales)
    return output, input_scales, input_azp


# qqq ops
def marlin_qqq_gemm(a: torch.Tensor, b_q_weight: torch.Tensor,
                    s_tok: torch.Tensor, s_ch: torch.Tensor,
                    s_group: torch.Tensor, workspace: torch.Tensor,
                    size_m: int, size_n: int, size_k: int) -> torch.Tensor:
    raise NotImplementedError()


# gguf
def ggml_dequantize(W: torch.Tensor, quant_type: int, m: int,
                    n: int) -> torch.Tensor:
    raise NotImplementedError()


def ggml_mul_mat_vec_a8(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_type: int,
    row: int,
) -> torch.Tensor:
    raise NotImplementedError()


def ggml_mul_mat_a8(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_type: int,
    row: int,
) -> torch.Tensor:
    raise NotImplementedError()


# mamba
def causal_conv1d_fwd(x: torch.Tensor, weight: torch.Tensor,
                      bias_: Optional[torch.Tensor],
                      conv_states: Optional[torch.Tensor],
                      query_start_loc: Optional[torch.Tensor],
                      cache_indices: Optional[torch.Tensor],
                      has_initial_state: Optional[torch.Tensor],
                      silu_activation: bool) -> torch.Tensor:
    raise NotImplementedError()


def causal_conv1d_update(
        x: torch.Tensor, conv_state: torch.Tensor, weight: torch.Tensor,
        bias_: Optional[torch.Tensor], silu_activation: bool,
        cache_seqlens: Optional[torch.Tensor],
        conv_state_indices: Optional[torch.Tensor]) -> torch.Tensor:
    raise NotImplementedError()


def selective_scan_fwd(
        u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
        C: torch.Tensor, D_: Optional[torch.Tensor],
        z_: Optional[torch.Tensor], delta_bias_: Optional[torch.Tensor],
        delta_softplus: bool, query_start_loc: Optional[torch.Tensor],
        cache_indices: Optional[torch.Tensor],
        has_initial_state: Optional[torch.Tensor], ssm_states: torch.Tensor):
    raise NotImplementedError()


# moe
def moe_align_block_size(topk_ids: torch.Tensor, num_experts: int,
                         block_size: int, sorted_token_ids: torch.Tensor,
                         experts_ids: torch.Tensor,
                         num_tokens_post_pad: torch.Tensor) -> None:
    # PyTorch implementation of moe_align_block_size.
    # Sort tokens by expert assignment with block-aligned padding.
    # This is the same logic as vllm's CUDA kernel but in Python.
    max_num_tokens_padded = sorted_token_ids.numel()
    num_tokens = topk_ids.numel()
    
    # Count tokens per expert
    tokens_per_expert = torch.zeros(num_experts, dtype=torch.int32, device=topk_ids.device)
    for i in range(num_tokens):
        tokens_per_expert[topk_ids.view(-1)[i]] += 1
    
    # Compute padded counts (align to block_size)
    cumsum = 0
    sorted_idx = 0
    for expert_id in range(num_experts):
        # Collect all tokens for this expert
        cnt = tokens_per_expert[expert_id].item()
        for i in range(num_tokens):
            if topk_ids.view(-1)[i].item() == expert_id:
                if sorted_idx < max_num_tokens_padded:
                    sorted_token_ids[sorted_idx] = i
                    sorted_idx += 1
        # Pad to block_size boundary
        padded_cnt = ((cnt + block_size - 1) // block_size) * block_size
        for _ in range(padded_cnt - cnt):
            if sorted_idx < max_num_tokens_padded:
                sorted_token_ids[sorted_idx] = num_tokens  # padding sentinel
                sorted_idx += 1
        # Expert id for each block
        num_blocks = padded_cnt // block_size
        for b in range(num_blocks):
            block_idx = cumsum // block_size + b
            if block_idx < experts_ids.numel():
                experts_ids[block_idx] = expert_id
        cumsum += padded_cnt
    
    num_tokens_post_pad.fill_(sorted_idx)


def invoke_fused_moe_kernel(
    A: torch.Tensor, 
    B: torch.Tensor, 
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    topk_weights: torch.Tensor, 
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool, 
    top_k: int,
    config: Dict[str, Any],
    compute_type,
    use_fp8_w8a8: bool,
    use_int8_w8a16: bool,
) -> None:
    # PyTorch implementation of fused MoE GEMM kernel.
    # For each block of sorted tokens belonging to the same expert,
    # compute C[token] = A[token] @ B[expert].T (optionally weighted).
    #
    # This replaces the Triton/CUDA fused_moe_kernel that base image expects
    # via ixf_F.vllm_invoke_fused_moe_kernel (which doesn't exist).
    num_tokens = A.shape[0]
    block_size = config.get('BLOCK_SIZE_M', 64)
    num_valid = num_tokens_post_padded.item() if isinstance(num_tokens_post_padded, torch.Tensor) else num_tokens_post_padded
    num_blocks = (num_valid + block_size - 1) // block_size
    
    for block_idx in range(min(num_blocks, expert_ids.numel())):
        expert_id = expert_ids[block_idx].item()
        start = block_idx * block_size
        end = min(start + block_size, num_valid)
        
        # Get token indices for this block
        token_indices = sorted_token_ids[start:end]
        # Filter out padding sentinels (index >= num_tokens)
        valid_mask = token_indices < num_tokens
        if not valid_mask.any():
            continue
        valid_indices = token_indices[valid_mask].long()
        
        # Gather input tokens
        a_block = A[valid_indices]  # (valid_count, K)
        # Expert weight: B is (num_experts, N, K) → B[expert_id] is (N, K)
        w = B[expert_id]  # (N, K)
        # GEMM: output = input @ weight.T
        out = torch.matmul(a_block.to(w.dtype), w.t())  # (valid_count, N)
        
        if mul_routed_weight:
            # Apply routing weights
            # valid_indices are flattened (token_idx * top_k + k)
            # We need to map back to (token_idx, k) to get the weight
            token_idx = valid_indices // top_k
            k_idx = valid_indices % top_k
            weights = topk_weights[token_idx, k_idx].unsqueeze(1).to(out.dtype)
            out = out * weights
        
        # Scatter back
        C[valid_indices] = out.to(C.dtype)


# ---------- topk_softmax: CUDA kernel → PyTorch fallback ----------
# moe_topk_softmax_v3.cu: fused warp-shuffle kernel, 64 experts, zero SMEM.
# Precompiled during Docker build → .so cached by torch.
# If not found, JIT from .cu source. PyTorch last resort.
_moe_topk_ext = None
_moe_topk_init_done = False

_ix_bridge_mod = None
_ix_bridge_init_done = False

def _init_ix_bridge():
    """Try to load ix_bridge which calls ixformer::infer::topk_softmax() via C++ pybind."""
    global _ix_bridge_mod, _ix_bridge_init_done
    _ix_bridge_init_done = True
    try:
        from ex_engine.python.ix_bridge import is_available, topk_softmax as _ix_ts
        if is_available():
            _ix_bridge_mod = True
            logger.info("topk_softmax: ix_bridge → ixformer::infer::topk_softmax() LOADED")
            return
    except Exception as e:
        logger.info("topk_softmax: ix_bridge unavailable (%s)", e)
    # Also try direct import from workspace
    try:
        import sys
        for p in ['/workspace/ex_engine/python', '/workspace/ex_engine',
                  '/usr/local/corex/lib/python3/dist-packages/ex_engine/python']:
            if p not in sys.path:
                sys.path.insert(0, p)
        from ix_bridge import is_available, topk_softmax as _ix_ts
        if is_available():
            _ix_bridge_mod = True
            logger.info("topk_softmax: ix_bridge (direct) → ixformer::infer LOADED")
            return
    except Exception as e:
        logger.info("topk_softmax: ix_bridge direct import failed (%s)", e)


def _init_moe_topk():
    global _moe_topk_ext, _moe_topk_init_done
    _moe_topk_init_done = True
    # 0. Try _moe_C (CUB-based, proven on BI-V100 real hardware 2026-08-11)
    try:
        import _moe_C as ext
        if hasattr(ext, 'topk_softmax'):
            _moe_topk_ext = ext
            logger.info("topk_softmax: loaded _moe_C (CUB BlockReduce, WARP_SIZE=64)")
            return
    except ImportError:
        pass
    # 0b. Try loading from torch cache
    import glob as _glob
    for pattern in [
        "/root/.cache/torch_extensions/py310_cu102/_moe_C/_moe_C.so",
        "/root/.cache/torch_extensions/*/_moe_C/*.so",
    ]:
        for so_path in _glob.glob(pattern):
            try:
                torch.ops.load_library(so_path)
                import _moe_C as ext
                _moe_topk_ext = ext
                logger.info("topk_softmax: loaded _moe_C from %s", so_path)
                return
            except Exception:
                pass
    # 0c. Try ix_bridge (calls ixformer C++ SDK if available)
    _init_ix_bridge()
    if _ix_bridge_mod:
        return
    # 1. Try import old precompiled module (torch cache from Docker build)
    try:
        import moe_topk_softmax_v3 as ext
        _moe_topk_ext = ext
        logger.info("topk_softmax: loaded precompiled moe_topk_softmax_v3")
        return
    except ImportError:
        pass
    # 2. Try loading from known .so paths
    import glob
    so_patterns = [
        "/workspace/ex_engine/build/moe_topk_softmax_v3*.so",
        "/root/.cache/torch_extensions/*/moe_topk_softmax_v3/*.so",
        "/tmp/torch_extensions/*/moe_topk_softmax_v3/*.so",
    ]
    for pattern in so_patterns:
        for so_path in glob.glob(pattern):
            try:
                torch.ops.load_library(so_path)
                # After load_library, the pybind module should be importable
                import moe_topk_softmax_v3 as ext
                _moe_topk_ext = ext
                logger.info("topk_softmax: loaded CUDA kernel from %s", so_path)
                return
            except Exception:
                pass
    # 3. JIT compile from .cu source
    import os
    search_paths = [
        "/workspace/ex_engine/csrc/moe_topk_softmax_v3.cu",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "moe_topk_softmax_v3.cu"),
    ]
    for base in ["/usr/local/corex/lib/python3/dist-packages/vllm/model_executor/models",
                 "/usr/local/corex/lib64/python3/dist-packages/vllm/model_executor/models"]:
        search_paths.append(os.path.join(base, "moe_topk_softmax_v3.cu"))
    for cu_path in search_paths:
        if os.path.isfile(cu_path):
            try:
                from torch.utils.cpp_extension import load
                ext = load(
                    name="moe_topk_softmax_v3",
                    sources=[cu_path],
                    extra_cuda_cflags=["-O3"],
                    verbose=False,
                )
                _moe_topk_ext = ext
                logger.info("topk_softmax: JIT compiled CUDA kernel from %s", cu_path)
                return
            except Exception as e:
                logger.warning("topk_softmax: JIT compile failed (%s)", e)
                break  # Don't retry same source with different paths
    logger.warning("topk_softmax: CUDA kernel unavailable — PyTorch fallback (SLOW)")


def topk_softmax(topk_weights: torch.Tensor, topk_ids: torch.Tensor,
                 token_expert_indicies: torch.Tensor,
                 gating_output: float) -> None:
    global _moe_topk_ext, _moe_topk_init_done
    if not _moe_topk_init_done:
        _init_moe_topk()

    # Priority 0: ix_bridge → ixformer::infer::topk_softmax() (fastest, uses SDK)
    if _ix_bridge_mod:
        try:
            from ex_engine.python.ix_bridge import topk_softmax as _ix_topk
            gating = gating_output if isinstance(gating_output, torch.Tensor) else gating_output
            topk_k = topk_weights.shape[1]
            weights, ids = _ix_topk(gating, topk_k, renormalize=False)
            topk_weights.copy_(weights.to(topk_weights.dtype))
            topk_ids.copy_(ids.to(topk_ids.dtype))
            # token_expert_indicies not produced by ix_bridge, fill with topk_ids
            token_expert_indicies.copy_(ids.to(token_expert_indicies.dtype))
            return
        except Exception as e:
            logger.warning("topk_softmax ix_bridge failed (%s), trying CUDA kernel", e)

    # Priority 1: CUDA kernel (_moe_C or moe_topk_softmax_v3)
    if _moe_topk_ext is not None:
        try:
            gating = gating_output if isinstance(gating_output, torch.Tensor) else gating_output
            if hasattr(_moe_topk_ext, 'topk_softmax'):
                # _moe_C style: in-place (vllm standard API)
                _moe_topk_ext.topk_softmax(topk_weights, topk_ids,
                                           token_expert_indicies, gating.float())
            elif hasattr(_moe_topk_ext, 'moe_topk_softmax'):
                # old v3 style: returns tuple
                topk_k = topk_weights.shape[1]
                results = _moe_topk_ext.moe_topk_softmax(gating, topk_k, False)
                topk_weights.copy_(results[0].to(topk_weights.dtype))
                topk_ids.copy_(results[1].to(topk_ids.dtype))
                token_expert_indicies.copy_(results[2].to(token_expert_indicies.dtype))
            return
        except Exception as e:
            logger.warning("topk_softmax CUDA kernel failed (%s), falling back to PyTorch", e)
            _moe_topk_ext = None  # disable permanently on failure

    # Priority 2: PyTorch fallback (always works)
    if isinstance(gating_output, torch.Tensor):
        probs = torch.softmax(gating_output.float(), dim=-1)
    else:
        probs = torch.softmax(gating_output, dim=-1)
    topk = topk_weights.shape[1]
    tw, ti = torch.topk(probs, topk, dim=-1)
    topk_weights.copy_(tw.to(topk_weights.dtype))
    topk_ids.copy_(ti.to(topk_ids.dtype))
    token_expert_indicies.copy_(
        torch.arange(topk, device=topk_ids.device, dtype=topk_ids.dtype)
        .unsqueeze(0).expand_as(topk_ids))


if supports_moe_ops and hasattr(torch.ops._moe_C, "marlin_gemm_moe"):

    @register_fake("_moe_C::marlin_gemm_moe")
    def marlin_gemm_moe_fake(a: torch.Tensor, b_q_weights: torch.Tensor,
                             sorted_ids: torch.Tensor,
                             topk_weights: torch.Tensor,
                             topk_ids: torch.Tensor, b_scales: torch.Tensor,
                             b_zero_points: torch.Tensor, g_idx: torch.Tensor,
                             perm: torch.Tensor, workspace: torch.Tensor,
                             b_q_type: ScalarType, size_m: int, size_n: int,
                             size_k: int, is_k_full: bool, num_experts: int,
                             topk: int, moe_block_size: int,
                             replicate_input: bool,
                             apply_weights: bool) -> torch.Tensor:
        return torch.empty((size_m, topk, size_n),
                           dtype=a.dtype,
                           device=a.device)


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
) -> None:
    slot_mapping = slot_mapping.to(torch.int32) 
    ixf_F.vllm_cache_ops_reshape_and_cache(key, value, key_cache,
                                             value_cache, slot_mapping)


def reshape_and_cache_flash(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: float,
    v_scale: float,
) -> None:
    ixf_F.reshape_and_cache_flash(key, value, key_cache,
                                    value_cache, slot_mapping,
                                    kv_cache_dtype, k_scale,
                                    v_scale)

def reshape_and_cache_flashinfer(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: float, # for fp8
    v_scale: float, # for fp8
    kv_cache_format: str = "NHD",
    key_cache_scales: torch.Tensor = None, # for int8
    value_cache_scales: torch.Tensor = None, # for int8
) -> None:
    ixf_F.paged_attention_cache_appended(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_format,
        key_cache_scales,
        value_cache_scales,
    )

def copy_blocks(key_caches: List[torch.Tensor],
                value_caches: List[torch.Tensor],
                block_mapping: torch.Tensor) -> None:
    # ixformer vllm_copy_cache expects dict {src_block: [dst_blocks...]}
    # vllm 0.6.3 passes a Tensor of shape [N, 2] with (src, dst) pairs
    if isinstance(block_mapping, torch.Tensor):
        mapping_dict = {}
        bm = block_mapping.cpu()
        for i in range(bm.shape[0]):
            src = int(bm[i, 0])
            dst = int(bm[i, 1])
            if src not in mapping_dict:
                mapping_dict[src] = []
            mapping_dict[src].append(dst)
        ixf_F.vllm_copy_cache(key_caches, value_caches, mapping_dict)
    else:
        ixf_F.vllm_copy_cache(key_caches, value_caches, block_mapping)


def swap_blocks(src: torch.Tensor, dst: torch.Tensor,
                block_mapping: torch.Tensor) -> None:
    # Same issue: ixformer expects dict, vllm passes Tensor
    if isinstance(block_mapping, torch.Tensor):
        mapping_dict = {}
        bm = block_mapping.cpu()
        for i in range(bm.shape[0]):
            s = int(bm[i, 0])
            d = int(bm[i, 1])
            mapping_dict[s] = d
        ixf_F.vllm_swap_blocks(src, dst, mapping_dict)
    else:
        ixf_F.vllm_swap_blocks(src, dst, block_mapping)


def convert_fp8(output: torch.Tensor,
                input: torch.Tensor,
                scale: float = 1.0,
                kv_dtype: str = "fp8") -> None:
    raise NotImplementedError()


def get_device_attribute(attribute: int, device: int) -> int:
    raise NotImplementedError()


def get_max_shared_memory_per_block_device_attribute(device: int) -> int:
    # BI-V100 SMEM = 49152 bytes (48KB), confirmed via ixsmi
    # Was incorrectly hardcoded to 32KB (32768), limiting Triton tile sizes
    # and potentially constraining ixformer internal SMEM allocation.
    return 49152


# custom ar
def init_custom_ar(meta: torch.Tensor, rank_data: torch.Tensor,
                   handles: List[str], offsets: List[int], rank: int,
                   full_nvlink: bool) -> int:
    raise NotImplementedError()


def should_custom_ar(inp: torch.Tensor, max_size: int, world_size: int,
                     full_nvlink: bool) -> bool:
    raise NotImplementedError()


def all_reduce_reg(fa: int, inp: torch.Tensor, out: torch.Tensor) -> None:
    raise NotImplementedError()


def all_reduce_unreg(fa: int, inp: torch.Tensor, reg_buffer: torch.Tensor,
                     out: torch.Tensor) -> None:
    raise NotImplementedError()


def dispose(fa: int) -> None:
    raise NotImplementedError()


def meta_size() -> int:
    raise NotImplementedError()


def register_buffer(fa: int, t: torch.Tensor, handles: List[str],
                    offsets: List[int]) -> None:
    raise NotImplementedError()


def get_graph_buffer_ipc_meta(fa: int) -> Tuple[List[str], List[int]]:
    raise NotImplementedError()


def register_graph_buffers(fa: int, handles: List[str],
                           offsets: List[List[int]]) -> None:
    raise NotImplementedError()


# Add our new features here..

# broadcast
class Async_helper():
    # For now, the comm and the other kernels are in the same stream, so we can remove the stream wait..
    def wait(self,):
        return True


def broadcast(tensor, src=0, group=None, async_op=False):
    cdist.broadcast(tensor,src,group,async_op=True)
    if async_op:
        return Async_helper()
    else:
        pass

# w8a16
def linear_w8a16(x: torch.Tensor, qweight: torch.Tensor, scales:torch.Tensor,
                 group_size: int = -1, format: str = "TN")-> torch.Tensor:
    return ixf_F.w8a16(x, qweight, scales, format="TN", group_size=group_size)


## lora sgmv / bgmv
def sbgmv_expand(x: torch.Tensor,
                w_t_all: torch.Tensor,
                y: torch.Tensor,
                b_seq_start_loc: torch.Tensor = None,
                seq_len_tensor: torch.Tensor = None,
                lora_indices_tensor: torch.Tensor = None,
                batches: int = -1,
                max_seq_length: int = -1,
                token_nums: int = -1,
                add_input=True,
                ):
    '''
    x: inputs
    w_t_all: lora weight
    y: output

    y += x@wt_t_all
    '''
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert w_t_all.dtype in [
        torch.float16,
        torch.bfloat16,
    ]

    assert x.is_contiguous()
    # assert y.is_contiguous()
    if x.dtype == torch.float:
        x = x.to(w_t_all.dtype)

    if w_t_all.ndim == 4:  # shape:(lora_num,1,size,rank)
        assert w_t_all.size(1) == 1
        w_t_all = w_t_all.squeeze(dim=1)
    else:
        assert w_t_all.ndim == 3  # shape:(lora_num,size,rank)
    assert w_t_all.is_contiguous()

    assert add_input == True

    lora_indices = lora_indices_tensor.cpu().tolist()
    lora_num = w_t_all.shape[0]

    ## 单一lora model, 且所有request均使用lora
    if lora_num == 1 and all(x == lora_indices[0] for x in lora_indices):
        if lora_indices[0] != -1:
            w_t = w_t_all[0]
            y += torch.matmul(x, w_t.t())
    ## 多个lora model
    else:
        ## prefill
        if batches != -1:
            for i, lora_id, start, seq_len in zip(range(batches), lora_indices, b_seq_start_loc, seq_len_tensor):
                if lora_id != -1:
                    xi = x[start: start+seq_len]
                    w_t = w_t_all[lora_id]
                    y[start:start+seq_len] += (xi @ w_t.t())
        ## decode
        else:
            batches = x.shape[0]
            for i, lora_id in zip(range(batches), lora_indices):
                if lora_id != -1:
                    xi = x[i].unsqueeze(0)
                    w_t = w_t_all[lora_id]
                    y[i] += (xi @ w_t.t()).squeeze(0)

    return y


def sbgmv_shrink(x: torch.Tensor,
                w_t_all: torch.Tensor,
                y: torch.Tensor,
                b_seq_start_loc: torch.Tensor = None,
                seq_len_tensor: torch.Tensor = None,
                lora_indices_tensor: torch.Tensor = None,
                batches: int = -1,
                max_seq_length: int = -1,
                token_nums: int = -1,
                scale: float = 1.0,):
    """
    xx: inputs
    w_t_all: lora weight
    y: output
    scale: float

    y = x@w_t_all * scale
    """
    assert x.dtype == w_t_all.dtype
    assert x.dtype in [torch.float16, torch.bfloat16]
    assert x.is_contiguous()
    assert y.is_contiguous()

    if w_t_all.ndim == 4:  # shape:(lora_num,1,size,rank)
        assert w_t_all.size(1) == 1
        w_t_all = w_t_all.squeeze(dim=1)
    else:
        assert w_t_all.ndim == 3  # shape:(lora_num,size,rank)
    assert w_t_all.is_contiguous()
    
    lora_num = w_t_all.shape[0]
    lora_indices = lora_indices_tensor.cpu().tolist()

    ## 单一lora model, 且所有request均使用lora
    if lora_num == 1 and all(x == lora_indices[0] for x in lora_indices):
        if lora_indices[0] != -1:
            w_t = w_t_all[0]
            y = torch.matmul(x, w_t.t()) * scale
    ## 多个lora model
    else:
        ## prefill
        if batches != -1:
            for i, lora_id, start, seq_len in zip(range(batches), lora_indices, b_seq_start_loc, seq_len_tensor):
                if lora_id != -1:
                    xi = x[start: start+seq_len]
                    w_t = w_t_all[lora_id]
                    y[start:start+seq_len] = (xi @ w_t.t())* scale
        ## decode
        else:
            batches = x.shape[0]
            for i, lora_id in zip(range(batches), lora_indices):
                if lora_id != -1:
                    xi = x[i].unsqueeze(0)
                    w_t = w_t_all[lora_id]
                    y[i] = (xi @ w_t.t()).squeeze(0) * scale

    return y

# temporary fix for https://github.com/vllm-project/vllm/issues/5456
# TODO: remove this in v0.6.0
names_and_values = globals()
names_and_values_to_update = {}
# prepare variables to avoid dict size change during iteration
k, v, arg = None, None, None
fn_type = type(lambda x: x)
for k, v in names_and_values.items():
    # find functions that are defined in this file and have torch.Tensor
    # in their annotations. `arg == "torch.Tensor"` is used to handle
    # the case when users use `import __annotations__` to turn type
    # hints into strings.
    if isinstance(v, fn_type) \
        and v.__code__.co_filename == __file__ \
        and any(arg is torch.Tensor or arg == "torch.Tensor"
                for arg in v.__annotations__.values()):
        names_and_values_to_update[k] = hint_on_error(v)

names_and_values.update(names_and_values_to_update)
del names_and_values_to_update, names_and_values, v, k, fn_type