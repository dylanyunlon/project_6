"""
EngineX Operator Registry — CCCL-style policy_selector for BI-V100

Maps each operator to its best available implementation:
  Tier 1: Native .so via ctypes/dlopen (libcorex_gdn.so, libixattn.so, etc.)
  Tier 2: ixformer Python ops (ixf_F.silu_and_mul, etc.)
  Tier 3: PyTorch fallback (torch.nn.functional, manual loops)

Modeled after CCCL dispatch_reduce.cuh → PolicySelector → tuning_reduce.cuh chain:
  CCCL picks {threads, items, algorithm} per SM arch
  We pick {backend, tile_size, num_warps} per BI-V100 hardware constraints
"""

import ctypes
import logging
import os
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger("enginex.registry")


class Backend(IntEnum):
    """Dispatch tiers — same ordering as CCCL's dispatch priority."""
    NATIVE_SO = 0      # dlopen .so — fastest, hardware-fused
    IXFORMER = 1       # ixformer Python ops — vendor-provided
    PYTORCH = 2        # torch fallback — slowest but always works


@dataclass
class HardwareProfile:
    """BI-V100 hardware constants (from HARDWARE_PROBE_20260808.md)."""
    sm_count: int = 16
    smem_per_sm: int = 49152        # 48KB confirmed
    max_threads_per_block: int = 1024
    warp_size: int = 32
    mem_bandwidth_gbps: float = 900.0
    per_sm_bandwidth_gbps: float = 56.25  # 900/16
    compute_capability: str = "bi_v100"
    cuda_version: str = "10.2"
    driver_version: str = "3.2.1"


@dataclass
class OperatorImpl:
    """A single implementation of an operator."""
    name: str
    backend: Backend
    fn: Optional[Callable] = None
    so_path: Optional[str] = None
    available: bool = False
    load_error: Optional[str] = None


@dataclass
class OperatorEntry:
    """
    One logical operator with multiple implementations.
    Mirrors CCCL's policy_selector: each entry has a chain of candidates
    sorted by priority. At dispatch time, we pick the first available.
    """
    op_name: str
    impls: List[OperatorImpl] = field(default_factory=list)
    active: Optional[OperatorImpl] = None

    def select_best(self) -> Optional[OperatorImpl]:
        """Pick first available impl (lowest Backend enum = highest priority)."""
        for impl in sorted(self.impls, key=lambda x: x.backend):
            if impl.available:
                self.active = impl
                return impl
        return None


# ---------------------------------------------------------------------------
# .so probe paths — where to look for native kernels
# ---------------------------------------------------------------------------
_SO_SEARCH_PATHS = [
    "/usr/local/corex/lib64",
    "/usr/local/corex/lib",
    "/workspace/enginex/lib",
    "/home/claude/project_6/enginex/lib",
]


def _probe_so(name: str) -> Optional[str]:
    """Try to find a .so file by name in known search paths."""
    for d in _SO_SEARCH_PATHS:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def _try_dlopen(path: str) -> Tuple[Optional[ctypes.CDLL], Optional[str]]:
    """Attempt dlopen, return (handle, error_or_None)."""
    try:
        handle = ctypes.CDLL(path)
        return handle, None
    except OSError as e:
        return None, str(e)


def _try_import_ixformer():
    """Probe ixformer availability."""
    try:
        import ixformer.functions as ixf_F
        return ixf_F, None
    except ImportError as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# The global registry
# ---------------------------------------------------------------------------
class OperatorRegistry:
    """
    Central operator registry — the EngineX equivalent of CCCL's
    DeviceReducePolicy / DeviceScanPolicy / DeviceTopkPolicy system.

    Usage:
        reg = get_registry()
        moe_topk = reg.get_op("moe_topk_softmax")
        if moe_topk:
            moe_topk(topk_weights, topk_ids, token_expert_indices, gating_output)
    """

    def __init__(self):
        self.hw = HardwareProfile()
        self.ops: Dict[str, OperatorEntry] = {}
        self.ixf_F = None
        self._probed = False

    def probe(self):
        """
        One-time hardware + library probe.
        Called automatically on first get_op().
        """
        if self._probed:
            return
        self._probed = True

        logger.info(f"EngineX probe: SM={self.hw.sm_count}, "
                    f"SMEM={self.hw.smem_per_sm}, "
                    f"BW={self.hw.mem_bandwidth_gbps} GB/s")

        # Probe ixformer
        self.ixf_F, ixf_err = _try_import_ixformer()
        if self.ixf_F:
            logger.info("EngineX: ixformer.functions available")
        else:
            logger.warning(f"EngineX: ixformer not available: {ixf_err}")

        # Register all operators
        self._register_gdn_ops()
        self._register_moe_ops()
        self._register_fa2_ops()
        self._register_activation_ops()
        self._register_norm_ops()
        self._register_attention_ops()
        self._register_cache_ops()
        self._register_sampling_ops()

        # Select best impl for each
        for name, entry in self.ops.items():
            best = entry.select_best()
            if best:
                logger.info(f"EngineX [{name}]: using {best.backend.name} "
                            f"({best.name})")
            else:
                logger.error(f"EngineX [{name}]: NO IMPLEMENTATION AVAILABLE")

    def _add_op(self, op_name: str, impl: OperatorImpl):
        if op_name not in self.ops:
            self.ops[op_name] = OperatorEntry(op_name=op_name)
        self.ops[op_name].impls.append(impl)

    def get_op(self, name: str) -> Optional[Callable]:
        """Get the best available implementation for an operator."""
        self.probe()
        entry = self.ops.get(name)
        if entry and entry.active and entry.active.fn:
            return entry.active.fn
        return None

    def get_backend(self, name: str) -> Optional[Backend]:
        """Get which backend is active for an operator."""
        self.probe()
        entry = self.ops.get(name)
        if entry and entry.active:
            return entry.active.backend
        return None

    # ------------------------------------------------------------------
    # GDN (GatedDeltaNet) operators
    # From log: corex_gdn.py:56 loads /usr/local/corex/lib64/libcorex_gdn.so
    # ------------------------------------------------------------------
    def _register_gdn_ops(self):
        # Tier 1: native .so
        so_path = _probe_so("libcorex_gdn.so")
        if so_path:
            handle, err = _try_dlopen(so_path)
            from enginex.ops.gdn import make_native_gdn_decode, make_native_gdn_prefill
            self._add_op("gdn_decode", OperatorImpl(
                name="corex_gdn_decode", backend=Backend.NATIVE_SO,
                fn=make_native_gdn_decode(handle) if handle else None,
                so_path=so_path, available=handle is not None,
                load_error=err))
            self._add_op("gdn_prefill", OperatorImpl(
                name="corex_gdn_prefill", backend=Backend.NATIVE_SO,
                fn=make_native_gdn_prefill(handle) if handle else None,
                so_path=so_path, available=handle is not None,
                load_error=err))
        # Tier 2: our FlashQLA SM70 kernel (.so compiled from gdn_forward.cu)
        flash_so = _probe_so("flash_qla_sm70_gdn_strided.so")
        if not flash_so:
            # Check build directory
            for d in ["/workspace/qwen3_6_scripts/flash_qla_sm70/build",
                      "/home/claude/project_6/qwen3_6_scripts/flash_qla_sm70/build"]:
                candidate = os.path.join(d, "flash_qla_sm70_gdn_strided.so")
                if os.path.isfile(candidate):
                    flash_so = candidate
                    break
        if flash_so:
            from enginex.ops.gdn import make_flashqla_gdn_prefill
            handle, err = _try_dlopen(flash_so)
            self._add_op("gdn_prefill", OperatorImpl(
                name="flashqla_sm70_prefill", backend=Backend.NATIVE_SO,
                fn=make_flashqla_gdn_prefill(flash_so) if handle else None,
                so_path=flash_so, available=handle is not None,
                load_error=err))

        # Tier 3: PyTorch fallback
        from enginex.ops.gdn import gdn_decode_pytorch, gdn_prefill_pytorch
        self._add_op("gdn_decode", OperatorImpl(
            name="pytorch_gdn_decode", backend=Backend.PYTORCH,
            fn=gdn_decode_pytorch, available=True))
        self._add_op("gdn_prefill", OperatorImpl(
            name="pytorch_gdn_prefill", backend=Backend.PYTORCH,
            fn=gdn_prefill_pytorch, available=True))

    # ------------------------------------------------------------------
    # MoE operators
    # From log: vllm_moe_topk_softmax missing from ixformer.functions
    #           corex_moe.py:339 uses expert-grouped-wmma kernel
    # ------------------------------------------------------------------
    def _register_moe_ops(self):
        # Tier 2: ixformer (but topk_softmax is KNOWN MISSING)
        if self.ixf_F:
            has_topk = hasattr(self.ixf_F, 'vllm_moe_topk_softmax')
            has_fused = hasattr(self.ixf_F, 'vllm_invoke_fused_moe_kernel')
            has_align = hasattr(self.ixf_F, 'vllm_moe_align_block_size')

            if has_topk:
                self._add_op("moe_topk_softmax", OperatorImpl(
                    name="ixf_moe_topk_softmax", backend=Backend.IXFORMER,
                    fn=self.ixf_F.vllm_moe_topk_softmax, available=True))
            if has_fused:
                self._add_op("moe_fused_kernel", OperatorImpl(
                    name="ixf_fused_moe", backend=Backend.IXFORMER,
                    fn=self.ixf_F.vllm_invoke_fused_moe_kernel, available=True))
            if has_align:
                self._add_op("moe_align_block_size", OperatorImpl(
                    name="ixf_moe_align", backend=Backend.IXFORMER,
                    fn=self.ixf_F.vllm_moe_align_block_size, available=True))

        # Tier 3: PyTorch fallback (THE FIX for the topk_softmax crash)
        from enginex.ops.moe import (moe_topk_softmax_pytorch,
                                      moe_fused_kernel_pytorch,
                                      moe_align_block_size_pytorch)
        self._add_op("moe_topk_softmax", OperatorImpl(
            name="pytorch_moe_topk_softmax", backend=Backend.PYTORCH,
            fn=moe_topk_softmax_pytorch, available=True))
        self._add_op("moe_fused_kernel", OperatorImpl(
            name="pytorch_fused_moe", backend=Backend.PYTORCH,
            fn=moe_fused_kernel_pytorch, available=True))
        self._add_op("moe_align_block_size", OperatorImpl(
            name="pytorch_moe_align", backend=Backend.PYTORCH,
            fn=moe_align_block_size_pytorch, available=True))

    # ------------------------------------------------------------------
    # FA2 (Flash Attention 2) operators
    # From log: corex_fa2.py:333 "CoreX FA2 packed prefill" B=2 Hq=4 Hkv=1 D=256
    #           corex_fa2.py:507 "CoreX paged FA2 chunked prefill"
    # ------------------------------------------------------------------
    def _register_fa2_ops(self):
        # Tier 2: ixformer flash_attn
        if self.ixf_F:
            import ixformer
            has_fa = hasattr(ixformer, 'flash_attn_varlen_func')
            has_fa_pad = hasattr(ixformer, 'flash_attn_func')
            if has_fa:
                self._add_op("fa2_varlen", OperatorImpl(
                    name="ixf_flash_attn_varlen", backend=Backend.IXFORMER,
                    fn=ixformer.flash_attn_varlen_func, available=True))
            if has_fa_pad:
                self._add_op("fa2_padded", OperatorImpl(
                    name="ixf_flash_attn_padded", backend=Backend.IXFORMER,
                    fn=ixformer.flash_attn_func, available=True))

        # Tier 2: libixattn.so (confirmed present in hardware probe)
        ixattn_so = _probe_so("libixattn.so")
        if ixattn_so:
            handle, err = _try_dlopen(ixattn_so)
            self._add_op("fa2_native", OperatorImpl(
                name="libixattn", backend=Backend.NATIVE_SO,
                so_path=ixattn_so, available=handle is not None,
                load_error=err))

        # Tier 3: xformers SDPA fallback (what we currently use)
        from enginex.ops.attention import fa2_xformers_fallback
        self._add_op("fa2_varlen", OperatorImpl(
            name="xformers_sdpa_fallback", backend=Backend.PYTORCH,
            fn=fa2_xformers_fallback, available=True))
        self._add_op("fa2_padded", OperatorImpl(
            name="xformers_sdpa_fallback", backend=Backend.PYTORCH,
            fn=fa2_xformers_fallback, available=True))

    # ------------------------------------------------------------------
    # Activation ops (silu_and_mul, gelu, etc.)
    # These work via ixformer — confirmed in hardware probe
    # ------------------------------------------------------------------
    def _register_activation_ops(self):
        if self.ixf_F:
            for op_name, ixf_name in [
                ("silu_and_mul", "silu_and_mul"),
                ("gelu_and_mul", "gelu_and_mul"),
                ("gelu_tanh_and_mul", "gelu_tanh_and_mul"),
            ]:
                fn = getattr(self.ixf_F, ixf_name, None)
                if fn:
                    self._add_op(op_name, OperatorImpl(
                        name=f"ixf_{ixf_name}", backend=Backend.IXFORMER,
                        fn=fn, available=True))

        # PyTorch fallbacks
        from enginex.ops.activations import (silu_and_mul_pytorch,
                                              gelu_and_mul_pytorch,
                                              gelu_tanh_and_mul_pytorch)
        self._add_op("silu_and_mul", OperatorImpl(
            name="pytorch_silu_and_mul", backend=Backend.PYTORCH,
            fn=silu_and_mul_pytorch, available=True))
        self._add_op("gelu_and_mul", OperatorImpl(
            name="pytorch_gelu_and_mul", backend=Backend.PYTORCH,
            fn=gelu_and_mul_pytorch, available=True))
        self._add_op("gelu_tanh_and_mul", OperatorImpl(
            name="pytorch_gelu_tanh_and_mul", backend=Backend.PYTORCH,
            fn=gelu_tanh_and_mul_pytorch, available=True))

    # ------------------------------------------------------------------
    # Norm ops (rms_norm, fused_add_rms_norm)
    # ------------------------------------------------------------------
    def _register_norm_ops(self):
        if self.ixf_F:
            for op_name, ixf_name in [
                ("rms_norm", "rms_norm"),
                ("fused_add_rms_norm", "fused_add_rms_norm"),
            ]:
                fn = getattr(self.ixf_F, ixf_name, None)
                if fn:
                    self._add_op(op_name, OperatorImpl(
                        name=f"ixf_{ixf_name}", backend=Backend.IXFORMER,
                        fn=fn, available=True))

        from enginex.ops.norm import rms_norm_pytorch, fused_add_rms_norm_pytorch
        self._add_op("rms_norm", OperatorImpl(
            name="pytorch_rms_norm", backend=Backend.PYTORCH,
            fn=rms_norm_pytorch, available=True))
        self._add_op("fused_add_rms_norm", OperatorImpl(
            name="pytorch_fused_add_rms_norm", backend=Backend.PYTORCH,
            fn=fused_add_rms_norm_pytorch, available=True))

    # ------------------------------------------------------------------
    # Paged attention ops
    # ------------------------------------------------------------------
    def _register_attention_ops(self):
        if self.ixf_F:
            fn_v1 = getattr(self.ixf_F,
                            'vllm_single_query_cached_kv_attention', None)
            fn_v2 = getattr(self.ixf_F,
                            'vllm_single_query_cached_kv_attention_v2', None)
            if fn_v1:
                self._add_op("paged_attention_v1", OperatorImpl(
                    name="ixf_paged_attn_v1", backend=Backend.IXFORMER,
                    fn=fn_v1, available=True))
            if fn_v2:
                self._add_op("paged_attention_v2", OperatorImpl(
                    name="ixf_paged_attn_v2", backend=Backend.IXFORMER,
                    fn=fn_v2, available=True))

        from enginex.ops.attention import (paged_attention_v1_pytorch,
                                            paged_attention_v2_pytorch)
        self._add_op("paged_attention_v1", OperatorImpl(
            name="pytorch_paged_attn_v1", backend=Backend.PYTORCH,
            fn=paged_attention_v1_pytorch, available=True))
        self._add_op("paged_attention_v2", OperatorImpl(
            name="pytorch_paged_attn_v2", backend=Backend.PYTORCH,
            fn=paged_attention_v2_pytorch, available=True))

    # ------------------------------------------------------------------
    # Cache ops (reshape_and_cache, copy_blocks, swap_blocks)
    # ------------------------------------------------------------------
    def _register_cache_ops(self):
        if self.ixf_F:
            for op_name, ixf_name in [
                ("reshape_and_cache", "vllm_cache_ops_reshape_and_cache"),
                ("copy_blocks", "copy_blocks"),
                ("swap_blocks", "swap_blocks"),
            ]:
                fn = getattr(self.ixf_F, ixf_name, None)
                if fn:
                    self._add_op(op_name, OperatorImpl(
                        name=f"ixf_{ixf_name}", backend=Backend.IXFORMER,
                        fn=fn, available=True))

        from enginex.ops.cache import (reshape_and_cache_pytorch,
                                        copy_blocks_pytorch,
                                        swap_blocks_pytorch)
        self._add_op("reshape_and_cache", OperatorImpl(
            name="pytorch_reshape_cache", backend=Backend.PYTORCH,
            fn=reshape_and_cache_pytorch, available=True))
        self._add_op("copy_blocks", OperatorImpl(
            name="pytorch_copy_blocks", backend=Backend.PYTORCH,
            fn=copy_blocks_pytorch, available=True))
        self._add_op("swap_blocks", OperatorImpl(
            name="pytorch_swap_blocks", backend=Backend.PYTORCH,
            fn=swap_blocks_pytorch, available=True))

    # ------------------------------------------------------------------
    # Sampling ops (rotary_embedding, topk)
    # ------------------------------------------------------------------
    def _register_sampling_ops(self):
        if self.ixf_F:
            fn = getattr(self.ixf_F, 'vllm_rotary_embedding_neox', None)
            if fn:
                self._add_op("rotary_embedding", OperatorImpl(
                    name="ixf_rotary", backend=Backend.IXFORMER,
                    fn=fn, available=True))

        from enginex.ops.sampling import rotary_embedding_pytorch
        self._add_op("rotary_embedding", OperatorImpl(
            name="pytorch_rotary", backend=Backend.PYTORCH,
            fn=rotary_embedding_pytorch, available=True))

    def summary(self) -> str:
        """Print a summary of all operators and their active backends."""
        self.probe()
        lines = ["EngineX Operator Registry Summary",
                 "=" * 50]
        for name, entry in sorted(self.ops.items()):
            active = entry.active
            if active:
                lines.append(
                    f"  {name:30s} → {active.backend.name:12s} ({active.name})")
            else:
                lines.append(f"  {name:30s} → MISSING")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_global_registry: Optional[OperatorRegistry] = None


def get_registry() -> OperatorRegistry:
    global _global_registry
    if _global_registry is None:
        _global_registry = OperatorRegistry()
    return _global_registry
