"""
EngineX Bridge — wires EngineX dispatch into vllm's _custom_ops.py

This module monkey-patches _custom_ops functions to use EngineX's
three-tier dispatch instead of direct ixformer calls.

The key fix: vllm_moe_topk_softmax is MISSING from ixformer.functions
on our BI-V100 image, causing every MoE forward pass to crash.
EngineX provides a PyTorch replacement that keeps the model running.

Usage in patch_ops.sh:
    python -c "import enginex.bridge; enginex.bridge.patch_custom_ops()"

Or at runtime startup:
    from enginex.bridge import patch_custom_ops
    patch_custom_ops()
"""

import importlib
import logging
import sys

logger = logging.getLogger("enginex.bridge")


def patch_custom_ops():
    """
    Patch vllm._custom_ops to use EngineX dispatch.

    Strategy: only patch functions that are KNOWN BROKEN.
    We do NOT touch working ixformer ops (silu_and_mul, rms_norm, etc.)
    because ixformer's implementations are faster.

    From docker log analysis, the BROKEN ops are:
      1. topk_softmax — AttributeError: no 'vllm_moe_topk_softmax'
      2. invoke_fused_moe_kernel — falls back to PyTorch on topk failure
      3. moe_align_block_size — cascading failure from topk
    """
    from enginex.dispatch.registry import get_registry

    reg = get_registry()
    reg.probe()

    logger.info("EngineX bridge: patching broken ops in _custom_ops")
    logger.info(reg.summary())

    # Only patch if the module is already imported
    custom_ops = sys.modules.get('vllm._custom_ops')
    if custom_ops is None:
        try:
            custom_ops = importlib.import_module('vllm._custom_ops')
        except ImportError:
            logger.warning("vllm._custom_ops not found — skipping bridge")
            return

    # ---- Patch 1: moe_topk_softmax (THE critical fix) ----
    moe_topk = reg.get_op("moe_topk_softmax")
    if moe_topk:
        original = getattr(custom_ops, 'topk_softmax', None)
        if original:
            # Check if the original actually works
            try:
                import ixformer.functions as ixf_F
                _ = ixf_F.vllm_moe_topk_softmax
                logger.info("EngineX: ixformer.vllm_moe_topk_softmax exists, "
                            "keeping original")
            except (ImportError, AttributeError):
                logger.info("EngineX: patching topk_softmax → EngineX "
                            f"({reg.get_backend('moe_topk_softmax').name})")
                custom_ops.topk_softmax = moe_topk

    # ---- Patch 2: moe_align_block_size ----
    moe_align = reg.get_op("moe_align_block_size")
    if moe_align:
        try:
            import ixformer.functions as ixf_F
            _ = ixf_F.vllm_moe_align_block_size
        except (ImportError, AttributeError):
            logger.info("EngineX: patching moe_align_block_size → EngineX")
            custom_ops.moe_align_block_size = moe_align

    # ---- Report status ----
    n_patched = 0
    for op_name in reg.ops:
        backend = reg.get_backend(op_name)
        if backend is not None:
            n_patched += 1
    logger.info(f"EngineX bridge: {n_patched} operators registered, "
                f"patched broken ops")


def patch_qwen3_5_moe():
    """
    Patch the MoE dispatch in qwen3_5.py to use EngineX.

    The model code tries:
      1st: corex_moe.py (not in our image)
      2nd: ixformer fused_moe (crashes on topk_softmax)
      3rd: PyTorch loop (works but slow)

    With EngineX, the topk_softmax fallback prevents the crash,
    so tier 2 (ixformer) works for the GEMM even though topk
    is handled by our PyTorch replacement.
    """
    from enginex.dispatch.registry import get_registry
    reg = get_registry()
    reg.probe()

    # The actual patching happens through _custom_ops
    # Since qwen3_5.py calls ops.topk_softmax(), which calls _custom_ops,
    # patching _custom_ops is sufficient.
    logger.info("EngineX: MoE dispatch chain patched via _custom_ops bridge")
