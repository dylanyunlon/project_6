"""ex_topk_bridge.py — ctypes bridge for ex_factor_0.so topk_softmax

CCCL pattern: ex_registry → ex_dispatch → kernel
Python bridge: ctypes.CDLL → ex_dispatch_moe_topk_softmax()

Usage:
    from ex_engine.python.ex_topk_bridge import ex_topk_softmax
    ex_topk_softmax(topk_weights, topk_ids, token_expert_indices, gating_output)
"""
import ctypes
import os
import glob
import logging
import torch

logger = logging.getLogger("ex_topk_bridge")

_lib = None
_dispatch_fn = None


def _load():
    global _lib, _dispatch_fn
    if _dispatch_fn is not None:
        return True

    # Search for ex_factor_0.so
    search = [
        os.path.join(os.path.dirname(__file__), "..", "build"),
        "/workspace/ex_engine/build",
        os.path.join(os.path.dirname(__file__), ".."),
    ]
    # Also check vllm model path (where build.sh factor compile puts it)
    for p in ["/usr/local/corex/lib64/python3/dist-packages/vllm/model_executor/models/ex_engine",
              "/usr/local/corex/lib/python3/dist-packages/vllm/model_executor/models/ex_engine"]:
        search.append(p)

    for d in search:
        so = os.path.join(d, "ex_factor_0.so")
        if os.path.isfile(so):
            try:
                _lib_local = ctypes.CDLL(so)
                fn = _lib_local.ex_dispatch_moe_topk_softmax
                fn.restype = ctypes.c_int
                fn.argtypes = [
                    ctypes.c_void_p,  # float* topk_weights
                    ctypes.c_void_p,  # int32_t* topk_ids
                    ctypes.c_void_p,  # const float* logits
                    ctypes.c_int,     # T
                    ctypes.c_int,     # E
                    ctypes.c_int,     # top_k
                    ctypes.c_void_p,  # stream
                ]
                _lib = _lib_local
                _dispatch_fn = fn
                logger.info("ex_factor_0.so loaded from %s", so)
                return True
            except Exception as e:
                logger.warning("Failed to load %s: %s", so, e)

    return False


def ex_topk_softmax(topk_weights: torch.Tensor,
                    topk_ids: torch.Tensor,
                    token_expert_indices: torch.Tensor,
                    gating_output: torch.Tensor) -> None:
    """Drop-in replacement for _custom_ops.topk_softmax using ex_factor_0.so.

    Same interface as vllm._custom_ops.topk_softmax:
      topk_weights: (T, K) float32, output
      topk_ids: (T, K) int32, output
      token_expert_indices: (T, K) int32, output (ignored by ex kernel)
      gating_output: (T, E) float32, input
    """
    if not _load():
        raise RuntimeError("ex_factor_0.so not available")

    T, E = gating_output.shape
    K = topk_weights.shape[1]

    # Get CUDA stream
    stream = torch.cuda.current_stream().cuda_stream

    ret = _dispatch_fn(
        topk_weights.data_ptr(),
        topk_ids.data_ptr(),
        gating_output.data_ptr(),
        T, E, K,
        stream,
    )
    if ret != 0:
        raise RuntimeError(f"ex_dispatch_moe_topk_softmax returned {ret}")

    # token_expert_indices: vllm expects (T, K) with values k_idx * T + t_idx
    # ex kernel doesn't write this, fill it here
    if token_expert_indices is not None:
        T_t = torch.arange(T, device=topk_ids.device, dtype=torch.int32)
        for k in range(K):
            token_expert_indices[:, k] = k * T + T_t
