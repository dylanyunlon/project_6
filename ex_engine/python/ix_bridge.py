"""
ix_bridge.py — Load ix_moe_bridge.so and expose ixformer::infer functions to Python.

LOAD CHAIN:
  1. Try precompiled ix_moe_bridge.so (from Docker build)
  2. Try JIT compile ix_moe_bridge.cpp (fallback)
  3. If both fail → functions return None (caller must handle)

USAGE:
  from ex_engine.python.ix_bridge import topk_softmax, moe_group_gemm, ...
  
  if topk_softmax is not None:
      topk_softmax(weights, ids, indices, gating)
  else:
      # fallback to Python implementation
"""
import os
import sys
import glob
import logging
import importlib

logger = logging.getLogger("ex_engine.ix_bridge")

_bridge = None
_loaded = False


def _find_so():
    """Find precompiled ix_moe_bridge*.so."""
    search_dirs = [
        os.path.join(os.path.dirname(__file__), ".."),
        os.path.join(os.path.dirname(__file__), "..", "build"),
        "/workspace/ex_engine/build",
        "/workspace/ex_engine",
    ]
    # Also check site-packages
    try:
        import ex_engine
        search_dirs.append(os.path.dirname(ex_engine.__file__))
        search_dirs.append(os.path.join(os.path.dirname(ex_engine.__file__), "build"))
    except ImportError:
        pass
    
    for d in search_dirs:
        for so in glob.glob(os.path.join(d, "ix_moe_bridge*.so")):
            return so
    return None


def _load():
    """Load the bridge module."""
    global _bridge, _loaded
    if _loaded:
        return _bridge
    _loaded = True
    
    # Method 1: Try precompiled .so
    so_path = _find_so()
    if so_path:
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("ix_moe_bridge", so_path)
            _bridge = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_bridge)
            logger.info(f"Loaded ix_moe_bridge from: {so_path}")
            funcs = [x for x in dir(_bridge) if not x.startswith('_')]
            logger.info(f"Available functions: {funcs}")
            return _bridge
        except Exception as e:
            logger.warning(f"Failed to load {so_path}: {e}")
    
    # Method 2: Try JIT compile
    try:
        import torch
        from torch.utils.cpp_extension import load
        
        cpp_path = None
        for p in [
            os.path.join(os.path.dirname(__file__), "..", "csrc", "ix_moe_bridge.cpp"),
            "/workspace/ex_engine/csrc/ix_moe_bridge.cpp",
        ]:
            if os.path.exists(p):
                cpp_path = p
                break
        
        if cpp_path is None:
            logger.warning("ix_moe_bridge.cpp not found for JIT compile")
            return None
        
        # Find libixformer.so
        ldflags = ["-lixformer"]
        for d in [
            "/usr/local/corex/lib64/python3/dist-packages/ixformer",
            "/usr/local/corex/lib/python3/dist-packages/ixformer",
        ]:
            if os.path.exists(os.path.join(d, "libixformer.so")):
                ldflags.insert(0, f"-L{d}")
                ldflags.insert(1, f"-Wl,-rpath,{d}")
                break
        
        _bridge = load(
            name="ix_moe_bridge",
            sources=[cpp_path],
            extra_cflags=["-O2", "-std=c++17"],
            extra_ldflags=ldflags,
            verbose=False,
        )
        logger.info(f"JIT compiled ix_moe_bridge from: {cpp_path}")
        return _bridge
        
    except Exception as e:
        logger.warning(f"JIT compile failed: {e}")
    
    return None


def _get_fn(name):
    """Get a function from the bridge, or None."""
    mod = _load()
    if mod is None:
        return None
    return getattr(mod, name, None)


# ============================================================================
# Public API — each is None if bridge not available
# ============================================================================

def topk_softmax(topk_weights, topk_ids, token_expert_indices, gating_output):
    fn = _get_fn("topk_softmax")
    if fn is None:
        raise RuntimeError("ix_moe_bridge: topk_softmax not available")
    fn(topk_weights, topk_ids, token_expert_indices, gating_output)


def moe_gen_idx(expert_id, expert_num):
    fn = _get_fn("moe_gen_idx")
    if fn is None:
        raise RuntimeError("ix_moe_bridge: moe_gen_idx not available")
    return fn(expert_id, expert_num)


def moe_expand_input(input_tensor, gather_index, combine_idx, topk):
    fn = _get_fn("moe_expand_input")
    if fn is None:
        raise RuntimeError("ix_moe_bridge: moe_expand_input not available")
    return fn(input_tensor, gather_index, combine_idx, topk)


def moe_group_gemm(output, inputs, weights, tokens_per_experts, output_n):
    fn = _get_fn("moe_group_gemm")
    if fn is None:
        raise RuntimeError("ix_moe_bridge: moe_group_gemm not available")
    fn(output, inputs, weights, tokens_per_experts, output_n)


def silu_and_mul(input_tensor):
    fn = _get_fn("silu_and_mul")
    if fn is None:
        raise RuntimeError("ix_moe_bridge: silu_and_mul not available")
    return fn(input_tensor)


def moe_combine_result(input_tensor, weight):
    fn = _get_fn("moe_combine_result")
    if fn is None:
        raise RuntimeError("ix_moe_bridge: moe_combine_result not available")
    return fn(input_tensor, weight)


def paged_attention(out, query, key_cache, value_cache, num_kv_heads, scale,
                    block_tables, context_lens, block_size, max_context_len):
    fn = _get_fn("paged_attention")
    if fn is None:
        raise RuntimeError("ix_moe_bridge: paged_attention not available")
    return fn(out, query, key_cache, value_cache, num_kv_heads, scale,
              block_tables, context_lens, block_size, max_context_len)


def rms_norm(output, input_tensor, weight, eps):
    fn = _get_fn("rms_norm")
    if fn is None:
        raise RuntimeError("ix_moe_bridge: rms_norm not available")
    fn(output, input_tensor, weight, eps)


def linear(input_tensor, weight):
    fn = _get_fn("linear")
    if fn is None:
        raise RuntimeError("ix_moe_bridge: linear not available")
    return fn(input_tensor, weight)


def reshape_and_cache(key, value, key_cache, value_cache, slot_mapping):
    fn = _get_fn("reshape_and_cache")
    if fn is None:
        raise RuntimeError("ix_moe_bridge: reshape_and_cache not available")
    fn(key, value, key_cache, value_cache, slot_mapping)


def rotary_embedding(positions, query, key, head_size, cos_sin_cache):
    fn = _get_fn("rotary_embedding")
    if fn is None:
        raise RuntimeError("ix_moe_bridge: rotary_embedding not available")
    fn(positions, query, key, head_size, cos_sin_cache)


# Convenience: check if bridge is available
def is_available():
    return _load() is not None
