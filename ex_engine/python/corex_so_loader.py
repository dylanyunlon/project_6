"""corex_so_loader.py — Unified loader for all 12 prebuilt CoreX .so modules.

CCCL pattern: device_reduce policy_selector — enumerate available kernels at
init, expose a stable Python API, fall back gracefully when .so unavailable.

The 12 prebuilt .so files expose these operator families:

  GDN decode pipeline (5 .so):
    corex_gdn_causal_conv    → .causal_conv_update(conv_state, mixed_qkv, weight)
    corex_gdn_packed_decode  → .packed_decode(temporal_state, packed_qkv, b, a, A_log, dt_bias)
    corex_gdn_beta_decay     → .beta_decay(b, a, A_log, dt_bias)
    corex_gdn_qk_map         → .qk_map(q, k, num_v_heads)
    corex_gdn_gated_norm     → .apply_inverse(x, z)

  Attention pipeline (3 .so):
    corex_attn_head_rms_norm → .prepare(x, eps) + .apply_inverse(x, z)
    corex_paged_kv_gather    → .gather(key_cache, val_cache, block_tables, context_lens)
    corex_fused_paged_prefill → .forward(q, k_cache, v_cache, ...)

  KV cache transfer (1 .so):
    corex_block_major_kv_transfer → .transfer(src, dst, mapping)

  MoE pipeline (3 .so):
    corex_moe_direct_routed  → .w13(hidden, w13, expert_ids)
                              + .w2_reduce(act, w2, expert_ids, weights)
    corex_moe_weight_gather  → .gather(w13, w2, expert_ids)
    corex_moe_exact_reduce   → .serial_float(expert_out, weights)

Usage:
    from ex_engine.python.corex_so_loader import corex
    if corex.gdn_causal_conv is not None:
        out = corex.gdn_causal_conv.causal_conv_update(...)

    # Or import from vllm install root (patch_ops.sh deploys there):
    from corex_so_loader import corex
"""

import importlib.util
import logging
import os
import sys
from typing import Optional

logger = logging.getLogger("corex_so_loader")

# All 12 .so modules in load order
_SO_MANIFEST = [
    "corex_gdn_causal_conv",
    "corex_gdn_packed_decode",
    "corex_gdn_beta_decay",
    "corex_gdn_qk_map",
    "corex_gdn_gated_norm",
    "corex_attn_head_rms_norm",
    "corex_paged_kv_gather",
    "corex_fused_paged_prefill",
    "corex_block_major_kv_transfer",
    "corex_moe_direct_routed",
    "corex_moe_weight_gather",
    "corex_moe_exact_reduce",
]


def _find_so_dir() -> Optional[str]:
    """Find the directory containing prebuilt CoreX .so files.

    Search order:
      1. COREX_SO_DIR env var
      2. vllm install roots (where patch_ops.sh installs them)
      3. Bundled prebuilt directory (repo-relative)
      4. /usr/local/corex/lib64/
    """
    candidates = []

    env = os.getenv("COREX_SO_DIR")
    if env:
        candidates.append(env)

    # vllm install roots (patch_ops.sh copies .so here)
    for p in sys.path:
        if "vllm" in p or "dist-packages" in p:
            candidates.append(p)
            # Also check parent/vllm/model_executor/models/
            candidates.append(os.path.join(p, "vllm", "model_executor", "models"))

    # Repo-relative prebuilt bundle
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(here, "..", "..", "qwen3_6_scripts",
                                   "prebuilt", "corex-3.2.3-ivcore10"))
    candidates.append(os.path.join(here, "..", "..", "qwen3_6_scripts"))

    # System CoreX
    candidates.append("/usr/local/corex/lib64/")

    for d in candidates:
        d = os.path.normpath(d)
        if os.path.isdir(d):
            test_so = os.path.join(d, "corex_gdn_causal_conv.so")
            if os.path.isfile(test_so):
                return d

    return None


def _load_so(name: str, so_dir: str):
    """Load a single .so by name from so_dir via importlib."""
    so_path = os.path.join(so_dir, f"{name}.so")
    if not os.path.isfile(so_path):
        return None
    try:
        spec = importlib.util.spec_from_file_location(name, so_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        logger.warning("Failed to load %s: %s", so_path, e)
        return None


class CoreXModules:
    """Container for all loaded CoreX .so modules.

    Each attribute is either the loaded module or None.
    Attribute names drop the 'corex_' prefix for brevity.
    """

    def __init__(self):
        self._loaded = {}
        self._so_dir = None

        so_dir = _find_so_dir()
        if so_dir is None:
            logger.info("CoreX prebuilt .so directory not found — all modules disabled")
            for name in _SO_MANIFEST:
                short = name.replace("corex_", "", 1)
                setattr(self, short, None)
                self._loaded[name] = False
            return

        self._so_dir = so_dir
        logger.info("CoreX .so directory: %s", so_dir)

        loaded_count = 0
        for name in _SO_MANIFEST:
            mod = _load_so(name, so_dir)
            short = name.replace("corex_", "", 1)
            setattr(self, short, mod)
            self._loaded[name] = mod is not None
            if mod is not None:
                loaded_count += 1

        logger.info("CoreX: %d/%d .so loaded from %s",
                     loaded_count, len(_SO_MANIFEST), so_dir)

    def summary(self) -> str:
        """Return a human-readable summary of loaded modules."""
        lines = [f"CoreX .so loader ({self._so_dir or 'NOT FOUND'})"]
        for name in _SO_MANIFEST:
            status = "✓" if self._loaded.get(name) else "✗"
            short = name.replace("corex_", "", 1)
            mod = getattr(self, short, None)
            if mod is not None:
                funcs = [f for f in dir(mod) if not f.startswith("_")]
                lines.append(f"  {status} {name} → .{', .'.join(funcs)}")
            else:
                lines.append(f"  {status} {name}")
        return "\n".join(lines)

    @property
    def all_loaded(self) -> bool:
        return all(self._loaded.values())

    @property
    def loaded_count(self) -> int:
        return sum(1 for v in self._loaded.values() if v)


# Singleton — initialized on first import
corex = CoreXModules()
