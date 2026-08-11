#!/usr/bin/env python3
"""
Precompile ix_moe_bridge.cpp during Docker build.

This bridges Python ↔ ixformer::infer C++ API (topk_softmax, group_gemm, etc).
Source: upstream_ref/xllm/xllm/core/kernels/ilu/fused_moe.cpp call pattern
"""
import os
import sys
import glob
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("precompile_ix_bridge")

def find_ixformer_libs():
    """Find libixformer.so and related libraries for linking."""
    extra_ldflags = []
    ixf_lib_dirs = set()
    
    try:
        import ixformer
        ixf_dir = os.path.dirname(ixformer.__file__)
        for so in glob.glob(os.path.join(ixf_dir, "*.so")):
            if "cpython" not in so:
                extra_ldflags.append(so)
                ixf_lib_dirs.add(os.path.dirname(so))
        for so in glob.glob(os.path.join(ixf_dir, "_ixformer_torch*.so")):
            if so not in extra_ldflags:
                extra_ldflags.append(so)
    except ImportError:
        logger.warning("ixformer not installed")
    
    corex_lib = "/usr/local/corex/lib64"
    if os.path.isdir(corex_lib):
        for lib in ["libixformer.so", "libixattn.so", "libcublas.so"]:
            p = os.path.join(corex_lib, lib)
            if os.path.exists(p) and p not in extra_ldflags:
                extra_ldflags.append(p)
                ixf_lib_dirs.add(corex_lib)
    
    for d in ixf_lib_dirs:
        extra_ldflags.append(f"-Wl,-rpath,{d}")
    
    return extra_ldflags

def main():
    # Find the .cpp source
    search = [
        "/workspace/ex_engine/csrc/ix_moe_bridge.cpp",
        os.path.join(os.path.dirname(__file__), "csrc", "ix_moe_bridge.cpp"),
    ]
    
    cpp_path = None
    for p in search:
        if os.path.isfile(p):
            cpp_path = p
            break
    
    if not cpp_path:
        logger.error("ix_moe_bridge.cpp not found in: %s", search)
        sys.exit(1)
    
    logger.info("Compiling ix_moe_bridge from %s", cpp_path)
    
    extra_ldflags = find_ixformer_libs()
    logger.info("Link flags: %s", extra_ldflags)
    
    if not extra_ldflags:
        logger.error("No ixformer libraries found — cannot compile bridge")
        sys.exit(1)
    
    try:
        from torch.utils.cpp_extension import load
        mod = load(
            name="ix_moe_bridge",
            sources=[cpp_path],
            extra_cflags=["-O2", "-std=c++17"],
            extra_ldflags=extra_ldflags,
            verbose=True,
        )
        fns = [x for x in dir(mod) if not x.startswith("_")]
        logger.info("SUCCESS: ix_moe_bridge compiled with functions: %s", fns)
    except Exception as e:
        logger.error("FAILED to compile ix_moe_bridge: %s", e)
        # Also try ix_full_bridge.cpp
        full_path = cpp_path.replace("ix_moe_bridge", "ix_full_bridge")
        if os.path.isfile(full_path):
            logger.info("Trying ix_full_bridge.cpp instead...")
            try:
                mod = load(
                    name="ix_full_bridge",
                    sources=[full_path],
                    extra_cflags=["-O2", "-std=c++17"],
                    extra_ldflags=extra_ldflags,
                    verbose=True,
                )
                fns = [x for x in dir(mod) if not x.startswith("_")]
                logger.info("SUCCESS: ix_full_bridge compiled with functions: %s", fns)
            except Exception as e2:
                logger.error("FAILED ix_full_bridge too: %s", e2)
                sys.exit(1)
        else:
            sys.exit(1)

if __name__ == "__main__":
    main()
