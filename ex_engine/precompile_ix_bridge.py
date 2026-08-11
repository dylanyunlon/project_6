#!/usr/bin/env python3
"""
precompile_ix_bridge.py — Compile ix_moe_bridge.cpp → ix_moe_bridge.so

Links against libixformer.so in the base image to expose:
  - topk_softmax (the missing vllm_moe_topk_softmax)
  - moe_gen_idx, moe_expand_input, moe_group_gemm
  - silu_and_mul, moe_combine_result
  - paged_attention, rms_norm, linear, reshape_and_cache, rotary_embedding

Build chain:
  precompile_ix_bridge.py
    → torch.utils.cpp_extension.load("ix_moe_bridge", ...)
      → g++ -shared ix_moe_bridge.cpp -lixformer -L/path/to/ixformer
        → ix_moe_bridge.cpython-310-x86_64-linux-gnu.so
"""
import os
import sys
import glob
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ix_bridge_compile")

def find_ixformer_paths():
    """Find libixformer.so and ixformer include paths in base image."""
    lib_dirs = set()
    include_dirs = set()
    
    # Search paths for libixformer.so
    search = [
        "/usr/local/corex/lib64/python3/dist-packages/ixformer",
        "/usr/local/corex/lib/python3/dist-packages/ixformer",
        "/usr/local/lib/python3.10/site-packages/ixformer",
    ]
    
    for d in search:
        so = os.path.join(d, "libixformer.so")
        if os.path.exists(so):
            lib_dirs.add(d)
            logger.info(f"Found libixformer.so at: {so}")
            # Also check for csrc/include
            inc = os.path.join(d, "csrc", "include")
            if os.path.isdir(inc):
                include_dirs.add(inc)
    
    # Also search LD_LIBRARY_PATH
    for d in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if os.path.exists(os.path.join(d, "libixformer.so")):
            lib_dirs.add(d)
    
    # Fallback: find anywhere
    if not lib_dirs:
        for so in glob.glob("/usr/**/libixformer.so", recursive=True):
            lib_dirs.add(os.path.dirname(so))
            logger.info(f"Found libixformer.so at: {so}")
    
    return list(lib_dirs), list(include_dirs)


def find_source():
    """Find ix_moe_bridge.cpp."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "csrc", "ix_moe_bridge.cpp"),
        "/workspace/ex_engine/csrc/ix_moe_bridge.cpp",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def main():
    import torch
    from torch.utils.cpp_extension import load
    
    src = find_source()
    if not src:
        logger.error("ix_moe_bridge.cpp not found!")
        sys.exit(1)
    
    lib_dirs, include_dirs = find_ixformer_paths()
    if not lib_dirs:
        logger.warning("libixformer.so not found — bridge will fail at runtime")
        logger.warning("This is expected if building outside the base image")
    
    # Build flags
    extra_ldflags = []
    for d in lib_dirs:
        extra_ldflags.extend([f"-L{d}", "-Wl,-rpath," + d])
    extra_ldflags.append("-lixformer")
    
    extra_include = include_dirs[:]
    # Our own headers
    here = os.path.dirname(os.path.abspath(__file__))
    extra_include.append(os.path.join(here, "include"))
    extra_include.append(os.path.join(here, "csrc", "ilu"))
    
    extra_cflags = ["-O2", "-std=c++17"]
    
    logger.info(f"Source: {src}")
    logger.info(f"Lib dirs: {lib_dirs}")
    logger.info(f"Include dirs: {extra_include}")
    logger.info(f"Ldflags: {extra_ldflags}")
    
    build_dir = os.path.join(here, "build")
    os.makedirs(build_dir, exist_ok=True)
    
    try:
        mod = load(
            name="ix_moe_bridge",
            sources=[src],
            extra_cflags=extra_cflags,
            extra_ldflags=extra_ldflags,
            extra_include_paths=extra_include,
            build_directory=build_dir,
            verbose=True,
        )
        logger.info(f"SUCCESS: ix_moe_bridge compiled")
        logger.info(f"Functions: {[x for x in dir(mod) if not x.startswith('_')]}")
        
        # Copy .so to known location
        for so in glob.glob(os.path.join(build_dir, "*.so")):
            dst = os.path.join(here, os.path.basename(so))
            import shutil
            shutil.copy2(so, dst)
            logger.info(f"Copied: {so} → {dst}")
            
    except Exception as e:
        logger.error(f"COMPILE FAILED: {e}")
        logger.error("MoE will fall back to corex_moe.py (if base image has it)")
        # Don't exit 1 — let Docker build continue
        

if __name__ == "__main__":
    main()
