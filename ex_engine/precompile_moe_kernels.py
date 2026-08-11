#!/usr/bin/env python3
"""
Precompile _moe_C extension: topk_softmax + moe_align_block_size.

Proven on real BI-V100 hardware:
- WARP_SIZE=64 (not 32)
- cub/block/block_reduce.cuh (not cub/cub.cuh which pulls radix_sort)
- -cl-fast-relaxed-math (not --use_fast_math which is nvcc-only)
"""
import os, sys, logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("precompile_moe")

def main():
    import torch
    from torch.utils.cpp_extension import load

    base = os.path.dirname(os.path.abspath(__file__))
    v055 = os.path.join(base, "csrc", "moe_v055")

    sources = [
        os.path.join(v055, "topk_softmax_kernels.cu"),
        os.path.join(v055, "moe_align_block_size_kernels.cu"),
        os.path.join(v055, "moe_pybind.cpp"),
    ]
    for s in sources:
        if not os.path.exists(s):
            logger.error("MISSING: %s", s)
            sys.exit(1)

    include_paths = [
        v055,
        os.path.join(base, "csrc", "moe"),
        os.path.join(base, "csrc"),
        "/usr/local/corex/include",
    ]

    logger.info("Sources: %s", sources)
    logger.info("Compiling _moe_C...")

    try:
        mod = load(
            name="_moe_C",
            sources=sources,
            extra_include_paths=include_paths,
            extra_cuda_cflags=["-O3", "-cl-fast-relaxed-math"],
            extra_cflags=["-O2", "-std=c++17"],
            verbose=True,
        )
        fns = [x for x in dir(mod) if not x.startswith("_")]
        logger.info("SUCCESS: _moe_C functions: %s", fns)
    except Exception as e:
        logger.error("FAILED: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
