#!/usr/bin/env bash
# Build libcccl_allocator.so — LD_PRELOAD .so for CUB CachingDeviceAllocator
#
# Usage:
#   bash build_cccl_preload.sh [output_dir]
#
# On BI-V100 with CoreX SDK:
#   bash build_cccl_preload.sh /workspace/qwen3_6_scripts/cccl_preload
#
# The .so intercepts cudaMalloc/cudaFree and routes through CUB's
# caching allocator, bypassing CoreX's "expandable segment not supported"
# ASSERT in CUDACachingAllocator.cpp:545.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-${SCRIPT_DIR}}"
SRC="${SCRIPT_DIR}/cccl_allocator_preload.cu"
OUT="${OUTPUT_DIR}/libcccl_allocator.so"

# Find CoreX clang++ (preferred) or system g++
if [[ -x /usr/local/corex-3.2.3/bin/clang++ ]]; then
    CXX=/usr/local/corex-3.2.3/bin/clang++
    echo "[build] Using CoreX clang++: ${CXX}"
elif [[ -x /usr/local/corex/bin/clang++ ]]; then
    CXX=/usr/local/corex/bin/clang++
    echo "[build] Using CoreX clang++ (alt): ${CXX}"
else
    CXX=g++
    echo "[build] CoreX clang++ not found, falling back to g++"
fi

# Find CUDA include path
CUDA_INC=""
for candidate in \
    /usr/local/corex/include \
    /usr/local/cuda/include \
    /usr/local/corex/lib64/clang/16/include \
    ; do
    if [[ -f "${candidate}/cuda_runtime_api.h" ]]; then
        CUDA_INC="${candidate}"
        break
    fi
done

# Find CUDA lib path for linking
CUDA_LIB=""
for candidate in \
    /usr/local/corex/lib64 \
    /usr/local/cuda/lib64 \
    ; do
    if [[ -f "${candidate}/libcudart.so" ]]; then
        CUDA_LIB="${candidate}"
        break
    fi
done

if [[ -z "${CUDA_INC}" ]]; then
    echo "[WARN] cuda_runtime_api.h not found — trying compile anyway"
fi

echo "[build] CUDA include: ${CUDA_INC:-system}"
echo "[build] CUDA lib: ${CUDA_LIB:-system}"
echo "[build] Source: ${SRC}"
echo "[build] Output: ${OUT}"

# Build as shared library
# -x cuda or -x c++ depending on compiler
if [[ "${CXX}" == *clang++* ]]; then
    # CoreX clang++ can compile .cu natively
    ${CXX} \
        -shared -fPIC \
        -O2 \
        ${CUDA_INC:+-I"${CUDA_INC}"} \
        ${CUDA_LIB:+-L"${CUDA_LIB}"} \
        -lcudart \
        -ldl \
        -std=c++17 \
        -o "${OUT}" \
        "${SRC}"
else
    # g++ needs .cu renamed or treated as C++
    # cuda_runtime_api.h should still work with host compiler
    ${CXX} \
        -shared -fPIC \
        -O2 \
        ${CUDA_INC:+-I"${CUDA_INC}"} \
        ${CUDA_LIB:+-L"${CUDA_LIB}"} \
        -lcudart \
        -ldl \
        -std=c++17 \
        -x c++ \
        -o "${OUT}" \
        "${SRC}"
fi

if [[ -f "${OUT}" ]]; then
    SIZE=$(stat -c%s "${OUT}" 2>/dev/null || stat -f%z "${OUT}" 2>/dev/null || echo "?")
    echo "[build] SUCCESS: ${OUT} (${SIZE} bytes)"
    echo ""
    echo "Usage:"
    echo "  LD_PRELOAD=${OUT} CCCL_ALLOC_DEBUG=1 python3 -c 'import torch; t=torch.zeros(1024, device=\"cuda\")'"
    echo ""
    echo "In computility-run.yaml, add to env:"
    echo "  - name: LD_PRELOAD"
    echo "    value: /workspace/qwen3_6_scripts/cccl_preload/libcccl_allocator.so"
    echo "  - name: PYTORCH_CUDA_ALLOC_CONF"
    echo "    value: expandable_segments:True"
else
    echo "[build] FAILED"
    exit 1
fi
