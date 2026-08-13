#!/usr/bin/env bash
set -euo pipefail

# Build the CCCL CachingDeviceAllocator LD_PRELOAD .so
#
# Usage: bash build_cccl_preload_allocator.sh [output_dir]
# Default output: ./cccl_preload_allocator.so
#
# Test:  LD_PRELOAD=./cccl_preload_allocator.so python3 -c "import torch; x=torch.zeros(1024,device='cuda'); del x; print('OK')"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR=${1:-${SCRIPT_DIR}}
OUTPUT=${OUTPUT_DIR}/cccl_preload_allocator.so
SOURCE=${SCRIPT_DIR}/cccl_preload_allocator.cu

COREX_ROOT=${COREX_ROOT:-/usr/local/corex-3.2.3}

# Check if we have the corex compiler
if [[ -x "${COREX_ROOT}/bin/clang++" ]]; then
    COMPILER="${COREX_ROOT}/bin/clang++"
    echo "[build] Using corex clang++: ${COMPILER}"

    "${COMPILER}" \
        -std=c++17 -O3 -shared -fPIC \
        --cuda-path="${COREX_ROOT}" --cuda-gpu-arch=ivcore10 \
        --no-cuda-version-check -D_GLIBCXX_USE_CXX11_ABI=0 \
        -I"${COREX_ROOT}/include" \
        "${SOURCE}" \
        -L"${COREX_ROOT}/lib64" -lcudart -ldl \
        -Wl,-rpath,"${COREX_ROOT}/lib64" \
        -o "${OUTPUT}"
else
    # Fallback: try to compile as pure C++ (no CUDA kernels needed)
    # The allocator is entirely host-side code
    echo "[build] corex clang++ not found, trying system g++"
    echo "[build] Note: this is HOST-only code, no GPU kernels involved"

    # Find cuda include path
    CUDA_INC=""
    for p in /usr/local/corex-3.2.3/include /usr/local/cuda/include /usr/local/corex/include; do
        if [[ -d "$p" ]]; then CUDA_INC="$p"; break; fi
    done

    CUDA_LIB=""
    for p in /usr/local/corex-3.2.3/lib64 /usr/local/cuda/lib64 /usr/local/corex/lib64; do
        if [[ -d "$p" ]]; then CUDA_LIB="$p"; break; fi
    done

    if [[ -z "$CUDA_INC" || -z "$CUDA_LIB" ]]; then
        echo "[build] ERROR: Cannot find CUDA headers/libs" >&2
        exit 2
    fi

    # Rename .cu → .cpp for g++ (it's all host code anyway)
    TMP_CPP=$(mktemp /tmp/cccl_alloc_XXXXXX.cpp)
    cp "${SOURCE}" "${TMP_CPP}"

    g++ -std=c++17 -O3 -shared -fPIC \
        -D_GLIBCXX_USE_CXX11_ABI=0 \
        -I"${CUDA_INC}" \
        "${TMP_CPP}" \
        -L"${CUDA_LIB}" -lcudart -ldl \
        -Wl,-rpath,"${CUDA_LIB}" \
        -o "${OUTPUT}"

    rm -f "${TMP_CPP}"
fi

# Verify
if [[ ! -s "${OUTPUT}" ]]; then
    echo "[build] ERROR: output is empty" >&2
    exit 2
fi

# Check it's a proper shared library with our symbols
if command -v nm &>/dev/null; then
    HAS_MALLOC=$(nm -D "${OUTPUT}" 2>/dev/null | grep -c "T cudaMalloc" || true)
    HAS_FREE=$(nm -D "${OUTPUT}" 2>/dev/null | grep -c "T cudaFree" || true)
    if [[ "$HAS_MALLOC" -gt 0 && "$HAS_FREE" -gt 0 ]]; then
        echo "[build] OK: cudaMalloc and cudaFree symbols exported"
    else
        echo "[build] WARNING: symbol check inconclusive (nm output may differ)"
    fi
fi

echo "[build] Built: ${OUTPUT} ($(stat -c%s "${OUTPUT}" 2>/dev/null || stat -f%z "${OUTPUT}") bytes)"
echo ""
echo "Test command:"
echo "  LD_PRELOAD=${OUTPUT} python3 -c \"import torch; x=torch.zeros(1024,device='cuda'); del x; print('OK')\""
echo ""
echo "Production usage (add to computility-run.yaml or Dockerfile CMD):"
echo "  LD_PRELOAD=${OUTPUT} python3 -m vllm.entrypoints.openai.api_server ..."
echo ""
echo "Tuning env vars:"
echo "  CCCL_ALLOC_BIN_GROWTH=8       # Geometric growth factor"
echo "  CCCL_ALLOC_MIN_BIN=3          # Min bin (growth^3 = 512B)"
echo "  CCCL_ALLOC_MAX_BIN=13         # Max bin (8^13 = ~550MB)"
echo "  CCCL_ALLOC_MAX_CACHED_MB=4096 # Max 4GB cached per device"
echo "  CCCL_ALLOC_DEBUG=1            # Print every alloc/free"
