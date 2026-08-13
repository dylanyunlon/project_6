#!/usr/bin/env bash
# Build cccl_moe_sort_scatter.so using CCCL upstream headers
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SCRIPT_DIR}/cccl_moe_sort_scatter.cu"
INC="${SCRIPT_DIR}/cccl_preload/include"
OUT="${1:-${SCRIPT_DIR}/prebuilt/corex-3.2.3-ivcore10/cccl_moe_sort_scatter.so}"

[[ -f "${SRC}" ]] || { echo "Source not found: ${SRC}"; exit 2; }
[[ -d "${INC}/cub" ]] || { echo "CCCL include tree missing: ${INC}/cub"; exit 2; }

# Find NVCC or corex clang
NVCC=""
for candidate in \
    /usr/local/corex/bin/nvcc \
    /usr/local/cuda/bin/nvcc \
    ; do
    if [[ -x "${candidate}" ]]; then
        NVCC="${candidate}"
        break
    fi
done

TORCH_INC=$(python3 -c "import torch; print(torch.utils.cpp_extension.include_paths()[0])" 2>/dev/null)
TORCH_LIB=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path + '/../lib')" 2>/dev/null || echo "")
PYTHON_INC=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])" 2>/dev/null)

echo "[build] NVCC: ${NVCC:-not found}"
echo "[build] CCCL: ${INC}"
echo "[build] Torch: ${TORCH_INC}"
echo "[build] Output: ${OUT}"

if [[ -n "${NVCC}" ]]; then
    "${NVCC}" \
        -shared --compiler-options -fPIC \
        -O3 -std=c++17 \
        -I"${INC}" \
        -I"${TORCH_INC}" \
        -I"${TORCH_INC}/torch/csrc/api/include" \
        ${PYTHON_INC:+-I"${PYTHON_INC}"} \
        -DCUB_WRAPPED_NAMESPACE=cccl_moe \
        -DTORCH_EXTENSION_NAME=cccl_moe_sort_scatter \
        -x cu \
        -o "${OUT}" "${SRC}" \
        -ltorch -lc10 -ltorch_cuda -ltorch_cpu \
        ${TORCH_LIB:+-L"${TORCH_LIB}"} \
        2>&1
else
    echo "[build] No nvcc found, trying torch JIT at runtime"
    python3 -c "
from torch.utils.cpp_extension import load
mod = load(
    name='cccl_moe_sort_scatter',
    sources=['${SRC}'],
    extra_include_paths=['${INC}'],
    extra_cuda_cflags=['-O3', '-DCUB_WRAPPED_NAMESPACE=cccl_moe'],
    verbose=True,
)
print('[build] JIT compiled successfully')
import shutil, os
# Copy to output
src_so = os.path.join(os.path.dirname(mod.__file__), 'cccl_moe_sort_scatter.so')
if os.path.exists(src_so):
    os.makedirs(os.path.dirname('${OUT}'), exist_ok=True)
    shutil.copy2(src_so, '${OUT}')
    print(f'[build] Copied to ${OUT}')
" 2>&1
fi

if [[ -f "${OUT}" ]]; then
    echo "[build] SUCCESS: ${OUT} ($(stat -c%s "${OUT}" 2>/dev/null || echo '?') bytes)"
else
    echo "[build] FAILED"
    exit 1
fi
