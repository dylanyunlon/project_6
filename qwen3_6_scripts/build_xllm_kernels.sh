#!/bin/bash
# build_xllm_kernels.sh — Compile xllm CUDA kernels into .so on BI-V100
#
# Uses corex's own CUB (/usr/local/corex/include/cub/) NOT cccl_upstream
# Source: ex_engine/xllm_kernels/cuda/
# Output: qwen3_6_scripts/prebuilt/corex-3.2.3-ivcore10/
#
# Run: bash qwen3_6_scripts/build_xllm_kernels.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CUDA_DIR="${PROJECT_DIR}/ex_engine/xllm_kernels/cuda"
HEADER_DIR="${CUDA_DIR}/headers"
PREBUILT_DIR="${SCRIPT_DIR}/prebuilt/corex-3.2.3-ivcore10"
mkdir -p "$PREBUILT_DIR"

build_kernel() {
    local name="$1"
    local cu_file="$2"
    echo "=== Building ${name}.so ==="
    python3 -c "
import os, glob, shutil
from torch.utils.cpp_extension import load

mod = load(
    name='${name}',
    sources=['${cu_file}'],
    extra_cflags=['-std=c++17'],
    extra_include_paths=['${HEADER_DIR}', '/usr/local/corex/include'],
    verbose=True,
)

import torch.utils.cpp_extension as ext
build_dir = ext._get_build_directory('${name}', verbose=False)
for f in glob.glob(os.path.join(build_dir, '*.so')):
    dst = '${PREBUILT_DIR}/${name}.so'
    shutil.copy2(f, dst)
    sz = os.path.getsize(dst)
    print(f'✓ ${name}.so ({sz} bytes) → {dst}')
    break

fns = [x for x in dir(mod) if not x.startswith('_')]
print(f'Functions: {fns}')
"
}

echo "Building xllm CUDA kernels for BI-V100 (ivcore10)"
echo "Using corex CUB: /usr/local/corex/include/cub/"
echo ""

# Build each kernel
build_kernel "xllm_norm"       "${CUDA_DIR}/norm.cu"
build_kernel "xllm_activation" "${CUDA_DIR}/activation.cu"
build_kernel "xllm_rope"       "${CUDA_DIR}/rope.cu"
build_kernel "xllm_block_copy" "${CUDA_DIR}/block_copy.cu"
build_kernel "xllm_cache"      "${CUDA_DIR}/reshape_paged_cache.cu"

echo ""
echo "=== All kernels built ==="
ls -la "${PREBUILT_DIR}"/xllm_*.so 2>/dev/null
