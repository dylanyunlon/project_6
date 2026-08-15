#!/bin/bash
# Build corex_batched_gemm.so — CUTLASS batched GEMM pybind for MoE decode
#
# Run on BI-V100:
#   bash build_corex_batched_gemm.sh
#
# Output: corex_batched_gemm.so (deploy to vllm package dir)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EX_ENGINE="$PROJ_ROOT/ex_engine"

# Source files
BIND_CPP="$EX_ENGINE/xllm_kernels/cuda/bindings/corex_batched_gemm_bind.cpp"
KERNEL_CU="$EX_ENGINE/xllm_kernels/cuda/corex_batched_gemm_kernel.cu"

# CUTLASS headers from cat_files (Iluvatar CoreX fork)
CUTLASS_INCLUDE="/usr/local/corex/include"
if [ ! -d "$CUTLASS_INCLUDE/cutlass" ]; then
    # Fallback: check corex-samples
    CUTLASS_INCLUDE="/usr/local/corex/samples/cutlass/include"
fi

# PyTorch/libtorch paths
TORCH_DIR=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null || echo "")
TORCH_INCLUDE=$(python3 -c "import torch; print(torch.utils.cpp_extension.include_paths()[0])" 2>/dev/null || echo "/usr/local/corex/lib/python3/dist-packages/torch/include")
TORCH_LIB=$(python3 -c "import torch; print(torch.utils.cpp_extension.library_paths()[0])" 2>/dev/null || echo "/usr/local/corex/lib/python3/dist-packages/torch/lib")
PYTHON_INCLUDE=$(python3 -c "from sysconfig import get_path; print(get_path('include'))")

echo "[build] CUTLASS_INCLUDE=$CUTLASS_INCLUDE"
echo "[build] TORCH_INCLUDE=$TORCH_INCLUDE"
echo "[build] TORCH_LIB=$TORCH_LIB"

BUILD_DIR="/tmp/build_corex_batched_gemm"
mkdir -p "$BUILD_DIR"
OUT_SO="$SCRIPT_DIR/prebuilt/corex-3.2.3-ivcore10/corex_batched_gemm.so"

# Step 1: Compile CUTLASS kernel .cu → .o
echo "[build] compiling kernel..."
nvcc -c "$KERNEL_CU" \
    -o "$BUILD_DIR/kernel.o" \
    -I "$CUTLASS_INCLUDE" \
    -I "$TORCH_INCLUDE" \
    -I "$TORCH_INCLUDE/torch/csrc/api/include" \
    --gpu-architecture=ivcore10 \
    -std=c++17 -O2 \
    --expt-relaxed-constexpr \
    -Xcompiler -fPIC

# Step 2: Compile pybind .cpp → .o
echo "[build] compiling pybind wrapper..."
g++ -c "$BIND_CPP" \
    -o "$BUILD_DIR/bind.o" \
    -I "$TORCH_INCLUDE" \
    -I "$TORCH_INCLUDE/torch/csrc/api/include" \
    -I "$PYTHON_INCLUDE" \
    -I "$CUTLASS_INCLUDE" \
    -std=c++17 -O2 -fPIC \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    -DTORCH_EXTENSION_NAME=corex_batched_gemm

# Step 3: Link → .so
echo "[build] linking..."
g++ -shared \
    "$BUILD_DIR/kernel.o" \
    "$BUILD_DIR/bind.o" \
    -o "$OUT_SO" \
    -L "$TORCH_LIB" \
    -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
    -L /usr/local/corex/lib64 -lcudart \
    -Wl,-rpath,"$TORCH_LIB" \
    -Wl,-rpath,/usr/local/corex/lib64

echo "[build] ✓ built $OUT_SO"
echo "[build] size: $(du -h "$OUT_SO" | cut -f1)"

# Quick import test
python3 -c "
import torch
torch.ops.load_library('$OUT_SO')
import importlib.util
spec = importlib.util.spec_from_file_location('corex_batched_gemm', '$OUT_SO')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('[build] ✓ import OK, functions:', [x for x in dir(mod) if not x.startswith('_')])
" 2>&1 || echo "[build] import test skipped (no GPU)"

echo "[build] done"
