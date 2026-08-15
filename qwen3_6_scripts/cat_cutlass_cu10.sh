#!/bin/bash
# cat_cutlass_cu10.sh — Cat the critical Cu10 CUTLASS files

SAMPLES="/usr/local/corex-samples-3.2.3_x86_64/samples/cutlass"
TF_INC="/usr/local/corex-3.2.3/lib64/python3/dist-packages/tensorflow/include/third_party/gpus/cuda/include"

echo "=== 1. iluvatar_mma.hpp — the TCU intrinsics ==="
cat /usr/local/corex/include/crt/iluvatar_mma.hpp
echo ""

echo "=== 2. mma_cu10.h — CUTLASS arch Cu10 MMA ==="
cat "${SAMPLES}/include/cutlass/arch/mma_cu10.h"
echo ""

echo "=== 3. default_mma_core_cu10.h — threadblock MMA core for Cu10 ==="
cat "${SAMPLES}/include/cutlass/gemm/threadblock/default_mma_core_cu10.h"
echo ""

echo "=== 4. batched_gemm.cu — the example we'll modify for grouped GEMM ==="
cat "${SAMPLES}/examples/05_batched_gemm/batched_gemm.cu"
echo ""

echo "=== 5. cutlass samples tree (full, no depth limit) ==="
find "${SAMPLES}" -type f | sort
echo ""

echo "=== 6. gemm_universal.h (device level) ==="
cat "${SAMPLES}/include/cutlass/gemm/device/gemm_universal.h" | head -100
echo "... ($(wc -l < "${SAMPLES}/include/cutlass/gemm/device/gemm_universal.h") total lines)"
echo ""

echo "=== 7. gemm_batched.h (device level) ==="
cat "${SAMPLES}/include/cutlass/gemm/device/gemm_batched.h" | head -100
echo "... ($(wc -l < "${SAMPLES}/include/cutlass/gemm/device/gemm_batched.h") total lines)"
echo ""

echo "=== 8. mma_tensor_op.h (warp level) ==="
cat "${SAMPLES}/include/cutlass/gemm/warp/mma_tensor_op.h" | head -100
echo "... ($(wc -l < "${SAMPLES}/include/cutlass/gemm/warp/mma_tensor_op.h") total lines)"
