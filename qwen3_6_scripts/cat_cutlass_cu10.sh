#!/bin/bash
# cat_cutlass_cu10.sh — Cat the critical Cu10 CUTLASS files into cat_files/

SAMPLES="/usr/local/corex-samples-3.2.3_x86_64/samples/cutlass"
OUTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/cat_files"
mkdir -p "$OUTDIR"

echo "Output dir: $OUTDIR"

cp /usr/local/corex/include/crt/iluvatar_mma.hpp "$OUTDIR/iluvatar_mma.hpp"
echo "✓ iluvatar_mma.hpp"

cp "${SAMPLES}/include/cutlass/arch/mma_cu10.h" "$OUTDIR/mma_cu10.h"
echo "✓ mma_cu10.h"

cp "${SAMPLES}/include/cutlass/gemm/threadblock/default_mma_core_cu10.h" "$OUTDIR/default_mma_core_cu10.h"
echo "✓ default_mma_core_cu10.h"

cp "${SAMPLES}/examples/05_batched_gemm/batched_gemm.cu" "$OUTDIR/batched_gemm.cu"
echo "✓ batched_gemm.cu"

cp "${SAMPLES}/include/cutlass/gemm/device/gemm_universal.h" "$OUTDIR/gemm_universal.h"
echo "✓ gemm_universal.h"

cp "${SAMPLES}/include/cutlass/gemm/device/gemm_batched.h" "$OUTDIR/gemm_batched.h"
echo "✓ gemm_batched.h"

cp "${SAMPLES}/include/cutlass/gemm/warp/mma_tensor_op.h" "$OUTDIR/mma_tensor_op.h"
echo "✓ mma_tensor_op.h"

cp "${SAMPLES}/include/cutlass/gemm/warp/mma_tensor_op_policy.h" "$OUTDIR/mma_tensor_op_policy.h"
echo "✓ mma_tensor_op_policy.h"

cp "${SAMPLES}/include/cutlass/gemm/warp/mma_tensor_op_tile_iterator.h" "$OUTDIR/mma_tensor_op_tile_iterator.h"
echo "✓ mma_tensor_op_tile_iterator.h"

cp "${SAMPLES}/include/cutlass/gemm/warp/default_mma_tensor_op.h" "$OUTDIR/default_mma_tensor_op.h"
echo "✓ default_mma_tensor_op.h"

cp "${SAMPLES}/include/cutlass/gemm/threadblock/default_mma_core.h" "$OUTDIR/default_mma_core.h"
echo "✓ default_mma_core.h"

cp "${SAMPLES}/include/cutlass/gemm/device/default_gemm_configuration.h" "$OUTDIR/default_gemm_configuration.h"
echo "✓ default_gemm_configuration.h"

cp "${SAMPLES}/include/cutlass/gemm/kernel/default_gemm.h" "$OUTDIR/default_gemm.h"
echo "✓ default_gemm.h"

cp "${SAMPLES}/include/cutlass/gemm/kernel/default_gemm_universal.h" "$OUTDIR/default_gemm_universal.h"
echo "✓ default_gemm_universal.h"

# Also grab the ixinfer.h
cp /usr/local/corex/include/ixinfer.h "$OUTDIR/ixinfer.h" 2>/dev/null && echo "✓ ixinfer.h"

# Full tree
find "${SAMPLES}" -type f | sort > "$OUTDIR/cutlass_samples_tree.txt"
echo "✓ cutlass_samples_tree.txt"

echo ""
echo "=== Files saved ==="
ls -lh "$OUTDIR/"
echo ""
echo "=== Commit these with: git add cat_files/ && git commit && git push ==="
