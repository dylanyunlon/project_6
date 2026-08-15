#!/bin/bash
# cat_cutlass_cu10_part2.sh — Cat tensorop examples and arch files
SAMPLES="/usr/local/corex-samples-3.2.3_x86_64/samples/cutlass"
OUTDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/cat_files"
mkdir -p "$OUTDIR"

cp "${SAMPLES}/examples/08_turing_tensorop_gemm/turing_tensorop_gemm.cu" "$OUTDIR/turing_tensorop_gemm.cu"
echo "✓ turing_tensorop_gemm.cu"

cp "${SAMPLES}/examples/00_basic_gemm/basic_gemm.cu" "$OUTDIR/basic_gemm.cu"
echo "✓ basic_gemm.cu"

cp "${SAMPLES}/include/cutlass/arch/arch.h" "$OUTDIR/arch.h" 2>/dev/null
echo "✓ arch.h"

# Get the Cu10 arch tag definition
grep -rl "struct Cu10" "${SAMPLES}/include/" 2>/dev/null | while read f; do
    base=$(basename "$f")
    cp "$f" "$OUTDIR/arch_${base}"
    echo "✓ arch_${base} (contains Cu10 definition)"
done

# Get the cutlass.h to see CUTLASS_ARCH_CU10_SUPPORTED
cp "${SAMPLES}/include/cutlass/cutlass.h" "$OUTDIR/cutlass.h"
echo "✓ cutlass.h"

# Get gemm_batched.h full (we only had head before)
cp "${SAMPLES}/include/cutlass/gemm/device/gemm_batched.h" "$OUTDIR/gemm_batched_full.h"
echo "✓ gemm_batched_full.h"

# Get gemm.h (device level)
cp "${SAMPLES}/include/cutlass/gemm/device/gemm.h" "$OUTDIR/gemm_device.h"
echo "✓ gemm_device.h"

# Get the CMakeLists for batched_gemm and tensorop examples
cp "${SAMPLES}/examples/05_batched_gemm/CMakeLists.txt" "$OUTDIR/CMakeLists_batched_gemm.txt"
cp "${SAMPLES}/examples/08_turing_tensorop_gemm/CMakeLists.txt" "$OUTDIR/CMakeLists_tensorop_gemm.txt"
echo "✓ CMakeLists"

echo ""
ls -lh "$OUTDIR/"
echo ""
echo "git add cat_files/ && git commit -m 'data: Cu10 CUTLASS part 2' && git push"
