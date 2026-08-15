#!/bin/bash
# probe_cutlass_cu10.sh — Find and cat all Cu10 CUTLASS files on BI-V100

echo "=== 1. Find ALL cu10/Cu10/ivcorex files ==="
find /usr/local/corex -name "*cu10*" -o -name "*Cu10*" -o -name "*ivcorex*" 2>/dev/null | sort
echo ""

echo "=== 2. Find cutlass include tree ==="
find /usr/local/corex -path "*/cutlass/*.h" 2>/dev/null | head -5
# Maybe it's in a different path
find / -path "*/cutlass/gemm/*" -name "*.h" 2>/dev/null | head -20
echo ""

echo "=== 3. Find cutlass examples ==="
find / -path "*/cutlass*" -name "batched_gemm*" 2>/dev/null | head -10
find / -path "*/cutlass*" -name "*.cu" 2>/dev/null | head -20
echo ""

echo "=== 4. Find mma_cu10.h ==="
find / -name "mma_cu10*" 2>/dev/null | head -10
find / -name "*mma_tensor_op*" 2>/dev/null | head -10
echo ""

echo "=== 5. Find default_mma_core_cu10.h ==="
find / -name "default_mma_core*" 2>/dev/null | head -10
echo ""

echo "=== 6. Find __ivcorex_matrix_mad in any header ==="
grep -rl "__ivcorex_matrix_mad" /usr/local/corex/include/ 2>/dev/null | head -10
grep -rl "__ivcorex_matrix_mad" /usr/local/corex/lib64/ 2>/dev/null | head -10
echo ""

echo "=== 7. Find cutlass samples / examples dirs ==="
find / -maxdepth 5 -type d -name "cutlass*" 2>/dev/null | head -10
find / -maxdepth 6 -type d -path "*/examples/*gemm*" 2>/dev/null | head -10
echo ""

echo "=== 8. Check pip/conda packages for cutlass ==="
pip3 list 2>/dev/null | grep -i cutlass
find /usr/local/lib/python3.10 -path "*cutlass*" 2>/dev/null | head -10
echo ""

echo "=== 9. ixformer SDK includes ==="
find /usr/local/corex -path "*/ixformer*" -name "*.h" 2>/dev/null | head -20
echo ""

echo "=== 10. Check if cutlass headers shipped with corex SDK ==="
ls /usr/local/corex/include/cutlass/ 2>/dev/null | head -20
ls /usr/local/corex/share/ 2>/dev/null | head -10
find /usr/local/corex/share -name "*.h" -path "*cutlass*" 2>/dev/null | head -10
echo ""

echo "=== 11. Search for Cu10 arch definition ==="
grep -rl "Cu10\|cu10\|IVCORE" /usr/local/corex/include/ 2>/dev/null | head -20
