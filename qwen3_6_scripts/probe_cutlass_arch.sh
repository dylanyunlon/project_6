#!/bin/bash
# probe_cutlass_arch.sh — Discover CUTLASS and __CUDA_ARCH__ on BI-V100
# Run: bash qwen3_6_scripts/probe_cutlass_arch.sh

echo "=== 1. libcutlass.so symbols (grouped GEMM related) ==="
nm -D /usr/local/corex/lib64/libcutlass.so 2>/dev/null | grep -i "gemm_grouped\|grouped_gemm\|batched_gemm\|GemmGrouped" | head -20
echo "(total cutlass symbols: $(nm -D /usr/local/corex/lib64/libcutlass.so 2>/dev/null | wc -l))"
echo ""

echo "=== 2. libcutlass.so ALL exported symbols (first 50) ==="
nm -D /usr/local/corex/lib64/libcutlass.so 2>/dev/null | grep " T " | head -50
echo ""

echo "=== 3. CUTLASS headers present? ==="
find /usr/local/corex/include -path "*/cutlass/*" -name "*.h" 2>/dev/null | head -20
echo "(total cutlass headers: $(find /usr/local/corex/include -path "*/cutlass/*" -name "*.h" 2>/dev/null | wc -l))"
echo ""

echo "=== 4. CUTLASS grouped GEMM headers? ==="
find /usr/local/corex/include -path "*/cutlass/*" -name "*group*" 2>/dev/null
find /usr/local/corex/include -path "*/cutlass/*" -name "*batch*" 2>/dev/null
echo ""

echo "=== 5. CUTLASS version ==="
grep -r "CUTLASS_MAJOR\|CUTLASS_MINOR\|CUTLASS_PATCH" /usr/local/corex/include/cutlass/cutlass.h 2>/dev/null | head -5
echo ""

echo "=== 6. __CUDA_ARCH__ value on ivcore10 ==="
cat > /tmp/probe_arch.cu << 'CUDA'
#include <cstdio>

__global__ void print_arch() {
#ifdef __CUDA_ARCH__
    if (threadIdx.x == 0) {
        printf("__CUDA_ARCH__ = %d\n", __CUDA_ARCH__);
    }
#else
    if (threadIdx.x == 0) {
        printf("__CUDA_ARCH__ not defined (host code)\n");
    }
#endif
}

// Host-side macro check
int main() {
    printf("Host: __CUDA_ARCH__ ");
#ifdef __CUDA_ARCH__
    printf("= %d\n", __CUDA_ARCH__);
#else
    printf("not defined (expected for host)\n");
#endif

    print_arch<<<1, 1>>>();
    cudaDeviceSynchronize();

    // Also check compute capability via runtime API
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("cudaDeviceProp: major=%d minor=%d\n", prop.major, prop.minor);
    printf("Device name: %s\n", prop.name);
    printf("SM count: %d\n", prop.multiProcessorCount);
    printf("Shared mem per block: %zu\n", prop.sharedMemPerBlock);
    printf("Shared mem per SM: %zu\n", prop.sharedMemPerMultiprocessor);
    printf("Warp size: %d\n", prop.warpSize);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Clock rate: %d kHz\n", prop.clockRate);
    printf("Memory clock: %d kHz\n", prop.memoryClockRate);
    printf("Memory bus width: %d bits\n", prop.memoryBusWidth);
    printf("L2 cache: %d bytes\n", prop.l2CacheSize);
    printf("Total global mem: %zu bytes\n", prop.totalGlobalMem);

    return 0;
}
CUDA

echo "Compiling..."
/usr/local/corex/bin/clang++ --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex \
    -I/usr/local/corex/include -L/usr/local/corex/lib64 -lcudart \
    /tmp/probe_arch.cu -o /tmp/probe_arch 2>&1

if [ -f /tmp/probe_arch ]; then
    echo "Compile: SUCCESS"
    /tmp/probe_arch
else
    echo "Compile: FAILED"
fi
echo ""

echo "=== 7. PyTorch CUDA arch ==="
python3 -c "
import torch
print(f'torch.cuda.get_device_capability() = {torch.cuda.get_device_capability()}')
print(f'torch.cuda.get_device_name() = {torch.cuda.get_device_name()}')
print(f'torch.cuda.get_arch_list() = {torch.cuda.get_arch_list()}')
print(f'torch.version.cuda = {torch.version.cuda}')
" 2>/dev/null
echo ""

echo "=== 8. cublasLt symbols (batched GEMM) ==="
nm -D /usr/local/corex/lib64/libcublasLt.so 2>/dev/null | grep -i "matmul\|batch\|group" | head -20
echo "(total cublasLt symbols: $(nm -D /usr/local/corex/lib64/libcublasLt.so 2>/dev/null | wc -l))"
echo ""

echo "=== 9. libcuinfer.so (if it has grouped GEMM) ==="
nm -D /usr/local/corex/lib64/libcuinfer.so 2>/dev/null | grep -i "gemm\|matmul\|expert\|moe" | head -20
echo ""

echo "=== 10. ixformer _ixformer_torch.so — check for WMMA/gemm symbols ==="
TORCH_SO=$(find /usr/local/corex -name "_ixformer_torch*.so" 2>/dev/null | head -1)
if [ -n "$TORCH_SO" ]; then
    nm -D "$TORCH_SO" 2>/dev/null | grep -i "wmma\|gemm\|moe\|expert\|group" | head -30
fi
echo ""

echo "=== 11. CUTLASS GEMM templates check ==="
ls /usr/local/corex/include/cutlass/gemm/ 2>/dev/null
ls /usr/local/corex/include/cutlass/gemm/device/ 2>/dev/null
echo ""

echo "=== 12. Quick CUTLASS compile test ==="
cat > /tmp/test_cutlass.cu << 'CUDA'
#include <cutlass/cutlass.h>
#include <cstdio>

int main() {
    printf("CUTLASS version: %d.%d.%d\n",
           CUTLASS_MAJOR, CUTLASS_MINOR, CUTLASS_PATCH);
    return 0;
}
CUDA

/usr/local/corex/bin/clang++ --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex \
    -I/usr/local/corex/include -L/usr/local/corex/lib64 -lcudart -lcutlass \
    /tmp/test_cutlass.cu -o /tmp/test_cutlass 2>&1

if [ -f /tmp/test_cutlass ]; then
    echo "CUTLASS compile: SUCCESS"
    /tmp/test_cutlass
else
    echo "CUTLASS compile: FAILED (may need different include path)"
    # Try alternative paths
    find /usr/local/corex -name "cutlass.h" -path "*/cutlass/*" 2>/dev/null
fi
