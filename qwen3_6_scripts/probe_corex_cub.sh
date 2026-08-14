#!/bin/bash
# probe_corex_cub.sh — Discover what CUB/CCCL corex ships
# Run on real machine: bash qwen3_6_scripts/probe_corex_cub.sh

echo "=== corex include tree ==="
ls /usr/local/corex/include/ 2>/dev/null
echo ""

echo "=== CUB version ==="
grep -r "CUB_VERSION\|CUB_MAJOR\|CUB_MINOR\|CUB_PATCH" /usr/local/corex/include/cub/version.cuh 2>/dev/null || \
grep -r "CUB_VERSION" /usr/local/corex/include/cub/*.cuh 2>/dev/null | head -5
echo ""

echo "=== Thrust version ==="
grep "THRUST_VERSION" /usr/local/corex/include/thrust/version.h 2>/dev/null | head -3
echo ""

echo "=== CUB block primitives ==="
ls /usr/local/corex/include/cub/block/ 2>/dev/null
echo ""

echo "=== CUB warp primitives ==="
ls /usr/local/corex/include/cub/warp/ 2>/dev/null
echo ""

echo "=== CUB device algorithms ==="
ls /usr/local/corex/include/cub/device/ 2>/dev/null
echo ""

echo "=== __shfl_down_sync in corex headers ==="
grep -r "__shfl_down_sync\|__shfl_sync" /usr/local/corex/include/cub/util_ptx.cuh 2>/dev/null | head -5
echo ""

echo "=== corex CUDA version ==="
grep "CUDART_VERSION\|CUDA_VERSION" /usr/local/corex/include/cuda.h 2>/dev/null | head -3
cat /usr/local/corex/version.txt 2>/dev/null
echo ""

echo "=== Quick compile test: corex's own CUB ==="
cat > /tmp/test_corex_cub.cu << 'EOF'
#include <cub/block/block_reduce.cuh>
#include <cstdio>

__global__ void test_kernel(float* out) {
    using BlockReduce = cub::BlockReduce<float, 256>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float val = (float)threadIdx.x;
    float sum = BlockReduce(temp_storage).Sum(val);
    if (threadIdx.x == 0) out[0] = sum;
}

int main() {
    float *d_out;
    cudaMalloc(&d_out, sizeof(float));
    test_kernel<<<1, 256>>>(d_out);
    float result;
    cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("BlockReduce sum(0..255) = %.0f (expected 32640)\n", result);
    cudaFree(d_out);
    return 0;
}
EOF

/usr/local/corex/bin/clang++ --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex \
    -I/usr/local/corex/include -L/usr/local/corex/lib64 -lcudart \
    /tmp/test_corex_cub.cu -o /tmp/test_corex_cub 2>&1

if [ -f /tmp/test_corex_cub ]; then
    echo "Compile: SUCCESS"
    /tmp/test_corex_cub
else
    echo "Compile: FAILED"
fi
