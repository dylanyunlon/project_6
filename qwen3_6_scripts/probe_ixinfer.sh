#!/bin/bash
# probe_ixinfer.sh — cat the ixinfer.h header and check cublas_v2.h for batched GEMM signatures
echo "=== 1. ixinfer.h full content ==="
cat /usr/local/corex/include/ixinfer.h
echo ""
echo "=== 2. cublas_v2.h batched GEMM declarations ==="
grep -B2 -A15 "StridedBatched\|GemmBatched" /usr/local/corex/include/cublas_v2.h 2>/dev/null
echo ""
echo "=== 3. cublasLt.h — MatmulDescCreate signature ==="
grep -B2 -A10 "cublasLtMatmulDescCreate\|cublasComputeType_t\|CUBLAS_COMPUTE" /usr/local/corex/include/cublasLt.h 2>/dev/null | head -40
echo ""
echo "=== 4. Quick batched GEMM functional test ==="
cat > /tmp/test_batched_gemm.cu << 'CUDA'
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    // Simulate: 8 experts, each doing (1, 128) x (128, 256) = (1, 256)
    // Strided batch: A is (8, 1, 128), B is (8, 128, 256), C is (8, 1, 256)
    int batch = 8;
    int m = 1, n = 256, k = 128;

    __half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, batch * m * k * sizeof(__half));
    cudaMalloc(&d_B, batch * k * n * sizeof(__half));
    cudaMalloc(&d_C, batch * m * n * sizeof(__half));

    // Fill with ones for testing
    __half one_h = __float2half(1.0f);
    // (skip fill for speed, just test the API call)

    __half alpha_h = __float2half(1.0f);
    __half beta_h = __float2half(0.0f);

    // cublasHgemmStridedBatched:
    // C[i] = alpha * A[i] * B[i] + beta * C[i]
    // A[i]: m x k, B[i]: k x n, C[i]: m x n
    cublasStatus_t st = cublasHgemmStridedBatched(
        handle,
        CUBLAS_OP_N,    // transa
        CUBLAS_OP_N,    // transb
        n,              // m (cublas is column-major, so swap m/n for row-major)
        m,              // n
        k,              // k
        &alpha_h,
        d_B, n, (long long)(k * n),   // B, ldb, strideB
        d_A, k, (long long)(m * k),   // A, lda, strideA
        &beta_h,
        d_C, n, (long long)(m * n),   // C, ldc, strideC
        batch
    );

    cudaDeviceSynchronize();
    printf("cublasHgemmStridedBatched status: %d (0=SUCCESS)\n", st);

    // Also test GemmStridedBatchedEx for fp16 compute
    float alpha_f = 1.0f, beta_f = 0.0f;
    st = cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        n, m, k,
        &alpha_f,
        d_B, CUDA_R_16F, n, (long long)(k * n),
        d_A, CUDA_R_16F, k, (long long)(m * k),
        &beta_f,
        d_C, CUDA_R_16F, n, (long long)(m * n),
        batch,
        CUDA_R_32F,     // computeType
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );
    cudaDeviceSynchronize();
    printf("cublasGemmStridedBatchedEx(fp16in_fp32compute) status: %d\n", st);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);
    printf("Batched GEMM test done.\n");
    return 0;
}
CUDA

/usr/local/corex/bin/clang++ --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex \
    -I/usr/local/corex/include -L/usr/local/corex/lib64 -lcudart -lcublas \
    /tmp/test_batched_gemm.cu -o /tmp/test_batched_gemm 2>&1

if [ -f /tmp/test_batched_gemm ]; then
    echo "Compile: SUCCESS"
    /tmp/test_batched_gemm
else
    echo "Compile: FAILED"
fi
