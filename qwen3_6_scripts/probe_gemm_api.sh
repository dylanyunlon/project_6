#!/bin/bash
# probe_gemm_api.sh — Discover cuinfer + cublasLt GEMM APIs on BI-V100
# Run: bash qwen3_6_scripts/probe_gemm_api.sh

echo "=== 1. cuinfer ALL exported symbols ==="
nm -D /usr/local/corex/lib64/libcuinfer.so 2>/dev/null | grep " T " | head -80
echo "..."
echo "(total T symbols: $(nm -D /usr/local/corex/lib64/libcuinfer.so 2>/dev/null | grep ' T ' | wc -l))"
echo ""

echo "=== 2. cuinfer headers ==="
find /usr/local/corex/include -name "*cuinfer*" -o -name "*infer*" 2>/dev/null | head -20
echo ""

echo "=== 3. cuinfer.h content (if exists) ==="
for f in /usr/local/corex/include/cuinfer.h /usr/local/corex/include/cuinfer/cuinfer.h; do
    if [ -f "$f" ]; then
        echo "--- $f ---"
        cat "$f"
    fi
done
echo ""

echo "=== 4. ALL cuinfer*.h files ==="
find /usr/local/corex -name "cuinfer*.h" 2>/dev/null | while read f; do
    echo "--- $f ($(wc -l < "$f") lines) ---"
    cat "$f"
    echo ""
done
echo ""

echo "=== 5. cublasLt ALL matmul-related symbols ==="
nm -D /usr/local/corex/lib64/libcublasLt.so 2>/dev/null | grep -i "matmul\|Matmul" | head -30
echo ""

echo "=== 6. cublasLt header ==="
find /usr/local/corex/include -name "cublasLt*" 2>/dev/null
for f in /usr/local/corex/include/cublasLt.h; do
    if [ -f "$f" ]; then
        echo "--- $f exists, $(wc -l < "$f") lines ---"
        # Just show the matmul function declarations
        grep -A5 "cublasLtMatmul\b" "$f" | head -30
    fi
done
echo ""

echo "=== 7. cublas strided batched GEMM ==="
nm -D /usr/local/corex/lib64/libcublas.so 2>/dev/null | grep -i "stridedbatch\|StridedBatch\|gemmBatched\|GemmBatched" | head -20
echo ""

echo "=== 8. cublas.h — batched GEMM declarations ==="
for f in /usr/local/corex/include/cublas_v2.h /usr/local/corex/include/cublas.h; do
    if [ -f "$f" ]; then
        echo "--- $f ---"
        grep -B1 -A8 "StridedBatch\|gemmBatched" "$f" | head -60
    fi
done
echo ""

echo "=== 9. Quick cublasLt compile + link test ==="
cat > /tmp/test_cublaslt.cu << 'CUDA'
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cstdio>

int main() {
    cublasLtHandle_t handle;
    cublasLtCreate(&handle);
    printf("cublasLtCreate: OK\n");

    // Check if we can create a matmul descriptor
    cublasLtMatmulDesc_t matmulDesc;
    cublasStatus_t st = cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_16F);
    if (st == CUBLAS_STATUS_SUCCESS) {
        printf("cublasLtMatmulDescCreate(COMPUTE_16F): OK\n");
        cublasLtMatmulDescDestroy(matmulDesc);
    } else {
        printf("cublasLtMatmulDescCreate(COMPUTE_16F): FAILED (%d)\n", st);
    }

    st = cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F);
    if (st == CUBLAS_STATUS_SUCCESS) {
        printf("cublasLtMatmulDescCreate(COMPUTE_32F): OK\n");
        cublasLtMatmulDescDestroy(matmulDesc);
    } else {
        printf("cublasLtMatmulDescCreate(COMPUTE_32F): FAILED (%d)\n", st);
    }

    cublasLtDestroy(handle);
    printf("Done.\n");
    return 0;
}
CUDA

/usr/local/corex/bin/clang++ --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex \
    -I/usr/local/corex/include -L/usr/local/corex/lib64 -lcudart -lcublasLt -lcublas \
    /tmp/test_cublaslt.cu -o /tmp/test_cublaslt 2>&1

if [ -f /tmp/test_cublaslt ]; then
    echo "Compile: SUCCESS"
    /tmp/test_cublaslt
else
    echo "Compile: FAILED"
fi
echo ""

echo "=== 10. Quick cuinfer compile + link test ==="
cat > /tmp/test_cuinfer.cu << 'CUDA'
#include <cstdio>
#include <cuda_fp16.h>

// Forward declare cuinfer functions (we don't have headers)
extern "C" {
    // cuinferCustomGemm — guess signature from nm
    int cuinferCustomGemm(void* handle, int transa, int transb,
                          int m, int n, int k,
                          const void* alpha,
                          const void* A, int lda,
                          const void* B, int ldb,
                          const void* beta,
                          void* C, int ldc,
                          int computeType, void* stream);
    int cuinferCustomGemmEx(void* handle, int transa, int transb,
                            int m, int n, int k,
                            const void* alpha,
                            const void* A, int Atype, int lda,
                            const void* B, int Btype, int ldb,
                            const void* beta,
                            void* C, int Ctype, int ldc,
                            int computeType, int algo);
}

int main() {
    printf("cuinfer forward decl OK — link test only\n");
    // Just verify the library links
    return 0;
}
CUDA

/usr/local/corex/bin/clang++ --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex \
    -I/usr/local/corex/include -L/usr/local/corex/lib64 -lcudart -lcuinfer \
    /tmp/test_cuinfer.cu -o /tmp/test_cuinfer 2>&1

if [ -f /tmp/test_cuinfer ]; then
    echo "Compile: SUCCESS (cuinfer links)"
else
    echo "Compile: FAILED"
fi
echo ""

echo "=== 11. ixformer.matmul and ixformer.linear — Python API check ==="
python3 << 'PY'
import inspect, ixformer

# Check matmul
if hasattr(ixformer, 'matmul'):
    print(f"ixformer.matmul signature: {inspect.signature(ixformer.matmul)}")

# Check linear  
if hasattr(ixformer, 'linear'):
    print(f"ixformer.linear signature: {inspect.signature(ixformer.linear)}")

# Check linear_allreduce
if hasattr(ixformer, 'linear_allreduce'):
    print(f"ixformer.linear_allreduce signature: {inspect.signature(ixformer.linear_allreduce)}")

# Check gemv
if hasattr(ixformer, 'gemv'):
    print(f"ixformer.gemv signature: {inspect.signature(ixformer.gemv)}")

# Check for any batched/grouped GEMM
for name in sorted(dir(ixformer)):
    if any(k in name.lower() for k in ['batch', 'group', 'expert', 'moe', 'fused_moe']):
        obj = getattr(ixformer, name)
        try:
            sig = inspect.signature(obj)
            print(f"ixformer.{name}: {sig}")
        except:
            print(f"ixformer.{name}: (no signature)")
PY
echo ""

echo "=== 12. cublas GemmStridedBatched link test ==="
cat > /tmp/test_batched.cu << 'CUDA'
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Test: can we call cublasHgemmStridedBatched?
    // Just check it links — don't actually allocate memory
    printf("cublasHgemmStridedBatched: ");
    // void* just to test the symbol resolves
    typedef cublasStatus_t (*fn_t)(cublasHandle_t, cublasOperation_t, cublasOperation_t,
                                   int, int, int,
                                   const __half*, const __half*, int, long long,
                                   const __half*, int, long long,
                                   const __half*, __half*, int, long long, int);
    fn_t fn = (fn_t)cublasHgemmStridedBatched;
    if (fn) printf("symbol found\n"); else printf("NULL\n");

    // Also check cublasSgemmStridedBatched
    printf("cublasSgemmStridedBatched: ");
    typedef cublasStatus_t (*fn2_t)(cublasHandle_t, cublasOperation_t, cublasOperation_t,
                                    int, int, int,
                                    const float*, const float*, int, long long,
                                    const float*, int, long long,
                                    const float*, float*, int, long long, int);
    fn2_t fn2 = (fn2_t)cublasSgemmStridedBatched;
    if (fn2) printf("symbol found\n"); else printf("NULL\n");

    cublasDestroy(handle);
    return 0;
}
CUDA

/usr/local/corex/bin/clang++ --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex \
    -I/usr/local/corex/include -L/usr/local/corex/lib64 -lcudart -lcublas \
    /tmp/test_batched.cu -o /tmp/test_batched 2>&1

if [ -f /tmp/test_batched ]; then
    echo "Compile: SUCCESS"
    /tmp/test_batched
else
    echo "Compile: FAILED"
fi
