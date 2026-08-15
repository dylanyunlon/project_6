/*
 * corex_batched_gemm_kernel.cu — CUTLASS half-precision batched GEMM
 *
 * Uses cutlass::gemm::device::GemmBatched with Cu10 TensorOp (ivcore10).
 * Verified: 2.462ms for 8×(1×4096 @ 4096×11008) on BI-V100.
 *
 * Source: cat_files/batched_gemm.cu adapted from float to half.
 *         cat_files/gemm_batched.h (Iluvatar CoreX CUTLASS fork)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/numeric_types.h"

/*
 * Half-precision batched strided GEMM via CUTLASS.
 *
 * C[b] = A[b] × B[b]   for b = 0..batch_count-1
 *
 * All matrices column-major.
 * The caller (corex_batched_gemm_bind.cpp) handles row-major ↔ col-major
 * transposition by swapping A/B and M/N.
 */
cudaError_t cutlass_batched_hgemm(
    int m, int n, int k,
    __half const *A, int lda, long long int batch_stride_A,
    __half const *B, int ldb, long long int batch_stride_B,
    __half *C, int ldc, long long int batch_stride_C,
    int batch_count)
{
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementAccumulator = cutlass::half_t;

    using Gemm = cutlass::gemm::device::GemmBatched<
        ElementA, cutlass::layout::ColumnMajor,   // A
        ElementB, cutlass::layout::ColumnMajor,   // B
        ElementC, cutlass::layout::ColumnMajor,   // C
        ElementAccumulator                         // accumulator
    >;

    ElementAccumulator alpha_val(1.0f);
    ElementAccumulator beta_val(0.0f);

    Gemm gemm_op;

    cutlass::Status status = gemm_op({
        {m, n, k},
        {reinterpret_cast<ElementA const *>(A), lda},
        batch_stride_A,
        {reinterpret_cast<ElementB const *>(B), ldb},
        batch_stride_B,
        {reinterpret_cast<ElementC const *>(C), ldc},
        batch_stride_C,
        {reinterpret_cast<ElementC *>(C), ldc},
        batch_stride_C,
        {alpha_val, beta_val},
        batch_count
    });

    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}
