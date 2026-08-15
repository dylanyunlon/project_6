/*
 * corex_batched_gemm_bind.cpp — pybind11 wrapper for CUTLASS batched GEMM
 *
 * Verified on BI-V100: 2.462ms for 8-expert MoE decode (1×4096 @ 4096×11008)
 * vs 4.6ms for 8× torch.matmul, vs 10.36ms for Python F.linear loop.
 *
 * Call from qwen3_5.py MoE decode path (T==1):
 *   import corex_batched_gemm
 *   gate_up = corex_batched_gemm.batched_gemm_fp16(x, w13_sel)  # (K, 2*I)
 *   expert_out = corex_batched_gemm.batched_gemm_fp16(act, w2_sel)  # (K, H)
 *
 * Source: cat_files/batched_gemm.cu (CUTLASS GemmBatched)
 *         cat_files/gemm_batched.h  (Iluvatar CoreX fork)
 *
 * Build: see qwen3_6_scripts/build_corex_batched_gemm.sh
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Forward declaration — implemented in corex_batched_gemm_kernel.cu
// which uses CUTLASS GemmBatched with half precision
cudaError_t cutlass_batched_hgemm(
    int m, int n, int k,
    __half const *A, int lda, long long int batch_stride_A,
    __half const *B, int ldb, long long int batch_stride_B,
    __half *C, int ldc, long long int batch_stride_C,
    int batch_count);

/*
 * batched_gemm_fp16: (batch, M, K) × (batch, K, N) → (batch, M, N)
 *
 * For MoE decode:
 *   gate_up: x=(K,1,H), w13=(K,2I,H) → matmul(x, w13.T) → (K,1,2I)
 *     i.e. batch=K=topk, M=1, K_dim=H, N=2I
 *   down:    act=(K,1,I), w2=(K,H,I) → matmul(act, w2.T) → (K,1,H)
 *     i.e. batch=K=topk, M=1, K_dim=I, N=H
 *
 * Both A and B must be contiguous fp16 tensors on CUDA.
 */
torch::Tensor batched_gemm_fp16(
    torch::Tensor A,   // (batch, M, K)
    torch::Tensor B)   // (batch, N, K) — row-major weight, will be transposed
{
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(A.scalar_type() == torch::kFloat16 &&
                B.scalar_type() == torch::kFloat16,
                "inputs must be float16");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(),
                "inputs must be contiguous");
    TORCH_CHECK(A.dim() == 3 && B.dim() == 3,
                "inputs must be 3D (batch, rows, cols)");

    int batch = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(1);
    TORCH_CHECK(B.size(0) == batch, "batch size mismatch");
    TORCH_CHECK(B.size(2) == K, "K dimension mismatch");

    // Output: (batch, M, N)
    auto C = torch::zeros({batch, M, N}, A.options());

    // CUTLASS uses column-major internally.
    // Our tensors are row-major: A(M,K), B(N,K)
    // We compute C = A × B^T in row-major = B × A^T in col-major
    // So pass: col-major B(K,N) × A(K,M) → C(N,M), then C is (M,N) row-major
    //
    // Actually for simplicity, compute as:
    //   C(M,N) = A(M,K) × B^T(K,N)
    // In col-major: m_cm=N, n_cm=M, k_cm=K
    //   A_cm = B^T  → B stored as (N,K) row = (K,N) col, lda=K
    //   B_cm = A^T  → A stored as (M,K) row = (K,M) col, ldb=K
    //   C_cm        → C stored as (M,N) row = (N,M) col, ldc=N

    int m_cm = N;
    int n_cm = M;
    int k_cm = K;
    int lda_cm = K;   // B^T leading dim in col-major
    int ldb_cm = K;   // A^T leading dim in col-major
    int ldc_cm = N;   // C leading dim in col-major

    long long int stride_A_cm = (long long int)N * K;  // B batch stride
    long long int stride_B_cm = (long long int)M * K;  // A batch stride
    long long int stride_C_cm = (long long int)M * N;  // C batch stride

    auto status = cutlass_batched_hgemm(
        m_cm, n_cm, k_cm,
        reinterpret_cast<const __half*>(B.data_ptr<at::Half>()),
        lda_cm, stride_A_cm,
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        ldb_cm, stride_B_cm,
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        ldc_cm, stride_C_cm,
        batch);

    TORCH_CHECK(status == cudaSuccess,
                "CUTLASS batched HGEMM failed: ", cudaGetErrorString(status));
    return C;
}

/*
 * moe_decode_fused: Full MoE decode path using batched GEMM.
 *
 * hidden_states: (1, H)
 * w13_sel: (K, 2*I, H)  — selected expert gate+up weights
 * w2_sel:  (K, H, I)    — selected expert down weights
 * topk_weights: (K,)    — routing weights
 *
 * Returns: (1, H) — weighted sum of expert outputs
 */
torch::Tensor moe_decode_fused(
    torch::Tensor hidden_states,   // (1, H)
    torch::Tensor w13_sel,         // (K, 2*I, H)
    torch::Tensor w2_sel,          // (K, H, I)
    torch::Tensor topk_weights)    // (K,)
{
    int K_experts = w13_sel.size(0);
    int two_I = w13_sel.size(1);
    int H = w13_sel.size(2);
    int I = two_I / 2;

    // Expand hidden_states to (K, 1, H) for batched GEMM
    auto x = hidden_states.expand({K_experts, 1, H}).contiguous();

    // Step 1: gate_up = batched_gemm(x, w13_sel) → (K, 1, 2*I)
    auto gate_up = batched_gemm_fp16(x, w13_sel);  // (K, 1, 2I)
    gate_up = gate_up.squeeze(1);                    // (K, 2I)

    // Step 2: SiLU activation + multiply
    auto chunks = gate_up.chunk(2, /*dim=*/1);
    auto act = torch::silu(chunks[0]) * chunks[1];  // (K, I)
    act = act.unsqueeze(1);                          // (K, 1, I)

    // Step 3: expert_out = batched_gemm(act, w2_sel) → (K, 1, H)
    auto expert_out = batched_gemm_fp16(act, w2_sel);  // (K, 1, H)
    expert_out = expert_out.squeeze(1);                  // (K, H)

    // Step 4: Weighted reduction
    auto out = (expert_out * topk_weights.unsqueeze(1)).sum(0, true);  // (1, H)
    return out.to(hidden_states.dtype());
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUTLASS batched GEMM for MoE decode (BI-V100, Cu10 TensorOp)";
    m.def("batched_gemm_fp16", &batched_gemm_fp16,
          "Batched GEMM: (B,M,K) x (B,N,K)^T -> (B,M,N) in fp16",
          py::arg("A"), py::arg("B"));
    m.def("moe_decode_fused", &moe_decode_fused,
          "Full MoE decode: hidden(1,H) + w13(K,2I,H) + w2(K,H,I) + weights(K) -> out(1,H)",
          py::arg("hidden_states"), py::arg("w13_sel"),
          py::arg("w2_sel"), py::arg("topk_weights"));
}
