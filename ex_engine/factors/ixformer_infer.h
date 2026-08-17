// ex_engine/factors/ixformer_infer.h
//
// Layer 7: ixformer::infer namespace contract
//
// Upstream parallel: kernels/ilu/ixformer.h (~140 lines)
//   Declares every function in ixformer::infer that the ilu kernel
//   wrappers (Layer 5-6) call through to.
//
// On BI-V100, this namespace is implemented by two sources:
//
//   1. BASE IMAGE (libixformer.so from corex SDK 3.2.3):
//      PRESENT — these symbols exist in nm -D of the .so:
//        silu_and_mul
//        rms_norm
//        residual_rms_norm
//        xllm_rotary_embedding (aka vllm_rotary_embedding_neox)
//        xllm_reshape_and_cache
//        xllm_paged_attention (v1/v2)
//        ixinfer_flash_attn_unpad_with_block_tables
//        ixformer_linear / ixformer_linear_ex
//        topk_softmax
//        moe_compute_token_index_api
//        moe_expand_input
//        moe_w16a16_group_gemm
//        moe_output_reduce_sum
//
//   2. EX FACTORS (.so files from ex_engine/build):
//      For MISSING ops that may not be in all base image versions.
//      The EX factor .so exports the same symbol → dlopen replaces it.
//
// Signature source: Verbatim from upstream xllm ixformer.h + utils.h,
//   cross-referenced with cat_files/symbol_dumps nm -D output.

#ifndef EX_FACTORS_IXFORMER_INFER_H
#define EX_FACTORS_IXFORMER_INFER_H

#include <torch/all.h>
#include <ATen/Tensor.h>
#include <optional>
#include <string>

namespace ixformer {
namespace infer {

// =====================================================================
// Attention kernels
// =====================================================================

// Flash attention with block tables (prefill path)
// Source: ixinfer flash attention unpadded variant
// BI-V100 status: PRESENT in base image
torch::Tensor ixinfer_flash_attn_unpad_with_block_tables(
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& out,
    torch::Tensor& block_tables,
    torch::Tensor& cu_seq_q,
    torch::Tensor& cu_seq_k,
    int64_t max_seq_q,
    int64_t max_seq_k,
    bool is_causal,
    int64_t window_left,
    int64_t window_right,
    double scale,
    double softcap,
    bool sqrt_alibi,
    const std::optional<torch::Tensor>& alibi_slopes,
    const std::optional<torch::Tensor>& sinks,
    std::optional<torch::Tensor>& lse);

// Paged attention (decode path, v1 or v2 selected internally)
// BI-V100 status: PRESENT
torch::Tensor xllm_paged_attention(
    torch::Tensor& out,
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    int64_t num_kv_heads,
    double scale,
    torch::Tensor& block_tables,
    torch::Tensor& context_lens,
    int64_t block_size,
    int64_t max_context_len,
    const std::optional<torch::Tensor>& alibi_slopes,
    bool causal,
    int32_t window_left,
    int32_t window_right,
    double softcap,
    bool enable_cuda_graph,
    bool use_sqrt_alibi,
    const std::optional<torch::Tensor>& sinks);

// =====================================================================
// Activation kernels
// =====================================================================

// SiLU-and-mul: out = silu(input[:half]) * input[half:]
// BI-V100 status: PRESENT
void silu_and_mul(torch::Tensor& input, torch::Tensor& output);

// =====================================================================
// Linear / GEMM kernels
// =====================================================================

// ixformer linear: fused matmul with optional activation
// BI-V100 status: PRESENT
torch::Tensor ixformer_linear(
    torch::Tensor& input,
    torch::Tensor& weight,
    int64_t act_type,
    const std::optional<torch::Tensor>& bias,
    const std::optional<torch::Tensor>& out,
    const std::optional<bool> persistent);

// ixformer linear extended: simplified interface
// BI-V100 status: PRESENT
torch::Tensor ixformer_linear_ex(
    torch::Tensor& input,
    torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::Tensor>& out);

// =====================================================================
// Cache management kernels
// =====================================================================

// Write KV into paged cache
// BI-V100 status: PRESENT
void xllm_reshape_and_cache(
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& slot_mapping,
    int64_t key_token_stride,
    int64_t value_token_stride);

// =====================================================================
// Rotary embedding kernels
// =====================================================================

// Apply rotary position encoding
// BI-V100 status: PRESENT
void xllm_rotary_embedding(
    torch::Tensor& positions,
    torch::Tensor& query,
    torch::Tensor& key,
    int64_t head_size,
    torch::Tensor& cos_sin_cache,
    bool is_neox);

// =====================================================================
// Normalization kernels
// =====================================================================

// Fused residual + RMS normalization
// BI-V100 status: PRESENT
void residual_rms_norm(
    torch::Tensor& input,
    torch::Tensor& residual,
    torch::Tensor& weight,
    torch::Tensor& output,
    torch::Tensor& residual_output,
    const std::optional<torch::Tensor>& fused_bias,
    double alpha,
    double eps,
    bool is_post);

// RMS normalization
// BI-V100 status: PRESENT
void rms_norm(
    torch::Tensor& input,
    torch::Tensor& weight,
    torch::Tensor& output,
    const std::optional<torch::Tensor>& fused_bias,
    double eps);

// =====================================================================
// MoE kernels
// =====================================================================

// Fused softmax + top-k for MoE routing
// BI-V100 status: PRESENT (confirmed in base image symbol dump)
// Calls CUDA kernel: moe_topk_softmax_kernels.cuh (Layer 8)
void topk_softmax(
    torch::Tensor& topk_weights,
    torch::Tensor& topk_indices,
    torch::Tensor& token_expert_indices,
    torch::Tensor& gating_output,
    bool renormalize);

// 3-phase permutation index computation for MoE token dispatch
// BI-V100 status: PRESENT
// Calls CUDA kernel: moe_compute_index.cu (Layer 9)
void moe_compute_token_index_api(
    torch::Tensor& topk_ids,
    torch::Tensor& src_dst,
    torch::Tensor& dst_src,
    torch::Tensor& expert_sizes_gpu,
    const c10::optional<torch::Tensor>& expert_mask,
    const c10::optional<torch::Tensor>& expert_sizes_cpu,
    const c10::optional<torch::Tensor>& expand_tokens_gpu,
    int64_t start_expert_id,
    int64_t end_expert_id,
    int64_t num_experts);

// Gather tokens from natural order to expert-sorted order
// BI-V100 status: PRESENT
void moe_expand_input(
    torch::Tensor outputs,
    torch::Tensor inputs,
    torch::Tensor dst_to_src,
    const c10::optional<torch::Tensor>& src_to_dst,
    int64_t dst_tokens,
    int64_t expand_factor);

// Group GEMM for MoE expert computation (half-precision)
// BI-V100 status: PRESENT
// This is the primary compute bottleneck.
void moe_w16a16_group_gemm(
    torch::Tensor output,
    torch::Tensor inputs,
    torch::Tensor weights,
    torch::Tensor tokens_per_experts,
    const c10::optional<torch::Tensor>& dst_to_src,
    const c10::optional<torch::Tensor>& bias,
    std::string format,
    int64_t persistent,
    int64_t output_n);

// Weighted combine of expert outputs
// BI-V100 status: PRESENT
// Calls CUDA kernel: moe_combine.cu (Layer 10)
void moe_output_reduce_sum(
    torch::Tensor outputs,
    torch::Tensor inputs,
    const c10::optional<torch::Tensor>& mul_weight,
    const c10::optional<torch::Tensor>& mask,
    const c10::optional<torch::Tensor>& extra_residual,
    double scaling_factor);

}  // namespace infer
}  // namespace ixformer

#endif  // EX_FACTORS_IXFORMER_INFER_H
