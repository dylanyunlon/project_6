// ex_engine/factors/ilu_ops_api.h
//
// Layer 4: Dispatch signature contract (canonical API header)
//
// Upstream parallel: kernels/ilu/ilu_ops_api.h (~160 lines)
//   Defines the signature contract between:
//     Layer 2-3 (orchestrators) → Layer 5-6 (kernel wrappers)
//     Layer 5-6 (kernel wrappers) → Layer 7 (ixformer::infer namespace)
//
// Every function declared here has exactly one implementation path:
//   ilu_ops_api.h declaration
//     → kernels/ilu/*.cpp wrapper (Layer 5-6)
//       → ixformer::infer::* (Layer 7, from ixformer.h)
//         → CUDA kernel (Layer 8-10)
//
// On BI-V100, ixformer::infer is the base image's libixformer.so.
// The EX engine replaces MISSING ops by providing .so factors that
// export the same infer:: signatures.
//
// This file is a verbatim-structure copy of the upstream xllm ilu_ops_api.h,
// with only the include paths adjusted for our build tree.

#ifndef EX_FACTORS_ILU_OPS_API_H
#define EX_FACTORS_ILU_OPS_API_H

#include <torch/all.h>
#include <ATen/Tensor.h>
#include <optional>
#include <string>
#include <vector>
#include <tuple>

// =========================================================================
// Namespace: xllm::kernel::ilu
// =========================================================================
// Each function maps to exactly one ixformer::infer call.
// The function bodies live in separate .cpp files (Layer 5-6).

namespace xllm {
namespace kernel {
namespace ilu {

// ---- Attention ops (Layer 6: attention.cpp) ----

void reshape_paged_cache(
    torch::Tensor& key,
    std::optional<torch::Tensor>& value,
    torch::Tensor& key_cache,
    std::optional<torch::Tensor>& value_cache,
    torch::Tensor& slot_mapping);

void batch_prefill(
    torch::Tensor& query,
    const torch::Tensor& key,
    const std::optional<torch::Tensor>& value,
    torch::Tensor& output,
    std::optional<torch::Tensor>& output_lse,
    const std::optional<torch::Tensor>& q_cu_seq_lens,
    const std::optional<torch::Tensor>& kv_cu_seq_lens,
    const std::optional<torch::Tensor>& alibi_slope,
    const std::optional<torch::Tensor>& attn_bias,
    const std::optional<torch::Tensor>& q_quant_scale,
    const std::optional<torch::Tensor>& k_quant_scale,
    const std::optional<torch::Tensor>& v_quant_scale,
    const torch::Tensor& block_tables,
    int64_t max_query_len,
    int64_t max_seq_len,
    float scale,
    bool is_causal,
    int64_t window_size_left,
    int64_t window_size_right,
    const std::string& compute_dtype,
    bool return_lse);

void batch_decode(
    torch::Tensor& query,
    const torch::Tensor& k_cache,
    torch::Tensor& output,
    const torch::Tensor& block_table,
    const torch::Tensor& seq_lens,
    const std::optional<torch::Tensor>& v_cache,
    std::optional<torch::Tensor>& output_lse,
    const std::optional<torch::Tensor>& q_quant_scale,
    const std::optional<torch::Tensor>& k_cache_quant_scale,
    const std::optional<torch::Tensor>& v_cache_quant_scale,
    const std::optional<torch::Tensor>& out_quant_scale,
    const std::optional<torch::Tensor>& alibi_slope,
    const std::optional<torch::Tensor>& mask,
    const std::string& compute_dtype,
    int64_t max_seq_len,
    int64_t window_size_left,
    int64_t window_size_right,
    float scale,
    bool return_lse,
    bool is_causal,
    int64_t kv_cache_quant_bit_size);

// ---- Normalization ops (Layer 6: norm.cpp) ----

void residual_layer_norm(
    torch::Tensor& input,
    torch::Tensor& output,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& weight,
    std::optional<torch::Tensor>& bias,
    std::optional<torch::Tensor>& residual_out,
    double eps);

void rms_norm(
    torch::Tensor& output,
    torch::Tensor& input,
    torch::Tensor& weight,
    double eps);

// ---- Activation ops (Layer 6: activation.cpp) ----

void act_and_mul(
    torch::Tensor out,
    torch::Tensor input,
    const std::string& act_mode);

// ---- RoPE ops (Layer 6: rope.cpp) ----

void apply_rope_pos_ids_cos_sin_cache(
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& cos_sin_cache,
    torch::Tensor& positions,
    bool interleave);

// ---- Linear / matmul ops (Layer 6: matmul.cpp) ----

torch::Tensor matmul(
    torch::Tensor a,
    torch::Tensor b,
    std::optional<torch::Tensor> bias);

// ---- MoE ops (Layer 5: fused_moe.cpp) ----

// Step 1: Router — softmax + topk
std::tuple<torch::Tensor, torch::Tensor> moe_active_topk(
    const torch::Tensor& input,
    int64_t topk,
    int64_t num_expert_group,
    int64_t topk_group,
    bool normalize,
    const std::optional<torch::Tensor>& mask,
    const std::string& normed_by,
    const std::string& scoring_func,
    double route_scale,
    const std::optional<torch::Tensor>& e_score_correction_bias);

// Step 2: Generate permutation indices
std::vector<torch::Tensor> moe_gen_idx(
    torch::Tensor& expert_id,
    int64_t expert_num);

// Step 3: Expand input by topk
torch::Tensor moe_expand_input(
    const torch::Tensor& input,
    const torch::Tensor& gather_index,
    const torch::Tensor& combine_idx,
    int64_t topk);

// Step 4+6: Group GEMM
torch::Tensor group_gemm(
    torch::Tensor& input,
    torch::Tensor& weight,
    torch::Tensor& tokens_per_experts,
    const std::optional<torch::Tensor>& dst_to_src,
    torch::Tensor& output);

// Step 7: Weighted combine
torch::Tensor moe_combine_result(
    torch::Tensor& input,
    torch::Tensor& weight);

}  // namespace ilu
}  // namespace kernel
}  // namespace xllm

#endif  // EX_FACTORS_ILU_OPS_API_H
