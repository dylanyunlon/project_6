// ix_moe_bridge.cpp — dlopen bridge to ixformer::infer MoE functions
//
// PURPOSE: base image libixformer.so has these C++ symbols but the Python
// binding (_C.so) doesn't expose them as ixformer.functions.vllm_moe_topk_softmax.
// This bridge compiles against the ixformer.h declarations and links to libixformer.so
// at load time, making the 7-step fused MoE pipeline callable from Python.
//
// BUILD: torch.utils.cpp_extension.load() with -lixformer -L/path/to/lib
//
// CALL CHAIN:
//   Python: ix_bridge.topk_softmax(weights, ids, indices, gating)
//     → ix_moe_bridge.so: ix_topk_softmax()
//       → libixformer.so: ixformer::infer::topk_softmax()
//         → CUDA kernel on BI-V100
//
// SOURCE REFERENCE: upstream_ref/xllm_latest/core/kernels/ilu/ixformer.h
//                   upstream_ref/xllm_latest/core/kernels/ilu/fused_moe.cpp

#include <torch/extension.h>
#include <optional>
#include <tuple>
#include <vector>
#include <string>

// ============================================================================
// Declarations from ixformer.h — these symbols live in libixformer.so
// The linker resolves them at .so load time via -lixformer
// ============================================================================
namespace ixformer::infer {

void topk_softmax(torch::Tensor& topk_weights,
                  torch::Tensor& topk_indices,
                  torch::Tensor& token_expert_indices,
                  torch::Tensor& gating_output,
                  bool renormalize);

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

void moe_expand_input(torch::Tensor outputs,
                      torch::Tensor inputs,
                      torch::Tensor dst_to_src,
                      const c10::optional<torch::Tensor>& src_to_dst,
                      int64_t dst_tokens,
                      int64_t expand_factor);

void moe_w16a16_group_gemm(torch::Tensor output,
                           torch::Tensor inputs,
                           torch::Tensor weights,
                           torch::Tensor tokens_per_experts,
                           const c10::optional<torch::Tensor>& dst_to_src,
                           const c10::optional<torch::Tensor>& bias,
                           std::string format,
                           int64_t persistent,
                           int64_t output_n);

void moe_output_reduce_sum(torch::Tensor outputs,
                           torch::Tensor inputs,
                           const c10::optional<torch::Tensor>& mul_weight,
                           const c10::optional<torch::Tensor>& mask,
                           const c10::optional<torch::Tensor>& extra_residual,
                           double scaling_factor);

void silu_and_mul(torch::Tensor& input, torch::Tensor& output);

void rms_norm(torch::Tensor& input,
              torch::Tensor& weight,
              torch::Tensor& output,
              const std::optional<torch::Tensor>& fused_bias,
              double eps);

void residual_rms_norm(torch::Tensor& input,
                       torch::Tensor& residual,
                       torch::Tensor& weight,
                       torch::Tensor& output,
                       torch::Tensor& residual_output,
                       const std::optional<torch::Tensor>& fused_bias,
                       double alpha,
                       double eps,
                       bool is_post);

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

torch::Tensor ixformer_linear(torch::Tensor& input,
                              torch::Tensor& weight,
                              int64_t act_type,
                              const std::optional<torch::Tensor>& bias,
                              const std::optional<torch::Tensor>& out,
                              const std::optional<bool> persistent);

void xllm_reshape_and_cache(torch::Tensor& key,
                            torch::Tensor& value,
                            torch::Tensor& key_cache,
                            torch::Tensor& value_cache,
                            torch::Tensor& slot_mapping,
                            int64_t key_token_stride,
                            int64_t value_token_stride);

void xllm_rotary_embedding(torch::Tensor& positions,
                           torch::Tensor& query,
                           torch::Tensor& key,
                           int64_t head_size,
                           torch::Tensor& cos_sin_cache,
                           bool is_neox);

}  // namespace ixformer::infer

// ============================================================================
// Python wrappers — match the signatures from ixformer_sdk/inference/functions/vllm.py
// ============================================================================

// --- MoE Step 1: topk_softmax (the missing function!) ---
void ix_topk_softmax(torch::Tensor topk_weights,
                     torch::Tensor topk_ids,
                     torch::Tensor token_expert_indices,
                     torch::Tensor gating_output) {
    ixformer::infer::topk_softmax(
        topk_weights, topk_ids, token_expert_indices, gating_output, false);
}

// --- MoE Step 2: compute token index ---
std::vector<torch::Tensor> ix_moe_gen_idx(torch::Tensor expert_id,
                                           int64_t expert_num) {
    auto src_dst = expert_id.new_empty({expert_id.numel()});
    auto dst_src = torch::empty_like(src_dst);
    auto expert_sizes_gpu = expert_id.new_empty({expert_num});

    ixformer::infer::moe_compute_token_index_api(
        expert_id, src_dst, dst_src, expert_sizes_gpu,
        c10::nullopt, c10::nullopt, c10::nullopt,
        0, expert_num, expert_num);

    auto expert_sizes_cumsum = expert_sizes_gpu.cumsum(-1);
    return {src_dst, dst_src, expert_sizes_gpu, expert_sizes_cumsum};
}

// --- MoE Step 3: expand input ---
torch::Tensor ix_moe_expand_input(torch::Tensor input,
                                   torch::Tensor gather_index,
                                   torch::Tensor combine_idx,
                                   int64_t topk) {
    int64_t dst_tokens = input.size(0) * topk;
    auto output = input.new_empty({dst_tokens, input.size(1)});
    ixformer::infer::moe_expand_input(
        output, input, combine_idx, gather_index, dst_tokens, topk);
    return output;
}

// --- MoE Step 4: group GEMM (w13: gate+up projection) ---
void ix_moe_group_gemm(torch::Tensor output,
                        torch::Tensor inputs,
                        torch::Tensor weights,
                        torch::Tensor tokens_per_experts,
                        int64_t output_n) {
    ixformer::infer::moe_w16a16_group_gemm(
        output, inputs, weights, tokens_per_experts,
        c10::nullopt, c10::nullopt,
        "auto", 0, output_n);
}

// --- MoE Step 5: silu_and_mul activation ---
torch::Tensor ix_silu_and_mul(torch::Tensor input) {
    int64_t half_dim = input.size(-1) / 2;
    auto output = input.new_empty({input.sizes()[0], half_dim});
    ixformer::infer::silu_and_mul(input, output);
    return output;
}

// --- MoE Step 6: group GEMM (w2: down projection) ---
// (reuses ix_moe_group_gemm above)

// --- MoE Step 7: combine result ---
torch::Tensor ix_moe_combine_result(torch::Tensor input, torch::Tensor weight) {
    input = input.view({-1, weight.size(1), input.size(1)});
    auto output = input.new_empty({input.size(0), input.size(2)});
    ixformer::infer::moe_output_reduce_sum(
        output, input, weight, c10::nullopt, c10::nullopt, 1.0);
    return output;
}

// --- Attention: paged attention ---
torch::Tensor ix_paged_attention(
    torch::Tensor out,
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    int64_t num_kv_heads,
    double scale,
    torch::Tensor block_tables,
    torch::Tensor context_lens,
    int64_t block_size,
    int64_t max_context_len) {
    return ixformer::infer::xllm_paged_attention(
        out, query, key_cache, value_cache,
        num_kv_heads, scale, block_tables, context_lens,
        block_size, max_context_len,
        std::nullopt, true, -1, -1, 0.0, false, false, std::nullopt);
}

// --- Norm ---
void ix_rms_norm(torch::Tensor output, torch::Tensor input,
                  torch::Tensor weight, double eps) {
    ixformer::infer::rms_norm(input, weight, output, std::nullopt, eps);
}

void ix_fused_add_rms_norm(torch::Tensor input, torch::Tensor residual,
                            torch::Tensor weight, torch::Tensor output,
                            double eps) {
    ixformer::infer::residual_rms_norm(
        input, residual, weight, output, residual, std::nullopt, 1.0, eps, false);
}

// --- Linear ---
torch::Tensor ix_linear(torch::Tensor input, torch::Tensor weight) {
    return ixformer::infer::ixformer_linear(
        input, weight, 0, std::nullopt, std::nullopt, std::nullopt);
}

// --- Cache ---
void ix_reshape_and_cache(torch::Tensor key, torch::Tensor value,
                           torch::Tensor key_cache, torch::Tensor value_cache,
                           torch::Tensor slot_mapping) {
    ixformer::infer::xllm_reshape_and_cache(
        key, value, key_cache, value_cache, slot_mapping,
        key.stride(0), value.stride(0));
}

// --- RoPE ---
void ix_rotary_embedding(torch::Tensor positions, torch::Tensor query,
                          torch::Tensor key, int64_t head_size,
                          torch::Tensor cos_sin_cache) {
    ixformer::infer::xllm_rotary_embedding(
        positions, query, key, head_size, cos_sin_cache, true);
}


// ============================================================================
// Module registration — 14 functions matching ixformer::infer API
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "ix_moe_bridge: dlopen bridge to libixformer.so MoE + inference ops";

    // MoE pipeline (7 steps)
    m.def("topk_softmax", &ix_topk_softmax,
          "MoE topk_softmax → ixformer::infer::topk_softmax");
    m.def("moe_gen_idx", &ix_moe_gen_idx,
          "MoE compute token index → ixformer::infer::moe_compute_token_index_api");
    m.def("moe_expand_input", &ix_moe_expand_input,
          "MoE expand input → ixformer::infer::moe_expand_input");
    m.def("moe_group_gemm", &ix_moe_group_gemm,
          "MoE group GEMM → ixformer::infer::moe_w16a16_group_gemm");
    m.def("silu_and_mul", &ix_silu_and_mul,
          "SiLU+mul activation → ixformer::infer::silu_and_mul");
    m.def("moe_combine_result", &ix_moe_combine_result,
          "MoE combine → ixformer::infer::moe_output_reduce_sum");

    // Attention
    m.def("paged_attention", &ix_paged_attention,
          "Paged attention → ixformer::infer::xllm_paged_attention");

    // Norm
    m.def("rms_norm", &ix_rms_norm,
          "RMSNorm → ixformer::infer::rms_norm");
    m.def("fused_add_rms_norm", &ix_fused_add_rms_norm,
          "Fused residual + RMSNorm → ixformer::infer::residual_rms_norm");

    // Linear
    m.def("linear", &ix_linear,
          "GEMM → ixformer::infer::ixformer_linear");

    // Cache
    m.def("reshape_and_cache", &ix_reshape_and_cache,
          "KV cache → ixformer::infer::xllm_reshape_and_cache");

    // RoPE
    m.def("rotary_embedding", &ix_rotary_embedding,
          "RoPE → ixformer::infer::xllm_rotary_embedding");
}
