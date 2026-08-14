// ix_full_bridge_v2.cpp — Complete bridge to ALL ixformer::infer C++ functions
//
// Base image has ixformer::infer namespace with 14 functions.
// Previous ix_full_bridge.cpp only bridged 4 (silu_and_mul, rms_norm,
// fused_add_rms_norm, linear). This file bridges ALL 14.
//
// The base image's _ixformer_torch.cpython-310.so and libixformer.so
// export these symbols in the ixformer::infer namespace (confirmed by nm -D).
//
// Compile:
//   torch.utils.cpp_extension.load(
//     name="ix_full_bridge_v2",
//     sources=["ix_full_bridge_v2.cpp"],
//     extra_ldflags=[<all ixformer .so files>, "-Wl,-rpath,..."],
//     extra_cflags=["-O2", "-std=c++17"],
//   )
//
// Upstream reference: xllm_latest/core/kernels/ilu/ixformer.h

#include <torch/extension.h>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

// ============================================================================
// Forward declarations — ixformer::infer namespace from base image .so
// Signatures EXACTLY match upstream_ref/xllm_latest/core/kernels/ilu/ixformer.h
// ============================================================================
namespace ixformer { namespace infer {

// --- Attention ---
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
    const c10::optional<torch::Tensor>& alibi_slopes,
    const c10::optional<torch::Tensor>& sinks,
    c10::optional<torch::Tensor>& lse);

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
    const c10::optional<torch::Tensor>& alibi_slopes,
    bool causal,
    int32_t window_left,
    int32_t window_right,
    double softcap,
    bool enable_cuda_graph,
    bool use_sqrt_alibi,
    const c10::optional<torch::Tensor>& sinks);

// --- Activation ---
void silu_and_mul(torch::Tensor& input, torch::Tensor& output);

// --- Linear ---
torch::Tensor ixformer_linear(torch::Tensor& input,
                              torch::Tensor& weight,
                              int64_t act_type,
                              const c10::optional<torch::Tensor>& bias,
                              const c10::optional<torch::Tensor>& out,
                              const c10::optional<bool> persistent);

torch::Tensor ixformer_linear_ex(torch::Tensor& input,
                                 torch::Tensor& weight,
                                 const c10::optional<torch::Tensor>& bias,
                                 const c10::optional<torch::Tensor>& out);

// --- Cache ---
void xllm_reshape_and_cache(torch::Tensor& key,
                             torch::Tensor& value,
                             torch::Tensor& key_cache,
                             torch::Tensor& value_cache,
                             torch::Tensor& slot_mapping,
                             int64_t key_token_stride,
                             int64_t value_token_stride);

// --- RoPE ---
void xllm_rotary_embedding(torch::Tensor& positions,
                            torch::Tensor& query,
                            torch::Tensor& key,
                            int64_t head_size,
                            torch::Tensor& cos_sin_cache,
                            bool is_neox);

// --- Norm ---
void residual_rms_norm(torch::Tensor& input,
                       torch::Tensor& residual,
                       torch::Tensor& weight,
                       torch::Tensor& output,
                       torch::Tensor& residual_output,
                       const c10::optional<torch::Tensor>& fused_bias,
                       double alpha,
                       double eps,
                       bool is_post);

void rms_norm(torch::Tensor& input,
              torch::Tensor& weight,
              torch::Tensor& output,
              const c10::optional<torch::Tensor>& fused_bias,
              double eps);

// --- MoE ---
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

}}  // namespace ixformer::infer


// ============================================================================
// Python wrappers — thin wrappers that match ix_bridge.py's expected API
// ============================================================================

// --- silu_and_mul ---
torch::Tensor ix_silu_and_mul(torch::Tensor input) {
    int64_t half_dim = input.size(-1) / 2;
    auto output = input.new_empty({input.size(0), half_dim});
    ixformer::infer::silu_and_mul(input, output);
    return output;
}

// --- rms_norm ---
void ix_rms_norm(torch::Tensor output, torch::Tensor input,
                 torch::Tensor weight, double eps) {
    ixformer::infer::rms_norm(input, weight, output,
                              /*fused_bias=*/c10::nullopt, eps);
}

// --- fused_add_rms_norm ---
// residual_rms_norm does: output = rms_norm(input + alpha*residual, weight, eps)
//                         residual_output = input + alpha*residual
void ix_fused_add_rms_norm(torch::Tensor input, torch::Tensor residual,
                           torch::Tensor weight, torch::Tensor output,
                           torch::Tensor residual_output, double eps) {
    ixformer::infer::residual_rms_norm(input, residual, weight,
                                       output, residual_output,
                                       /*fused_bias=*/c10::nullopt,
                                       /*alpha=*/1.0, eps,
                                       /*is_post=*/false);
}

// --- linear ---
torch::Tensor ix_linear(torch::Tensor input, torch::Tensor weight,
                        const c10::optional<torch::Tensor>& bias) {
    auto input_2d = input.view({-1, input.size(-1)});
    int64_t m = input_2d.size(0);
    if (m <= 1 && !bias.has_value()) {
        return ixformer::infer::ixformer_linear_ex(
            input, weight, bias, /*out=*/c10::optional<torch::Tensor>());
    }
    return ixformer::infer::ixformer_linear(
        input, weight, /*act_type=*/0, bias,
        /*out=*/c10::nullopt, /*persistent=*/c10::nullopt);
}

// --- rotary_embedding ---
void ix_rotary_embedding(torch::Tensor positions, torch::Tensor query,
                         torch::Tensor key, int64_t head_size,
                         torch::Tensor cos_sin_cache, bool is_neox) {
    ixformer::infer::xllm_rotary_embedding(
        positions, query, key, head_size, cos_sin_cache, is_neox);
}

// --- reshape_and_cache ---
void ix_reshape_and_cache(torch::Tensor key, torch::Tensor value,
                          torch::Tensor key_cache, torch::Tensor value_cache,
                          torch::Tensor slot_mapping) {
    // token stride = product of dims after dim 0 for key/value
    // key shape: [num_tokens, num_heads, head_dim]
    int64_t key_token_stride = 1;
    for (int i = 1; i < key.dim(); i++) key_token_stride *= key.size(i);
    int64_t value_token_stride = 1;
    for (int i = 1; i < value.dim(); i++) value_token_stride *= value.size(i);

    ixformer::infer::xllm_reshape_and_cache(
        key, value, key_cache, value_cache, slot_mapping,
        key_token_stride, value_token_stride);
}

// --- paged_attention (decode) ---
torch::Tensor ix_paged_attention(
    torch::Tensor output, torch::Tensor query,
    torch::Tensor key_cache, torch::Tensor value_cache,
    int64_t num_kv_heads, double scale,
    torch::Tensor block_tables, torch::Tensor context_lens,
    int64_t block_size, int64_t max_context_len,
    const c10::optional<torch::Tensor>& alibi_slopes) {
    return ixformer::infer::xllm_paged_attention(
        output, query, key_cache, value_cache,
        num_kv_heads, scale, block_tables, context_lens,
        block_size, max_context_len, alibi_slopes,
        /*causal=*/true, /*window_left=*/-1, /*window_right=*/-1,
        /*softcap=*/0.0, /*enable_cuda_graph=*/false,
        /*use_sqrt_alibi=*/false, /*sinks=*/c10::nullopt);
}

// --- flash_attn_prefill ---
torch::Tensor ix_flash_attn_prefill(
    torch::Tensor query, torch::Tensor key_cache, torch::Tensor value_cache,
    torch::Tensor output, torch::Tensor block_tables,
    torch::Tensor cu_seq_q, torch::Tensor cu_seq_k,
    int64_t max_query_len, int64_t max_seq_len,
    double scale, bool is_causal,
    int64_t window_left, int64_t window_right) {
    c10::optional<torch::Tensor> lse = c10::nullopt;
    return ixformer::infer::ixinfer_flash_attn_unpad_with_block_tables(
        query, key_cache, value_cache, output, block_tables,
        cu_seq_q, cu_seq_k, max_query_len, max_seq_len,
        is_causal, window_left, window_right, scale,
        /*softcap=*/0.0, /*sqrt_alibi=*/false,
        /*alibi_slopes=*/c10::nullopt, /*sinks=*/c10::nullopt, lse);
}

// --- MoE: topk_softmax ---
// Returns (topk_weights, topk_ids, token_expert_indices)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ix_topk_softmax(torch::Tensor gating_output, int64_t topk, bool renormalize) {
    int64_t num_tokens = gating_output.size(0);
    auto topk_weights = torch::empty({num_tokens, topk},
        torch::dtype(torch::kFloat32).device(gating_output.device()));
    auto topk_ids = torch::empty({num_tokens, topk},
        torch::dtype(torch::kInt32).device(gating_output.device()));
    auto token_expert_indices = torch::empty({num_tokens, topk},
        torch::dtype(torch::kInt32).device(gating_output.device()));

    auto gating_f32 = gating_output.to(torch::kFloat32);
    ixformer::infer::topk_softmax(
        topk_weights, topk_ids, token_expert_indices, gating_f32, renormalize);

    return std::make_tuple(topk_weights, topk_ids, token_expert_indices);
}

// --- MoE: moe_gen_idx ---
// Equivalent to xllm::kernel::ilu::moe_gen_idx
std::vector<torch::Tensor>
ix_moe_gen_idx(torch::Tensor expert_id, int64_t expert_num) {
    auto src_dst = expert_id.new_empty({expert_id.numel()});
    auto dst_src = torch::empty_like(src_dst);
    auto expert_sizes_gpu = expert_id.new_empty({expert_num});

    ixformer::infer::moe_compute_token_index_api(
        expert_id, src_dst, dst_src, expert_sizes_gpu,
        /*expert_mask=*/c10::nullopt,
        /*expert_sizes_cpu=*/c10::nullopt,
        /*expand_tokens_gpu=*/c10::nullopt,
        /*start_expert_id=*/0,
        /*end_expert_id=*/expert_num,
        /*num_experts=*/expert_num);

    auto expert_sizes_cumsum = expert_sizes_gpu.cumsum(-1);
    return {src_dst, dst_src, expert_sizes_gpu, expert_sizes_cumsum};
}

// --- MoE: moe_expand_input ---
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

// --- MoE: group_gemm ---
torch::Tensor ix_group_gemm(torch::Tensor inputs, torch::Tensor weights,
                            torch::Tensor tokens_per_experts,
                            int64_t output_n) {
    int64_t total_tokens = inputs.size(0);
    auto output = inputs.new_empty({total_tokens, output_n});
    ixformer::infer::moe_w16a16_group_gemm(
        output, inputs, weights, tokens_per_experts,
        /*dst_to_src=*/c10::nullopt,
        /*bias=*/c10::nullopt,
        /*format=*/"default",
        /*persistent=*/0,
        output_n);
    return output;
}

// --- MoE: moe_combine_result ---
torch::Tensor ix_moe_combine_result(torch::Tensor input, torch::Tensor weight) {
    // input: [T*topk, H], weight: [T, topk]
    auto input_3d = input.view({-1, weight.size(1), input.size(1)});
    auto output = input.new_empty({input_3d.size(0), input_3d.size(2)});
    ixformer::infer::moe_output_reduce_sum(
        output, input_3d, weight,
        /*mask=*/c10::nullopt,
        /*extra_residual=*/c10::nullopt,
        /*scaling_factor=*/1.0);
    return output;
}

// --- MoE: fused_moe_forward (7-step pipeline) ---
// This is the full fused MoE forward: topk → gen_idx → expand → gemm(w13) →
// silu_mul → gemm(w2) → combine
torch::Tensor ix_fused_moe_forward(
    torch::Tensor hidden_states,
    torch::Tensor router_logits,
    torch::Tensor w13,       // [num_experts, 2*intermediate, hidden]
    torch::Tensor w2,        // [num_experts, hidden, intermediate]
    int64_t topk,
    int64_t num_experts,
    bool renormalize) {

    // Step 1: topk_softmax
    auto [topk_weights, topk_ids, token_expert_indices] =
        ix_topk_softmax(router_logits, topk, renormalize);

    if (renormalize) {
        auto sum = topk_weights.sum(-1, /*keepdim=*/true);
        topk_weights = topk_weights / sum;
    }

    // Step 2: moe_gen_idx
    auto idx_results = ix_moe_gen_idx(topk_ids.view({-1}), num_experts);
    auto& src_dst = idx_results[0];
    auto& dst_src = idx_results[1];
    auto& expert_sizes_gpu = idx_results[2];

    // Step 3: moe_expand_input
    auto expanded = ix_moe_expand_input(hidden_states, src_dst, dst_src, topk);

    // Step 4: group_gemm (w13: gate_up projection)
    int64_t intermediate_2x = w13.size(1);
    auto gate_up = ix_group_gemm(expanded, w13.view({-1, w13.size(2)}),
                                 expert_sizes_gpu, intermediate_2x);

    // Step 5: silu_and_mul
    auto activated = ix_silu_and_mul(gate_up);

    // Step 6: group_gemm (w2: down projection)
    int64_t hidden_size = w2.size(1);
    auto down = ix_group_gemm(activated, w2.view({-1, w2.size(2)}),
                              expert_sizes_gpu, hidden_size);

    // Step 7: moe_combine_result
    auto output = ix_moe_combine_result(down, topk_weights);

    return output;
}


// ============================================================================
// Module registration — ALL 14 functions + fused pipeline
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Activation
    m.def("silu_and_mul", &ix_silu_and_mul,
          "Fused SiLU+mul activation via ixformer::infer");

    // Norm
    m.def("rms_norm", &ix_rms_norm,
          "RMSNorm via ixformer::infer");
    m.def("fused_add_rms_norm", &ix_fused_add_rms_norm,
          "Residual + RMSNorm via ixformer::infer");

    // Linear
    m.def("linear", &ix_linear,
          "GEMM via ixformer::infer (linear/linear_ex)");

    // RoPE
    m.def("rotary_embedding", &ix_rotary_embedding,
          "Rotary position embedding via ixformer::infer");

    // Cache
    m.def("reshape_and_cache", &ix_reshape_and_cache,
          "KV cache reshape+store via ixformer::infer");

    // Attention
    m.def("paged_attention", &ix_paged_attention,
          "Paged attention decode via ixformer::infer");
    m.def("flash_attn_prefill", &ix_flash_attn_prefill,
          "Flash attention prefill via ixformer::infer");

    // MoE (individual steps)
    m.def("topk_softmax", &ix_topk_softmax,
          "MoE topk+softmax routing via ixformer::infer");
    m.def("moe_gen_idx", &ix_moe_gen_idx,
          "MoE compute token index via ixformer::infer");
    m.def("moe_expand_input", &ix_moe_expand_input,
          "MoE expand input for expert dispatch via ixformer::infer");
    m.def("group_gemm", &ix_group_gemm,
          "MoE grouped GEMM via ixformer::infer");
    m.def("moe_combine_result", &ix_moe_combine_result,
          "MoE output reduce sum via ixformer::infer");

    // MoE (fused 7-step pipeline)
    m.def("fused_moe_forward", &ix_fused_moe_forward,
          "Complete fused MoE forward (7-step pipeline) via ixformer::infer");
}
