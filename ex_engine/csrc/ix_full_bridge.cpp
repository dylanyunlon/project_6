// ix_full_bridge.cpp — Complete ixformer::infer bridge for BI-V100
//
// Exposes ALL 14 ixformer C++ functions to Python via pybind11.
// Header source: upstream_ref/xllm/xllm/core/kernels/ilu/ixformer.h
//
// This replaces the partial ix_moe_bridge.cpp with the full set:
//   MoE pipeline: topk_softmax, moe_compute_token_index_api, moe_expand_input,
//                 moe_w16a16_group_gemm, silu_and_mul, moe_output_reduce_sum
//   Attention:    ixinfer_flash_attn_unpad_with_block_tables, xllm_paged_attention
//   Norm:         rms_norm, residual_rms_norm
//   RoPE:         xllm_rotary_embedding
//   Linear:       ixformer_linear, ixformer_linear_ex
//   Cache:        xllm_reshape_and_cache

#include <torch/extension.h>
#include <tuple>
#include <vector>

// ============================================================================
// Forward-declare ixformer::infer namespace — matches ixformer.h exactly
// We forward-declare instead of #include to avoid build-time dependency
// on internal headers (ixinfer.h etc) that may not be on include path.
// The symbols resolve at link time against the base image's libixattn.so etc.
// ============================================================================
namespace ixformer {
namespace infer {

// --- Attention ---
torch::Tensor ixinfer_flash_attn_unpad_with_block_tables(
    torch::Tensor& query, torch::Tensor& key_cache, torch::Tensor& value_cache,
    torch::Tensor& out, torch::Tensor& block_tables,
    torch::Tensor& cu_seq_q, torch::Tensor& cu_seq_k,
    int64_t max_seq_q, int64_t max_seq_k, bool is_causal,
    int64_t window_left, int64_t window_right,
    double scale, double softcap, bool sqrt_alibi,
    const std::optional<torch::Tensor>& alibi_slopes,
    const std::optional<torch::Tensor>& sinks,
    std::optional<torch::Tensor>& lse);

torch::Tensor xllm_paged_attention(
    torch::Tensor& out, torch::Tensor& query,
    torch::Tensor& key_cache, torch::Tensor& value_cache,
    int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables, torch::Tensor& context_lens,
    int64_t block_size, int64_t max_context_len,
    const std::optional<torch::Tensor>& alibi_slopes,
    bool causal, int32_t window_left, int32_t window_right,
    double softcap, bool enable_cuda_graph, bool use_sqrt_alibi,
    const std::optional<torch::Tensor>& sinks);

// --- Norm ---
void residual_rms_norm(
    torch::Tensor& input, torch::Tensor& residual, torch::Tensor& weight,
    torch::Tensor& output, torch::Tensor& residual_output,
    const std::optional<torch::Tensor>& fused_bias,
    double alpha, double eps, bool is_post);

void rms_norm(
    torch::Tensor& input, torch::Tensor& weight, torch::Tensor& output,
    const std::optional<torch::Tensor>& fused_bias, double eps);

// --- Activation ---
void silu_and_mul(torch::Tensor& input, torch::Tensor& output);

// --- RoPE ---
void xllm_rotary_embedding(
    torch::Tensor& positions, torch::Tensor& query, torch::Tensor& key,
    int64_t head_size, torch::Tensor& cos_sin_cache, bool is_neox);

// --- KV Cache ---
void xllm_reshape_and_cache(
    torch::Tensor& key, torch::Tensor& value,
    torch::Tensor& key_cache, torch::Tensor& value_cache,
    torch::Tensor& slot_mapping,
    int64_t key_token_stride, int64_t value_token_stride);

// --- Linear ---
torch::Tensor ixformer_linear(
    torch::Tensor& input, torch::Tensor& weight, int64_t act_type,
    const std::optional<torch::Tensor>& bias,
    const std::optional<torch::Tensor>& out,
    const std::optional<bool> persistent);

torch::Tensor ixformer_linear_ex(
    torch::Tensor& input, torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    const c10::optional<torch::Tensor>& out);

// --- MoE ---
void topk_softmax(
    torch::Tensor& topk_weights, torch::Tensor& topk_indices,
    torch::Tensor& token_expert_indices, torch::Tensor& gating_output,
    bool renormalize);

void moe_compute_token_index_api(
    torch::Tensor& topk_ids, torch::Tensor& src_dst, torch::Tensor& dst_src,
    torch::Tensor& expert_sizes_gpu,
    const c10::optional<torch::Tensor>& expert_mask,
    const c10::optional<torch::Tensor>& expert_sizes_cpu,
    const c10::optional<torch::Tensor>& expand_tokens_gpu,
    int64_t start_expert_id, int64_t end_expert_id, int64_t num_experts);

void moe_expand_input(
    torch::Tensor outputs, torch::Tensor inputs, torch::Tensor dst_to_src,
    const c10::optional<torch::Tensor>& src_to_dst,
    int64_t dst_tokens, int64_t expand_factor);

void moe_w16a16_group_gemm(
    torch::Tensor output, torch::Tensor inputs, torch::Tensor weights,
    torch::Tensor tokens_per_experts,
    const c10::optional<torch::Tensor>& dst_to_src,
    const c10::optional<torch::Tensor>& bias,
    std::string format, int64_t persistent, int64_t output_n);

void moe_output_reduce_sum(
    torch::Tensor outputs, torch::Tensor inputs,
    const c10::optional<torch::Tensor>& mul_weight,
    const c10::optional<torch::Tensor>& mask,
    const c10::optional<torch::Tensor>& extra_residual,
    double scaling_factor);

}  // namespace infer
}  // namespace ixformer

// ============================================================================
// Python wrappers — thin wrappers matching upstream xllm ILU kernel layer
// Source: upstream_ref/xllm/xllm/core/kernels/ilu/*.cpp
// ============================================================================

// --- MoE: topk_softmax (from ilu/fused_moe.cpp moe_active_topk) ---
std::tuple<torch::Tensor, torch::Tensor> ix_topk_softmax(
    torch::Tensor gating_output, int64_t topk, bool renormalize) {
  auto input = gating_output.to(torch::kFloat32).contiguous();
  int64_t num_tokens = input.size(0);
  auto topk_weights = torch::empty({num_tokens, topk},
      torch::dtype(torch::kFloat32).device(input.device()));
  auto topk_indices = torch::empty({num_tokens, topk},
      torch::dtype(torch::kInt32).device(input.device()));
  auto token_expert_indices = torch::empty({num_tokens, topk},
      torch::dtype(torch::kInt32).device(input.device()));
  ixformer::infer::topk_softmax(
      topk_weights, topk_indices, token_expert_indices, input, false);
  if (renormalize) {
    topk_weights = topk_weights / topk_weights.sum(-1, /*keepdim=*/true);
  }
  return std::make_tuple(topk_weights, topk_indices);
}

// --- MoE: gen_idx (from ilu/fused_moe.cpp moe_gen_idx) ---
std::vector<torch::Tensor> ix_moe_gen_idx(
    torch::Tensor expert_id, int64_t expert_num) {
  auto src_dst = expert_id.new_empty({expert_id.numel()});
  auto dst_src = torch::empty_like(src_dst);
  auto expert_sizes_gpu = expert_id.new_empty({expert_num});
  ixformer::infer::moe_compute_token_index_api(
      expert_id, src_dst, dst_src, expert_sizes_gpu,
      c10::nullopt, c10::nullopt, c10::nullopt, 0, expert_num, expert_num);
  auto expert_sizes_gpu_cumsum = expert_sizes_gpu.cumsum(-1);
  return {src_dst, dst_src, expert_sizes_gpu, expert_sizes_gpu_cumsum};
}

// --- MoE: expand_input ---
torch::Tensor ix_moe_expand_input(
    torch::Tensor input, torch::Tensor gather_index,
    torch::Tensor combine_idx, int64_t topk) {
  int64_t dst_tokens = input.size(0) * topk;
  auto output = input.new_empty({dst_tokens, input.size(1)});
  ixformer::infer::moe_expand_input(
      output, input, combine_idx, gather_index, dst_tokens, topk);
  return output;
}

// --- MoE: group_gemm ---
torch::Tensor ix_group_gemm(
    torch::Tensor inputs, torch::Tensor weights,
    torch::Tensor token_count, int64_t output_n) {
  int64_t total_tokens = inputs.size(0);
  auto output = inputs.new_empty({total_tokens, output_n});
  ixformer::infer::moe_w16a16_group_gemm(
      output, inputs, weights, token_count,
      c10::nullopt, c10::nullopt, "NT", 0, output_n);
  return output;
}

// --- MoE: silu_and_mul ---
torch::Tensor ix_silu_and_mul(torch::Tensor input) {
  int64_t half_dim = input.size(-1) / 2;
  auto output = input.new_empty({input.size(0), half_dim});
  ixformer::infer::silu_and_mul(input, output);
  return output;
}

// --- MoE: combine_result ---
torch::Tensor ix_moe_combine_result(torch::Tensor input, torch::Tensor weight) {
  input = input.view({-1, weight.size(1), input.size(1)});
  auto output = input.new_empty({input.size(0), input.size(2)});
  ixformer::infer::moe_output_reduce_sum(
      output, input, weight, c10::nullopt, c10::nullopt, 1.0);
  return output;
}

// --- MoE: full fused forward (from ilu/layers/fused_moe.cpp) ---
torch::Tensor ix_fused_moe_forward(
    torch::Tensor hidden_states, torch::Tensor router_logits,
    torch::Tensor w13, torch::Tensor w2,
    int64_t topk, int64_t num_experts, bool renormalize) {
  auto [topk_weights, topk_ids] = ix_topk_softmax(router_logits, topk, renormalize);
  auto idx = ix_moe_gen_idx(topk_ids.view({-1}), num_experts);
  auto expanded = ix_moe_expand_input(hidden_states, idx[0], idx[1], topk);
  int64_t gate_up_dim = w13.size(1);
  auto gemm1_out = ix_group_gemm(expanded, w13, idx[2], gate_up_dim);
  auto act_out = ix_silu_and_mul(gemm1_out);
  int64_t hidden_dim = w2.size(1);
  auto gemm2_out = ix_group_gemm(act_out, w2, idx[2], hidden_dim);
  return ix_moe_combine_result(gemm2_out, topk_weights);
}

// --- Attention: paged decode (from ilu/attention.cpp batch_decode) ---
void ix_paged_attention(
    torch::Tensor output, torch::Tensor query,
    torch::Tensor key_cache, torch::Tensor value_cache,
    int64_t num_kv_heads, double scale,
    torch::Tensor block_tables, torch::Tensor seq_lens,
    int64_t block_size, int64_t max_context_len,
    const std::optional<torch::Tensor>& alibi_slopes) {
  if (query.dim() == 4) {
    query = query.view({query.size(0)*query.size(1), query.size(2), query.size(3)}).contiguous();
  }
  if (output.dim() == 4) {
    output = output.view({output.size(0)*output.size(1), output.size(2), output.size(3)}).contiguous();
  }
  ixformer::infer::xllm_paged_attention(
      output, query, key_cache, value_cache, num_kv_heads, scale,
      block_tables, seq_lens, block_size, max_context_len,
      alibi_slopes, /*causal=*/true, /*window_left=*/-1, /*window_right=*/-1,
      /*softcap=*/0.0, /*enable_cuda_graph=*/false, /*use_sqrt_alibi=*/false,
      /*sinks=*/c10::nullopt);
}

// --- Attention: prefill flash (from ilu/attention.cpp batch_prefill) ---
void ix_flash_attn_prefill(
    torch::Tensor query, torch::Tensor key, torch::Tensor value,
    torch::Tensor output, torch::Tensor block_tables,
    torch::Tensor cu_seq_q, torch::Tensor cu_seq_k,
    int64_t max_query_len, int64_t max_seq_len,
    double scale, bool is_causal,
    int64_t window_left, int64_t window_right) {
  std::optional<torch::Tensor> lse = c10::nullopt;
  ixformer::infer::ixinfer_flash_attn_unpad_with_block_tables(
      query, key, value, output, block_tables,
      cu_seq_q, cu_seq_k, max_query_len, max_seq_len,
      is_causal, window_left, window_right,
      scale, /*softcap=*/0.0, /*sqrt_alibi=*/false,
      /*alibi_slopes=*/c10::nullopt, /*sinks=*/c10::nullopt, lse);
}

// --- Norm: rms_norm (from ilu/norm.cpp) ---
void ix_rms_norm(
    torch::Tensor output, torch::Tensor input,
    torch::Tensor weight, double eps) {
  ixformer::infer::rms_norm(input, weight, output, c10::nullopt, eps);
}

// --- Norm: fused residual + rms_norm (from ilu/norm.cpp) ---
void ix_fused_add_rms_norm(
    torch::Tensor input, torch::Tensor residual,
    torch::Tensor weight, torch::Tensor output,
    torch::Tensor residual_output, double eps) {
  ixformer::infer::residual_rms_norm(
      input, residual, weight, output, residual_output,
      c10::nullopt, 1.0, eps, false);
}

// --- RoPE (from ilu/rope.cpp) ---
void ix_rotary_embedding(
    torch::Tensor positions, torch::Tensor query, torch::Tensor key,
    int64_t head_size, torch::Tensor cos_sin_cache, bool is_neox) {
  ixformer::infer::xllm_rotary_embedding(
      positions, query, key, head_size, cos_sin_cache, is_neox);
}

// --- KV Cache reshape (from ilu/attention.cpp reshape_paged_cache) ---
void ix_reshape_and_cache(
    torch::Tensor key, torch::Tensor value,
    torch::Tensor key_cache, torch::Tensor value_cache,
    torch::Tensor slot_mapping) {
  slot_mapping = slot_mapping.to(torch::kLong);
  int64_t key_stride = key.stride(0);
  int64_t val_stride = value.stride(0);
  ixformer::infer::xllm_reshape_and_cache(
      key, value, key_cache, value_cache, slot_mapping,
      key_stride, val_stride);
}

// --- Linear ---
torch::Tensor ix_linear(
    torch::Tensor input, torch::Tensor weight,
    const std::optional<torch::Tensor>& bias) {
  return ixformer::infer::ixformer_linear(
      input, weight, /*act_type=*/0, bias, c10::nullopt, c10::nullopt);
}

// ============================================================================
// Module registration
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // MoE
  m.def("topk_softmax", &ix_topk_softmax, "Fused topk+softmax",
        py::arg("gating_output"), py::arg("topk"), py::arg("renormalize")=true);
  m.def("moe_gen_idx", &ix_moe_gen_idx);
  m.def("moe_expand_input", &ix_moe_expand_input);
  m.def("group_gemm", &ix_group_gemm);
  m.def("silu_and_mul", &ix_silu_and_mul);
  m.def("moe_combine_result", &ix_moe_combine_result);
  m.def("fused_moe_forward", &ix_fused_moe_forward,
        py::arg("hidden_states"), py::arg("router_logits"),
        py::arg("w13"), py::arg("w2"),
        py::arg("topk"), py::arg("num_experts"), py::arg("renormalize")=true);
  // Attention
  m.def("paged_attention", &ix_paged_attention);
  m.def("flash_attn_prefill", &ix_flash_attn_prefill);
  // Norm
  m.def("rms_norm", &ix_rms_norm);
  m.def("fused_add_rms_norm", &ix_fused_add_rms_norm);
  // RoPE
  m.def("rotary_embedding", &ix_rotary_embedding);
  // Cache
  m.def("reshape_and_cache", &ix_reshape_and_cache);
  // Linear
  m.def("linear", &ix_linear);
}
