// ix_full_bridge_v2.cpp — Bridge to ixformer C++ functions + MoE pipeline
//
// Forward declarations use REAL symbols from nm -D symbol dumps:
//   _ixformer_torch.so → namespace ixformer_torch_ext (7 functions)
//   moe_ops_impl.cu    → namespace ixformer::infer    (5 MoE functions, self-compiled)
//
// Symbol dump verified:
//   ixformer_torch_ext::silu_and_mul_forward(at::Tensor&, at::Tensor&)
//   ixformer_torch_ext::rms_norm_forward(at::Tensor&, at::Tensor&, at::Tensor&, double)
//   ixformer_torch_ext::fused_add_rms_norm_forward(at::Tensor&, at::Tensor&, at::Tensor&, double, double)
//   ixformer_torch_ext::ixformer_linear(at::Tensor&, at::Tensor&, c10::optional<at::Tensor>, c10::optional<at::Tensor>)
//   ixformer_torch_ext::ixformer_linear_ex(at::Tensor&, at::Tensor&, c10::optional<at::Tensor>)
//   ixformer_torch_ext::vllm_rotary_embedding_neox(at::Tensor&, at::Tensor&, at::Tensor&, long, at::Tensor&, long, bool)
//   ixformer_torch_ext::vllm_cache_ops_reshape_and_cache(at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, long, long)
//   ixformer_torch_ext::vllm_single_query_cached_kv_attention(13 params — see below)
//
// NOT available in any .so (confirmed by nm -D on all 4 .so files):
//   ixinfer_flash_attn_unpad_with_block_tables — DOES NOT EXIST
//   xllm_paged_attention — DOES NOT EXIST
//   topk_softmax, moe_w16a16_group_gemm, etc — NOT in libixformer.so
//     (provided by moe_ops_impl.cu instead)

#include <torch/extension.h>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

// ============================================================================
// Forward declarations — ixformer_torch_ext namespace from _ixformer_torch.so
// Signatures EXACTLY match nm -D | c++filt output
// ============================================================================
namespace ixformer_torch_ext {

// silu_and_mul_forward(at::Tensor&, at::Tensor&)
void silu_and_mul_forward(at::Tensor& input, at::Tensor& output);

// rms_norm_forward(at::Tensor&, at::Tensor&, at::Tensor&, double)
// Real ixformer signature order: (input, weight, output, eps)
void rms_norm_forward(at::Tensor& input, at::Tensor& weight,
                      at::Tensor& output, double eps);

// fused_add_rms_norm_forward(at::Tensor&, at::Tensor&, at::Tensor&, double, double)
void fused_add_rms_norm_forward(at::Tensor& input, at::Tensor& residual,
                                at::Tensor& weight, double eps, double alpha);

// ixformer_linear(at::Tensor&, at::Tensor&, c10::optional<at::Tensor> const&, c10::optional<at::Tensor> const&)
at::Tensor ixformer_linear(at::Tensor& input, at::Tensor& weight,
                           c10::optional<at::Tensor> const& bias,
                           c10::optional<at::Tensor> const& out);

// ixformer_linear_ex(at::Tensor&, at::Tensor&, c10::optional<at::Tensor> const&)
at::Tensor ixformer_linear_ex(at::Tensor& input, at::Tensor& weight,
                              c10::optional<at::Tensor> const& bias);

// vllm_rotary_embedding_neox(at::Tensor&, at::Tensor&, at::Tensor&, long, at::Tensor&, long, bool)
void vllm_rotary_embedding_neox(at::Tensor& positions, at::Tensor& query,
                                at::Tensor& key, int64_t head_size,
                                at::Tensor& cos_sin_cache,
                                int64_t max_position, bool is_neox);

// vllm_cache_ops_reshape_and_cache(at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, long, long)
void vllm_cache_ops_reshape_and_cache(at::Tensor& key, at::Tensor& value,
                                      at::Tensor& key_cache,
                                      at::Tensor& value_cache,
                                      at::Tensor& slot_mapping,
                                      int64_t key_token_stride,
                                      int64_t value_token_stride);

// vllm_single_query_cached_kv_attention(at::Tensor& x13)
// Full signature from nm -D:
//   (at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
//    double, at::Tensor&, at::Tensor&, long, long, long, bool,
//    c10::optional<at::Tensor> const&)
void vllm_single_query_cached_kv_attention(
    at::Tensor& output, at::Tensor& query,
    at::Tensor& key_cache, at::Tensor& value_cache,
    at::Tensor& head_mapping, double scale,
    at::Tensor& block_tables, at::Tensor& context_lens,
    int64_t block_size, int64_t max_context_len, int64_t num_kv_heads,
    bool is_neox,
    c10::optional<at::Tensor> const& alibi_slopes);

}  // namespace ixformer_torch_ext

// ============================================================================
// MoE functions are NOT available in this POC image's ixformer .so files.
// MoE path uses separate prebuilt .so (corex_moe_*.so, gemm_grouped.so)
// dispatched via ix_fused_moe.py at runtime.
// ============================================================================


// ============================================================================
// Python wrappers — thin wrappers matching ix_bridge.py's expected API
// ============================================================================

// --- silu_and_mul ---
torch::Tensor ix_silu_and_mul(torch::Tensor input) {
    int64_t half_dim = input.size(-1) / 2;
    auto output = input.new_empty({input.size(0), half_dim});
    ixformer_torch_ext::silu_and_mul_forward(input, output);
    return output;
}

// --- rms_norm ---
void ix_rms_norm(torch::Tensor output, torch::Tensor input,
                 torch::Tensor weight, double eps) {
    // pybind receives (output, input, weight, eps)
    // ixformer expects (input, weight, output, eps)
    ixformer_torch_ext::rms_norm_forward(input, weight, output, eps);
}

// --- fused_add_rms_norm ---
void ix_fused_add_rms_norm(torch::Tensor input, torch::Tensor residual,
                           torch::Tensor weight, double eps) {
    ixformer_torch_ext::fused_add_rms_norm_forward(
        input, residual, weight, eps, /*alpha=*/1.0);
}

// --- linear ---
torch::Tensor ix_linear(torch::Tensor input, torch::Tensor weight,
                        const c10::optional<torch::Tensor>& bias) {
    auto input_2d = input.view({-1, input.size(-1)});
    int64_t m = input_2d.size(0);
    if (m <= 1 && !bias.has_value()) {
        return ixformer_torch_ext::ixformer_linear_ex(input, weight, bias);
    }
    return ixformer_torch_ext::ixformer_linear(
        input, weight, bias, /*out=*/c10::optional<at::Tensor>());
}

// --- rotary_embedding ---
void ix_rotary_embedding(torch::Tensor positions, torch::Tensor query,
                         torch::Tensor key, int64_t head_size,
                         torch::Tensor cos_sin_cache, bool is_neox) {
    int64_t max_position = cos_sin_cache.size(0);
    ixformer_torch_ext::vllm_rotary_embedding_neox(
        positions, query, key, head_size, cos_sin_cache, max_position, is_neox);
}

// --- reshape_and_cache ---
void ix_reshape_and_cache(torch::Tensor key, torch::Tensor value,
                          torch::Tensor key_cache, torch::Tensor value_cache,
                          torch::Tensor slot_mapping) {
    int64_t key_token_stride = 1;
    for (int i = 1; i < key.dim(); i++) key_token_stride *= key.size(i);
    int64_t value_token_stride = 1;
    for (int i = 1; i < value.dim(); i++) value_token_stride *= value.size(i);

    ixformer_torch_ext::vllm_cache_ops_reshape_and_cache(
        key, value, key_cache, value_cache, slot_mapping,
        key_token_stride, value_token_stride);
}

// --- paged_attention (decode only — no prefill available in .so) ---
void ix_paged_attention(
    torch::Tensor output, torch::Tensor query,
    torch::Tensor key_cache, torch::Tensor value_cache,
    torch::Tensor head_mapping, double scale,
    torch::Tensor block_tables, torch::Tensor context_lens,
    int64_t block_size, int64_t max_context_len, int64_t num_kv_heads,
    const c10::optional<torch::Tensor>& alibi_slopes) {
    ixformer_torch_ext::vllm_single_query_cached_kv_attention(
        output, query, key_cache, value_cache,
        head_mapping, scale, block_tables, context_lens,
        block_size, max_context_len, num_kv_heads,
        /*is_neox=*/true, alibi_slopes);
}

// ============================================================================
// Module registration
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Activation
    m.def("silu_and_mul", &ix_silu_and_mul,
          "Fused SiLU+mul via ixformer_torch_ext");

    // Norm
    m.def("rms_norm", &ix_rms_norm,
          "RMSNorm via ixformer_torch_ext");
    m.def("fused_add_rms_norm", &ix_fused_add_rms_norm,
          "Residual + RMSNorm via ixformer_torch_ext");

    // Linear
    m.def("linear", &ix_linear,
          "GEMM via ixformer_torch_ext");

    // RoPE
    m.def("rotary_embedding", &ix_rotary_embedding,
          "Rotary embedding via ixformer_torch_ext");

    // Cache
    m.def("reshape_and_cache", &ix_reshape_and_cache,
          "KV cache reshape+store via ixformer_torch_ext");

    // Attention (decode only)
    m.def("paged_attention", &ix_paged_attention,
          "Paged attention decode via ixformer_torch_ext");

    // MoE ops are NOT in this bridge — they use separate prebuilt .so files:
    //   corex_moe_topk_softmax.so, corex_moe_index_combine.so,
    //   gemm_grouped.so, corex_moe_exact_reduce.so, etc.
    // Dispatched via ix_fused_moe.py at runtime.
}