// ix_unified_bridge.cpp — Unified pybind11 bridge for all ixformer::infer APIs
//
// This is the single dlopen entry point that exposes the complete ixformer
// kernel API to Python.  It links against the base-image .so files at runtime:
//   - _ixformer_torch.cpython-310.so  (silu_and_mul, rms_norm, linear, etc.)
//   - libixformer.so                  (flash_attn, paged_attention)
//   - libixattn.so                    (attention kernels)
//
// The ixformer::infer symbols are resolved by the dynamic linker because
// the base image already has them loaded.  We just need to declare them
// (in ixformer.h) and call them.
//
// Namespace mapping:
//   ixformer::infer::*           → direct from ixformer.h (14 functions)
//   xllm::kernel::ilu::*         → wrappers from upstream xllm (搬运)
//
// Adapted from: upstream_ref/xllm/xllm/core/kernels/ilu/

#include <torch/extension.h>
#include <optional>
#include <vector>
#include <tuple>

#include "ixformer.h"
#include "ilu_ops_api.h"

using namespace ixformer;

// ============================================================================
// Direct ixformer::infer wrappers (thin Python-facing layer)
// ============================================================================

// --- Activation ---
static torch::Tensor py_silu_and_mul(torch::Tensor input) {
    int64_t d = input.size(-1) / 2;
    auto out = input.new_empty({input.size(0), d});
    infer::silu_and_mul(input, out);
    return out;
}

// --- Norm ---
static void py_rms_norm(torch::Tensor output, torch::Tensor input,
                        torch::Tensor weight, double eps) {
    std::optional<torch::Tensor> bias = std::nullopt;
    infer::rms_norm(input, weight, output, bias, eps);
}

static void py_fused_add_rms_norm(torch::Tensor input, torch::Tensor residual,
                                  torch::Tensor weight, double eps) {
    auto output = torch::empty_like(input);
    auto residual_out = torch::empty_like(input);
    std::optional<torch::Tensor> bias = std::nullopt;
    infer::residual_rms_norm(input, residual, weight, output, residual_out,
                             bias, /*alpha=*/1.0, eps, /*is_post=*/false);
    // Copy back in-place
    input.copy_(output);
    residual.copy_(residual_out);
}

// --- Linear ---
static torch::Tensor py_linear(torch::Tensor input, torch::Tensor weight,
                                const c10::optional<torch::Tensor>& bias) {
    std::vector<int64_t> out_shape = input.sizes().vec();
    if (!out_shape.empty()) {
        out_shape[out_shape.size() - 1] = weight.size(0);
    }
    auto output = input.new_empty(out_shape);
    c10::optional<torch::Tensor> out_opt = output;

    // Try linear_ex for small batch (decode), linear for larger
    if (input.size(0) <= 1 && input.size(-1) % 32 == 0 &&
        weight.size(0) % 2 == 0 && !bias.has_value()) {
        output = infer::ixformer_linear_ex(input, weight, bias, out_opt);
    } else {
        int64_t act_type = -1;
        c10::optional<bool> persistent = false;
        output = infer::ixformer_linear(input, weight, act_type, bias,
                                        out_opt, persistent);
    }
    return output;
}

// --- RoPE ---
static void py_rotary_embedding(torch::Tensor positions, torch::Tensor query,
                                torch::Tensor key, int64_t head_size,
                                torch::Tensor cos_sin_cache, bool is_neox) {
    infer::xllm_rotary_embedding(positions, query, key, head_size,
                                 cos_sin_cache, is_neox);
}

// --- KV Cache ---
static void py_reshape_and_cache(torch::Tensor key, torch::Tensor value,
                                 torch::Tensor key_cache,
                                 torch::Tensor value_cache,
                                 torch::Tensor slot_mapping) {
    int64_t key_stride = key.stride(0);
    int64_t val_stride = value.stride(0);
    infer::xllm_reshape_and_cache(key, value, key_cache, value_cache,
                                  slot_mapping, key_stride, val_stride);
}

// --- Attention: prefill ---
static torch::Tensor py_flash_attn_prefill(
    torch::Tensor query, torch::Tensor key_cache, torch::Tensor value_cache,
    torch::Tensor output, torch::Tensor block_tables,
    torch::Tensor cu_seq_q, torch::Tensor cu_seq_k,
    int64_t max_seq_q, int64_t max_seq_k,
    bool is_causal, double scale) {
    int64_t wl = -1, wr = -1;
    double softcap = 0.0;
    bool sqrt_alibi = false;
    std::optional<torch::Tensor> alibi = std::nullopt;
    std::optional<torch::Tensor> sinks = std::nullopt;
    std::optional<torch::Tensor> lse = std::nullopt;
    return infer::ixinfer_flash_attn_unpad_with_block_tables(
        query, key_cache, value_cache, output, block_tables,
        cu_seq_q, cu_seq_k, max_seq_q, max_seq_k,
        is_causal, wl, wr, scale, softcap, sqrt_alibi,
        alibi, sinks, lse);
}

// --- Attention: decode (paged) ---
static torch::Tensor py_paged_attention(
    torch::Tensor output, torch::Tensor query,
    torch::Tensor key_cache, torch::Tensor value_cache,
    int64_t num_kv_heads, double scale,
    torch::Tensor block_tables, torch::Tensor context_lens,
    int64_t block_size, int64_t max_context_len) {
    std::optional<torch::Tensor> alibi = std::nullopt;
    bool causal = true;
    int32_t wl = -1, wr = -1;
    double softcap = 0.0;
    bool enable_cuda_graph = false;
    bool sqrt_alibi = false;
    std::optional<torch::Tensor> sinks = std::nullopt;
    return infer::xllm_paged_attention(
        output, query, key_cache, value_cache,
        num_kv_heads, scale, block_tables, context_lens,
        block_size, max_context_len, alibi, causal, wl, wr,
        softcap, enable_cuda_graph, sqrt_alibi, sinks);
}

// --- MoE: topk_softmax ---
static std::tuple<torch::Tensor, torch::Tensor> py_moe_topk_softmax(
    torch::Tensor gating_output, int64_t topk, bool renormalize) {
    auto gating_f32 = gating_output.to(torch::kFloat32);
    int64_t n_tokens = gating_f32.size(0);
    auto topk_weights = torch::empty({n_tokens, topk},
        torch::dtype(torch::kFloat).device(gating_f32.device()));
    auto topk_indices = torch::empty({n_tokens, topk},
        torch::dtype(torch::kInt32).device(gating_f32.device()));
    auto token_expert_indices = torch::empty({n_tokens, topk},
        torch::dtype(torch::kInt32).device(gating_f32.device()));

    infer::topk_softmax(topk_weights, topk_indices, token_expert_indices,
                        gating_f32, false);
    if (renormalize) {
        auto sums = topk_weights.sum(-1, /*keepdim=*/true);
        topk_weights = topk_weights / sums;
    }
    return std::make_tuple(topk_weights, topk_indices);
}

// --- MoE: compute_token_index ---
static std::vector<torch::Tensor> py_moe_gen_idx(
    torch::Tensor expert_ids, int64_t num_experts) {
    auto src_dst = expert_ids.new_empty({expert_ids.numel()});
    auto dst_src = torch::empty_like(src_dst);
    auto expert_sizes = expert_ids.new_empty({num_experts});

    infer::moe_compute_token_index_api(
        expert_ids, src_dst, dst_src, expert_sizes,
        /*expert_mask=*/std::nullopt,
        /*expert_sizes_cpu=*/std::nullopt,
        /*expand_tokens_gpu=*/std::nullopt,
        /*start_expert_id=*/0,
        /*end_expert_id=*/num_experts,
        /*num_experts=*/num_experts);

    auto cumsum = expert_sizes.cumsum(-1);
    return {src_dst, dst_src, expert_sizes, cumsum};
}

// --- MoE: expand_input ---
static torch::Tensor py_moe_expand_input(
    torch::Tensor input, torch::Tensor gather_index,
    torch::Tensor combine_idx, int64_t topk) {
    int64_t dst_tokens = input.size(0) * topk;
    auto output = input.new_empty({dst_tokens, input.size(1)});
    infer::moe_expand_input(output, input, combine_idx, gather_index,
                            dst_tokens, topk);
    return output;
}

// --- MoE: group_gemm ---
static torch::Tensor py_moe_group_gemm(
    torch::Tensor input, torch::Tensor weight,
    torch::Tensor tokens_per_experts) {
    int64_t out_features = weight.size(-2);  // weight is [E, N, K] in TN format
    auto output = input.new_empty({input.size(0), out_features});
    infer::moe_w16a16_group_gemm(
        output, input, weight, tokens_per_experts,
        /*dst_to_src=*/std::nullopt,
        /*bias=*/std::nullopt,
        /*format=*/"TN",
        /*persistent=*/0,
        /*output_n=*/input.size(0));
    return output;
}

// --- MoE: combine_result (reduce_sum) ---
static torch::Tensor py_moe_combine_result(
    torch::Tensor input, torch::Tensor weights) {
    // input: [n_tokens, topk, hidden]  weights: [n_tokens, topk]
    auto inp_3d = input.view({-1, weights.size(1), input.size(-1)});
    auto output = input.new_empty({inp_3d.size(0), inp_3d.size(2)});
    infer::moe_output_reduce_sum(
        output, inp_3d, weights,
        /*mask=*/std::nullopt,
        /*extra_residual=*/std::nullopt,
        /*scaling_factor=*/1.0);
    return output;
}

// ============================================================================
// PYBIND11 MODULE — single entry point for all ixformer ops
// ============================================================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "ix_unified_bridge: complete ixformer::infer API for BI-V100";

    // Activation
    m.def("silu_and_mul", &py_silu_and_mul, "Fused SiLU+Mul");

    // Norm
    m.def("rms_norm", &py_rms_norm, "RMSNorm");
    m.def("fused_add_rms_norm", &py_fused_add_rms_norm,
          "Fused residual + RMSNorm (in-place)");

    // Linear
    m.def("linear", &py_linear, "ixformer GEMM (linear/linear_ex auto-select)");

    // RoPE
    m.def("rotary_embedding", &py_rotary_embedding, "Rotary position embedding");

    // KV Cache
    m.def("reshape_and_cache", &py_reshape_and_cache,
          "Reshape K/V into paged cache");

    // Attention
    m.def("flash_attn_prefill", &py_flash_attn_prefill,
          "Flash attention (prefill, unpadded, block tables)");
    m.def("paged_attention", &py_paged_attention,
          "Paged attention (decode)");

    // MoE
    m.def("moe_topk_softmax", &py_moe_topk_softmax,
          "MoE topk + softmax gating");
    m.def("moe_gen_idx", &py_moe_gen_idx,
          "MoE compute token→expert index mapping");
    m.def("moe_expand_input", &py_moe_expand_input,
          "MoE expand input by topk");
    m.def("moe_group_gemm", &py_moe_group_gemm,
          "MoE group GEMM (w16a16)");
    m.def("moe_combine_result", &py_moe_combine_result,
          "MoE reduce expert outputs (weighted sum)");
}
