// ex_engine/factors/kernel_elem_ops.cpp
//
// Layer 6: Element-wise kernel dispatch wrappers
//
// Upstream parallel: kernels/ilu/activation.cpp (30 lines)
//                  + kernels/ilu/norm.cpp (45 lines)
//                  + kernels/ilu/rope.cpp (20 lines)
//                  + kernels/ilu/group_gemm.cpp (25 lines)
//                  + kernels/ilu/matmul.cpp (~15 lines)
//
// Total upstream: ~135 lines across 5 files.
// Each function is a 3-5 line dispatch wrapper that calls ixformer::infer.
//
// These ops are PRESENT in the base image's ixformer — they don't need
// EX factor replacement. But they must be in the call chain because:
//   - Activation is Step 5 of the MoE pipeline (called between GEMM1 and GEMM2)
//   - RMSNorm is called before/after every decoder layer (2× per layer × 64 layers)
//   - RoPE is called once per attention layer (32 full attention + 4 GDN = 36)
//   - Group GEMM is Steps 4+6 (called twice per MoE layer × 64 layers)
//
// The presence in ixformer is confirmed by:
//   cat_files/symbol_dumps (nm -D output from real device)
//   SYSTEM_DESIGN.md PRESENT list

#include "ilu_ops_api.h"
#include "ixformer.h"

using namespace ixformer;

namespace xllm {
namespace kernel {
namespace ilu {

// =====================================================================
// Activation: silu_and_mul
// =====================================================================
// Upstream: kernels/ilu/activation.cpp::act_and_mul
// BI-V100 ixformer: PRESENT (silu_and_mul confirmed in symbol dump)
//
// Input: (tokens, 2 × intermediate_size)  — gate + up projections concatenated
// Output: (tokens, intermediate_size)     — silu(gate) × up
//
// For Qwen3.5: intermediate_size = 18944 / TP4 = 4736
// Each call processes 4736 × 2 = 9472 half values per token.
// At 200 tokens/batch decode: 200 × 9472 × 2 bytes = 3.6 MB bandwidth.

void act_and_mul(
    torch::Tensor out,
    torch::Tensor input,
    const std::string& act_mode) {

  if (act_mode == "silu") {
    infer::silu_and_mul(input, out);
  } else {
    // gelu_tanh_and_mul is MISSING from ixformer on BI-V100.
    // The EX factor system provides this as EX_FACTOR_GELU_TANH_MUL (id=3).
    // For now, fallback to PyTorch.
    LOG(FATAL) << "Unsupported act mode: " << act_mode
               << ", only silu is available via ixformer on BI-V100. "
               << "Use EX factor 3 for gelu_tanh.";
  }
}

// =====================================================================
// RMSNorm: rms_norm + residual_layer_norm
// =====================================================================
// Upstream: kernels/ilu/norm.cpp
// BI-V100 ixformer: PRESENT (rms_norm, fused_add_rms_norm confirmed)
//
// rms_norm: out = x × rsqrt(mean(x²) + eps) × weight
// residual_layer_norm: fused residual add + rms_norm
//
// Called 2× per decoder layer (pre-attention + post-attention norm).
// 64 layers × 2 = 128 calls per forward pass.

void rms_norm(
    torch::Tensor& output,
    torch::Tensor& input,
    torch::Tensor& weight,
    double eps) {

  std::optional<torch::Tensor> fused_bias = std::nullopt;
  infer::rms_norm(input, weight, output, fused_bias, eps);
}

void residual_layer_norm(
    torch::Tensor& input,
    torch::Tensor& output,
    std::optional<torch::Tensor>& residual,
    torch::Tensor& weight,
    std::optional<torch::Tensor>& bias,
    std::optional<torch::Tensor>& residual_out,
    double eps) {

  auto residual_ = residual.value_or(torch::zeros_like(input));
  torch::Tensor residual_out_ = residual_out.value_or(torch::zeros_like(input));
  infer::residual_rms_norm(
      input, residual_, weight, output, residual_out_,
      bias, /*alpha=*/1.0, eps, /*is_post=*/false);
}

// =====================================================================
// RoPE: Rotary Position Embedding
// =====================================================================
// Upstream: kernels/ilu/rope.cpp
// BI-V100 ixformer: PRESENT (vllm_rotary_embedding_neox confirmed)
//
// Applies cosine-sine rotation to query and key tensors.
// Called once per attention layer per forward pass.
// Qwen3.5: 36 attention layers (32 full + 4 GDN).

void apply_rope_pos_ids_cos_sin_cache(
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& cos_sin_cache,
    torch::Tensor& positions,
    bool interleave) {

  const int64_t head_size = cos_sin_cache.size(-1);
  // is_neox = !interleave (NeoX-style = non-interleaved)
  infer::xllm_rotary_embedding(
      positions, query, key, head_size, cos_sin_cache, !interleave);
}

// =====================================================================
// Group GEMM: Batched matrix multiplication for MoE experts
// =====================================================================
// Upstream: kernels/ilu/group_gemm.cpp
// BI-V100 ixformer: PRESENT (moe_w16a16_group_gemm confirmed)
//
// Performs A × B^T for each expert group simultaneously.
// tokens_per_experts defines the row count per group.
// Called twice per MoE layer: once for w13 (gate+up), once for w2 (down).
//
// For Qwen3.5 with TP4:
//   w13: (16 experts_local, 9472, 3584) — 16 experts × [inter*2, hidden]
//   w2:  (16 experts_local, 3584, 4736) — 16 experts × [hidden, inter]
//
// This is the primary compute bottleneck on BI-V100.
// sub694 shows prompt_tok >50K requests drop to 1-3 TPS — GEMM bound.

torch::Tensor group_gemm(
    torch::Tensor& input,
    torch::Tensor& weight,
    torch::Tensor& tokens_per_experts,
    const std::optional<torch::Tensor>& dst_to_src,
    torch::Tensor& output) {

  infer::moe_w16a16_group_gemm(
      output,
      input,
      weight,
      tokens_per_experts,
      dst_to_src,
      /*bias=*/std::nullopt,
      /*format=*/"TN",
      /*persistent=*/0,
      /*output_n=*/tokens_per_experts.sum().item<int64_t>());

  return output;
}

// =====================================================================
// Reshape and cache: KV cache management
// =====================================================================
// Upstream: kernels/ilu/attention.cpp::reshape_paged_cache
// BI-V100 ixformer: PRESENT (vllm_cache_ops_reshape_and_cache)
//
// Writes new KV pairs into paged cache at the positions specified
// by slot_mapping.

void reshape_paged_cache(
    torch::Tensor& key,
    std::optional<torch::Tensor>& value,
    torch::Tensor& key_cache,
    std::optional<torch::Tensor>& value_cache,
    torch::Tensor& slot_mapping) {

  auto value_ = value.value_or(torch::Tensor());
  auto value_cache_ = value_cache.value_or(torch::Tensor());

  int64_t key_token_stride = key.stride(0);
  int64_t value_token_stride = 0;
  if (value_.defined()) {
    value_token_stride = value_.stride(0);
  }
  slot_mapping = slot_mapping.to(at::kLong);

  infer::xllm_reshape_and_cache(
      key, value_, key_cache, value_cache_,
      slot_mapping, key_token_stride, value_token_stride);
}

// =====================================================================
// Matmul: General matrix multiplication
// =====================================================================
// Upstream: kernels/ilu/matmul.cpp
// Used for linear projections (q/k/v proj, out proj, gate proj)

torch::Tensor matmul(
    torch::Tensor a,
    torch::Tensor b,
    std::optional<torch::Tensor> bias) {

  return infer::ixformer_linear_ex(a, b, bias, std::nullopt);
}

}  // namespace ilu
}  // namespace kernel
}  // namespace xllm
