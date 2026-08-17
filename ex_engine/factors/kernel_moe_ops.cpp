// ex_engine/factors/kernel_moe_ops.cpp
//
// Layer 5: MoE kernel-level operations
//
// Upstream parallel: kernels/ilu/fused_moe.cpp (99 lines)
//   → moe_active_topk() → infer::topk_softmax
//   → moe_gen_idx() → infer::moe_compute_token_index_api
//   → moe_expand_input() → infer::moe_expand_input
//   → moe_combine_result() → infer::moe_output_reduce_sum
//
// Each function is a thin dispatch wrapper. On BI-V100, the call goes:
//   this .cpp → ixformer::infer::* (libixformer.so from base image)
//   OR
//   this .cpp → EX factor .so (our replacement for missing ops)
//
// The code here is intentionally minimal — the real logic lives in
// the CUDA kernels (Layer 8-10). This layer only handles:
//   1. Tensor type coercion (fp16 → fp32 for routing)
//   2. Output tensor allocation
//   3. Call-through to infer:: namespace

#include "ilu_ops_api.h"
#include "ixformer.h"

namespace xllm {
namespace kernel {
namespace ilu {

// =====================================================================
// Step 1: moe_active_topk — Router dispatch
// =====================================================================
// Upstream: kernels/ilu/fused_moe.cpp::moe_active_topk
// Converts input to float32, allocates output tensors, calls topk_softmax.
// The topk_softmax kernel (Layer 8) does fused softmax + topk in one pass.

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
    const std::optional<torch::Tensor>& e_score_correction_bias) {

  // Cast to float32 for numerical stability (half softmax overflows)
  torch::Tensor input_f32 = input.to(torch::kFloat32);

  // Allocate output tensors — matches upstream exactly
  auto reduce_weight = torch::empty(
      {input.size(0), topk},
      torch::dtype(torch::kFloat).device(input.device()));
  auto topk_indices = torch::empty(
      {input.size(0), topk},
      torch::dtype(torch::kInt32).device(input.device()));
  auto token_expert_indices = torch::empty(
      {input.size(0), topk},
      torch::dtype(torch::kInt32).device(input.device()));

  // Dispatch to ixformer::infer::topk_softmax
  // This calls the CUDA kernel in Layer 8 (moe_topk_softmax_kernels.cuh)
  ixformer::infer::topk_softmax(
      reduce_weight, topk_indices, token_expert_indices, input_f32,
      /*renormalize=*/false);

  // Renormalize weights (upstream does this post-kernel)
  if (normalize) {
    auto weight_sum = reduce_weight.sum(-1);
    reduce_weight = reduce_weight / weight_sum.unsqueeze(-1);
  }

  return std::make_tuple(reduce_weight, topk_indices);
}

// =====================================================================
// Step 2: moe_gen_idx — Permutation index generation
// =====================================================================
// Upstream: kernels/ilu/fused_moe.cpp::moe_gen_idx
// Calls the 3-phase CUDA kernel (histogram → prefix_sum → place)
// Returns {src_dst, dst_src, expert_sizes, expert_sizes_cumsum}

std::vector<torch::Tensor> moe_gen_idx(
    torch::Tensor& expert_id,
    int64_t expert_num) {

  auto src_dst = expert_id.new_empty({expert_id.numel()});
  auto dst_src = torch::empty_like(src_dst);
  auto expert_sizes_gpu = expert_id.new_empty({expert_num});
  auto expert_sizes_gpu_cumsum = expert_id.new_zeros({expert_id.numel() + 1});

  // Dispatch to ixformer::infer::moe_compute_token_index_api
  // This calls the 3-phase CUDA kernel in Layer 9
  ixformer::infer::moe_compute_token_index_api(
      expert_id, src_dst, dst_src, expert_sizes_gpu,
      /*expert_mask=*/std::nullopt,
      /*expert_sizes_cpu=*/std::nullopt,
      /*expand_tokens_gpu=*/std::nullopt,
      0, expert_num, expert_num);

  expert_sizes_gpu_cumsum = expert_sizes_gpu.cumsum(-1);

  return {src_dst, dst_src, expert_sizes_gpu, expert_sizes_gpu_cumsum};
}

// =====================================================================
// Step 3: moe_expand_input — Token gather/scatter
// =====================================================================
// Upstream: kernels/ilu/fused_moe.cpp::moe_expand_input
// Reorders tokens from natural order to expert-grouped order.

torch::Tensor moe_expand_input(
    const torch::Tensor& input,
    const torch::Tensor& gather_index,
    const torch::Tensor& combine_idx,
    int64_t topk) {

  int64_t dst_tokens = input.size(0) * topk;
  auto output = input.new_empty({dst_tokens, input.size(1)});

  // Dispatch to ixformer::infer::moe_expand_input
  ixformer::infer::moe_expand_input(
      output, input, combine_idx, gather_index, dst_tokens, topk);

  return output;
}

// =====================================================================
// Step 7: moe_combine_result — Weighted combine
// =====================================================================
// Upstream: kernels/ilu/fused_moe.cpp::moe_combine_result
// Reorders from expert-sorted back to token order with weighted sum.
// Calls the CUDA kernel in Layer 10 (moe_combine.cu)

torch::Tensor moe_combine_result(
    torch::Tensor& input,
    torch::Tensor& weight) {

  input = input.view({-1, weight.size(1), input.size(1)});
  auto output = input.new_empty({input.size(0), input.size(2)});

  // Dispatch to ixformer::infer::moe_output_reduce_sum
  ixformer::infer::moe_output_reduce_sum(
      output, input, weight,
      /*mask=*/std::nullopt,
      /*extra_residual=*/std::nullopt,
      /*scaling_factor=*/1.0);

  return output;
}

}  // namespace ilu
}  // namespace kernel
}  // namespace xllm
