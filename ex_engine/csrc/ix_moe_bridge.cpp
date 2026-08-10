// ix_moe_bridge.cpp — Bridge to ixformer C++ topk_softmax
//
// Problem: ixformer Python (ixformer.functions) lacks vllm_moe_topk_softmax
// Solution: Call ixformer::infer::topk_softmax() directly via C++ torch extension
//
// Source: upstream_ref/xllm/xllm/core/kernels/ilu/ixformer.h declares:
//   void topk_softmax(torch::Tensor&, torch::Tensor&, torch::Tensor&, 
//                     torch::Tensor&, bool);
// Source: upstream_ref/xllm/xllm/core/kernels/ilu/fused_moe.cpp shows usage:
//   infer::topk_softmax(reduce_weight, topk_indices, token_expert_indices, input_, false);

#include <torch/extension.h>

// Forward-declare ixformer C++ API (from ixformer.h in base image SDK)
namespace ixformer {
namespace infer {
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

}  // namespace infer
}  // namespace ixformer

// Python-callable wrappers
std::tuple<torch::Tensor, torch::Tensor> ix_moe_topk_softmax(
    torch::Tensor gating_output,  // (num_tokens, num_experts) float32
    int64_t topk,
    bool renormalize) {
  auto input = gating_output.to(torch::kFloat32).contiguous();
  int64_t num_tokens = input.size(0);
  
  auto topk_weights = torch::empty({num_tokens, topk},
      torch::dtype(torch::kFloat32).device(input.device()));
  auto topk_indices = torch::empty({num_tokens, topk},
      torch::dtype(torch::kInt32).device(input.device()));
  auto token_expert_indices = torch::empty({num_tokens, topk},
      torch::dtype(torch::kInt32).device(input.device()));
  
  ixformer::infer::topk_softmax(
      topk_weights, topk_indices, token_expert_indices, input, renormalize);
  
  // Renormalize if not done by kernel (match xllm behavior)
  if (!renormalize) {
    auto row_sum = topk_weights.sum(-1, /*keepdim=*/true);
    topk_weights = topk_weights / row_sum;
  }
  
  return std::make_tuple(topk_weights, topk_indices);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("topk_softmax", &ix_moe_topk_softmax,
        "Fused topk+softmax via ixformer C++ API (bypasses missing Python binding)",
        py::arg("gating_output"), py::arg("topk"), py::arg("renormalize") = true);
}
