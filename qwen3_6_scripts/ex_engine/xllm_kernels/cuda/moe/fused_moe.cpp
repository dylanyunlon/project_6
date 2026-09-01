/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "cuda_ops_api.h"
#include "utils.h"



namespace xllm::kernel::cuda {

torch::Tensor cutlass_fused_moe(
    const torch::Tensor& input,                   // [num_tokens, hidden]
    const torch::Tensor& token_selected_experts,  // [num_tokens, top_k]
    const torch::Tensor& token_final_scales,      // [num_tokens, top_k]
    const torch::Tensor&
        fc1_expert_weights,  // [num_experts, inter_dim, hidden]
    const torch::Tensor&
        fc2_expert_weights,  // [num_experts, hidden, inter_dim]
    torch::ScalarType output_dtype,
    const std::vector<torch::Tensor>& quant_scales,
    int32_t tp_size,
    int32_t tp_rank,
    int32_t ep_size,
    int32_t ep_rank,
    int32_t cluster_size,
    int32_t cluster_rank,
    const std::optional<torch::Tensor>& fc1_expert_biases,
    const std::optional<torch::Tensor>& fc2_expert_biases,
    const std::optional<torch::Tensor>& input_sf,
    const std::optional<torch::Tensor>& swiglu_alpha,
    const std::optional<torch::Tensor>& swiglu_beta,
    const std::optional<torch::Tensor>& swiglu_limit,
    const std::optional<torch::Tensor>& output,
    bool enable_alltoall,
    bool use_deepseek_fp8_block_scale,
    bool use_w4_group_scaling,
    bool use_mxfp8_act_scaling,
    bool min_latency_mode,
    bool use_packed_weights,
    int32_t tune_max_num_tokens,
    ActivationType activation_type) {
  int64_t num_tokens = input.size(0);
  int64_t hidden_size = fc2_expert_weights.size(1);
  int64_t inter_dim = fc1_expert_weights.size(1);
  int64_t top_k = token_selected_experts.size(1);

  int64_t num_rows = num_tokens;
  if (min_latency_mode) {
    num_rows *= fc2_expert_weights.size(0);
  }

  torch::Tensor result_output;
  if (output.has_value() && output.value().defined()) {
    result_output = output.value();
  } else {
    result_output = torch::zeros({num_rows, hidden_size},
                                 input.options().dtype(output_dtype));
  }

  if (Platform::is_support_ivcore10()) {
    // BI-V100 path: per-token expert gather + matmul + SiLU-gate + matmul
    // This replaces the tvm ffi CUTLASS path with native PyTorch ops.
    for (int64_t t = 0; t < num_tokens; ++t) {
      auto token = input[t].unsqueeze(0);  // [1, hidden]
      torch::Tensor accum = torch::zeros({1, hidden_size},
                                         input.options().dtype(output_dtype));
      for (int64_t k = 0; k < top_k; ++k) {
        int64_t expert_id = token_selected_experts[t][k].item<int64_t>();
        float scale = token_final_scales[t][k].item<float>();

        // gate_up = token @ fc1[expert].T  → [1, inter_dim]
        auto gate_up = torch::mm(token, fc1_expert_weights[expert_id].t());
        if (fc1_expert_biases.has_value()) {
          gate_up = gate_up + fc1_expert_biases.value()[expert_id];
        }

        // SwiGLU: split into gate and up, apply silu(gate) * up
        torch::Tensor act;
        if (activation_type == ActivationType::SWIGLU ||
            activation_type == ActivationType::SWIGLU_BIAS) {
          auto chunks = gate_up.chunk(2, /*dim=*/-1);
          act = torch::silu(chunks[0]) * chunks[1];
        } else if (activation_type == ActivationType::SILU) {
          act = torch::silu(gate_up);
        } else if (activation_type == ActivationType::GELU) {
          act = torch::gelu(gate_up);
        } else {
          act = gate_up;  // identity
        }

        // down = act @ fc2[expert].T  → [1, hidden]
        auto down = torch::mm(act, fc2_expert_weights[expert_id].t());
        if (fc2_expert_biases.has_value()) {
          down = down + fc2_expert_biases.value()[expert_id];
        }

        accum += down.to(output_dtype) * scale;
      }
      result_output[t] = accum.squeeze(0);
    }
    return result_output;
  }

  // Original NVIDIA GPU path (sm90/sm100/sm120) via tvm ffi
  TORCH_CHECK(false,
      "cutlass_fused_moe: no supported platform. "
      "BI-V100 should use ivcore10 path above; "
      "NVIDIA GPUs require sm90+.");
  return result_output;
}
}  // namespace xllm::kernel::cuda