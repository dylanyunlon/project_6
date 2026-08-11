/* ilu_ops_api.h — Standalone header for project_6 ex_engine.
 *
 * Adapted from xllm/core/kernels/ilu/ilu_ops_api.h.
 * Removes xllm-internal deps (glog, kernels/kernels.h, framework/*).
 * Only requires: torch, ixformer.h (ixformer::infer namespace).
 */
#pragma once

#include <torch/all.h>
#include <optional>
#include <iostream>
#include <stdexcept>

#include "ixformer.h"

using namespace ixformer;

/* ---- Minimal LOG(FATAL) replacement ------------------------------------ */
#ifndef LOG
struct FatalLogStream {
  std::ostringstream ss;
  [[noreturn]] ~FatalLogStream() noexcept(false) {
    std::cerr << ss.str() << std::endl;
    throw std::runtime_error(ss.str());
  }
  template <typename T> FatalLogStream& operator<<(const T& v) {
    ss << v; return *this;
  }
};
#define LOG(level) FatalLogStream()
#endif

namespace xllm::kernel::ilu {

void apply_rope_pos_ids_cos_sin_cache(torch::Tensor& query,
                                      torch::Tensor& key,
                                      torch::Tensor& cos_sin_cache,
                                      torch::Tensor& positions,
                                      bool interleave);

void act_and_mul(torch::Tensor out,
                 torch::Tensor input,
                 const std::string& act_mode);

void reshape_paged_cache(
    torch::Tensor& key,
    std::optional<torch::Tensor>& value,
    torch::Tensor& key_cache,
    std::optional<torch::Tensor>& value_cache,
    torch::Tensor& slot_mapping);

void batch_prefill(torch::Tensor& query,
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

void batch_decode(torch::Tensor& query,
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

void residual_layer_norm(torch::Tensor& input,
                         torch::Tensor& output,
                         std::optional<torch::Tensor>& residual,
                         torch::Tensor& weight,
                         std::optional<torch::Tensor>& bias,
                         std::optional<torch::Tensor>& residual_out,
                         double eps);

void rms_norm(torch::Tensor& output,
              torch::Tensor& input,
              torch::Tensor& weight,
              double eps);

torch::Tensor matmul(torch::Tensor a,
                     torch::Tensor b,
                     std::optional<torch::Tensor> bias);

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

std::vector<torch::Tensor> moe_gen_idx(torch::Tensor& expert_id,
                                       int64_t expert_num);

torch::Tensor moe_expand_input(const torch::Tensor& input,
                               const torch::Tensor& gather_index,
                               const torch::Tensor& combine_idx,
                               int64_t topk);

torch::Tensor group_gemm(torch::Tensor& input,
                         torch::Tensor& weight,
                         torch::Tensor& tokens_per_experts,
                         const std::optional<torch::Tensor>& dst_to_src,
                         torch::Tensor& output);

torch::Tensor moe_combine_result(torch::Tensor& input, torch::Tensor& weight);

}  // namespace xllm::kernel::ilu
