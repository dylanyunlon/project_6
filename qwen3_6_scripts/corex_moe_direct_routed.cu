#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

namespace {

constexpr int kExperts = 256;
constexpr int kTopK = 8;
constexpr int kHidden = 2048;
constexpr int kIntermediate = 128;
constexpr int kW13Rows = 2 * kIntermediate;
constexpr int kThreads = 256;
constexpr int kWarpSize = 32;

__device__ inline float warp_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__global__ void direct_w13_kernel(
    const __half* input, const __half* w13, const int64_t* expert_ids,
    __half* gate_up) {
  const int warp =
      (static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x) / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  if (warp >= kTopK * kW13Rows) {
    return;
  }

  const int slot = warp / kW13Rows;
  const int local_row = warp - slot * kW13Rows;
  const int64_t expert = expert_ids[slot];
  const int64_t weight_row =
      (expert * kW13Rows + local_row) * static_cast<int64_t>(kHidden);
  const __half2* input2 = reinterpret_cast<const __half2*>(input);
  const __half2* weight2 =
      reinterpret_cast<const __half2*>(w13 + weight_row);
  float sum = 0.0f;
  for (int index = lane; index < kHidden / 2; index += kWarpSize) {
    const __half2 x = input2[index];
    const __half2 weight = weight2[index];
    sum = fmaf(__half2float(weight.x), __half2float(x.x), sum);
    sum = fmaf(__half2float(weight.y), __half2float(x.y), sum);
  }
  sum = warp_sum(sum);
  if (lane == 0) {
    gate_up[warp] = __float2half_rn(sum);
  }
}

__global__ void direct_w2_reduce_kernel(
    const __half* activated, const __half* w2, const int64_t* expert_ids,
    const __half* weights, __half* output) {
  const int warp =
      (static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x) / kWarpSize;
  const int lane = threadIdx.x & (kWarpSize - 1);
  if (warp >= kHidden) {
    return;
  }

  float weighted_sum = 0.0f;
#pragma unroll
  for (int slot = 0; slot < kTopK; ++slot) {
    const int64_t expert = expert_ids[slot];
    const int64_t weight_row =
        (expert * kHidden + warp) * static_cast<int64_t>(kIntermediate);
    const __half2* activation2 = reinterpret_cast<const __half2*>(
        activated + slot * kIntermediate);
    const __half2* weight2 =
        reinterpret_cast<const __half2*>(w2 + weight_row);
    float expert_sum = 0.0f;
    for (int index = lane; index < kIntermediate / 2;
         index += kWarpSize) {
      const __half2 x = activation2[index];
      const __half2 weight = weight2[index];
      expert_sum = fmaf(
          __half2float(weight.x), __half2float(x.x), expert_sum);
      expert_sum = fmaf(
          __half2float(weight.y), __half2float(x.y), expert_sum);
    }
    expert_sum = warp_sum(expert_sum);
    if (lane == 0) {
      const __half expert_half = __float2half_rn(expert_sum);
      const __half product = __hmul(expert_half, weights[slot]);
      weighted_sum += __half2float(product);
    }
  }
  if (lane == 0) {
    output[warp] = __float2half_rn(weighted_sum);
  }
}

void check_half_cuda(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kFloat16,
              name, " must have dtype float16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_ids(const torch::Tensor& expert_ids) {
  TORCH_CHECK(expert_ids.is_cuda() && expert_ids.is_contiguous(),
              "expert_ids must be a contiguous CUDA tensor");
  TORCH_CHECK(expert_ids.scalar_type() == torch::kInt64,
              "expert_ids must have dtype int64");
  TORCH_CHECK(expert_ids.dim() == 1 && expert_ids.numel() == kTopK,
              "expert_ids must have shape (8,)");
}

}  // namespace

torch::Tensor direct_w13(const torch::Tensor& input,
                         const torch::Tensor& w13,
                         const torch::Tensor& expert_ids) {
  check_half_cuda(input, "input");
  check_half_cuda(w13, "w13");
  check_ids(expert_ids);
  TORCH_CHECK(input.dim() == 2 && input.size(0) == 1
                  && input.size(1) == kHidden,
              "input must have shape (1, 2048)");
  TORCH_CHECK(w13.dim() == 3 && w13.size(0) == kExperts
                  && w13.size(1) == kW13Rows
                  && w13.size(2) == kHidden,
              "w13 must have shape (256, 256, 2048)");

  auto output = torch::empty({kTopK, kW13Rows}, input.options());
  constexpr int kWarpsPerBlock = kThreads / kWarpSize;
  constexpr int kBlocks =
      (kTopK * kW13Rows + kWarpsPerBlock - 1) / kWarpsPerBlock;
  direct_w13_kernel<<<kBlocks, kThreads, 0,
                      at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(w13.data_ptr<at::Half>()),
      expert_ids.data_ptr<int64_t>(),
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor direct_w2_reduce(const torch::Tensor& activated,
                               const torch::Tensor& w2,
                               const torch::Tensor& expert_ids,
                               const torch::Tensor& weights) {
  check_half_cuda(activated, "activated");
  check_half_cuda(w2, "w2");
  check_half_cuda(weights, "weights");
  check_ids(expert_ids);
  TORCH_CHECK(activated.dim() == 2 && activated.size(0) == kTopK
                  && activated.size(1) == kIntermediate,
              "activated must have shape (8, 128)");
  TORCH_CHECK(w2.dim() == 3 && w2.size(0) == kExperts
                  && w2.size(1) == kHidden
                  && w2.size(2) == kIntermediate,
              "w2 must have shape (256, 2048, 128)");
  TORCH_CHECK(weights.dim() == 1 && weights.numel() == kTopK,
              "weights must have shape (8,)");

  auto output = torch::empty({1, kHidden}, activated.options());
  constexpr int kWarpsPerBlock = kThreads / kWarpSize;
  constexpr int kBlocks =
      (kHidden + kWarpsPerBlock - 1) / kWarpsPerBlock;
  direct_w2_reduce_kernel<<<kBlocks, kThreads, 0,
                            at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __half*>(activated.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(w2.data_ptr<at::Half>()),
      expert_ids.data_ptr<int64_t>(),
      reinterpret_cast<const __half*>(weights.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("w13", &direct_w13,
             "Direct selected-expert FP16 W13 matvec");
  module.def("w2_reduce", &direct_w2_reduce,
             "Direct selected-expert W2 matvec and routed reduction");
}
