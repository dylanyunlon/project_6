// cccl_moe_sort_scatter.cu — Block-level CUB MoE token dispatch
//
// Uses CUB BlockScan (already proven on BI-V100 in corex_moe_index_combine.cu)
// for histogram + prefix_sum + scatter. No device-level CUB API (conflicts
// with corex's thrust/complex.h on CUDA 10.2).
//
// Three kernels (same as corex_moe_index_combine but with CUB BlockRadixSort
// for the scatter step):
//   1. histogram — atomicAdd per expert
//   2. prefix_sum — CUB BlockScan ExclusiveSum
//   3. place — atomicAdd scatter into sorted positions
//
// Build: torch.utils.cpp_extension.load(
//   name="cccl_moe_sort_scatter",
//   sources=["cccl_moe_sort_scatter.cu"],
//   extra_cuda_cflags=["-O3"],
// )

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cub/block/block_scan.cuh>

// ========================================================================
// Block-level CUB kernels (proven on BI-V100 corex clang++)
// Same pattern as corex_moe_index_combine.cu
// ========================================================================

constexpr int32_t kBlock = 256;

__global__ void moe_histogram_kernel(
    const int32_t* __restrict__ expert_id,
    int32_t* __restrict__ expert_sizes,
    int64_t num_elements,
    int32_t num_experts) {
  int64_t tid = int64_t(blockIdx.x) * kBlock + threadIdx.x;
  if (tid < num_elements) {
    int32_t eid = expert_id[tid];
    if (eid >= 0 && eid < num_experts) {
      atomicAdd(&expert_sizes[eid], 1);
    }
  }
}

__global__ void moe_prefix_sum_kernel(
    const int32_t* __restrict__ expert_sizes,
    int32_t* __restrict__ expert_offsets,
    int32_t num_experts,
    int64_t* __restrict__ total_out) {
  using BlockScan = cub::BlockScan<int32_t, 256>;
  __shared__ typename BlockScan::TempStorage s_scan;

  int32_t val = (threadIdx.x < num_experts) ? expert_sizes[threadIdx.x] : 0;
  int32_t offset;
  BlockScan(s_scan).ExclusiveSum(val, offset);
  __syncthreads();

  if (threadIdx.x < num_experts) {
    expert_offsets[threadIdx.x] = offset;
  }
  if (threadIdx.x == 0 && total_out != nullptr) {
    // Last thread's offset + val = total
    *total_out = offset + val;
  }
}

__global__ void moe_place_kernel(
    const int32_t* __restrict__ expert_id,
    int32_t* __restrict__ expert_offsets,  // modified in-place by atomicAdd
    int32_t* __restrict__ dst_src,
    int32_t* __restrict__ src_dst,
    int64_t num_elements,
    int32_t num_experts) {
  int64_t flat_idx = int64_t(blockIdx.x) * kBlock + threadIdx.x;
  if (flat_idx >= num_elements) return;

  int32_t eid = expert_id[flat_idx];
  if (eid < 0 || eid >= num_experts) return;

  int32_t pos = atomicAdd(&expert_offsets[eid], 1);
  dst_src[pos] = static_cast<int32_t>(flat_idx);
  src_dst[flat_idx] = pos;
}


// Same proven 3-kernel approach as corex_moe_index_combine.cu but with
// an additional inverse-scatter output for full compatibility.

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
moe_sort_scatter(const torch::Tensor& expert_id, int64_t num_experts) {
  TORCH_CHECK(expert_id.is_cuda(), "expert_id must be on CUDA");
  auto stream = at::cuda::getCurrentCUDAStream();
  int64_t N = expert_id.numel();
  int32_t E = static_cast<int32_t>(num_experts);

  auto expert_id_i32 = expert_id.to(torch::kInt32).contiguous();
  auto opt_i32 = expert_id_i32.options();

  auto expert_sizes = torch::zeros({num_experts}, opt_i32);
  auto expert_offsets = torch::empty({num_experts}, opt_i32);
  auto dst_src = torch::empty({N}, opt_i32);
  auto src_dst = torch::empty({N}, opt_i32);

  int64_t grid = (N + kBlock - 1) / kBlock;

  // Step 1: histogram
  moe_histogram_kernel<<<grid, kBlock, 0, stream>>>(
      expert_id_i32.data_ptr<int32_t>(),
      expert_sizes.data_ptr<int32_t>(),
      N, E);

  // Step 2: prefix sum (CUB BlockScan)
  moe_prefix_sum_kernel<<<1, kBlock, 0, stream>>>(
      expert_sizes.data_ptr<int32_t>(),
      expert_offsets.data_ptr<int32_t>(),
      E, nullptr);

  // Step 3: scatter — place each token into its sorted position
  moe_place_kernel<<<grid, kBlock, 0, stream>>>(
      expert_id_i32.data_ptr<int32_t>(),
      expert_offsets.data_ptr<int32_t>(),
      dst_src.data_ptr<int32_t>(),
      src_dst.data_ptr<int32_t>(),
      N, E);

  return std::make_tuple(src_dst, dst_src, expert_sizes);
}


// ========================================================================
// pybind
// ========================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_sort_scatter", &moe_sort_scatter,
        "CUB DeviceRadixSort-based MoE token dispatch "
        "(sort expert_ids, compute offsets+sizes)");
}
