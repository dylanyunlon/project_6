// cccl_moe_sort_scatter.cu — CUB DeviceRadixSort-based MoE token dispatch
//
// Replaces the 3-kernel (histogram + prefix_sum + place) approach with:
//   1. DeviceRadixSort::SortPairs — sort (expert_id, token_idx) pairs by expert_id
//   2. DeviceHistogram::HistogramEven — count tokens per expert
//   3. DeviceScan::ExclusiveSum — prefix sum for expert offsets
//
// Uses CCCL upstream headers (in cccl_preload/include/) instead of corex CUB
// to avoid BI-V100 corex CUB bugs.
//
// Build: torch.utils.cpp_extension.load(
//   name="cccl_moe_sort_scatter",
//   sources=["cccl_moe_sort_scatter.cu"],
//   extra_include_paths=["cccl_preload/include"],
//   extra_cuda_cflags=["-O3", "-DCUB_WRAPPED_NAMESPACE=cccl_moe"],
// )

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

// Use CCCL CUB, not corex CUB
#define CUB_WRAPPED_NAMESPACE cccl_moe
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/block/block_scan.cuh>

// ========================================================================
// moe_sort_scatter: sort tokens by expert_id using CUB DeviceRadixSort
//
// Input:
//   expert_id: [N] int32, each in [0, num_experts)
//   num_experts: int
//
// Output:
//   sorted_indices: [N] int32 — original token indices sorted by expert
//   expert_offsets: [num_experts+1] int32 — exclusive prefix sum
//   expert_sizes:   [num_experts] int32 — count per expert
// ========================================================================

// Small kernel to build expert_sizes from sorted keys via boundary detection
__global__ void compute_expert_boundaries(
    const int32_t* __restrict__ sorted_keys,
    int32_t* __restrict__ expert_offsets,  // [num_experts + 1]
    int64_t N,
    int32_t num_experts) {
  // Initialize all to 0
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // First pass: detect boundaries
  if (tid < N) {
    int32_t cur = sorted_keys[tid];
    if (tid == 0) {
      // First element starts expert cur
      expert_offsets[cur] = 0;
    } else {
      int32_t prev = sorted_keys[tid - 1];
      if (cur != prev) {
        expert_offsets[cur] = tid;
      }
    }
    // Last element
    if (tid == N - 1) {
      expert_offsets[num_experts] = N;
    }
  }
}

// Fill gaps in expert_offsets (experts with 0 tokens)
__global__ void fill_offset_gaps(
    int32_t* __restrict__ expert_offsets,
    int32_t num_experts) {
  // Backward fill: if expert_offsets[i] == -1, copy from next non-(-1)
  // Single thread is fine for num_experts <= 256
  if (threadIdx.x != 0) return;

  // Fill from the end
  int32_t next_offset = expert_offsets[num_experts];  // = N
  for (int i = num_experts - 1; i >= 0; --i) {
    if (expert_offsets[i] == -1) {
      expert_offsets[i] = next_offset;
    } else {
      next_offset = expert_offsets[i];
    }
  }
}

// Compute expert_sizes from expert_offsets
__global__ void compute_expert_sizes(
    const int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ expert_sizes,
    int32_t num_experts) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_experts) {
    expert_sizes[tid] = expert_offsets[tid + 1] - expert_offsets[tid];
  }
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
moe_sort_scatter(const torch::Tensor& expert_id, int64_t num_experts) {
  TORCH_CHECK(expert_id.is_cuda(), "expert_id must be on CUDA");
  auto device = expert_id.device();
  auto stream = at::cuda::getCurrentCUDAStream();
  int64_t N = expert_id.numel();
  int32_t E = static_cast<int32_t>(num_experts);

  // Ensure int32
  auto keys_in = expert_id.to(torch::kInt32).contiguous();
  auto opt_i32 = keys_in.options();

  // Create value array: [0, 1, 2, ..., N-1]
  auto vals_in = torch::arange(N, opt_i32);

  // Allocate output
  auto sorted_keys = torch::empty({N}, opt_i32);
  auto sorted_vals = torch::empty({N}, opt_i32);

  // CUB DeviceRadixSort needs temp storage
  // First query size
  size_t temp_bytes = 0;
  cccl_moe::cub::DeviceRadixSort::SortPairs(
      nullptr, temp_bytes,
      keys_in.data_ptr<int32_t>(),
      sorted_keys.data_ptr<int32_t>(),
      vals_in.data_ptr<int32_t>(),
      sorted_vals.data_ptr<int32_t>(),
      static_cast<int>(N),
      0,                      // begin_bit
      sizeof(int32_t) * 8,    // end_bit (all bits, but only need log2(E) bits)
      stream);

  // Allocate temp storage
  auto temp_storage = torch::empty({static_cast<int64_t>(temp_bytes)},
                                    torch::dtype(torch::kUInt8).device(device));

  // Sort
  cccl_moe::cub::DeviceRadixSort::SortPairs(
      temp_storage.data_ptr(), temp_bytes,
      keys_in.data_ptr<int32_t>(),
      sorted_keys.data_ptr<int32_t>(),
      vals_in.data_ptr<int32_t>(),
      sorted_vals.data_ptr<int32_t>(),
      static_cast<int>(N),
      0,
      sizeof(int32_t) * 8,
      stream);

  // Compute expert offsets via boundary detection
  // Initialize to -1
  auto expert_offsets = torch::full({num_experts + 1}, -1, opt_i32);

  int block = 256;
  int grid = (N + block - 1) / block;
  compute_expert_boundaries<<<grid, block, 0, stream>>>(
      sorted_keys.data_ptr<int32_t>(),
      expert_offsets.data_ptr<int32_t>(),
      N, E);

  // Handle empty experts
  fill_offset_gaps<<<1, 1, 0, stream>>>(
      expert_offsets.data_ptr<int32_t>(), E);

  // Compute sizes from offsets
  auto expert_sizes = torch::empty({num_experts}, opt_i32);
  int grid2 = (E + block - 1) / block;
  compute_expert_sizes<<<grid2, block, 0, stream>>>(
      expert_offsets.data_ptr<int32_t>(),
      expert_sizes.data_ptr<int32_t>(),
      E);

  // sorted_vals = the original token indices, sorted by expert
  // expert_sizes = tokens per expert
  // sorted_keys not needed by caller, but sorted_vals is "dst_src"
  //
  // Build src_dst: inverse mapping
  // src_dst[sorted_vals[i]] = i
  auto src_dst = torch::empty({N}, opt_i32);

  // Simple inverse scatter kernel
  // For now use a tiny lambda — could be another kernel
  // But actually we can do it with scatter:
  // src_dst.scatter_(0, sorted_vals.long(), torch.arange(N))
  // This is a single CUDA kernel internally

  auto arange_n = torch::arange(N, opt_i32);
  src_dst.scatter_(0, sorted_vals.to(torch::kInt64), arange_n);

  return std::make_tuple(src_dst, sorted_vals, expert_sizes);
}


// ========================================================================
// pybind
// ========================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_sort_scatter", &moe_sort_scatter,
        "CUB DeviceRadixSort-based MoE token dispatch "
        "(sort expert_ids, compute offsets+sizes)");
}
