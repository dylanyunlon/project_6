#!/bin/bash
# probe_warp64.sh — Verify warp_size=64 behavior on BI-V100
# and test siboehm kernel 10 with WARPSIZE=64

echo "=== 1. Warp behavior probe ==="
cat > /tmp/probe_warp.cu << 'CUDA'
#include <cstdio>
#include <cuda_runtime.h>

__global__ void probe_warp_info() {
    if (threadIdx.x == 0) {
        printf("warpSize (built-in) = %d\n", warpSize);
    }

    // Each thread reports its warp ID and lane ID
    int warp_id = threadIdx.x / warpSize;
    int lane_id = threadIdx.x % warpSize;

    // Only first thread of each warp reports
    if (lane_id == 0) {
        printf("  thread %3d: warp_id=%d, lane_id=%d, warpSize=%d\n",
               threadIdx.x, warp_id, lane_id, warpSize);
    }
}

__global__ void test_syncwarp_64() {
    // Test if __syncwarp with 64-bit mask works
    // On NVIDIA: mask is 32-bit (unsigned int)
    // On BI-V100 with warp_size=64: need 64-bit mask?

    __shared__ int data[128];
    int tid = threadIdx.x;
    int warp_id = tid / warpSize;
    int lane = tid % warpSize;

    // Write
    data[tid] = tid * 2;

    // Try standard __syncwarp()
    __syncwarp();

    // Read neighbor's data within the warp
    int neighbor = (lane + 1) % warpSize + warp_id * warpSize;
    int val = data[neighbor];

    if (tid == 0) {
        printf("__syncwarp() works: thread 0 read thread 1's value = %d (expected 2)\n", val);
    }
    if (tid == 63) {
        printf("__syncwarp() works: thread 63 read thread 0's value = %d (expected 0)\n",
               data[warp_id * warpSize]);
    }
}

__global__ void test_shfl_down_64() {
    int tid = threadIdx.x;
    int lane = tid % warpSize;
    float val = (float)tid;

    // __shfl_down_sync with full mask
    // On NVIDIA: mask = 0xFFFFFFFF (32 bits)
    // On BI-V100: what mask for 64 threads?
    float result = __shfl_down_sync(0xFFFFFFFF, val, 1);

    if (tid == 0) {
        printf("shfl_down(0xFFFFFFFF, tid=0, delta=1) = %.0f (expected 1.0)\n", result);
    }
    if (tid == 31) {
        printf("shfl_down(0xFFFFFFFF, tid=31, delta=1) = %.0f (expected 32.0 if warp=64, else 31.0)\n", result);
    }
    if (tid == 32) {
        printf("shfl_down(0xFFFFFFFF, tid=32, delta=1) = %.0f (expected 33.0)\n", result);
    }
    if (tid == 63) {
        printf("shfl_down(0xFFFFFFFF, tid=63, delta=1) = %.0f (expected 63.0 if wraps, else 0)\n", result);
    }
}

int main() {
    printf("--- Warp info (128 threads, 1 block) ---\n");
    probe_warp_info<<<1, 128>>>();
    cudaDeviceSynchronize();

    printf("\n--- __syncwarp test (128 threads) ---\n");
    test_syncwarp_64<<<1, 128>>>();
    cudaDeviceSynchronize();

    printf("\n--- __shfl_down_sync test (64 threads) ---\n");
    test_shfl_down_64<<<1, 64>>>();
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("\nCUDA error: %s\n", cudaGetErrorString(err));
    } else {
        printf("\nAll warp probes completed successfully.\n");
    }
    return 0;
}
CUDA

/usr/local/corex/bin/clang++ --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex \
    -I/usr/local/corex/include -L/usr/local/corex/lib64 -lcudart \
    /tmp/probe_warp.cu -o /tmp/probe_warp 2>&1

if [ -f /tmp/probe_warp ]; then
    echo "Compile: SUCCESS"
    /tmp/probe_warp
else
    echo "Compile: FAILED"
fi

echo ""
echo "=== 2. Kernel 10 warp tiling compile test (WARPSIZE=64) ==="
cat > /tmp/test_k10_warp64.cu << 'CUDA'
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// BI-V100 warp size
const int WARPSIZE = 64;

// Minimal kernel 10 structure from siboehm, with WARPSIZE=64
// Just test that the indexing compiles and the launch config is valid
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    hgemmWarptiling(int M, int N, int K, float alpha,
                    const __half *A, const __half *B, float beta, __half *C) {
  const uint warpIdx = threadIdx.x / WARPSIZE;
  const uint warpCol = warpIdx % (BN / WN);
  const uint warpRow = warpIdx / (BN / WN);

  constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER;
  constexpr uint WSUBN = WN / WNITER;

  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

  __shared__ __half As[BM * BK];
  __shared__ __half Bs[BK * BN];

  // Just write a marker to verify the kernel launched
  if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
      C[0] = __float2half(42.0f);
  }
}

int main() {
    // Config adapted for WARPSIZE=64:
    // 128 threads = 2 warps of 64
    // BM=128, BN=128 → 2 warps need to cover 128x128
    // WM=64, WN=128 → 2 warps: (128/128)*(128/64) = 1*2 = 2 warps ✓
    const int NUM_THREADS = 128;
    const int BM = 128, BN = 128, BK = 16;
    const int WM = 64, WN = 128;
    const int WNITER = 4;
    const int TM = 4, TN = 4;

    // Verify constraints
    constexpr int NUM_WARPS = NUM_THREADS / 64;  // 2
    // (BN/WN) * (BM/WM) should == NUM_WARPS
    printf("NUM_WARPS=%d, (BN/WN)*(BM/WM)=%d\n", NUM_WARPS, (BN/WN)*(BM/WM));

    constexpr int WMITER_check = (WM * WN) / (64 * TM * TN * WNITER);
    printf("WMITER=%d\n", WMITER_check);
    printf("WSUBM=%d, WSUBN=%d\n", WM/WMITER_check, WN/WNITER);

    // Threads in warp: WSUBN/TN * WSUBM/TM should <= 64
    int WSUBM = WM / WMITER_check;
    int WSUBN = WN / WNITER;
    printf("threads_per_warp_check: (WSUBN/TN)*(WSUBM/TM) = %d (need <=64)\n",
           (WSUBN/TN) * (WSUBM/TM));

    // Allocate tiny test
    __half *d_A, *d_B, *d_C;
    int M=128, N=128, K=16;
    cudaMalloc(&d_A, M*K*sizeof(__half));
    cudaMalloc(&d_B, K*N*sizeof(__half));
    cudaMalloc(&d_C, M*N*sizeof(__half));
    cudaMemset(d_C, 0, M*N*sizeof(__half));

    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 block(NUM_THREADS);

    hgemmWarptiling<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<grid, block>>>(M, N, K, 1.0f, d_A, d_B, 0.0f, d_C);
    cudaDeviceSynchronize();

    __half h_c;
    cudaMemcpy(&h_c, d_C, sizeof(__half), cudaMemcpyDeviceToHost);
    printf("C[0] = %.1f (expected 42.0)\n", __half2float(h_c));

    cudaError_t err = cudaGetLastError();
    printf("CUDA error: %s\n", cudaGetErrorString(err));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
CUDA

/usr/local/corex/bin/clang++ --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex \
    -I/usr/local/corex/include -L/usr/local/corex/lib64 -lcudart \
    /tmp/test_k10_warp64.cu -o /tmp/test_k10_warp64 2>&1

if [ -f /tmp/test_k10_warp64 ]; then
    echo "Compile: SUCCESS"
    /tmp/test_k10_warp64
else
    echo "Compile: FAILED"
fi
