#!/bin/bash
# probe_k10_perf.sh — Diagnose kernel 10 performance on BI-V100
set -eo pipefail

echo "=== Kernel 10 parameter space exploration ==="
cat > /tmp/probe_k10_perf.cu << 'CUDA'
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
const int WARPSIZE = 64;

// Minimal kernel 10 — load + compute, no frills
namespace wt {
template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB>
__device__ void loadFromGmem(int N, int K, const __half *A, const __half *B,
                             __half *As, __half *Bs, int innerRowA, int innerColA,
                             int innerRowB, int innerColB) {
  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    __half a0 = A[(innerRowA + offset) * K + innerColA * 4 + 0];
    __half a1 = A[(innerRowA + offset) * K + innerColA * 4 + 1];
    __half a2 = A[(innerRowA + offset) * K + innerColA * 4 + 2];
    __half a3 = A[(innerRowA + offset) * K + innerColA * 4 + 3];
    As[(innerColA * 4 + 0) * BM + innerRowA + offset] = a0;
    As[(innerColA * 4 + 1) * BM + innerRowA + offset] = a1;
    As[(innerColA * 4 + 2) * BM + innerRowA + offset] = a2;
    As[(innerColA * 4 + 3) * BM + innerRowA + offset] = a3;
  }
  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    Bs[(innerRowB + offset) * BN + innerColB * 4 + 0] =
        B[(innerRowB + offset) * N + innerColB * 4 + 0];
    Bs[(innerRowB + offset) * BN + innerColB * 4 + 1] =
        B[(innerRowB + offset) * N + innerColB * 4 + 1];
    Bs[(innerRowB + offset) * BN + innerColB * 4 + 2] =
        B[(innerRowB + offset) * N + innerColB * 4 + 2];
    Bs[(innerRowB + offset) * BN + innerColB * 4 + 3] =
        B[(innerRowB + offset) * N + innerColB * 4 + 3];
  }
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void
processFromSmem(float *regM, float *regN, float *threadResults, const __half *As,
                const __half *Bs, const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint i = 0; i < TM; ++i) {
        regM[wSubRowIdx * TM + i] = __half2float(
            As[(dotIdx * BM) + warpRow * WM + wSubRowIdx * WSUBM +
               threadRowInWarp * TM + i]);
      }
    }
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      for (uint i = 0; i < TN; ++i) {
        regN[wSubColIdx * TN + i] = __half2float(
            Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
               threadColInWarp * TN + i]);
      }
    }
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
          for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                          (wSubColIdx * TN) + resIdxN] +=
                regM[wSubRowIdx * TM + resIdxM] *
                regN[wSubColIdx * TN + resIdxN];
          }
        }
      }
    }
  }
}
} // namespace wt

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    hgemmWarptiling(int M, int N, int K, float alpha, const __half *A,
                    const __half *B, float beta, __half *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;
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

  A += cRow * BM * K;
  B += cCol * BN;
  C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

  const uint innerRowA = threadIdx.x / (BK / 4);
  const uint innerColA = threadIdx.x % (BK / 4);
  constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
  const uint innerRowB = threadIdx.x / (BN / 4);
  const uint innerColB = threadIdx.x % (BN / 4);
  constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

  float threadResults[WMITER * TM * WNITER * TN] = {0.0f};
  float regM[WMITER * TM] = {0.0f};
  float regN[WNITER * TN] = {0.0f};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
        N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
    __syncthreads();
    wt::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                        TN>(regM, regN, threadResults, As, Bs, warpRow, warpCol,
                            threadRowInWarp, threadColInWarp);
    A += BK;
    B += BK * N;
    __syncthreads();
  }

  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      __half *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
      for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 1) {
          uint idx = (threadRowInWarp * TM + resIdxM) * N +
                     threadColInWarp * TN + resIdxN;
          float c_old = __half2float(C_interim[idx]);
          const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
          C_interim[idx] = __float2half(alpha * threadResults[i] + beta * c_old);
        }
      }
    }
  }
}

float bench_kernel(int M, int N, int K, const __half *A, const __half *B, __half *C,
                   int warmup, int iters) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Config: same as our current broken one
    constexpr int NUM_THREADS = 128;
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int WM = 64, WN = 128;
    constexpr int WNITER = 4;
    constexpr int TM = 4, TN = 4;

    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 block(NUM_THREADS);

    for (int i = 0; i < warmup; i++)
        hgemmWarptiling<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
            <<<grid, block>>>(M, N, K, 1.0f, A, B, 0.0f, C);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < iters; i++)
        hgemmWarptiling<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
            <<<grid, block>>>(M, N, K, 1.0f, A, B, 0.0f, C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iters;
}

int main() {
    int M = 256, N = 256, K = 256;
    size_t sizeA = M * K * sizeof(__half);
    size_t sizeB = K * N * sizeof(__half);
    size_t sizeC = M * N * sizeof(__half);

    __half *dA, *dB, *dC;
    cudaMalloc(&dA, sizeA);
    cudaMalloc(&dB, sizeB);
    cudaMalloc(&dC, sizeC);
    cudaMemset(dA, 0, sizeA);
    cudaMemset(dB, 0, sizeB);

    // Print config
    constexpr int NUM_THREADS = 128;
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int WM = 64, WN = 128;
    constexpr int WNITER = 4;
    constexpr int TM = 4, TN = 4;
    constexpr int WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;

    printf("=== Config ===\n");
    printf("WARPSIZE=%d NUM_THREADS=%d NUM_WARPS=%d\n", WARPSIZE, NUM_THREADS, NUM_THREADS/WARPSIZE);
    printf("BM=%d BN=%d BK=%d\n", BM, BN, BK);
    printf("WM=%d WN=%d WNITER=%d WMITER=%d\n", WM, WN, WNITER, WMITER);
    printf("WSUBM=%d WSUBN=%d\n", WSUBM, WSUBN);
    printf("TM=%d TN=%d\n", TM, TN);
    printf("threadResults size = %d floats = %d bytes\n",
           WMITER*TM*WNITER*TN, WMITER*TM*WNITER*TN*4);
    printf("regM size = %d, regN size = %d\n", WMITER*TM, WNITER*TN);
    printf("grid = (%d, %d)\n", CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    printf("As size = %d halfs = %d bytes\n", BM*BK, BM*BK*2);
    printf("Bs size = %d halfs = %d bytes\n", BK*BN, BK*BN*2);
    printf("rowStrideA = %d, rowStrideB = %d\n", (NUM_THREADS*4)/BK, NUM_THREADS/(BN/4));

    // Small benchmark
    printf("\n=== Bench 256x256 ===\n");
    float ms = bench_kernel(256, 256, 256, dA, dB, dC, 5, 20);
    printf("  kernel 10: %.3f ms\n", ms);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    // Now test with bigger matrix
    M = 256; N = 11008; K = 4096;
    cudaMalloc(&dA, M*K*sizeof(__half));
    cudaMalloc(&dB, K*N*sizeof(__half));
    cudaMalloc(&dC, M*N*sizeof(__half));
    cudaMemset(dA, 0, M*K*sizeof(__half));
    cudaMemset(dB, 0, K*N*sizeof(__half));

    printf("\n=== Bench 256x4096@4096x11008 ===\n");
    ms = bench_kernel(M, N, K, dA, dB, dC, 2, 5);
    printf("  kernel 10: %.3f ms\n", ms);

    err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("CUDA error: %s\n", cudaGetErrorString(err));

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}
CUDA

/usr/local/corex/bin/clang++ --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex \
    -I/usr/local/corex/include -L/usr/local/corex/lib64 -lcudart \
    -O2 /tmp/probe_k10_perf.cu -o /tmp/probe_k10_perf 2>&1

if [ -f /tmp/probe_k10_perf ]; then
    echo "Compile: SUCCESS"
    /tmp/probe_k10_perf
else
    echo "Compile: FAILED"
fi
