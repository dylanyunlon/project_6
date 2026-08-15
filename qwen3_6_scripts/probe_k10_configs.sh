#!/bin/bash
# probe_k10_configs.sh — Test multiple kernel 10 configs on BI-V100
set -eo pipefail

cat > /tmp/probe_k10_configs.cu << 'CUDA'
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))
const int WARPSIZE = 64;

// Same kernel code as before
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

template <int BM, int BN, int BK, int WM, int WN, int WNITER, int TM, int TN, int NT>
float bench(int M, int N, int K, const __half *A, const __half *B, __half *C) {
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 block(NT);

    // warmup
    for (int i = 0; i < 3; i++)
        hgemmWarptiling<BM, BN, BK, WM, WN, WNITER, TM, TN, NT>
            <<<grid, block>>>(M, N, K, 1.0f, A, B, 0.0f, C);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < 10; i++)
        hgemmWarptiling<BM, BN, BK, WM, WN, WNITER, TM, TN, NT>
            <<<grid, block>>>(M, N, K, 1.0f, A, B, 0.0f, C);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  CUDA error: %s\n", cudaGetErrorString(err));
        return -1.0f;
    }
    return ms / 10.0f;
}

float bench_cublas(int M, int N, int K, const __half *A, const __half *B, __half *C) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    __half alpha_h = __float2half(1.0f), beta_h = __float2half(0.0f);

    for (int i = 0; i < 3; i++)
        cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha_h, B, N, A, K, &beta_h, C, N);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < 10; i++)
        cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha_h, B, N, A, K, &beta_h, C, N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cublasDestroy(handle);
    return ms / 10.0f;
}

int main() {
    int M = 256, N = 256, K = 256;
    __half *dA, *dB, *dC;
    cudaMalloc(&dA, M*K*sizeof(__half));
    cudaMalloc(&dB, K*N*sizeof(__half));
    cudaMalloc(&dC, M*N*sizeof(__half));
    cudaMemset(dA, 0, M*K*sizeof(__half));
    cudaMemset(dB, 0, K*N*sizeof(__half));

    float ms_cublas = bench_cublas(M, N, K, dA, dB, dC);
    printf("cublas baseline 256x256: %.3f ms\n\n", ms_cublas);

    // Config A: current (broken)
    printf("Config A: BM128 BN128 BK16 WM64 WN128 WNITER4 TM4 TN4 NT128\n");
    float msA = bench<128,128,16, 64,128, 4, 4,4, 128>(M,N,K,dA,dB,dC);
    printf("  %.3f ms  (%.1fx cublas)\n\n", msA, msA/ms_cublas);

    // Config B: fewer WNITER, bigger TM
    printf("Config B: BM128 BN128 BK16 WM64 WN64 WNITER2 TM8 TN4 NT128\n");
    float msB = bench<128,128,16, 64,64, 2, 8,4, 128>(M,N,K,dA,dB,dC);
    printf("  %.3f ms  (%.1fx cublas)\n\n", msB, msB/ms_cublas);

    // Config C: 256 threads (4 warps of 64)
    printf("Config C: BM128 BN128 BK16 WM64 WN64 WNITER2 TM4 TN4 NT256\n");
    float msC = bench<128,128,16, 64,64, 2, 4,4, 256>(M,N,K,dA,dB,dC);
    printf("  %.3f ms  (%.1fx cublas)\n\n", msC, msC/ms_cublas);

    // Config D: smaller block, more blocks for 16 SMs
    printf("Config D: BM64 BN64 BK16 WM64 WN64 WNITER4 TM4 TN4 NT64\n");
    float msD = bench<64,64,16, 64,64, 4, 4,4, 64>(M,N,K,dA,dB,dC);
    printf("  %.3f ms  (%.1fx cublas)\n\n", msD, msD/ms_cublas);

    // Config E: WMITER=1 by design
    printf("Config E: BM128 BN64 BK16 WM64 WN64 WNITER1 TM4 TN4 NT128\n");
    float msE = bench<128,64,16, 64,64, 1, 4,4, 128>(M,N,K,dA,dB,dC);
    printf("  %.3f ms  (%.1fx cublas)\n\n", msE, msE/ms_cublas);

    // Config F: bigger BK=32
    printf("Config F: BM128 BN128 BK32 WM64 WN128 WNITER4 TM4 TN4 NT256\n");
    float msF = bench<128,128,32, 64,128, 4, 4,4, 256>(M,N,K,dA,dB,dC);
    printf("  %.3f ms  (%.1fx cublas)\n\n", msF, msF/ms_cublas);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    // Big matrix
    M = 256; N = 11008; K = 4096;
    cudaMalloc(&dA, (long long)M*K*sizeof(__half));
    cudaMalloc(&dB, (long long)K*N*sizeof(__half));
    cudaMalloc(&dC, (long long)M*N*sizeof(__half));
    cudaMemset(dA, 0, (long long)M*K*sizeof(__half));
    cudaMemset(dB, 0, (long long)K*N*sizeof(__half));

    printf("=== Big matrix 256x4096 @ 4096x11008 ===\n");
    ms_cublas = bench_cublas(M, N, K, dA, dB, dC);
    printf("cublas: %.3f ms\n", ms_cublas);

    msA = bench<128,128,16, 64,128, 4, 4,4, 128>(M,N,K,dA,dB,dC);
    printf("Config A: %.3f ms (%.1fx)\n", msA, msA/ms_cublas);
    msB = bench<128,128,16, 64,64, 2, 8,4, 128>(M,N,K,dA,dB,dC);
    printf("Config B: %.3f ms (%.1fx)\n", msB, msB/ms_cublas);
    msC = bench<128,128,16, 64,64, 2, 4,4, 256>(M,N,K,dA,dB,dC);
    printf("Config C: %.3f ms (%.1fx)\n", msC, msC/ms_cublas);
    msF = bench<128,128,32, 64,128, 4, 4,4, 256>(M,N,K,dA,dB,dC);
    printf("Config F: %.3f ms (%.1fx)\n", msF, msF/ms_cublas);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
CUDA

echo "=== Compiling ==="
/usr/local/corex/bin/clang++ --cuda-gpu-arch=ivcore10 --cuda-path=/usr/local/corex \
    -I/usr/local/corex/include -L/usr/local/corex/lib64 -lcudart -lcublas \
    -O2 /tmp/probe_k10_configs.cu -o /tmp/probe_k10_configs 2>&1

if [ -f /tmp/probe_k10_configs ]; then
    echo "Compile: SUCCESS"
    /tmp/probe_k10_configs
else
    echo "Compile: FAILED"
fi
