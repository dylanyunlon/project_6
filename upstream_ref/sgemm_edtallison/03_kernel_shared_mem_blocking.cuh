#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
    // output C block we want to compute with this threadBlock
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    // allocate buffer for current block in fast SMEM (shared between all threads in block)
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    // the inner row and col that we are accessing in this specific thread
    const uint threadRow = threadIdx.x / BLOCKSIZE; // note similarity to previous kernel
    const uint threadCol = threadIdx.x % BLOCKSIZE;

    // advance pointers to the starting positions (they are input as pointers to first elements in the matrices)
    A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0 (the start of the relevant row)
    B += cCol * BLOCKSIZE;                        // row=0, col=cCol (top of relevant col)
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

    float tmp = 0.0;
    for (int bkIdx=0; bkIdx < K; bkIdx+=BLOCKSIZE) { // shifting the whole block along the row of A and col of B
        // have each thread load one of the elements in A and B
        // make the threadCol (=threadIdx.x) the consecutive index
        // to allow GMEM access coalescing
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

        // ensure cache is fully populated
        __syncthreads();
        A += BLOCKSIZE; // for next iteration
        B += BLOCKSIZE * N;

        // execute the dotproduct on the currently cached block
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
            tmp += As[threadRow * BLOCKSIZE + dotIdx] *
                   Bs[dotIdx * BLOCKSIZE + threadCol];
        }
        // sync so faster threads don't fetch the next block into cache
        _syncthreads();
    }
    C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
}