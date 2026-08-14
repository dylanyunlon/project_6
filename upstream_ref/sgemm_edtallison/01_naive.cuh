# pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

/*

Matrix sizes:
MxK * KxN = MxN

*/

__global__ void sgemm_naive(
    int M, int N, int K, // sizes
    float alpha, const float *A, const float *B, float beta, float *C // pointers used to point to matrices
) {
    // compute position in C that this thread is responsible for
    // "which block" * "width of block" to get to start of block + "which thread"
    const uint x = blockIdx.x * blockDim.x + threadIdx.x; // "which row?" (inverted from graphical intuition, confusingly)
    const uint y = blockIdx.y * blockDim.y + threadIdx.y; // "which column?"

    // if M or N are not multiples of 32, there will be "extra"/"remainder" threads on the last block in x/y.
    // we don't want those leftover threads to do anything (tile quantisation)
    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) { // K is the size of the row in A, col in B i.e. the dot product
            // A: x * K gives the start of relevant row, i enumerates across the row (col by col)
            // B: y gives the relevant column, i * N enumerates down the column, (row by row)
            tmp += A[x * K + i] * B[i * N + y];
        }
        // C = alpha*(A@B) + beta*C
        // x * N takes to start of relevant row, y moves across to the relevant column
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}
