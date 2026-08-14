#pragma once

#include "kernels/01_naive.cuh"
#include "kernels/02_kernel_global_mem_coalesce.cuh"
#include "kernels/03_kernel_shared_mem_blocking.cuh"
#include "kernels/04_kernel_1D_blocktiling.cuh"
#include "kernels/05_kernel_2D_blocktiling.cuh"
#include "kernels/06_kernel_vectorize.cuh"
#include "kernels/07_kernel_resolve_bank_conflicts.cuh"
#include "kernels/08_kernel_bank_extra_col.cuh"
#include "kernels/09_kernel_autotuned.cuh"
#include "kernels/10_kernel_warptiling.cuh"
#include "kernels/11_kernel_double_buffering.cuh"
#include "kernels/12_kernel_double_buffering.cuh"