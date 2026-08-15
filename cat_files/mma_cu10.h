/***************************************************************************************************
* Copyright (c) 2021 Iluvatar CoreX. All rights reserved.
* Copyright Declaration: This software, including all of its code and documentation,
* except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
* Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
* Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
* CoreX. No user of this software shall have any right, ownership or interest in this software and
* any use of this software shall be in compliance with the terms and conditions of the End User
* License Agreement.
 **************************************************************************************************/

/*! \file
    \brief Matrix Multiply for BigIsland 1st generation
*/

#pragma once

#include "cutlass/arch/mma.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/gemm.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace arch {

/// BigIsland Tensor Core tile format - EM orinted vector type definitions
/// fp32
typedef float v4float_t __attribute__((ext_vector_type(4)));
/// s32
typedef int32_t v4int32_t __attribute__((ext_vector_type(4)));
/// u32
typedef uint32_t v4uint32_t __attribute__((ext_vector_type(4)));
/// fp16
typedef uint16_t v4half_t __attribute__((ext_vector_type(4)));
/// bf16
typedef uint16_t v4bfloat16_t __attribute__((ext_vector_type(4)));
/// s8
typedef int8_t v4int8_t __attribute__((ext_vector_type(4)));
/// u8
typedef uint8_t v4uint8_t __attribute__((ext_vector_type(4)));

////////////////////////////////////////////////////////////////////////////////
//
// Matrix multiply accumulate 161616 - U32 accumulation
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: U32 = U8 * U8 + U32
template <typename LayoutA, typename LayoutB, typename LayoutC>
struct Mma<
  gemm::GemmShape<16, 16, 16>,
  64,
  uint8_t,
  LayoutA,
  uint8_t,
  LayoutB,
  uint32_t,
  LayoutC,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 16, 16>;

  using ElementA = uint8_t;
  using FragmentA = Array<uint8_t, 4>;

  using ElementB = uint8_t;
  using FragmentB = Array<uint8_t, 4>;

  using ElementC = uint;
  using FragmentC = Array<uint, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Cu10;

  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {
#if CUTLASS_ARCH_CU10_SUPPORTED
    v4uint8_t src_A;
    v4uint8_t src_B;
    v4uint32_t src_C;
    v4uint32_t dst_D;

    src_A[0] = a[0];
    src_A[1] = a[1];
    src_A[2] = a[2];
    src_A[3] = a[3];
    src_B[0] = b[0];
    src_B[1] = b[1];
    src_B[2] = b[2];
    src_B[3] = b[3];
    src_C[0] = c[0];
    src_C[1] = c[1];
    src_C[2] = c[2];
    src_C[3] = c[3];

    dst_D = __ivcorex_matrix_mad_u32x4_u8x4(src_A, src_B, src_C);

    d[0] = dst_D[0];
    d[1] = dst_D[1];
    d[2] = dst_D[2];
    d[3] = dst_D[3];
#else
    assert(0);
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix multiply accumulate 161616 - S32 accumulation
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: S32 = S8 * S8 + S32
template <typename LayoutA, typename LayoutB, typename LayoutC>
struct Mma<
  gemm::GemmShape<16, 16, 16>,
  64,
  int8_t,
  LayoutA,
  int8_t,
  LayoutB,
  int,
  LayoutC,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 16, 16>;

  using ElementA = int8_t;
  using FragmentA = Array<int8_t, 4>;

  using ElementB = int8_t;
  using FragmentB = Array<int8_t, 4>;

  using ElementC = int;
  using FragmentC = Array<int, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Cu10;

  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {
#if CUTLASS_ARCH_CU10_SUPPORTED
    v4int8_t src_A;
    v4int8_t src_B;
    v4int32_t src_C;
    v4int32_t dst_D;

    src_A[0] = a[0];
    src_A[1] = a[1];
    src_A[2] = a[2];
    src_A[3] = a[3];
    src_B[0] = b[0];
    src_B[1] = b[1];
    src_B[2] = b[2];
    src_B[3] = b[3];
    src_C[0] = c[0];
    src_C[1] = c[1];
    src_C[2] = c[2];
    src_C[3] = c[3];

    dst_D = __ivcorex_matrix_mad_i32x4_i8x4(src_A, src_B, src_C);

    d[0] = dst_D[0];
    d[1] = dst_D[1];
    d[2] = dst_D[2];
    d[3] = dst_D[3];
#else
    assert(0);
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////
//
// Matrix multiply accumulate 161616 - FP32 accumulation
//
////////////////////////////////////////////////////////////////////////////////

/// Matrix multiply-add operation: FP32 = FP16 * FP16 + FP32
template <typename LayoutA, typename LayoutB, typename LayoutC>
struct Mma<
  gemm::GemmShape<16, 16, 16>,
  64,
  cutlass::half_t,
  LayoutA,
  cutlass::half_t,
  LayoutB,
  float,
  LayoutC,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 16, 16>;

  using ElementA = cutlass::half_t;
  using FragmentA = Array<half_t, 4>;

  using ElementB = cutlass::half_t;
  using FragmentB = Array<half_t, 4>;

  using ElementC = float;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Cu10;

  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {
    v4half_t src_A;
    v4half_t src_B;
    v4float_t src_C;
    v4float_t dst_D;

    src_A[0] = half_t(a[0]).storage;
    src_A[1] = half_t(a[1]).storage;
    src_A[2] = half_t(a[2]).storage;
    src_A[3] = half_t(a[3]).storage;
    src_B[0] = half_t(b[0]).storage;
    src_B[1] = half_t(b[1]).storage;
    src_B[2] = half_t(b[2]).storage;
    src_B[3] = half_t(b[3]).storage;
    src_C[0] = c[0];
    src_C[1] = c[1];
    src_C[2] = c[2];
    src_C[3] = c[3];

    dst_D = __ivcorex_matrix_mad_f32x4_f16x4(src_A, src_B, src_C);
#if 0
if(threadIdx.x == 0)
printf(
  ">>> After\n"
  "A: %f, %f, %f, %f\n"
  "B: %f, %f, %f, %f\n"
  "C: %f, %f, %f, %f\n"
  "D: %f, %f, %f, %f\n\n",
  float(a[0]), float(a[1]), float(a[2]), float(a[3]),
  float(b[0]), float(b[1]), float(b[2]), float(b[3]),
  float(src_C[0]), float(src_C[1]), float(src_C[2]), float(src_C[3]),
  float(d[0]), float(d[1]), float(d[2]), float(d[3])
);
#endif

    d[0] = dst_D[0];
    d[1] = dst_D[1];
    d[2] = dst_D[2];
    d[3] = dst_D[3];

  }
};

/// Matrix multiply-add operation: FP32 = BF16 * BF16 + FP32
template <typename LayoutA, typename LayoutB, typename LayoutC>
struct Mma<
  gemm::GemmShape<16, 16, 16>,
  64,
  bfloat16_t,
  LayoutA,
  bfloat16_t,
  LayoutB,
  float,
  LayoutC,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16, 16, 16>;

  using ElementA = bfloat16_t;
  using FragmentA = Array<bfloat16_t, 4>;

  using ElementB = bfloat16_t;
  using FragmentB = Array<bfloat16_t, 4>;

  using ElementC = float;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Cu10;

  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {
    v4bfloat16_t src_A;
    v4bfloat16_t src_B;
    v4float_t src_C;
    v4float_t dst_D;

    src_A[0] = bfloat16_t(a[0]).storage;
    src_A[1] = bfloat16_t(a[1]).storage;
    src_A[2] = bfloat16_t(a[2]).storage;
    src_A[3] = bfloat16_t(a[3]).storage;
    src_B[0] = bfloat16_t(b[0]).storage;
    src_B[1] = bfloat16_t(b[1]).storage;
    src_B[2] = bfloat16_t(b[2]).storage;
    src_B[3] = bfloat16_t(b[3]).storage;
    src_C[0] = c[0];
    src_C[1] = c[1];
    src_C[2] = c[2];
    src_C[3] = c[3];
#if __clang_major__ >= 16
    dst_D = __ivcorex_matrix_mad_f32x4_bf16x4(src_A, src_B, src_C);
#else
    dst_D = __ivcorex_matrix_mad_f32_bf16(src_A, src_B, src_C);
#endif
    d[0] = dst_D[0];
    d[1] = dst_D[1];
    d[2] = dst_D[2];
    d[3] = dst_D[3];
  }
};

/// Matrix multiply-add operation: FP32 = FP32 * FP32 + FP32
template <typename LayoutA, typename LayoutB, typename LayoutC>
struct Mma<
  gemm::GemmShape<16,16,16>,
  64,
  float,
  LayoutA,
  float,
  LayoutB,
  float,
  LayoutC,
  OpMultiplyAdd> {

  using Shape = gemm::GemmShape<16,16,16>;

  using ElementA = float;
  using FragmentA = Array<float, 4>;

  using ElementB = float;
  using FragmentB = Array<float, 4>;

  using ElementC = float;
  using FragmentC = Array<float, 4>;

  using Operator = OpMultiplyAdd;
  using ArchTag = arch::Cu10;

  CUTLASS_HOST_DEVICE
  void operator()(
    FragmentC &d,
    FragmentA const &a,
    FragmentB const &b,
    FragmentC const &c
  ) const {
    v4float_t src_A;
    v4float_t src_B;
    v4float_t src_C;
    v4float_t dst_D;

    src_A[0] = a[0];
    src_A[1] = a[1];
    src_A[2] = a[2];
    src_A[3] = a[3];
    src_B[0] = b[0];
    src_B[1] = b[1];
    src_B[2] = b[2];
    src_B[3] = b[3];
    src_C[0] = c[0];
    src_C[1] = c[1];
    src_C[2] = c[2];
    src_C[3] = c[3];

    dst_D = __ivcorex_matrix_mad_f32x4_f32x4(src_A, src_B, src_C);

    d[0] = dst_D[0];
    d[1] = dst_D[1];
    d[2] = dst_D[2];
    d[3] = dst_D[3];
  }
};

////////////////////////////////////////////////////////////////////////////////
}
}
