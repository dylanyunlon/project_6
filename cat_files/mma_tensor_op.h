/***************************************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

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
    \brief Templates implementing warp-level matrix multiply-accumulate operations targeting
      Tensor Cores.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/platform/platform.h"

#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/arch/mma.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/warp/mma.h"

#include "cutlass/gemm/warp/mma_tensor_op_policy.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace detail {

template <typename T, typename S, int N, FloatRoundStyle Round>
struct ConvertAndPack {

  using Converter = NumericArrayConverter<T, S, N, Round>;

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<S, N> const &source) {
    Converter converter;

    return converter(source);
  }
};

template <typename T, int N, FloatRoundStyle Round>
struct ConvertAndPack<T, T, N, Round> {

  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const &source) {
		return source;
  }
};

template <int N, FloatRoundStyle Round>
struct ConvertAndPack<bfloat16_t, float, N, Round> {

  using Converter = NumericArrayConverter<bfloat16_t, float, N, Round>;

  CUTLASS_HOST_DEVICE
  Array<bfloat16_t, N> operator()(Array<float, N> const &source) {
    Converter converter;

    Array<float, N> tmp;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      int idx = (((i << 1) & 2) | ((i >> 1) & 1) | (i & 0xfffffffc));
      tmp[i] = source[idx];
    }

    return converter(tmp);
  }
};

template <int N, FloatRoundStyle Round>
struct ConvertAndPack<half_t, float, N, Round> {

  using Converter = NumericArrayConverter<half_t, float, N, Round>;

  CUTLASS_HOST_DEVICE
  Array<half_t, N> operator()(Array<float, N> const &source) {
    Converter converter;

    Array<float, N> tmp;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      int idx = (((i << 1) & 2) | ((i >> 1) & 1) | (i & 0xfffffffc));
      tmp[i] = source[idx];
    }

    return converter(tmp);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace detail

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math instructions.
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  /// Data type of A elements
  typename ElementA_,
  /// Layout of A matrix (concept: MatrixLayout)
  typename LayoutA_,
  /// Data type of B elements
  typename ElementB_,
  /// Layout of B matrix (concept: MatrixLayout)
  typename LayoutB_,
  /// Element type of C matrix
  typename ElementC_,
  /// Layout of C matrix (concept: MatrixLayout)
  typename LayoutC_,
  /// Policy describing warp-level MmaTensorOp (concept: MmaTensorOp policy)
  typename Policy_,
  /// Number of partitions along K dimension
  int PartitionsK_ = 1,
  /// Store the accumulators in row major or column major.
  /// Iluvatar Tensor Core always stores accumulators in row major
  bool AccumulatorsInRowMajor = true,
  /// Used for partial specialization
  typename Enable = bool
>
class MmaTensorOp {
public:
  /// Shape of warp-level matrix operation (concept: GemmShape)
  using Shape = Shape_;

  /// Data type of multiplicand A
  using ElementA = ElementA_;

  /// Layout of multiplicand A
  using LayoutA = LayoutA_;

  /// Data type of multiplicand B
  using ElementB = ElementB_;

  /// Layout of multiplicand B
  using LayoutB = LayoutB_;

  /// Data type of accumulator matrix C
  using ElementC = ElementC_;

  /// Layout of accumulator matrix C
  using LayoutC = LayoutC_;

  /// Shape of the warp in units of thread (concept: MmaLanePolicySimt)
  using Policy = Policy_;

  /// Underlying matrix multiply operator (concept: arch::Mma)
  using ArchMmaOperator = typename Policy::Operator;

  /// Architecture tag from underlying instruction
  using ArchTag = typename ArchMmaOperator::ArchTag;

  /// Indicates class of matrix operator
  using OperatorClass = arch::OpClassTensorOp;

  /// Shape of underlying instruction
  using InstructionShape = typename ArchMmaOperator::Shape;

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = ComplexTransform::kNone;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = ComplexTransform::kNone;

  /// Number of threads participating in warp-level matrix product
  static int const kThreadCount = NUM_THREADS_PER_WARP;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

public:
  /// FIXME(Peter Han): workaround to adapt to simt epilogue, need to remove
  struct ThreadMma {
    using ElementC = ElementC;
  };

  /// Iterates over the A operand in memory
  using IteratorA = MmaTensorOpMultiplicandTileIterator<
    MatrixShape<Shape::kM, Policy::Operator::Shape::kK>,
    Operand::kA,
    ElementA,
    LayoutA,
    InstructionShape,
    kThreadCount,
    kPartitionsK>;

  /// Storage for A tile
  using FragmentA = typename IteratorA::Fragment;

  /// Storage for transformed A tile
  using TransformedFragmentA =
      Array<typename ArchMmaOperator::ElementA, FragmentA::kElements>;

  /// Iterates over the B operand in memory
  using IteratorB = MmaTensorOpMultiplicandTileIterator<
    MatrixShape<Policy::Operator::Shape::kK, Shape::kN>,
    Operand::kB,
    ElementB,
    LayoutB,
    InstructionShape,
    kThreadCount,
    kPartitionsK>;

  /// Storage for B tile
  using FragmentB = typename IteratorB::Fragment;

  /// Storage for transformed B tile
  using TransformedFragmentB =
      Array<typename ArchMmaOperator::ElementB, FragmentB::kElements>;

  /// Iterates over the C operand in memory
  using IteratorC = MmaTensorOpAccumulatorTileIterator<
    MatrixShape<Shape::kM, Shape::kN>,
    ElementC,
    LayoutC,
    InstructionShape>;

  /// Storage for C tile
  using FragmentC = typename IteratorC::Fragment;

  static_assert(
    !(Shape::kM % Policy::Operator::Shape::kM) &&
    !(Shape::kN % Policy::Operator::Shape::kN) &&
    !(Shape::kK % Policy::Operator::Shape::kK),
    "Shape of warp-level Mma must be divisible by operator shape.");

  using MmaIterations = gemm::GemmShape<
    (Shape::kM + ArchMmaOperator::Shape::kM - 1) / ArchMmaOperator::Shape::kM,
    (Shape::kN + ArchMmaOperator::Shape::kN - 1) / ArchMmaOperator::Shape::kN,
    InstructionShape::kK / Policy::Operator::Shape::kK
  >;

public:

  /// Underlying matrix multiply operator (concept: arch::Mma)
  ArchMmaOperator mma;

public:

  //
  // Methods
  //

  /// Ctor
  CUTLASS_DEVICE
  MmaTensorOp() {}

  /// Performs a warp-level matrix multiply-accumulate operation
  CUTLASS_DEVICE
  void operator()(
    FragmentC &D,
    TransformedFragmentA const &A,
    TransformedFragmentB const &B,
    FragmentC const &C
  ) const {

    using MmaOperandA = typename ArchMmaOperator::FragmentA;
    using MmaOperandB = typename ArchMmaOperator::FragmentB;
    using MmaOperandC = typename ArchMmaOperator::FragmentC;

    D = C;

    MmaOperandA const *ptr_A = reinterpret_cast<MmaOperandA const *>(&A);
    MmaOperandB const *ptr_B = reinterpret_cast<MmaOperandB const *>(&B);
    MmaOperandC *ptr_D = reinterpret_cast<MmaOperandC *>(&D);

    // Serpentine visitation order maximizing reuse of Rb
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < MmaIterations::kK; ++k) {
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < MmaIterations::kM; ++m) {
        CUTLASS_PRAGMA_UNROLL
        for (int n = 0; n < MmaIterations::kN; ++n) {
          int n_serpentine = ((m % 2) ? (MmaIterations::kN - 1 - n) : n);

          /// assume A is column-major in VRF, B is row-major in VRF
          if(AccumulatorsInRowMajor) {
            mma(
              ptr_D[n_serpentine + m * MmaIterations::kN],
              ptr_A[m + k * MmaIterations::kM],
              ptr_B[n_serpentine + k * MmaIterations::kN],
              ptr_D[n_serpentine + m * MmaIterations::kN]);
          } else {
            mma(
              ptr_D[m + n_serpentine * MmaIterations::kM],
              ptr_A[m + k * MmaIterations::kM],
              ptr_B[n_serpentine + k * MmaIterations::kN],
              ptr_D[m + n_serpentine * MmaIterations::kM]);
          }
        }
      }
    }
  }

  /// Transform the mma operands to the required types
  CUTLASS_DEVICE
  void transform(TransformedFragmentA &dst_A, TransformedFragmentB &dst_B,
                 FragmentA const &A, FragmentB const &B) const {

    //
    // Define conversions from source type to instruction type
    //
    FloatRoundStyle const kRoundA =
        PreferredRoundingMode<typename ArchMmaOperator::ElementA,
                              ElementA>::kRound;
    FloatRoundStyle const kRoundB =
        PreferredRoundingMode<typename ArchMmaOperator::ElementB,
                              ElementB>::kRound;
    detail::ConvertAndPack<typename ArchMmaOperator::ElementA, ElementA,
                            FragmentA::kElements / 2, kRoundA>
          convert_A;
    NumericArrayConverter<typename ArchMmaOperator::ElementB, ElementB,
                            FragmentB::kElements, kRoundB>
          convert_B;
    Array<ElementA, FragmentA::kElements / 2> const *ptr_A =
          reinterpret_cast<Array<ElementA, FragmentA::kElements / 2> const *>(&A);
    Array<typename ArchMmaOperator::ElementA, FragmentA::kElements / 2> *
          ptr_dst_A = reinterpret_cast<Array<typename ArchMmaOperator::ElementA,
                                             FragmentA::kElements / 2> *>(&dst_A);

    dst_B = convert_B(B);

    ptr_dst_A[0] = convert_A(ptr_A[0]);
    ptr_dst_A[1] = convert_A(ptr_A[1]);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
