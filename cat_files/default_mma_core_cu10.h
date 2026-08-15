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
    \brief Defines basic properties needed by CTA-level GEMMs assuming expectations about data
      layout of the global memory fragments, data types, and internal tile sizes.

      Partial specializations for threadblock::Mma operations targeting TensorOp instructions.

      Aims at TensorOp of the first generation BigIsland.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"

#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h"
#include "cutlass/layout/tensor_op_multiplicand.h"
#include "cutlass/layout/tensor_op_em.h"

#include "cutlass/gemm/warp/mma_tensor_op_policy.h"
#include "cutlass/gemm/warp/mma_tensor_op.h"
#include "cutlass/gemm/warp/default_mma_tensor_op.h"
#include "cutlass/gemm/threadblock/default_mma_core.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////
///
/// Specialization: A: row-major, B: row-major, TT
///
/// This uses the default warp-level operator given tile sizes
///
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Data type of A operand
    typename ElementA_,
    /// Data type of B operand
    typename ElementB_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Stages
    int Stages,
    /// Operation performed by GEMM
    typename Operator_>
struct DefaultMmaCore<Shape_,
                      WarpShape_,
                      GemmShape<16, 16, 16>,
                      ElementA_,
                      layout::RowMajor,
                      ElementB_,
                      layout::RowMajor,
                      ElementC_,
                      LayoutC_,
                      arch::OpClassTensorOp,
                      Stages,
                      Operator_> {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = GemmShape<16, 16, 16>;
  using ElementA = ElementA_;
  using LayoutA = layout::RowMajor;
  using ElementB = ElementB_;
  using LayoutB = layout::RowMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using OperatorClass = arch::OpClassTensorOp;

  static int const kStages = Stages;

  /// Default Operator
  using Operator = Operator_;

  /// Warp thread arrangement
  using WarpThreadArrangement = layout::PitchLinearShape<16, 4>;

  /// Number of warps present
  using WarpCount = GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    Shape::kK / WarpShape::kK
  >;

  /// Don't support split K within CTA
  static_assert(Shape::kK == WarpShape::kK,
    "Threadblock-scoped GEMM shape K should equal warp-scoped GEMM shape K"
  );

  // Divisibility requirements
  static_assert(
    !(Shape::kM % WarpShape::kM) &&
    !(Shape::kN % WarpShape::kN) &&
    !(Shape::kK % WarpShape::kK),
    "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size."
  );

  // Divisibility requirements
  static_assert(
    !(WarpShape::kM % 16) &&
    !(WarpShape::kN % 16) &&
    !(WarpShape::kK % 16),
    "Threadblock-scoped GEMM should be divisible by 16."
  );

  /// Number of threads per warp
  static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  /// Size of a threadblock-scoped access
  static int const kAccessSizeInBits = 32;

  /// Number of A elemnts per access
  static int const kElementsPerAccessA = kAccessSizeInBits / sizeof_bits<ElementA>::value;

  /// Number of A elemnts per access
  static int const kElementsPerAccessB = kAccessSizeInBits / sizeof_bits<ElementB>::value;

  //
  // Shared memory layouts
  //

  #if BLOCK_LOAD_STORE
    using SmemLayoutA = layout::TensorOpEm<sizeof_bits<ElementA>::value, LayoutA>;
    using SmemLayoutB = layout::TensorOpEm<sizeof_bits<ElementB>::value, LayoutB>;
  #else
    using SmemLayoutA = layout::TensorOpMultiplicand<sizeof_bits<ElementA>::value, LayoutA>;
    using SmemLayoutB = layout::TensorOpMultiplicand<sizeof_bits<ElementB>::value, LayoutB>;
  #endif

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  ///
  using IteratorThreadMapA = transform::PitchLinear2DThreadTileWarpRakedThreadMap<
    layout::PitchLinearShape<Shape::kK, Shape::kM>,
    kThreads,
    WarpThreadArrangement,
    layout::PitchLinearShape<kElementsPerAccessA, kElementsPerAccessA>
  >;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileIterator<
    MatrixShape<Shape::kM, Shape::kK>,
    ElementA,
    SmemLayoutA,
    1,
    IteratorThreadMapA
  >;

  /// Policy of iterator B
  using IteratorThreadMapB = transform::PitchLinear2DThreadTileWarpRakedThreadMap<
    layout::PitchLinearShape<Shape::kN, Shape::kK>,
    kThreads,
    WarpThreadArrangement,
    layout::PitchLinearShape<kElementsPerAccessB, kElementsPerAccessB>
  >;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileIterator<
    MatrixShape<Shape::kK, Shape::kN>,
    ElementB,
    SmemLayoutB,
    0,
    IteratorThreadMapB
  >;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level tensor op
  using Policy = gemm::warp::MmaTensorOpPolicy<
    arch::Mma<
      gemm::GemmShape<16, 16, 16>,
      NUM_THREADS_PER_WARP,
      ElementA,
      LayoutA,
      ElementB,
      LayoutB,
      ElementC,
      layout::RowMajor,
      arch::OpMultiplyAdd
    >,
    MatrixShape<1, 1>
  >;

  using MmaTensorOp = typename gemm::warp::DefaultMmaTensorOp<
    WarpShape,
    gemm::GemmShape<16, 16, 16>,
    ElementA,
    SmemLayoutA,
    ElementB,
    SmemLayoutB,
    ElementC,
    LayoutC,
    arch::OpMultiplyAdd
  >::Type;

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<
    MmaTensorOp,
    MatrixShape<0, 0>,
    MatrixShape<0, 0>,
    WarpCount::kK
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
///
/// Specialization: A: row-major, B: column-major, TN
///
/// This uses the default warp-level operator given tile sizes
///
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Data type of A operand
    typename ElementA_,
    /// Data type of B operand
    typename ElementB_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Stages
    int Stages,
    /// Operation performed by GEMM
    typename Operator_>
struct DefaultMmaCore<Shape_,
                      WarpShape_,
                      GemmShape<16, 16, 16>,
                      ElementA_,
                      layout::RowMajor,
                      ElementB_,
                      layout::ColumnMajor,
                      ElementC_,
                      LayoutC_,
                      arch::OpClassTensorOp,
                      Stages,
                      Operator_> {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = GemmShape<16, 16, 16>;
  using ElementA = ElementA_;
  using LayoutA = layout::RowMajor;
  using ElementB = ElementB_;
  using LayoutB = layout::ColumnMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using OperatorClass = arch::OpClassTensorOp;

  static int const kStages = Stages;

  /// Default Operator
  using Operator = Operator_;

  /// Warp thread arrangement
  using WarpThreadArrangement = layout::PitchLinearShape<16, 4>;

  /// Number of warps present
  using WarpCount = GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    Shape::kK / WarpShape::kK
  >;

  /// Don't support split K within CTA
  static_assert(Shape::kK == WarpShape::kK,
    "Threadblock-scoped GEMM shape K should equal warp-scoped GEMM shape K"
  );

  // Divisibility requirements
  static_assert(
    !(Shape::kM % WarpShape::kM) &&
    !(Shape::kN % WarpShape::kN) &&
    !(Shape::kK % WarpShape::kK),
    "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size."
  );

  // Divisibility requirements
  static_assert(
    !(WarpShape::kM % 16) &&
    !(WarpShape::kN % 16) &&
    !(WarpShape::kK % 16),
    "Threadblock-scoped GEMM should be divisible by 16."
  );

  /// Number of threads per warp
  static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  /// Size of a threadblock-scoped access
  static int const kAccessSizeInBits = 32;

  /// Number of A elemnts per access
  static int const kElementsPerAccessA = kAccessSizeInBits / sizeof_bits<ElementA>::value;

  /// Number of A elemnts per access
  static int const kElementsPerAccessB = kAccessSizeInBits / sizeof_bits<ElementB>::value;

  //
  // Shared memory layouts
  //

  #if BLOCK_LOAD_STORE
    using SmemLayoutA = layout::TensorOpEm<sizeof_bits<ElementA>::value, LayoutA>;
    using SmemLayoutB = layout::TensorOpMultiplicand<sizeof_bits<ElementB>::value, LayoutB>;
  #else
    using SmemLayoutA = layout::TensorOpMultiplicand<sizeof_bits<ElementA>::value, LayoutA>;
    using SmemLayoutB = layout::TensorOpMultiplicand<sizeof_bits<ElementB>::value, LayoutB>;
  #endif

  //

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  ///
  using IteratorThreadMapA = transform::PitchLinear2DThreadTileWarpRakedThreadMap<
    layout::PitchLinearShape<Shape::kK, Shape::kM>,
    kThreads,
    WarpThreadArrangement,
    layout::PitchLinearShape<kElementsPerAccessA, kElementsPerAccessA>
  >;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileIterator<
    MatrixShape<Shape::kM, Shape::kK>,
    ElementA,
    SmemLayoutA,
    1,
    IteratorThreadMapA
  >;

  /// Policy of iterator B
  using IteratorThreadMapB = transform::PitchLinear2DThreadTileWarpRakedThreadMap<
    layout::PitchLinearShape<Shape::kK, Shape::kN>,
    kThreads,
    WarpThreadArrangement,
    layout::PitchLinearShape<kElementsPerAccessB, kElementsPerAccessB>
  >;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileIterator<
    MatrixShape<Shape::kK, Shape::kN>,
    ElementB,
    SmemLayoutB,
    0,
    IteratorThreadMapB
  >;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level tensor op
  using Policy = gemm::warp::MmaTensorOpPolicy<
    arch::Mma<
      gemm::GemmShape<16, 16, 16>,
      NUM_THREADS_PER_WARP,
      ElementA,
      LayoutA,
      ElementB,
      LayoutB,
      ElementC,
      layout::RowMajor,
      arch::OpMultiplyAdd
    >,
    MatrixShape<1, 1>
  >;

  using MmaTensorOp = typename gemm::warp::DefaultMmaTensorOp<
    WarpShape,
    gemm::GemmShape<16, 16, 16>,
    ElementA,
    SmemLayoutA,
    ElementB,
    SmemLayoutB,
    ElementC,
    LayoutC,
    arch::OpMultiplyAdd
  >::Type;

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<
    MmaTensorOp,
    MatrixShape<0, 0>,
    MatrixShape<0, 0>,
    WarpCount::kK
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
///
/// Specialization: A: column-major, B: row-major, NT
///
/// This uses the default warp-level operator given tile sizes
///
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Data type of A operand
    typename ElementA_,
    /// Data type of B operand
    typename ElementB_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Stages
    int Stages,
    /// Operation performed by GEMM
    typename Operator_>
struct DefaultMmaCore<Shape_,
                      WarpShape_,
                      GemmShape<16, 16, 16>,
                      ElementA_,
                      layout::ColumnMajor,
                      ElementB_,
                      layout::RowMajor,
                      ElementC_,
                      LayoutC_,
                      arch::OpClassTensorOp,
                      Stages,
                      Operator_> {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = GemmShape<16, 16, 16>;
  using ElementA = ElementA_;
  using LayoutA = layout::ColumnMajor;
  using ElementB = ElementB_;
  using LayoutB = layout::RowMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using OperatorClass = arch::OpClassTensorOp;

  static int const kStages = Stages;

  /// Default Operator
  using Operator = Operator_;

  /// Warp thread arrangement
  using WarpThreadArrangement = layout::PitchLinearShape<16, 4>;

  /// Number of warps present
  using WarpCount = GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    Shape::kK / WarpShape::kK
  >;

  /// Don't support split K within CTA
  static_assert(Shape::kK == WarpShape::kK,
    "Threadblock-scoped GEMM shape K should equal warp-scoped GEMM shape K"
  );

  // Divisibility requirements
  static_assert(
    !(Shape::kM % WarpShape::kM) &&
    !(Shape::kN % WarpShape::kN) &&
    !(Shape::kK % WarpShape::kK),
    "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size."
  );

  // Divisibility requirements
  static_assert(
    !(WarpShape::kM % 16) &&
    !(WarpShape::kN % 16) &&
    !(WarpShape::kK % 16),
    "Threadblock-scoped GEMM should be divisible by 16."
  );

  /// Number of threads per warp
  static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  /// Size of a threadblock-scoped access
  static int const kAccessSizeInBits = 32;

  /// Number of A elemnts per access
  static int const kElementsPerAccessA = kAccessSizeInBits / sizeof_bits<ElementA>::value;

  /// Number of A elemnts per access
  static int const kElementsPerAccessB = kAccessSizeInBits / sizeof_bits<ElementB>::value;

  //
  // Shared memory layouts
  //

  #if BLOCK_LOAD_STORE
    using SmemLayoutA = layout::TensorOpMultiplicand<sizeof_bits<ElementA>::value, LayoutA>;
    using SmemLayoutB = layout::TensorOpEm<sizeof_bits<ElementB>::value, LayoutB>;
  #else
    using SmemLayoutA = layout::TensorOpMultiplicand<sizeof_bits<ElementA>::value, LayoutA>;
    using SmemLayoutB = layout::TensorOpMultiplicand<sizeof_bits<ElementB>::value, LayoutB>;
  #endif

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  ///
  using IteratorThreadMapA = transform::PitchLinear2DThreadTileWarpRakedThreadMap<
    layout::PitchLinearShape<Shape::kM, Shape::kK>,
    kThreads,
    WarpThreadArrangement,
    layout::PitchLinearShape<kElementsPerAccessB, kElementsPerAccessB>
  >;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileIterator<
    MatrixShape<Shape::kM, Shape::kK>,
    ElementA,
    SmemLayoutA,
    1,
    IteratorThreadMapA
  >;

  /// Policy of iterator B
  using IteratorThreadMapB = transform::PitchLinear2DThreadTileWarpRakedThreadMap<
    layout::PitchLinearShape<Shape::kN, Shape::kK>,
    kThreads,
    WarpThreadArrangement,
    layout::PitchLinearShape<kElementsPerAccessA, kElementsPerAccessA>
  >;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileIterator<
    MatrixShape<Shape::kK, Shape::kN>,
    ElementB,
    SmemLayoutB,
    0,
    IteratorThreadMapB
  >;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level tensor op
  using Policy = gemm::warp::MmaTensorOpPolicy<
    arch::Mma<
      gemm::GemmShape<16, 16, 16>,
      NUM_THREADS_PER_WARP,
      ElementA,
      LayoutA,
      ElementB,
      LayoutB,
      ElementC,
      layout::RowMajor,
      arch::OpMultiplyAdd
    >,
    MatrixShape<1, 1>
  >;

  using MmaTensorOp = typename gemm::warp::DefaultMmaTensorOp<
    WarpShape,
    gemm::GemmShape<16, 16, 16>,
    ElementA,
    SmemLayoutA,
    ElementB,
    SmemLayoutB,
    ElementC,
    LayoutC,
    arch::OpMultiplyAdd
  >::Type;

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<
    MmaTensorOp,
    MatrixShape<0, 0>,
    MatrixShape<0, 0>,
    WarpCount::kK
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
///
/// Specialization: A: column-major, B: column-major, NN
///
/// This uses the default warp-level operator given tile sizes
///
template <
    /// Shape of threadblock-scoped matrix multiply operator (concept:
    /// GemmShape)
    typename Shape_,
    /// Shape of warp-level matrix multiply operator (concept: GemmShape)
    typename WarpShape_,
    /// Data type of A operand
    typename ElementA_,
    /// Data type of B operand
    typename ElementB_,
    /// Data type of accumulator
    typename ElementC_,
    /// Layout of accumulator
    typename LayoutC_,
    /// Stages
    int Stages,
    /// Operation performed by GEMM
    typename Operator_>
struct DefaultMmaCore<Shape_,
                      WarpShape_,
                      GemmShape<16, 16, 16>,
                      ElementA_,
                      layout::ColumnMajor,
                      ElementB_,
                      layout::ColumnMajor,
                      ElementC_,
                      LayoutC_,
                      arch::OpClassTensorOp,
                      Stages,
                      Operator_> {
  using Shape = Shape_;
  using WarpShape = WarpShape_;
  using InstructionShape = GemmShape<16, 16, 16>;
  using ElementA = ElementA_;
  using LayoutA = layout::ColumnMajor;
  using ElementB = ElementB_;
  using LayoutB = layout::ColumnMajor;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using OperatorClass = arch::OpClassTensorOp;

  static int const kStages = Stages;

  /// Default Operator
  using Operator = Operator_;

  /// Warp thread arrangement
  using WarpThreadArrangement = layout::PitchLinearShape<16, 4>;

  /// Number of warps present
  using WarpCount = GemmShape<
    Shape::kM / WarpShape::kM,
    Shape::kN / WarpShape::kN,
    Shape::kK / WarpShape::kK
  >;

  /// Don't support split K within CTA
  static_assert(Shape::kK == WarpShape::kK,
    "Threadblock-scoped GEMM shape K should equal warp-scoped GEMM shape K"
  );

  // Divisibility requirements
  static_assert(
    !(Shape::kM % WarpShape::kM) &&
    !(Shape::kN % WarpShape::kN) &&
    !(Shape::kK % WarpShape::kK),
    "Threadblock-scoped GEMM should be divisible by warp-scoped GEMM size."
  );

  // Divisibility requirements
  static_assert(
    !(WarpShape::kM % 16) &&
    !(WarpShape::kN % 16) &&
    !(WarpShape::kK % 16),
    "Threadblock-scoped GEMM should be divisible by 16."
  );

  /// Number of threads per warp
  static int const kWarpSize = warp::WarpSize<arch::OpClassTensorOp>::value;

  /// Number of threads total
  static int const kThreads = WarpCount::kCount * kWarpSize;

  /// Size of a threadblock-scoped access
  static int const kAccessSizeInBits = 32;

  /// Number of A elemnts per access
  static int const kElementsPerAccessA = kAccessSizeInBits / sizeof_bits<ElementA>::value;

  /// Number of A elemnts per access
  static int const kElementsPerAccessB = kAccessSizeInBits / sizeof_bits<ElementB>::value;

  //
  // Shared memory layouts
  //
  using SmemLayoutA = layout::TensorOpMultiplicand<sizeof_bits<ElementA>::value, LayoutA>;
  using SmemLayoutB = layout::TensorOpMultiplicand<sizeof_bits<ElementB>::value, LayoutB>;

  //

  //
  // Iterators to write to shared memory
  //

  /// ThreadMap of iterator A
  ///
  using IteratorThreadMapA = transform::PitchLinear2DThreadTileWarpRakedThreadMap<
    layout::PitchLinearShape<Shape::kM, Shape::kK>,
    kThreads,
    WarpThreadArrangement,
    layout::PitchLinearShape<kElementsPerAccessA, kElementsPerAccessA>
  >;

  /// Shared memory iterator to A operand
  using SmemIteratorA = transform::threadblock::RegularTileIterator<
    MatrixShape<Shape::kM, Shape::kK>,
    ElementA,
    SmemLayoutA,
    1,
    IteratorThreadMapA
  >;

  /// Policy of iterator B
  using IteratorThreadMapB = transform::PitchLinear2DThreadTileWarpRakedThreadMap<
    layout::PitchLinearShape<Shape::kK, Shape::kN>,
    kThreads,
    WarpThreadArrangement,
    layout::PitchLinearShape<kElementsPerAccessB, kElementsPerAccessB>
  >;

  /// Shared memory iterator to B operand
  using SmemIteratorB = transform::threadblock::RegularTileIterator<
    MatrixShape<Shape::kK, Shape::kN>,
    ElementB,
    SmemLayoutB,
    0,
    IteratorThreadMapB
  >;

  //
  // Warp-level matrix multiply operator
  //

  // Define the warp-level tensor op
  using Policy = gemm::warp::MmaTensorOpPolicy<
    arch::Mma<
      gemm::GemmShape<16, 16, 16>,
      NUM_THREADS_PER_WARP,
      ElementA,
      LayoutA,
      ElementB,
      LayoutB,
      ElementC,
      layout::RowMajor,
      arch::OpMultiplyAdd
    >,
    MatrixShape<1, 1>
  >;

  using MmaTensorOp = typename gemm::warp::DefaultMmaTensorOp<
    WarpShape,
    gemm::GemmShape<16, 16, 16>,
    ElementA,
    SmemLayoutA,
    ElementB,
    SmemLayoutB,
    ElementC,
    LayoutC,
    arch::OpMultiplyAdd
  >::Type;

  /// Policy used to define MmaPipelined
  using MmaPolicy = MmaPolicy<
    MmaTensorOp,
    MatrixShape<0, 0>,
    MatrixShape<0, 0>,
    WarpCount::kK
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
