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
    \brief Defines iterators used by warp-level matrix multiply operations targeting Tensor Cores.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/gemm/gemm.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor_op_multiplicand.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {

////////////////////////////////////////////////////////////////////////////////

template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Operand identity
    Operand Operand,
    /// Data type of elements
    typename Element_,
    /// Layout of operand
    typename Layout_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Number of threads participating in one matrix operation
    int Threads,
    /// Number of partitions along K dimension
    int PartitionsK = 1>
class MmaTensorOpMultiplicandTileIterator;

////////////////////////////////////////////////////////////////////////////////
/// Specialization for row-major A operands of 4bytes width type
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Data type of elements
  typename Element_,
  /// Shape of one matrix product operation (conecpt: PitchLinearShape)
  typename InstructionShape_,
  /// Number of partitions along K dimension
  int PartitionsK>
class MmaTensorOpMultiplicandTileIterator<
  Shape_,
  Operand::kA,
  Element_,
  layout::TensorOpMultiplicand<32, layout::RowMajor>,
  InstructionShape_,
  NUM_THREADS_PER_WARP,
  PartitionsK> {

public:

  /// Shape of tile to load (Concept: MatrixShape)
  using Shape = Shape_;

  /// Operand type
  static Operand const kOperand = Operand::kA;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = layout::TensorOpMultiplicand<32, layout::RowMajor>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Number of participating threads
  static int const kThreads = NUM_THREADS_PER_WARP;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Packed Size
  static int const kPackedSize = 32 / sizeof_bits<Element>::value;

  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero");
  static_assert(Shape::kColumn> 0, "Shape::kColumn must be greater than zero");
  static_assert(Shape::kColumn == InstructionShape::kK, "Shape::kColumn must equal InstructionShape::kK");
  static_assert((!(Shape::kRow % layout::EmShape::kRow) &&
                 !(Shape::kColumn % layout::EmShape::kColumn)),
                 "Shape must be divisibile by EM shape.");

  /// Internal structure of iterator - made public to enable introspection
  struct Detail {

    /// Determine access shape in units of elements per access
    using AccessShape = layout::PitchLinearShape<kPackedSize, 1>;

    /// Determine iterations
    using Iterations = MatrixShape<Shape::kRow / InstructionShape::kM, 1>;
  };

  /// Access type
  using AccessType = Array<Element, Detail::AccessShape::kCount>;

public:

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kCount / kThreads>;

private:

  /// Stride
  int stride_;

  /// Offset in units of element
  int offset_ = 0;

  /// Pointers holding same stride
  Element* pointers_[4];

public:

  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef ref,
    int lane_id
  ) : stride_(ref.stride(0)) {

    Element* ptr = ref.data() + ref.offset({lane_id / 16, lane_id % 16});

    CUTLASS_PRAGMA_UNROLL
    for(int i = 0; i < 4; ++i) {
      pointers_[i] = ptr + i * 64;
    }
  }

  /// Adds a pointer offset to interal pointer(s) to advance through memory
  /// So far, isn't used anywhere. Offset must be muliple of Layout::TileShape
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    offset_ += int(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    offset_ += tile_offset.row() * Shape::kRow * stride_ +
               tile_offset.column() * (Shape::kColumn / layout::EmShape::kColumn) * layout::EmShape::kCount;
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {
    add_tile_offset({0, 1});
    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    add_tile_offset({0, -1});
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int idx = 0;
    int offset = ptr_offset + offset_;

    CUTLASS_PRAGMA_UNROLL
    for(int r = 0; r < Detail::Iterations::kRow; ++r) {
      CUTLASS_PRAGMA_UNROLL
      for(int i = 0; i < 4; ++i) {
        dst_ptr[idx] = *reinterpret_cast<AccessType*>(pointers_[i] + offset);
        ++idx;
      }
      offset += stride_ * 16;
    }
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};

////////////////////////////////////////////////////////////////////////////////
/// Specialization for column-major A operands of 4bytes width type
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Data type of elements
  typename Element_,
  /// Shape of one matrix product operation (conecpt: PitchLinearShape)
  typename InstructionShape_,
  /// Number of partitions along K dimension
  int PartitionsK>
class MmaTensorOpMultiplicandTileIterator<
  Shape_,
  Operand::kA,
  Element_,
  layout::TensorOpMultiplicand<32, layout::ColumnMajor>,
  InstructionShape_,
  NUM_THREADS_PER_WARP,
  PartitionsK> {

public:

  /// Shape of tile to load (Concept: MatrixShape)
  using Shape = Shape_;

  /// Operand type
  static Operand const kOperand = Operand::kA;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = layout::TensorOpMultiplicand<32, layout::ColumnMajor>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Number of participating threads
  static int const kThreads = NUM_THREADS_PER_WARP;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Packed Size
  static int const kPackedSize = 32 / sizeof_bits<Element>::value;

  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero");
  static_assert(Shape::kColumn> 0, "Shape::kColumn must be greater than zero");
  static_assert(Shape::kColumn == InstructionShape::kK, "Shape::kColumn must equal InstructionShape::kK");
  static_assert((!(Shape::kRow % layout::EmShape::kRow) &&
                 !(Shape::kColumn % layout::EmShape::kColumn)),
                 "Shape must be divisibile by EM shape.");

  /// Internal structure of iterator - made public to enable introspection
  struct Detail {

    /// Determine access shape in units of elements per access
    using AccessShape = layout::PitchLinearShape<kPackedSize, 1>;

    /// Determine iterations
    using Iterations = MatrixShape<Shape::kRow / InstructionShape::kM, 1>;
  };

  /// Access type
  using AccessType = Array<Element, Detail::AccessShape::kCount>;

public:

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kCount / kThreads>;

private:

  /// Stride
  int stride_;

  /// Offset in units of element
  int offset_ = 0;

  /// Pointers holding same stride
  Element* pointers_[4];

public:

  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef ref,
    int lane_id
  ) : stride_(ref.stride(0)) {

    int const r = lane_id / 16;
    int const c = lane_id % 16;
    Element* ptr = ref.data();

    CUTLASS_PRAGMA_UNROLL
    for(int i = 0; i < 4; ++i) {
      pointers_[i] = ptr + ref.offset({r + i * 4, c});
    }
  }

  /// Adds a pointer offset to interal pointer(s) to advance through memory
  /// So far, isn't used anywhere. Offset must be muliple of Layout::TileShape
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    offset_ += int(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    offset_ += tile_offset.column() * Shape::kColumn * stride_ +
               tile_offset.row() * (Shape::kRow / layout::EmShape::kRow) * layout::EmShape::kCount;
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {
    add_tile_offset({0, 1});
    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    add_tile_offset({0, -1});
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int idx = 0;
    int offset = ptr_offset + offset_;

    CUTLASS_PRAGMA_UNROLL
    for(int r = 0; r < Detail::Iterations::kRow; ++r) {
      CUTLASS_PRAGMA_UNROLL
      for(int i = 0; i < 4; ++i) {
        dst_ptr[idx] = *reinterpret_cast<AccessType*>(pointers_[i] + offset);
        ++idx;
      }
      offset += layout::EmShape::kCount;
    }
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};

////////////////////////////////////////////////////////////////////////////////
/// Specialization for row-major B operands of 4bytes width type
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Data type of elements
  typename Element_,
  /// Shape of one matrix product operation (conecpt: PitchLinearShape)
  typename InstructionShape_,
  /// Number of partitions along K dimension
  int PartitionsK>
class MmaTensorOpMultiplicandTileIterator<
  Shape_,
  Operand::kB,
  Element_,
  layout::TensorOpMultiplicand<32, layout::RowMajor>,
  InstructionShape_,
  NUM_THREADS_PER_WARP,
  PartitionsK> {

public:

  /// Shape of tile to load (Concept: MatrixShape)
  using Shape = Shape_;

  /// Operand type
  static Operand const kOperand = Operand::kA;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = layout::TensorOpMultiplicand<32, layout::RowMajor>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Number of participating threads
  static int const kThreads = NUM_THREADS_PER_WARP;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Packed Size
  static int const kPackedSize = 32 / sizeof_bits<Element>::value;

  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero");
  static_assert(Shape::kColumn> 0, "Shape::kColumn must be greater than zero");
  static_assert(Shape::kRow == InstructionShape::kK, "Shape::kRow must equal InstructionShape::kK");
  static_assert((!(Shape::kRow % layout::EmShape::kRow) &&
                 !(Shape::kColumn % layout::EmShape::kColumn)),
                 "Shape must be divisibile by EM shape.");

  /// Internal structure of iterator - made public to enable introspection
  struct Detail {

    /// Determine access shape in units of elements per access
    using AccessShape = layout::PitchLinearShape<kPackedSize, 1>;

    /// Determine iterations
    using Iterations = MatrixShape<1, Shape::kColumn / InstructionShape::kN>;
  };

  /// Access type
  using AccessType = Array<Element, Detail::AccessShape::kCount>;

public:

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kCount / kThreads>;

private:

  /// Stride
  int stride_;

  /// Offset in units of element
  int offset_ = 0;

  /// Pointers holding same stride
  Element* pointers_[4];

public:

  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef ref,
    int lane_id
  ) : stride_(ref.stride(0)) {

    Element* ptr = ref.data();
    int const r = lane_id / 16;
    int const c = lane_id % 16;

    CUTLASS_PRAGMA_UNROLL
    for(int i = 0; i < 4; ++i) {
      pointers_[i] = ptr + ref.offset({r + i * 4, c});
    }
  }

  /// Adds a pointer offset to interal pointer(s) to advance through memory
  /// So far, isn't used anywhere. Offset must be muliple of Layout::TileShape
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    offset_ += int(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    offset_ += tile_offset.row() * Shape::kRow * stride_ +
               tile_offset.column() * (Shape::kColumn / layout::EmShape::kColumn) * layout::EmShape::kCount;
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {
    add_tile_offset({1, 0});
    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    add_tile_offset({-1, 0});
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int idx = 0;
    int offset = ptr_offset + offset_;

    CUTLASS_PRAGMA_UNROLL
    for(int c = 0; c < Detail::Iterations::kColumn; ++c) {
      CUTLASS_PRAGMA_UNROLL
      for(int i = 0; i < 4; ++i) {
        dst_ptr[idx] = *reinterpret_cast<AccessType*>(pointers_[i] + offset);
        ++idx;
      }
      offset += layout::EmShape::kCount;
    }
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};

////////////////////////////////////////////////////////////////////////////////
/// Specialization for column-major B operands of 4bytes width type
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Data type of elements
  typename Element_,
  /// Shape of one matrix product operation (conecpt: PitchLinearShape)
  typename InstructionShape_,
  /// Number of partitions along K dimension
  int PartitionsK>
class MmaTensorOpMultiplicandTileIterator<
  Shape_,
  Operand::kB,
  Element_,
  layout::TensorOpMultiplicand<32, layout::ColumnMajor>,
  InstructionShape_,
  NUM_THREADS_PER_WARP,
  PartitionsK> {

public:

  /// Shape of tile to load (Concept: MatrixShape)
  using Shape = Shape_;

  /// Operand type
  static Operand const kOperand = Operand::kB;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = layout::TensorOpMultiplicand<32, layout::ColumnMajor>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Number of participating threads
  static int const kThreads = NUM_THREADS_PER_WARP;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Packed Size
  static int const kPackedSize = 32 / sizeof_bits<Element>::value;

  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero");
  static_assert(Shape::kColumn> 0, "Shape::kColumn must be greater than zero");
  static_assert(Shape::kRow == InstructionShape::kK, "Shape::kRow must equal InstructionShape::kK");
  static_assert((!(Shape::kRow % layout::EmShape::kRow) &&
                 !(Shape::kColumn % layout::EmShape::kColumn)),
                 "Shape must be divisibile by EM shape.");

  /// Internal structure of iterator - made public to enable introspection
  struct Detail {

    /// Determine access shape in units of elements per access
    using AccessShape = layout::PitchLinearShape<kPackedSize, 1>;

    /// Determine iterations
    using Iterations = MatrixShape<1, Shape::kColumn / InstructionShape::kN>;
  };

  /// Access type
  using AccessType = Array<Element, Detail::AccessShape::kCount>;

public:

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kCount / kThreads>;

private:

  /// Stride
  int stride_;

  /// Offset in units of element
  int offset_ = 0;

  /// Pointers holding same stride
  Element* pointers_[4];

public:

  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef ref,
    int lane_id
  ) : stride_(ref.stride(0)) {

    Element* ptr = ref.data();
    int const r = lane_id / 16;
    int const c = lane_id % 16;

    CUTLASS_PRAGMA_UNROLL
    for(int i = 0; i < 4; ++i) {
      pointers_[i] = ptr + ref.offset({r + i * 4, c});
    }
  }

  /// Adds a pointer offset to interal pointer(s) to advance through memory
  /// So far, isn't used anywhere. Offset must be muliple of Layout::TileShape
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    offset_ += int(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    offset_ += tile_offset.column() * Shape::kColumn * stride_ +
               tile_offset.row() * (Shape::kRow / layout::EmShape::kRow) * layout::EmShape::kCount;
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {
    add_tile_offset({1, 0});
    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    add_tile_offset({-1, 0});
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int idx = 0;
    int offset = ptr_offset + offset_;

    CUTLASS_PRAGMA_UNROLL
    for(int c = 0; c < Detail::Iterations::kColumn; ++c) {
      CUTLASS_PRAGMA_UNROLL
      for(int i = 0; i < 4; ++i) {
        dst_ptr[idx] = *reinterpret_cast<AccessType*>(pointers_[i] + offset);
        idx++;
      }
      offset += stride_ * 16;
    }
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///                           2bytes   (half/bhalf)                          ///
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/// Specialization for row-major A operands of 2bytes width type
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Data type of elements
  typename Element_,
  /// Shape of one matrix product operation (conecpt: PitchLinearShape)
  typename InstructionShape_,
  /// Number of partitions along K dimension
  int PartitionsK>
class MmaTensorOpMultiplicandTileIterator<
  Shape_,
  Operand::kA,
  Element_,
  layout::TensorOpMultiplicand<16, layout::RowMajor>,
  InstructionShape_,
  NUM_THREADS_PER_WARP,
  PartitionsK> {

public:

  /// Shape of tile to load (Concept: MatrixShape)
  using Shape = Shape_;

  /// Operand type
  static Operand const kOperand = Operand::kA;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = layout::TensorOpMultiplicand<16, layout::RowMajor>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Number of participating threads
  static int const kThreads = NUM_THREADS_PER_WARP;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Packed Size
  static int const kPackedSize = 32 / sizeof_bits<Element>::value;

  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero");
  static_assert(Shape::kColumn> 0, "Shape::kColumn must be greater than zero");
  static_assert(Shape::kColumn == InstructionShape::kK, "Shape::kColumn must equal InstructionShape::kK");
  static_assert((!(Shape::kRow % layout::EmShape::kRow) &&
                 !(Shape::kColumn % layout::EmShape::kColumn)),
                 "Shape must be divisibile by EM shape.");

  /// Internal structure of iterator - made public to enable introspection
  struct Detail {

    /// Determine access shape in units of elements per access
    using AccessShape = layout::PitchLinearShape<kPackedSize, 1>;

    /// Determine iterations
    using Iterations = MatrixShape<Shape::kRow / InstructionShape::kM, 1>;
  };

  /// Access type
  using AccessType = Array<Element, Detail::AccessShape::kCount>;

public:

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kCount / kThreads>;

private:

  /// Stride
  int stride_;

  /// Pointers holding same stride
  Element* pointers_[2][2];

  /// Iterations along row dimension in units of em
  int iteration_row_ = 0;

  /// Iterations along column dimension in units of em
  int iteration_column_ = 0;

public:

  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef ref,
    int lane_id
  ) : stride_(ref.stride(0)) {

    Element* ptr = ref.data();
    int r = (lane_id / 16) * 2;
    int c = lane_id % 16;

    pointers_[0][0] = ptr + ref.offset({r,     c});
    pointers_[0][1] = ptr + ref.offset({r + 8, c});
    pointers_[1][0] = ptr + ref.offset({r,     c + 16});
    pointers_[1][1] = ptr + ref.offset({r + 8, c + 16});
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    iteration_row_ += tile_offset.row() * (Shape::kRow / layout::EmShape::kRow);
    iteration_column_ += tile_offset.column();
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {
    add_tile_offset({0, 1});
    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    add_tile_offset({0, -1});
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int const col_offset = (iteration_column_ / 2) * layout::EmShape::kCount * 2 + ptr_offset;
    int idx = 0;

    if(iteration_column_ & 1) {
      CUTLASS_PRAGMA_UNROLL
      for(int r = 0; r < Detail::Iterations::kRow; ++r) {
        int const row_offset = (iteration_row_ + r) * layout::EmShape::kRow * stride_;

        CUTLASS_PRAGMA_UNROLL
        for(int i = 0; i < 2; ++i) {
          dst_ptr[idx] = *reinterpret_cast<AccessType*>(
            pointers_[1][i] + col_offset + row_offset);
          ++idx;
        }
      }
    } else {
      CUTLASS_PRAGMA_UNROLL
      for(int r = 0; r < Detail::Iterations::kRow; ++r) {
        int const row_offset = (iteration_row_ + r) * layout::EmShape::kRow * stride_;

        CUTLASS_PRAGMA_UNROLL
        for(int i = 0; i < 2; ++i) {
          dst_ptr[idx] = *reinterpret_cast<AccessType*>(
            pointers_[0][i] + col_offset + row_offset);
          ++idx;
        }
      }
    }
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};

////////////////////////////////////////////////////////////////////////////////
/// Specialization for column-major A operands of 2bytes width type
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Data type of elements
  typename Element_,
  /// Shape of one matrix product operation (conecpt: PitchLinearShape)
  typename InstructionShape_,
  /// Number of partitions along K dimension
  int PartitionsK>
class MmaTensorOpMultiplicandTileIterator<
  Shape_,
  Operand::kA,
  Element_,
  layout::TensorOpMultiplicand<16, layout::ColumnMajor>,
  InstructionShape_,
  NUM_THREADS_PER_WARP,
  PartitionsK> {

public:

  /// Shape of tile to load (Concept: MatrixShape)
  using Shape = Shape_;

  /// Operand type
  static Operand const kOperand = Operand::kA;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = layout::TensorOpMultiplicand<16, layout::ColumnMajor>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Number of participating threads
  static int const kThreads = NUM_THREADS_PER_WARP;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Packed Size
  static int const kPackedSize = 32 / sizeof_bits<Element>::value;

  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero");
  static_assert(Shape::kColumn> 0, "Shape::kColumn must be greater than zero");
  static_assert((!(Shape::kRow % layout::EmShape::kRow) &&
                 !(Shape::kColumn % layout::EmShape::kColumn)),
                 "Shape must be divisibile by EM shape.");

  /// Internal structure of iterator - made public to enable introspection
  struct Detail {

    /// Determine access shape in units of elements per access
    using AccessShape = layout::PitchLinearShape<kPackedSize, 1>;

    /// Determine iterations
    using Iterations = MatrixShape<Shape::kRow / InstructionShape::kM, 1>;
  };

  /// Access type
  using AccessType = Array<Element, Detail::AccessShape::kCount>;

public:

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kCount / kThreads>;

private:

  /// Stride
  int stride_;

  /// Pointers holding same stride
  Element* pointers_[2][2];

  /// Iterations along row dimension in units of em
  int iteration_row_ = 0;

  /// Iterations along column dimension in units of em
  int iteration_column_ = 0;

public:

  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef ref,
    int lane_id
  ) : stride_(ref.stride(0)) {

    Element* ptr = ref.data();
    int r = (lane_id / 16) * 2;
    int c = lane_id % 16;

    pointers_[0][0] = ptr + ref.offset({r,      c});
    pointers_[0][1] = ptr + ref.offset({r + 8,  c});
    pointers_[1][0] = ptr + ref.offset({r + 16, c});
    pointers_[1][1] = ptr + ref.offset({r + 24, c});
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    iteration_row_ += tile_offset.row() * (Shape::kRow / layout::EmShape::kRow);
    iteration_column_ += tile_offset.column();
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {
    add_tile_offset({0, 1});
    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    add_tile_offset({0, -1});
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int idx = 0;
    int const col_offset = iteration_column_ * layout::EmShape::kColumn * stride_ + ptr_offset;

    if(iteration_row_ & 1) {
      CUTLASS_PRAGMA_UNROLL
      for(int r = 0; r < Detail::Iterations::kRow; ++r) {
        int const row_offset = ((iteration_row_ + r) / 2) * layout::EmShape::kCount * 2;

        CUTLASS_PRAGMA_UNROLL
        for(int i = 0; i < 2; ++i) {
          dst_ptr[idx] = *reinterpret_cast<AccessType*>(
            pointers_[(r + 1) & 1][i] + col_offset + row_offset);
          ++idx;
        }
      }
    } else {
      CUTLASS_PRAGMA_UNROLL
      for(int r = 0; r < Detail::Iterations::kRow; ++r) {
        int const row_offset = ((iteration_row_ + r) / 2) * layout::EmShape::kCount * 2;

        CUTLASS_PRAGMA_UNROLL
        for(int i = 0; i < 2; ++i) {
          dst_ptr[idx] = *reinterpret_cast<AccessType*>(
            pointers_[r & 1][i] + col_offset + row_offset);
          ++idx;
        }
      }
    }
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};

////////////////////////////////////////////////////////////////////////////////
/// Specialization for row-major B operands of 2bytes width type
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Data type of elements
  typename Element_,
  /// Shape of one matrix product operation (conecpt: PitchLinearShape)
  typename InstructionShape_,
  /// Number of partitions along K dimension
  int PartitionsK>
class MmaTensorOpMultiplicandTileIterator<
  Shape_,
  Operand::kB,
  Element_,
  layout::TensorOpMultiplicand<16, layout::RowMajor>,
  InstructionShape_,
  NUM_THREADS_PER_WARP,
  PartitionsK> {

public:

  /// Shape of tile to load (Concept: MatrixShape)
  using Shape = Shape_;

  /// Operand type
  static Operand const kOperand = Operand::kB;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = layout::TensorOpMultiplicand<16, layout::RowMajor>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Number of participating threads
  static int const kThreads = NUM_THREADS_PER_WARP;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Packed Size
  static int const kPackedSize = 32 / sizeof_bits<Element>::value;

  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero");
  static_assert(Shape::kColumn> 0, "Shape::kColumn must be greater than zero");
  static_assert((!(Shape::kRow % layout::EmShape::kRow) &&
                 !(Shape::kColumn % layout::EmShape::kColumn)),
                 "Shape must be divisibile by EM shape.");

  /// Internal structure of iterator - made public to enable introspection
  struct Detail {

    /// Determine access shape in units of elements per access
    using AccessShape = layout::PitchLinearShape<kPackedSize, 1>;

    /// Determine iterations
    using Iterations = MatrixShape<1, Shape::kColumn / InstructionShape::kN>;
  };

  /// Access type
  using AccessType = Array<Element, Detail::AccessShape::kCount>;

public:

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kCount / kThreads>;

private:

  /// Stride
  int stride_;

  /// Pointers holding same stride
  Element* pointers_[2][2];

  /// Iterations along row dimension in units of em
  int iteration_row_ = 0;

  /// Iterations along column dimension in units of em
  int iteration_column_ = 0;

public:

  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef ref,
    int lane_id
  ) : stride_(ref.stride(0)) {

    int r = (lane_id / 16) * 2;
    int c = lane_id % 16;

    pointers_[0][0] = ref.data() + ref.offset({r,     c});
    pointers_[0][1] = ref.data() + ref.offset({r + 8, c});
    pointers_[1][0] = ref.data() + ref.offset({r,     c + 16});
    pointers_[1][1] = ref.data() + ref.offset({r + 8, c + 16});
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    iteration_row_ += tile_offset.row();
    iteration_column_ += tile_offset.column() * (Shape::kColumn / layout::EmShape::kColumn);
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {
    add_tile_offset({1, 0});
    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    add_tile_offset({-1, 0});
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int idx = 0;
    int const row_offset = iteration_row_ * layout::EmShape::kRow * stride_ + ptr_offset;

    if(iteration_column_ & 1) {
      CUTLASS_PRAGMA_UNROLL
      for(int c = 0; c < Detail::Iterations::kColumn; ++c) {
        int const col_offset = ((iteration_column_ + c) / 2) * layout::EmShape::kCount * 2;

        CUTLASS_PRAGMA_UNROLL
        for(int i = 0; i < 2; ++i) {
          dst_ptr[idx] = *reinterpret_cast<AccessType*>(pointers_[(c + 1)& 1][i] + col_offset + row_offset);
          ++idx;
        }
      }
    } else {
      CUTLASS_PRAGMA_UNROLL
      for(int c = 0; c < Detail::Iterations::kColumn; ++c) {
        int const col_offset = ((iteration_column_ + c) / 2) * layout::EmShape::kCount * 2;

        CUTLASS_PRAGMA_UNROLL
        for(int i = 0; i < 2; ++i) {
          dst_ptr[idx] = *reinterpret_cast<AccessType*>(pointers_[c & 1][i] + col_offset + row_offset);
          ++idx;
        }
      }
    }
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};

////////////////////////////////////////////////////////////////////////////////
/// Specialization for column-major B operands of 2bytes width type
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Data type of elements
  typename Element_,
  /// Shape of one matrix product operation (conecpt: PitchLinearShape)
  typename InstructionShape_,
  /// Number of partitions along K dimension
  int PartitionsK>
class MmaTensorOpMultiplicandTileIterator<
  Shape_,
  Operand::kB,
  Element_,
  layout::TensorOpMultiplicand<16, layout::ColumnMajor>,
  InstructionShape_,
  NUM_THREADS_PER_WARP,
  PartitionsK> {

public:

  /// Shape of tile to load (Concept: MatrixShape)
  using Shape = Shape_;

  /// Operand type
  static Operand const kOperand = Operand::kB;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = layout::TensorOpMultiplicand<16, layout::ColumnMajor>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Number of participating threads
  static int const kThreads = NUM_THREADS_PER_WARP;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Packed Size
  static int const kPackedSize = 32 / sizeof_bits<Element>::value;

  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero");
  static_assert(Shape::kColumn> 0, "Shape::kColumn must be greater than zero");
  static_assert((!(Shape::kRow % layout::EmShape::kRow) &&
                 !(Shape::kColumn % layout::EmShape::kColumn)),
                 "Shape must be divisibile by EM shape.");

  /// Internal structure of iterator - made public to enable introspection
  struct Detail {

    /// Determine access shape in units of elements per access
    using AccessShape = layout::PitchLinearShape<kPackedSize, 1>;

    /// Determine iterations
    using Iterations = MatrixShape<1, Shape::kColumn / InstructionShape::kN>;
  };

  /// Access type
  using AccessType = Array<Element, Detail::AccessShape::kCount>;

public:

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kCount / kThreads>;

private:

  /// Stride
  int stride_;

  /// Pointers holding same stride
  Element* pointers_[2][2];

  /// Iterations along row dimension in units of em
  int iteration_row_ = 0;

  /// Iterations along column dimension in units of em
  int iteration_column_ = 0;

public:

  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef ref,
    int lane_id
  ) : stride_(ref.stride(0)) {

    Element* ptr = ref.data();
    int r = (lane_id / 16) * 2;
    int c = lane_id % 16;

    pointers_[0][0] = ptr + int(ref.offset({r,      c}));
    pointers_[0][1] = ptr + int(ref.offset({r + 8,  c}));
    pointers_[1][0] = ptr + int(ref.offset({r + 16, c}));
    pointers_[1][1] = ptr + int(ref.offset({r + 24, c}));
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    iteration_row_ += tile_offset.row();
    iteration_column_ += tile_offset.column() * (Shape::kColumn / layout::EmShape::kColumn);
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {
    add_tile_offset({1, 0});
    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    add_tile_offset({-1, 0});
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int const row_offset = (iteration_row_ / 2) * layout::EmShape::kCount * 2 + ptr_offset;
    int idx = 0;

    if(iteration_row_ & 1) {
      CUTLASS_PRAGMA_UNROLL
      for(int c = 0; c < Detail::Iterations::kColumn; ++c) {
        int const col_offset = (iteration_column_ + c) * layout::EmShape::kColumn * stride_;

        CUTLASS_PRAGMA_UNROLL
        for(int i = 0; i < 2; ++i) {
          dst_ptr[idx] = *reinterpret_cast<AccessType*>(
            pointers_[1][i] + row_offset + col_offset);
          idx++;
        }
      }
    } else {
      CUTLASS_PRAGMA_UNROLL
      for(int c = 0; c < Detail::Iterations::kColumn; ++c) {
        int const col_offset = (iteration_column_ + c) * layout::EmShape::kColumn * stride_;

        CUTLASS_PRAGMA_UNROLL
        for(int i = 0; i < 2; ++i) {
          dst_ptr[idx] = *reinterpret_cast<AccessType*>(
            pointers_[0][i] + row_offset + col_offset);
          idx++;
        }
      }
    }
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///                           1byte   (int8/uint8)                          ///
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/// Specialization for row-major A operands of 1byte width type
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Data type of elements
  typename Element_,
  /// Shape of one matrix product operation (conecpt: PitchLinearShape)
  typename InstructionShape_,
  /// Number of partitions along K dimension
  int PartitionsK>
class MmaTensorOpMultiplicandTileIterator<
  Shape_,
  Operand::kA,
  Element_,
  layout::TensorOpMultiplicand<8, layout::RowMajor>,
  InstructionShape_,
  NUM_THREADS_PER_WARP,
  PartitionsK> {

public:

  /// Shape of tile to load (Concept: MatrixShape)
  using Shape = Shape_;

  /// Operand type
  static Operand const kOperand = Operand::kA;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = layout::TensorOpMultiplicand<8, layout::RowMajor>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Number of participating threads
  static int const kThreads = NUM_THREADS_PER_WARP;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Packed Size
  static int const kPackedSize = 32 / sizeof_bits<Element>::value;

  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero");
  static_assert(Shape::kColumn> 0, "Shape::kColumn must be greater than zero");
  static_assert(Shape::kColumn == InstructionShape::kK, "Shape::kColumn must equal InstructionShape::kK");
  static_assert((!(Shape::kRow % layout::EmShape::kRow) &&
                 !(Shape::kColumn % layout::EmShape::kColumn)),
                 "Shape must be divisibile by EM shape.");

  /// Internal structure of iterator - made public to enable introspection
  struct Detail {

    /// Determine access shape in units of elements per access
    using AccessShape = layout::PitchLinearShape<kPackedSize, 1>;

    /// Determine iterations
    using Iterations = MatrixShape<Shape::kRow / InstructionShape::kM, 1>;
  };

  /// Access type
  using AccessType = Array<Element, Detail::AccessShape::kCount>;

public:

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kCount / kThreads>;

private:

  /// Stride
  int stride_;

  /// Pointers holding same stride
  Element* pointers_[4];

  /// Iterations along row dimension in units of em
  int iteration_row_ = 0;

  /// Iterations along column dimension in units of em
  int iteration_column_ = 0;

public:

  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef ref,
    int lane_id
  ) : stride_(ref.stride(0)) {

    Element* ptr = ref.data();
    int r = (lane_id / 16) * 4;
    int c = lane_id % 16;

    pointers_[0] = ptr + ref.offset({r, c});
    pointers_[1] = ptr + ref.offset({r, c + 16});
    pointers_[2] = ptr + ref.offset({r, c + 32});
    pointers_[3] = ptr + ref.offset({r, c + 48});
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    iteration_row_ += tile_offset.row() * (Shape::kRow / layout::EmShape::kRow);
    iteration_column_ += tile_offset.column();
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {
    add_tile_offset({0, 1});
    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    add_tile_offset({0, -1});
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int const col_offset = (iteration_column_ / 4) * layout::EmShape::kCount * 4 + ptr_offset;
    int idx = 0;

    if((iteration_column_ & 3) == 3) {
      CUTLASS_PRAGMA_UNROLL
      for(int r = 0; r < Detail::Iterations::kRow; ++r) {
        int const row_offset = (iteration_row_ + r) * layout::EmShape::kRow * stride_;

        dst_ptr[idx] = *reinterpret_cast<AccessType*>(
          pointers_[3] + col_offset + row_offset);
        ++idx;
      }
    } else if((iteration_column_ & 3) == 2) {
      CUTLASS_PRAGMA_UNROLL
      for(int r = 0; r < Detail::Iterations::kRow; ++r) {
        int const row_offset = (iteration_row_ + r) * layout::EmShape::kRow * stride_;

        dst_ptr[idx] = *reinterpret_cast<AccessType*>(
          pointers_[2] + col_offset + row_offset);
        ++idx;
      }
    } else if((iteration_column_ & 3) == 1) {
      CUTLASS_PRAGMA_UNROLL
      for(int r = 0; r < Detail::Iterations::kRow; ++r) {
        int const row_offset = (iteration_row_ + r) * layout::EmShape::kRow * stride_;

        dst_ptr[idx] = *reinterpret_cast<AccessType*>(
          pointers_[1] + col_offset + row_offset);
        ++idx;
      }
    } else {
      CUTLASS_PRAGMA_UNROLL
      for(int r = 0; r < Detail::Iterations::kRow; ++r) {
        int const row_offset = (iteration_row_ + r) * layout::EmShape::kRow * stride_;

        dst_ptr[idx] = *reinterpret_cast<AccessType*>(
          pointers_[0] + col_offset + row_offset);
        ++idx;
      }
    }
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};

////////////////////////////////////////////////////////////////////////////////
/// Specialization for column-major A operands of 1byte width type
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Data type of elements
  typename Element_,
  /// Shape of one matrix product operation (conecpt: PitchLinearShape)
  typename InstructionShape_,
  /// Number of partitions along K dimension
  int PartitionsK>
class MmaTensorOpMultiplicandTileIterator<
  Shape_,
  Operand::kA,
  Element_,
  layout::TensorOpMultiplicand<8, layout::ColumnMajor>,
  InstructionShape_,
  NUM_THREADS_PER_WARP,
  PartitionsK> {

public:

  /// Shape of tile to load (Concept: MatrixShape)
  using Shape = Shape_;

  /// Operand type
  static Operand const kOperand = Operand::kA;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = layout::TensorOpMultiplicand<8, layout::ColumnMajor>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Number of participating threads
  static int const kThreads = NUM_THREADS_PER_WARP;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Packed Size
  static int const kPackedSize = 32 / sizeof_bits<Element>::value;

  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero");
  static_assert(Shape::kColumn> 0, "Shape::kColumn must be greater than zero");
  static_assert((!(Shape::kRow % layout::EmShape::kRow) &&
                 !(Shape::kColumn % layout::EmShape::kColumn)),
                 "Shape must be divisibile by EM shape.");

  /// Internal structure of iterator - made public to enable introspection
  struct Detail {

    /// Determine access shape in units of elements per access
    using AccessShape = layout::PitchLinearShape<kPackedSize, 1>;

    /// Determine iterations
    using Iterations = MatrixShape<Shape::kRow / InstructionShape::kM, 1>;
  };

  /// Access type
  using AccessType = Array<Element, Detail::AccessShape::kCount>;

public:

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kCount / kThreads>;

private:

  /// Stride
  int stride_;

  /// Pointers holding same stride
  Element* pointers_[4];

  /// Iterations along row dimension in units of em
  int iteration_row_ = 0;

  /// Iterations along column dimension in units of em
  int iteration_column_ = 0;

public:

  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef ref,
    int lane_id
  ) : stride_(ref.stride(0)) {

    Element* ptr = ref.data();
    int r = (lane_id / 16) * 4;
    int c = lane_id % 16;

    pointers_[0] = ptr + ref.offset({r,      c});
    pointers_[1] = ptr + ref.offset({r + 16, c});
    pointers_[2] = ptr + ref.offset({r + 32, c});
    pointers_[3] = ptr + ref.offset({r + 48, c});
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    iteration_row_ += tile_offset.row() * (Shape::kRow / layout::EmShape::kRow);
    iteration_column_ += tile_offset.column();
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {
    add_tile_offset({0, 1});
    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    add_tile_offset({0, -1});
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int idx = 0;
    int const col_offset = iteration_column_ * layout::EmShape::kColumn * stride_ + ptr_offset;

    if((iteration_row_ & 3) == 3) {
      CUTLASS_PRAGMA_UNROLL
      for(int r = 0; r < Detail::Iterations::kRow; ++r) {
        int const row_offset = ((iteration_row_ + r) / 4) * layout::EmShape::kCount * 4;

        dst_ptr[idx] = *reinterpret_cast<AccessType*>(
          pointers_[(r + 3) & 3] + col_offset + row_offset);
        ++idx;
      }
    } else if((iteration_row_ & 3) == 2) {
      CUTLASS_PRAGMA_UNROLL
      for(int r = 0; r < Detail::Iterations::kRow; ++r) {
        int const row_offset = ((iteration_row_ + r) / 4) * layout::EmShape::kCount * 4;

        dst_ptr[idx] = *reinterpret_cast<AccessType*>(
          pointers_[(r + 2) & 3] + col_offset + row_offset);
        ++idx;
      }
    } else if((iteration_row_ & 3) == 1) {
      CUTLASS_PRAGMA_UNROLL
      for(int r = 0; r < Detail::Iterations::kRow; ++r) {
        int const row_offset = ((iteration_row_ + r) / 4) * layout::EmShape::kCount * 4;

        dst_ptr[idx] = *reinterpret_cast<AccessType*>(
          pointers_[(r + 1) & 3] + col_offset + row_offset);
        ++idx;
      }
    } else {
      CUTLASS_PRAGMA_UNROLL
      for(int r = 0; r < Detail::Iterations::kRow; ++r) {
        int const row_offset = ((iteration_row_ + r) / 4) * layout::EmShape::kCount * 4;

        dst_ptr[idx] = *reinterpret_cast<AccessType*>(
          pointers_[r & 3] + col_offset + row_offset);
        ++idx;
      }
    }
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};

////////////////////////////////////////////////////////////////////////////////
/// Specialization for row-major B operands of 1byte width type
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Data type of elements
  typename Element_,
  /// Shape of one matrix product operation (conecpt: PitchLinearShape)
  typename InstructionShape_,
  /// Number of partitions along K dimension
  int PartitionsK>
class MmaTensorOpMultiplicandTileIterator<
  Shape_,
  Operand::kB,
  Element_,
  layout::TensorOpMultiplicand<8, layout::RowMajor>,
  InstructionShape_,
  NUM_THREADS_PER_WARP,
  PartitionsK> {

public:

  /// Shape of tile to load (Concept: MatrixShape)
  using Shape = Shape_;

  /// Operand type
  static Operand const kOperand = Operand::kB;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = layout::TensorOpMultiplicand<8, layout::RowMajor>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Number of participating threads
  static int const kThreads = NUM_THREADS_PER_WARP;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Packed Size
  static int const kPackedSize = 32 / sizeof_bits<Element>::value;

  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero");
  static_assert(Shape::kColumn> 0, "Shape::kColumn must be greater than zero");
  static_assert((!(Shape::kRow % layout::EmShape::kRow) &&
                 !(Shape::kColumn % layout::EmShape::kColumn)),
                 "Shape must be divisibile by EM shape.");

  /// Internal structure of iterator - made public to enable introspection
  struct Detail {

    /// Determine access shape in units of elements per access
    using AccessShape = layout::PitchLinearShape<kPackedSize, 1>;

    /// Determine iterations
    using Iterations = MatrixShape<1, Shape::kColumn / InstructionShape::kN>;
  };

  /// Access type
  using AccessType = Array<Element, Detail::AccessShape::kCount>;

public:

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kCount / kThreads>;

private:

  /// Stride
  int stride_;

  /// Pointers holding same stride
  Element* pointers_[4];

  /// Iterations along row dimension in units of em
  int iteration_row_ = 0;

  /// Iterations along column dimension in units of em
  int iteration_column_ = 0;

public:

  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef ref,
    int lane_id
  ) : stride_(ref.stride(0)) {

    int r = (lane_id / 16) * 4;
    int c = lane_id % 16;

    pointers_[0] = ref.data() + ref.offset({r, c});
    pointers_[1] = ref.data() + ref.offset({r, c + 16});
    pointers_[2] = ref.data() + ref.offset({r, c + 32});
    pointers_[3] = ref.data() + ref.offset({r, c + 48});
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    iteration_row_ += tile_offset.row();
    iteration_column_ += tile_offset.column() * (Shape::kColumn / layout::EmShape::kColumn);
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {
    add_tile_offset({1, 0});
    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    add_tile_offset({-1, 0});
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int idx = 0;
    int const row_offset = iteration_row_ * layout::EmShape::kRow * stride_ + ptr_offset;

    if((iteration_column_ & 3) == 3) {
      CUTLASS_PRAGMA_UNROLL
      for(int c = 0; c < Detail::Iterations::kColumn; ++c) {
        int const col_offset = ((iteration_column_ + c) / 4) * layout::EmShape::kCount * 4;

        dst_ptr[idx] = *reinterpret_cast<AccessType*>(pointers_[(c + 3) & 3] + col_offset + row_offset);
        ++idx;
      }
    } else if((iteration_column_ & 3) == 2) {
      CUTLASS_PRAGMA_UNROLL
      for(int c = 0; c < Detail::Iterations::kColumn; ++c) {
        int const col_offset = ((iteration_column_ + c) / 4) * layout::EmShape::kCount * 4;

        dst_ptr[idx] = *reinterpret_cast<AccessType*>(pointers_[(c + 2) & 3] + col_offset + row_offset);
        ++idx;
      }
    } else if((iteration_column_ & 3) == 1) {
      CUTLASS_PRAGMA_UNROLL
      for(int c = 0; c < Detail::Iterations::kColumn; ++c) {
        int const col_offset = ((iteration_column_ + c) / 4) * layout::EmShape::kCount * 4;

        dst_ptr[idx] = *reinterpret_cast<AccessType*>(pointers_[(c + 1) & 3] + col_offset + row_offset);
        ++idx;
      }
    } else {
      CUTLASS_PRAGMA_UNROLL
      for(int c = 0; c < Detail::Iterations::kColumn; ++c) {
        int const col_offset = ((iteration_column_ + c) / 4) * layout::EmShape::kCount * 4;

        dst_ptr[idx] = *reinterpret_cast<AccessType*>(pointers_[c & 3] + col_offset + row_offset);
        ++idx;
      }
    }
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};

////////////////////////////////////////////////////////////////////////////////
/// Specialization for column-major B operands of 1byte width type
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Data type of elements
  typename Element_,
  /// Shape of one matrix product operation (conecpt: PitchLinearShape)
  typename InstructionShape_,
  /// Number of partitions along K dimension
  int PartitionsK>
class MmaTensorOpMultiplicandTileIterator<
  Shape_,
  Operand::kB,
  Element_,
  layout::TensorOpMultiplicand<8, layout::ColumnMajor>,
  InstructionShape_,
  NUM_THREADS_PER_WARP,
  PartitionsK> {

public:

  /// Shape of tile to load (Concept: MatrixShape)
  using Shape = Shape_;

  /// Operand type
  static Operand const kOperand = Operand::kB;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = layout::TensorOpMultiplicand<8, layout::ColumnMajor>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Number of participating threads
  static int const kThreads = NUM_THREADS_PER_WARP;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Packed Size
  static int const kPackedSize = 32 / sizeof_bits<Element>::value;

  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero");
  static_assert(Shape::kColumn> 0, "Shape::kColumn must be greater than zero");
  static_assert((!(Shape::kRow % layout::EmShape::kRow) &&
                 !(Shape::kColumn % layout::EmShape::kColumn)),
                 "Shape must be divisibile by EM shape.");

  /// Internal structure of iterator - made public to enable introspection
  struct Detail {

    /// Determine access shape in units of elements per access
    using AccessShape = layout::PitchLinearShape<kPackedSize, 1>;

    /// Determine iterations
    using Iterations = MatrixShape<1, Shape::kColumn / InstructionShape::kN>;
  };

  /// Access type
  using AccessType = Array<Element, Detail::AccessShape::kCount>;

public:

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kCount / kThreads>;

private:

  /// Stride
  int stride_;

  /// Pointers holding same stride
  Element* pointers_[4];

  /// Iterations along row dimension in units of em
  int iteration_row_ = 0;

  /// Iterations along column dimension in units of em
  int iteration_column_ = 0;

public:

  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef ref,
    int lane_id
  ) : stride_(ref.stride(0)) {

    Element* ptr = ref.data();
    int r = (lane_id / 16) * 4;
    int c = lane_id % 16;

    pointers_[0] = ptr + ref.offset({r,      c});
    pointers_[1] = ptr + ref.offset({r + 16, c});
    pointers_[2] = ptr + ref.offset({r + 32, c});
    pointers_[3] = ptr + ref.offset({r + 48, c});
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    iteration_row_ += tile_offset.row();
    iteration_column_ += tile_offset.column() * (Shape::kColumn / layout::EmShape::kColumn);
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {
    add_tile_offset({1, 0});
    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    add_tile_offset({-1, 0});
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int const row_offset = (iteration_row_ / 4) * layout::EmShape::kCount * 4 + ptr_offset;
    int idx = 0;

    if((iteration_row_ & 3) == 3) {
      CUTLASS_PRAGMA_UNROLL
      for(int c = 0; c < Detail::Iterations::kColumn; ++c) {
        int const col_offset = (iteration_column_ + c) * layout::EmShape::kColumn * stride_;

        dst_ptr[idx] = *reinterpret_cast<AccessType*>(
          pointers_[3] + row_offset + col_offset);
        idx++;
      }
    } else if((iteration_row_ & 3) == 2) {
      CUTLASS_PRAGMA_UNROLL
      for(int c = 0; c < Detail::Iterations::kColumn; ++c) {
        int const col_offset = (iteration_column_ + c) * layout::EmShape::kColumn * stride_;

        dst_ptr[idx] = *reinterpret_cast<AccessType*>(
          pointers_[2] + row_offset + col_offset);
        idx++;
      }
    } else if((iteration_row_ & 3) == 1) {
      CUTLASS_PRAGMA_UNROLL
      for(int c = 0; c < Detail::Iterations::kColumn; ++c) {
        int const col_offset = (iteration_column_ + c) * layout::EmShape::kColumn * stride_;

        dst_ptr[idx] = *reinterpret_cast<AccessType*>(
          pointers_[1] + row_offset + col_offset);
        idx++;
      }
    } else {
      CUTLASS_PRAGMA_UNROLL
      for(int c = 0; c < Detail::Iterations::kColumn; ++c) {
        int const col_offset = (iteration_column_ + c) * layout::EmShape::kColumn * stride_;

        dst_ptr[idx] = *reinterpret_cast<AccessType*>(
          pointers_[0] + row_offset + col_offset);
        idx++;
      }
    }

  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};

////////////////////////////////////////////////////////////////////////////////

template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Element type
  typename Element_,
  /// Layout of operand in memory
  typename Layout_,
  /// Shape of one matrix product operation (concept: MatrixShape)
  typename InstructionShape_>
class MmaTensorOpAccumulatorTileIterator;

////////////////////////////////////////////////////////////////////////////////
/// Specialization for C operands of row-major layouts
///
/// Concept: MutableRandomAccessContiguousTileIteratorConcept |
///          WriteableRandomAccessContiguousTileIteratorConcept
///
template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Data type of A elements
  typename Element_,
  /// Shape of one matrix product operation (concept: MatrixShape)
  typename InstructionShape_
>
class MmaTensorOpAccumulatorTileIterator<
  Shape_,
  Element_,
  layout::RowMajor,
  InstructionShape_> {
public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Element type
  using Element = Element_;

  /// Layout of accumulators in memory
  using Layout = layout::RowMajor;

  using WarpThreadArrangement = layout::PitchLinearShape<16, 4>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  //
  // Derived quantities
  //

  static_assert(
    (!(Shape::kRow % WarpThreadArrangement::kStrided)) &&
    (!(Shape::kColumn % WarpThreadArrangement::kContiguous)),
    "Warp-level GEMM shape must be divisible by the arrangement of threads in the warp.");
  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero.");
  static_assert(Shape::kColumn > 0, "Shape::kColumn must be greater than zero.");
  static_assert(Shape::kRow / WarpThreadArrangement::kStrided> 0,
                "Shape::kRow / WarpThreadArrangement::kStrided must be greater than zero.");
  static_assert(Shape::kColumn / WarpThreadArrangement::kContiguous > 0,
                "Shape::kColumn / WarpThreadArrangement::kContiguous must be greater than zero.");

  /// Packed size
  static int const kPackedSize = 32 / sizeof_bits<Element>::value;

  /// Access shape
  using AccessShape = layout::PitchLinearShape<kPackedSize, 1>;

  /// Shape in vectors
  using ShapeVec = MatrixShape<
    Shape::kRow,
    Shape::kColumn / AccessShape::kContiguous
  >;

  /// Thread-level shape in vec of a fragment
  using ThreadShape = MatrixShape<
    ShapeVec::kRow / WarpThreadArrangement::kStrided,
    ShapeVec::kColumn / WarpThreadArrangement::kContiguous
  >;

  /// Number of individual loads within one instruction result
  using IterationsInner = MatrixShape<
    InstructionShape::kM / WarpThreadArrangement::kStrided / kPackedSize,
    InstructionShape::kN / WarpThreadArrangement::kContiguous
  >;

  /// Number of iterations in units of instruction shape
  using Iterations = MatrixShape<
    Shape::kRow / InstructionShape::kM,
    Shape::kColumn / InstructionShape::kN
  >;

  /// Delta in units of elements
  using Delta = MatrixShape<
    WarpThreadArrangement::kStrided,
    WarpThreadArrangement::kContiguous * AccessShape::kContiguous
  >;

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, ThreadShape::kCount * AccessShape::kCount>;

private:

  TensorRef ref_;

  MatrixCoord init_offset_;

public:

  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator(
    TensorRef const &ref,
    int lane_id
  ): ref_(ref) {

    init_offset_ = TensorCoord(
      lane_id / 16 * kPackedSize,
      lane_id % 16
    );
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_pointer_offset(LongIndex offset) {
    ref_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_tile_offset(TensorCoord const &coord) {

    ref_.add_coord_offset(coord * make_Coord(Shape::kRow, Shape::kColumn));

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator++() {
    // deliberate no-op
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator--() {
    // deliberate no-op
    return *this;
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(
    Fragment &frag,                             ///< fragment to be loaded from memory
    Index pointer_offset) const {               ///< linear offset (in units of Element) when loading

    CUTLASS_PRAGMA_UNROLL
    for(int m = 0; m < Iterations::kRow; ++m) {
      CUTLASS_PRAGMA_UNROLL
      for(int n = 0; n < Iterations::kColumn; ++n) {
        CUTLASS_PRAGMA_UNROLL
        for(int inner_n = 0; inner_n < IterationsInner::kColumn; ++inner_n) {
          CUTLASS_PRAGMA_UNROLL
          for(int inner_m = 0; inner_m < IterationsInner::kRow; ++inner_m) {
            TensorCoord offset(m * inner_m, n * inner_n);

            Array<Element, AccessShape::kCount> const * src_ptr =
              reinterpret_cast<Array<Element, AccessShape::kCount> const *>(
                ref_.data() + pointer_offset + ref_.offset(init_offset_ + offset));

            Array<Element, AccessShape::kCount> *dst_ptr =
              reinterpret_cast<Array<Element, AccessShape::kCount>*>(&frag) +
              inner_m + inner_n * IterationsInner::kRow +
              n * IterationsInner::kCount + m * Iterations::kColumn * IterationsInner::kCount;

            *dst_ptr = src_ptr[0];
          }
        }
      }
    }
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment to memory at the location pointed to by the iterator
  CUTLASS_HOST_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) const {

    CUTLASS_PRAGMA_UNROLL
    for(int m = 0; m < Iterations::kRow; ++m) {
      CUTLASS_PRAGMA_UNROLL
      for(int n = 0; n < Iterations::kColumn; ++n) {
        CUTLASS_PRAGMA_UNROLL
        for(int inner_n = 0; inner_n < IterationsInner::kColumn; ++inner_n) {
          CUTLASS_PRAGMA_UNROLL
          for(int inner_m = 0; inner_m < IterationsInner::kRow; ++inner_m) {
            TensorCoord offset((m * IterationsInner::kRow + inner_m) * Delta::kRow,
                               (n * IterationsInner::kColumn + inner_n) * Delta::kColumn);
            Array<Element, AccessShape::kCount> * dst_ptr =
              reinterpret_cast<Array<Element, AccessShape::kCount> *>(
                ref_.data() + pointer_offset + ref_.offset(offset + init_offset_));

            Array<Element, AccessShape::kCount> const * src_ptr =
              reinterpret_cast<Array<Element, AccessShape::kCount> const *>(&frag) +
              inner_m + IterationsInner::kRow * inner_n + n * IterationsInner::kCount +
              m * Iterations::kColumn * IterationsInner::kCount;

            *dst_ptr = *src_ptr;
          }
        }
      }
    }
  }

  /// Stores a fragment to memory at the location pointed to by the iterator
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) const {
    store_with_pointer_offset(frag, 0);
  }
};

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
/// Specialization for row-major A operands of 4bytes width type
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Data type of elements
  typename Element_,
  /// Shape of one matrix product operation (conecpt: PitchLinearShape)
  typename InstructionShape_,
  /// Number of partitions along K dimension
  int PartitionsK>
class MmaTensorOpMultiplicandTileIterator<
  Shape_,
  Operand::kA,
  Element_,
  layout::TensorOpEm<32, layout::RowMajor>,
  InstructionShape_,
  NUM_THREADS_PER_WARP,
  PartitionsK> {

public:

  /// Shape of tile to load (Concept: MatrixShape)
  using Shape = Shape_;

  /// Operand type
  static Operand const kOperand = Operand::kA;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = layout::TensorOpEm<32, layout::RowMajor>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Number of participating threads
  static int const kThreads = NUM_THREADS_PER_WARP;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Packed Size
  static int const kPackedSize = 32 / sizeof_bits<Element>::value;

  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero");
  static_assert(Shape::kColumn> 0, "Shape::kColumn must be greater than zero");
  static_assert(Shape::kColumn == InstructionShape::kK, "Shape::kColumn must equal InstructionShape::kK");
  static_assert((!(Shape::kRow % layout::EmShape::kRow) &&
                 !(Shape::kColumn % layout::EmShape::kColumn)),
                 "Shape must be divisibile by EM shape.");

  /// Internal structure of iterator - made public to enable introspection
  struct Detail {

    /// Determine access shape in units of elements per access
    using AccessShape = layout::PitchLinearShape<kPackedSize, 1>;

    /// Determine iterations
    using Iterations = MatrixShape<Shape::kRow / InstructionShape::kM, 1>;
  };

  /// Access type
#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
  using AccessType = Array<Element, 4>;
#else
  using AccessType = Array<Element, Detail::AccessShape::kCount>;
#endif

public:

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kCount / kThreads>;

private:

  /// Stride
  int stride_;

  /// Offset in units of element
  int offset_ = 0;

  /// Pointers holding same stride
#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
  Element* pointer_;
#else
  Element* pointers_[4];
#endif

public:

  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef ref,
    int lane_id
  ) : stride_(ref.stride(0)) {

#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
    pointer_ = ref.data();
#else
    Element* ptr = ref.data() + ref.offset({lane_id / 16, lane_id % 16});

    CUTLASS_PRAGMA_UNROLL
    for(int i = 0; i < 4; ++i) {
      pointers_[i] = ptr + i * 64;
    }
#endif
  }

  /// Adds a pointer offset to interal pointer(s) to advance through memory
  /// So far, isn't used anywhere. Offset must be muliple of Layout::TileShape
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    offset_ += int(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    offset_ += tile_offset.row() * Shape::kRow * stride_ +
               tile_offset.column() * (Shape::kColumn / layout::EmShape::kColumn) * layout::EmShape::kCount;
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {
    add_tile_offset({0, 1});
    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    add_tile_offset({0, -1});
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int offset = offset_ + ptr_offset;
    CUTLASS_PRAGMA_UNROLL
    for(int r = 0; r < Detail::Iterations::kRow; ++r) {
      auto tmp = __builtin_bi_slb_blkld_fx4((unsigned)((unsigned long long)(pointer_ + offset)), 0);
      dst_ptr[r][0] = tmp[0];
      dst_ptr[r][1] = tmp[1];
      dst_ptr[r][2] = tmp[2];
      dst_ptr[r][3] = tmp[3];
      offset += stride_ * 16;
    }
  }
#else
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int idx = 0;
    int offset = offset_ + ptr_offset;

    CUTLASS_PRAGMA_UNROLL
    for(int r = 0; r < Detail::Iterations::kRow; ++r) {
      CUTLASS_PRAGMA_UNROLL
      for(int i = 0; i < 4; ++i) {
        dst_ptr[idx] = *reinterpret_cast<AccessType*>(pointers_[i] + offset);
        ++idx;
      }
      offset += stride_ * 16;
    }
  }
#endif

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};

////////////////////////////////////////////////////////////////////////////////
/// Specialization for column-major A operands of 4bytes width type
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Data type of elements
  typename Element_,
  /// Shape of one matrix product operation (conecpt: PitchLinearShape)
  typename InstructionShape_,
  /// Number of partitions along K dimension
  int PartitionsK>
class MmaTensorOpMultiplicandTileIterator<
  Shape_,
  Operand::kA,
  Element_,
  layout::TensorOpEm<32, layout::ColumnMajor>,
  InstructionShape_,
  NUM_THREADS_PER_WARP,
  PartitionsK> {

public:

  /// Shape of tile to load (Concept: MatrixShape)
  using Shape = Shape_;

  /// Operand type
  static Operand const kOperand = Operand::kA;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = layout::TensorOpEm<32, layout::ColumnMajor>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Number of participating threads
  static int const kThreads = NUM_THREADS_PER_WARP;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Packed Size
  static int const kPackedSize = 32 / sizeof_bits<Element>::value;

  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero");
  static_assert(Shape::kColumn> 0, "Shape::kColumn must be greater than zero");
  static_assert(Shape::kColumn == InstructionShape::kK, "Shape::kColumn must equal InstructionShape::kK");
  static_assert((!(Shape::kRow % layout::EmShape::kRow) &&
                 !(Shape::kColumn % layout::EmShape::kColumn)),
                 "Shape must be divisibile by EM shape.");

  /// Internal structure of iterator - made public to enable introspection
  struct Detail {

    /// Determine access shape in units of elements per access
    using AccessShape = layout::PitchLinearShape<kPackedSize, 1>;

    /// Determine iterations
    using Iterations = MatrixShape<Shape::kRow / InstructionShape::kM, 1>;
  };

  /// Access type
#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
  using AccessType = Array<Element, 4>;
#else
  using AccessType = Array<Element, Detail::AccessShape::kCount>;
#endif

public:

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kCount / kThreads>;

private:

  /// Stride
  int stride_;

  /// Offset in units of element
  int offset_ = 0;

  /// Pointers holding same stride
#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
  Element* pointer_;
#else
  Element* pointers_[4];
#endif

public:

  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef ref,
    int lane_id
  ) : stride_(ref.stride(0)) {

    Element* ptr = ref.data();
#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
    pointer_ = ptr;
#else
    int const r = lane_id / 16;
    int const c = lane_id % 16;

    CUTLASS_PRAGMA_UNROLL
    for(int i = 0; i < 4; ++i) {
      pointers_[i] = ptr + ref.offset({r + i * 4, c});
    }
#endif
  }

  /// Adds a pointer offset to interal pointer(s) to advance through memory
  /// So far, isn't used anywhere. Offset must be muliple of Layout::TileShape
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    offset_ += int(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    offset_ += tile_offset.column() * Shape::kColumn * stride_ +
               tile_offset.row() * (Shape::kRow / layout::EmShape::kRow) * layout::EmShape::kCount;
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {
    add_tile_offset({0, 1});
    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    add_tile_offset({0, -1});
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int offset = offset_ + ptr_offset;
    CUTLASS_PRAGMA_UNROLL
    for(int r = 0; r < Detail::Iterations::kRow; ++r) {
      auto tmp = __builtin_bi_slb_blkld_fx4((unsigned)((unsigned long long)(pointer_ + offset)), 0);
      dst_ptr[r][0] = tmp[0];
      dst_ptr[r][1] = tmp[1];
      dst_ptr[r][2] = tmp[2];
      dst_ptr[r][3] = tmp[3];
      offset += layout::EmShape::kCount;
    }
  }
#else
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int idx = 0;
    int offset = offset_ + ptr_offset;

    CUTLASS_PRAGMA_UNROLL
    for(int r = 0; r < Detail::Iterations::kRow; ++r) {
      CUTLASS_PRAGMA_UNROLL
      for(int i = 0; i < 4; ++i) {
        dst_ptr[idx] = *reinterpret_cast<AccessType*>(pointers_[i] + offset);
        ++idx;
      }
      offset += layout::EmShape::kCount;
    }
  }
#endif

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};

////////////////////////////////////////////////////////////////////////////////
/// Specialization for row-major B operands of 4bytes width type
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Data type of elements
  typename Element_,
  /// Shape of one matrix product operation (conecpt: PitchLinearShape)
  typename InstructionShape_,
  /// Number of partitions along K dimension
  int PartitionsK>
class MmaTensorOpMultiplicandTileIterator<
  Shape_,
  Operand::kB,
  Element_,
  layout::TensorOpEm<32, layout::RowMajor>,
  InstructionShape_,
  NUM_THREADS_PER_WARP,
  PartitionsK> {

public:

  /// Shape of tile to load (Concept: MatrixShape)
  using Shape = Shape_;

  /// Operand type
  static Operand const kOperand = Operand::kA;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = layout::TensorOpEm<32, layout::RowMajor>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Number of participating threads
  static int const kThreads = NUM_THREADS_PER_WARP;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Packed Size
  static int const kPackedSize = 32 / sizeof_bits<Element>::value;

  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero");
  static_assert(Shape::kColumn> 0, "Shape::kColumn must be greater than zero");
  static_assert(Shape::kRow == InstructionShape::kK, "Shape::kRow must equal InstructionShape::kK");
  static_assert((!(Shape::kRow % layout::EmShape::kRow) &&
                 !(Shape::kColumn % layout::EmShape::kColumn)),
                 "Shape must be divisibile by EM shape.");

  /// Internal structure of iterator - made public to enable introspection
  struct Detail {

    /// Determine access shape in units of elements per access
    using AccessShape = layout::PitchLinearShape<kPackedSize, 1>;

    /// Determine iterations
    using Iterations = MatrixShape<1, Shape::kColumn / InstructionShape::kN>;
  };

  /// Access type
#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
  using AccessType = Array<Element, 4>;
#else
  using AccessType = Array<Element, Detail::AccessShape::kCount>;
#endif

public:

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kCount / kThreads>;

private:

  /// Stride
  int stride_;

  /// Offset in units of element
  int offset_ = 0;

  /// Pointers holding same stride
#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
  Element* pointer_;
#else
  Element* pointers_[4];
#endif

public:

  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef ref,
    int lane_id
  ) : stride_(ref.stride(0)) {

    Element* ptr = ref.data();
#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
    pointer_ = ptr;
#else
    int const r = lane_id / 16;
    int const c = lane_id % 16;

    CUTLASS_PRAGMA_UNROLL
    for(int i = 0; i < 4; ++i) {
      pointers_[i] = ptr + ref.offset({r + i * 4, c});
    }
#endif
  }

  /// Adds a pointer offset to interal pointer(s) to advance through memory
  /// So far, isn't used anywhere. Offset must be muliple of Layout::TileShape
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    offset_ += int(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    offset_ += tile_offset.row() * Shape::kRow * stride_ +
               tile_offset.column() * (Shape::kColumn / layout::EmShape::kColumn) * layout::EmShape::kCount;
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {
    add_tile_offset({1, 0});
    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    add_tile_offset({-1, 0});
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int offset = offset_ + ptr_offset;

    CUTLASS_PRAGMA_UNROLL
    for(int c = 0; c < Detail::Iterations::kColumn; ++c) {
      auto tmp = __builtin_bi_slb_blkld_fx4((unsigned)((unsigned long long)(pointer_ + offset)), 0);
      dst_ptr[c][0] = tmp[0];
      dst_ptr[c][1] = tmp[1];
      dst_ptr[c][2] = tmp[2];
      dst_ptr[c][3] = tmp[3];
      offset += layout::EmShape::kCount;
    }
  }
#else
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int idx = 0;
    int offset = offset_ + ptr_offset;

    CUTLASS_PRAGMA_UNROLL
    for(int c = 0; c < Detail::Iterations::kColumn; ++c) {
      CUTLASS_PRAGMA_UNROLL
      for(int i = 0; i < 4; ++i) {
        dst_ptr[idx] = *reinterpret_cast<AccessType*>(pointers_[i] + offset);
        ++idx;
      }
      offset += layout::EmShape::kCount;
    }
  }
#endif

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};

////////////////////////////////////////////////////////////////////////////////
/// Specialization for column-major B operands of 4bytes width type
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Data type of elements
  typename Element_,
  /// Shape of one matrix product operation (conecpt: PitchLinearShape)
  typename InstructionShape_,
  /// Number of partitions along K dimension
  int PartitionsK>
class MmaTensorOpMultiplicandTileIterator<
  Shape_,
  Operand::kB,
  Element_,
  layout::TensorOpEm<32, layout::ColumnMajor>,
  InstructionShape_,
  NUM_THREADS_PER_WARP,
  PartitionsK> {

public:

  /// Shape of tile to load (Concept: MatrixShape)
  using Shape = Shape_;

  /// Operand type
  static Operand const kOperand = Operand::kB;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = layout::TensorOpEm<32, layout::ColumnMajor>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Number of participating threads
  static int const kThreads = NUM_THREADS_PER_WARP;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Packed Size
  static int const kPackedSize = 32 / sizeof_bits<Element>::value;

  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero");
  static_assert(Shape::kColumn> 0, "Shape::kColumn must be greater than zero");
  static_assert(Shape::kRow == InstructionShape::kK, "Shape::kRow must equal InstructionShape::kK");
  static_assert((!(Shape::kRow % layout::EmShape::kRow) &&
                 !(Shape::kColumn % layout::EmShape::kColumn)),
                 "Shape must be divisibile by EM shape.");

  /// Internal structure of iterator - made public to enable introspection
  struct Detail {

    /// Determine access shape in units of elements per access
    using AccessShape = layout::PitchLinearShape<kPackedSize, 1>;

    /// Determine iterations
    using Iterations = MatrixShape<1, Shape::kColumn / InstructionShape::kN>;
  };

  /// Access type
#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
  using AccessType = Array<Element, 4>;
#else
  using AccessType = Array<Element, Detail::AccessShape::kCount>;
#endif

public:

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kCount / kThreads>;

private:

  /// Stride
  int stride_;

  /// Offset in units of element
  int offset_ = 0;

  /// Pointers holding same stride
#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
  Element* pointer_;
#else
  Element* pointers_[4];
#endif

public:

  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef ref,
    int lane_id
  ) : stride_(ref.stride(0)) {

    Element* ptr = ref.data();
#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
  pointer_ = ptr;
#else
    int const r = lane_id / 16;
    int const c = lane_id % 16;

    CUTLASS_PRAGMA_UNROLL
    for(int i = 0; i < 4; ++i) {
      pointers_[i] = ptr + ref.offset({r + i * 4, c});
    }
#endif
  }

  /// Adds a pointer offset to interal pointer(s) to advance through memory
  /// So far, isn't used anywhere. Offset must be muliple of Layout::TileShape
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    offset_ += int(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    offset_ += tile_offset.column() * Shape::kColumn * stride_ +
               tile_offset.row() * (Shape::kRow / layout::EmShape::kRow) * layout::EmShape::kCount;
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {
    add_tile_offset({1, 0});
    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    add_tile_offset({-1, 0});
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int offset = offset_ + ptr_offset;

    CUTLASS_PRAGMA_UNROLL
    for(int c = 0; c < Detail::Iterations::kColumn; ++c) {
      auto tmp = __builtin_bi_slb_blkld_fx4((unsigned)((unsigned long long)(pointer_ + offset)), 0);
      dst_ptr[c][0] = tmp[0];
      dst_ptr[c][1] = tmp[1];
      dst_ptr[c][2] = tmp[2];
      dst_ptr[c][3] = tmp[3];
      offset += stride_ * 16;
    }
  }
#else
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int idx = 0;
    int offset = offset_ + ptr_offset;

    CUTLASS_PRAGMA_UNROLL
    for(int c = 0; c < Detail::Iterations::kColumn; ++c) {
      CUTLASS_PRAGMA_UNROLL
      for(int i = 0; i < 4; ++i) {
        dst_ptr[idx] = *reinterpret_cast<AccessType*>(pointers_[i] + offset);
        idx++;
      }
      offset += stride_ * 16;
    }
  }
#endif

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///                           2bytes   (half/bhalf)                          ///
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/// Specialization for row-major A operands of 2bytes width type
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Data type of elements
  typename Element_,
  /// Shape of one matrix product operation (conecpt: PitchLinearShape)
  typename InstructionShape_,
  /// Number of partitions along K dimension
  int PartitionsK>
class MmaTensorOpMultiplicandTileIterator<
  Shape_,
  Operand::kA,
  Element_,
  layout::TensorOpEm<16, layout::RowMajor>,
  InstructionShape_,
  NUM_THREADS_PER_WARP,
  PartitionsK> {

public:

  /// Shape of tile to load (Concept: MatrixShape)
  using Shape = Shape_;

  /// Operand type
  static Operand const kOperand = Operand::kA;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = layout::TensorOpEm<16, layout::RowMajor>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Number of participating threads
  static int const kThreads = NUM_THREADS_PER_WARP;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Packed Size
  static int const kPackedSize = 32 / sizeof_bits<Element>::value;

  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero");
  static_assert(Shape::kColumn> 0, "Shape::kColumn must be greater than zero");
  static_assert(Shape::kColumn == InstructionShape::kK, "Shape::kColumn must equal InstructionShape::kK");
  static_assert((!(Shape::kRow % layout::EmShape::kRow) &&
                 !(Shape::kColumn % layout::EmShape::kColumn)),
                 "Shape must be divisibile by EM shape.");

  /// Internal structure of iterator - made public to enable introspection
  struct Detail {

    /// Determine access shape in units of elements per access
    using AccessShape = layout::PitchLinearShape<kPackedSize, 1>;

    /// Determine iterations
    using Iterations = MatrixShape<Shape::kRow / InstructionShape::kM, 1>;
  };

  /// Access type
#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
  using AccessType = Array<Element, 4>;
#else
  using AccessType = Array<Element, Detail::AccessShape::kCount>;
#endif

public:

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kCount / kThreads>;

private:

  /// Stride
  int stride_;

  /// Pointers holding same stride
#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
  Element* pointer_;
#else
  Element* pointers_[4];
#endif

  /// Iterations along row dimension in units of em
  int iteration_row_ = 0;

  /// Iterations along column dimension in units of em
  int iteration_column_ = 0;

public:

  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef ref,
    int lane_id
  ) : stride_(ref.stride(0)) {

    Element* ptr = ref.data();
#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
    pointer_ = ptr;
#else
    int r = (lane_id / 16) * 2;
    int c = lane_id % 16;

    pointers_[0] = ptr + ref.offset({r,     c});
    pointers_[1] = ptr + ref.offset({r + 8, c});
#endif
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    iteration_row_ += tile_offset.row() * (Shape::kRow / layout::EmShape::kRow);
    iteration_column_ += tile_offset.column();
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {
    add_tile_offset({0, 1});
    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    add_tile_offset({0, -1});
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int const col_offset = iteration_column_ * layout::EmShape::kCount + ptr_offset;
    CUTLASS_PRAGMA_UNROLL
    for(int r = 0; r < Detail::Iterations::kRow; ++r) {
      int const row_offset = (iteration_row_ + r) * layout::EmShape::kRow * stride_;

      auto tmp = __builtin_bi_slb_blkld_fx2((unsigned)((unsigned long long)(pointer_ + row_offset + col_offset)), 0);
      AccessType* at = reinterpret_cast<AccessType*>(&tmp);
      dst_ptr[r][0] = at->at(0).get();
      dst_ptr[r][1] = at->at(1).get();
      dst_ptr[r][2] = at->at(2).get();
      dst_ptr[r][3] = at->at(3).get();
    }
  }
#else
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int const col_offset = iteration_column_ * layout::EmShape::kCount + ptr_offset;
    CUTLASS_PRAGMA_UNROLL
    for(int r = 0; r < Detail::Iterations::kRow; ++r) {
      int const row_offset = (iteration_row_ + r) * layout::EmShape::kRow * stride_;

      CUTLASS_PRAGMA_UNROLL
      for(int i = 0; i < 2; ++i) {
        dst_ptr[r * 2 + i] = *reinterpret_cast<AccessType*>(pointers_[i] + col_offset + row_offset);
      }
    }
  }
#endif

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};

////////////////////////////////////////////////////////////////////////////////
/// Specialization for column-major A operands of 2bytes width type
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Data type of elements
  typename Element_,
  /// Shape of one matrix product operation (conecpt: PitchLinearShape)
  typename InstructionShape_,
  /// Number of partitions along K dimension
  int PartitionsK>
class MmaTensorOpMultiplicandTileIterator<
  Shape_,
  Operand::kA,
  Element_,
  layout::TensorOpEm<16, layout::ColumnMajor>,
  InstructionShape_,
  NUM_THREADS_PER_WARP,
  PartitionsK> {

public:

  /// Shape of tile to load (Concept: MatrixShape)
  using Shape = Shape_;

  /// Operand type
  static Operand const kOperand = Operand::kA;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = layout::TensorOpEm<16, layout::ColumnMajor>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Number of participating threads
  static int const kThreads = NUM_THREADS_PER_WARP;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Packed Size
  static int const kPackedSize = 32 / sizeof_bits<Element>::value;

  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero");
  static_assert(Shape::kColumn> 0, "Shape::kColumn must be greater than zero");
  static_assert((!(Shape::kRow % layout::EmShape::kRow) &&
                 !(Shape::kColumn % layout::EmShape::kColumn)),
                 "Shape must be divisibile by EM shape.");

  /// Internal structure of iterator - made public to enable introspection
  struct Detail {

    /// Determine access shape in units of elements per access
    using AccessShape = layout::PitchLinearShape<kPackedSize, 1>;

    /// Determine iterations
    using Iterations = MatrixShape<Shape::kRow / InstructionShape::kM, 1>;
  };

  /// Access type
#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
  using AccessType = Array<Element, 4>;
#else
  using AccessType = Array<Element, Detail::AccessShape::kCount>;
#endif

public:

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kCount / kThreads>;

private:

  /// Stride
  int stride_;

  /// Pointers holding same stride
#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
  Element* pointer_;
#else
  Element* pointers_[2];
#endif

  /// Iterations along row dimension in units of em
  int iteration_row_ = 0;

  /// Iterations along column dimension in units of em
  int iteration_column_ = 0;

public:

  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef ref,
    int lane_id
  ) : stride_(ref.stride(0)) {

    Element* ptr = ref.data();
#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
    pointer_ = ptr;
#else
    int r = (lane_id / 16) * 2;
    int c = lane_id % 16;

    pointers_[0] = ptr + ref.offset({r,      c});
    pointers_[1] = ptr + ref.offset({r + 8,  c});
#endif
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    iteration_row_ += tile_offset.row() * (Shape::kRow / layout::EmShape::kRow);
    iteration_column_ += tile_offset.column();
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {
    add_tile_offset({0, 1});
    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    add_tile_offset({0, -1});
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int const col_offset = iteration_column_ * layout::EmShape::kColumn * stride_ + ptr_offset;
    CUTLASS_PRAGMA_UNROLL
    for(int r = 0; r < Detail::Iterations::kRow; ++r) {
      int const row_offset = (iteration_row_ + r) * layout::EmShape::kCount;

      auto tmp = __builtin_bi_slb_blkld_fx2((unsigned)((unsigned long long)(pointer_ + row_offset + col_offset)), 0);
      AccessType* at = reinterpret_cast<AccessType*>(&tmp);
      dst_ptr[r][0] = at->at(0).get();
      dst_ptr[r][1] = at->at(1).get();
      dst_ptr[r][2] = at->at(2).get();
      dst_ptr[r][3] = at->at(3).get();
    }
  }
#else
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int const col_offset = iteration_column_ * layout::EmShape::kColumn * stride_ + ptr_offset;
    CUTLASS_PRAGMA_UNROLL
    for(int r = 0; r < Detail::Iterations::kRow; ++r) {
      int const row_offset = (iteration_row_ + r) * layout::EmShape::kCount;

      CUTLASS_PRAGMA_UNROLL
      for(int i = 0; i < 2; ++i) {
        dst_ptr[r * 2 + i] = *reinterpret_cast<AccessType*>(pointers_[i] + col_offset + row_offset);
      }
    }
  }
#endif

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};

////////////////////////////////////////////////////////////////////////////////
/// Specialization for row-major B operands of 2bytes width type
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Data type of elements
  typename Element_,
  /// Shape of one matrix product operation (conecpt: PitchLinearShape)
  typename InstructionShape_,
  /// Number of partitions along K dimension
  int PartitionsK>
class MmaTensorOpMultiplicandTileIterator<
  Shape_,
  Operand::kB,
  Element_,
  layout::TensorOpEm<16, layout::RowMajor>,
  InstructionShape_,
  NUM_THREADS_PER_WARP,
  PartitionsK> {

public:

  /// Shape of tile to load (Concept: MatrixShape)
  using Shape = Shape_;

  /// Operand type
  static Operand const kOperand = Operand::kB;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = layout::TensorOpEm<16, layout::RowMajor>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Number of participating threads
  static int const kThreads = NUM_THREADS_PER_WARP;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Packed Size
  static int const kPackedSize = 32 / sizeof_bits<Element>::value;

  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero");
  static_assert(Shape::kColumn> 0, "Shape::kColumn must be greater than zero");
  static_assert((!(Shape::kRow % layout::EmShape::kRow) &&
                 !(Shape::kColumn % layout::EmShape::kColumn)),
                 "Shape must be divisibile by EM shape.");

  /// Internal structure of iterator - made public to enable introspection
  struct Detail {

    /// Determine access shape in units of elements per access
    using AccessShape = layout::PitchLinearShape<kPackedSize, 1>;

    /// Determine iterations
    using Iterations = MatrixShape<1, Shape::kColumn / InstructionShape::kN>;
  };

  /// Access type
#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
  using AccessType = Array<Element, 4>;
#else
  using AccessType = Array<Element, Detail::AccessShape::kCount>;
#endif

public:

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kCount / kThreads>;

private:

  /// Stride
  int stride_;

  /// Pointers holding same stride
#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
  Element* pointer_;
#else
  Element* pointers_[2];
#endif

  /// Iterations along row dimension in units of em
  int iteration_row_ = 0;

  /// Iterations along column dimension in units of em
  int iteration_column_ = 0;

public:

  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef ref,
    int lane_id
  ) : stride_(ref.stride(0)) {

#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
    pointer_ = ref.data();
#else
    int r = (lane_id / 16) * 2;
    int c = lane_id % 16;

    pointers_[0] = ref.data() + ref.offset({r,     c});
    pointers_[1] = ref.data() + ref.offset({r + 8, c});
#endif
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    iteration_row_ += tile_offset.row();
    iteration_column_ += tile_offset.column() * (Shape::kColumn / layout::EmShape::kColumn);
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {
    add_tile_offset({1, 0});
    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    add_tile_offset({-1, 0});
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int const row_offset = iteration_row_ * layout::EmShape::kRow * stride_ + ptr_offset;

    CUTLASS_PRAGMA_UNROLL
    for(int c = 0; c < Detail::Iterations::kColumn; ++c) {
      int const col_offset = (iteration_column_ + c) * layout::EmShape::kCount;

      auto tmp = __builtin_bi_slb_blkld_fx2((unsigned)((unsigned long long)(pointer_ + row_offset + col_offset)), 0);
      AccessType* at = reinterpret_cast<AccessType*>(&tmp);
      dst_ptr[c][0] = at->at(0).get();
      dst_ptr[c][1] = at->at(1).get();
      dst_ptr[c][2] = at->at(2).get();
      dst_ptr[c][3] = at->at(3).get();
    }
  }
#else
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int const row_offset = iteration_row_ * layout::EmShape::kRow * stride_ + ptr_offset;

    CUTLASS_PRAGMA_UNROLL
    for(int c = 0; c < Detail::Iterations::kColumn; ++c) {
      int const col_offset = (iteration_column_ + c) * layout::EmShape::kCount;

      CUTLASS_PRAGMA_UNROLL
      for(int i = 0; i < 2; ++i) {
        dst_ptr[c * 2 + i] = *reinterpret_cast<AccessType*>(pointers_[i] + col_offset + row_offset);
      }
    }
  }
#endif

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};

////////////////////////////////////////////////////////////////////////////////
/// Specialization for column-major B operands of 2bytes width type
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Data type of elements
  typename Element_,
  /// Shape of one matrix product operation (conecpt: PitchLinearShape)
  typename InstructionShape_,
  /// Number of partitions along K dimension
  int PartitionsK>
class MmaTensorOpMultiplicandTileIterator<
  Shape_,
  Operand::kB,
  Element_,
  layout::TensorOpEm<16, layout::ColumnMajor>,
  InstructionShape_,
  NUM_THREADS_PER_WARP,
  PartitionsK> {

public:

  /// Shape of tile to load (Concept: MatrixShape)
  using Shape = Shape_;

  /// Operand type
  static Operand const kOperand = Operand::kB;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = layout::TensorOpEm<16, layout::ColumnMajor>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Number of participating threads
  static int const kThreads = NUM_THREADS_PER_WARP;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Packed Size
  static int const kPackedSize = 32 / sizeof_bits<Element>::value;

  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero");
  static_assert(Shape::kColumn> 0, "Shape::kColumn must be greater than zero");
  static_assert((!(Shape::kRow % layout::EmShape::kRow) &&
                 !(Shape::kColumn % layout::EmShape::kColumn)),
                 "Shape must be divisibile by EM shape.");

  /// Internal structure of iterator - made public to enable introspection
  struct Detail {

    /// Determine access shape in units of elements per access
    using AccessShape = layout::PitchLinearShape<kPackedSize, 1>;

    /// Determine iterations
    using Iterations = MatrixShape<1, Shape::kColumn / InstructionShape::kN>;
  };

  /// Access type
#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
  using AccessType = Array<Element, 4>;
#else
  using AccessType = Array<Element, Detail::AccessShape::kCount>;
#endif

public:

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kCount / kThreads>;

private:

  /// Stride
  int stride_;

  /// Pointers holding same stride
#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
  Element* pointer_;
#else
  Element* pointers_[2];
#endif

  /// Iterations along row dimension in units of em
  int iteration_row_ = 0;

  /// Iterations along column dimension in units of em
  int iteration_column_ = 0;

public:

  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef ref,
    int lane_id
  ) : stride_(ref.stride(0)) {

    Element* ptr = ref.data();
#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
    pointer_ = ptr;
#else
    int r = (lane_id / 16) * 2;
    int c = lane_id % 16;

    pointers_[0] = ptr + int(ref.offset({r,      c}));
    pointers_[1] = ptr + int(ref.offset({r + 8,  c}));
#endif
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    iteration_row_ += tile_offset.row();
    iteration_column_ += tile_offset.column() * (Shape::kColumn / layout::EmShape::kColumn);
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {
    add_tile_offset({1, 0});
    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    add_tile_offset({-1, 0});
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int const row_offset = iteration_row_ * layout::EmShape::kCount + ptr_offset;

    CUTLASS_PRAGMA_UNROLL
    for(int c = 0; c < Detail::Iterations::kColumn; ++c) {
      int const col_offset = (iteration_column_ + c) * layout::EmShape::kColumn * stride_;

      auto tmp = __builtin_bi_slb_blkld_fx2((unsigned)((unsigned long long)(pointer_ + row_offset + col_offset)), 0);
      AccessType* at = reinterpret_cast<AccessType*>(&tmp);
      dst_ptr[c][0] = at->at(0).get();
      dst_ptr[c][1] = at->at(1).get();
      dst_ptr[c][2] = at->at(2).get();
      dst_ptr[c][3] = at->at(3).get();
    }
  }
#else
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int const row_offset = iteration_row_ * layout::EmShape::kCount + ptr_offset;

    CUTLASS_PRAGMA_UNROLL
    for(int c = 0; c < Detail::Iterations::kColumn; ++c) {
      int const col_offset = (iteration_column_ + c) * layout::EmShape::kColumn * stride_;

      CUTLASS_PRAGMA_UNROLL
      for(int i = 0; i < 2; ++i) {
        dst_ptr[c * 2 + i] = *reinterpret_cast<AccessType*>(pointers_[i] + row_offset + col_offset);
      }
    }
  }
#endif

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///                           1byte   (int8/uint8)                          ///
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
/// Specialization for row-major A operands of 1byte width type
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Data type of elements
  typename Element_,
  /// Shape of one matrix product operation (conecpt: PitchLinearShape)
  typename InstructionShape_,
  /// Number of partitions along K dimension
  int PartitionsK>
class MmaTensorOpMultiplicandTileIterator<
  Shape_,
  Operand::kA,
  Element_,
  layout::TensorOpEm<8, layout::RowMajor>,
  InstructionShape_,
  NUM_THREADS_PER_WARP,
  PartitionsK> {

public:

  /// Shape of tile to load (Concept: MatrixShape)
  using Shape = Shape_;

  /// Operand type
  static Operand const kOperand = Operand::kA;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = layout::TensorOpEm<8, layout::RowMajor>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Number of participating threads
  static int const kThreads = NUM_THREADS_PER_WARP;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Packed Size
  static int const kPackedSize = 32 / sizeof_bits<Element>::value;

  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero");
  static_assert(Shape::kColumn> 0, "Shape::kColumn must be greater than zero");
  static_assert(Shape::kColumn == InstructionShape::kK, "Shape::kColumn must equal InstructionShape::kK");
  static_assert((!(Shape::kRow % layout::EmShape::kRow) &&
                 !(Shape::kColumn % layout::EmShape::kColumn)),
                 "Shape must be divisibile by EM shape.");

  /// Internal structure of iterator - made public to enable introspection
  struct Detail {

    /// Determine access shape in units of elements per access
    using AccessShape = layout::PitchLinearShape<kPackedSize, 1>;

    /// Determine iterations
    using Iterations = MatrixShape<Shape::kRow / InstructionShape::kM, 1>;
  };

  /// Access type
#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
  using AccessType = Array<Element, 4>;
#else
  using AccessType = Array<Element, Detail::AccessShape::kCount>;
#endif

public:

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kCount / kThreads>;

private:

  /// Stride
  int stride_;

  /// Pointers holding same stride
  Element* pointer_;

  /// Iterations along row dimension in units of em
  int iteration_row_ = 0;

  /// Iterations along column dimension in units of em
  int iteration_column_ = 0;

public:

  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef ref,
    int lane_id
  ) : stride_(ref.stride(0)) {

    Element* ptr = ref.data();
#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
    pointer_ = ptr;
#else
    int r = (lane_id / 16) * 4;
    int c = lane_id % 16;

    pointer_ = ptr + ref.offset({r, c});
#endif
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    iteration_row_ += tile_offset.row() * (Shape::kRow / layout::EmShape::kRow);
    iteration_column_ += tile_offset.column();
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {
    add_tile_offset({0, 1});
    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    add_tile_offset({0, -1});
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int const col_offset = iteration_column_ * layout::EmShape::kCount + ptr_offset;

    CUTLASS_PRAGMA_UNROLL
    for(int r = 0; r < Detail::Iterations::kRow; ++r) {
      int const row_offset = (iteration_row_ + r) * layout::EmShape::kRow * stride_;

      auto tmp = __builtin_bi_slb_blkld_fx1((unsigned)((unsigned long long)(pointer_ + row_offset + col_offset)), 0);
      AccessType* at = reinterpret_cast<AccessType*>(&tmp);
      dst_ptr[r][0] = at->at(0).get();
      dst_ptr[r][1] = at->at(1).get();
      dst_ptr[r][2] = at->at(2).get();
      dst_ptr[r][3] = at->at(3).get();
    }
  }
#else
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int const col_offset = iteration_column_ * layout::EmShape::kCount + ptr_offset;

    CUTLASS_PRAGMA_UNROLL
    for(int r = 0; r < Detail::Iterations::kRow; ++r) {
      int const row_offset = (iteration_row_ + r) * layout::EmShape::kRow * stride_;

      dst_ptr[r] = *reinterpret_cast<AccessType*>(pointer_ + col_offset + row_offset);
    }
  }
#endif

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};

////////////////////////////////////////////////////////////////////////////////
/// Specialization for column-major A operands of 1byte width type
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Data type of elements
  typename Element_,
  /// Shape of one matrix product operation (conecpt: PitchLinearShape)
  typename InstructionShape_,
  /// Number of partitions along K dimension
  int PartitionsK>
class MmaTensorOpMultiplicandTileIterator<
  Shape_,
  Operand::kA,
  Element_,
  layout::TensorOpEm<8, layout::ColumnMajor>,
  InstructionShape_,
  NUM_THREADS_PER_WARP,
  PartitionsK> {

public:

  /// Shape of tile to load (Concept: MatrixShape)
  using Shape = Shape_;

  /// Operand type
  static Operand const kOperand = Operand::kA;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = layout::TensorOpEm<8, layout::ColumnMajor>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Number of participating threads
  static int const kThreads = NUM_THREADS_PER_WARP;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Packed Size
  static int const kPackedSize = 32 / sizeof_bits<Element>::value;

  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero");
  static_assert(Shape::kColumn> 0, "Shape::kColumn must be greater than zero");
  static_assert((!(Shape::kRow % layout::EmShape::kRow) &&
                 !(Shape::kColumn % layout::EmShape::kColumn)),
                 "Shape must be divisibile by EM shape.");

  /// Internal structure of iterator - made public to enable introspection
  struct Detail {

    /// Determine access shape in units of elements per access
    using AccessShape = layout::PitchLinearShape<kPackedSize, 1>;

    /// Determine iterations
    using Iterations = MatrixShape<Shape::kRow / InstructionShape::kM, 1>;
  };

  /// Access type
#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
  using AccessType = Array<Element, 4>;
#else
  using AccessType = Array<Element, Detail::AccessShape::kCount>;
#endif

public:

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kCount / kThreads>;

private:

  /// Stride
  int stride_;

  /// Pointers holding same stride
  Element* pointer_;

  /// Iterations along row dimension in units of em
  int iteration_row_ = 0;

  /// Iterations along column dimension in units of em
  int iteration_column_ = 0;

public:

  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef ref,
    int lane_id
  ) : stride_(ref.stride(0)) {

    Element* ptr = ref.data();
#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
    pointer_ = ptr;
#else
    int r = (lane_id / 16) * 4;
    int c = lane_id % 16;

    pointer_ = ptr + ref.offset({r, c});
#endif
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    iteration_row_ += tile_offset.row() * (Shape::kRow / layout::EmShape::kRow);
    iteration_column_ += tile_offset.column();
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {
    add_tile_offset({0, 1});
    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    add_tile_offset({0, -1});
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int const col_offset = iteration_column_ * layout::EmShape::kColumn * stride_ + ptr_offset;

    CUTLASS_PRAGMA_UNROLL
    for(int r = 0; r < Detail::Iterations::kRow; ++r) {
      int const row_offset = (iteration_row_ + r) * layout::EmShape::kCount;

      auto tmp = __builtin_bi_slb_blkld_fx1((unsigned)((unsigned long long)(pointer_ + row_offset + col_offset)), 0);
      AccessType* at = reinterpret_cast<AccessType*>(&tmp);
      dst_ptr[r][0] = at->at(0).get();
      dst_ptr[r][1] = at->at(1).get();
      dst_ptr[r][2] = at->at(2).get();
      dst_ptr[r][3] = at->at(3).get();
    }
  }
#else
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int const col_offset = iteration_column_ * layout::EmShape::kColumn * stride_ + ptr_offset;

    CUTLASS_PRAGMA_UNROLL
    for(int r = 0; r < Detail::Iterations::kRow; ++r) {
      int const row_offset = (iteration_row_ + r) * layout::EmShape::kCount;

      dst_ptr[r] = *reinterpret_cast<AccessType*>(pointer_ + col_offset + row_offset);
    }
  }
#endif

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};

////////////////////////////////////////////////////////////////////////////////
/// Specialization for row-major B operands of 1byte width type
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Data type of elements
  typename Element_,
  /// Shape of one matrix product operation (conecpt: PitchLinearShape)
  typename InstructionShape_,
  /// Number of partitions along K dimension
  int PartitionsK>
class MmaTensorOpMultiplicandTileIterator<
  Shape_,
  Operand::kB,
  Element_,
  layout::TensorOpEm<8, layout::RowMajor>,
  InstructionShape_,
  NUM_THREADS_PER_WARP,
  PartitionsK> {

public:

  /// Shape of tile to load (Concept: MatrixShape)
  using Shape = Shape_;

  /// Operand type
  static Operand const kOperand = Operand::kB;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = layout::TensorOpEm<8, layout::RowMajor>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Number of participating threads
  static int const kThreads = NUM_THREADS_PER_WARP;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Packed Size
  static int const kPackedSize = 32 / sizeof_bits<Element>::value;

  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero");
  static_assert(Shape::kColumn> 0, "Shape::kColumn must be greater than zero");
  static_assert((!(Shape::kRow % layout::EmShape::kRow) &&
                 !(Shape::kColumn % layout::EmShape::kColumn)),
                 "Shape must be divisibile by EM shape.");

  /// Internal structure of iterator - made public to enable introspection
  struct Detail {

    /// Determine access shape in units of elements per access
    using AccessShape = layout::PitchLinearShape<kPackedSize, 1>;

    /// Determine iterations
    using Iterations = MatrixShape<1, Shape::kColumn / InstructionShape::kN>;
  };

  /// Access type
#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
  using AccessType = Array<Element, 4>;
#else
  using AccessType = Array<Element, Detail::AccessShape::kCount>;
#endif

public:

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kCount / kThreads>;

private:

  /// Stride
  int stride_;

  /// Pointers holding same stride
  Element* pointer_;

  /// Iterations along row dimension in units of em
  int iteration_row_ = 0;

  /// Iterations along column dimension in units of em
  int iteration_column_ = 0;

public:

  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef ref,
    int lane_id
  ) : stride_(ref.stride(0)) {

#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
    pointer_ = ref.data();
#else
    int r = (lane_id / 16) * 4;
    int c = lane_id % 16;

    pointer_ = ref.data() + ref.offset({r, c});
#endif
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    iteration_row_ += tile_offset.row();
    iteration_column_ += tile_offset.column() * (Shape::kColumn / layout::EmShape::kColumn);
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {
    add_tile_offset({1, 0});
    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    add_tile_offset({-1, 0});
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int const row_offset = iteration_row_ * layout::EmShape::kRow * stride_ + ptr_offset;

    CUTLASS_PRAGMA_UNROLL
    for(int c = 0; c < Detail::Iterations::kColumn; ++c) {
      int const col_offset = (iteration_column_ + c) * layout::EmShape::kCount;

      auto tmp = __builtin_bi_slb_blkld_fx1((unsigned)((unsigned long long)(pointer_ + row_offset + col_offset)), 0);
      AccessType* at = reinterpret_cast<AccessType*>(&tmp);
      dst_ptr[c][0] = at->at(0).get();
      dst_ptr[c][1] = at->at(1).get();
      dst_ptr[c][2] = at->at(2).get();
      dst_ptr[c][3] = at->at(3).get();
    }
  }
#else
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int const row_offset = iteration_row_ * layout::EmShape::kRow * stride_ + ptr_offset;

    CUTLASS_PRAGMA_UNROLL
    for(int c = 0; c < Detail::Iterations::kColumn; ++c) {
      int const col_offset = (iteration_column_ + c) * layout::EmShape::kCount;

      dst_ptr[c] = *reinterpret_cast<AccessType*>(pointer_ + col_offset + row_offset);
    }
  }
#endif

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};

////////////////////////////////////////////////////////////////////////////////
/// Specialization for column-major B operands of 1byte width type
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
  /// Size of the matrix to load (concept: MatrixShape)
  typename Shape_,
  /// Data type of elements
  typename Element_,
  /// Shape of one matrix product operation (conecpt: PitchLinearShape)
  typename InstructionShape_,
  /// Number of partitions along K dimension
  int PartitionsK>
class MmaTensorOpMultiplicandTileIterator<
  Shape_,
  Operand::kB,
  Element_,
  layout::TensorOpEm<8, layout::ColumnMajor>,
  InstructionShape_,
  NUM_THREADS_PER_WARP,
  PartitionsK> {

public:

  /// Shape of tile to load (Concept: MatrixShape)
  using Shape = Shape_;

  /// Operand type
  static Operand const kOperand = Operand::kB;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = layout::TensorOpEm<8, layout::ColumnMajor>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Number of participating threads
  static int const kThreads = NUM_THREADS_PER_WARP;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Packed Size
  static int const kPackedSize = 32 / sizeof_bits<Element>::value;

  static_assert(Shape::kRow > 0, "Shape::kRow must be greater than zero");
  static_assert(Shape::kColumn> 0, "Shape::kColumn must be greater than zero");
  static_assert((!(Shape::kRow % layout::EmShape::kRow) &&
                 !(Shape::kColumn % layout::EmShape::kColumn)),
                 "Shape must be divisibile by EM shape.");

  /// Internal structure of iterator - made public to enable introspection
  struct Detail {

    /// Determine access shape in units of elements per access
    using AccessShape = layout::PitchLinearShape<kPackedSize, 1>;

    /// Determine iterations
    using Iterations = MatrixShape<1, Shape::kColumn / InstructionShape::kN>;
  };

  /// Access type
#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
  using AccessType = Array<Element, 4>;
#else
  using AccessType = Array<Element, Detail::AccessShape::kCount>;
#endif

public:

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kCount / kThreads>;

private:

  /// Stride
  int stride_;

  /// Pointers holding same stride
  Element* pointer_;

  /// Iterations along row dimension in units of em
  int iteration_row_ = 0;

  /// Iterations along column dimension in units of em
  int iteration_column_ = 0;

public:

  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef ref,
    int lane_id
  ) : stride_(ref.stride(0)) {

    Element* ptr = ref.data();
#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
    pointer_ = ptr;
#else
    int r = (lane_id / 16) * 4;
    int c = lane_id % 16;

    pointer_ = ptr + ref.offset({r, c});
#endif
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {
    iteration_row_ += tile_offset.row();
    iteration_column_ += tile_offset.column() * (Shape::kColumn / layout::EmShape::kColumn);
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {
    add_tile_offset({1, 0});
    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    add_tile_offset({-1, 0});
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

#if SLB_BLOCK_LOAD_INSTRINSIC_ENABLED
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int const row_offset = iteration_row_ * layout::EmShape::kCount + ptr_offset;

    CUTLASS_PRAGMA_UNROLL
    for(int c = 0; c < Detail::Iterations::kColumn; ++c) {
      int const col_offset = (iteration_column_ + c) * layout::EmShape::kColumn * stride_;

      auto tmp = __builtin_bi_slb_blkld_fx1((unsigned)((unsigned long long)(pointer_ + row_offset + col_offset)), 0);
      AccessType* at = reinterpret_cast<AccessType*>(&tmp);
      dst_ptr[c][0] = at->at(0).get();
      dst_ptr[c][1] = at->at(1).get();
      dst_ptr[c][2] = at->at(2).get();
      dst_ptr[c][3] = at->at(3).get();
    }
  }
#else
  CUTLASS_HOST_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index ptr_offset) const {
    AccessType *dst_ptr = reinterpret_cast<AccessType*>(&frag);

    int const row_offset = iteration_row_ * layout::EmShape::kCount + ptr_offset;

    CUTLASS_PRAGMA_UNROLL
    for(int c = 0; c < Detail::Iterations::kColumn; ++c) {
      int const col_offset = (iteration_column_ + c) * layout::EmShape::kColumn * stride_;

      dst_ptr[c] = *reinterpret_cast<AccessType*>(pointer_ + row_offset + col_offset);
    }
  }
#endif

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};
} // namespace warp
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
