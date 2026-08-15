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
    \brief Definitions for GEMM structures
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_clamp.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace device {

////////////////////////////////////////////////////////////////////////////////

template <
  typename OperatorClass,
  typename ArchTag,
  typename ElementA,
  typename ElementB,
  typename ElementC,
  typename ElementAccumulator
>
struct DefaultGemmConfiguration;

////////////////////////////////////////////////////////////////////////////////

/// FIXME(Peter Han): Need to update configuration according to perf results, so
/// that could archieve good performance by default.

template <
  typename ArchTag,
  typename ElementA,
  typename ElementB,
  typename ElementC,
  typename ElementAccumulator>
struct DefaultGemmConfiguration<
  arch::OpClassSimt,
  ArchTag,
  ElementA,
  ElementB,
  ElementC,
  ElementAccumulator> {

  static int const kAlignmentA = 1;
  static int const kAlignmentB = 1;
  using ThreadblockShape = GemmShape<128, 128, 8>;
  using WarpShape = GemmShape<64, 64, 8>;
  using InstructionShape = GemmShape<1, 1, 1>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombination<
    ElementC,
    1,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Operator = arch::OpMultiplyAdd;
};

////////////////////////////////////////////////////////////////////////////////

template <
  typename ArchTag,
  typename ElementC>
struct DefaultGemmConfiguration<arch::OpClassSimt, ArchTag, int8_t, int8_t, ElementC, int32_t> {

  static int const kAlignmentA = 4;
  static int const kAlignmentB = 4;
  using ThreadblockShape = GemmShape<128, 128, 32>;
  using WarpShape = GemmShape<64, 64, 32>;
  using InstructionShape = GemmShape<1, 1, 4>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombinationClamp<
    ElementC,
    1,
    int32_t,
    float
  >;

  using Operator = arch::OpMultiplyAdd;
};

////////////////////////////////////////////////////////////////////////////////

template <
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp,
  arch::Cu10,
  int8_t,
  int8_t,
  ElementC,
  int32_t> {

  using ElementA = int8_t;
  using ElementB = int8_t;
  using ElementAccumulator = int32_t;
  static int const kAlignmentA = MEMORY_ACCESS_SIZE / sizeof_bits<ElementA>::value;
  static int const kAlignmentB = MEMORY_ACCESS_SIZE / sizeof_bits<ElementB>::value;

  using ThreadblockShape = GemmShape<256, 256, 32>;
  using WarpShape = GemmShape<64, 64, 32>;
  using InstructionShape = GemmShape<16, 16, 16>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombination<
    ElementC,
    MEMORY_ACCESS_SIZE / sizeof_bits<ElementC>::value,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Operator = arch::OpMultiplyAdd;
};

template <
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp,
  arch::Cu10,
  uint8_t,
  uint8_t,
  ElementC,
  uint32_t> {

  using ElementA = uint8_t;
  using ElementB = uint8_t;
  using ElementAccumulator = uint32_t;
  static int const kAlignmentA = MEMORY_ACCESS_SIZE / sizeof_bits<ElementA>::value;
  static int const kAlignmentB = MEMORY_ACCESS_SIZE / sizeof_bits<ElementB>::value;

  using ThreadblockShape = GemmShape<256, 256, 32>;
  using WarpShape = GemmShape<64, 64, 32>;
  using InstructionShape = GemmShape<16, 16, 16>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombination<
    ElementC,
    MEMORY_ACCESS_SIZE / sizeof_bits<ElementC>::value,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Operator = arch::OpMultiplyAdd;
};

template <
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp,
  arch::Cu10,
  half_t,
  half_t,
  ElementC,
  float> {

  using ElementA = half_t;
  using ElementB = half_t;
  using ElementAccumulator = float;
  static int const kAlignmentA = MEMORY_ACCESS_SIZE / sizeof_bits<ElementA>::value;
  static int const kAlignmentB = MEMORY_ACCESS_SIZE / sizeof_bits<ElementB>::value;

  using ThreadblockShape = GemmShape<128, 128, 32>;
  using WarpShape = GemmShape<32, 32, 32>;
  using InstructionShape = GemmShape<16, 16, 16>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombination<
    ElementC,
    MEMORY_ACCESS_SIZE / sizeof_bits<ElementC>::value,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Operator = arch::OpMultiplyAdd;
};

template <
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp,
  arch::Cu10,
  bfloat16_t,
  bfloat16_t,
  ElementC,
  float> {

  using ElementA = bfloat16_t;
  using ElementB = bfloat16_t;
  using ElementAccumulator = float;
  static int const kAlignmentA = 32 / sizeof_bits<ElementA>::value;
  static int const kAlignmentB = 32 / sizeof_bits<ElementB>::value;

  using ThreadblockShape = GemmShape<128, 128, 32>;
  using WarpShape = GemmShape<32, 32, 32>;
  using InstructionShape = GemmShape<16, 16, 16>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombination<
    ElementC,
    MEMORY_ACCESS_SIZE / sizeof_bits<ElementC>::value,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Operator = arch::OpMultiplyAdd;
};

template <
  typename ElementC>
struct DefaultGemmConfiguration<
  arch::OpClassTensorOp,
  arch::Cu10,
  float,
  float,
  ElementC,
  float> {

  using ElementA = float;
  using ElementB = float;
  using ElementAccumulator = float;
  static int const kAlignmentA = 32 / sizeof_bits<ElementA>::value;
  static int const kAlignmentB = 32 / sizeof_bits<ElementB>::value;

  using ThreadblockShape = GemmShape<128, 128, 32>;
  using WarpShape = GemmShape<32, 32, 32>;
  using InstructionShape = GemmShape<16, 16, 16>;
  static int const kStages = 2;

  using EpilogueOutputOp = epilogue::thread::LinearCombination<
    ElementC,
    MEMORY_ACCESS_SIZE / sizeof_bits<ElementC>::value,
    ElementAccumulator,
    ElementAccumulator
  >;

  using Operator = arch::OpMultiplyAdd;
};

////////////////////////////////////////////////////////////////////////////////
} // namespace device
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
