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
    \brief Policy describing implementation details of warp-level GEMM targeting Tensor Cores.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/gemm/gemm.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Policy
template <
  typename Operator_,        ///< hardware instruction(s) performing TensorOp (concept: arch::Mma)
  typename OpDelta_          ///< distance between operations (concept: MatrixShape)
>
struct MmaTensorOpPolicy {

  using Operator = Operator_;    ///< hardware instruction(s) performing TensorOp (concept: arch::Mma)
  using OpDelta = OpDelta_;      ///< distance between operations (concept: MatrixShape)
  using MmaShape = typename Operator::Shape;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass
