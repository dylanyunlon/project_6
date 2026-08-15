/* Copyright 2019 Iluvatar-CoreX - All Rights Reserved
 * Unauthorized copying of this file, via any medium is strictly prohibited
 * Proprietary and confidential
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
 * DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#if defined(_MSC_VER)
#pragma message("crt/iluvatar_mma.h is an internal header file and must not be used directly.  Please use mma.h instead.")
#else
#warning "crt/iluvatar_mma.h is an internal header file and must not be used directly.  Please use mma.h instead."
#endif
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_CUDA_MMA_H__
#endif

#if !defined(__ILUVATAR_MMA_HPP__)
#define __ILUVATAR_MMA_HPP__

#include <cuda_fp16.h>

#define __CUDA_MMA_DEVICE_DECL__ static __device__ __inline__

#if defined(__cplusplus) && defined(__CUDACC__)

#if !defined(__CUDA_ARCH__) || defined(__ILUVATAR__)

#if !defined(__CUDA_ARCH__) || defined(__ivcore10__)
#define __BI__ 1
#endif
#if !defined(__CUDA_ARCH__) || defined(__ivcore11__)
#define __MR__ 1
#endif

namespace nvcuda {
namespace wmma {

  /// Convert tile internal coordinate to offset relative to origin of current tile
  template <int ElementSize, int Layout>
  __device__ __inline__
  int64_t CoordToOffset(int row, int column);

  template <>
  __device__ __inline__
  int64_t CoordToOffset<32, layout_t::mem_row_major>(int row, int column) {
    return row * 16 + column;
  }

  template <>
  __device__ __inline__
  int64_t CoordToOffset<16, layout_t::mem_row_major>(int row, int column) {
    int const i = column / 16;
    int const c = column % 16;
    int const r = row / 2;

    return (i * 256 + r / 4 * 128) + ((r * 32 + c * 2 + (row & 1) + i * 64) & 127);
  }

  template <>
  __device__ __inline__
  int64_t CoordToOffset<8, layout_t::mem_row_major>(int row, int column) {
    int const i = column / 16;
    int const c = column % 16;
    int const r = row / 4;
    return i * 256 + ((r * 64 + c * 4 + (row & 3) + 64 * i) & 255);
  }

  template <>
  __device__ __inline__
  int64_t CoordToOffset<32, layout_t::mem_col_major>(int row, int column) {
    return ((column >> 2) & 3) * 64 + (row & 3) * 16 + (((row >> 2) & 3) ^ ((column >> 2) & 3)) * 4 + (column & 3);
  }

  template <>
  __device__ __inline__
  int64_t CoordToOffset<16, layout_t::mem_col_major>(int row, int column) {
    int const r = row >> 1;
    return (column >> 2) * 128 + (r & 3) * 32 + ((r >> 3) ^ (column >> 3)) * 16 +
          ((r >> 2 & 1) ^ (column >> 2 & 1)) * 8 + (column & 3) * 2 + (row & 1);
  }

  template <>
  __device__ __inline__
  int64_t CoordToOffset<8, layout_t::mem_col_major>(int row, int column) {
    int const r = row >> 2;
    return (column >> 2) * 256 + (r & 3) * 64 + ((r >> 2) ^ (column / 4)) * 16 +
          (column & 3) * 4 + (row & 3);
  }

  template <typename MatrixType, typename PtrType>
  __CUDA_MMA_DEVICE_DECL__ void __imma_ld_col_b8(MatrixType* a, const PtrType* p) {
    int laneId = __ivcorex_lane_id();

    for (int quarter_tile = 0; quarter_tile < 4; quarter_tile++) {
      int row = (laneId / 16) * 4 + 16 * quarter_tile;
      int column = laneId % 16;
      int offset = CoordToOffset<8, layout_t::mem_col_major>(row, column);
      a[quarter_tile] = *((int*)(p + offset));
    }
  }

  template <typename MatrixType, typename PtrType>
  __CUDA_MMA_DEVICE_DECL__ void __imma_ld_row_b8(MatrixType* a, const PtrType* p) {
    int laneId = __ivcorex_lane_id();

    for (int quarter_tile = 0; quarter_tile < 4; quarter_tile++) {
      int row = (laneId / 16) * 4;
      int column = laneId % 16 + 16 * quarter_tile;
      int offset = CoordToOffset<8, layout_t::mem_row_major>(row, column);
      a[quarter_tile] = *((int*)(p + offset));
    }
  }

  template <typename MatrixType, typename PtrType>
  __CUDA_MMA_DEVICE_DECL__ void __imma_ld_row_b32(MatrixType* a, const PtrType* p) {
    int laneId = __ivcorex_lane_id();

    for(int quarter_tile = 0; quarter_tile < 4; quarter_tile++) {
      int row = laneId / 16 + 4 * quarter_tile;
      int column = laneId % 16;
      int offset = CoordToOffset<32, layout_t::mem_row_major>(row, column);
      a[quarter_tile] = *(p + offset);
    }
  }

  //
  // Load functions for frags of A, B, C, D: I8, I8, I32, I32
  //
  /************************************************* 
  Function:       load_matrix_sync_tcu
  Description:    load data from slb to matrix a and b with row and col major
  Input:          a            destionation fragment
                  p            source address in slb
                  WarpMIndex   M direction's tcu position in a block area
                  WarpNIndex   N direction's tcu position in a block area
                  WarpKIndex   K direction's tcu position in a block area
  *************************************************/
  template<class MMBI>
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync_tcu(FragmentARowB8<MMBI>& a, const void* p, unsigned WarpMIndex, unsigned WarpKIndex) {
    unsigned SLBaTCUIndex = WarpMIndex * a.getBlockKloopCnt() * 2 + WarpKIndex * 2;
    const unsigned TCUEmStride = 64;
    unsigned SLBaTCUOffset = SLBaTCUIndex * TCUEmStride;
    a[0] = *((unsigned int*)p + SLBaTCUOffset + a.getRowEMOffset(SLBaTCUIndex % 4));
    a[1] = *((unsigned int*)p + SLBaTCUOffset + a.getRowEMOffset((SLBaTCUIndex + 1) % 4) + TCUEmStride);
  }

template<class MMBI>
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync_tcu(FragmentAColB8<MMBI>& a, const void* p, unsigned WarpMIndex, unsigned WarpKIndex) {
    const unsigned TCUEmStrideX4 = 256;
    const unsigned WarpEmStride = TCUEmStrideX4 * 4;
    unsigned SLBaWarpMIndex = (WarpMIndex / 4) * WarpEmStride;
    unsigned SLBaTCUOffset = SLBaWarpMIndex + ((2 * WarpKIndex) % (a.getBlockKloopCnt() * 2) * TCUEmStrideX4);
    a[0] = *((unsigned int*)p + SLBaTCUOffset + a.getColEMOffset(WarpMIndex % 4));
    a[1] = *((unsigned int*)p + SLBaTCUOffset + a.getColEMOffset(WarpMIndex % 4) + TCUEmStrideX4);
  }

template<class MMBI>
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync_tcu(FragmentBRowB8<MMBI>& a, const void* p, unsigned WarpNIndex, unsigned WarpKIndex) {
    unsigned SLBbTCUIndex = WarpKIndex * 2 * a.getBlockNloopCnt() + WarpNIndex;
    const unsigned TCUEmStride = 64;
    int SLBbTCUOffset = SLBbTCUIndex * TCUEmStride;
    a[0] = *((unsigned int*)p + SLBbTCUOffset + a.getRowEMOffset(SLBbTCUIndex % 4));
    a[1] = *((unsigned int*)p + SLBbTCUOffset + TCUEmStride * a.getBlockNloopCnt() + a.getRowEMOffset((SLBbTCUIndex + a.getBlockNloopCnt()) % 4));
  }

template<class MMBI>
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync_tcu(FragmentBColB8<MMBI>& a, const void* p, unsigned WarpNIndex, unsigned WarpKIndex) {
    unsigned SLBbTCUIndex = WarpNIndex * a.getBlockKloopCnt() * 2 + WarpKIndex * 2;
    const unsigned TCUEmStrideX4 = 256;
    unsigned SLBbTCUOffset = SLBbTCUIndex / 4 * TCUEmStrideX4;
    a[0] = *((unsigned int*)p + SLBbTCUOffset + a.getColEMOffset(SLBbTCUIndex % 4));
    a[1] = *((unsigned int*)p + SLBbTCUOffset + a.getColEMOffset((SLBbTCUIndex + 1) % 4));
  }

/*****************************************************/

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 16, 64, signed char, row_major>& a, const signed char* p, unsigned ldm) {
    __imma_ld_row_b8<unsigned int, signed char>(&(a.x[0]), p);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 16, 16, 64, signed char, col_major>& a, const signed char* p, unsigned ldm) {
    __imma_ld_col_b8<unsigned int, signed char>(&(a.x[0]), p);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 16, 16, 64, signed int>& a, const signed int* p, unsigned ldm, layout_t layout) {
    assert(layout == mem_row_major && "mem_col_major not supported for accumulator!");
    __imma_ld_row_b32<signed int, signed int>(&(a.x[0]), p);
  }

#ifdef __BI__
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 64, 64, 16, signed char, col_major>& a, const signed char* p, unsigned ldm) {
    __imma_ld_col_b8<unsigned int, signed char>(&(a.x[0]), p);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 64, 64, 16, signed char, row_major>& a, const signed char* p, unsigned ldm) {
    __imma_ld_row_b8<unsigned int, signed char>(&(a.x[0]), p);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 64, 64, 16, signed int>& a, const signed int* p, unsigned ldm, layout_t layout) {
    assert(layout == mem_row_major && "mem_col_major not supported for accumulator!");

    for (int tile_row = 0; tile_row < 4; tile_row++) {
      for (int tile_column = 0; tile_column < 4; tile_column++) {
        int tile_num = 4 * tile_row + tile_column;
        const signed int* ptr = p + 16 * 16 * tile_num;
        __imma_ld_row_b32<signed int, signed int>(&(a.x[tile_num * 4]), ptr);
      }
    }    
  }
#endif /* __BI__ */

#ifdef __MR__
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 64, 64, 32, signed char, col_major>& a, const signed char* p, unsigned ldm) {
    for(int tile_num = 0; tile_num < 2; tile_num++) {
      const signed char* ptr = p + 16 * 64 * tile_num;
      __imma_ld_col_b8<unsigned int, signed char>(&(a.x[tile_num * 4]), ptr);
    }
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 64, 64, 32, signed char, row_major>& a, const signed char* p, unsigned ldm) {
    for(int tile_num = 0; tile_num < 2; tile_num++) {
      const signed char* ptr = p + 16 * 64 * tile_num;
      __imma_ld_row_b8<unsigned int, signed char>(&(a.x[tile_num * 4]), ptr);
    }
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 64, 64, 32, signed int>& a, const signed int* p, unsigned ldm, layout_t layout) {
    assert(layout == mem_row_major && "mem_col_major not supported for accumulator!");

    for (int tile_row = 0; tile_row < 4; tile_row++) {
      for (int tile_column = 0; tile_column < 4; tile_column++) {
        int tile_num = 4 * tile_row + tile_column;
        const signed int* ptr = p + 16 * 16 * tile_num;
        __imma_ld_row_b32<signed int, signed int>(&(a.x[tile_num * 4]), ptr);
      }
    }    
  }
#endif /* __MR__ */

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 64, 64, signed char, row_major>& a, const signed char* p, unsigned ldm) {
    __imma_ld_row_b8<unsigned int, signed char>(&(a.x[0]), p);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 16, 64, 64, signed char, row_major>& a, const signed char* p, unsigned ldm) {
    for (int tile_num = 0; tile_num < 4; tile_num++) {
      const signed char* ptr = p + 16 * 64 * tile_num;
      __imma_ld_row_b8<unsigned int, signed char>(&(a.x[tile_num * 4]), ptr);
    }
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 16, 64, 64, signed int>& a, const signed int* p, unsigned ldm, layout_t layout) {
    assert(layout == mem_row_major && "mem_col_major not supported for accumulator!");
    for (int tile_num = 0; tile_num < 4; tile_num++) {
      const signed int* ptr = p + 16 * 16 * tile_num;
      __imma_ld_row_b32<signed int, signed int>(&(a.x[tile_num * 4]), ptr);
    }
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 64, 16, 64, signed char, col_major>& a, const signed char* p, unsigned ldm) {
    for (int tile_num = 0; tile_num < 4; tile_num++) {
      const signed char* ptr = p + 16 * 64 * tile_num;
      __imma_ld_col_b8<unsigned int, signed char>(&(a.x[tile_num * 4]), ptr);
    }
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 64, 16, 64, signed char, col_major>& a, const signed char* p, unsigned ldm) {
    __imma_ld_col_b8<unsigned int, signed char>(&(a.x[0]), p);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 64, 16, 64, signed int>& a, const signed int* p, unsigned ldm, layout_t layout) {
    assert(layout == mem_row_major && "mem_col_major not supported for accumulator!");

    for (int tile_num = 0; tile_num < 4; tile_num++) {
      const signed int* ptr = p + 16 * 16 * tile_num;
      __imma_ld_row_b32<signed int, signed int>(&(a.x[tile_num * 4]), ptr);
    }
  }

  //
  // Load functions for frags of A, B, C, D: U8, U8, U32, U32
  //
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 16, 64, unsigned char, row_major>& a, const unsigned char* p, unsigned ldm) {
    __imma_ld_row_b8<unsigned int, unsigned char>(&(a.x[0]), p);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 16, 16, 64, unsigned char, col_major>& a, const unsigned char* p, unsigned ldm) {
    __imma_ld_col_b8<unsigned int, unsigned char>(&(a.x[0]), p);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 16, 16, 64, unsigned int>& a, const unsigned int* p, unsigned ldm, layout_t layout) {
    assert(layout == mem_row_major && "mem_col_major not supported for accumulator!");
    __imma_ld_row_b32<unsigned int, unsigned int>(&(a.x[0]), p);
  }

#ifdef __BI__
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 64, 64, 16, unsigned char, col_major>& a, const unsigned char* p, unsigned ldm) {
    __imma_ld_col_b8<unsigned int, unsigned char>(&(a.x[0]), p);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 64, 64, 16, unsigned char, row_major>& a, const unsigned char* p, unsigned ldm) {
    __imma_ld_row_b8<unsigned int, unsigned char>(&(a.x[0]), p);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 64, 64, 16, unsigned int>& a, const unsigned int* p, unsigned ldm, layout_t layout) {
    assert(layout == mem_row_major && "mem_col_major not supported for accumulator!");

    for (int tile_row = 0; tile_row < 4; tile_row++) {
      for (int tile_column = 0; tile_column < 4; tile_column++) {
        int tile_num = 4 * tile_row + tile_column;
        const unsigned int* ptr = p + 16 * 16 * tile_num;
        __imma_ld_row_b32<unsigned int, unsigned int>(&(a.x[tile_num * 4]), ptr);
      }
    }    
  }
#endif /* __BI__ */

#ifdef __MR__
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 64, 64, 32, unsigned char, col_major>& a, const unsigned char* p, unsigned ldm) {
    for(int tile_num = 0; tile_num < 2; tile_num++) {
      const unsigned char* ptr = p + 16 * 64 * tile_num;
      __imma_ld_col_b8<unsigned int, unsigned char>(&(a.x[tile_num * 4]), ptr);
    }
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 64, 64, 32, unsigned char, row_major>& a, const unsigned char* p, unsigned ldm) {
    for(int tile_num = 0; tile_num < 2; tile_num++) {
      const unsigned char* ptr = p + 16 * 64 * tile_num;
      __imma_ld_row_b8<unsigned int, unsigned char>(&(a.x[tile_num * 4]), ptr);
    }
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 64, 64, 32, unsigned int>& a, const unsigned int* p, unsigned ldm, layout_t layout) {
    assert(layout == mem_row_major && "mem_col_major not supported for accumulator!");

    for (int tile_row = 0; tile_row < 4; tile_row++) {
      for (int tile_column = 0; tile_column < 4; tile_column++) {
        int tile_num = 4 * tile_row + tile_column;
        const unsigned int* ptr = p + 16 * 16 * tile_num;
        __imma_ld_row_b32<unsigned int, unsigned int>(&(a.x[tile_num * 4]), ptr);
      }
    }    
  }
#endif /* __MR__ */

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 64, 64, unsigned char, row_major>& a, const unsigned char* p, unsigned ldm) {
    __imma_ld_row_b8<unsigned int, unsigned char>(&(a.x[0]), p);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 16, 64, 64, unsigned char, row_major>& a, const unsigned char* p, unsigned ldm) {
    for (int tile_num = 0; tile_num < 4; tile_num++) {
      const unsigned char* ptr = p + 16 * 64 * tile_num;
      __imma_ld_row_b8<unsigned int, unsigned char>(&(a.x[tile_num * 4]), ptr);
    }
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 16, 64, 64, unsigned int>& a, const unsigned int* p, unsigned ldm, layout_t layout) {
    assert(layout == mem_row_major && "mem_col_major not supported for accumulator!");
    for (int tile_num = 0; tile_num < 4; tile_num++) {
      const unsigned int* ptr = p + 16 * 16 * tile_num;
      __imma_ld_row_b32<unsigned int, unsigned int>(&(a.x[tile_num * 4]), ptr);
    }
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 64, 16, 64, unsigned char, col_major>& a, const unsigned char* p, unsigned ldm) {
    for (int tile_num = 0; tile_num < 4; tile_num++) {
      const unsigned char* ptr = p + 16 * 64 * tile_num;
      __imma_ld_col_b8<unsigned int, unsigned char>(&(a.x[tile_num * 4]), ptr);
    }
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 64, 16, 64, unsigned char, col_major>& a, const unsigned char* p, unsigned ldm) {
    __imma_ld_col_b8<unsigned int, unsigned char>(&(a.x[0]), p);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 64, 16, 64, unsigned int>& a, const unsigned int* p, unsigned ldm, layout_t layout) {
    assert(layout == mem_row_major && "mem_col_major not supported for accumulator!");
    for (int tile_num = 0; tile_num < 4; tile_num++) {
      const unsigned int* ptr = p + 16 * 16 * tile_num;
      __imma_ld_row_b32<unsigned int, unsigned int>(&(a.x[tile_num * 4]), ptr);
    }
  }

  template <typename MatrixType, typename PtrType>
  __CUDA_MMA_DEVICE_DECL__ void __hmma_ld_row_b16(MatrixType* a, const PtrType* p) {
    int laneId = __ivcorex_lane_id();
  
    for (int quarter_tile = 0; quarter_tile < 4; quarter_tile++) {
      int row = (laneId / 16) * 2 + 8 * (quarter_tile % 2);
      int column = laneId % 16 + 16 * (quarter_tile / 2);
      int offset = CoordToOffset<16, layout_t::mem_row_major>(row, column);
      a[quarter_tile] = *((unsigned int*)(p + offset));
    }
  }

  template <typename MatrixType, typename PtrType>
  __CUDA_MMA_DEVICE_DECL__ void __hmma_ld_col_b16(MatrixType* a, const PtrType* p) {
    int laneId = __ivcorex_lane_id();

    for (int quarter_tile = 0; quarter_tile < 4; quarter_tile++) {
      int row = (laneId / 16) * 2 + 8 * quarter_tile;
      int column = laneId % 16;
      int offset = CoordToOffset<16, layout_t::mem_col_major>(row, column);
      a[quarter_tile] = *((unsigned int*)(p + offset));
    }
  }

  //
  // Load functions for frags of A, B, C, D: F16, F16, F32, F32
  //

  /************************************************* 
  Function:       load_matrix_sync_tcu
  Description:    load data from slb to matrix a and b with row and col major
  Input:          a            destionation fragment
                  p            source address in slb
                  WarpMIndex   M direction's tcu position in a block area
                  WarpNIndex   N direction's tcu position in a block area
                  WarpKIndex   K direction's tcu position in a block area
  *************************************************/
  template<class MMBI>
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync_tcu(FragmentARowB16<MMBI>& a, const void* p, unsigned WarpMIndex, unsigned WarpKIndex) {
    int laneId = __ivcorex_lane_id();
    unsigned SLBaTCUIndex = WarpMIndex * a.getBlockKloopCnt() + WarpKIndex;
    unsigned RowEmOffset = (SLBaTCUIndex & 1) ? (laneId ^ 0x20) : laneId;
    const unsigned TCUEmStride = 128;
    int SLBbTCUOffset = SLBaTCUIndex * TCUEmStride;
    a[0] = *((unsigned int*)p + SLBbTCUOffset + RowEmOffset);
    a[1] = *((unsigned int*)p + SLBbTCUOffset + RowEmOffset + 64);
  }

template<class MMBI>
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync_tcu(FragmentAColB16<MMBI>& a, const void* p, unsigned WarpMIndex, unsigned WarpKIndex) {
    unsigned SLBaTCUIndex = WarpMIndex / 2 * a.getBlockKloopCnt() + WarpKIndex;
    const unsigned TCUEmStrideX2 = 256;
    unsigned SLBbTCUOffset = SLBaTCUIndex * TCUEmStrideX2;
    unsigned EmIdx = (WarpMIndex & 1) * 2;
    a[0] = *((unsigned int*)p + SLBbTCUOffset + a.getColEMOffset(EmIdx % 4));
    a[1] = *((unsigned int*)p + SLBbTCUOffset + a.getColEMOffset((EmIdx + 1) % 4));
  }
template<class MMBI>
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync_tcu(FragmentBRowB16<MMBI>& a, const void* p, unsigned WarpNIndex, unsigned WarpKIndex) {
    int laneId = __ivcorex_lane_id();
    unsigned SLBbTCUIndex = WarpKIndex * a.getBlockNloopCnt() + WarpNIndex;
    unsigned   RowEmOffset = (SLBbTCUIndex & 1) ? (laneId ^ 0x20) : laneId;
    const unsigned TCUEmStride = 128;
    int SLBbTCUOffset = SLBbTCUIndex * TCUEmStride;
    a[0] = *((unsigned int*)p + SLBbTCUOffset + RowEmOffset);
    a[1] = *((unsigned int*)p + SLBbTCUOffset + RowEmOffset + 64);
  }

template<class MMBI>
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync_tcu(FragmentBColB16<MMBI>& a, const void* p, unsigned WarpNIndex, unsigned WarpKIndex) {
    unsigned SLBbTCUIndex = WarpKIndex / 2 * a.getBlockNloopCnt() + WarpNIndex;
    const unsigned TCUEmStrideX2 = 256;
    unsigned SLBbTCUOffset = SLBbTCUIndex * TCUEmStrideX2;
    unsigned EmIdx = (WarpKIndex & 1) * 2;
    a[0] = *((unsigned int*)p + SLBbTCUOffset + a.getColEMOffset(EmIdx % 4));
    a[1] = *((unsigned int*)p + SLBbTCUOffset + a.getColEMOffset((EmIdx + 1) % 4));
  }

/*****************************************************/

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 16, 32, __half, row_major>& a, const __half* p, unsigned ldm) {
    __hmma_ld_row_b16<unsigned int, __half>(&(a.x[0]), p);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 16, 16, 32, __half, col_major>& a, const __half* p, unsigned ldm) {
    __hmma_ld_col_b16<unsigned int, __half>(&(a.x[0]), p);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 16, 16, 32, float>& a, const float* p, unsigned ldm, layout_t layout) {
    assert(layout == mem_row_major && "mem_col_major not supported for accumulator!");
    __imma_ld_row_b32<float, float>(&(a.x[0]), p);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 32, 32, 16, __half, col_major>& a, const __half* p, unsigned ldm) {
    __hmma_ld_col_b16<unsigned int, __half>(&(a.x[0]), p);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 32, 32, 16, __half, row_major>& a, const __half* p, unsigned ldm) {
    __hmma_ld_row_b16<unsigned int, __half>(&(a.x[0]), p);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 32, 32, 16, float>& a, const float* p, unsigned ldm, layout_t layout) {
    assert(layout == mem_row_major && "mem_col_major not supported for accumulator!");
    for (int tile_row = 0; tile_row < 2; tile_row++) {
      for (int tile_column = 0; tile_column < 2; tile_column++) {
        int tile_num = 2 * tile_row + tile_column;
        const float* ptr = p + 16 * 16 * tile_num;
        __imma_ld_row_b32<float, float>(&(a.x[tile_num * 4]), ptr);
      }
    }
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 32, 32, __half, row_major>& a, const __half* p, unsigned ldm) {
    __hmma_ld_row_b16<unsigned int, __half>(&(a.x[0]), p);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 16, 32, 32, __half, row_major>& a, const __half* p, unsigned ldm) {
    for (int tile_num = 0; tile_num < 2; tile_num++) {
      const __half* ptr = p + 16 * 32 * tile_num;
      __hmma_ld_row_b16<unsigned int, __half>(&(a.x[tile_num * 4]), ptr);
    }
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 16, 32, 32, float>& a, const float* p, unsigned ldm, layout_t layout) {
    assert(layout == mem_row_major && "mem_col_major not supported for accumulator!");
    for (int tile_num = 0; tile_num < 2; tile_num++) {
      const float* ptr = p + 16 * 16 * tile_num;
      __imma_ld_row_b32<float, float>(&(a.x[tile_num * 4]), ptr);
    }
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 32, 16, 32, __half, col_major>& a, const __half* p, unsigned ldm) {
    for (int tile_num = 0; tile_num < 2; tile_num++) {
      const __half* ptr = p + 16 * 32 * tile_num;
      __hmma_ld_col_b16<unsigned int, __half>(&(a.x[tile_num * 4]), ptr);
    }
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 32, 16, 32, __half, col_major>& a, const __half* p, unsigned ldm) {
    __hmma_ld_col_b16<unsigned int, __half>(&(a.x[0]), p);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 32, 16, 32, float>& a, const float* p, unsigned ldm, layout_t layout) {
    assert(layout == mem_row_major && "mem_col_major not supported for accumulator!");
    for (int tile_num = 0; tile_num < 2; tile_num++) {
      const float* ptr = p + 16 * 16 * tile_num;
      __imma_ld_row_b32<float, float>(&(a.x[tile_num * 4]), ptr);
    }
  }

  //
  // Load functions for frags of A, B, C, D: F32, F32, F32, F32
  //
  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 16, 16, float, row_major>& a, const float* p, unsigned ldm) {
    __imma_ld_row_b32<float, float>(&(a.x[0]), p);
  }

  template <typename MatrixType, typename PtrType>
  __CUDA_MMA_DEVICE_DECL__ void __imma_ld_col_b32(MatrixType* a, const PtrType* p) {
    int laneId = __ivcorex_lane_id();

    for(int quarter_tile = 0; quarter_tile < 4; quarter_tile++) {
      int row = laneId / 16 + 4 * quarter_tile;
      int column = laneId % 16;
      int offset = CoordToOffset<32, layout_t::mem_col_major>(row, column);
      a[quarter_tile] = *(p + offset);
    }
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_a, 16, 16, 16, float, col_major>& a, const float* p, unsigned ldm) {
    __imma_ld_col_b32<float, float>(&(a.x[0]), p);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 16, 16, 16, float, row_major>& a, const float* p, unsigned ldm) {
    __imma_ld_row_b32<float, float>(&(a.x[0]), p);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<matrix_b, 16, 16, 16, float, col_major>& a, const float* p, unsigned ldm) {
    __imma_ld_col_b32<float, float>(&(a.x[0]), p);
  }

  __CUDA_MMA_DEVICE_DECL__ void load_matrix_sync(fragment<accumulator, 16, 16, 16, float>& a, const float* p, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major)
      __imma_ld_row_b32<float, float>(&(a.x[0]), p);
  }

  template <typename MatrixType, typename PtrType>
  __CUDA_MMA_DEVICE_DECL__ void __imma_st_row_b32(const MatrixType* a, PtrType* p) {
    int laneId = __ivcorex_lane_id();

    for(int quarter_tile = 0; quarter_tile < 4; quarter_tile++) {
      int row = laneId / 16 + 4 * quarter_tile;
      int column = laneId % 16;
      int offset = CoordToOffset<32, layout_t::mem_row_major>(row, column);
      *(p + offset) = a[quarter_tile];
    }
  }

  //
  // Store functions for frags of A, B, C, D: I8, I8, I32, I32
  //
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(signed int* p, const fragment<accumulator, 16, 16, 64, signed int>& a, unsigned ldm, layout_t layout) {
    assert(layout == mem_row_major && "mem_col_major not supported for accumulator!");
    __imma_st_row_b32<signed int, signed int>(&(a.x[0]), p);
  }

#ifdef __BI__
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(signed int* p, const fragment<accumulator, 64, 64, 16, signed int>& a, unsigned ldm, layout_t layout) {
    assert(layout == mem_row_major && "mem_col_major not supported for accumulator!");
    for (int tile_row = 0; tile_row < 4; tile_row++) {
      for (int tile_column = 0; tile_column < 4; tile_column++) {
        int tile_num = 4 * tile_row + tile_column;
        signed int* ptr = p + 16 * 16 * tile_num;
        __imma_st_row_b32<signed int, signed int>(&(a.x[tile_num * 4]), ptr);
      }
    } 
  }
#endif /* __BI__ */

#ifdef __MR__
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(signed int* p, const fragment<accumulator, 64, 64, 32, signed int>& a, unsigned ldm, layout_t layout) {
    assert(layout == mem_row_major && "mem_col_major not supported for accumulator!");
    for (int tile_row = 0; tile_row < 4; tile_row++) {
      for (int tile_column = 0; tile_column < 4; tile_column++) {
        int tile_num = 4 * tile_row + tile_column;
        signed int* ptr = p + 16 * 16 * tile_num;
        __imma_st_row_b32<signed int, signed int>(&(a.x[tile_num * 4]), ptr);
      }
    } 
  }
#endif /* __MR__ */

  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(signed int* p, const fragment<accumulator, 16, 64, 64, signed int>& a, unsigned ldm, layout_t layout) {
    assert(layout == mem_row_major && "mem_col_major not supported for accumulator!");
    for (int tile_num = 0; tile_num < 4; tile_num++) {
      signed int* ptr = p + 16 * 16 * tile_num;
      __imma_st_row_b32<signed int, signed int>(&(a.x[tile_num * 4]), ptr);
    }
  }

  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(signed int* p, const fragment<accumulator, 64, 16, 64, signed int>& a, unsigned ldm, layout_t layout) {
    assert(layout == mem_row_major && "mem_col_major not supported for accumulator!");
    for (int tile_num = 0; tile_num < 4; tile_num++) {
      signed int* ptr = p + 16 * 16 * tile_num;
      __imma_st_row_b32<signed int, signed int>(&(a.x[tile_num * 4]), ptr);
    }
  }

  //
  // Store functions for frags of A, B, C, D: U8, U8, U32, U32
  //
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(unsigned int* p, const fragment<accumulator, 16, 16, 64, unsigned int>& a, unsigned ldm, layout_t layout) {
    assert(layout == mem_row_major && "mem_col_major not supported for accumulator!");
    __imma_st_row_b32<unsigned int, unsigned int>(&(a.x[0]), p);
  }

#ifdef __BI__
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(unsigned int* p, const fragment<accumulator, 64, 64, 16, unsigned int>& a, unsigned ldm, layout_t layout) {
    assert(layout == mem_row_major && "mem_col_major not supported for accumulator!");
    for (int tile_row = 0; tile_row < 4; tile_row++) {
      for (int tile_column = 0; tile_column < 4; tile_column++) {
        int tile_num = 4 * tile_row + tile_column;
        unsigned int* ptr = p + 16 * 16 * tile_num;
        __imma_st_row_b32<unsigned int, unsigned int>(&(a.x[tile_num * 4]), ptr);
      }
    } 
  }
#endif /* __BI__ */

#ifdef __MR__
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(unsigned int* p, const fragment<accumulator, 64, 64, 32, unsigned int>& a, unsigned ldm, layout_t layout) {
    assert(layout == mem_row_major && "mem_col_major not supported for accumulator!");
    for (int tile_row = 0; tile_row < 4; tile_row++) {
      for (int tile_column = 0; tile_column < 4; tile_column++) {
        int tile_num = 4 * tile_row + tile_column;
        unsigned int* ptr = p + 16 * 16 * tile_num;
        __imma_st_row_b32<unsigned int, unsigned int>(&(a.x[tile_num * 4]), ptr);
      }
    } 
  }
#endif /* __MR__ */

  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(unsigned int* p, const fragment<accumulator, 16, 64, 64, unsigned int>& a, unsigned ldm, layout_t layout) {
    assert(layout == mem_row_major && "mem_col_major not supported for accumulator!");
    for (int tile_num = 0; tile_num < 4; tile_num++) {
      unsigned int* ptr = p + 16 * 16 * tile_num;
      __imma_st_row_b32<unsigned int, unsigned int>(&(a.x[tile_num * 4]), ptr);
    }
  }

  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(unsigned int* p, const fragment<accumulator, 64, 16, 64, unsigned int>& a, unsigned ldm, layout_t layout) {
    assert(layout == mem_row_major && "mem_col_major not supported for accumulator!");
    for (int tile_num = 0; tile_num < 4; tile_num++) {
      unsigned int* ptr = p + 16 * 16 * tile_num;
      __imma_st_row_b32<unsigned int, unsigned int>(&(a.x[tile_num * 4]), ptr);
    }
  }

  //
  // Store functions for frags of A, B, C, D: F16, F16, F32, F32
  //
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(float *p, const fragment<accumulator, 16, 16, 32, float>& a, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major) {
      __imma_st_row_b32<float, float>(&(a.x[0]), p);
    }
  }

  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(float *p, const fragment<accumulator, 32, 32, 16, float>& a, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major) {
      for (int tile_row = 0; tile_row < 2; tile_row++) {
        for (int tile_column = 0; tile_column < 2; tile_column++) {
          int tile_num = 2 * tile_row + tile_column;
          float* ptr = p + 16 * 16 * tile_num;
          __imma_st_row_b32<float, float>(&(a.x[tile_num * 4]), ptr);
        }
      }
    }
  }

  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(float *p, const fragment<accumulator, 16, 32, 32, float>& a, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major){
      for (int tile_num = 0; tile_num < 2; tile_num++) {
        float* ptr = p + 16 * 16 * tile_num;
        __imma_st_row_b32<float, float>(&(a.x[tile_num * 4]), ptr);
      }
    }
  }

  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(float *p, const fragment<accumulator, 32, 16, 32, float>& a, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major) {
      for (int tile_num = 0; tile_num < 2; tile_num++) {
        float* ptr = p + 16 * 16 * tile_num;
        __imma_st_row_b32<float, float>(&(a.x[tile_num * 4]), ptr);
      }
    }
  }

  //
  // Store functions for frags of A, B, C, D: F32, F32, F32, F32
  // 
  __CUDA_MMA_DEVICE_DECL__ void store_matrix_sync(float *p, const fragment<accumulator, 16, 16, 16, float>& a, unsigned ldm, layout_t layout) {
    if (layout == mem_row_major)
      __imma_st_row_b32<float, float>(&(a.x[0]), p);
  }

  // 
  // MMA functions for A, B, C, D: I8, I8, I32, I32
  //
#ifdef __MR__
  template<class MMBI>
  __CUDA_MMA_DEVICE_DECL__ void mma_sync_tcu(fragmentIx<MMBI, accumulator, signed int>& d, const FragmentARowB8<MMBI>& a, const FragmentBColB8<MMBI>& b, const fragmentIx<MMBI, accumulator, signed int>& c) {
      *((v4i32*)&d) = __ivcorex_matrix_mad_i32x4_i8x8(*(v2i32*)&a, *(v2i32*)&b, *(v4i32*)&c);
  }

  template<class MMBI>
  __CUDA_MMA_DEVICE_DECL__ void mma_sync_tcu(fragmentIx<MMBI, accumulator, signed int>& d, const FragmentAColB8<MMBI>& a, const FragmentBRowB8<MMBI>& b, const fragmentIx<MMBI, accumulator, signed int>& c) {
      *((v4i32*)&d) = __ivcorex_matrix_mad_i32x4_i8x8(*(v2i32*)&a, *(v2i32*)&b, *(v4i32*)&c);
  }

  template<class MMBI>
  __CUDA_MMA_DEVICE_DECL__ void mma_sync_tcu(fragmentIx<MMBI, accumulator, signed int>& d, const FragmentARowB8<MMBI>& a, const FragmentBRowB8<MMBI>& b, const fragmentIx<MMBI, accumulator, signed int>& c) {
      *((v4i32*)&d) = __ivcorex_matrix_mad_i32x4_i8x8(*(v2i32*)&a, *(v2i32*)&b, *(v4i32*)&c);
  }

  template<class MMBI>
  __CUDA_MMA_DEVICE_DECL__ void mma_sync_tcu(fragmentIx<MMBI, accumulator, signed int>& d, const FragmentAColB8<MMBI>& a, const FragmentBColB8<MMBI>& b, const fragmentIx<MMBI, accumulator, signed int>& c) {
      *((v4i32*)&d) = __ivcorex_matrix_mad_i32x4_i8x8(*(v2i32*)&a, *(v2i32*)&b, *(v4i32*)&c);
  }
#endif /* __MR__ */

  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 16, 16, 64, signed int>& d, const fragment<matrix_a, 16, 16, 64, signed char, row_major>& a, const fragment<matrix_b, 16, 16, 64, signed char, col_major>& b, const fragment<accumulator, 16, 16, 64, signed int>& c) {
#ifdef __BI__
    /**
     * A: A0A1A2A3    B:  B0
     *                    B1
     *                    B2
     *                    B3
     *    
    */
    // A0 * B0 + A1 * B1 + A2 * B2 + A3 * B3
    *(reinterpret_cast<v4i32*>(&(d.x))) = __ivcorex_matrix_mad_i32x4_i8x4(*(reinterpret_cast<const v4i8*>(&(a.x))), *(reinterpret_cast<const v4i8*>(&(b.x))), reinterpret_cast<const v4i32&>(c.x));

    for (int tile_num = 1; tile_num < 4; tile_num++) {
      *(reinterpret_cast<v4i32*>(&(d.x))) = __ivcorex_matrix_mad_i32x4_i8x4(*(reinterpret_cast<const v4i8*>(&(a.x)) + tile_num), *(reinterpret_cast<const v4i8*>(&(b.x)) + tile_num), reinterpret_cast<const v4i32&>(d.x));
    }
#endif /* __BI__ */

#ifdef __MR__
    /**
     * A: A0A1        B:  B0
     *                    B1
     *    
    */
    // A0 * B0 + A1 * B1
    *(reinterpret_cast<v4i32*>(&(d.x))) = __ivcorex_matrix_mad_i32x4_i8x8(*(reinterpret_cast<const v8i8*>(&(a.x))), *(reinterpret_cast<const v8i8*>(&(b.x))), reinterpret_cast<const v4i32&>(c.x));

    for (int tile_num = 1; tile_num < 2; tile_num++) {
      *(reinterpret_cast<v4i32*>(&(d.x))) = __ivcorex_matrix_mad_i32x4_i8x8(*(reinterpret_cast<const v8i8*>(&(a.x)) + tile_num), *(reinterpret_cast<const v8i8*>(&(b.x)) + tile_num), reinterpret_cast<const v4i32&>(d.x));
    }
#endif /* __MR__ */
  }

#ifdef __BI__
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 64, 64, 16, signed int>& d, const fragment<matrix_a, 64, 64, 16, signed char, col_major>& a, const fragment<matrix_b, 64, 64, 16, signed char, row_major>& b, const fragment<accumulator, 64, 64, 16, signed int>& c) {
    /*
      A:  A0      B:  B0B1B2B3
          A1
          A2
          A3
    */
    for (int row_tile = 0; row_tile < 4; row_tile++) {
      for (int column_tile = 0; column_tile < 4; column_tile++) {
        *(reinterpret_cast<v4i32*>(&(d.x)) + 4 * row_tile + column_tile) = __ivcorex_matrix_mad_i32x4_i8x4(*(reinterpret_cast<const v4i8*>(&(a.x)) + row_tile), *(reinterpret_cast<const v4i8*>(&(b.x)) + column_tile), *(reinterpret_cast<const v4i32*>(&(c.x)) + 4 * row_tile + column_tile));
      }
    }
  }
#endif /* __BI__ */

#ifdef __MR__
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 64, 64, 32, signed int>& d, const fragment<matrix_a, 64, 64, 32, signed char, col_major>& a, const fragment<matrix_b, 64, 64, 32, signed char, row_major>& b, const fragment<accumulator, 64, 64, 32, signed int>& c) {
    /*
      A:  A0      B:  B0B1B2B3    C: C0C1C2C3
          A1                         C4C5C6C7
          A2                         C8C9C10C11
          A3                         C12C13C14C15
    */
    v2i32 a_tile, b_tile;
    int a_start_vreg, b_start_vreg;
    for (int row_tile = 0; row_tile < 4; row_tile++) {
      a_start_vreg = row_tile;
      a_tile[0] = *(reinterpret_cast<const int*>(&(a.x)) + a_start_vreg);
      a_tile[1] = *(reinterpret_cast<const int*>(&(a.x)) + 4 + a_start_vreg);
      

      for (int column_tile = 0; column_tile < 4; column_tile++) {
        b_start_vreg = column_tile;
        b_tile[0] = *(reinterpret_cast<const int*>(&(b.x)) + b_start_vreg);
        b_tile[1] = *(reinterpret_cast<const int*>(&(b.x)) + 4 + b_start_vreg);

        *(reinterpret_cast<v4i32*>(&(d.x)) + 4 * row_tile + column_tile) = __ivcorex_matrix_mad_i32x4_i8x8(reinterpret_cast<const v8i8&>(a_tile), reinterpret_cast<const v8i8&>(b_tile), *(reinterpret_cast<const v4i32*>(&(c.x)) + 4 * row_tile + column_tile));
      }
    }
}
#endif /* __MR__ */

  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 16, 64, 64, signed int>& d, const fragment<matrix_a, 16, 64, 64, signed char, row_major>& a, const fragment<matrix_b, 16, 64, 64, signed char, row_major>& b, const fragment<accumulator, 16, 64, 64, signed int>& c) {
#ifdef __BI__
    /*
      A:  A0A1A2A3    B:  B0B1B2B3        C:  C0C1C2C3
                          B4B5B6B7
                          B8B9B10B11
                          B12B13B14B15
    */
    // A0 * B0 + A1 * B4 + A2 * B8 + A3 * B12
    // A0 * B1 + A1 * B5 + A2 * B9 + A3 * B13
    // A0 * B2 + A1 * B6 + A2 * B10 + A3 * B14
    // A0 * B3 + A1 * B7 + A2 * B11 + A3 * B15
    for (int start_tile = 0; start_tile < 4; start_tile++) {
      *(reinterpret_cast<v4i32*>(&(d.x)) + start_tile) = __ivcorex_matrix_mad_i32x4_i8x4(*(reinterpret_cast<const v4i8*>(&(a.x))), *(reinterpret_cast<const v4i8*>(&(b.x)) + start_tile), *(reinterpret_cast<const v4i32*>(&(c.x)) + start_tile));
    }

    for (int start_tile = 0; start_tile < 4; start_tile++) {
      for (int tile_step = 1; tile_step < 4; tile_step++) {
        *(reinterpret_cast<v4i32*>(&(d.x)) + start_tile) = __ivcorex_matrix_mad_i32x4_i8x4(*(reinterpret_cast<const v4i8*>(&(a.x)) + tile_step), *(reinterpret_cast<const v4i8*>(&(b.x)) + start_tile + 4 * tile_step), *(reinterpret_cast<const v4i32*>(&(d.x)) + start_tile));
      }
    }
#endif /* __BI__ */

#ifdef __MR__
    /*
      A:  A0A1        B:  B0B1B2B3        C:  C0C1C2C3
                          B4B5B6B7
    */
    // A0 * B0 + A1 * B4 
    // A0 * B1 + A1 * B5 
    // A0 * B2 + A1 * B6
    // A0 * B3 + A1 * B7
    v2i32 b_tile;
    int start_vreg = 0;

    for (int start_tile = 0; start_tile < 4; start_tile++) {
      start_vreg = start_tile;
      b_tile[0] = *(reinterpret_cast<const int*>(&(b.x)) + start_vreg);
      b_tile[1] = *(reinterpret_cast<const int*>(&(b.x)) + 4 + start_vreg);
      *(reinterpret_cast<v4i32*>(&(d.x)) + start_tile) = __ivcorex_matrix_mad_i32x4_i8x8(reinterpret_cast<const v8i8&>(a.x), reinterpret_cast<const v8i8&>(b_tile), *(reinterpret_cast<const v4i32*>(&(c.x)) + start_tile));
    }

    for (int start_tile = 0; start_tile < 4; start_tile++) {
      start_vreg = 8 + start_tile;
      b_tile[0] = *(reinterpret_cast<const int*>(&(b.x)) + start_vreg);
      b_tile[1] = *(reinterpret_cast<const int*>(&(b.x)) + 4 + start_vreg);
      *(reinterpret_cast<v4i32*>(&(d.x)) + start_tile) = __ivcorex_matrix_mad_i32x4_i8x8(*(reinterpret_cast<const v8i8*>(&(a.x)) + 1), reinterpret_cast<const v8i8&>(b_tile), *(reinterpret_cast<const v4i32*>(&(d.x)) + start_tile));
    }
#endif /* __MR__ */
  }

  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 64, 16, 64, signed int>& d, const fragment<matrix_a, 64, 16, 64, signed char, col_major>& a, const fragment<matrix_b, 64, 16, 64, signed char, col_major>& b, const fragment<accumulator, 64, 16, 64, signed int>& c) {
#ifdef __BI__
    /*
      A:  A0 A4 A8  A12     B:  B0      C:  C0
          A1 A5 A9  A13         B1          C1
          A2 A6 A10 A14         B2          C2
          A3 A7 A11 A15         B3          C3

    */
    // A0 * B0 + A4 * B1 + A8 * B2 + A12 * B3
    // A1 * B0 + A5 * B1 + A9 * B2 + A13 * B3
    // A2 * B0 + A6 * B1 + A10 * B2 + A14 * B3
    // A3 * B0 + A7 * B1 + A11 * B2 + A15 * B3
    for (int start_tile = 0; start_tile < 4; start_tile++) {
      *(reinterpret_cast<v4i32*>(&(d.x)) + start_tile) = __ivcorex_matrix_mad_i32x4_i8x4(*(reinterpret_cast<const v4i8*>(&(a.x)) + start_tile), *(reinterpret_cast<const v4i8*>(&(b.x))), *(reinterpret_cast<const v4i32*>(&(c.x)) + start_tile));
    }

    for (int start_tile = 0; start_tile < 4; start_tile++) {
      for (int tile_step = 1; tile_step < 4; tile_step++) {
        *(reinterpret_cast<v4i32*>(&(d.x)) + start_tile) = __ivcorex_matrix_mad_i32x4_i8x4(*(reinterpret_cast<const v4i8*>(&(a.x)) + start_tile + 4 * tile_step), *(reinterpret_cast<const v4i8*>(&(b.x)) + tile_step), *(reinterpret_cast<const v4i32*>(&(d.x)) + start_tile));
      }
    }
#endif /* __BI__ */

#ifdef __MR__
    /*
      A:  A0 A4        B:  B0      C:  C0
          A1 A5            B1          C1
          A2 A6                        C2
          A3 A7                        C3

    */
    // A0 * B0 + A4 * B1 
    // A1 * B0 + A5 * B1
    // A2 * B0 + A6 * B1
    // A3 * B0 + A7 * B1
    v2i32 a_tile;
    int start_vreg = 0;

    for (int start_tile = 0; start_tile < 4; start_tile++) {
      start_vreg = start_tile;
      a_tile[0] = *(reinterpret_cast<const int*>(&(a.x)) + start_vreg);
      a_tile[1] = *(reinterpret_cast<const int*>(&(a.x)) + 4 + start_vreg);
      *(reinterpret_cast<v4i32*>(&(d.x)) + start_tile) = __ivcorex_matrix_mad_i32x4_i8x8(reinterpret_cast<const v8i8&>(a_tile), reinterpret_cast<const v8i8&>(b.x), *(reinterpret_cast<const v4i32*>(&(c.x)) + start_tile));
    }

    for (int start_tile = 0; start_tile < 4; start_tile++) {
      start_vreg = 8 + start_tile;
      a_tile[0] = *(reinterpret_cast<const int*>(&(a.x)) + start_vreg);
      a_tile[1] = *(reinterpret_cast<const int*>(&(a.x)) + 4 + start_vreg);
      *(reinterpret_cast<v4i32*>(&(d.x)) + start_tile) = __ivcorex_matrix_mad_i32x4_i8x8(reinterpret_cast<const v8i8&>(a_tile), *(reinterpret_cast<const v8i8*>(&(b.x)) + 1), *(reinterpret_cast<const v4i32*>(&(d.x)) + start_tile));
    }
#endif /* __MR__ */
  }

  // 
  // MMA functions for A, B, C, D: U8, U8, U32, U32
  //
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 16, 16, 64, unsigned int>& d, const fragment<matrix_a, 16, 16, 64, unsigned char, row_major>& a, const fragment<matrix_b, 16, 16, 64, unsigned char, col_major>& b, const fragment<accumulator, 16, 16, 64, unsigned int>& c) {
#ifdef __BI__
    /**
     * A: A0A1A2A3    B:  B0
     *                    B1
     *                    B2
     *                    B3
     *    
    */
    // A0 * B0 + A1 * B1 + A2 * B2 + A3 * B3
    *(reinterpret_cast<v4u32*>(&(d.x))) = __ivcorex_matrix_mad_u32x4_u8x4(*(reinterpret_cast<const v4u8*>(&(a.x))), *(reinterpret_cast<const v4u8*>(&(b.x))), reinterpret_cast<const v4u32&>(c.x));

    for (int tile_num = 1; tile_num < 4; tile_num++) {
      *(reinterpret_cast<v4u32*>(&(d.x))) = __ivcorex_matrix_mad_u32x4_u8x4(*(reinterpret_cast<const v4u8*>(&(a.x)) + tile_num), *(reinterpret_cast<const v4u8*>(&(b.x)) + tile_num), reinterpret_cast<const v4u32&>(d.x));
    }
#endif /* __BI__ */

#ifdef __MR__
    /**
     * A: A0A1        B:  B0
     *                    B1
     *    
    */
    // A0 * B0 + A1 * B1
    *(reinterpret_cast<v4u32*>(&(d.x))) = __ivcorex_matrix_mad_u32x4_u8x8(*(reinterpret_cast<const v8u8*>(&(a.x))), *(reinterpret_cast<const v8u8*>(&(b.x))), reinterpret_cast<const v4u32&>(c.x));

    for (int tile_num = 1; tile_num < 2; tile_num++) {
      *(reinterpret_cast<v4u32*>(&(d.x))) = __ivcorex_matrix_mad_u32x4_u8x8(*(reinterpret_cast<const v8u8*>(&(a.x)) + tile_num), *(reinterpret_cast<const v8u8*>(&(b.x)) + tile_num), reinterpret_cast<const v4u32&>(d.x));
    }
#endif /* __MR__ */
  }

#ifdef __BI__
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 64, 64, 16, unsigned int>& d, const fragment<matrix_a, 64, 64, 16, unsigned char, col_major>& a, const fragment<matrix_b, 64, 64, 16, unsigned char, row_major>& b, const fragment<accumulator, 64, 64, 16, unsigned int>& c) {
    /*
      A:  A0      B:  B0B1B2B3
          A1
          A2
          A3
    */
    for (int row_tile = 0; row_tile < 4; row_tile++) {
      for (int column_tile = 0; column_tile < 4; column_tile++) {
        *(reinterpret_cast<v4i32*>(&(d.x)) + 4 * row_tile + column_tile) = __ivcorex_matrix_mad_i32x4_i8x4(*(reinterpret_cast<const v4i8*>(&(a.x)) + row_tile), *(reinterpret_cast<const v4i8*>(&(b.x)) + column_tile), *(reinterpret_cast<const v4i32*>(&(c.x)) + 4 * row_tile + column_tile));
      }
    }
  }
#endif /* __BI__ */

#ifdef __MR__
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 64, 64, 32, unsigned int>& d, const fragment<matrix_a, 64, 64, 32, unsigned char, col_major>& a, const fragment<matrix_b, 64, 64, 32, unsigned char, row_major>& b, const fragment<accumulator, 64, 64, 32, unsigned int>& c) {
    /*
      A:  A0      B:  B0B1B2B3    C: C0C1C2C3
          A1                         C4C5C6C7
          A2                         C8C9C10C11
          A3                         C12C13C14C15
    */
    v2i32 a_tile, b_tile;
    int a_start_vreg, b_start_vreg;
    for (int row_tile = 0; row_tile < 4; row_tile++) {
      a_start_vreg = row_tile;
      a_tile[0] = *(reinterpret_cast<const unsigned int*>(&(a.x)) + a_start_vreg);
      a_tile[1] = *(reinterpret_cast<const unsigned int*>(&(a.x)) + 4 + a_start_vreg);
      

      for (int column_tile = 0; column_tile < 4; column_tile++) {
        b_start_vreg = column_tile;
        b_tile[0] = *(reinterpret_cast<const unsigned int*>(&(b.x)) + b_start_vreg);
        b_tile[1] = *(reinterpret_cast<const unsigned int*>(&(b.x)) + 4 + b_start_vreg);

        *(reinterpret_cast<v4u32*>(&(d.x)) + 4 * row_tile + column_tile) = __ivcorex_matrix_mad_i32x4_i8x8(reinterpret_cast<const v8u8&>(a_tile), reinterpret_cast<const v8u8&>(b_tile), *(reinterpret_cast<const v4u32*>(&(c.x)) + 4 * row_tile + column_tile));
      }
    }
}
#endif /* __MR__ */

  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 16, 64, 64, unsigned int>& d, const fragment<matrix_a, 16, 64, 64, unsigned char, row_major>& a, const fragment<matrix_b, 16, 64, 64, unsigned char, row_major>& b, const fragment<accumulator, 16, 64, 64, unsigned int>& c) {
#ifdef __BI__
    /*
      A:  A0A1A2A3    B:  B0B1B2B3        C:  C0C1C2C3
                          B4B5B6B7
                          B8B9B10B11
                          B12B13B14B15
    */
    // A0 * B0 + A1 * B4 + A2 * B8 + A3 * B12
    // A0 * B1 + A1 * B5 + A2 * B9 + A3 * B13
    // A0 * B2 + A1 * B6 + A2 * B10 + A3 * B14
    // A0 * B3 + A1 * B7 + A2 * B11 + A3 * B15
    for (int start_tile = 0; start_tile < 4; start_tile++) {
      *(reinterpret_cast<v4u32*>(&(d.x)) + start_tile) = __ivcorex_matrix_mad_u32x4_u8x4(*(reinterpret_cast<const v4u8*>(&(a.x))), *(reinterpret_cast<const v4u8*>(&(b.x)) + start_tile), *(reinterpret_cast<const v4u32*>(&(c.x)) + start_tile));
    }

    for (int start_tile = 0; start_tile < 4; start_tile++) {
      for (int tile_step = 1; tile_step < 4; tile_step++) {
        *(reinterpret_cast<v4u32*>(&(d.x)) + start_tile) = __ivcorex_matrix_mad_u32x4_u8x4(*(reinterpret_cast<const v4u8*>(&(a.x)) + tile_step), *(reinterpret_cast<const v4u8*>(&(b.x)) + start_tile + 4 * tile_step), *(reinterpret_cast<const v4u32*>(&(d.x)) + start_tile));
      }
    }
#endif /* __BI__ */

#ifdef __MR__
    /*
      A:  A0A1        B:  B0B1B2B3        C:  C0C1C2C3
                          B4B5B6B7
    */
    // A0 * B0 + A1 * B4 
    // A0 * B1 + A1 * B5 
    // A0 * B2 + A1 * B6
    // A0 * B3 + A1 * B7
    v2i32 b_tile;
    int start_vreg = 0;

    for (int start_tile = 0; start_tile < 4; start_tile++) {
      start_vreg = start_tile;
      b_tile[0] = *(reinterpret_cast<const unsigned int*>(&(b.x)) + start_vreg);
      b_tile[1] = *(reinterpret_cast<const unsigned int*>(&(b.x)) + 4 + start_vreg);
      *(reinterpret_cast<v4u32*>(&(d.x)) + start_tile) = __ivcorex_matrix_mad_u32x4_u8x8(reinterpret_cast<const v8u8&>(a.x), reinterpret_cast<const v8u8&>(b_tile), *(reinterpret_cast<const v4u32*>(&(c.x)) + start_tile));
    }

    for (int start_tile = 0; start_tile < 4; start_tile++) {
      start_vreg = 8 + start_tile;
      b_tile[0] = *(reinterpret_cast<const unsigned int*>(&(b.x)) + start_vreg);
      b_tile[1] = *(reinterpret_cast<const unsigned int*>(&(b.x)) + 4 + start_vreg);
      *(reinterpret_cast<v4u32*>(&(d.x)) + start_tile) = __ivcorex_matrix_mad_u32x4_u8x8(*(reinterpret_cast<const v8u8*>(&(a.x)) + 1), reinterpret_cast<const v8u8&>(b_tile), *(reinterpret_cast<const v4u32*>(&(d.x)) + start_tile));
    }
#endif /* __MR__ */
  }

  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 64, 16, 64, unsigned int>& d, const fragment<matrix_a, 64, 16, 64, unsigned char, col_major>& a, const fragment<matrix_b, 64, 16, 64, unsigned char, col_major>& b, const fragment<accumulator, 64, 16, 64, unsigned int>& c) {
#ifdef __BI__
    /*
      A:  A0 A4 A8  A12     B:  B0      C:  C0
          A1 A5 A9  A13         B1          C1
          A2 A6 A10 A14         B2          C2
          A3 A7 A11 A15         B3          C3

    */
    // A0 * B0 + A4 * B1 + A8 * B2 + A12 * B3
    // A1 * B0 + A5 * B1 + A9 * B2 + A13 * B3
    // A2 * B0 + A6 * B1 + A10 * B2 + A14 * B3
    // A3 * B0 + A7 * B1 + A11 * B2 + A15 * B3
    for (int start_tile = 0; start_tile < 4; start_tile++) {
      *(reinterpret_cast<v4u32*>(&(d.x)) + start_tile) = __ivcorex_matrix_mad_u32x4_u8x4(*(reinterpret_cast<const v4u8*>(&(a.x)) + start_tile), *(reinterpret_cast<const v4u8*>(&(b.x))), *(reinterpret_cast<const v4u32*>(&(c.x)) + start_tile));
    }

    for (int start_tile = 0; start_tile < 4; start_tile++) {
      for (int tile_step = 1; tile_step < 4; tile_step++) {
        *(reinterpret_cast<v4u32*>(&(d.x)) + start_tile) = __ivcorex_matrix_mad_u32x4_u8x4(*(reinterpret_cast<const v4u8*>(&(a.x)) + start_tile + 4 * tile_step), *(reinterpret_cast<const v4u8*>(&(b.x)) + tile_step), *(reinterpret_cast<const v4u32*>(&(d.x)) + start_tile));
      }
    }
#endif /* __BI__ */

#ifdef __MR__
    /*
      A:  A0 A4        B:  B0      C:  C0
          A1 A5            B1          C1
          A2 A6                        C2
          A3 A7                        C3

    */
    // A0 * B0 + A4 * B1 
    // A1 * B0 + A5 * B1
    // A2 * B0 + A6 * B1
    // A3 * B0 + A7 * B1
    v2i32 a_tile;
    int start_vreg = 0;

    for (int start_tile = 0; start_tile < 4; start_tile++) {
      start_vreg = start_tile;
      a_tile[0] = *(reinterpret_cast<const unsigned int*>(&(a.x)) + start_vreg);
      a_tile[1] = *(reinterpret_cast<const unsigned int*>(&(a.x)) + 4 + start_vreg);
      *(reinterpret_cast<v4u32*>(&(d.x)) + start_tile) = __ivcorex_matrix_mad_u32x4_u8x8(reinterpret_cast<const v8u8&>(a_tile), reinterpret_cast<const v8u8&>(b.x), *(reinterpret_cast<const v4u32*>(&(c.x)) + start_tile));
    }

    for (int start_tile = 0; start_tile < 4; start_tile++) {
      start_vreg = 8 + start_tile;
      a_tile[0] = *(reinterpret_cast<const unsigned int*>(&(a.x)) + start_vreg);
      a_tile[1] = *(reinterpret_cast<const unsigned int*>(&(a.x)) + 4 + start_vreg);
      *(reinterpret_cast<v4u32*>(&(d.x)) + start_tile) = __ivcorex_matrix_mad_u32x4_u8x8(reinterpret_cast<const v8u8&>(a_tile), *(reinterpret_cast<const v8u8*>(&(b.x)) + 1), *(reinterpret_cast<const v4u32*>(&(d.x)) + start_tile));
    }
#endif /* __MR__ */
  }

  // 
  // MMA functions for A, B, C, D: F16, F16, F32, F32
  //
template<class MMBI>
  __CUDA_MMA_DEVICE_DECL__ void mma_sync_tcu(fragmentIx<MMBI, accumulator, float>& d, const FragmentARowB16<MMBI>& a, const FragmentBColB16<MMBI>& b, const fragmentIx<MMBI, accumulator, float>& c) {
    *((v4f32*)&d) = __ivcorex_matrix_mad_f32x4_f16x4(*(v4f16*)&a, *(v4f16*)&b, *(v4f32*)&c);
  }

template<class MMBI>
  __CUDA_MMA_DEVICE_DECL__ void mma_sync_tcu(fragmentIx<MMBI, accumulator, float>& d, const FragmentAColB16<MMBI>& a, const FragmentBRowB16<MMBI>& b, const fragmentIx<MMBI, accumulator, float>& c) {
    *((v4f32*)&d) = __ivcorex_matrix_mad_f32x4_f16x4(*(v4f16*)&a, *(v4f16*)&b, *(v4f32*)&c);
  }

template<class MMBI>
  __CUDA_MMA_DEVICE_DECL__ void mma_sync_tcu(fragmentIx<MMBI, accumulator, float>& d, const FragmentARowB16<MMBI>& a, const FragmentBRowB16<MMBI>& b, const fragmentIx<MMBI, accumulator, float>& c) {
    *((v4f32*)&d) = __ivcorex_matrix_mad_f32x4_f16x4(*(v4f16*)&a, *(v4f16*)&b, *(v4f32*)&c);
  }

template<class MMBI>
  __CUDA_MMA_DEVICE_DECL__ void mma_sync_tcu(fragmentIx<MMBI, accumulator, float>& d, const FragmentAColB16<MMBI>& a, const FragmentBColB16<MMBI>& b, const fragmentIx<MMBI, accumulator, float>& c) {
    *((v4f32*)&d) = __ivcorex_matrix_mad_f32x4_f16x4(*(v4f16*)&a, *(v4f16*)&b, *(v4f32*)&c);
  }

  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 16, 16, 32, float>& d, const fragment<matrix_a, 16, 16, 32, __half, row_major>& a, const fragment<matrix_b, 16, 16, 32, __half, col_major>& b, const fragment<accumulator, 16, 16, 32, float>& c) {

    *(reinterpret_cast<v4f32*>(&(d.x))) = __ivcorex_matrix_mad_f32x4_f16x4(*(reinterpret_cast<const v4f16*>(&(a.x))), *(reinterpret_cast<const v4f16*>(&(b.x))), reinterpret_cast<const v4f32&>(c.x));
    
    *(reinterpret_cast<v4f32*>(&(d.x))) = __ivcorex_matrix_mad_f32x4_f16x4(*(reinterpret_cast<const v4f16*>(&(a.x)) + 1), *(reinterpret_cast<const v4f16*>(&(b.x)) + 1), reinterpret_cast<const v4f32&>(d.x));
  }

  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 32, 16, float>& d, const fragment<matrix_a, 32, 32, 16, __half, col_major>& a, const fragment<matrix_b, 32, 32, 16, __half, row_major>& b, const fragment<accumulator, 32, 32, 16, float>& c) {
    /**
     * A: A0   B: B0B1
     *    A1
    */
    for (int row_tile = 0; row_tile < 2; row_tile++) {
      for (int column_tile = 0; column_tile < 2; column_tile++) {
        *(reinterpret_cast<v4f32*>(&(d.x)) + 2 * row_tile + column_tile) = __ivcorex_matrix_mad_f32x4_f16x4(*(reinterpret_cast<const v4f16*>(&(a.x)) + row_tile), *(reinterpret_cast<const v4f16*>(&(b.x)) + column_tile), *(reinterpret_cast<const v4f32*>(&(c.x)) + 2 * row_tile + column_tile));
      }
    }
  }

  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 16, 32, 32, float>& d, const fragment<matrix_a, 16, 32, 32, __half, row_major>& a, const fragment<matrix_b, 16, 32, 32, __half, row_major>& b, const fragment<accumulator, 16, 32, 32, float>& c) {
    /*
      A: A0A1   B: B0B1                 a.x[0]          a.x[2]
                  -------         B0 =>   ------    B1 => -------
                   B2B3                 a.x[1]          a.x[3]
    */
    // A0 * B0 + A1 * B2
    *(reinterpret_cast<v4f32*>(&(d.x))) = __ivcorex_matrix_mad_f32x4_f16x4(*(reinterpret_cast<const v4f16*>(&(a.x))), *(reinterpret_cast<const v4f16*>(&(b.x))), reinterpret_cast<const v4f32&>(c.x)); 

    *(reinterpret_cast<v4f32*>(&(d.x))) = __ivcorex_matrix_mad_f32x4_f16x4(*(reinterpret_cast<const v4f16*>(&(a.x)) + 1), *(reinterpret_cast<const v4f16*>(&(b.x)) + 2), *(reinterpret_cast<const v4f32*>(&(c.x))));

    // A0 * B1 + A1 * B3
    *(reinterpret_cast<v4f32*>(&(d.x)) + 1) = __ivcorex_matrix_mad_f32x4_f16x4(*(reinterpret_cast<const v4f16*>(&(a.x))), *(reinterpret_cast<const v4f16*>(&(b.x)) + 1), *(reinterpret_cast<const v4f32*>(&(c.x)) + 1)); 

    *(reinterpret_cast<v4f32*>(&(d.x)) + 1) = __ivcorex_matrix_mad_f32x4_f16x4(*(reinterpret_cast<const v4f16*>(&(a.x)) + 1), *(reinterpret_cast<const v4f16*>(&(b.x)) + 3), *(reinterpret_cast<const v4f32*>(&(d.x)) + 1)); 
  }

    __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 32, 16, 32, float>& d, const fragment<matrix_a, 32, 16, 32, __half, col_major>& a, const fragment<matrix_b, 32, 16, 32, __half, col_major>& b, const fragment<accumulator, 32, 16, 32, float>& c) {
    /*
       A: |A0|A2|         B: |B0|                a.x[0]          a.x[2]       C: C0
         -------             ----        A0 =>   ------    A1 => -------         --
          |A1|A3|            |B1|                a.x[1]          a.x[3]          C1

    */
    // A0 * B0 + A2 * B1
    *(reinterpret_cast<v4f32*>(&(d.x))) = __ivcorex_matrix_mad_f32x4_f16x4(*(reinterpret_cast<const v4f16*>(&(a.x))), *(reinterpret_cast<const v4f16*>(&(b.x))), reinterpret_cast<const v4f32&>(c.x));

    *(reinterpret_cast<v4f32*>(&(d.x))) = __ivcorex_matrix_mad_f32x4_f16x4(*(reinterpret_cast<const v4f16*>(&(a.x)) + 2), *(reinterpret_cast<const v4f16*>(&(b.x)) + 1), *(reinterpret_cast<const v4f32*>(&(d.x))));

    // A1 * B0 + A3 * B1
    *(reinterpret_cast<v4f32*>(&(d.x)) + 1) = __ivcorex_matrix_mad_f32x4_f16x4(*(reinterpret_cast<const v4f16*>(&(a.x)) + 1), *(reinterpret_cast<const v4f16*>(&(b.x))), *(reinterpret_cast<const v4f32*>(&(c.x)) + 1));

    *(reinterpret_cast<v4f32*>(&(d.x)) + 1) = __ivcorex_matrix_mad_f32x4_f16x4(*(reinterpret_cast<const v4f16*>(&(a.x)) + 3), *(reinterpret_cast<const v4f16*>(&(b.x)) + 1), *(reinterpret_cast<const v4f32*>(&(d.x)) + 1));
    }

  // 
  // MMA functions for A, B, C, D: F32, F32, F32, F32
  //
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 16, 16, 16, float>& d, const fragment<matrix_a, 16, 16, 16, float, row_major>& a, const fragment<matrix_b,16, 16, 16, float, col_major>& b, const fragment<accumulator, 16, 16, 16, float>& c) {
     *(reinterpret_cast<v4f32*>(&(d.x))) = __ivcorex_matrix_mad_f32x4_f32x4(*(reinterpret_cast<const v4f32*>(&(a.x))), *(reinterpret_cast<const v4f32*>(&(b.x))), *(reinterpret_cast<const v4f32*>(&(c.x))));
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 16, 16, 16, float>& d, const fragment<matrix_a, 16, 16, 16, float, row_major>& a, const fragment<matrix_b,16, 16, 16, float, row_major>& b, const fragment<accumulator, 16, 16, 16, float>& c) {
     *(reinterpret_cast<v4f32*>(&(d.x))) = __ivcorex_matrix_mad_f32x4_f32x4(*(reinterpret_cast<const v4f32*>(&(a.x))), *(reinterpret_cast<const v4f32*>(&(b.x))), *(reinterpret_cast<const v4f32*>(&(c.x))));
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 16, 16, 16, float>& d, const fragment<matrix_a, 16, 16, 16, float, col_major>& a, const fragment<matrix_b,16, 16, 16, float, col_major>& b, const fragment<accumulator, 16, 16, 16, float>& c) {
     *(reinterpret_cast<v4f32*>(&(d.x))) = __ivcorex_matrix_mad_f32x4_f32x4(*(reinterpret_cast<const v4f32*>(&(a.x))), *(reinterpret_cast<const v4f32*>(&(b.x))), *(reinterpret_cast<const v4f32*>(&(c.x))));
  }
  
  __CUDA_MMA_DEVICE_DECL__ void mma_sync(fragment<accumulator, 16, 16, 16, float>& d, const fragment<matrix_a, 16, 16, 16, float, col_major>& a, const fragment<matrix_b,16, 16, 16, float, row_major>& b, const fragment<accumulator, 16, 16, 16, float>& c) {
    *(reinterpret_cast<v4f32*>(&(d.x))) = __ivcorex_matrix_mad_f32x4_f32x4(*(reinterpret_cast<const v4f32*>(&(a.x))), *(reinterpret_cast<const v4f32*>(&(b.x))), *(reinterpret_cast<const v4f32*>(&(c.x))));
  }
};
};

#undef __DEF_IF_HOST
#undef __BI__
#undef __MR__
#undef __CUDA_MMA_DEVICE_DECL__
#endif /* !__CUDA_ARCH__ || __ILUVATAR__ */

#endif /* __cplusplus && __CUDACC__ */

#endif   /* __ILUVATAR_MMA_HPP__ */

#if defined(__UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_CUDA_MMA_H__)
#undef __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#undef __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_CUDA_MMA_H__
#endif