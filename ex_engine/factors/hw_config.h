// ex_engine/factors/hw_config.h
//
// Layer 1: Hardware descriptor + per-algorithm tuning tables
//
// Upstream parallel: xllm_models/llm/qwen3_5.h (model-level config)
//   → defines num_experts=64, top_k=8, hidden_size=3584, intermediate=18944
//   → binds GDN layers [1,7,13,19] vs full attention layers
//
// This file defines the BI-V100 hardware descriptor and per-factor
// tuning tables. Every downstream factor file #includes this to get
// compile-time constants that are calibrated to the real device.
//
// Source: cat_files/arch.h (real nm -D symbol dump from BI-V100)
//         cat_files/iluvatar_mma.hpp (tensor core instruction set)
//         sub694rizhi.txt TPS distribution (avg 6.2, target >12)

#ifndef EX_FACTORS_HW_CONFIG_H
#define EX_FACTORS_HW_CONFIG_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =========================================================================
// BI-V100 hardware constants (from real device probes)
// =========================================================================
// Source: cat_files/arch.h, HARDWARE_PROBE_20260808.md
#define EX_SM_MAJOR          7
#define EX_SM_MINOR          0
#define EX_SM_COUNT          16
#define EX_WARP_SIZE         32
#define EX_MAX_THREADS_SM    2048
#define EX_SMEM_PER_SM       49152   // bytes
#define EX_SMEM_PER_BLOCK    49152   // bytes, BI-V100 no bank partitioning
#define EX_L2_CACHE_BYTES    (6 * 1024 * 1024)  // 6 MB
#define EX_MEM_BW_GBS        900.0f  // total ~900 GB/s (56 GB/s per SM × 16)

// Register file: 65536 32-bit registers per SM, 256 per warp
// Source: probed via cudaDeviceProp
#define EX_REGS_PER_SM       65536
#define EX_REGS_PER_BLOCK    65536

// CoreX clang/16 CUDA 10.2 compatibility constraints
// Source: cat_files/cutlass.h (Iluvatar CoreX copyright block)
// - No cp.async (requires SM80+)
// - No TMA (requires SM90+)
// - Warp shuffle via __shfl_sync, __shfl_xor_sync, __shfl_down_sync
// - atomicAdd for float available (SM70)
// - half2 arithmetic available via __hmul2, __hadd2 etc.
#define EX_HAS_CP_ASYNC      0
#define EX_HAS_TMA           0
#define EX_HAS_WARP_SHUFFLE  1
#define EX_HAS_HALF2         1

// =========================================================================
// Qwen3.5-27B model constants (from SYSTEM_DESIGN.md)
// =========================================================================
#define EX_NUM_LAYERS        64   // decoder layers
#define EX_NUM_EXPERTS       64   // routed experts per MoE layer
#define EX_MOE_TOPK          8    // active experts per token
#define EX_HIDDEN_SIZE       3584 // hidden dimension
#define EX_INTERMEDIATE_SIZE 18944 // MoE intermediate (per expert, pre-TP)
#define EX_NUM_HEADS         28   // attention heads
#define EX_NUM_KV_HEADS      4    // GQA kv heads
#define EX_HEAD_DIM          128  // per-head dimension
#define EX_GDN_LAYERS_COUNT  4    // GatedDeltaNet layers: [1,7,13,19]
#define EX_ATTN_LAYERS_COUNT 32   // full attention layers (of 36 attention layers)
#define EX_MOE_LAYERS_COUNT  64   // all 64 layers have MoE

// TP=4 sliced sizes (actual runtime)
#define EX_TP_SIZE           4
#define EX_INTER_PER_TP      (EX_INTERMEDIATE_SIZE / EX_TP_SIZE)  // 4736

// =========================================================================
// Per-factor tuning parameter structs
// =========================================================================
// Mirrors CCCL's per-algorithm ReducePassPolicy/ScanPolicy pattern
// Each factor reads these at compile time or load time to set
// grid/block dims and SMEM allocation.

typedef struct {
    int threads_per_block;
    int items_per_thread;  // values per thread (VPT)
    int vec_size;          // vector load width
    int smem_bytes;        // shared memory per block
    int num_warps;         // threads_per_block / WARP_SIZE
    int num_stages;        // software pipeline stages
    int grid_scale;        // multiplier for grid dim (1 = 1:1 with N)
} ex_factor_tune_t;

// =========================================================================
// BI-V100 tuning tables — one entry per factor
// =========================================================================
// Source: CCCL tuning headers adapted via muh toolchain
//         sub168 TPS benchmarks, sub694 per-request TPS distribution
//
// Key tuning rationale per sub694 data:
//   - avg TPS = 6.2, ceiling TPS = 11.9 (decode-only, low prompt_tok)
//   - 227/824 requests below 5 TPS → bottleneck is prefill-heavy requests
//   - prompt_tok >100K → TPS drops to 0.4-1.2 (GEMM bound)
//   - prompt_tok <10K → TPS reaches 9-11 (compute matches HW)
//   - Goal: double TPS on prompt_tok 10K-50K range (bulk of traffic)

// Factor 0: MOE_TOPK_SOFTMAX
// 64 experts, VPT=2 → 32 threads = 1 warp per token row
// 4 warps per CTA → 4 tokens per CTA
// No SMEM needed (all warp shuffle)
// Upstream: moe_topk_softmax_kernels.cuh topkGatingSoftmax
static const ex_factor_tune_t EX_TUNE_MOE_TOPK_SOFTMAX = {
    .threads_per_block = 128,  // 4 warps × 32
    .items_per_thread  = 2,    // 64 experts / 32 threads
    .vec_size          = 1,    // float, no vectorization
    .smem_bytes        = 0,    // pure warp shuffle
    .num_warps         = 4,
    .num_stages        = 1,
    .grid_scale        = 1,    // ceil(N / 4) blocks
};

// Factor 1: MOE_ALIGN_BLOCK
// Simple histogram + prefix sum, 256 threads per block
// SMEM for CUB BlockScan
static const ex_factor_tune_t EX_TUNE_MOE_ALIGN_BLOCK = {
    .threads_per_block = 256,
    .items_per_thread  = 1,
    .vec_size          = 1,
    .smem_bytes        = 1024,  // CUB BlockScan TempStorage
    .num_warps         = 8,
    .num_stages        = 1,
    .grid_scale        = 1,
};

// Factor 2: MOE_FUSED_GEMM
// Group GEMM: cublas or CUTLASS batched
// BI-V100: cublas SM70 hgemm, M×N×K per expert
// threads/smem managed by cublas internally
static const ex_factor_tune_t EX_TUNE_MOE_FUSED_GEMM = {
    .threads_per_block = 256,  // cublas managed
    .items_per_thread  = 4,
    .vec_size          = 8,    // half8 loads
    .smem_bytes        = 49152, // full SMEM for GEMM tiles
    .num_warps         = 8,
    .num_stages        = 2,    // SW pipeline (no cp.async, manual prefetch)
    .grid_scale        = 1,
};

// Factor 3: GELU_TANH_MUL
// Element-wise, memory-bandwidth bound
// 256 threads, vec4 half loads
static const ex_factor_tune_t EX_TUNE_GELU_TANH_MUL = {
    .threads_per_block = 256,
    .items_per_thread  = 4,
    .vec_size          = 4,    // half4 = 8 bytes
    .smem_bytes        = 0,    // pure register
    .num_warps         = 8,
    .num_stages        = 1,
    .grid_scale        = 1,
};

// Factor 4: BATCHED_ROTARY
// Per-head, per-position rotation
// Each thread handles one (cos, sin) pair
static const ex_factor_tune_t EX_TUNE_BATCHED_ROTARY = {
    .threads_per_block = 512,
    .items_per_thread  = 2,    // 2 elements per rotation pair
    .vec_size          = 2,
    .smem_bytes        = 0,
    .num_warps         = 16,
    .num_stages        = 1,
    .grid_scale        = 1,
};

// Factor 5: GDN_CHUNK_FWD
// Chunked prefill: intra-chunk QK^T + inter-chunk state update
// Critical: fp32 accumulation to prevent NaN
// chunk_size=64 (from fla upstream), head_dim=128
// Each CTA processes one (batch, head, chunk)
static const ex_factor_tune_t EX_TUNE_GDN_CHUNK_FWD = {
    .threads_per_block = 128,
    .items_per_thread  = 4,
    .vec_size          = 4,
    .smem_bytes        = 32768, // Q,K,V tiles for chunk_size=64, dim=128, fp16
    .num_warps         = 4,
    .num_stages        = 2,
    .grid_scale        = 1,
};

// Factor 6: GDN_RECURRENT
// Single-step decode: conv1d_state update + delta_rule recurrence
// Lightweight: one token per step
static const ex_factor_tune_t EX_TUNE_GDN_RECURRENT = {
    .threads_per_block = 128,
    .items_per_thread  = 4,
    .vec_size          = 4,
    .smem_bytes        = 8192,  // conv state + temporal state
    .num_warps         = 4,
    .num_stages        = 1,
    .grid_scale        = 1,
};

// Factor 7: CACHE_APPEND (reshape_and_cache for paged KV)
// Each thread handles one token's K or V slice
static const ex_factor_tune_t EX_TUNE_CACHE_APPEND = {
    .threads_per_block = 256,
    .items_per_thread  = 4,
    .vec_size          = 4,
    .smem_bytes        = 0,
    .num_warps         = 8,
    .num_stages        = 1,
    .grid_scale        = 1,
};

// Factor 8: RESHAPE_CACHE_FLASH
// Optimized cache write for flash attention layout
static const ex_factor_tune_t EX_TUNE_RESHAPE_CACHE_FLASH = {
    .threads_per_block = 256,
    .items_per_thread  = 4,
    .vec_size          = 4,
    .smem_bytes        = 0,
    .num_warps         = 8,
    .num_stages        = 1,
    .grid_scale        = 1,
};

// =========================================================================
// Tuning table array (indexed by factor_id)
// =========================================================================
static const ex_factor_tune_t* const EX_TUNE_TABLE[] = {
    &EX_TUNE_MOE_TOPK_SOFTMAX,   // 0
    &EX_TUNE_MOE_ALIGN_BLOCK,    // 1
    &EX_TUNE_MOE_FUSED_GEMM,     // 2
    &EX_TUNE_GELU_TANH_MUL,      // 3
    &EX_TUNE_BATCHED_ROTARY,     // 4
    &EX_TUNE_GDN_CHUNK_FWD,      // 5
    &EX_TUNE_GDN_RECURRENT,      // 6
    &EX_TUNE_CACHE_APPEND,        // 7
    &EX_TUNE_RESHAPE_CACHE_FLASH, // 8
};

#ifdef __cplusplus
}
#endif

#endif // EX_FACTORS_HW_CONFIG_H
