// muh/test/compile_test.cpp — Compile-time verification of muh tuning headers
//
// Build: g++ -std=c++17 -I muh/include muh/test/compile_test.cpp -o muh_test
// Run:   ./muh_test

#include "muh/muh.cuh"

#include <cassert>
#include <cstdio>

#define CHECK_NONZERO(expr, name) \
  do { auto _v = (expr); if (_v == 0) { std::fprintf(stderr, "FAIL: %s == 0\n", name); failures++; } else { passes++; } } while(0)

#define CHECK_EQ(expr, expected, name) \
  do { auto _v = (expr); if (_v != (expected)) { std::fprintf(stderr, "FAIL: %s == %d, expected %d\n", name, (int)_v, (int)(expected)); failures++; } else { passes++; } } while(0)

#define CHECK_TRUE(expr, name) \
  do { if (!(expr)) { std::fprintf(stderr, "FAIL: %s\n", name); failures++; } else { passes++; } } while(0)

int main() {
  using namespace muh::tuning; // bring enum values into scope
  int passes = 0;
  int failures = 0;

  auto hw = muh::target_hw;

  // --- Hardware descriptor ---
  CHECK_TRUE(hw.vendor == muh::hardware_capability::vendor_t::iluvatar, "target_hw.vendor");
  CHECK_NONZERO(hw.warp_size, "target_hw.warp_size");

  // --- reduce: default (run_to_run) ---
  {
    using namespace muh::tuning::reduce;
    auto ps = policy_selector{
      .accum_t = muh::tuning::type_t::float32,
      .operation_t = muh::tuning::op_kind_t::plus,
      .offset_size = 4, .accum_size = 4,
    };
    auto p = ps(hw);
    CHECK_EQ(p.multi_tile.threads_per_block, 512, "reduce.f32.threads");
    CHECK_EQ(p.multi_tile.vec_size, 2, "reduce.f32.vec_size");
    CHECK_EQ(p.multi_tile.reduce_algorithm, BLOCK_REDUCE_WARP_REDUCTIONS, "reduce.f32.algo");
  }

  // --- reduce: deterministic (gpu_to_gpu) ---
  {
    using namespace muh::tuning::reduce;
    auto ps = policy_selector{
      .accum_t = muh::tuning::type_t::float32,
      .operation_t = muh::tuning::op_kind_t::plus,
      .offset_size = 4, .accum_size = 4,
      .determinism = determinism_t::gpu_to_gpu,
    };
    auto p = ps(hw);
    CHECK_EQ(p.multi_tile.reduce_algorithm, BLOCK_REDUCE_RAKING, "reduce.det.algo=RAKING");
    CHECK_EQ(p.multi_tile.vec_size, 1, "reduce.det.vec_size=1");
    CHECK_EQ(p.multi_tile.load_modifier, LOAD_DEFAULT, "reduce.det.load=DEFAULT");
  }

  // --- reduce: nondeterministic ---
  {
    using namespace muh::tuning::reduce;
    auto ps = policy_selector{
      .accum_t = muh::tuning::type_t::float32,
      .operation_t = muh::tuning::op_kind_t::plus,
      .offset_size = 4, .accum_size = 4,
      .determinism = determinism_t::not_guaranteed,
    };
    auto p = ps(hw);
    CHECK_EQ(p.multi_tile.reduce_algorithm, BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC,
             "reduce.nondet.algo=NONDETERMINISTIC");
  }

  // --- topk: verify VECTORIZE and correct bits_per_pass ---
  {
    using namespace muh::tuning::topk;
    // 2-byte keys (fp16 logits — LLM hot path)
    auto p2 = policy_selector{.key_size = 2}(hw);
    CHECK_EQ(p2.load_algorithm, BLOCK_LOAD_VECTORIZE, "topk.2B.load=VECTORIZE");
    CHECK_EQ(p2.bits_per_pass, 11, "topk.2B.bits=11");  // CCCL: case 2 → 11
    CHECK_EQ(p2.items_per_thread, 8, "topk.2B.items=8"); // 4*4/2=8
    CHECK_EQ(p2.threads_per_block, 512, "topk.2B.threads=512");

    // 4-byte keys
    auto p4 = policy_selector{.key_size = 4}(hw);
    CHECK_EQ(p4.bits_per_pass, 11, "topk.4B.bits=11");
    CHECK_EQ(p4.items_per_thread, 4, "topk.4B.items=4"); // 4*4/4=4

    // 1-byte keys
    auto p1 = policy_selector{.key_size = 1}(hw);
    CHECK_EQ(p1.bits_per_pass, 8, "topk.1B.bits=8");
    CHECK_EQ(p1.items_per_thread, 16, "topk.1B.items=16"); // 4*4/1=16
  }

  // --- scan: lookback + lookahead ---
  {
    using namespace muh::tuning::scan;
    auto ps = policy_selector{
      .input_value_size = 4, .accum_size = 4, .offset_size = 4,
      .input_type = muh::tuning::type_t::float32,
      .accum_type = muh::tuning::type_t::float32,
      .operation_t = muh::tuning::op_kind_t::plus,
      .is_primitive_accum = true,
    };
    auto p = ps(hw);
    CHECK_EQ(p.lookback.threads_per_block, 384, "scan.f32.lookback.threads=384");
    CHECK_EQ(p.lookback.items_per_thread, 22, "scan.f32.lookback.items=22");
    CHECK_NONZERO(p.lookahead.reduce_and_scan_warps, "scan.f32.lookahead.warps");

    // 8-byte scan: was SMEM overflow with SM100 values (416*23*8=76544 > 49152)
    auto ps8 = policy_selector{
      .input_value_size = 8, .accum_size = 8, .offset_size = 4,
      .input_type = type_t::int64, .accum_type = type_t::int64,
      .operation_t = op_kind_t::plus, .is_primitive_accum = true,
    };
    auto p8 = ps8(hw);
    CHECK_EQ(p8.lookback.items_per_thread, 14, "scan.8B.items=14(derived)");
    CHECK_TRUE(p8.lookback.threads_per_block * p8.lookback.items_per_thread * 8 <= 49152,
               "scan.8B.tile_fits_48KB_smem");
  }

  // --- batch_memcpy: two-tier ---
  {
    using namespace muh::tuning::batch_memcpy;
    auto p = policy_selector{}(hw);
    CHECK_EQ(p.small_buffer.threads_per_block, 128, "batch_memcpy.small.threads=128");
    CHECK_EQ(p.small_buffer.buffers_per_thread, 4, "batch_memcpy.small.bufs=4");
    CHECK_EQ(p.small_buffer.warp_level_threshold, 128, "batch_memcpy.small.warp_thresh=128");
    CHECK_EQ(p.small_buffer.block_level_threshold, 8192, "batch_memcpy.small.block_thresh=8192");
    CHECK_EQ(p.large_buffer.threads_per_block, 256, "batch_memcpy.large.threads=256");
    CHECK_EQ(p.large_buffer.bytes_per_thread, 32, "batch_memcpy.large.bytes=32");
  }

  // --- transform: three-policy ---
  {
    using namespace muh::tuning::transform;
    auto ps = policy_selector{
      .min_elem_size = 2, .max_elem_size = 2, .num_inputs = 1,
      .all_contiguous = true, .all_trivially_relocatable = true,
      .requires_stable_address = false,
    };
    auto p = ps(hw);
    CHECK_NONZERO(p.vectorized.threads_per_block, "transform.vectorized.threads");
    CHECK_NONZERO(p.vectorized.vec_size, "transform.vectorized.vec_size");
    CHECK_NONZERO(p.async_copy.threads_per_block, "transform.async_copy.threads");
    CHECK_EQ(p.prefetch.threads_per_block, 256, "transform.prefetch.threads=256");
    CHECK_EQ(p.fill.threads_per_block, 256, "transform.fill.threads=256");
  }

  // --- for_each ---
  {
    using namespace muh::tuning::for_each;
    auto p = policy_selector{}(hw);
    CHECK_EQ(p.threads_per_block, 256, "for.threads=256");
    CHECK_EQ(p.items_per_thread, 4, "for.items=4");
  }

  // --- Report ---
  std::printf("\nmuh compile test: %d passed, %d failed\n", passes, failures);
  return failures > 0 ? 1 : 0;
}
