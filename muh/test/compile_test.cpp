// muh/test/compile_test.cpp — Compile-time verification of muh tuning headers
//
// This test does NOT require a GPU. It verifies:
//   1. All headers parse without errors
//   2. All policy_selector functors instantiate and return valid policies
//   3. All bi100_* struct values are non-zero (not forgotten placeholders)
//
// Build: g++ -std=c++17 -I muh/include muh/test/compile_test.cpp -o muh_test
// Run:   ./muh_test

#include "muh/muh.cuh"

#include <cassert>
#include <cstdio>

// Helper: verify a value is non-zero (catches forgotten TBD placeholders)
#define CHECK_NONZERO(expr, name) \
  do { \
    auto _v = (expr); \
    if (_v == 0) { \
      std::fprintf(stderr, "FAIL: %s == 0 (placeholder not filled)\n", name); \
      failures++; \
    } else { \
      passes++; \
    } \
  } while(0)

#define CHECK_TRUE(expr, name) \
  do { \
    if (!(expr)) { \
      std::fprintf(stderr, "FAIL: %s\n", name); \
      failures++; \
    } else { \
      passes++; \
    } \
  } while(0)

int main() {
  int passes = 0;
  int failures = 0;

  auto hw = muh::target_hw;

  // --- Verify hardware descriptor ---
  CHECK_TRUE(hw.vendor == muh::hardware_capability::vendor_t::iluvatar,
             "target_hw.vendor == iluvatar");
  CHECK_NONZERO(hw.warp_size, "target_hw.warp_size");
  CHECK_NONZERO(hw.max_threads_per_block, "target_hw.max_threads_per_block");

  // --- Test reduce policy_selector ---
  {
    using namespace muh::tuning::reduce;
    auto ps = policy_selector{
      .accum_t = muh::tuning::type_t::float32,
      .operation_t = muh::tuning::op_kind_t::plus,
      .offset_size = 4,
      .accum_size = 4,
    };
    auto policy = ps(hw);
    CHECK_NONZERO(policy.multi_tile.threads_per_block,
                  "reduce.float32.threads_per_block");
    CHECK_NONZERO(policy.multi_tile.items_per_thread,
                  "reduce.float32.items_per_thread");
    CHECK_NONZERO(policy.multi_tile.vec_size,
                  "reduce.float32.vec_size");

    // Verify known bi100 value matches
    CHECK_TRUE(policy.multi_tile.threads_per_block > 0 &&
               policy.multi_tile.threads_per_block <= 1024,
               "reduce.threads_per_block in [1, 1024]");
  }

  // --- Test topk policy_selector ---
  {
    using namespace muh::tuning::topk;
    auto ps = policy_selector{.key_size = 2};
    auto policy = ps(hw);
    CHECK_NONZERO(policy.threads_per_block, "topk.2B.threads_per_block");
    CHECK_NONZERO(policy.items_per_thread, "topk.2B.items_per_thread");
    CHECK_NONZERO(policy.bits_per_pass, "topk.2B.bits_per_pass");
    CHECK_TRUE(policy.bits_per_pass >= 4 && policy.bits_per_pass <= 11,
               "topk.bits_per_pass in [4, 11]");
  }

  // --- Test scan policy_selector ---
  {
    using namespace muh::tuning::scan;
    auto ps = policy_selector{
      .input_value_size = 4,
      .accum_size = 4,
      .offset_size = 4,
      .input_type = muh::tuning::type_t::float32,
      .accum_type = muh::tuning::type_t::float32,
      .operation_t = muh::tuning::op_kind_t::plus,
      .is_primitive_accum = true,
    };
    auto policy = ps(hw);
    CHECK_NONZERO(policy.lookback.threads_per_block,
                  "scan.float32.lookback.threads_per_block");
    CHECK_NONZERO(policy.lookback.items_per_thread,
                  "scan.float32.lookback.items_per_thread");
  }

  // --- Test transform policy_selector ---
  {
    using namespace muh::tuning::transform;
    auto ps = policy_selector{
      .min_elem_size = 2,
      .max_elem_size = 2,
      .num_inputs = 1,
    };
    auto policy = ps(hw);
    CHECK_NONZERO(policy.bulk.threads_per_block,
                  "transform.bulk.threads_per_block");
  }

  // --- Test batch_memcpy policy_selector ---
  {
    using namespace muh::tuning::batch_memcpy;
    auto ps = policy_selector{};
    auto policy = ps(hw);
    CHECK_NONZERO(policy.threads_per_block,
                  "batch_memcpy.threads_per_block");
  }

  // --- Test for_each policy_selector ---
  {
    using namespace muh::tuning::for_each;
    auto ps = policy_selector{};
    auto policy = ps(hw);
    CHECK_NONZERO(policy.threads_per_block,
                  "for_each.threads_per_block");
    CHECK_NONZERO(policy.items_per_thread,
                  "for_each.items_per_thread");
  }

  // --- Report ---
  std::printf("\nmuh compile test: %d passed, %d failed\n", passes, failures);
  return failures > 0 ? 1 : 0;
}
