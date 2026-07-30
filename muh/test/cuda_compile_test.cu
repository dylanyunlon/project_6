// muh/test/cuda_compile_test.cu — CUDA compilation test
//
// Verifies muh headers compile under nvcc/clang CUDA mode.
// Does NOT require GPU execution — just compilation.

#include "muh/muh.cuh"

__global__ void dummy_kernel() {
  // Instantiate policy selectors in device code to verify
  // all constexpr paths compile on the device side
  auto hw = muh::hardware_capability::bi_v100();

  // Reduce
  auto rp = muh::tuning::reduce::policy_selector{
    .accum_t = muh::tuning::type_t::float32,
    .operation_t = muh::tuning::op_kind_t::plus,
    .offset_size = 4,
    .accum_size = 4,
  }(hw);
  (void)rp;

  // Topk
  auto tp = muh::tuning::topk::policy_selector{.key_size = 2}(hw);
  (void)tp;
}

int main() {
  // Host-side test (same as compile_test.cpp core)
  auto hw = muh::target_hw;

  auto reduce_policy = muh::tuning::reduce::policy_selector{
    .accum_t = muh::tuning::type_t::float32,
    .operation_t = muh::tuning::op_kind_t::plus,
    .offset_size = 4,
    .accum_size = 4,
  }(hw);

  printf("CUDA compile test passed: reduce.threads=%d\n",
         reduce_policy.multi_tile.threads_per_block);
  return 0;
}
