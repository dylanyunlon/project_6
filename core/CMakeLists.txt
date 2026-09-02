# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Commit: 494f293b5629 · feat · PR #2260  (adapted for Iluvatar BI-V100)

cmake_minimum_required(VERSION 3.18)

# ---------- layerwise split KV cache library ----------
add_library(layerwise_split_kv STATIC
  framework/kv_cache/kv_cache_layerwise.cpp
  framework/kv_cache/kv_cache_estimation_layerwise.cpp
  framework/parallel_state/mapping_ilu.cpp
  distributed_runtime/layerwise_split_engine_ext.cpp
  distributed_runtime/layerwise_split_master.cpp
  runtime/worker_layerwise_init.cpp
  config/parallel_config_layerwise.cpp
)

target_include_directories(layerwise_split_kv PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(layerwise_split_kv PUBLIC
  gflags
  glog::glog
  torch
)

# Iluvatar BI-V100 build: define USE_ILU
if(USE_ILU)
  target_compile_definitions(layerwise_split_kv PUBLIC USE_ILU)
endif()

# ---------- tests ----------
if(BUILD_TESTING)
  add_executable(test_layerwise_split
    ${CMAKE_CURRENT_SOURCE_DIR}/../tests/core/test_layerwise_split_kv_cache.cpp
  )
  target_link_libraries(test_layerwise_split PRIVATE
    layerwise_split_kv
    GTest::gtest_main
    gflags
    glog::glog
  )
  add_test(NAME LayerwiseSplitKVTests COMMAND test_layerwise_split)
endif()
