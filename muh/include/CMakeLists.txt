cmake_minimum_required(VERSION 3.18)
project(muh LANGUAGES CXX)

# muh is a header-only library
add_library(muh INTERFACE)
target_include_directories(muh INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(muh INTERFACE cxx_std_17)

# If CCCL is available, link it
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../cccl_upstream/cub/cub/config.cuh")
  target_include_directories(muh INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cccl_upstream/cub
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cccl_upstream/thrust
    ${CMAKE_CURRENT_SOURCE_DIR}/../../cccl_upstream/libcudacxx/include
  )
  target_compile_definitions(muh INTERFACE MUH_HAS_CCCL=1)
endif()

# Compile test — verifies all headers parse without errors
# This is a host-only test (no GPU needed)
option(MUH_BUILD_TESTS "Build muh compile tests" ON)

if(MUH_BUILD_TESTS)
  add_executable(muh_compile_test
    ${CMAKE_CURRENT_SOURCE_DIR}/../test/compile_test.cpp
  )
  target_link_libraries(muh_compile_test PRIVATE muh)

  # If we have a CUDA compiler, also test .cu compilation
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_executable(muh_cuda_compile_test
      ${CMAKE_CURRENT_SOURCE_DIR}/../test/cuda_compile_test.cu
    )
    target_link_libraries(muh_cuda_compile_test PRIVATE muh)
    set_target_properties(muh_cuda_compile_test PROPERTIES CUDA_STANDARD 17)
  endif()
endif()
