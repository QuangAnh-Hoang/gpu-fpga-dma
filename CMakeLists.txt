cmake_minimum_required(VERSION 3.16)
project(gpu_fpga_dma LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# nvcc 12.x defaults to C++17 for device code; CMake 3.16 rejects CUDA_STANDARD 17.

# A6000 is Ampere (sm_86). CMake 3.16 predates CMAKE_CUDA_ARCHITECTURES, so set
# the gencode flag directly. Override CUDA_ARCH on the command line if needed.
set(CUDA_ARCH "86" CACHE STRING "CUDA compute capability (e.g. 86 for A6000)")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}")

# ---- XRT ----
# Honor XILINX_XRT if set, else default install location.
if(DEFINED ENV{XILINX_XRT})
  set(XRT_ROOT $ENV{XILINX_XRT})
else()
  set(XRT_ROOT /opt/xilinx/xrt)
endif()
message(STATUS "Using XRT at ${XRT_ROOT}")

add_executable(gpu_pull src/gpu_pull.cu)

# Milestone B: FPGA DMAs into GPU VRAM (Phase-0 spike + full push).
add_executable(gpu_push_spike src/gpu_push_spike.cu)
add_executable(gpu_push src/gpu_push.cu)

foreach(t gpu_pull gpu_push_spike gpu_push)
  target_include_directories(${t} PRIVATE ${XRT_ROOT}/include)
  target_link_directories(${t} PRIVATE ${XRT_ROOT}/lib)
  target_link_libraries(${t} PRIVATE xrt_coreutil uuid pthread)
endforeach()

# The CUDA language links the CUDA runtime (cudart) automatically.
