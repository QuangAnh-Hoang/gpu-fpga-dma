// Milestone C Phase 0 — measure the GPU->FPGA request path (deal-breaker 2).
//
// The control plane needs the GPU to write ID lists + doorbells into FPGA HBM
// through the P2P BAR (Milestone A mapping, cudaHostRegisterDefault). Reads
// were proven at 4.40 GB/s; writes were never measured. This benchmark:
//   1. maps an FPGA HBM P2P bo and registers it with CUDA,
//   2. has a GPU kernel write patterns of 4 KB .. 4 MB into it (timed),
//   3. validates the data actually landed in FPGA HBM (read back via the map),
//   4. also measures host-CPU memcpy into the BAR (the fallback submission path).
//
// Gate: >= ~0.5 GB/s and correct => GPU-driven submission is viable.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

#include <cuda_runtime.h>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA error %s at %d: %s\n", cudaGetErrorName(_e),  \
                   __LINE__, cudaGetErrorString(_e));                          \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

__global__ void bar_write(volatile uint32_t* dst, uint32_t base, size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) dst[i] = base + (uint32_t)i;
  __threadfence_system();
}

int main(int argc, char** argv) {
  std::string xclbin =
      "/home/hoangqa/workspace/Vitis_Accel_Examples/hello_world/"
      "build_dir.hw.xilinx_u55c_gen3x16_xdma_3_202210_1/vadd.xclbin";
  if (argc > 2 && !std::strcmp(argv[1], "-x")) xclbin = argv[2];

  const size_t BYTES = 16u << 20;
  auto device = xrt::device(0);
  auto uuid = device.load_xclbin(xclbin);
  auto krnl = xrt::kernel(device, uuid, "vadd");
  auto bo = xrt::bo(device, BYTES, xrt::bo::flags::p2p, krnl.group_id(2));
  uint32_t* map = bo.map<uint32_t*>();

  CUDA_CHECK(cudaHostRegister(map, BYTES, cudaHostRegisterDefault));
  uint32_t* d_bar = nullptr;
  CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_bar, map, 0));
  std::printf("FPGA HBM BAR mapped, GPU device ptr %p\n", (void*)d_bar);

  cudaEvent_t a, b;
  CUDA_CHECK(cudaEventCreate(&a));
  CUDA_CHECK(cudaEventCreate(&b));

  std::printf("\n%-10s %-16s %-16s\n", "Size", "GPU->BAR GB/s", "CPU->BAR GB/s");
  bool all_ok = true;
  for (size_t sz : {4096ul, 65536ul, 1ul << 20, 4ul << 20}) {
    size_t n = sz / 4;
    uint32_t seed = 0xB0000000u + (uint32_t)sz;

    // correctness once
    bar_write<<<64, 256>>>(d_bar, seed, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    size_t bad = 0;
    for (size_t i = 0; i < n; i++)
      if (map[i] != seed + (uint32_t)i) ++bad;
    if (bad) all_ok = false;

    // GPU write bandwidth
    const int iters = 20;
    bar_write<<<64, 256>>>(d_bar, seed, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(a));
    for (int it = 0; it < iters; it++) bar_write<<<64, 256>>>(d_bar, seed, n);
    CUDA_CHECK(cudaEventRecord(b));
    CUDA_CHECK(cudaEventSynchronize(b));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    double gpu_gbps = (double)sz * iters / (1024.0 * 1024 * 1024) / (ms / 1000.0);

    // CPU write bandwidth (fallback path)
    std::vector<uint32_t> src(n, seed);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; it++) std::memcpy(map, src.data(), sz);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_gbps = (double)sz * iters / (1024.0 * 1024 * 1024) /
                      std::chrono::duration<double>(t1 - t0).count();

    char szs[16];
    std::snprintf(szs, sizeof(szs), sz >= (1ul << 20) ? "%zuMB" : "%zuKB",
                  sz >= (1ul << 20) ? sz >> 20 : sz >> 10);
    std::printf("%-10s %-16.2f %-16.2f %s\n", szs, gpu_gbps, cpu_gbps,
                bad ? "VALIDATION FAILED" : "ok");
  }

  std::printf("\n%s\n", all_ok ? "G0 RESULT: GPU->BAR writes CORRECT"
                               : "G0 RESULT: GPU->BAR writes BROKEN");
  CUDA_CHECK(cudaHostUnregister(map));
  return all_ok ? 0 : 1;
}
