// Milestone B — Phase 0 spike: prove the FPGA can DMA into GPU VRAM.
//
// Flow:
//   1. cudaMalloc a 64KB-aligned GPU buffer, zero it.
//   2. Create a host_only XRT bo (size N) so XRT binds the kernel's AXI-master
//      pointer arg to the HMSS aperture and reserves a translation slot.
//   3. Pin the GPU buffer via /dev/fpga_gdr (nvidia_p2p) -> GPU bus address.
//   4. Re-point the HMSS slot covering the bo's aperture offset at that GPU bus
//      address (raw register writes through the driver).
//   5. Run gdr_write -> kernel writes out[i]=seed+i through the aperture, which
//      now routes to GPU VRAM.
//   6. Copy the GPU buffer back and validate byte-for-byte.
//
// This is intentionally verbose: it prints every address/register so the novel
// HMSS-to-GPU step can be debugged on first contact with hardware.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>

#include <cuda_runtime.h>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "../driver/fpga_gdr.h"

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t _e = (call);                                                    \
    if (_e != cudaSuccess) {                                                    \
      std::cerr << "CUDA error " << cudaGetErrorName(_e) << " at " << __LINE__  \
                << ": " << cudaGetErrorString(_e) << "\n";                      \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

static const uint64_t GPU_PAGE = 64 * 1024;

static uint32_t hmss_rd(int fd, uint32_t off) {
  struct fpga_gdr_reg r{};
  r.offset = off;
  if (ioctl(fd, FPGA_GDR_HMSS_READ, &r) < 0) {
    perror("HMSS_READ");
    std::exit(EXIT_FAILURE);
  }
  return r.value;
}
static void hmss_wr(int fd, uint32_t off, uint32_t val) {
  struct fpga_gdr_reg r{};
  r.offset = off;
  r.value = val;
  if (ioctl(fd, FPGA_GDR_HMSS_WRITE, &r) < 0) {
    perror("HMSS_WRITE");
    std::exit(EXIT_FAILURE);
  }
}

static std::string arg(int c, char** v, const char* a, const char* b,
                       const char* d) {
  for (int i = 1; i < c - 1; ++i)
    if (!std::strcmp(v[i], a) || !std::strcmp(v[i], b)) return v[i + 1];
  return d;
}

int main(int argc, char** argv) {
  const std::string xclbin = arg(argc, argv, "--xclbin_file", "-x", "");
  const int dev = std::stoi(arg(argc, argv, "--device_id", "-d", "0"));
  const uint32_t seed = 0xC0DE0000u;
  const size_t N = GPU_PAGE;                 // 64 KB spike transfer
  const size_t count = N / sizeof(uint32_t);
  if (xclbin.empty()) {
    std::cerr << "usage: " << argv[0] << " -x gdr_write.xclbin [-d dev]\n";
    return 1;
  }

  // 1) GPU buffer, 64KB-aligned, zeroed.
  void* raw = nullptr;
  CUDA_CHECK(cudaMalloc(&raw, N + GPU_PAGE));
  uint64_t va = ((uint64_t)raw + GPU_PAGE - 1) & ~(GPU_PAGE - 1);
  CUDA_CHECK(cudaMemset((void*)va, 0, N));
  CUDA_CHECK(cudaDeviceSynchronize());
  std::printf("GPU buffer: raw=%p aligned_va=0x%lx size=%zu\n", raw, va, N);

  // 2) XRT host_only bo + kernel.
  auto device = xrt::device(dev);
  auto uuid = device.load_xclbin(xclbin);
  auto krnl = xrt::kernel(device, uuid, "gdr_write");
  auto bo = xrt::bo(device, N, xrt::bo::flags::host_only, krnl.group_id(0));
  uint64_t ap = bo.address();
  std::printf("host_only bo aperture device addr = 0x%lx\n", ap);

  // 3) Pin the GPU buffer -> bus address.
  int fd = open("/dev/fpga_gdr", O_RDWR);
  if (fd < 0) { perror("open /dev/fpga_gdr"); return 1; }
  struct fpga_gdr_pin pin{};
  pin.va = va;
  pin.len = N;
  if (ioctl(fd, FPGA_GDR_PIN, &pin) < 0) { perror("PIN"); return 1; }
  std::printf("pinned: n_pages=%u page_size=%u gpu_bus[0]=0x%lx\n",
              pin.n_pages, pin.page_size, (unsigned long)pin.dma_addrs[0]);

  // 4) Inspect HMSS, compute the slot for this bo, re-point it at the GPU.
  uint32_t entry_num = hmss_rd(fd, HMSS_REG_ENTRY_NUM);
  uint32_t base_lo = hmss_rd(fd, HMSS_REG_BASE_LO);
  uint32_t base_hi = hmss_rd(fd, HMSS_REG_BASE_HI);
  uint32_t range_log = hmss_rd(fd, HMSS_REG_ADDR_RANGE);
  uint64_t base = ((uint64_t)base_hi << 32) | base_lo;
  uint64_t range = range_log ? (1ULL << range_log) : 0;
  std::printf("HMSS: entry_num=%u base=0x%lx range=2^%u(%lu) \n", entry_num,
              base, range_log, range);
  if (entry_num == 0 || range == 0) {
    std::fprintf(stderr,
                 "HMSS not programmed. Run: xbutil configure --host-mem "
                 "--size <MB> first.\n");
    return 1;
  }
  uint64_t slot_sz = range / entry_num;
  uint64_t off = ap - base;
  uint32_t slot = (uint32_t)(off / slot_sz);
  std::printf("aperture offset=0x%lx slot_sz=0x%lx -> slot=%u\n", off, slot_sz,
              slot);

  // Disable, write the GPU bus addr into this slot, re-enable.
  uint64_t gpu_bus = pin.dma_addrs[0];
  hmss_wr(fd, HMSS_REG_ENTRY_NUM, 0);
  hmss_wr(fd, HMSS_REG_PTABLE + slot * 8 + 0, (uint32_t)(gpu_bus & 0xFFFFFFFF));
  hmss_wr(fd, HMSS_REG_PTABLE + slot * 8 + 4, (uint32_t)(gpu_bus >> 32));
  hmss_wr(fd, HMSS_REG_ENTRY_NUM, entry_num);
  std::printf("HMSS slot %u re-pointed at GPU bus 0x%lx; reading back: ", slot,
              gpu_bus);
  std::printf("lo=0x%x hi=0x%x\n", hmss_rd(fd, HMSS_REG_PTABLE + slot * 8),
              hmss_rd(fd, HMSS_REG_PTABLE + slot * 8 + 4));

  // 5) Run the FPGA kernel: writes out[i]=seed+i through the aperture -> GPU.
  std::printf("running gdr_write (seed=0x%x, count=%zu)...\n", seed, count);
  auto run = krnl(bo, seed, (int)count);
  run.wait();

  // 6) Validate from GPU memory.
  std::vector<uint32_t> host(count);
  CUDA_CHECK(cudaMemcpy(host.data(), (void*)va, N, cudaMemcpyDeviceToHost));
  size_t bad = 0, first = (size_t)-1;
  for (size_t i = 0; i < count; ++i) {
    uint32_t exp = seed + (uint32_t)i;
    if (host[i] != exp) {
      if (first == (size_t)-1) first = i;
      ++bad;
    }
  }
  std::printf("GPU[0..3] = 0x%x 0x%x 0x%x 0x%x\n", host[0], host[1], host[2],
              host[3]);
  if (bad == 0) {
    std::printf("SPIKE PASSED: FPGA wrote %zu bytes into GPU VRAM, validated.\n",
                N);
  } else {
    std::printf("SPIKE FAILED: %zu mismatches, first at i=%zu got 0x%x exp 0x%x\n",
                bad, first, host[first], seed + (uint32_t)first);
  }

  ioctl(fd, FPGA_GDR_UNPIN, 0);
  close(fd);
  CUDA_CHECK(cudaFree(raw));
  return bad ? 2 : 0;
}
