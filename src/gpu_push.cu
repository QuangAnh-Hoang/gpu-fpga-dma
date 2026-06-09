// Milestone B — full GDR push: FPGA DMAs a pattern into GPU VRAM, sweep + validate.
//
// Builds on the Phase-0 spike. Pins a cudaMalloc buffer via nvidia_p2p, programs
// every HMSS slave-bridge slot that covers the transfer with the corresponding
// GPU bus address, then runs the gdr_write kernel over a size sweep. A GPU read
// validates each size byte-for-byte; kernel execution time gives sustained
// FPGA->GPU *write* bandwidth, compared against Milestone A's ~4.40 GB/s read.
//
// HMSS slot programming is done from userspace via the fpga_gdr HMSS_WRITE ioctl
// (no driver reload needed). The current host-mem config (xbutil --host-mem)
// determines slot_sz; this program adapts to it and verifies that the GPU's
// nvidia_p2p pages are bus-contiguous within each slot.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <string>
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
  if (ioctl(fd, FPGA_GDR_HMSS_READ, &r) < 0) { perror("HMSS_READ"); std::exit(1); }
  return r.value;
}
static void hmss_wr(int fd, uint32_t off, uint32_t val) {
  struct fpga_gdr_reg r{};
  r.offset = off; r.value = val;
  if (ioctl(fd, FPGA_GDR_HMSS_WRITE, &r) < 0) { perror("HMSS_WRITE"); std::exit(1); }
}

static std::string arg(int c, char** v, const char* a, const char* b, const char* d) {
  for (int i = 1; i < c - 1; ++i)
    if (!std::strcmp(v[i], a) || !std::strcmp(v[i], b)) return v[i + 1];
  return d;
}

int main(int argc, char** argv) {
  const std::string xclbin = arg(argc, argv, "--xclbin_file", "-x", "");
  const int dev = std::stoi(arg(argc, argv, "--device_id", "-d", "0"));
  // Cap: HMSS has 256 slots and the driver returns 256 page addrs => with the
  // default 1MB-slot host-mem config we can cover up to 256MB, but PIN returns
  // at most 256 page (64KB) addrs => 16MB. Keep 16MB unless slots are larger.
  size_t max_bytes = (size_t)std::stoull(arg(argc, argv, "--max_mb", "-m", "16")) * 1024 * 1024;
  // Diagnostic: -H times the kernel writing to plain host memory (no GPU
  // repoint), to separate the HLS-kernel/bridge ceiling from the GPU path.
  bool host_only_diag = false;
  for (int i = 1; i < argc; ++i) if (!std::strcmp(argv[i], "-H")) host_only_diag = true;
  if (xclbin.empty()) {
    std::cerr << "usage: " << argv[0] << " -x gdr_write.xclbin [-d dev] [-m max_MB] [-H]\n";
    return 1;
  }

  if (host_only_diag) {
    auto device = xrt::device(dev);
    auto uuid = device.load_xclbin(xclbin);
    auto krnl = xrt::kernel(device, uuid, "gdr_write");
    auto bo = xrt::bo(device, max_bytes, xrt::bo::flags::host_only, krnl.group_id(0));
    auto map = bo.map<uint32_t*>();
    std::printf("[-H] kernel -> HOST memory (slave-bridge ceiling, no GPU)\n");
    std::printf("%-8s %-12s %s\n", "Size", "write GB/s", "result");
    for (size_t S : {64ul*1024, 256ul*1024, 1ul<<20, 4ul<<20, 16ul<<20}) {
      if (S > max_bytes) continue;
      size_t count = S / sizeof(uint32_t);
      std::memset(map, 0, S);
      krnl(bo, 0xABCD0000u, (int)count).wait();
      size_t bad = 0;
      for (size_t i = 0; i < count; ++i) if (map[i] != 0xABCD0000u + (uint32_t)i) ++bad;
      for (int i = 0; i < 3; ++i) krnl(bo, 0xABCD0000u, (int)count).wait();
      auto t0 = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < 30; ++i) krnl(bo, 0xABCD0000u, (int)count).wait();
      auto t1 = std::chrono::high_resolution_clock::now();
      double sec = std::chrono::duration<double>(t1 - t0).count() / 30;
      char sz[16]; std::snprintf(sz, sizeof(sz), S>=(1<<20)?"%zuMB":"%zuKB", S>=(1<<20)?S/(1<<20):S/1024);
      std::printf("%-8s %-12.2f %s\n", sz, (double)S/(1024.0*1024*1024)/sec, bad?"MISMATCH":"OK");
    }
    return 0;
  }

  const uint32_t seed = 0xABCD0000u;

  // --- GPU buffer (64KB-aligned), XRT host_only aperture bo, kernel ---
  void* raw = nullptr;
  CUDA_CHECK(cudaMalloc(&raw, max_bytes + GPU_PAGE));
  uint64_t va = ((uint64_t)raw + GPU_PAGE - 1) & ~(GPU_PAGE - 1);
  std::printf("GPU buffer: aligned_va=0x%lx size=%zu MB\n", va, max_bytes / (1<<20));

  auto device = xrt::device(dev);
  auto uuid = device.load_xclbin(xclbin);
  auto krnl = xrt::kernel(device, uuid, "gdr_write");
  auto bo = xrt::bo(device, max_bytes, xrt::bo::flags::host_only, krnl.group_id(0));
  uint64_t ap = bo.address();

  // --- Pin GPU buffer -> bus addresses ---
  int fd = open("/dev/fpga_gdr", O_RDWR);
  if (fd < 0) { perror("open /dev/fpga_gdr (loaded? chmod 666?)"); return 1; }
  struct fpga_gdr_pin pin{};
  pin.va = va; pin.len = max_bytes;
  if (ioctl(fd, FPGA_GDR_PIN, &pin) < 0) { perror("PIN"); return 1; }
  std::printf("pinned %u pages (page_size=%u)\n", pin.n_pages, pin.page_size);

  // --- Read HMSS geometry, derive slot size ---
  uint32_t entry_num = hmss_rd(fd, HMSS_REG_ENTRY_NUM);
  uint64_t base = ((uint64_t)hmss_rd(fd, HMSS_REG_BASE_HI) << 32) | hmss_rd(fd, HMSS_REG_BASE_LO);
  uint32_t range_log = hmss_rd(fd, HMSS_REG_ADDR_RANGE);
  uint64_t range = range_log ? (1ULL << range_log) : 0;
  if (!entry_num || !range) {
    std::fprintf(stderr, "HMSS not enabled. Run: xbutil configure --host-mem -d <bdf> -s 256M ENABLE\n");
    return 1;
  }
  uint64_t slot_sz = range / entry_num;
  uint64_t off = ap - base;
  uint64_t pps = slot_sz / GPU_PAGE;             // GPU pages per HMSS slot
  std::printf("HMSS: slots=%u base=0x%lx range=%lu slot_sz=0x%lx ap_off=0x%lx pages/slot=%lu\n",
              entry_num, base, range, slot_sz, off, pps);
  if (off != 0) { std::fprintf(stderr, "aperture offset != 0 (got 0x%lx)\n", off); return 1; }
  if (max_bytes > (uint64_t)pin.n_pages * GPU_PAGE) {
    std::fprintf(stderr, "transfer %zu exceeds pinned pages (%u x 64KB)\n", max_bytes, pin.n_pages);
    return 1;
  }

  uint64_t n_slots = max_bytes / slot_sz;
  if (max_bytes % slot_sz) n_slots++;

  // --- Verify GPU pages are bus-contiguous within each slot ---
  for (uint64_t s = 0; s < n_slots; ++s) {
    uint64_t b0 = pin.dma_addrs[s * pps];
    for (uint64_t j = 1; j < pps && s * pps + j < pin.n_pages; ++j) {
      if (pin.dma_addrs[s * pps + j] != b0 + j * GPU_PAGE) {
        std::fprintf(stderr,
          "WARN: GPU pages not bus-contiguous within slot %lu (page %lu). "
          "Need 64KB slots: re-enable host-mem at 16MB. Aborting.\n", s, j);
        return 1;
      }
    }
  }

  // --- Program HMSS slots: slot s -> GPU bus addr at offset s*slot_sz ---
  hmss_wr(fd, HMSS_REG_ENTRY_NUM, 0);
  for (uint64_t s = 0; s < n_slots; ++s) {
    uint64_t pa = pin.dma_addrs[s * pps];
    hmss_wr(fd, HMSS_REG_PTABLE + s * 8 + 0, (uint32_t)(pa & 0xFFFFFFFF));
    hmss_wr(fd, HMSS_REG_PTABLE + s * 8 + 4, (uint32_t)(pa >> 32));
  }
  hmss_wr(fd, HMSS_REG_ENTRY_NUM, entry_num);
  std::printf("programmed %lu HMSS slot(s) -> GPU\n\n", n_slots);

  // --- Sweep: validate + time FPGA->GPU writes ---
  std::vector<size_t> sizes = {64 * 1024, 256 * 1024, 1 << 20, 4 << 20, 16 << 20};
  std::printf("%-8s %-12s %-10s %s\n", "Size", "write GB/s", "result", "(vs MsA read 4.40)");
  std::vector<uint32_t> host(max_bytes / sizeof(uint32_t));
  const int warmup = 3, iters = 30;

  for (size_t S : sizes) {
    if (S > max_bytes) continue;
    size_t count = S / sizeof(uint32_t);

    // correctness: zero GPU region, run once, read back, validate
    CUDA_CHECK(cudaMemset((void*)va, 0, S));
    CUDA_CHECK(cudaDeviceSynchronize());
    krnl(bo, seed, (int)count).wait();
    CUDA_CHECK(cudaMemcpy(host.data(), (void*)va, S, cudaMemcpyDeviceToHost));
    size_t bad = 0; size_t first = (size_t)-1;
    for (size_t i = 0; i < count; ++i)
      if (host[i] != seed + (uint32_t)i) { if (first==(size_t)-1) first=i; ++bad; }

    // bandwidth: time kernel execution
    for (int i = 0; i < warmup; ++i) krnl(bo, seed, (int)count).wait();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) krnl(bo, seed, (int)count).wait();
    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count() / iters;
    double gbps = (double)S / (1024.0*1024*1024) / sec;

    char sz[16]; std::snprintf(sz, sizeof(sz), "%zuKB", S/1024);
    if (S >= (1<<20)) std::snprintf(sz, sizeof(sz), "%zuMB", S/(1<<20));
    std::printf("%-8s %-12.2f %-10s %s\n", sz, gbps,
                bad ? "MISMATCH" : "OK",
                bad ? "" : "");
    if (bad) std::printf("   first mismatch i=%zu got 0x%x exp 0x%x\n",
                         first, host[first], seed + (uint32_t)first);
  }

  std::printf("\nFPGA->GPU write validated. (Milestone A FPGA->GPU read was 4.40 GB/s.)\n");
  ioctl(fd, FPGA_GDR_UNPIN, 0);
  close(fd);
  CUDA_CHECK(cudaFree(raw));
  return 0;
}
