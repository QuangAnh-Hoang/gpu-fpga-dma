// Host driver for rd_probe: maps read-health of the HOST[0] slave bridge
// across window slots (GPU VRAM) and CMA host memory, for 32-bit, single-beat
// 512-bit, and 4-beat-burst read shapes.
//
// Pass A: ctrl/sq aimed at the VRAM window (slots 0..15 of the bridge).
// Pass B: ctrl/sq aimed at the second host_only bo (CMA, slots 16+).
// Expected marker at offset X: 0x51000000 | (X >> 20)  (i.e., slot index).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "../driver/fpga_gdr.h"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA error %s at %d\n", cudaGetErrorName(_e),      \
                   __LINE__);                                                  \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

static const uint64_t GPU_PAGE = 64 * 1024;
static const uint32_t WIN_BYTES = 16u << 20;

static uint32_t hmss_rd(int fd, uint32_t off) {
  struct fpga_gdr_reg r{};
  r.offset = off;
  if (ioctl(fd, FPGA_GDR_HMSS_READ, &r) < 0) { perror("rd"); exit(1); }
  return r.value;
}
static void hmss_wr(int fd, uint32_t off, uint32_t val) {
  struct fpga_gdr_reg r{};
  r.offset = off;
  r.value = val;
  if (ioctl(fd, FPGA_GDR_HMSS_WRITE, &r) < 0) { perror("wr"); exit(1); }
}

__global__ void fill_markers(uint32_t* win) {
  // marker word at every 64KB: 0x51000000 | (byte_off >> 20)
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t step = (size_t)gridDim.x * blockDim.x;
  size_t n = WIN_BYTES / (64 * 1024);
  for (; i < n; i += step) {
    size_t off = i * 64 * 1024;
    win[off / 4] = 0x51000000u | (uint32_t)(off >> 16);
  }
}

static void decode(const char* tag, uint32_t* r) {
  // r = 7 beats x 16 words copied from out[0..6]
  std::printf("%s boot=%08x end=%08x\n", tag, r[0], r[6 * 16]);
  for (int i = 0; i < 4; i++) {
    uint32_t* b = r + (1 + i) * 16;
    std::printf("  probe%d: stage=%08x w32=%08x b512=%08x burst=%08x\n", i,
                b[0], b[1], b[2], b[3]);
  }
}

int main(int argc, char** argv) {
  std::string xclbin = argc > 1 ? argv[1] : "hw/build_rd_probe/rd_probe.xclbin";
  auto device = xrt::device(0);
  auto uuid = device.load_xclbin(xclbin);
  auto krnl = xrt::kernel(device, uuid, "rd_probe");

  auto out_bo = xrt::bo(device, WIN_BYTES, xrt::bo::flags::host_only,
                        krnl.group_id(2));
  auto q_bo = xrt::bo(device, WIN_BYTES, xrt::bo::flags::host_only,
                      krnl.group_id(1));
  std::printf("out=0x%lx q=0x%lx\n", out_bo.address(), q_bo.address());

  // pinned VRAM window behind HMSS slots 0..15
  void* raw = nullptr;
  CUDA_CHECK(cudaMalloc(&raw, WIN_BYTES + GPU_PAGE));
  uint64_t win_va = ((uint64_t)raw + GPU_PAGE - 1) & ~(GPU_PAGE - 1);
  CUDA_CHECK(cudaMemset((void*)win_va, 0, WIN_BYTES));
  fill_markers<<<64, 256>>>((uint32_t*)win_va);
  CUDA_CHECK(cudaDeviceSynchronize());

  int fd = open("/dev/fpga_gdr", O_RDWR);
  if (fd < 0) { perror("open"); return 1; }
  struct fpga_gdr_pin pin{};
  pin.va = win_va;
  pin.len = WIN_BYTES;
  if (ioctl(fd, FPGA_GDR_PIN, &pin) < 0) { perror("PIN"); return 1; }
  uint32_t entry_num = hmss_rd(fd, HMSS_REG_ENTRY_NUM);
  uint32_t range_log = hmss_rd(fd, HMSS_REG_ADDR_RANGE);
  uint64_t slot_sz = (1ULL << range_log) / entry_num;
  uint64_t pps = slot_sz / GPU_PAGE;
  hmss_wr(fd, HMSS_REG_ENTRY_NUM, 0);
  for (uint64_t s = 0; s < WIN_BYTES / slot_sz; s++) {
    uint64_t pa = pin.dma_addrs[s * pps];
    hmss_wr(fd, HMSS_REG_PTABLE + s * 8 + 0, (uint32_t)pa);
    hmss_wr(fd, HMSS_REG_PTABLE + s * 8 + 4, (uint32_t)(pa >> 32));
  }
  hmss_wr(fd, HMSS_REG_ENTRY_NUM, entry_num);
  std::printf("HMSS: %lu slots of %lu KB -> VRAM\n", WIN_BYTES / slot_sz,
              slot_sz >> 10);

  // markers in CMA bo
  auto qmap = q_bo.map<uint32_t*>();
  std::memset(qmap, 0, WIN_BYTES);
  for (uint64_t off = 0; off < WIN_BYTES; off += 64 * 1024)
    qmap[off / 4] = 0x51000000u | (uint32_t)(off >> 16);
  __sync_synchronize();

  uint32_t res[7 * 16];

  // ---- pass A: VRAM window (slots 0,8,9,15) ----
  CUDA_CHECK(cudaMemset((void*)(win_va + (15u << 20) + 65536), 0, 7 * 64));
  uint64_t res_off = (15u << 20) + 65536;  // out results: slot15+64K
  {
    auto run = krnl(out_bo, out_bo,
                    xrt::bo(out_bo, 7 * 64, res_off),  // sub-bo for results
                    (uint64_t)(64 * 1024), (uint64_t)((8u << 20) + 65536),
                    (uint64_t)(9u << 20), (uint64_t)(14u << 20));
    auto st = run.wait(std::chrono::seconds(10));
    std::printf("passA state: %s\n",
                st == ERT_CMD_STATE_COMPLETED ? "COMPLETED" : "TIMEOUT");
    CUDA_CHECK(cudaMemcpy(res, (void*)(win_va + res_off), sizeof(res),
                          cudaMemcpyDeviceToHost));
    decode("A(VRAM)", res);
  }

  // ---- pass B: CMA bo (bridge slots 16,24,25,30 via q_bo offsets) ----
  {
    auto run = krnl(q_bo, q_bo, xrt::bo(out_bo, 7 * 64, res_off),
                    (uint64_t)(64 * 1024), (uint64_t)((8u << 20) + 65536),
                    (uint64_t)(9u << 20), (uint64_t)(14u << 20));
    auto st = run.wait(std::chrono::seconds(10));
    std::printf("passB state: %s\n",
                st == ERT_CMD_STATE_COMPLETED ? "COMPLETED" : "TIMEOUT");
    CUDA_CHECK(cudaMemcpy(res, (void*)(win_va + res_off), sizeof(res),
                          cudaMemcpyDeviceToHost));
    decode("B(CMA)", res);
  }

  // ---- pass C: STALENESS LAW test — change CMA markers, re-read SAME addrs.
  // If the kernel reports the OLD values, some element on the ULP read path
  // caches per-address and never sees host-path updates.
  for (uint64_t off = 0; off < WIN_BYTES; off += 64 * 1024)
    qmap[off / 4] = 0xC4C4E000u | (uint32_t)(off >> 16);  // new marker
  __sync_synchronize();
  usleep(100000);
  {
    auto run = krnl(q_bo, q_bo, xrt::bo(out_bo, 7 * 64, res_off),
                    (uint64_t)(64 * 1024), (uint64_t)((8u << 20) + 65536),
                    (uint64_t)(9u << 20), (uint64_t)(14u << 20));
    auto st = run.wait(std::chrono::seconds(10));
    std::printf("passC state: %s (markers changed to 0xC4C4Exxx)\n",
                st == ERT_CMD_STATE_COMPLETED ? "COMPLETED" : "TIMEOUT");
    CUDA_CHECK(cudaMemcpy(res, (void*)(win_va + res_off), sizeof(res),
                          cudaMemcpyDeviceToHost));
    decode("C(CMA again)", res);
    std::printf("verdict: %s\n",
                (res[16 + 1] >> 16) == 0xC4C4 ? "FRESH - no stale cache across runs"
                                              : "STALE - read cache confirmed");
  }

  ioctl(fd, FPGA_GDR_UNPIN, 0);
  close(fd);
  CUDA_CHECK(cudaFree(raw));
  return 0;
}
