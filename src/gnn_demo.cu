// Milestone C — end-to-end GPU-request -> FPGA-response demo (round design).
//
// Emulated intelligent storage: the static node-feature table lives in FPGA
// HBM; requests are batches of node IDs written into a pinned GPU-VRAM window;
// one kernel launch gathers one batch and streams the packed rows back into
// the window's ring (Milestone B data plane, ~12.9 GB/s); a GPU consumer
// kernel polls the DONE record locally in VRAM and validates every word.
//
// Round design rationale: on this U55C shell, a running kernel's re-reads of
// any address return only the first-read value (host/GPU updates after start
// are never observed), so doorbell polling is impossible — instead the batch
// metadata travels as scalar args and the kernel is re-launched per round
// (~100 us, amortized). Established via rd_probe + prestart experiments.
//
// Layout/spec: src/queue_layout.h. Submission modes:
//   -S gpu (default: ids staged by GPU into the window) | host (cudaMemcpy)

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#include "../driver/fpga_gdr.h"
#include "queue_layout.h"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA error %s at %d: %s\n", cudaGetErrorName(_e),  \
                   __LINE__, cudaGetErrorString(_e));                          \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

static const uint64_t GPU_PAGE = 64 * 1024;

// Copy ids into the window's id region (device-local stores).
__global__ void stage_ids(uint32_t* win_ids, const uint32_t* ids, uint32_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += s) win_ids[i] = ids[i];
  __threadfence_system();
}

// One-block consumer: poll DONE (locally in VRAM), then validate the ring.
__global__ void consume(volatile uint32_t* win, const uint32_t* ids,
                        uint32_t batch_id, uint32_t n_ids, uint32_t row_beats,
                        int validate, unsigned long long* err_cnt) {
  volatile uint32_t* cq_done = win + GNN_CQ_DONE_OFF / 4;
  const uint32_t row_words = row_beats * 16;
  if (threadIdx.x == 0) {
    while (!(cq_done[0] == batch_id && cq_done[2] == GNN_DONE_MAGIC)) {
    }
  }
  __syncthreads();
  if (validate) {
    uint32_t total_words = n_ids * row_words;
    for (uint32_t w = threadIdx.x + blockIdx.x * blockDim.x; w < total_words;
         w += blockDim.x * gridDim.x) {
      uint32_t row = w / row_words;
      uint32_t d = w % row_words;
      if (win[w] != gnn_feat_word(ids[row], d)) atomicAdd(err_cnt, 1ull);
    }
  }
}

__global__ void clear_done(volatile uint32_t* win) {
  win[GNN_CQ_DONE_OFF / 4] = 0xFFFFFFFFu;
  win[GNN_CQ_DONE_OFF / 4 + 2] = 0;
  __threadfence_system();
}

// ---------------- host helpers ----------------

static uint32_t hmss_rd(int fd, uint32_t off) {
  struct fpga_gdr_reg r{};
  r.offset = off;
  if (ioctl(fd, FPGA_GDR_HMSS_READ, &r) < 0) { perror("HMSS_READ"); exit(1); }
  return r.value;
}
static void hmss_wr(int fd, uint32_t off, uint32_t val) {
  struct fpga_gdr_reg r{};
  r.offset = off;
  r.value = val;
  if (ioctl(fd, FPGA_GDR_HMSS_WRITE, &r) < 0) { perror("HMSS_WRITE"); exit(1); }
}

static std::string arg(int c, char** v, const char* a, const char* d) {
  for (int i = 1; i < c - 1; ++i)
    if (!std::strcmp(v[i], a)) return v[i + 1];
  return d;
}

int main(int argc, char** argv) {
  const std::string xclbin =
      arg(argc, argv, "-x", "hw/build_gnn_gather/gnn_gather.xclbin");
  const uint32_t n_nodes = std::stoul(arg(argc, argv, "-n", "2097152"));
  const uint32_t row_bytes = std::stoul(arg(argc, argv, "-r", "512"));
  const std::string submit_mode = arg(argc, argv, "-S", "gpu");
  const uint32_t row_beats = row_bytes / 64;
  const uint32_t row_words = row_bytes / 4;
  if (row_bytes % 64 || !row_beats) {
    std::fprintf(stderr, "row_bytes must be a multiple of 64\n");
    return 1;
  }
  std::printf("nodes=%u row=%uB (%u beats) submit=%s\n", n_nodes, row_bytes,
              row_beats, submit_mode.c_str());

  // ---- XRT: device, kernel, buffers ----
  auto device = xrt::device(0);
  auto uuid = device.load_xclbin(xclbin);
  auto krnl = xrt::kernel(device, uuid, "gnn_gather");

  auto feat_bo = xrt::bo(device, (uint64_t)n_nodes * row_bytes, krnl.group_id(1));
  auto out_bo = xrt::bo(device, GNN_WIN_BYTES, xrt::bo::flags::host_only,
                        krnl.group_id(2));

  std::printf("filling %.1f MB feature table...\n",
              (double)n_nodes * row_bytes / (1 << 20));
  auto fmap = feat_bo.map<uint32_t*>();
  for (uint32_t i = 0; i < n_nodes; i++)
    for (uint32_t d = 0; d < row_words; d++)
      fmap[(size_t)i * row_words + d] = gnn_feat_word(i, d);
  feat_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // ---- pinned GPU window + HMSS (programmed once) ----
  void* raw = nullptr;
  CUDA_CHECK(cudaMalloc(&raw, GNN_WIN_BYTES + GPU_PAGE));
  uint64_t win_va = ((uint64_t)raw + GPU_PAGE - 1) & ~(GPU_PAGE - 1);
  CUDA_CHECK(cudaMemset((void*)win_va, 0, GNN_WIN_BYTES));
  CUDA_CHECK(cudaDeviceSynchronize());
  volatile uint32_t* d_win = (volatile uint32_t*)win_va;
  uint32_t* d_win_ids = (uint32_t*)(win_va + GNN_SLOT_IDS_OFF);

  int fd = open("/dev/fpga_gdr", O_RDWR);
  if (fd < 0) { perror("open /dev/fpga_gdr"); return 1; }
  struct fpga_gdr_pin pin{};
  pin.va = win_va;
  pin.len = GNN_WIN_BYTES;
  if (ioctl(fd, FPGA_GDR_PIN, &pin) < 0) { perror("PIN"); return 1; }

  uint32_t entry_num = hmss_rd(fd, HMSS_REG_ENTRY_NUM);
  uint64_t base = ((uint64_t)hmss_rd(fd, HMSS_REG_BASE_HI) << 32) |
                  hmss_rd(fd, HMSS_REG_BASE_LO);
  uint32_t range_log = hmss_rd(fd, HMSS_REG_ADDR_RANGE);
  uint64_t slot_sz = (1ULL << range_log) / entry_num;
  uint64_t pps = slot_sz / GPU_PAGE;
  if (out_bo.address() != base) {
    std::fprintf(stderr, "aperture offset != 0 (bo 0x%lx base 0x%lx)\n",
                 out_bo.address(), base);
    return 1;
  }
  hmss_wr(fd, HMSS_REG_ENTRY_NUM, 0);
  for (uint64_t s = 0; s < GNN_WIN_BYTES / slot_sz; s++) {
    uint64_t pa = pin.dma_addrs[s * pps];
    for (uint64_t j = 1; j < pps; j++)
      if (pin.dma_addrs[s * pps + j] != pa + j * GPU_PAGE) {
        std::fprintf(stderr, "GPU pages not contiguous in slot %lu\n", s);
        return 1;
      }
    hmss_wr(fd, HMSS_REG_PTABLE + s * 8 + 0, (uint32_t)pa);
    hmss_wr(fd, HMSS_REG_PTABLE + s * 8 + 4, (uint32_t)(pa >> 32));
  }
  hmss_wr(fd, HMSS_REG_ENTRY_NUM, entry_num);
  std::printf("HMSS window programmed (%lu slots of %lu KB)\n",
              GNN_WIN_BYTES / slot_sz, slot_sz >> 10);

  unsigned long long* d_err;
  CUDA_CHECK(cudaMalloc(&d_err, sizeof(unsigned long long)));
  uint32_t* d_ids;
  CUDA_CHECK(cudaMalloc(&d_ids, (size_t)GNN_SQ_MAX_IDS * 4));
  cudaStream_t s_con;
  CUDA_CHECK(cudaStreamCreate(&s_con));

  std::mt19937 rng(42);
  uint32_t batch_seq = 0;
  bool all_pass = true;

  // One round: stage ids -> clear DONE -> launch consumer -> launch kernel ->
  // wait both -> return validation result. ms_out gets total round time.
  auto run_round = [&](const std::vector<uint32_t>& ids, int validate,
                       double* ms_out) -> bool {
    uint32_t n = ids.size();
    uint32_t bid = ++batch_seq;
    CUDA_CHECK(cudaMemcpy(d_ids, ids.data(), (size_t)n * 4,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_err, 0, sizeof(unsigned long long)));

    auto t0 = std::chrono::high_resolution_clock::now();
    if (submit_mode == "gpu") {
      stage_ids<<<64, 256>>>(d_win_ids, d_ids, n);
      CUDA_CHECK(cudaDeviceSynchronize());
    } else {
      CUDA_CHECK(cudaMemcpy(d_win_ids, ids.data(), (size_t)n * 4,
                            cudaMemcpyHostToDevice));
    }
    clear_done<<<1, 1>>>(d_win);
    CUDA_CHECK(cudaDeviceSynchronize());

    consume<<<1, 1024, 0, s_con>>>(d_win, d_ids, bid, n, row_beats, validate,
                                   d_err);
    auto run = krnl(out_bo /*sq: id reads target the window*/, feat_bo, out_bo,
                    bid, n, row_beats);
    auto st = run.wait(std::chrono::seconds(20));
    if (st != ERT_CMD_STATE_COMPLETED) {
      std::printf("KERNEL TIMEOUT (batch %u)\n", bid);
      std::exit(3);
    }
    CUDA_CHECK(cudaStreamSynchronize(s_con));
    auto t1 = std::chrono::high_resolution_clock::now();
    if (ms_out)
      *ms_out = std::chrono::duration<double>(t1 - t0).count() * 1e3;

    unsigned long long errs = 0;
    CUDA_CHECK(cudaMemcpy(&errs, d_err, sizeof(errs), cudaMemcpyDeviceToHost));
    if (validate && errs) std::printf("    (%llu word mismatches)\n", errs);
    return errs == 0;
  };

  // split arbitrary-size requests into ring-capped rounds
  auto run_batches = [&](const std::vector<uint32_t>& ids, int validate,
                         double* ms_out) -> bool {
    uint32_t max_rows = GNN_RING_BYTES / row_bytes;
    if (max_rows > GNN_SQ_MAX_IDS) max_rows = GNN_SQ_MAX_IDS;
    bool ok = true;
    double total_ms = 0;
    for (size_t off = 0; off < ids.size(); off += max_rows) {
      size_t n = std::min((size_t)max_rows, ids.size() - off);
      std::vector<uint32_t> part(ids.begin() + off, ids.begin() + off + n);
      double ms = 0;
      ok &= run_round(part, validate, &ms);
      total_ms += ms;
    }
    if (ms_out) *ms_out = total_ms;
    return ok;
  };

  // ---------------- G1: functionality ----------------
  std::printf("\n--- G1 functionality (validate every word) ---\n");
  struct Trial {
    const char* name;
    uint32_t n_ids, count;
    bool zipf;
  };
  std::vector<Trial> trials = {
      {"tiny", 16, 1, false},        {"small", 1024, 1, false},
      {"medium", 16384, 2, false},   {"ring-split", 65536, 4, false},
      {"zipf-skew", 32768, 2, true}, {"back2back", 4096, 8, false},
  };
  for (auto& t : trials) {
    bool ok = true;
    for (uint32_t k = 0; k < t.count && ok; k++) {
      std::vector<uint32_t> ids(t.n_ids);
      if (t.zipf) {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        for (auto& v : ids)
          v = (uint32_t)((n_nodes - 1) * std::pow(u(rng), 3.0));
      } else {
        std::uniform_int_distribution<uint32_t> u(0, n_nodes - 1);
        for (auto& v : ids) v = u(rng);
      }
      ok &= run_batches(ids, /*validate=*/1, nullptr);
    }
    std::printf("  %-12s n_ids=%-6u x%-2u : %s\n", t.name, t.n_ids, t.count,
                ok ? "PASS" : "FAIL");
    all_pass &= ok;
  }
  std::printf("G1 %s\n", all_pass ? "PASSED" : "FAILED");

  // ---------------- G2: performance ----------------
  std::printf("\n--- G2 performance (row=%uB, no validation) ---\n", row_bytes);
  std::printf("%-8s %-10s %-12s %-12s\n", "n_ids", "ms/batch", "GB/s",
              "Mfeat/s");
  for (uint32_t n_ids : {1024u, 4096u, 16384u}) {
    std::vector<uint32_t> ids(n_ids);
    std::uniform_int_distribution<uint32_t> u(0, n_nodes - 1);
    for (auto& v : ids) v = u(rng);
    double ms = 0;
    run_batches(ids, 0, &ms);  // warm-up
    const int iters = 10;
    double sum = 0;
    for (int i = 0; i < iters; i++) {
      run_batches(ids, 0, &ms);
      sum += ms;
    }
    double avg = sum / iters;
    double bytes = (double)n_ids * row_bytes;
    std::printf("%-8u %-10.3f %-12.2f %-12.1f\n", n_ids, avg,
                bytes / (1 << 30) / (avg / 1e3), n_ids / (avg / 1e3) / 1e6);
  }

  ioctl(fd, FPGA_GDR_UNPIN, 0);
  close(fd);
  CUDA_CHECK(cudaFree(raw));
  std::printf("\ndone\n");
  return all_pass ? 0 : 1;
}
