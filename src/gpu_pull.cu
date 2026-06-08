// Milestone A — GPU pulls from FPGA HBM over PCIe P2P.
//
// The Alveo's HBM output buffer is allocated as an XRT P2P buffer object; XRT
// returns a host-mapped pointer that is really the resized BAR window over that
// HBM region. The GPU registers that BAR window as I/O memory
// (cudaHostRegisterIoMemory) and reads straight from FPGA HBM into GPU memory
// over a single PCIe hop — no host bounce.
//
// This is GPU-initiated peer access (not GDR in the strict sense), used as the
// topology/bandwidth reference. See plan: Milestone A.
//
// Pattern source: the prebuilt hello_world `vadd` kernel. With in0[i]=in1[i]=i,
// the kernel writes out[i] = 2*i into HBM; the GPU reads it back and validates.

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>

// XRT
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      std::cerr << "CUDA error " << cudaGetErrorName(_e) << " ("               \
                << cudaGetErrorString(_e) << ") at " << __FILE__ << ":"        \
                << __LINE__ << " -> " << #call << std::endl;                   \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// Non-fatal variant: returns the cudaError_t for the caller to handle.
static inline cudaError_t cuda_try(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::cerr << "CUDA warning: " << cudaGetErrorName(e) << " ("
              << cudaGetErrorString(e) << ") during " << what << std::endl;
  }
  return e;
}

// ---------------------------------------------------------------------------
// CUDA kernel: GPU reads directly from the FPGA BAR window (src) into GPU mem.
// Fallback / comparison path for the cudaMemcpy route.
// ---------------------------------------------------------------------------
__global__ void p2p_read_kernel(unsigned int* __restrict__ dst,
                                const unsigned int* __restrict__ src,
                                size_t n) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) dst[i] = src[i];
}

// ---------------------------------------------------------------------------
// Tiny argument parser (keeps the program self-contained; no Vitis common dep).
// ---------------------------------------------------------------------------
static std::string get_arg(int argc, char** argv, const std::string& a,
                           const std::string& b, const std::string& def) {
  for (int i = 1; i < argc - 1; ++i) {
    if (a == argv[i] || b == argv[i]) return argv[i + 1];
  }
  return def;
}

static double bytes_to_gb(size_t b) { return (double)b / (1024.0 * 1024.0 * 1024.0); }

struct BwResult {
  double avg_ms;
  double best_ms;
  double avg_gbps;
  double best_gbps;
};

// Time `iters` runs of a transfer of `bytes`, return stats.
template <typename Fn>
static BwResult time_transfer(size_t bytes, int warmup, int iters, Fn run,
                              cudaStream_t stream, cudaEvent_t start,
                              cudaEvent_t stop) {
  for (int i = 0; i < warmup; ++i) run();
  CUDA_CHECK(cudaStreamSynchronize(stream));

  double total_ms = 0.0;
  double best_ms = 1e30;
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(cudaEventRecord(start, stream));
    run();
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    total_ms += ms;
    best_ms = std::min(best_ms, (double)ms);
  }
  double avg_ms = total_ms / iters;
  BwResult r;
  r.avg_ms = avg_ms;
  r.best_ms = best_ms;
  r.avg_gbps = bytes_to_gb(bytes) / (avg_ms / 1000.0);
  r.best_gbps = bytes_to_gb(bytes) / (best_ms / 1000.0);
  return r;
}

int main(int argc, char** argv) {
  const std::string binaryFile =
      get_arg(argc, argv, "--xclbin_file", "-x", "");
  const int device_index = std::stoi(get_arg(argc, argv, "--device_id", "-d", "0"));
  // Default 128 MB: the vadd `gmem0` HBM bank (512 MB) holds BOTH an input and
  // the P2P output, so each must stay well under 512 MB or XRT returns EPERM.
  const size_t max_bytes =
      (size_t)std::stoull(get_arg(argc, argv, "--max_mb", "-m", "128")) * 1024 * 1024;

  if (binaryFile.empty()) {
    std::cerr << "Usage: " << argv[0]
              << " -x <vadd.xclbin> [-d device_index] [-m max_MB]" << std::endl;
    return EXIT_FAILURE;
  }

  const size_t count_max = max_bytes / sizeof(unsigned int);
  std::cout << "Max transfer: " << (max_bytes / (1024 * 1024)) << " MB ("
            << count_max << " uints)\n";

  // -------------------------------------------------------------------------
  // 1) XRT: open device, load xclbin, set up vadd kernel + buffers.
  // -------------------------------------------------------------------------
  std::cout << "Open device " << device_index << "\n";
  auto device = xrt::device(device_index);
  std::cout << "Load xclbin " << binaryFile << "\n";
  auto uuid = device.load_xclbin(binaryFile);
  auto krnl = xrt::kernel(device, uuid, "vadd");

  std::cout << "Allocate HBM buffers (P2P output)\n";
  auto bo0 = xrt::bo(device, max_bytes, krnl.group_id(0));  // in1
  auto bo1 = xrt::bo(device, max_bytes, krnl.group_id(1));  // in2
  // Output buffer in HBM, exposed over the P2P BAR window.
  auto p2p_out =
      xrt::bo(device, max_bytes, xrt::bo::flags::p2p, krnl.group_id(2));

  auto in0 = bo0.map<unsigned int*>();
  auto in1 = bo1.map<unsigned int*>();
  for (size_t i = 0; i < count_max; ++i) {
    in0[i] = (unsigned int)i;
    in1[i] = (unsigned int)i;
  }
  bo0.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo1.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  // host-mapped pointer == BAR window over HBM
  void* p2p_ptr = p2p_out.map<void*>();
  std::cout << "P2P BAR window mapped at host VA " << p2p_ptr << "\n";

  std::cout << "Run vadd kernel -> HBM holds out[i] = 2*i\n";
  auto run = krnl(bo0, bo1, p2p_out, (int)count_max);
  run.wait();
  // NOTE: deliberately NO sync(FROM_DEVICE) — the GPU reads HBM via the BAR.

  // -------------------------------------------------------------------------
  // 2) GPU: register the FPGA BAR window as I/O memory, get a device pointer.
  // -------------------------------------------------------------------------
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::cout << "GPU: " << prop.name << " (sm_" << prop.major << prop.minor
            << ")\n";

  // The xocl P2P BAR is backed by ZONE_DEVICE struct pages, so cudaHostRegister
  // with the IoMemory flag (meant for raw pfn-mapped IO) is rejected; the
  // Default flag registers it as ordinary pinnable memory and works on the
  // NVIDIA proprietary driver. Try IoMemory first, fall back to Default.
  std::cout << "cudaHostRegister(BAR window)...\n";
  const char* reg_kind = "IoMemory";
  cudaError_t reg = cudaHostRegister(p2p_ptr, max_bytes, cudaHostRegisterIoMemory);
  if (reg != cudaSuccess) {
    std::cout << "  IoMemory rejected (" << cudaGetErrorString(reg)
              << "); retrying with Default flag...\n";
    reg = cudaHostRegister(p2p_ptr, max_bytes, cudaHostRegisterDefault);
    reg_kind = "Default";
  }
  if (reg != cudaSuccess) {
    std::cerr << "FATAL: GPU could not register the FPGA P2P BAR ("
              << cudaGetErrorString(reg) << ").\n"
                 "  Requires the NVIDIA *proprietary* kernel module.\n"
                 "  Check:  modinfo nvidia | grep license   (need 'NVIDIA', not "
                 "'Dual MIT/GPL')\n"
                 "  Also confirm: xbutil reports 'P2P Status: enabled' and IOMMU "
                 "is off (cat /proc/cmdline).\n";
    return EXIT_FAILURE;
  }
  std::cout << "  registered via " << reg_kind << " flag\n";

  unsigned int* d_peer = nullptr;
  CUDA_CHECK(cudaHostGetDevicePointer((void**)&d_peer, p2p_ptr, 0));
  std::cout << "Device-accessible peer pointer: " << (void*)d_peer << "\n";

  unsigned int* d_gpu = nullptr;
  CUDA_CHECK(cudaMalloc((void**)&d_gpu, max_bytes));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  // -------------------------------------------------------------------------
  // 3) Correctness: pull the whole buffer FPGA->GPU and validate out[i]==2*i.
  //    Try cudaMemcpy first; if it errors, fall back to the read kernel.
  // -------------------------------------------------------------------------
  const int threads = 256;
  const int blocks = 1024;
  bool memcpy_ok = true;

  std::cout << "\n--- Correctness check (FPGA HBM -> GPU) ---\n";
  cudaError_t cp = cuda_try(
      cudaMemcpy(d_gpu, d_peer, max_bytes, cudaMemcpyDefault),
      "cudaMemcpy(FPGA P2P -> GPU)");
  if (cp != cudaSuccess) {
    memcpy_ok = false;
    std::cout << "cudaMemcpy path failed; using read-kernel path instead.\n";
    p2p_read_kernel<<<blocks, threads, 0, stream>>>(d_gpu, d_peer, count_max);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  std::vector<unsigned int> h_check(count_max);
  CUDA_CHECK(cudaMemcpy(h_check.data(), d_gpu, max_bytes, cudaMemcpyDeviceToHost));

  size_t mismatches = 0;
  size_t first_bad = (size_t)-1;
  for (size_t i = 0; i < count_max; ++i) {
    unsigned int expect = 2u * (unsigned int)i;
    if (h_check[i] != expect) {
      if (first_bad == (size_t)-1) first_bad = i;
      ++mismatches;
    }
  }
  if (mismatches) {
    std::cerr << "VALIDATION FAILED: " << mismatches << " mismatches, first at i="
              << first_bad << " got " << h_check[first_bad] << " expected "
              << (2u * (unsigned int)first_bad) << "\n";
    return EXIT_FAILURE;
  }
  std::cout << "Pattern validated: out[i] == 2*i for all " << count_max
            << " elements. TEST PASSED\n";

  // -------------------------------------------------------------------------
  // 4) Bandwidth sweep: FPGA->GPU via memcpy and via read-kernel, per size.
  //    Plus a host->GPU pinned reference as the single-hop PCIe ceiling.
  // -------------------------------------------------------------------------
  const int warmup = 3, iters = 30;
  std::vector<size_t> sizes_mb = {1, 4, 16, 64, 128};

  // Reference ceiling: pinned host -> GPU H2D at max size.
  unsigned int* h_pinned = nullptr;
  CUDA_CHECK(cudaHostAlloc((void**)&h_pinned, max_bytes, cudaHostAllocDefault));
  std::memset(h_pinned, 0, max_bytes);
  BwResult ref = time_transfer(
      max_bytes, warmup, iters,
      [&]() {
        CUDA_CHECK(cudaMemcpyAsync(d_gpu, h_pinned, max_bytes,
                                   cudaMemcpyHostToDevice, stream));
      },
      stream, ev_start, ev_stop);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  const double ceiling_gbps = ref.best_gbps;

  std::cout << "\n--- Bandwidth sweep (single PCIe hop) ---\n";
  std::cout << std::left << std::setw(8) << "Size" << std::setw(16)
            << "memcpy GB/s" << std::setw(16) << "kernel GB/s" << std::setw(14)
            << "%ceiling" << "\n";

  for (size_t mb : sizes_mb) {
    size_t bytes = mb * 1024 * 1024;
    if (bytes > max_bytes) continue;
    size_t count = bytes / sizeof(unsigned int);

    double memcpy_gbps = 0.0;
    if (memcpy_ok) {
      BwResult r = time_transfer(
          bytes, warmup, iters,
          [&]() {
            CUDA_CHECK(cudaMemcpyAsync(d_gpu, d_peer, bytes, cudaMemcpyDefault,
                                       stream));
          },
          stream, ev_start, ev_stop);
      memcpy_gbps = r.best_gbps;
    }

    BwResult rk = time_transfer(
        bytes, warmup, iters,
        [&]() {
          p2p_read_kernel<<<blocks, threads, 0, stream>>>(d_gpu, d_peer, count);
        },
        stream, ev_start, ev_stop);

    double best_path = std::max(memcpy_gbps, rk.best_gbps);
    std::cout << std::left << std::setw(8) << (std::to_string(mb) + "MB")
              << std::setw(16) << std::fixed << std::setprecision(2)
              << (memcpy_ok ? memcpy_gbps : 0.0) << std::setw(16) << rk.best_gbps
              << std::setw(14) << std::setprecision(1)
              << (100.0 * best_path / ceiling_gbps) << "\n";
  }

  std::cout << "\nReference ceiling (pinned host->GPU H2D, " << (max_bytes/(1024*1024))
            << "MB): " << std::fixed << std::setprecision(2) << ceiling_gbps
            << " GB/s\n";
  std::cout << "Topology note: GPU and FPGA are under different CPU root "
               "complexes; P2P crosses the Infinity Fabric.\n";

  // -------------------------------------------------------------------------
  // Cleanup
  // -------------------------------------------------------------------------
  CUDA_CHECK(cudaEventDestroy(ev_start));
  CUDA_CHECK(cudaEventDestroy(ev_stop));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFreeHost(h_pinned));
  CUDA_CHECK(cudaFree(d_gpu));
  CUDA_CHECK(cudaHostUnregister(p2p_ptr));

  std::cout << "\nDone.\n";
  return 0;
}
