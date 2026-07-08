# Control Plane — NVMeVirt + Emulated Middle Layer

This directory holds the **control plane** of the Intelligent Storage Emulator (ISE),
built on [NVMeVirt](nvmevirt/). It now contains two cooperating pieces:

1. **NVMeVirt** — a Linux kernel module that exposes a fully virtual, DRAM-backed NVMe
   controller as a real PCIe device to the OS. Same NVMe register model, admin/IO queues,
   and doorbells a physical SSD presents, but with software-programmable latency and
   bandwidth. This is the **baseline** the ISE is calibrated against.
2. **The emulated middle layer** (`nvmevirt/coalesce.c` + `backend_*.c` +
   `coalesce_backend.h`) — an in-kernel **staging + sort-reduce coalescing layer** that
   sits between storage and GPU VRAM and gathers GNN feature rows on the GPU's behalf. This
   is the current research vehicle: a CPU **golden model** of a near-storage coalescing
   engine, wired to swappable storage backends and GPU-delivery planes.

## Why a middle layer — memory architecture for GNN training on large graphs

GNN training over billion-scale graphs (e.g. IGB-260M, ~138 GB of features) is bound by
**feature-gather memory traffic**, not compute. Each sampled mini-batch needs the feature
rows of thousands of node IDs scattered across a table far larger than GPU HBM, so the
features must stream from storage every step. Two properties make this expensive:

- **I/O amplification.** A feature row (e.g. 128×fp32 = 512 B) is much smaller than a
  storage page (4 KB). Reading one row naively drags in a whole page — up to `A = page/row`
  (≈8×) wasted bytes.
- **Redundancy.** Within and across mini-batches the same hot nodes/pages are re-requested
  many times (measured ~246× per page per epoch on IGB-medium). That reuse is mostly
  *temporal* / cross-batch, invisible to a per-request path.

The middle layer is the architectural response: interpose a **reduction stage between the
SSD and the GPU** that (a) coalesces many sub-page row requests that hit the same page into
one page read, and (b) delivers **only the useful rows** across the GPU PCIe link. Staging
lives *off-GPU* (host DRAM today, FPGA HBM later), so the GPU link carries ≈1 byte per
useful byte instead of `A` bytes per useful byte — a ×A amplification win on top of the ×C
win from coalescing away redundant reads.

This CPU implementation is the **oracle**: it lets us measure achievable coalescing,
delivery bandwidth, and the crossover against a GPU-cache design (GIDS/BaM) *before*
committing the reduction engine to FPGA. The two-stage bucket→reduce model
(`coalesce_sort=3`) directly mirrors the RTL Tier-1/Tier-2 coalescer it will size (see
[`docs/coalescing_engine_architecture.md`](../docs/coalescing_engine_architecture.md) and
[`docs/middle-layer-assessment.md`](../docs/middle-layer-assessment.md)).

## Architecture — three decoupled planes

The middle layer is deliberately split into three planes joined by two kernel-internal
vtables ([`coalesce_backend.h`](nvmevirt/coalesce_backend.h)), so each end can be swapped
without touching the coalescing logic or the GPU-facing ABI:

```
        GPU app (DGL / libisefs)
              │  node-ID requests          ▲  completed rows
              ▼                            │
   ┌───────── L1 ring ABI (FROZEN) ──────────────────┐   NORTH interface
   │   /dev/nvmev_l1: l1_req (32B) / l1_cqe (16B)    │   (feature-store contract)
   └──────────────────────┬──────────────────────────┘
                          │
   ┌──────────────────────▼───────────────────────────┐
   │  MIDDLE  (coalesce.c) — pure logic               │
   │   page-sharded kthreads → window → sort/bucket   │
   │   → page-keyed coalesce → waiter fan-out         │
   │   golden model of the FPGA Tier-1/2 coalescer    │
   └──────┬───────────────────────────────┬───────────┘
          │ co_deliver_ops (FRONT)        │ co_backend_ops (SOUTH)
          ▼                               ▼
   ┌──────────────────┐            ┌──────────────────────────────┐
   │ GPU delivery:    │            │ Storage backend (unified):   │
   │  • cpu   WC-store│            │  • model   emulated DRAM     │
   │  • gpupull GPU   │            │  • nvme    real SSD via bio  │
   │    persistent    │            │  • direct  real SSD via own  │
   │    gather kernel │            │            poll-mode NVMe QP │
   └──────────────────┘            └──────────────────────────────┘
```

- **North — the L1 ring ABI** is the feature-store interface, byte-frozen
  (`BUILD_BUG_ON` in `coalesce.c`). A GPU app produces node-ID requests into a per-shard
  MPSC ring and consumes completed rows; `sw/libisefs` + `sw/isefs.py` wrap it as a DGL
  `DataLoader` (`ISE_DGLDataLoader`). This never changes as the backends below it change.
- **Middle — the coalescing logic** shards requests by page across dedicated isolated CPU
  cores (no locking, no missed coalescing), accumulates a window, sorts/buckets it, and
  reduces same-page requests to one backend read with a per-page **waiter list** that
  scatters the fetched page back to every requesting row. This is the piece the FPGA
  ultimately replaces; on CPU it is instrumented (`/proc/nvmev/coalesce`) as the oracle.
- **Front — `co_deliver_ops`** decides how a finished row reaches VRAM: `cpu` (CPU
  write-combining store to the GPU BAR, the simplest model) or `gpupull` (a persistent GPU
  gather kernel pulls rows from host staging — ~4× faster, meets single-NVMe bandwidth).
- **South — `co_backend_ops`** is the **unified storage interface** (below).

## The unified backend interface — emulated *and* real NVMe

The whole point of `co_backend_ops` is that the identical middle-and-north stack drives
**either an emulated backend or a real NVMe device**, selected at load time by the
`coalesce_backend` parameter with no change to the coalescing logic, the L1 ABI, or the GPU
app:

| Backend | Provider | `coalesce_backend=` | What it is |
|---------|----------|---------------------|------------|
| **Emulated DRAM** | [`backend_model.c`](nvmevirt/backend_model.c) | *(empty)* | Pages served from a DRAM model with a programmable per-page latency ring. Zero real I/O — for correctness, coalescing-ratio, and delivery-plane measurement in isolation. |
| **Real SSD (block layer)** | [`backend_nvme.c`](nvmevirt/backend_nvme.c) | `/dev/nvmeXn1[,...]` | One `bio` (`REQ_OP_READ`) per unique coalesced page through the **stock kernel NVMe driver**. Multi-device page striping supported. The CPU-in-path reference. **Deprecated due to significantly high software overhead leading to sub-optimal performance** |
| **[Recommended] Real SSD (owned QP)** | [`backend_direct.c`](nvmevirt/backend_direct.c) | `pci:DDDD:BB:DD.F` | The middle layer **unbinds the SSD and drives its own poll-mode NVMe I/O queue-pair** (one SQ/CQ per shard, SQE built in place, doorbell rung, CQ polled — no block layer, no IRQ). Removes the south-side CPU tax; the executable reference for the future FPGA NVMe initiator. |

Because the emulated NVMe *front* (NVMeVirt's virtual controller) and the storage *backend*
are independent, the same emulated device the OS sees can be backed by DRAM for a clean
oracle run, then re-pointed at a physical Samsung SSD for a faithful end-to-end run — the
GNN training script (`sw/example_graphsage_isefs.py`) and the L1 contract are byte-identical
across all three. This is what lets the CPU golden model and the real-hardware path share
one codebase, and it is the same seam the **FPGA data plane** will slot behind (a
`backend_fpga` provider) when the coalescer moves to silicon.

Delivery plane and backend are orthogonal: `coalesce_deliver=0` (CPU WC-store) or `=1`
(GPU-pull) compose with any of the three backends. `gpu-pull + direct` (GPU pulls rows the
SSD DMA'd into host staging) is the fully CPU-out-of-the-byte-path configuration verified
end-to-end.

## Role in the larger ISE research

NVMeVirt-based emulation stays **CPU-based on purpose**: it is the golden model that guides
FPGA architecture exploration, and the calibration harness that proves the emulated device
behaves like real NVMe hardware. Concretely, binding **[BaM](vendor/bam) / libnvm**
(GPU-initiated NVMe) directly to the NVMeVirt virtual device lets us:

- **Verify the emulated device behaves like real NVMe hardware** — the same GPU-initiated
  submission/completion path, doorbell rings, and PRP/SGL transfers that run against a
  physical NVMe SSD run unchanged against NVMeVirt.
- **Calibrate emulated performance against a real device** — NVMeVirt's per-command latency
  and derived bandwidth are tunable (`/proc/nvmev/read_times`, `/proc/nvmev/write_times`,
  `/proc/nvmev/io_lat_ns`, and the `read_time`/`write_time` module parameters), so the
  emulator can be dialed in until BaM microbenchmarks match the target SSD (e.g. a Samsung
  990 PRO, the default `BASE_SSD`).

Once the baseline is trusted, the middle layer measures the memory-architecture question —
how much feature-gather traffic coalescing removes, at what CPU cost — and hands the FPGA
data plane (`../driver/`, `../hw/`) a sized, validated Tier-1/Tier-2 coalescer to implement.
BaM/GIDS is the head-to-head baseline (GPU-cache design) the middle layer's staging-and-reduce
design is measured against (`sw/bench_ise_vs_gids.py`): GIDS wins when the working set fits
HBM, the middle layer wins on datasets that exceed it — the regime that motivates the whole
project.

## Layout

- [`nvmevirt/`](nvmevirt/) — the NVMeVirt kernel module (`nvmev.ko`), emulated NVMe front +
  the middle layer:
  - [`coalesce.c`](nvmevirt/coalesce.c) / [`coalesce.h`](nvmevirt/coalesce.h) — L1 ring, the
    page-sharded coalescing logic, delivery planes, `/proc/nvmev/coalesce`. Built under
    `CONFIG_NVMEVIRT_MIDLAYER`.
  - [`coalesce_backend.h`](nvmevirt/coalesce_backend.h) — the `co_deliver_ops` (front) and
    `co_backend_ops` (south) vtables that decouple the three planes.
  - [`backend_model.c`](nvmevirt/backend_model.c) / [`backend_nvme.c`](nvmevirt/backend_nvme.c)
    / [`backend_direct.c`](nvmevirt/backend_direct.c) — the three storage backends.
  - [`gpu_mem.c`](nvmevirt/gpu_mem.c) — GPUDirect VRAM pinning + `VA→dst_phys` translation
    used by both delivery planes.
- [`vendor/bam/`](vendor/bam/) — upstream BaM (bundles libnvm), vendored as a submodule. The
  GPU-initiated NVMe stack used to calibrate the emulated device against real hardware.
- [`Makefile`](Makefile) — build/load/unload orchestration for both `nvmev.ko` and
  `libnvm.ko`, plus the `reload-8shard` helper for middle-layer runs.

The userspace feature-store library, DGL integration, and benchmarks live in
[`../sw/`](../sw/) (`libisefs`, `isefs.py`, `microbench_featfetch.py`,
`bench_ise_vs_gids.py`).

## Quick start

Before starting, reference the NVMeVirt document on reserving host DRAM and CPU cores in
GRUB. The quick start assumes something like:
`GRUB_CMDLINE_LINUX="memmap=64G\\\$64G isolcpus=8-15,24-31"` (whole physical cores isolated
for the shard kthreads and their SMT siblings).

```sh
# Build and load the emulated NVMe controller (reserves 64G at phys 64G by default)
make build
sudo make load                 # insmod nvmev.ko; waits for PCI dev 0001:10:00.0

# --- Middle-layer runs (pick a backend) ----------------------------------
# Emulated DRAM backend, 8 shards on isolated cores, GPU-pull delivery:
sudo make reload-8shard DELIVER=1
# Real SSD via the middle-layer-owned poll-mode NVMe QP:
sudo make reload-8shard BACKEND=pci:0000:62:00.0 DELIVER=1
# Real SSD via the stock block layer:
sudo make reload-8shard BACKEND=/dev/nvme7n1

# --- BaM calibration path (independent of the middle layer) --------------
make build-libnvm              # kernel module built with gcc-9 (see note below)
sudo make load-libnvm          # unbinds stock nvme, insmods libnvm.ko, exposes /dev/libnvm*

sudo make status               # module + PCI + procfs status
```

Enable the middle layer with `coalesce_enable=1` and watch `/proc/nvmev/coalesce` for
coalescing ratio, amplification, window depth, and per-backend read counts. See `make help`
for the full target list and parameters (`MEMMAP_START`, `MEMMAP_SIZE`, `CPUS`, `RING`,
`DELIVER`, `BACKEND`, `EXTRA_PARAMS`).

Key middle-layer module parameters (load-time `0444` unless noted): `coalesce_cores`,
`coalesce_cpu_base`, `coalesce_split` (split-role SMT issuer/reaper), `coalesce_deliver`
(0=CPU WC-store, 1=GPU-pull), `coalesce_backend` (selects the storage backend above),
`coalesce_ring`; and live `0644` knobs `coalesce_window`, `coalesce_flush_min`,
`coalesce_flush_us`, `coalesce_sort` (0=window sort, 3=two-stage bucket→reduce).

## Requirements

The versions below are the verified, working toolchain on this host. The kernel-module
compiler and the NVIDIA driver flavor are strict requirements, not merely recommendations.

| Component | Required | Verified on this host |
|-----------|----------|-----------------------|
| **Linux kernel** | 5.15.x with headers/build tree present (`/lib/modules/$(uname -r)/build`) | `5.15.0-139-generic` |
| **Kernel-module compiler** | **gcc-9** — must match the kernel's `CONFIG_CC_VERSION_TEXT` (gcc 9.4.0). The system default gcc (13) will not produce a loadable module. | `gcc-9 (9.4.0)`, system `gcc 13.1.0` |
| **CUDA Toolkit** | 12.x (used to build BaM/libnvm's CUDA components and the GPU-pull gather kernel) | `CUDA 12.8` (nvcc V12.8.61) |
| **NVIDIA driver** | **Open kernel module** (`Dual MIT/GPL`), 570.x. Required because `libnvm.ko`, the middle layer's `gpu_mem.c`, and the optional NVMeVirt `GPU_DIRECT` path use the GPL-only `nvidia_p2p_*` symbols — the **proprietary** driver fails to load with `Unknown symbol nvidia_p2p_*`. Source tree with `nv-p2p.h` + `Module.symvers` must be present. | `570.172.08` open module, src at `/usr/src/nvidia-570.172.08` |
| **GPU** | Must support the NVIDIA open kernel module and GPUDirect P2P (e.g. A6000/A100) | — |

### Notes

- **gcc-9 for the module only.** BaM's `module/Makefile.in` does not forward `CC`, so the
  `build-libnvm` target forces `make -C build/module CC=/usr/bin/gcc-9` (overridable via
  `LIBNVM_CC`). A "compiler differs" Kbuild warning here is a false positive — both are
  9.4.0; Kbuild is only comparing the `gcc` vs `gcc-9` strings.
- **Open driver is load-time critical.** Both `libnvm.ko` and the middle layer's GPUDirect
  path link against `nvidia_p2p_*`. Under the proprietary driver they build but fail to
  load. NVMeVirt's *emulated NVMe front* is a pure DRAM-backed control plane and does not
  need the open driver; the GPU-delivery planes (`gpu_mem.c`) and the BaM calibration stack
  do.
- **`bare make` prints help, it does not build.** Use `make build` (or `make -C control
  build`). The middle-layer ioctls (`VA→dst_phys`, staging/descriptor info) need a *fresh*
  `nvmev.ko` load — `ENOTTY` from a probe usually means a stale module.
- **CPU isolation is a measured cost, not incidental.** Every shard core the middle layer
  consumes is a host core the FPGA is meant to give back; runs report cores-consumed
  alongside throughput. `coalesce_split=1` puts the issuer and the WC-store reaper on SMT
  siblings of the same physical core.
</content>
</invoke>
