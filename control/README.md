# Control Plane — NVMeVirt Emulation Framework

This directory holds the **control plane** of the Intelligent Storage Emulator (ISE),
built on [NVMeVirt](nvmevirt/). NVMeVirt is a Linux kernel module that exposes a
fully virtual, DRAM-backed NVMe controller as a real PCIe device to the OS — the same
NVMe register model, admin/IO queues, and doorbells a physical SSD presents, but with
software-programmable latency and bandwidth.

## Role: baseline for the Intelligent Storage Emulator

NVMeVirt is the **baseline** the ISE is built on. Because it emulates a standards-compliant
NVMe device in DRAM, it can be driven by any real NVMe software stack unmodified — in
particular the **[BaM](vendor/bam) (Big accelerator Memory) / libnvm GPU-initiated NVMe
stack**. Binding BaM directly to the NVMeVirt virtual device lets us:

- **Verify the emulated device behaves like real NVMe hardware** — the same GPU-initiated
  submission/completion path, doorbell rings, and PRP/SGL data transfers that run against a
  physical NVMe SSD run unchanged against NVMeVirt.
- **Calibrate emulated performance against a real device** — NVMeVirt's per-command
  latency and derived bandwidth are tunable (via `/proc/nvmev/read_times`,
  `/proc/nvmev/write_times`, `/proc/nvmev/io_lat_ns`, and the `read_time`/`write_time` module parameters), so the
  emulator can be dialed in until BaM microbenchmarks match the numbers measured on the
  target SSD (e.g. a Samsung 990 PRO, the default `BASE_SSD`).

Once the baseline is trusted, the FPGA data plane (`driver/`, `hw/`) and the near-storage
gather/coalesce engine are layered on top — NVMeVirt stays the control plane while the
FPGA owns the GPU-memory data path. 

**For now, NVMeVirt-based emulation is completely CPU-based to serve as golden model guiding further architecture exploration on FPGA.**

## Layout

- [`nvmevirt/`](nvmevirt/) — the NVMeVirt kernel module (`nvmev.ko`). Emulates a
  Samsung 970 PRO-class SSD by default (`CONFIG_NVMEVIRT_SSD`).
- [`vendor/bam/`](vendor/bam/) — upstream BaM (bundles libnvm), vendored as a submodule.
  The GPU-initiated NVMe driver + userspace library used to exercise the emulated device.
- [`Makefile`](Makefile) — build/load/unload orchestration for both `nvmev.ko` and
  `libnvm.ko`.

## Quick start

Before starting, reference NVMeVirt document on how to reserve both host DRAM and CPU cores in GRUB. For reference, the following quick start assumes this line in GRUB:  `GRUB_CMDLINE_LINUX="memmap=64G\\\$64G isolcpus=16-31"`

```sh
# Build and load the emulated NVMe controller (reserves 64G at phys 64G by default)
make build
sudo make load                 # insmod nvmev.ko; waits for PCI dev 0001:10:00.0

# Attach the BaM / libnvm GPU-initiated NVMe stack to the virtual device
make build-libnvm              # kernel module built with gcc-9 (see note below)
sudo make load-libnvm          # unbinds stock nvme, insmods libnvm.ko, exposes /dev/libnvm*

sudo make status               # module + PCI + procfs status
```

See `make help` for the full target list and configurable parameters
(`MEMMAP_START`, `MEMMAP_SIZE`, `CPUS`).

## Requirements

The versions below are the verified, working toolchain on this host. The kernel-module
compiler and the NVIDIA driver flavor are strict requirements, not merely recommendations.

| Component | Required | Verified on this host |
|-----------|----------|-----------------------|
| **Linux kernel** | 5.15.x with headers/build tree present (`/lib/modules/$(uname -r)/build`) | `5.15.0-139-generic` |
| **Kernel-module compiler** | **gcc-9** — must match the kernel's `CONFIG_CC_VERSION_TEXT` (gcc 9.4.0). The system default gcc (13) will not produce a loadable module. | `gcc-9 (9.4.0)`, system `gcc 13.1.0` |
| **CUDA Toolkit** | 12.x (used to build BaM/libnvm's CUDA components and GPUDirect P2P) | `CUDA 12.8` (nvcc V12.8.61) |
| **NVIDIA driver** | **Open kernel module** (`Dual MIT/GPL`), 570.x. Required because `libnvm.ko` (and the optional NVMeVirt `GPU_DIRECT` path) uses the GPL-only `nvidia_p2p_*` symbols — the **proprietary** driver will fail to load with `Unknown symbol nvidia_p2p_*`. Source tree with `nv-p2p.h` + `Module.symvers` must be present. | `570.172.08` open module, src at `/usr/src/nvidia-570.172.08` |
| **GPU** | Must support the NVIDIA open kernel module and GPUDirect P2P (e.g. A6000/A100) | — |

### Notes

- **gcc-9 for the module only.** BaM's `module/Makefile.in` does not forward `CC`, so the
  `build-libnvm` target forces `make -C build/module CC=/usr/bin/gcc-9` (overridable via
  `LIBNVM_CC`). A "compiler differs" Kbuild warning here is a false positive — both are
  9.4.0; Kbuild is only comparing the `gcc` vs `gcc-9` strings.
- **Open driver is load-time critical.** Both `libnvm.ko` and NVMeVirt's optional
  `CONFIG_NVMEVIRT_GPU_DIRECT` build link against `nvidia_p2p_*`. Under the proprietary
  driver they build but fail to load. For the ISE integration NVMeVirt itself is kept as a
  pure DRAM-backed control plane (GPUDirect handled by the FPGA driver), so the strict
  open-driver dependency is carried by the BaM verification stack.
