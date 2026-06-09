# gpu-fpga-dma

FPGA↔GPU data path experiments on an Alveo U55C + RTX A6000 (see per-milestone
sections below).

- **Milestone A** — GPU pulls from FPGA HBM (GPU-initiated peer read). ~4.40 GB/s.
- **Milestone B** — true GPUDirect RDMA: the FPGA is DMA master and writes into
  GPU VRAM. Validated byte-for-byte; **~12.87 GB/s** (Gen3 x16 line rate) with the
  512-bit kernel.

---

# Milestone B — FPGA DMAs into GPU memory (true GDR)

The FPGA becomes the DMA master and writes a known pattern straight into a
`cudaMalloc` buffer; a GPU kernel validates it byte-for-byte.

### How it works
1. An HLS kernel ([hw/gdr_write.cpp](hw/gdr_write.cpp)) has an AXI master wired to
   `HOST[0]` — the XDMA shell's slave bridge (HMSS). Writes through it become PCIe
   memory-write TLPs whose destination is set by the HMSS address-translation
   page table.
2. The [fpga_gdr](driver/fpga_gdr.c) kernel module pins the GPU buffer via the
   legacy `nvidia_p2p_*` API (`nvidia_p2p_get_pages` + `dma_map_pages`, peer =
   the Alveo) to get the GPU's PCIe bus addresses, and exposes raw read/write of
   the HMSS registers.
3. [src/gpu_push.cu](src/gpu_push.cu) programs each HMSS slot covering the
   transfer with the matching GPU bus address (instead of host CMA), runs the
   kernel, and validates from GPU memory.

This is the literal GDR model and reuses the **installed XDMA shell** — no QDMA
platform or custom PCIe RTL required.

### Build & run
```bash
# 1) build the FPGA kernel (~44 min place & route)
./hw/build_xclbin.sh                 # -> hw/build/gdr_write.xclbin
# 2) build host programs
cmake -B build && cmake --build build -j
# 3) load driver + enable the slave bridge (needs sudo), then run
./scripts/spike.sh                   # Phase-0: one 64KB write, gates everything
./build/gpu_push -x hw/build/gdr_write.xclbin -d 0 -m 16
```
`scripts/spike.sh` loads `fpga_gdr.ko`, runs `xbutil configure --host-mem -s 256M
ENABLE`, and `chmod 666 /dev/fpga_gdr`. After that, `gpu_push` needs no sudo.

### Results (this host)

The kernel ([hw/gdr_write.cpp](hw/gdr_write.cpp)) uses a **512-bit** AXI master
(`ap_uint<512>`, pipelined II=1, `num_write_outstanding=32`,
`max_write_burst_length=64`) ⇒ 64 B/cycle × 300 MHz = 19.2 GB/s theoretical.

```
Size     write GB/s   result
64KB     2.45         OK
256KB    6.32         OK
1MB      10.26        OK
4MB      12.44        OK
16MB     12.87        OK          # FPGA wrote into GPU VRAM, validated
```
- **Correctness: PASSED** — FPGA writes 64KB–16MB into GPU VRAM, validated
  byte-for-byte.
- **Sustained write ≈ 12.87 GB/s** at 16 MB — essentially the **Gen3 x16 line
  rate** of the U55C link, and ~2.9× the Milestone A read (4.40 GB/s). The
  bottleneck is now the **PCIe link**, not the kernel.
- History: the original **32-bit** kernel sustained only **0.40 GB/s** (kernel-
  limited; a `-H` host-target run matched it). Widening to 512-bit + deeper
  outstanding writes gave a **~32× speedup** to line rate.

> Note: the `-H` host-target diagnostic is only meaningful right after a fresh
> host-mem enable; once a GPU run has repointed the HMSS slots, re-enable host-mem
> before trusting it.

### Constraints / notes
- HMSS slot size = `host_mem_size / 256`. At `-s 256M` that's **1 MB/slot**; the
  256-page PIN return then caps a single transfer at **16 MB**. For >16 MB, add a
  driver `PROGRAM` ioctl (programs all slots in-kernel) and/or use 64 KB slots
  (`-s 16M`).
- `gpu_push` verifies the GPU's nvidia_p2p pages are **bus-contiguous within each
  slot** before programming, and aborts with guidance otherwise.
- IOMMU is off (`amd_iommu=off`), so the nvidia_p2p DMA addresses equal GPU BAR1
  physical addresses.

---

# Milestone A — GPU pulls from FPGA HBM

Lowest-effort end-to-end FPGA→GPU data path over a single PCIe hop, used as the
**topology/bandwidth reference** for later GPUDirect (GDR) work.

## How it works

1. A buffer in the Alveo's HBM is allocated as an XRT **P2P** buffer object
   (`xrt::bo::flags::p2p`). XRT returns a host-mapped pointer that is really the
   resized **BAR window** over that HBM region.
2. The prebuilt `vadd` kernel fills that HBM buffer with a known pattern
   (`out[i] = in0[i] + in1[i]`, with `in0[i]=in1[i]=i` → `out[i] = 2*i`).
3. The GPU registers the BAR window as I/O memory
   (`cudaHostRegister(..., cudaHostRegisterIoMemory)`), gets a device pointer
   (`cudaHostGetDevicePointer`), and reads straight from FPGA HBM into GPU
   memory — no host bounce.

This is **GPU-initiated peer access**, not GDR in the strict sense, but it moves
data FPGA→GPU over a single PCIe hop and proves the whole physical path.

## This host (verified)

| | |
|---|---|
| GPU | NVIDIA RTX A6000 @ `0000:c1:00.0`, CUDA 12.8, driver 570 (open) |
| FPGA | Alveo U55C (HBM) @ `0000:21:00.1`, shell `xilinx_u55c_gen3x16_xdma_base_3`, XRT 2.17 |
| P2P | `xbutil` reports **P2P Status: enabled**, 32 GB P2P BAR |
| IOMMU | `amd_iommu=off`, `pcie_aspm=off` (ideal for P2P) |
| Topology | GPU under root complex `c0`, FPGA under `20` — **different CPU root complexes**, so P2P crosses the Threadripper Infinity Fabric |

## Build & run

```bash
source /opt/xilinx/xrt/setup.sh
cmake -B build && cmake --build build -j
./scripts/run.sh                 # or:
./build/gpu_pull -x <vadd.xclbin> -d 0 -m 128
```

Options: `-x` xclbin, `-d` device index, `-m` max transfer size in MB.
Keep `-m` ≤ ~240: vadd's `gmem0` HBM bank (512 MB) holds both an input buffer
and the P2P output, so each buffer must stay well under the bank size.

## Results (this host)

```
--- Correctness check (FPGA HBM -> GPU) ---
Pattern validated: out[i] == 2*i for all 33554432 elements. TEST PASSED

--- Bandwidth sweep (single PCIe hop) ---
Size    memcpy GB/s     kernel GB/s     %ceiling
1MB     4.32            4.22            17.3
...
128MB   4.40            4.31            17.6

Reference ceiling (pinned host->GPU H2D, 128MB): 24.98 GB/s
```

Sustained FPGA→GPU P2P read is **~4.40 GB/s**, flat across transfer sizes and
flat across 1–8 concurrent streams — i.e. a genuine **fabric ceiling**, not a
latency/outstanding-request limit. The cap comes from the GPU and FPGA living
under **different CPU root complexes** (P2P crosses the Threadripper Infinity
Fabric). This 4.40 GB/s is the topology/bandwidth reference for later GDR work;
co-locating both cards under one PCIe switch/root complex is the lever to raise
it.

## Driver requirement (important)

Two GPU-side requirements, both non-obvious:

1. **NVIDIA proprietary kernel module** (the **open** module, `Dual MIT/GPL`,
   rejects mapping a foreign PCIe BAR — both `cudaHostRegisterIoMemory` and
   dmabuf import fail).
2. **Register with `cudaHostRegisterDefault`, not `cudaHostRegisterIoMemory`.**
   xocl backs the P2P BAR with ZONE_DEVICE struct pages, so `IoMemory` (for raw
   pfn-mapped IO) returns *invalid argument*; the `Default` flag registers it as
   ordinary pinnable memory and works. `gpu_pull` tries `IoMemory` first and
   falls back to `Default` automatically.

Confirm the loaded module type:

```bash
modinfo nvidia | grep license   # need "NVIDIA"; "Dual MIT/GPL" = open = won't work
```

To switch module type (the installer at
`~/Downloads/NVIDIA-Linux-x86_64-570.172.08.run` builds either), from a text
console, then reboot:

```bash
sudo systemctl isolate multi-user.target          # drop to console
sudo sh ~/Downloads/NVIDIA-Linux-x86_64-570.172.08.run \
        --kernel-module-type=proprietary --silent --dkms
sudo reboot
```

Expected output: `TEST PASSED` (pattern `out[i]==2*i` validated end-to-end),
then a bandwidth table (memcpy vs read-kernel per size) and the FPGA→GPU figure
as a percentage of the measured pinned host→GPU ceiling.

## Acceptance gate

GPU reads the known pattern from FPGA HBM and validates it; sustained bandwidth
within ~10–15% of the single-hop ceiling. Because the GPU and FPGA sit under
different root complexes here, P2P bandwidth may be capped below the nominal
Gen3 x16 ceiling — the measured number **is** the topology reference this
milestone exists to produce.

## Troubleshooting

- **`cudaHostRegister(IoMemory)` fails**: peer routing between the GPU and FPGA
  root complexes is blocked. Check ACS (`lspci -vvv`), IOMMU state
  (`cat /proc/cmdline`), and that `xbutil examine -r platform` shows
  `P2P Status: enabled`.
- **`cudaMemcpyDefault` over the BAR errors/slow**: the program automatically
  falls back to a CUDA read-kernel (`out[i]=peer[i]`) and reports both paths.
- **xclbin rejected**: fall back to another U55C xclbin under
  `/opt/xilinx/firmware/u55c/gen3x16-xdma/base/test/`.
