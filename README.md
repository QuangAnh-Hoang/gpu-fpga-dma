# gpu-fpga-dma — Milestone A: GPU pulls from FPGA HBM

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
