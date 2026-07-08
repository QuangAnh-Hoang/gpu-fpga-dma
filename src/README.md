# ISEFS — feature-store library for the ISE middle layer

This directory is the **reusable userspace library** for driving the ISE middle layer in
[`../control/`](../control/). A GNN training loop hands it node IDs; it pushes them through
the kernel middle layer over `/dev/nvmev_l1`, the middle layer coalesces the storage reads,
and feature rows land P2P in GPU VRAM. The training side sees only the **frozen L1 ring
ABI**, so the kernel backend (emulated DRAM, real NVMe, or a future FPGA) can change
underneath without touching a line of application code.

Pair this with `../control/` (the kernel module) and you have everything needed to reuse the
project: load `nvmev.ko`, build the two shared libs here, and fetch features from Python.

## Contents

| File | Role |
|------|------|
| [`isefs.py`](isefs.py) | The Python API — `ISEFS` (CPU-deliver), `ISEFS_GPUPull` (GPU-pull), and `ISE_DGLDataLoader`, a drop-in DGL `DataLoader` whose batches carry the fetched features. This is the entry point. |
| [`isefs.c`](isefs.c) / [`isefs.h`](isefs.h) | `libisefs.so`: opens + mmaps the L1 ring, pins a VRAM destination, and issues fetches (`isefs_fetch` / `isefs_poll` / `isefs_fetch_wait`). Loaded by `isefs.py` via ctypes. |
| [`isefs_gpupull.cu`](isefs_gpupull.cu) | `libisefs_gpupull.so`: the persistent GPU gather kernel that pulls coalesced rows from host staging into VRAM (the `coalesce_deliver=1` delivery plane). |
| [`nvmev_l1.h`](nvmev_l1.h) | The L1 ring + ioctl **ABI** — a byte-for-byte mirror of [`../control/nvmevirt/coalesce.h`](../control/nvmevirt/coalesce.h). Included by `isefs.c` and `isefs_gpupull.cu`. |
| [`Makefile`](Makefile) | Builds the two shared libs into `obj/`. |
| [`Makefile.local`](Makefile.local) | Host-specific `NVCC` path, `include`d by the Makefile — adjust to your CUDA install. |

### The one hard contract

`nvmev_l1.h` **must stay byte-identical** to `../control/nvmevirt/coalesce.h` (the kernel
enforces this with `BUILD_BUG_ON`). The two headers move together; if you change the ring
layout on one side, change it on the other. Everything else here is ordinary userspace.

Also: `coalesce_row` (`/sys/module/nvmev/parameters/coalesce_row`) **must equal**
`feat_dim * dtype.itemsize`, and feature *n* must be stored on the backend at byte offset
`n * row`. A row never straddles a page.

## Build

```sh
# 1. Kernel side first (see ../control/README.md):
make -C ../control build && sudo make -C ../control reload

# 2. The libraries, into src/obj/:
make -C . isefs           # obj/libisefs.so         (CPU-deliver transport)
make -C . isefs_gpupull   # obj/libisefs_gpupull.so (GPU-pull kernel; needs CUDA)
```

`isefs.py` loads `obj/libisefs.so` and `obj/libisefs_gpupull.so` by path relative to itself,
so keep the built `obj/` next to these sources.

- **`Makefile.local`** sets `NVCC` (defaults to `/usr/local/cuda/bin/nvcc`). Edit it for your
  CUDA path. The Makefile's default target is a legacy `nvme_probe` stub — always name a
  target explicitly (`make isefs` / `make isefs_gpupull`).
- **Runtime Python deps:** `numpy`, `torch`, and `dgl` (for `ISE_DGLDataLoader`).

## Quick use (Python)

Run as root (pinning VRAM writes `/proc/nvmev/gpu_mem`) with the module loaded and
`coalesce_enable=1`:

```python
import torch
from isefs import ISEFS

fs = ISEFS(feat_dim=128)                 # row = 128 * 4B = 512B; sets coalesce_row
ids = torch.randint(0, 8192, (2048,))    # node IDs for one minibatch
rows = fs.fetch(ids)                      # -> [2048, 128] tensor in GPU VRAM
fs.close()
```

For training, wrap a DGL sampler so each batch arrives with its features already fetched:

```python
from isefs import ISE_DGLDataLoader
loader = ISE_DGLDataLoader(graph, train_nids, sampler, batch_size, dim=128)
for input_nodes, output_nodes, blocks, feat in loader:
    ...  # feat is the fetched feature tensor
```

Swap `ISEFS` for `ISEFS_GPUPull` (module loaded `coalesce_deliver=1`) to use the GPU-pull
delivery plane — same interface, higher delivery bandwidth.

