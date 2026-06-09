#!/usr/bin/env bash
# Milestone B — Phase 0 spike orchestration.
# Proves the FPGA can DMA a known pattern into GPU VRAM via the HMSS slave bridge.
#
# Requires sudo (insmod + xbutil configure). Run from the repo root.
set -euo pipefail

source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XCLBIN="${XCLBIN:-${REPO}/hw/build/gdr_write.xclbin}"
HOSTMEM_MB="${HOSTMEM_MB:-256}"
BDF="${BDF:-0000:21:00.1}"

[[ -f "$XCLBIN" ]] || { echo "xclbin not found: $XCLBIN (still building?)"; exit 1; }
[[ -x "${REPO}/build/gpu_push_spike" ]] || { echo "build first: cmake --build build"; exit 1; }

echo "== (1) load fpga_gdr.ko (nvidia must be loaded) =="
if ! lsmod | grep -q '^fpga_gdr'; then
  sudo insmod "${REPO}/driver/fpga_gdr.ko"
fi
sudo chmod 666 /dev/fpga_gdr

echo "== (2) enable HMSS host memory (${HOSTMEM_MB} MB) =="
sudo /opt/xilinx/xrt/bin/xbutil configure --host-mem -d "$BDF" -s "${HOSTMEM_MB}M" ENABLE
/opt/xilinx/xrt/bin/xbutil examine -d "$BDF" -r all 2>/dev/null | grep -iE 'host mem' || true

echo "== (3) run spike =="
"${REPO}/build/gpu_push_spike" -x "$XCLBIN" -d 0
rc=$?

echo "== dmesg tail =="
sudo dmesg | tail -8 | grep -iE 'fpga_gdr|p2p' || true
exit $rc
