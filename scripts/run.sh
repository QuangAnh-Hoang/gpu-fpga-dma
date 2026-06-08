#!/usr/bin/env bash
# Milestone A runner: GPU pulls a known pattern from FPGA HBM over PCIe P2P.
set -euo pipefail

# XRT runtime environment
source /opt/xilinx/xrt/setup.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${REPO_ROOT}/build/gpu_pull"

# Prebuilt U55C vadd.xclbin (HBM output bank, VBNV matches the loaded shell).
XCLBIN="${XCLBIN:-/home/hoangqa/workspace/Vitis_Accel_Examples/hello_world/build_dir.hw.xilinx_u55c_gen3x16_xdma_3_202210_1/vadd.xclbin}"

DEVICE_ID="${DEVICE_ID:-0}"
# 128 MB: vadd's gmem0 HBM bank (512 MB) holds both an input and the P2P output.
MAX_MB="${MAX_MB:-128}"

if [[ ! -x "$BIN" ]]; then
  echo "Build first: cmake -B build && cmake --build build -j" >&2
  exit 1
fi
if [[ ! -f "$XCLBIN" ]]; then
  echo "xclbin not found: $XCLBIN (override with XCLBIN=...)" >&2
  exit 1
fi

exec "$BIN" -x "$XCLBIN" -d "$DEVICE_ID" -m "$MAX_MB"
