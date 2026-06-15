#!/usr/bin/env bash
# Milestone C — run the GNN feature-gather queue-pair demo.
# Prereqs (once, needs sudo): scripts/spike.sh   (loads fpga_gdr.ko + host-mem)
set -euo pipefail
source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XCLBIN="${XCLBIN:-${REPO}/hw/build_gnn_gather/gnn_gather.xclbin}"
N_NODES="${N_NODES:-2097152}"     # 2M nodes
ROW_BYTES="${ROW_BYTES:-512}"     # 128-dim fp32
SUBMIT="${SUBMIT:-gpu}"           # gpu | host

[[ -f "$XCLBIN" ]] || { echo "xclbin not found: $XCLBIN"; exit 1; }
exec "${REPO}/build/gnn_demo" -x "$XCLBIN" -n "$N_NODES" -r "$ROW_BYTES" -S "$SUBMIT"
