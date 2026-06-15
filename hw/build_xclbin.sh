#!/usr/bin/env bash
# Build gdr_write.xclbin for the U55C (HMSS slave-bridge target).
# Long-running (~1-2 h) due to place & route.
set -euo pipefail

source /tools/Xilinx/Vitis/2024.1/settings64.sh

HW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLATFORM=xilinx_u55c_gen3x16_xdma_3_202210_1
KERNEL="${1:-gdr_write}"
BUILD="${HW_DIR}/build_${KERNEL}"
mkdir -p "$BUILD"
cd "$BUILD"

echo "[1/2] v++ compile (.xo) ..."
v++ -c -t hw --platform "$PLATFORM" -k "$KERNEL" \
    -o "${KERNEL}.xo" "${HW_DIR}/${KERNEL}.cpp" \
    --save-temps --log_dir "${BUILD}/logs" --report_dir "${BUILD}/reports"

echo "[2/2] v++ link (.xclbin) ..."
v++ -l -t hw --platform "$PLATFORM" \
    --config "${HW_DIR}/${KERNEL}.cfg" \
    -o "${KERNEL}.xclbin" "${KERNEL}.xo" \
    --save-temps --log_dir "${BUILD}/logs" --report_dir "${BUILD}/reports"

echo "DONE: ${BUILD}/${KERNEL}.xclbin"
