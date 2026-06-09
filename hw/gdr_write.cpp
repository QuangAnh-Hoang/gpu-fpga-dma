// Milestone B — FPGA-as-DMA-master kernel (512-bit, deep outstanding writes).
//
// A single AXI master (`m_axi_gmem`) writes a known pattern (out[i] = seed + i,
// per 32-bit word) to its output pointer. At link time that master is wired to
// HOST[0] (the XDMA shell's slave bridge / HMSS), so every write becomes a PCIe
// memory-write TLP whose destination is set by the HMSS page table. Our driver
// programs that page table with GPU BAR bus addresses, so the writes land in GPU
// VRAM.
//
// The interface is 512-bit wide (16 x uint32 per beat) and pipelined II=1, with a
// deep write-outstanding / burst config so the master can saturate the write
// path. 512-bit x 300 MHz = 19.2 GB/s theoretical ceiling.
//
// Arg layout (pointer out, uint32 seed, int n_elements) is unchanged from the
// 32-bit version, so the host program, HMSS slot programming, and HOST[0]
// connectivity are all unchanged. n_elements is still the number of 32-bit words.

#include <ap_int.h>
#include <stdint.h>

typedef ap_uint<512> wide_t;  // 16 x uint32 per beat

extern "C" {
void gdr_write(wide_t* out, uint32_t seed, int n_elements) {
#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem \
    num_write_outstanding = 32 num_read_outstanding = 2                 \
    max_write_burst_length = 64 max_read_burst_length = 2
#pragma HLS INTERFACE s_axilite port = out
#pragma HLS INTERFACE s_axilite port = seed
#pragma HLS INTERFACE s_axilite port = n_elements
#pragma HLS INTERFACE s_axilite port = return

    int n_beats = n_elements >> 4;  // 16 words per 512-bit beat

write_beats:
    for (int w = 0; w < n_beats; w++) {
#pragma HLS pipeline II = 1
        wide_t beat;
        uint32_t base = seed + (uint32_t)(w << 4);
        for (int l = 0; l < 16; l++) {
#pragma HLS unroll
            beat.range(32 * l + 31, 32 * l) = base + (uint32_t)l;  // out[w*16+l]
        }
        out[w] = beat;
    }
}
}
