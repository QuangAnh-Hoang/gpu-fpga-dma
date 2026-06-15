// Read-path probe for the HOST[0] slave bridge (Milestone C debug).
//
// For each of 4 runtime offsets, performs: a 32-bit read (ctrl port), a
// single-beat 512-bit read (sq port), and a 4-beat 512-bit pipelined read
// (burst candidate), then writes one result beat to out[1+i]. A final marker
// beat goes to out[6]. Because results are written after each probe step, a
// hang pinpoints exactly which offset+shape stalled. ctrl/sq/out pointers and
// all offsets are runtime args: one bitstream probes any slot of any bo.

#include <ap_int.h>
#include <stdint.h>

typedef ap_uint<512> beat_t;

extern "C" {
void rd_probe(volatile uint32_t* ctrl, beat_t* sq, beat_t* out, uint64_t o0,
              uint64_t o1, uint64_t o2, uint64_t o3) {
#pragma HLS INTERFACE m_axi port = ctrl offset = slave bundle = ctrlb \
    num_read_outstanding = 2 max_read_burst_length = 2
#pragma HLS INTERFACE m_axi port = sq offset = slave bundle = sqb \
    num_read_outstanding = 8 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem \
    num_write_outstanding = 8 max_write_burst_length = 8
#pragma HLS INTERFACE s_axilite port = ctrl
#pragma HLS INTERFACE s_axilite port = sq
#pragma HLS INTERFACE s_axilite port = out
#pragma HLS INTERFACE s_axilite port = o0
#pragma HLS INTERFACE s_axilite port = o1
#pragma HLS INTERFACE s_axilite port = o2
#pragma HLS INTERFACE s_axilite port = o3
#pragma HLS INTERFACE s_axilite port = return

    uint64_t offs[4];
    offs[0] = o0;
    offs[1] = o1;
    offs[2] = o2;
    offs[3] = o3;

    // boot marker first: proves kernel started + out path works
    {
        beat_t m = 0;
        m.range(31, 0) = 0xB007B007u;
        out[0] = m;
    }

probe:
    for (int i = 0; i < 4; i++) {
#pragma HLS pipeline off
        uint64_t off = offs[i];

        // step 1: 32-bit read
        uint32_t w32 = ctrl[off / 4];
        {
            beat_t r = 0;
            r.range(31, 0) = 0xAAAA0001u;
            r.range(63, 32) = w32;
            out[1 + i] = r;  // partial: 32-bit done
        }

        // step 2: single-beat 512-bit read
        beat_t b1 = sq[off / 64];

        // step 3: 4-beat pipelined 512-bit read
        uint32_t acc = 0;
    burst:
        for (int k = 0; k < 4; k++) {
#pragma HLS pipeline II = 1
            acc ^= (uint32_t)sq[off / 64 + k].range(31, 0);
        }

        {
            beat_t r = 0;
            r.range(31, 0) = 0xAAAA0003u;  // all three shapes done
            r.range(63, 32) = w32;
            r.range(95, 64) = (uint32_t)b1.range(31, 0);
            r.range(127, 96) = acc;
            out[1 + i] = r;
        }
    }

    {
        beat_t m = 0;
        m.range(31, 0) = 0xD03E0D03u;  // all probes complete
        out[6] = m;
    }
}
}
