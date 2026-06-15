// Milestone C — gather-round kernel: the FPGA side of the GPU-request →
// FPGA-response queue pair, restructured as doorbell-free rounds.
//
// Platform law (established empirically with rd_probe/prestart experiments on
// this U55C shell): within one kernel execution, a ULP master that re-reads an
// address observes only its first-read value — host/GPU updates made after
// kernel start never become visible (HBM or slave-bridge paths alike). Across
// kernel executions everything is fresh. Therefore NO in-kernel polling:
//
//   * batch metadata (n_ids, row_beats, batch_id) arrives as SCALAR args;
//   * one launch processes one round; the host/GPU re-launches per round
//     (~100us, amortized over the round);
//   * all node IDs are bulk-read ONCE from the GPU-VRAM window into URAM
//     before any streaming, so bridge reads and bridge writes never overlap;
//   * no flow control: a round is capped at GNN_RING_BYTES; the consumer
//     drains the ring between rounds.
//
// Data plane unchanged: random row reads from the static HBM feature table
// (written pre-session), packed sequential 512-bit writes into the pinned
// GPU-VRAM ring through HOST[0] (Milestone B, ~12.9 GB/s), DONE record last
// (same AXI port => PCIe ordering makes it safe to poll from the GPU).

#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>

#include "../src/queue_layout.h"

typedef ap_uint<512> beat_t;

#define B(off) ((off) / 64)  // byte offset -> beat index

static void read_rows(beat_t* feat, const uint32_t* ids, int rows, int row_beats,
                      hls::stream<beat_t>& s) {
rows_loop:
    for (int r = 0; r < rows; r++) {
        uint64_t base = (uint64_t)ids[r] * (uint32_t)row_beats;
    beats_loop:
        for (int b = 0; b < row_beats; b++) {
#pragma HLS pipeline II = 1
            s << feat[base + b];
        }
    }
}

static void write_ring(beat_t* out, uint64_t wr_beat, int n_beats,
                       hls::stream<beat_t>& s) {
write_loop:
    for (int i = 0; i < n_beats; i++) {
#pragma HLS pipeline II = 1
        out[wr_beat + i] = s.read();
    }
}

static void gather_chunk(beat_t* feat, beat_t* out, const uint32_t* ids, int rows,
                         int row_beats, uint64_t wr_beat) {
#pragma HLS dataflow
    hls::stream<beat_t> s("rowstream");
#pragma HLS stream variable = s depth = 64
    read_rows(feat, ids, rows, row_beats, s);
    write_ring(out, wr_beat, rows * row_beats, s);
}

extern "C" {
void gnn_gather(beat_t* sq,       // window: node-id list (read once, upfront)
                beat_t* feat,     // HBM: static fixed-stride feature rows
                beat_t* out,      // window: ring + CQ records
                uint32_t batch_id,
                uint32_t n_ids,      // number of rows to gather (<= GNN_SQ_MAX_IDS)
                uint32_t row_beats)  // 64B beats per feature row
{
#pragma HLS INTERFACE m_axi port = sq offset = slave bundle = sqb \
    num_read_outstanding = 8 max_read_burst_length = 64
#pragma HLS INTERFACE m_axi port = feat offset = slave bundle = featb \
    num_read_outstanding = 32 max_read_burst_length = 32
#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem \
    num_write_outstanding = 32 max_write_burst_length = 64
#pragma HLS INTERFACE s_axilite port = sq
#pragma HLS INTERFACE s_axilite port = feat
#pragma HLS INTERFACE s_axilite port = out
#pragma HLS INTERFACE s_axilite port = batch_id
#pragma HLS INTERFACE s_axilite port = n_ids
#pragma HLS INTERFACE s_axilite port = row_beats
#pragma HLS INTERFACE s_axilite port = return

    // Phase 1: bulk-load ALL ids into URAM (bridge reads only, long bursts).
    static uint32_t ids_buf[GNN_SQ_MAX_IDS];  // 64K x 4B = 256 KB
#pragma HLS bind_storage variable = ids_buf type = ram_2p impl = uram

    uint32_t n_id_beats = (n_ids + 15) / 16;
load_ids:
    for (uint32_t i = 0; i < n_id_beats; i++) {
#pragma HLS pipeline II = 1
        beat_t b = sq[B(GNN_SLOT_IDS_OFF) + i];
    unpack:
        for (int l = 0; l < 16; l++) {
#pragma HLS unroll
            uint32_t idx = i * 16 + l;
            if (idx < GNN_SQ_MAX_IDS) ids_buf[idx] = (uint32_t)b.range(32 * l + 31, 32 * l);
        }
    }

    // Phase 2: gather + stream (bridge writes only; feat reads are HBM-side).
    uint32_t row_bytes = row_beats * 64;
    uint32_t done_rows = 0;
    uint32_t wr_total = 0;
chunks:
    while (done_rows < n_ids) {
        uint32_t rows = n_ids - done_rows;
        if (rows > GNN_CHUNK_ROWS) rows = GNN_CHUNK_ROWS;
        gather_chunk(feat, out, &ids_buf[done_rows], rows, row_beats,
                     (uint64_t)wr_total / 64);
        wr_total += rows * row_bytes;
        done_rows += rows;

        // progress record (after its data; same port => ordered)
        beat_t pg = 0;
        pg.range(31, 0) = wr_total;
        pg.range(63, 32) = batch_id;
        out[B(GNN_CQ_PROG_OFF)] = pg;
    }

    // DONE record
    beat_t dn = 0;
    dn.range(31, 0) = batch_id;
    dn.range(63, 32) = wr_total;
    dn.range(95, 64) = 0x600DD03Eu;
    out[B(GNN_CQ_DONE_OFF)] = dn;
}
}
