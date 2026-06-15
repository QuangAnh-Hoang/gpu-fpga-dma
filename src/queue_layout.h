// Milestone C — shared layout between the HLS gather-round kernel and the
// CUDA/host side.
//
// Doorbell-free round design (platform law: within one kernel execution, ULP
// re-reads of an address return only its first-read value, so in-kernel
// polling is impossible on this shell; everything is fresh across launches):
//   * one kernel launch = one round = one batch of node IDs;
//   * batch metadata travels as scalar kernel args;
//   * node IDs are written into the window BEFORE launch and bulk-read once;
//   * a round's output is capped at GNN_RING_BYTES; the consumer drains the
//     ring between rounds (no flow control inside a round).
//
// One 16 MB pinned GPU-VRAM window behind the HMSS (HOST[0]):
//   [0 .. GNN_RING_BYTES)   feature ring (round output, starts at 0 each round)
//   GNN_CQ_PROG_OFF   64B beat: [31:0]=bytes streamed so far [63:32]=batch_id
//   GNN_CQ_DONE_OFF   64B beat: [31:0]=batch_id [63:32]=total bytes
//                                [95:64]=0x600DD03E magic
//   GNN_SLOT_IDS_OFF  u32 ids[GNN_SQ_MAX_IDS]  (written pre-launch)
//
// FPGA HBM holds only the static feature table (filled before the session;
// pre-session writes are reliably visible to kernel reads).

#ifndef GNN_QUEUE_LAYOUT_H
#define GNN_QUEUE_LAYOUT_H

#include <stdint.h>

// ---- window geometry ----
#define GNN_WIN_BYTES (16u << 20)
#define GNN_RING_BYTES (8u << 20)

// ---- completion records (FPGA writes; GPU polls locally) ----
#define GNN_CQ_PROG_OFF (GNN_RING_BYTES + 64u)
#define GNN_CQ_DONE_OFF (GNN_RING_BYTES + 128u)
#define GNN_DONE_MAGIC 0x600DD03Eu

// ---- node-id list (GPU/host writes pre-launch, FPGA bulk-reads once) ----
#define GNN_SLOT_IDS_OFF (GNN_RING_BYTES + (1u << 20))  // window + 9 MB
#define GNN_SQ_MAX_IDS 65536u                           // 256 KB (fits URAM)

// ---- streaming ----
#define GNN_CHUNK_ROWS 256u  // progress-record granularity (rows)

// Deterministic synthetic feature: word d of node id's row. The host fills the
// HBM table with this; the GPU validator recomputes it for exact comparison.
#if defined(__CUDACC__)
__host__ __device__
#endif
static inline uint32_t gnn_feat_word(uint32_t id, uint32_t d) {
    return (id * 2654435761u) ^ (d * 40503u) ^ 0x9E3779B9u;
}

#endif  // GNN_QUEUE_LAYOUT_H
