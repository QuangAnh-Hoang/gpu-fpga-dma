/* SPDX-License-Identifier: GPL-2.0 */
/*
 * libisefs — userspace feature-store transport over the NVMeVirt L1 ring
 * (/dev/nvmev_l1). Wraps what sw/l1_gnn_probe.cu does by hand into a small,
 * reusable API for a DGL/PyG FeatureStore adapter (assessment §3.3.7, DD1):
 *
 *   isefs_open        open + mmap the L1 ring region
 *   isefs_set_dst     pin a VRAM minibatch buffer + resolve per-row bus addrs
 *   isefs_fetch       async: request feature[node_id[i]] -> dst row i
 *   isefs_poll        drain completions (row indices that landed)
 *   isefs_close       unpin + close
 *
 * The transport is the frozen L1 ABI (sw/nvmev_l1.h); the same code will drive
 * an FPGA consumer unchanged. Host (CPU) producer path; a GPU-push kernel can
 * use l1_push_req_dev() from nvmev_l1.h directly against the same rings.
 *
 * Requires CONFIG_NVMEVIRT_GPU_DIRECT (VRAM delivery) and permission to write
 * /proc/nvmev/gpu_mem (pinning) — typically run as root.
 */
#ifndef LIBISEFS_H
#define LIBISEFS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct isefs isefs_t;

#define ISEFS_MAX_DST 8	/* destination-buffer slots (for prefetch pipelining) */

/* row_bytes / page_bytes must match the module's coalesce_row / coalesce_page. */
isefs_t *isefs_open(const char *l1_dev, unsigned row_bytes, unsigned page_bytes);
void	 isefs_close(isefs_t *h);

/* Register a destination VRAM buffer in `slot` (0..ISEFS_MAX_DST-1). Multiple
 * slots let the caller double/triple-buffer: fetch batch k+1 into one slot while
 * batch k is still being consumed from another. dst_gpu_va = CUDA device pointer
 * cast to u64; capacity_rows = rows it holds. Pins + resolves bus addrs. 0/-1. */
int isefs_set_dst(isefs_t *h, unsigned slot, uint64_t dst_gpu_va, size_t capacity_rows);

/* Async fetch into `slot`: for i in [0,n) request feature[node_ids[i]] delivered
 * to slot-row (dst_off+i). Completion req_id = req_base + i. `dst_off` lets a
 * large batch be pushed in bounded chunks (each chunk targets the right rows)
 * while draining completions between chunks — required to avoid overflowing the
 * completion ring on big batches. Non-blocking per call. */
int isefs_fetch(isefs_t *h, unsigned slot, const uint32_t *node_ids, size_t n,
		uint64_t req_base, size_t dst_off);

/* Non-blocking: fill done[] with up to `max` completed req_ids; returns count. */
int isefs_poll(isefs_t *h, uint64_t *done, size_t max);

/* Convenience: single-slot (slot 0, req_base 0) fetch that blocks until all n
 * have completed. set_dst(h,0,..) must have been called. */
int isefs_fetch_wait(isefs_t *h, const uint32_t *node_ids, size_t n);

unsigned isefs_nr_shards(const isefs_t *h);

#ifdef __cplusplus
}
#endif
#endif /* LIBISEFS_H */
