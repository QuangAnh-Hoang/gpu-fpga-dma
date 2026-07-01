/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Milestone F1b — NVMeVirt -> userspace FPGA daemon forward ring (FQ/FCQ).
 *
 * NVMeVirt produces forward descriptors (a read to the FPGA-backed namespace)
 * onto the FQ and consumes completions from the FCQ; the userspace daemon
 * (gpu-fpga-dma/src/ise_dispatchd) does the opposite. The ring is a single
 * vmalloc_user region exposed mmap-able via /dev/nvmev_fpga.
 *
 * The struct layout here MUST stay byte-identical to the authoritative ABI in
 * gpu-fpga-dma/src/ise_fpga_ring.h (BUILD_BUG_ON in fpga_fwd.c asserts sizes).
 * This path uses NO nvidia_p2p — it only moves descriptors + posts the host-
 * resident NVMe CQ, preserving the D0 decoupling from the GPU driver.
 */
#ifndef NVMEV_FPGA_FWD_H
#define NVMEV_FPGA_FWD_H

#ifdef CONFIG_NVMEVIRT_FPGA
#include <linux/types.h>

#define ISF_MAGIC 0x49534651u /* 'I''S''F''Q' */
#define ISF_VERSION 1u
#define ISF_F_BATCH 0x0001u /* arg0 is a node-id list, count = list length */

/* FQ entry: one forwarded read. arg0 = first node id (plain) or list addr
 * (batch); count = n_rows (plain) or #ids (batch). */
struct ise_fwd_desc {
	__u64 req_id;
	__u64 arg0;
	__u32 count;
	__u32 row_bytes;
	__u64 gpu_dst; /* PRP1 GPU dest; 0 => daemon fixed window */
	__u32 sqid;
	__u16 cmd_id;
	__u16 flags;
};

/* FCQ entry: one completed request. offset = byte offset of this request's
 * row(s) in the GPU window ring (= slot * row_bytes), surfaced as NVMe result0;
 * lets the scheduler dedup/reorder rows (G1) — consumer locates data by offset. */
struct ise_fwd_cqe {
	__u64 req_id;
	__u32 offset;
	__u16 status;
	__u16 _pad;
};

/* Ring control header at offset 0 of the region. */
struct ise_fwd_ring {
	__u32 magic;
	__u32 version;
	__u32 fq_entries;
	__u32 fcq_entries;
	__u32 desc_size;
	__u32 cqe_size;
	__u32 fq_off;
	__u32 fcq_off;
	__u32 region_bytes;
	__u32 _rsvd;
	__u32 fq_head __attribute__((aligned(64)));  /* producer: NVMeVirt */
	__u32 fq_tail __attribute__((aligned(64)));  /* consumer: daemon   */
	__u32 fcq_head __attribute__((aligned(64))); /* producer: daemon   */
	__u32 fcq_tail __attribute__((aligned(64))); /* consumer: NVMeVirt */
};

struct nvmev_fpga {
	void *region; /* vmalloc_user'd ring region */
	size_t region_bytes;
	struct ise_fwd_ring *ring;
	bool ready;
};

/* lifecycle (called from NVMeV_init / NVMeV_exit) */
int nvmev_fpga_init(__u32 fq_entries, __u32 fcq_entries);
void nvmev_fpga_exit(void);
bool nvmev_fpga_ready(void);

/* FQ producer (NVMeVirt -> daemon): true if enqueued, false if full. */
bool nvmev_fpga_fq_push(const struct ise_fwd_desc *d);
/* FCQ consumer (daemon -> NVMeVirt): true if dequeued, false if empty. */
bool nvmev_fpga_fcq_pop(struct ise_fwd_cqe *c);

/* Pending NVMe command awaiting an FPGA completion (keyed by req_id). */
struct fpga_pend {
	__u64 req_id;
	__u32 sq_entry;
	__u16 sqid;
	__u16 cqid;
	__u16 cmd_id;
	__u16 busy;
};

/* Forward a read: allocate a pending slot and push the FQ descriptor.
 * Returns 0 on success, -EAGAIN (FQ full) or -EBUSY (pending table full). */
int nvmev_fpga_forward(__u16 sqid, __u16 cqid, __u32 sq_entry, __u16 cmd_id,
		       __u64 node_id, __u32 n_rows, __u32 row_bytes, __u64 gpu_dst,
		       __u16 flags);
/* Take (and free) the pending entry for a completed req_id; false if unknown. */
bool nvmev_fpga_pend_take(__u64 req_id, struct fpga_pend *out);

/* Drain the FCQ and post NVMe CQ entries (defined in io.c, needs __fill_cq_result).
 * Returns true if at least one command was completed. */
bool nvmev_fpga_complete_pending(void);

#endif /* CONFIG_NVMEVIRT_FPGA */
#endif /* NVMEV_FPGA_FWD_H */
