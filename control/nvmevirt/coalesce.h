/* SPDX-License-Identifier: GPL-2.0 */
/*
 * coalesce — L1-ring + dedicated sort-reduce cores (M1).
 *
 * Realizes the decided two-layer design (docs/nvmevirt-gnn-two-queue-model.md,
 * "Decided realization"):
 *
 *   GPU app --push node-id reqs--> [L1 ring x N shards]   (only interface)
 *                                        |
 *                          [N dedicated consumer kthreads in nvmev.ko]
 *                            page-keyed radix coalesce (STING), flush on
 *                            depth W / age T / shard-empty quiescence
 *                                        |
 *                          copy feature row ns->mapped --P2P--> VRAM
 *                            (nvmev_memcpy_to_gpu, GPUDirect RDMA)
 *                                        |
 *                          [completion ring] --> GPU aggregation kernel
 *
 * The app shards by page: it pushes request r into ring k = hash(line_tag)%N, so
 * every request for a page lands on one core -> that core coalesces its shard
 * with no locking and no missed coalescing. Region is one vmalloc_user block,
 * mmap-able via /dev/nvmev_l1; the ABI structs below must stay byte-identical to
 * the userspace mirror (sw/nvmev_l1.h).
 */
#ifndef NVMEV_COALESCE_H
#define NVMEV_COALESCE_H

#ifdef CONFIG_NVMEVIRT_FPGA
#include <linux/types.h>

struct proc_dir_entry;

#define NVMEV_L1_MAGIC   0x4e4c3151u /* 'NL1Q' */
#define NVMEV_L1_VERSION 1u

/* L1 request: GPU app -> module. dst_phys = the GPU BAR1 bus/physical address
 * where this node's feature row must land (BaM-style: the app resolves it from
 * the gpu_mem region page table, since VRAM pages are not phys-contiguous). One
 * row (<= 64 KB) always fits in a single registered page. */
struct l1_req {
	__u64 req_id;
	__u32 node_id;
	__u32 _pad0;
	__u64 dst_phys;
	__u16 flags;
	__u16 _pad1;
	__u32 _pad2;
};

/* Completion: module -> GPU aggregation kernel (req_id carries the app's slot). */
struct l1_cqe {
	__u64 req_id;
	__u16 status;
	__u16 _pad0;
	__u32 _pad1;
};

/* Per-shard ring control header (page-aligned at the start of each shard block).
 * rq is a bounded MPSC (Vyukov) ring: MANY GPU threads produce, the single
 * consumer kthread pops. Each rq slot has a sequence cell in rq_seq[] (init i);
 * producer reserves rq_head via atomic fetch-add, waits seq==pos, writes, sets
 * seq=pos+1; consumer reads when seq==pos+1, sets seq=pos+entries.
 * cq is SPSC: module produces (cq_head), GPU consumes (cq_tail). */
struct l1_ring {
	__u32 magic;
	__u32 version;
	__u32 shard;
	__u32 nr_shards;
	__u32 rq_entries; /* power of 2 */
	__u32 cq_entries; /* power of 2 */
	__u32 rq_off;	   /* byte offset of rq array from this shard base    */
	__u32 cq_off;	   /* byte offset of cq array from this shard base    */
	__u32 rq_seq_off;  /* byte offset of rq_seq[] (u32 per rq slot)       */
	__u32 shard_bytes;
	__u32 rq_head __attribute__((aligned(64))); /* producers reserve (MPSC) */
	__u32 rq_tail __attribute__((aligned(64))); /* consumer: module         */
	__u32 cq_head __attribute__((aligned(64))); /* producer: module         */
	__u32 cq_tail __attribute__((aligned(64))); /* consumer: GPU            */
};

/* Per-flush grouping algorithm (swept to measure sorter cost / ordering). */
enum co_sort { CO_SORT_RADIX = 0, CO_SORT_COMPARISON = 1, CO_SORT_HASH = 2 };

/* Lifecycle, called from NVMEV_STORAGE_INIT / _FINAL in main.c. Spawns the
 * consumer cores, allocates the ring region + /dev/nvmev_l1, and /proc counters. */
int nvmev_coalesce_init(struct proc_dir_entry *proc_root);
void nvmev_coalesce_exit(struct proc_dir_entry *proc_root);

#endif /* CONFIG_NVMEVIRT_FPGA */
#endif /* NVMEV_COALESCE_H */
