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

#ifdef CONFIG_NVMEVIRT_MIDLAYER
#include <linux/types.h>
#include <linux/ioctl.h>

struct proc_dir_entry;

/* VA->dst_phys translate ioctl on /dev/nvmev_l1 (M6). Given a registered VRAM
 * range, returns its GPU-page bus addresses so userspace (libisefs) can resolve
 * each feature row's dst_phys without parsing /proc/nvmev/gpu_mem. Mirrored in
 * sw/nvmev_l1.h. */
struct l1_translate {
	__u64 gpu_va;	/* in: start of the registered VRAM range          */
	__u64 len;	/* in: byte length                                 */
	__u64 phys;	/* in: user ptr to __u64[n_pages] (out: bus addrs) */
	__u32 n_pages;	/* in: capacity / out: pages written               */
	__u32 page_shift;/* out: GPU page shift (16 => 64 KB)              */
};
#define NVMEV_L1_IOC_TRANSLATE _IOWR('L', 1, struct l1_translate)

/* GPU-pull staging geometry (b2). The kernel exposes one contiguous, page-
 * granular staging region (all shards) at mmap file offset NVMEV_L1_MMAP_STAGING
 * so userspace can mmap + cudaHostRegister it once and the GPU-pull kernel can
 * read any feature row by a single byte offset. A row for (shard, slot) lives at
 * ((shard*pend_cap)+slot)*page; descriptors (b3) carry the absolute offset. */
struct l1_staging {
	__u64 mmap_off;		/* out: mmap() file offset of the staging region */
	__u64 bytes;		/* out: total region size                        */
	__u32 nshards;		/* out: shard count                              */
	__u32 pend_cap;		/* out: staging slots per shard                  */
	__u32 page;		/* out: staging page (bytes)                     */
	__u32 _pad;
};
#define NVMEV_L1_IOC_STGINFO _IOR('L', 2, struct l1_staging)

/* GPU-pull descriptor ring (b3). In gpu-pull delivery (coalesce_deliver=1) the
 * kernel does NOT WC-store rows or post CQEs; per finished row it enqueues one
 * gather descriptor here, and the GPU persistent kernel (b4) reads the row from
 * staging into VRAM and writes the l1_cqe itself. Per-shard SPSC ring (kernel
 * produces `head`, GPU consumes `tail`), mirror of the L1 ring layout. */
struct l1_gdesc {		/* one gather descriptor (32 B) */
	__u64 staging_off;	/* src: byte offset into the staging region      */
	__u64 dst;		/* dst: l1_req.dst_phys passthrough (GPU interprets) */
	__u32 req_id;		/* -> l1_cqe.req_id the GPU posts                */
	__u16 len;		/* = coalesce_row                                */
	__u16 status;		/* 0 = gather+deliver; !=0 = error (cqe only)    */
	__u64 _pad;
};
struct l1_gring {		/* per-shard descriptor-ring header (in g_desc) */
	__u32 magic;
	__u32 entries;		/* ring capacity (power of two)                  */
	__u32 desc_off;		/* byte offset (within this shard slice) to l1_gdesc[] */
	__u32 shard;
	__u32 head;		/* producer cursor (kernel)                      */
	__u32 tail;		/* consumer cursor (GPU) — flow control          */
	__u32 shard_bytes;	/* per-shard slice size                          */
	__u32 _pad;
};
struct l1_descinfo {		/* DESCINFO ioctl */
	__u64 mmap_off;		/* out: mmap() file offset of the descriptor region */
	__u64 bytes;		/* out: total region size                        */
	__u32 nshards;		/* out                                           */
	__u32 entries;		/* out: per-shard ring capacity                  */
	__u32 shard_bytes;	/* out: per-shard slice size                     */
	__u32 _pad;
};
#define NVMEV_L1_IOC_DESCINFO _IOR('L', 3, struct l1_descinfo)

/* mmap file offsets selecting each GPU-accessible region (0 = the L1 rings). */
#define NVMEV_L1_MMAP_STAGING (1ULL << 34)
#define NVMEV_L1_MMAP_DESC    (2ULL << 34)
#define NVMEV_L1_GRING_MAGIC  0x4e4c3147u /* 'NL1G' */

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

/* Per-flush grouping algorithm (swept to measure sorter cost / ordering).
 * RADIX/COMPARISON/HASH are exact global window dedup (the upper-bound
 * reference). BUCKET is the two-stage RTL oracle (BLRadix online bucketer +
 * bounded per-burst MergeSortReducer): online binning, arrival-adjacent
 * pre-filter, half-capacity burst eviction, dedup bounded to one burst — a
 * weaker, arrival-order-dependent coalescer that matches the hardware. */
enum co_sort { CO_SORT_RADIX = 0, CO_SORT_COMPARISON = 1, CO_SORT_HASH = 2,
	       CO_SORT_BUCKET = 3 };

/* Lifecycle, called from NVMEV_STORAGE_INIT / _FINAL in main.c. Spawns the
 * consumer cores, allocates the ring region + /dev/nvmev_l1, and /proc counters. */
int nvmev_coalesce_init(struct proc_dir_entry *proc_root);
void nvmev_coalesce_exit(struct proc_dir_entry *proc_root);

#endif /* CONFIG_NVMEVIRT_MIDLAYER */
#endif /* NVMEV_COALESCE_H */
