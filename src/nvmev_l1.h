/* SPDX-License-Identifier: GPL-2.0 */
/*
 * nvmev_l1.h — userspace/GPU mirror of the NVMeVirt L1-ring ABI.
 * MUST stay byte-identical to control/nvmevirt/coalesce.h (BUILD_BUG_ON kernel).
 * rq is a bounded MPSC (Vyukov) ring: many producers (GPU threads or CPU
 * threads), single consumer (the module kthread). cq is SPSC.
 */
#ifndef NVMEV_L1_USER_H
#define NVMEV_L1_USER_H

#include <stdint.h>
#include <stddef.h>
#include <sys/ioctl.h>

/* VA->dst_phys translate ioctl on /dev/nvmev_l1 (M6) — mirror of
 * control/nvmevirt/coalesce.h. Given a registered VRAM range, returns its
 * GPU-page bus addresses so libisefs can resolve each row's dst_phys. */
struct l1_translate {
	uint64_t gpu_va;	/* in                                       */
	uint64_t len;		/* in                                       */
	uint64_t phys;		/* in: ptr to uint64_t[n_pages] (filled out) */
	uint32_t n_pages;	/* in: capacity / out: count                */
	uint32_t page_shift;	/* out: 16 => 64 KB GPU pages               */
};
#define NVMEV_L1_IOC_TRANSLATE _IOWR('L', 1, struct l1_translate)

/* GPU-pull staging geometry (b2), mirror of control/nvmevirt/coalesce.h. */
struct l1_staging {
	uint64_t mmap_off;	/* out: mmap() file offset of the staging region */
	uint64_t bytes;		/* out: total region size                        */
	uint32_t nshards;	/* out                                           */
	uint32_t pend_cap;	/* out: staging slots per shard                  */
	uint32_t page;		/* out: staging page (bytes)                     */
	uint32_t _pad;
};
#define NVMEV_L1_IOC_STGINFO _IOR('L', 2, struct l1_staging)
#define NVMEV_L1_MMAP_STAGING (1ULL << 34)

/* GPU-pull descriptor ring (b3), mirror of control/nvmevirt/coalesce.h. */
struct l1_gdesc {		/* 32 B gather descriptor */
	uint64_t staging_off;	/* src: byte offset into the staging region      */
	uint64_t dst;		/* dst: l1_req.dst_phys passthrough              */
	uint32_t req_id;
	uint16_t len;		/* = row bytes                                   */
	uint16_t status;	/* 0 = gather+deliver; !=0 = error (cqe only)    */
	uint64_t _pad;
};
struct l1_gring {		/* per-shard descriptor-ring header */
	uint32_t magic, entries, desc_off, shard;
	uint32_t head;		/* producer (kernel)                             */
	uint32_t tail;		/* consumer (GPU / this probe)                   */
	uint32_t shard_bytes, _pad;
};
struct l1_descinfo {
	uint64_t mmap_off;
	uint64_t bytes;
	uint32_t nshards, entries, shard_bytes, _pad;
};
#define NVMEV_L1_IOC_DESCINFO _IOR('L', 3, struct l1_descinfo)
#define NVMEV_L1_MMAP_DESC    (2ULL << 34)
#define NVMEV_L1_GRING_MAGIC  0x4e4c3147u

/* descriptor ring accessors (mirror l1_shard/l1_rq for g_desc). */
static inline struct l1_gring *l1_gshard(void *desc_region, unsigned k,
					 uint32_t shard_bytes)
{
	return (struct l1_gring *)((uint8_t *)desc_region + (size_t)k * shard_bytes);
}
static inline struct l1_gdesc *l1_gdescs(struct l1_gring *g)
{
	return (struct l1_gdesc *)((uint8_t *)g + g->desc_off);
}

#define NVMEV_L1_MAGIC   0x4e4c3151u
#define NVMEV_L1_VERSION 1u

struct l1_req { uint64_t req_id; uint32_t node_id; uint32_t _pad0; uint64_t dst_phys; uint16_t flags; uint16_t _pad1; uint32_t _pad2; };
struct l1_cqe { uint64_t req_id; uint16_t status; uint16_t _pad0; uint32_t _pad1; };

struct l1_ring {
	uint32_t magic, version, shard, nr_shards;
	uint32_t rq_entries, cq_entries, rq_off, cq_off, rq_seq_off, shard_bytes;
	uint32_t rq_head __attribute__((aligned(64)));
	uint32_t rq_tail __attribute__((aligned(64)));
	uint32_t cq_head __attribute__((aligned(64)));
	uint32_t cq_tail __attribute__((aligned(64)));
};

#ifdef __CUDACC__
#define L1_HD __host__ __device__
#else
#define L1_HD static inline
#endif

L1_HD struct l1_ring *l1_shard(void *region, unsigned k)
{
	struct l1_ring *r0 = (struct l1_ring *)region;
	return (struct l1_ring *)((uint8_t *)region + (size_t)k * r0->shard_bytes);
}
L1_HD struct l1_req *l1_rq(struct l1_ring *r) { return (struct l1_req *)((uint8_t *)r + r->rq_off); }
L1_HD struct l1_cqe *l1_cq(struct l1_ring *r) { return (struct l1_cqe *)((uint8_t *)r + r->cq_off); }
L1_HD uint32_t *l1_rq_seq(struct l1_ring *r) { return (uint32_t *)((uint8_t *)r + r->rq_seq_off); }

L1_HD unsigned l1_shard_of(uint32_t node_id, unsigned row, unsigned page, unsigned nr_shards)
{
	uint64_t line_tag = ((uint64_t)node_id * row) & ~((uint64_t)page - 1);
	uint64_t h = line_tag * 0x9e3779b97f4a7c15ULL;
	return (unsigned)((h >> 40) % nr_shards);
}

/* ---- host producer/consumer (host code; nvcc compiles these host-only) ---- */
/* MPSC producer: reserve, wait for the cell, write, publish. Blocks if full. */
static inline void l1_push_req(struct l1_ring *r, const struct l1_req *req)
{
	uint32_t *seq = l1_rq_seq(r);
	uint32_t pos = __atomic_fetch_add(&r->rq_head, 1, __ATOMIC_RELAXED);
	uint32_t cell = pos & (r->rq_entries - 1);
	while (__atomic_load_n(&seq[cell], __ATOMIC_ACQUIRE) != pos)
		;
	l1_rq(r)[cell] = *req;
	__atomic_store_n(&seq[cell], pos + 1, __ATOMIC_RELEASE);
}

/* SPSC consumer of cq (single drainer). Returns 0 on success, -1 if empty. */
static inline int l1_pop_cqe(struct l1_ring *r, struct l1_cqe *out)
{
	uint32_t h = __atomic_load_n(&r->cq_head, __ATOMIC_ACQUIRE);
	uint32_t t = r->cq_tail;
	if (h == t)
		return -1;
	*out = l1_cq(r)[t & (r->cq_entries - 1)];
	__atomic_store_n(&r->cq_tail, t + 1, __ATOMIC_RELEASE);
	return 0;
}

/* ---- device producer (GPU push kernel) ---- */
#ifdef __CUDACC__
/* MPSC producer from GPU threads into host-pinned ring memory. Uses system-scope
 * atomics + fences so the CPU consumer sees the writes. */
__device__ static inline void l1_push_req_dev(struct l1_ring *r, const struct l1_req *req)
{
	volatile uint32_t *seq = l1_rq_seq(r);
	uint32_t pos = atomicAdd_system(&r->rq_head, 1u);
	uint32_t cell = pos & (r->rq_entries - 1);
	while (seq[cell] != pos)       /* wait for the cell's previous lap to drain */
		;
	l1_rq(r)[cell] = *req;
	__threadfence_system();        /* publish the payload before the seq flag */
	seq[cell] = pos + 1;
}
#endif /* __CUDACC__ */

#endif /* NVMEV_L1_USER_H */
