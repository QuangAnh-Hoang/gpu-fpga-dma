// SPDX-License-Identifier: GPL-2.0
/*
 * coalesce — L1-ring + dedicated sort-reduce cores (M1). See coalesce.h.
 *
 * N dedicated kthreads each own one L1 ring shard (no locking — the GPU shards
 * by page so all requests for a page reach one core). A consumer drains its
 * shard into a window, flushes on depth W / age T / shard-empty quiescence,
 * groups by page (radix | comparison | hash — swept to measure sorter cost),
 * models one backend read per unique page, delivers each waiter's feature row
 * ns->mapped -> VRAM via GPUDirect RDMA, and posts a completion.
 */
#ifdef CONFIG_NVMEVIRT_MIDLAYER

#include <linux/atomic.h>
#include <linux/delay.h>
#include <linux/fs.h>
#include <linux/kthread.h>
#include <linux/log2.h>
#include <linux/miscdevice.h>
#include <linux/mm.h>
#include <linux/moduleparam.h>
#include <linux/proc_fs.h>
#include <linux/sched/clock.h>
#include <linux/seq_file.h>
#include <linux/slab.h>
#include <linux/sort.h>
#include <linux/string.h>
#include <linux/version.h>
#include <linux/vmalloc.h>

#include "nvmev.h"
#include "gpu_mem.h"
#include "gpu_memcpy.h"
#include "coalesce.h"
#include "coalesce_backend.h"

#ifdef CONFIG_NVMEVIRT_GPU_DIRECT
#include <asm/fpu/api.h>
#endif

/* ---- tunables (swept) ---- */
static unsigned int coalesce_enable;      /* 0 = subsystem idle             */
module_param(coalesce_enable, uint, 0644);
static unsigned int coalesce_cores = 4;   /* N dedicated consumer cores     */
module_param(coalesce_cores, uint, 0444);
static unsigned int coalesce_cpu_base = 12;/* bind shards to base..base+N-1
					    * (DD4: whole cores 12-15 on isolcpus=8-15,24-31,
					    * disjoint from dispatcher 8 + io-workers 9-11) */
module_param(coalesce_cpu_base, uint, 0444);
static unsigned int coalesce_split;	  /* DD4-phase-2: 1 = split-role SMT (issue
					   * on the shard core, reap+deliver on its
					   * SMT sibling); 0 = one thread does both */
module_param(coalesce_split, uint, 0444);
static unsigned int coalesce_sibling_off = 16;/* CPU distance to the SMT sibling
					       * (sibling(k)=k+16 on this host) */
module_param(coalesce_sibling_off, uint, 0444);
static unsigned int coalesce_deliver;	  /* delivery plane: 0 = CPU WC-store to
					   * VRAM (default); 1 = GPU-pull (enqueue
					   * gather descriptors, GPU delivers, b3) */
module_param(coalesce_deliver, uint, 0444);
static unsigned int coalesce_ring = 262144;/* per-shard ring depth (pow2); also
					    * caps coalesce_window (g_wcap) + the
					    * isefs push limit (ring//4). ~180 B/entry
					    * /shard of rq+cq+win+scr+warena. */
module_param(coalesce_ring, uint, 0444);
static unsigned int coalesce_window = 4096;/* W: flush depth                */
module_param(coalesce_window, uint, 0644);
static unsigned int coalesce_flush_us = 200;/* T: max age before flush      */
module_param(coalesce_flush_us, uint, 0644);
/* Minimum fill before a transient rq_empty (ring momentarily drained) may flush.
 * Default 1 = flush on any empty (lowest latency, but a fast consumer flushes
 * tiny windows and captures almost no temporal reuse). Raise it (toward W, or the
 * bucket cap in bucket mode) so the window/bucket keeps accumulating across the
 * producer's inter-burst/inter-batch gaps — flushing only on W/bucket-full or the
 * age timer. Higher = more coalescing, more latency (bounded by coalesce_flush_us
 * and the bounded-push flow control). See the flush_w/flush_empty/flush_aged
 * counters to see which trigger dominates. */
static unsigned int coalesce_flush_min = 1;
module_param(coalesce_flush_min, uint, 0644);
static unsigned int coalesce_page = 4096;
module_param(coalesce_page, uint, 0644);
static unsigned int coalesce_row = 512;   /* feature row bytes             */
module_param(coalesce_row, uint, 0644);
static unsigned int coalesce_nsid = 1;    /* 1-based backing namespace     */
module_param(coalesce_nsid, uint, 0644);
static unsigned int coalesce_sort;        /* enum co_sort (0=radix default)*/
module_param(coalesce_sort, uint, 0644);
/* CO_SORT_BUCKET (RTL oracle) knobs — see §3.3.8. Bucket geometry is load-time
 * (sizes the store); shift is live. */
static unsigned int coalesce_bucket_bits = 8;  /* nbuckets = 1<<bits (BLRadix ways) */
module_param(coalesce_bucket_bits, uint, 0444);
static unsigned int coalesce_bucket_cap = 64;  /* burst = evict threshold (BLRadix halfBufferLen) */
module_param(coalesce_bucket_cap, uint, 0444);
static unsigned int coalesce_bucket_shift;     /* bin on page_index>>shift (burst-formation stride) */
module_param(coalesce_bucket_shift, uint, 0644);
unsigned int coalesce_read_ns;            /* modeled backend latency per page read
					   * (shared with backend_model.c) */
module_param(coalesce_read_ns, uint, 0644);
static unsigned int coalesce_no_merge;    /* 1 = one backend read per request (baseline, no coalescing) */
module_param(coalesce_no_merge, uint, 0644);
static unsigned int coalesce_staging = 256;/* staging buffers / in-flight pages per shard */
module_param(coalesce_staging, uint, 0444);
static unsigned int coalesce_timeout_us;  /* >0: force-complete backend reads stuck this long (0=off) */
module_param(coalesce_timeout_us, uint, 0644);
static char *coalesce_backend;            /* comma-list of real NVMe dev paths; empty = emulated DRAM */
module_param(coalesce_backend, charp, 0444);
MODULE_PARM_DESC(coalesce_backend, "comma-separated backend block devices (e.g. /dev/nvme0n1); empty = emulated DRAM");

#define CO_MAX_CORES 32

struct co_ent {
	u64 line_tag;
	u64 req_id;
	u64 dst_phys;
	u32 node_id;
	u32 _pad;
};

struct co_shard {
	struct l1_ring *ring;	/* points into g_region */
	struct l1_req *rq;
	struct l1_cqe *cq;
	u32 *rq_seq;		/* MPSC sequence cells */
	u32 rq_mask, cq_mask;
	struct l1_gring *gring;	/* GPU-pull descriptor ring header (in g_desc, b3) */
	struct l1_gdesc *gdesc;	/* descriptor array for this shard                 */
	struct task_struct *task;
	struct co_ent *win, *scr;
	u64 *htab;		/* hash-mode unique set */
	u32 htab_mask;
	u32 rcnt[256];		/* radix histogram (off-stack) */
	u32 win_n;
	u64 oldest_ns;
	u64 last_scan_ns;	/* throttles the timeout sweep (M2) */
	int id;

	/* ---- split-phase state (M1) ---- */
	bool win_sorted;	/* window sorted, now issuing                 */
	u32 issue_pos;		/* first un-issued window entry (partial issue) */
	u32 flush_uniq;		/* unique pages of the current window          */

	struct co_pend_page *pend;	/* pend_cap slots (1:1 with staging bufs) */
	u32 *pend_free;			/* stack of free pend-slot indices       */
	u32 pend_free_n;
	void **spool_kva;		/* staging buffer kernel-va, per slot    */
	struct page **spool_pg;		/* same buffer as struct page (bio path) */

	struct co_waiter *warena;	/* waiter pool (chained by .next)        */
	u32 wfree, wfree_n;		/* waiter freelist head + count          */

	struct llist_head done_list;	/* backend completions land here (-> reaper) */
	struct llist_head pend_return;	/* reaper hands finished pages back (-> issuer) */
	u32 gpu_held;			/* GPU-pull: pages awaiting GPU gather (reaper-local) */
	struct task_struct *reaper;	/* split-role sibling-B thread (or NULL)     */

	/* ---- Stage-A bucket store (CO_SORT_BUCKET only, M4) ---- */
	struct co_ent *bkt;		/* g_nbuckets * g_bucket_cap entries      */
	u32 *bkt_n;			/* per-bucket occupancy                   */
	u64 *bkt_last;			/* per-bucket last line_tag (adjacent pre-filter) */
	u32 bkt_total;			/* total binned across all buckets        */
} ____cacheline_aligned;

/* per-shard pending-page capacity (== staging buffers == in-flight pages) */
static u32 g_pend_cap;
static u32 g_warena_cap;
static u32 g_nbuckets, g_bucket_cap;	/* Stage-A geometry (M4) */
static bool co_backend_is_real;	/* real NVMe backend selected (vs emulated DRAM) */
static u32 g_ndev = 1;		/* number of striped backend devices (M5) */

static void *g_region;
static size_t g_region_bytes;
static u32 g_shard_bytes;
static u32 g_nshards;

/* Contiguous, GPU-accessible staging (b2, GPU-pull delivery). One vmalloc_user
 * region for all shards so a descriptor can name a row by a single offset and
 * userspace maps + cudaHostRegisters it once. Shard k, slot i lives at byte
 * offset ((k*g_pend_cap)+i)*coalesce_page. bio DMAs into it via vmalloc_to_page;
 * the CPU WC-store path reads spool_kva unchanged. */
static void *g_staging;
static size_t g_staging_bytes;

/* GPU-pull descriptor rings (b3): one contiguous vmalloc_user region, per-shard
 * slice = l1_gring header + l1_gdesc[entries]. mmap'd at NVMEV_L1_MMAP_DESC. */
static void *g_desc;
static size_t g_desc_bytes;
static u32 g_desc_shard_bytes;	/* per-shard slice (header + descriptors) */
static u32 g_desc_entries;	/* per-shard ring capacity (pow2)         */
static bool g_gpupull;		/* delivery plane = GPU-pull (staging held for GPU) */
static u32 g_wcap;	/* allocated window capacity; caps the live coalesce_window */
static struct co_shard g_shard[CO_MAX_CORES];
static bool g_running;

/* ---- counters ---- */
static struct {
	atomic64_t flushes, reqs_in, unique_pages, page_merges;
	atomic64_t bytes_useful, bytes_read, max_window, sort_ns;
	atomic64_t max_inflight;	/* peak in-flight backend pages / shard */
	atomic64_t timeouts, errors;	/* backend reads force-completed / errored (M2) */
	/* Stage-A/B oracle counters (CO_SORT_BUCKET, M4) */
	atomic64_t prefilter_hits;	/* arrival-adjacent dups caught by bkt_last */
	atomic64_t reduce_merges;	/* within-burst equal-key merges (Stage B)  */
	atomic64_t burst_full, burst_age; /* evictions by fullness / age            */
	/* Flush-trigger attribution: which condition sealed the window/bucket.
	 * flush_empty dominating with a low coalesce_x => the ring is draining
	 * before the window fills (raise coalesce_flush_min / flush_us / limit). */
	atomic64_t flush_w, flush_empty, flush_aged;
	atomic64_t burst_hist[8];	/* unique-pages-per-burst, log2-bucketed    */
	atomic64_t dev_reads[CO_MAX_DEVS]; /* backend reads issued per device (M5)  */
} g_stats;

/* cookie layout [39:32]=shard [31:16]=gen [15:0]=slot (M2): CO_COOKIE/CO_CK_*
 * now live in coalesce_backend.h so backend providers can decode the shard. */

/* co_pend_page.genstate = (gen16 << 8) | state8 — one word so the ISSUED->
 * terminal transition + gen check is a single cmpxchg (hardirq-safe, M3). */
#define GS_MAKE(gen, st)	(((u32)(u16)(gen) << 8) | (u8)(st))
#define GS_STATE(gs)		((u32)((gs) & 0xff))
#define GS_GEN(gs)		((u32)(((gs) >> 8) & 0xffff))

/* ---- MPSC rq consumer (single consumer = this kthread) ---- */
static bool pop_req(struct co_shard *s, struct l1_req *out)
{
	struct l1_ring *r = s->ring;
	u32 pos = r->rq_tail;
	u32 cell = pos & s->rq_mask;

	/* slot ready when its producer committed seq = pos+1 */
	if (smp_load_acquire(&s->rq_seq[cell]) != pos + 1)
		return false;
	*out = s->rq[cell];
	/* free the cell for the next lap (pos + entries) */
	smp_store_release(&s->rq_seq[cell], pos + r->rq_entries);
	r->rq_tail = pos + 1;
	return true;
}

static bool rq_empty(struct co_shard *s)
{
	u32 pos = s->ring->rq_tail;

	return smp_load_acquire(&s->rq_seq[pos & s->rq_mask]) != pos + 1;
}

static void push_cqe(struct co_shard *s, const struct l1_cqe *c)
{
	struct l1_ring *r = s->ring;

	for (;;) {
		u32 h = r->cq_head;
		u32 t = smp_load_acquire(&r->cq_tail);

		if (h - t < r->cq_entries) {
			s->cq[h & s->cq_mask] = *c;
			smp_store_release(&r->cq_head, h + 1);
			return;
		}
		/* completion ring full: the GPU is behind. Never drop. */
		if (kthread_should_stop())
			return;
		cond_resched();
	}
}

/* ---- per-flush grouping ---- */
static int cmp_tag(const void *a, const void *b)
{
	u64 x = ((const struct co_ent *)a)->line_tag;
	u64 y = ((const struct co_ent *)b)->line_tag;

	return (x > y) - (x < y);
}

static void swp_ent(void *a, void *b, int n)
{
	struct co_ent t = *(struct co_ent *)a;
	*(struct co_ent *)a = *(struct co_ent *)b;
	*(struct co_ent *)b = t;
}

/* LSD radix on the 32-bit page index (line_tag >> page_shift). 4 even passes -> a.
 * cnt is the shard's off-stack 256-entry histogram. */
static void radix_by_page(struct co_ent *a, struct co_ent *b, u32 n, u32 page_shift,
			  u32 *cnt)
{
	struct co_ent *src = a, *dst = b;
	int pass;

	for (pass = 0; pass < 4; pass++) {
		u32 i, sum = 0, sh = pass * 8;

		memset(cnt, 0, 256 * sizeof(*cnt));
		for (i = 0; i < n; i++)
			cnt[((u32)(src[i].line_tag >> page_shift) >> sh) & 0xff]++;
		for (i = 0; i < 256; i++) {
			u32 c = cnt[i];
			cnt[i] = sum;
			sum += c;
		}
		for (i = 0; i < n; i++)
			dst[cnt[((u32)(src[i].line_tag >> page_shift) >> sh) & 0xff]++] = src[i];
		swap(src, dst);
	}
	/* even #passes -> sorted data is back in a (src) */
}

static u32 count_adjacent(struct co_ent *a, u32 n)
{
	u32 i, uniq = 0;
	u64 prev = ~0ULL;

	for (i = 0; i < n; i++) {
		if (i == 0 || a[i].line_tag != prev)
			uniq++;
		prev = a[i].line_tag;
	}
	return uniq;
}

static u32 hash_unique(struct co_shard *s, u32 n)
{
	u32 i, uniq = 0;

	memset(s->htab, 0xff, (size_t)(s->htab_mask + 1) * sizeof(u64));
	for (i = 0; i < n; i++) {
		u64 key = s->win[i].line_tag;
		u32 h = (u32)((key * 0x9e3779b97f4a7c15ULL) >> 40) & s->htab_mask;

		while (s->htab[h] != ~0ULL) {
			if (s->htab[h] == key)
				goto next;
			h = (h + 1) & s->htab_mask;
		}
		s->htab[h] = key;
		uniq++;
next:;
	}
	return uniq;
}

/* ---- front-end delivery vtable (co_deliver_ops): CPU WC-copy to VRAM ----
 * The middle logic below reaches the GPU only through this vtable, so an FPGA
 * DMA delivery engine can replace it at M9 without touching the flush path
 * (assessment DD3). */
static int co_deliver_cpu_row(u64 dst_phys, const void *src, u32 len)
{
#ifdef CONFIG_NVMEVIRT_GPU_DIRECT
	if (dst_phys) {
		void __iomem *dst;

		rcu_read_lock();
		dst = nvmev_gpu_kmap(dst_phys);
		if (dst) {
			/* nvmev_gpu_kmap() rounds down to a PAGE_SIZE boundary and
			 * discards the low bits; add them back like __do_perform_io. */
			dst = (u8 __iomem *)dst + (dst_phys & (PAGE_SIZE - 1));
			kernel_fpu_begin();
			nvmev_memcpy_to_gpu(dst, src, len);
			kernel_fpu_end();
			rcu_read_unlock();
		} else {
			rcu_read_unlock();
			return 2; /* dst_phys not in a registered region */
		}
	}
	/* dst_phys==0: measurement-only (ring+coalesce), no copy */
#endif
	return 0;
}

static void co_deliver_cpu_cqe(struct co_shard *s, const struct l1_cqe *c)
{
	push_cqe(s, c);
}

static const struct co_deliver_ops co_deliver_cpu = {
	.name = "cpu-p2p",
	.row = co_deliver_cpu_row,
	.cqe = co_deliver_cpu_cqe,
};

/* ---- GPU-pull delivery (b3): enqueue a gather descriptor per finished row into
 * this shard's SPSC ring; the GPU persistent kernel (b4) reads the row from the
 * mmap'd staging into VRAM and posts the l1_cqe. The kernel posts no CQE here —
 * the row has not been delivered yet. Single producer (the reap thread), so the
 * head cursor needs no atomic; tail is the GPU's (flow control). ---- */
static void co_deliver_gpupull_deliver(struct co_shard *s,
				       const struct co_waiter *w,
				       const void *page_src, u16 status)
{
	struct l1_gring *g = s->gring;
	u64 off = ((const u8 *)page_src + w->off_in_page) - (const u8 *)g_staging;
	struct l1_gdesc d = {
		.staging_off = off,
		.dst = w->dst_phys,
		.req_id = (u32)w->req_id,
		.len = (u16)coalesce_row,
		.status = status,
	};

	for (;;) {
		u32 h = g->head;
		u32 t = smp_load_acquire(&g->tail);

		if (h - t < g->entries) {			/* room: publish */
			s->gdesc[h & (g->entries - 1)] = d;
			smp_store_release(&g->head, h + 1);
			return;
		}
		if (kthread_should_stop())	/* ring full, GPU behind — never drop */
			return;
		cond_resched();
	}
}

static const struct co_deliver_ops co_deliver_gpupull = {
	.name = "gpu-pull",
	.deliver = co_deliver_gpupull_deliver,
};

/* Active planes (single provider each until M9/M3 register alternates). */
static const struct co_deliver_ops *co_deliver = &co_deliver_cpu;
static const struct co_backend_ops *co_backend;

/* ---- per-shard pool allocators (single-consumer, no locks) ---- */
/* pending page + its staging buffer are one unified credit (assessment R2). */
static u32 pend_alloc(struct co_shard *s, u64 line_tag, u32 dev, u64 now)
{
	u32 slot;
	struct co_pend_page *p;

	if (!s->pend_free_n)
		return CO_NIL;
	slot = s->pend_free[--s->pend_free_n];
	p = &s->pend[slot];
	p->line_tag = line_tag;
	p->staging = s->spool_kva[slot];
	p->spage = s->spool_pg[slot];
	p->first_waiter = CO_NIL;
	p->n_waiters = 0;
	p->dev = (u8)dev;
	p->issue_ns = now;
	/* bump gen (invalidate any late completion of the prior use) + arm ISSUED.
	 * The slot is off every completion path here (freelist-owned), so a plain
	 * store is safe; subsequent transitions use cmpxchg. */
	WRITE_ONCE(p->genstate, GS_MAKE(GS_GEN(p->genstate) + 1, CO_PEND_ISSUED));
	return slot;
}

static void pend_free(struct co_shard *s, u32 slot)
{
	struct co_pend_page *p = &s->pend[slot];

	/* Only the consumer frees, and only after the (single) completion for this
	 * slot has been reaped, so no completer races this store. Keep gen. */
	WRITE_ONCE(p->genstate, GS_MAKE(GS_GEN(p->genstate), CO_PEND_FREE));
	s->pend_free[s->pend_free_n++] = slot;
}

static u32 waiter_alloc(struct co_shard *s)
{
	u32 w = s->wfree;

	if (w == CO_NIL)
		return CO_NIL;
	s->wfree = s->warena[w].next;
	s->wfree_n--;
	return w;
}

/* Return a whole per-page waiter chain to the freelist. */
static void waiter_free_chain(struct co_shard *s, u32 head)
{
	while (head != CO_NIL) {
		u32 nxt = s->warena[head].next;

		s->warena[head].next = s->wfree;
		s->wfree = head;
		s->wfree_n++;
		head = nxt;
	}
}

/* Async backend completion upcall. Runs in arbitrary context (M1: the shard's
 * consumer thread via backend_model->poll; M3: bio bi_end_io, possibly hardirq).
 * Does only cookie validation + llist_add — no alloc, no sleep, no GPU mapping
 * (liveness rule L5). A stale/duplicate completion (the slot timed out and was
 * reused, or the backend double-fired) fails the gen/state check and is dropped.
 *
 * M2 backend_model calls this single-threaded from ->poll, so the gen+state
 * check is race-free. M3 (bio hardirq) races the consumer's timeout sweep and
 * must make the ISSUED->terminal transition atomic (cmpxchg on ->state). */
void nvmev_co_backend_done(u64 cookie, int status)
{
	u32 shard = CO_CK_SHARD(cookie), slot = CO_CK_SLOT(cookie);
	u32 gen = CO_CK_GEN(cookie);
	struct co_shard *s;
	struct co_pend_page *p;

	u32 expect, want;

	if (shard >= g_nshards)
		return;
	s = &g_shard[shard];
	if (slot >= g_pend_cap)
		return;
	p = &s->pend[slot];
	/* Atomically require (gen match AND state==ISSUED) and transition. A stale
	 * or duplicate completion (slot timed out and reused, or double-fired)
	 * loses the cmpxchg and is dropped — safe against the consumer's concurrent
	 * timeout sweep / reuse even from hardirq context (M3). */
	expect = GS_MAKE(gen, CO_PEND_ISSUED);
	want = GS_MAKE(gen, status ? CO_PEND_ERROR : CO_PEND_DONE);
	if (cmpxchg(&p->genstate, expect, want) != expect)
		return;
	llist_add(&p->done_node, &s->done_list);
}

/* Force-complete backend reads stuck in ISSUED past coalesce_timeout_us (a lost
 * or dropped completion). Throttled to ~1 ms/shard. The waiters are delivered
 * with error status by co_reap; a genuine late completion is then rejected by
 * the gen bump on the slot's next allocation. */
static void co_timeout_scan(struct co_shard *s)
{
	u64 now, thresh;
	u32 i;

	if (!coalesce_timeout_us)
		return;
	now = local_clock();
	if (now - s->last_scan_ns < 1000000ULL)		/* ~1 ms throttle */
		return;
	s->last_scan_ns = now;
	thresh = (u64)coalesce_timeout_us * 1000;
	for (i = 0; i < g_pend_cap; i++) {
		struct co_pend_page *p = &s->pend[i];
		u32 gs = READ_ONCE(p->genstate);

		if (GS_STATE(gs) == CO_PEND_ISSUED && now - p->issue_ns >= thresh) {
			/* cmpxchg on the exact gs we read: if a real completion won
			 * the race in between, this fails and we skip. */
			if (cmpxchg(&p->genstate, gs,
				    GS_MAKE(GS_GEN(gs), CO_PEND_ERROR)) == gs) {
				llist_add(&p->done_node, &s->done_list);
				atomic64_inc(&g_stats.timeouts);
			}
		}
	}
}

/* Deliver one waiter's feature row from a staged page into VRAM (P2P), routed
 * through the front-end vtable. */
static void deliver_row(struct co_shard *s, const struct co_waiter *w,
			const void *page_src, u16 status)
{
	struct l1_cqe c = { .req_id = w->req_id, .status = status };

	if (co_deliver->deliver) {	/* GPU-pull: enqueue a gather descriptor */
		co_deliver->deliver(s, w, page_src, status);
		return;
	}
	if (!status)
		c.status = co_deliver->row(w->dst_phys,
					   (const u8 *)page_src + w->off_in_page,
					   coalesce_row);
	co_deliver->cqe(s, &c);
}

/* Reap phase (runs on the reaper thread under split-role, else the consumer):
 * drain completed pages, scatter each waiter's row to VRAM, then hand the page
 * back to the issuer via pend_return (lock-free) instead of freeing it here.
 * The issuer owns the freelists (co_drain_returns), so no freelist locking is
 * needed even when reap and issue run on different SMT threads (DD4-phase-2). */
static void co_reap(struct co_shard *s)
{
	struct llist_node *n = llist_del_all(&s->done_list);

	while (n) {
		struct co_pend_page *p =
			llist_entry(n, struct co_pend_page, done_node);
		u16 st = (GS_STATE(p->genstate) == CO_PEND_ERROR) ? 1 : 0;
		u32 wi = p->first_waiter;

		if (st)
			atomic64_inc(&g_stats.errors);
		n = n->next; /* grab before done_node is reused */
		while (wi != CO_NIL) {
			deliver_row(s, &s->warena[wi], p->staging, st);
			wi = s->warena[wi].next;
		}
		if (g_gpupull) {
			/* Descriptors now point into this page's staging; hold it
			 * until the GPU has gathered them (co_gpu_reclaim), else the
			 * issuer would reuse the buffer under the GPU's read. */
			p->gpu_wm = READ_ONCE(s->gring->head);
			p->gpu_held_next = s->gpu_held;
			s->gpu_held = (u32)(p - s->pend);
		} else {
			llist_add(&p->done_node, &s->pend_return);/* -> issuer frees it */
		}
	}
}

/* GPU-pull: release held pages whose descriptors the GPU has consumed (its
 * ring tail passed the page's watermark) back to the issuer via pend_return.
 * If force, release all (teardown — the GPU has stopped). Runs on the reaper. */
static void co_gpu_reclaim(struct co_shard *s, bool force)
{
	u32 tail = force ? 0 : smp_load_acquire(&s->gring->tail);
	u32 *link = &s->gpu_held;
	u32 idx = s->gpu_held;

	while (idx != CO_NIL) {
		struct co_pend_page *p = &s->pend[idx];
		u32 next = p->gpu_held_next;

		if (force || (s32)(tail - p->gpu_wm) >= 0) {
			*link = next;			/* unlink from the held list */
			llist_add(&p->done_node, &s->pend_return);
		} else {
			link = &p->gpu_held_next;	/* still in flight on the GPU */
		}
		idx = next;
	}
}

/* Issuer side: reclaim pages the reaper finished with (frees waiters + slot).
 * Only this thread touches the freelists, so they stay lock-free. */
static void co_drain_returns(struct co_shard *s)
{
	struct llist_node *n = llist_del_all(&s->pend_return);

	while (n) {
		struct co_pend_page *p =
			llist_entry(n, struct co_pend_page, done_node);

		n = n->next;
		waiter_free_chain(s, p->first_waiter);
		pend_free(s, (u32)(p - s->pend));
	}
}

/* Sort the window and count unique pages; begins the ISSUING phase. */
static void co_sort_window(struct co_shard *s)
{
	u32 n = s->win_n;
	u64 t0 = local_clock();
	u32 page_shift = ilog2(coalesce_page);

	switch (coalesce_sort) {
	case CO_SORT_COMPARISON:
		sort(s->win, n, sizeof(*s->win), cmp_tag, swp_ent);
		s->flush_uniq = count_adjacent(s->win, n);
		break;
	case CO_SORT_HASH:
		s->flush_uniq = hash_unique(s, n);
		break;
	case CO_SORT_RADIX:
	default:
		radix_by_page(s->win, s->scr, n, page_shift, s->rcnt);
		s->flush_uniq = count_adjacent(s->win, n);
		break;
	}
	atomic64_add(local_clock() - t0, &g_stats.sort_ns);
	s->issue_pos = 0;
	s->win_sorted = true;
}

/* Map a global page-aligned byte offset to (backend device, device-local byte
 * offset). Page-granular round-robin keeps every page whole on one device so a
 * feature row never spans devices (assessment §3.3.6). The emulated backend has
 * one DRAM namespace: device = nsid, offset = global. A single real device:
 * device 0, offset = global. */
static void co_map_backend(u64 global_tag, u32 *dev, u64 *dev_off)
{
	if (!co_backend_is_real) {
		*dev = coalesce_nsid ? coalesce_nsid - 1 : 0;
		*dev_off = global_tag;
	} else if (g_ndev > 1) {
		u32 page_shift = ilog2(coalesce_page);
		u64 page = global_tag >> page_shift;

		*dev = (u32)(page % g_ndev);
		*dev_off = (page / g_ndev) << page_shift;
	} else {
		*dev = 0;
		*dev_off = global_tag;
	}
}

/* Issue phase (split from completion). Issue one backend read per (adjacent)
 * run of equal line_tag into a staged page, recording the run's waiters in the
 * pending fan-out table; the modeled/real backend completes it later, and
 * co_reap() scatters the rows. Issue only as many pages as credits (staging
 * buffer ⇔ pending slot, plus waiter nodes) allow; keep the un-issued tail
 * resident and retry next lap — never block (liveness rule L3). When the whole
 * window is issued, charge counters and clear it.
 *
 * Synchronous providers (backend_sync, poll==NULL) complete inline here; async
 * providers (backend_model, real NVMe) return 0 and complete via co_reap. */
static void co_flush_issue(struct co_shard *s)
{
	u32 n = s->win_n, i = s->issue_pos;
	u64 now = local_clock();

	while (i < n) {
		u64 tag = s->win[i].line_tag;	/* global page-aligned offset */
		u32 run = 1, j, slot, dev;
		u64 dev_off;
		struct co_pend_page *p;
		struct co_bread rd;
		int rc;

		if (!coalesce_no_merge)	/* baseline issues one read per request */
			while (i + run < n && s->win[i + run].line_tag == tag)
				run++;

		if (!s->pend_free_n || s->wfree_n < run)
			break;		/* out of credits: defer to next lap */

		/* route this page to its backend device + device-local offset (M5) */
		co_map_backend(tag, &dev, &dev_off);
		slot = pend_alloc(s, tag, dev, now);
		p = &s->pend[slot];
		for (j = 0; j < run; j++) {
			struct co_ent *e = &s->win[i + j];
			u32 wi = waiter_alloc(s);

			s->warena[wi].req_id = e->req_id;
			s->warena[wi].dst_phys = e->dst_phys;
			s->warena[wi].off_in_page =
				(u16)(((u64)e->node_id * coalesce_row) &
				      (coalesce_page - 1));
			s->warena[wi].next = p->first_waiter;
			p->first_waiter = wi;
		}
		p->n_waiters = run;

		rd = (struct co_bread){
			.line_tag = dev_off, .dev = dev, .len = coalesce_page,
			.staging = p->staging, .spage = p->spage,
			.cookie = CO_COOKIE(s->id, GS_GEN(p->genstate), slot),
		};
		atomic64_inc(&g_stats.dev_reads[dev]);
		rc = co_backend->issue_read(&rd);
		if (rc == -EAGAIN) {
			/* backend queue full: undo this page, retry next lap. */
			waiter_free_chain(s, p->first_waiter);
			pend_free(s, slot);
			break;
		}
		if (rc < 0) {
			/* hard error: complete this run's waiters with error status
			 * (no data copied), free the slot, and move on — never spin
			 * on a page the backend refuses. */
			u32 h = p->first_waiter;

			while (h != CO_NIL) {
				deliver_row(s, &s->warena[h], p->staging, 1);
				h = s->warena[h].next;
			}
			atomic64_inc(&g_stats.errors);
			waiter_free_chain(s, p->first_waiter);
			pend_free(s, slot);
			i += run;
			continue;
		}
		if (co_backend->sync) {		/* staging is ready on return now */
			u32 h = p->first_waiter;

			while (h != CO_NIL) {
				deliver_row(s, &s->warena[h], p->staging, 0);
				h = s->warena[h].next;
			}
			waiter_free_chain(s, p->first_waiter);
			pend_free(s, slot);
		} else {			/* async: reaped later; track in-flight depth */
			u32 inuse = g_pend_cap - s->pend_free_n;

			if (inuse > (u32)atomic64_read(&g_stats.max_inflight))
				atomic64_set(&g_stats.max_inflight, inuse);
		}
		i += run;
	}
	s->issue_pos = i;

	if (i == n) {			/* whole window issued -> charge + clear */
		u32 uniq = s->flush_uniq;

		atomic64_inc(&g_stats.flushes);
		atomic64_add(n, &g_stats.reqs_in);
		atomic64_add(uniq, &g_stats.unique_pages);
		atomic64_add(n - uniq, &g_stats.page_merges);
		atomic64_add((u64)n * coalesce_row, &g_stats.bytes_useful);
		atomic64_add((u64)uniq * coalesce_page, &g_stats.bytes_read);
		if (n > (u32)atomic64_read(&g_stats.max_window))
			atomic64_set(&g_stats.max_window, n);
		s->win_n = 0;
		s->win_sorted = false;
		s->issue_pos = 0;
	}
}

static bool aged(struct co_shard *s)
{
	return local_clock() - s->oldest_ns >= (u64)coalesce_flush_us * 1000;
}

/* True once the shard has no window and nothing outstanding in the backend. */
static bool co_shard_idle(struct co_shard *s)
{
	return s->win_n == 0 && s->pend_free_n == g_pend_cap &&
	       llist_empty(&s->done_list) && llist_empty(&s->pend_return) &&
	       s->bkt_total == 0;
}

/* ---- Stage A/B (CO_SORT_BUCKET, the RTL oracle) ----
 * Stage A bins requests online by coarse page bits (BLRadix), with a per-bucket
 * arrival-adjacent pre-filter; a bucket evicts a burst on fullness or age.
 * Stage B sort-reduces the evicted burst (dedup bounded to the burst — weaker
 * than the exact window modes, matching the hardware) and hands it to the
 * unchanged split-phase issue path via s->win. Only one burst is issued at a
 * time (win_sorted), so the store and the in-flight burst never alias. */
static u32 co_bucket_of(u64 line_tag)
{
	u32 page_index = (u32)(line_tag >> ilog2(coalesce_page));

	return (page_index >> coalesce_bucket_shift) & (g_nbuckets - 1);
}

/* Seal bucket b's contents as the current burst: copy into s->win, sort-reduce,
 * arm ISSUING. `by_age` selects the eviction-reason counter. */
static void co_bucket_evict(struct co_shard *s, u32 b, bool by_age)
{
	u32 n = s->bkt_n[b];
	u32 uniq, h;

	if (!n)
		return;
	memcpy(s->win, &s->bkt[(size_t)b * g_bucket_cap], (size_t)n * sizeof(*s->win));
	s->win_n = n;
	s->bkt_total -= n;
	s->bkt_n[b] = 0;

	sort(s->win, n, sizeof(*s->win), cmp_tag, swp_ent);	/* Stage B: small burst */
	uniq = count_adjacent(s->win, n);
	s->flush_uniq = uniq;
	s->win_sorted = true;
	s->issue_pos = 0;

	atomic64_add(n - uniq, &g_stats.reduce_merges);
	atomic64_inc(by_age ? &g_stats.burst_age : &g_stats.burst_full);
	h = uniq ? ilog2(uniq) : 0;
	atomic64_inc(&g_stats.burst_hist[min_t(u32, h, 7)]);
}

/* Return the fullest non-empty bucket, or CO_NIL if the store is empty. */
static u32 co_bucket_fullest(struct co_shard *s)
{
	u32 b, best = CO_NIL, bestn = 0;

	for (b = 0; b < g_nbuckets; b++)
		if (s->bkt_n[b] > bestn) {
			bestn = s->bkt_n[b];
			best = b;
		}
	return best;
}

/* One consumer lap in bucket mode: finish any in-flight burst, else bin and
 * evict. */
static void co_bucket_lap(struct co_shard *s, u32 budget)
{
	struct l1_req req;
	u32 binned = 0;

	if (s->win_sorted) {		/* finish issuing the current burst first */
		co_flush_issue(s);
		return;
	}
	/* Left-over partial window from a prior non-bucket mode: drain it. */
	if (s->win_n) {
		co_sort_window(s);
		co_flush_issue(s);
		return;
	}

	while (binned < budget && pop_req(s, &req)) {
		u64 off = (u64)req.node_id * coalesce_row;
		u64 tag = off & ~((u64)coalesce_page - 1);
		u32 b = co_bucket_of(tag);
		struct co_ent *e;

		if (s->bkt_total == 0)
			s->oldest_ns = local_clock();
		if (tag == s->bkt_last[b])
			atomic64_inc(&g_stats.prefilter_hits);
		s->bkt_last[b] = tag;

		if (s->bkt_n[b] >= g_bucket_cap) {
			/* full: evict as a burst, then bin this req into the emptied
			 * bucket and stop to issue the sealed burst. */
			co_bucket_evict(s, b, false);
			atomic64_inc(&g_stats.flush_w);
			e = &s->bkt[(size_t)b * g_bucket_cap + s->bkt_n[b]];
			e->line_tag = tag;
			e->req_id = req.req_id;
			e->node_id = req.node_id;
			e->dst_phys = req.dst_phys;
			s->bkt_n[b]++;
			s->bkt_total++;
			break;
		}
		e = &s->bkt[(size_t)b * g_bucket_cap + s->bkt_n[b]];
		e->line_tag = tag;
		e->req_id = req.req_id;
		e->node_id = req.node_id;
		e->dst_phys = req.dst_phys;
		s->bkt_n[b]++;
		s->bkt_total++;
		binned++;
	}

	/* Quiescence / age: flush a partial bucket so latency stays bounded. On a
	 * transient rq_empty, only evict once the fullest bucket has coalesce_flush_min
	 * entries — else keep buckets resident across the producer's inter-batch gaps
	 * to catch cross-batch reuse. Age always evicts (latency bound). */
	if (!s->win_sorted && s->bkt_total) {
		bool age = aged(s);
		bool empty = rq_empty(s);

		if (age || empty) {
			u32 b = co_bucket_fullest(s);

			if (b != CO_NIL &&
			    (age || s->bkt_n[b] >= coalesce_flush_min)) {
				atomic64_inc(age ? &g_stats.flush_aged :
						   &g_stats.flush_empty);
				co_bucket_evict(s, b, true);
			}
		}
	}

	if (s->win_sorted)
		co_flush_issue(s);
}

/* Reap side: advance backend completions and deliver rows. Runs on the reaper
 * thread (split-role) or inline in the consumer (single-thread). */
static void co_reaper_lap(struct co_shard *s)
{
	if (co_backend->poll)
		co_backend->poll(s->id);
	co_timeout_scan(s);	/* rescue lost completions (throttled) */
	co_reap(s);		/* deliver + hand pages back via pend_return */
	if (g_gpupull)		/* release staging the GPU has finished gathering */
		co_gpu_reclaim(s, false);
}

/* Issue side: reclaim finished pages (credits), fill/coalesce, issue. */
static void co_issue_lap(struct co_shard *s, bool *got)
{
	u32 W = min_t(u32, coalesce_window, g_wcap);	/* cap to allocation */
	struct l1_req req;

	co_drain_returns(s);	/* reclaim credits the reaper handed back (rule L1) */

	if (coalesce_sort == CO_SORT_BUCKET) {
		co_bucket_lap(s, W);
		*got = s->bkt_total || s->win_sorted;
		return;
	}
	if (!s->win_sorted) {	/* fill until sealed; a sealed window is frozen */
		while (s->win_n < W && pop_req(s, &req)) {
			struct co_ent *e = &s->win[s->win_n];
			u64 off = (u64)req.node_id * coalesce_row;

			if (s->win_n == 0)
				s->oldest_ns = local_clock();
			e->line_tag = off & ~((u64)coalesce_page - 1);
			e->req_id = req.req_id;
			e->node_id = req.node_id;
			e->dst_phys = req.dst_phys;
			s->win_n++;
			*got = true;
		}
		if (s->win_n) {
			bool full = s->win_n >= W;
			bool age = aged(s);
			/* A transient rq_empty only flushes once the window holds at
			 * least coalesce_flush_min — otherwise keep accumulating across
			 * the producer's inter-burst gaps (rule: never past W or age). */
			bool empty = rq_empty(s) && s->win_n >= coalesce_flush_min;

			if (full || age || empty) {
				atomic64_inc(full ? &g_stats.flush_w :
					     age  ? &g_stats.flush_aged :
						    &g_stats.flush_empty);
				co_sort_window(s);	/* seal -> ISSUING */
			}
		}
	}
	if (s->win_sorted)
		co_flush_issue(s);	/* partial issue defers on credit exhaustion */
}

/* Force-complete any reads stuck in ISSUED so their waiters/slots are released
 * (teardown / dropped completions). */
static void co_force_complete(struct co_shard *s)
{
	u32 i;

	for (i = 0; i < g_pend_cap; i++) {
		struct co_pend_page *p = &s->pend[i];
		u32 gs = READ_ONCE(p->genstate);

		if (GS_STATE(gs) == CO_PEND_ISSUED &&
		    cmpxchg(&p->genstate, gs,
			    GS_MAKE(GS_GEN(gs), CO_PEND_ERROR)) == gs)
			llist_add(&p->done_node, &s->done_list);
	}
}

/* Issuer thread (sibling A), and the single-thread consumer when
 * coalesce_split=0 (then it reaps inline too). */
static int co_consumer(void *arg)
{
	struct co_shard *s = arg;
	bool split = coalesce_split && s->reaper;
	int guard;

	while (!kthread_should_stop()) {
		bool got = false;

		if (!coalesce_enable) {
			schedule_timeout_interruptible(1);
			continue;
		}
		if (!split)
			co_reaper_lap(s);	/* single-thread: reap here too */
		co_issue_lap(s, &got);
		cond_resched();
		if (!got)
			cpu_relax();
	}

	/* Teardown. Split: only issue the resident window/buckets so every request
	 * becomes a pending read; the reaper (stopped after us) drains them.
	 * Single: also reap + force-complete so no waiter leaks before pool free. */
	guard = 100000;
	while (guard-- > 0) {
		bool got = false;

		if (!split)
			co_reaper_lap(s);
		if (!s->win_sorted && s->win_n == 0 && s->bkt_total) {
			u32 b = co_bucket_fullest(s);	/* flush leftover buckets */

			if (b != CO_NIL)
				co_bucket_evict(s, b, true);
		}
		co_issue_lap(s, &got);
		if (split) {
			if (s->win_n == 0 && !s->win_sorted && s->bkt_total == 0)
				break;			/* everything issued */
		} else if (co_shard_idle(s)) {
			break;
		}
		cond_resched();
	}
	if (!split) {
		co_force_complete(s);
		co_reaper_lap(s);
		if (g_gpupull)
			co_gpu_reclaim(s, true);	/* GPU stopped: release held */
		co_drain_returns(s);
	}
	return 0;
}

/* Reaper thread (sibling B) — split-role only: reap + deliver while the issuer
 * (sibling A) fills and issues on the paired physical core. */
static int co_reaper(void *arg)
{
	struct co_shard *s = arg;
	int guard;

	while (!kthread_should_stop()) {
		if (!coalesce_enable) {
			schedule_timeout_interruptible(1);
			continue;
		}
		co_reaper_lap(s);
		cond_resched();
	}

	/* Teardown: the issuer has stopped (we are stopped after it), so no new
	 * reads arrive. Drain outstanding + force-complete stragglers. push_cqe
	 * bails on kthread_should_stop, so a departed GPU can't wedge us. */
	guard = 100000;
	while (!llist_empty(&s->done_list) && guard-- > 0) {
		co_reaper_lap(s);
		cond_resched();
	}
	co_force_complete(s);
	co_reap(s);
	if (g_gpupull) {
		/* the issuer (A) is already stopped, so we may touch the freelists:
		 * force-release held pages and free everything handed back. */
		co_gpu_reclaim(s, true);
		co_drain_returns(s);
	}
	return 0;
}

/* ---- region + device ---- */
static int co_mmap(struct file *file, struct vm_area_struct *vma)
{
	unsigned long sz = vma->vm_end - vma->vm_start;

	/* offset selects the region: GPU-pull staging (b2) / descriptor ring (b3)
	 * / the L1 rings (offset 0). */
	if (vma->vm_pgoff == (NVMEV_L1_MMAP_STAGING >> PAGE_SHIFT)) {
		if (!g_staging || sz > g_staging_bytes)
			return -EINVAL;
		vma->vm_pgoff = 0;	/* remap_vmalloc_range maps from region start */
		return remap_vmalloc_range(vma, g_staging, 0);
	}
	if (vma->vm_pgoff == (NVMEV_L1_MMAP_DESC >> PAGE_SHIFT)) {
		if (!g_desc || sz > g_desc_bytes)
			return -EINVAL;
		vma->vm_pgoff = 0;
		return remap_vmalloc_range(vma, g_desc, 0);
	}
	if (sz > g_region_bytes)
		return -EINVAL;
	return remap_vmalloc_range(vma, g_region, 0);
}

/* VA->dst_phys translate (M6): resolve a registered VRAM range's GPU-page bus
 * addresses for userspace, replacing the fragile /proc/nvmev/gpu_mem text parse. */
static long co_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
	struct l1_translate t;
	u64 *phys;
	u32 n = 0;
	long rc;

	if (cmd == NVMEV_L1_IOC_STGINFO) {	/* GPU-pull staging geometry (b2) */
		struct l1_staging g = {
			.mmap_off = NVMEV_L1_MMAP_STAGING,
			.bytes = g_staging_bytes,
			.nshards = g_nshards,
			.pend_cap = g_pend_cap,
			.page = coalesce_page,
		};

		return copy_to_user((void __user *)arg, &g, sizeof(g)) ? -EFAULT : 0;
	}
	if (cmd == NVMEV_L1_IOC_DESCINFO) {	/* GPU-pull descriptor-ring geometry (b3) */
		struct l1_descinfo g = {
			.mmap_off = NVMEV_L1_MMAP_DESC,
			.bytes = g_desc_bytes,
			.nshards = g_nshards,
			.entries = g_desc_entries,
			.shard_bytes = g_desc_shard_bytes,
		};

		return copy_to_user((void __user *)arg, &g, sizeof(g)) ? -EFAULT : 0;
	}
	if (cmd != NVMEV_L1_IOC_TRANSLATE)
		return -ENOTTY;
	if (copy_from_user(&t, (void __user *)arg, sizeof(t)))
		return -EFAULT;
	if (!t.n_pages || t.n_pages > (1u << 20))
		return -EINVAL;

	phys = kvmalloc_array(t.n_pages, sizeof(u64), GFP_KERNEL);
	if (!phys)
		return -ENOMEM;

	rc = nvmev_gpu_region_pages(t.gpu_va, t.len, phys, t.n_pages, &n);
	if (rc)
		goto out;
	if (copy_to_user((void __user *)(uintptr_t)t.phys, phys,
			 (size_t)n * sizeof(u64))) {
		rc = -EFAULT;
		goto out;
	}
	t.n_pages = n;
	t.page_shift = NVMEV_GPU_PAGE_SHIFT;
	if (copy_to_user((void __user *)arg, &t, sizeof(t)))
		rc = -EFAULT;
out:
	kvfree(phys);
	return rc;
}

static const struct file_operations co_dev_fops = {
	.owner = THIS_MODULE,
	.mmap = co_mmap,
	.unlocked_ioctl = co_ioctl,
	.compat_ioctl = co_ioctl,
};

static struct miscdevice co_misc = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = "nvmev_l1",
	.fops = &co_dev_fops,
};

/* ---- /proc/nvmev/coalesce ---- */
static int co_proc_show(struct seq_file *m, void *v)
{
	u64 reqs = atomic64_read(&g_stats.reqs_in);
	u64 up = atomic64_read(&g_stats.unique_pages);
	u64 useful = atomic64_read(&g_stats.bytes_useful);
	u64 read = atomic64_read(&g_stats.bytes_read);

	seq_printf(m, "enable          %u\n", coalesce_enable);
	seq_printf(m, "backend         %s\n", co_backend ? co_backend->name : "?");
	seq_printf(m, "cores           %u\n", g_nshards);
	seq_printf(m, "split           %u (reap on SMT sibling +%u)\n",
		   coalesce_split, coalesce_sibling_off);
	seq_printf(m, "deliver         %s\n", co_deliver ? co_deliver->name : "?");
	seq_printf(m, "window          %u\n", coalesce_window);
	seq_printf(m, "flush_us        %u\n", coalesce_flush_us);
	seq_printf(m, "flush_min       %u\n", coalesce_flush_min);
	seq_printf(m, "page            %u\n", coalesce_page);
	seq_printf(m, "row             %u\n", coalesce_row);
	seq_printf(m, "sort            %u\n", coalesce_sort);
	seq_printf(m, "read_ns         %u\n", coalesce_read_ns);
	seq_printf(m, "no_merge        %u\n", coalesce_no_merge);
	seq_printf(m, "staging         %u\n", g_pend_cap);
	seq_printf(m, "flushes         %llu\n", atomic64_read(&g_stats.flushes));
	seq_printf(m, "flush_w         %llu\n", atomic64_read(&g_stats.flush_w));
	seq_printf(m, "flush_empty     %llu\n", atomic64_read(&g_stats.flush_empty));
	seq_printf(m, "flush_aged      %llu\n", atomic64_read(&g_stats.flush_aged));
	seq_printf(m, "reqs_in         %llu\n", reqs);
	seq_printf(m, "unique_pages    %llu\n", up);
	seq_printf(m, "backend_reads   %llu\n", up);
	seq_printf(m, "ndev            %u\n", g_ndev);
	if (g_ndev > 1) {
		u32 d;

		seq_printf(m, "dev_reads      ");
		for (d = 0; d < g_ndev && d < CO_MAX_DEVS; d++)
			seq_printf(m, " %llu", atomic64_read(&g_stats.dev_reads[d]));
		seq_printf(m, "\n");
	}
	seq_printf(m, "page_merges     %llu\n", atomic64_read(&g_stats.page_merges));
	seq_printf(m, "bytes_useful    %llu\n", useful);
	seq_printf(m, "bytes_read      %llu\n", read);
	seq_printf(m, "max_window      %llu\n", atomic64_read(&g_stats.max_window));
	seq_printf(m, "inflight_peak   %llu\n", atomic64_read(&g_stats.max_inflight));
	seq_printf(m, "timeouts        %llu\n", atomic64_read(&g_stats.timeouts));
	seq_printf(m, "errors          %llu\n", atomic64_read(&g_stats.errors));
	seq_printf(m, "sort_ns         %llu\n", atomic64_read(&g_stats.sort_ns));
	if (coalesce_sort == CO_SORT_BUCKET) {
		int hi;

		seq_printf(m, "bucket_bits     %u\n", coalesce_bucket_bits);
		seq_printf(m, "bucket_cap      %u\n", coalesce_bucket_cap);
		seq_printf(m, "bucket_shift    %u\n", coalesce_bucket_shift);
		seq_printf(m, "prefilter_hits  %llu\n", atomic64_read(&g_stats.prefilter_hits));
		seq_printf(m, "reduce_merges   %llu\n", atomic64_read(&g_stats.reduce_merges));
		seq_printf(m, "burst_full      %llu\n", atomic64_read(&g_stats.burst_full));
		seq_printf(m, "burst_age       %llu\n", atomic64_read(&g_stats.burst_age));
		seq_printf(m, "burst_hist     ");
		for (hi = 0; hi < 8; hi++)	/* unique pages/burst: 1,2,4,8,..,128+ */
			seq_printf(m, " %llu", atomic64_read(&g_stats.burst_hist[hi]));
		seq_printf(m, "\n");
	}
	seq_printf(m, "coalesce_x1000  %llu\n", up ? reqs * 1000 / up : 0);
	seq_printf(m, "useful_frac1000 %llu\n", read ? useful * 1000 / read : 0);
	return 0;
}

static int co_proc_open(struct inode *inode, struct file *file)
{
	return single_open(file, co_proc_show, NULL);
}

static ssize_t co_proc_write(struct file *file, const char __user *buf,
			     size_t count, loff_t *ppos)
{
	memset(&g_stats, 0, sizeof(g_stats));
	return count;
}

#if LINUX_VERSION_CODE > KERNEL_VERSION(5, 0, 0)
static const struct proc_ops co_proc_fops = {
	.proc_open = co_proc_open, .proc_write = co_proc_write,
	.proc_read = seq_read, .proc_lseek = seq_lseek, .proc_release = single_release,
};
#else
static const struct file_operations co_proc_fops = {
	.open = co_proc_open, .write = co_proc_write,
	.read = seq_read, .llseek = seq_lseek, .release = single_release,
};
#endif

/* ---- per-shard split-phase pool alloc/free ---- */
static void co_shard_pools_free(struct co_shard *s)
{
	/* Staging buffers are slices of the shared g_staging vmalloc region (freed
	 * once in nvmev_coalesce_exit), so no per-slot page free here. */
	kvfree(s->spool_pg);
	kvfree(s->spool_kva);
	kvfree(s->pend_free);
	kvfree(s->pend);
	kvfree(s->warena);
	kvfree(s->bkt);
	kvfree(s->bkt_n);
	kvfree(s->bkt_last);
	s->spool_pg = NULL; s->spool_kva = NULL; s->pend_free = NULL;
	s->pend = NULL; s->warena = NULL;
	s->bkt = NULL; s->bkt_n = NULL; s->bkt_last = NULL;
}

static int co_shard_pools_alloc(struct co_shard *s)
{
	u32 i;

	init_llist_head(&s->done_list);
	init_llist_head(&s->pend_return);
	s->gpu_held = CO_NIL;
	s->win_sorted = false;
	s->issue_pos = 0;

	s->pend = kvcalloc(g_pend_cap, sizeof(*s->pend), GFP_KERNEL);
	s->pend_free = kvmalloc_array(g_pend_cap, sizeof(u32), GFP_KERNEL);
	s->spool_kva = kvcalloc(g_pend_cap, sizeof(void *), GFP_KERNEL);
	s->spool_pg = kvcalloc(g_pend_cap, sizeof(struct page *), GFP_KERNEL);
	s->warena = kvmalloc_array(g_warena_cap, sizeof(*s->warena), GFP_KERNEL);
	if (!s->pend || !s->pend_free || !s->spool_kva || !s->spool_pg || !s->warena)
		return -ENOMEM;

	{
		u32 k = (u32)(s - g_shard);	/* this shard's slice of g_staging */
		u8 *base = (u8 *)g_staging + (size_t)k * g_pend_cap * coalesce_page;

		for (i = 0; i < g_pend_cap; i++) {
			s->spool_kva[i] = base + (size_t)i * coalesce_page;
			s->spool_pg[i] = vmalloc_to_page(s->spool_kva[i]);
			s->pend_free[i] = i;	/* all slots free */
		}
	}
	s->pend_free_n = g_pend_cap;

	for (i = 0; i < g_warena_cap; i++)	/* waiter freelist chain */
		s->warena[i].next = (i + 1 < g_warena_cap) ? i + 1 : CO_NIL;
	s->wfree = 0;
	s->wfree_n = g_warena_cap;

	/* Stage-A bucket store (CO_SORT_BUCKET). Allocated always so coalesce_sort
	 * can be switched to bucket mode at runtime. */
	s->bkt = kvmalloc_array((size_t)g_nbuckets * g_bucket_cap,
				sizeof(*s->bkt), GFP_KERNEL);
	s->bkt_n = kvcalloc(g_nbuckets, sizeof(u32), GFP_KERNEL);
	s->bkt_last = kvmalloc_array(g_nbuckets, sizeof(u64), GFP_KERNEL);
	if (!s->bkt || !s->bkt_n || !s->bkt_last)
		return -ENOMEM;
	for (i = 0; i < g_nbuckets; i++)
		s->bkt_last[i] = ~0ULL;		/* no last key yet */
	s->bkt_total = 0;
	return 0;
}

int nvmev_coalesce_init(struct proc_dir_entry *proc_root)
{
	u32 ring = roundup_pow_of_two(coalesce_ring ? coalesce_ring : 1024);
	u32 hdr = round_up(sizeof(struct l1_ring), 64);
	u32 seqb = round_up(ring * sizeof(u32), 64);
	u32 rqb = ring * sizeof(struct l1_req);
	u32 cqb = ring * sizeof(struct l1_cqe);
	u32 k, htab_sz;
	int rc;

	BUILD_BUG_ON(sizeof(struct l1_req) != 32);
	BUILD_BUG_ON(sizeof(struct l1_cqe) != 16);

	/* Rows must not straddle a page: a request maps to a single line_tag and a
	 * single staged page (assessment §5.4). */
	WARN_ON_ONCE(coalesce_row == 0 || coalesce_page % coalesce_row != 0);

	/* In-flight pages per shard = staging buffers = pending slots (one unified
	 * credit, assessment R2); the backend's queue depth is sized to match. */
	g_pend_cap = roundup_pow_of_two(clamp_val(coalesce_staging, 1u, 1u << 16));

	/* Stage-A bucket geometry (CO_SORT_BUCKET). */
	g_nbuckets = 1u << clamp_val(coalesce_bucket_bits, 1u, 16u);
	g_bucket_cap = clamp_val(coalesce_bucket_cap, 2u, 4096u);

	/* Select + bring up the backend provider: real NVMe if coalesce_backend
	 * lists device paths, else the emulated DRAM model (M1). */
	if (coalesce_backend && coalesce_backend[0]) {
		const char *paths[CO_MAX_DEVS];
		char *dup = kstrdup(coalesce_backend, GFP_KERNEL);
		char *cur = dup, *tok;
		u32 ndev = 0;

		if (!dup)
			return -ENOMEM;
		while ((tok = strsep(&cur, ",")) != NULL && ndev < CO_MAX_DEVS)
			if (*tok)
				paths[ndev++] = tok;
		/* "pci:<BDF>" paths select the middle-layer-owned direct NVMe QP
		 * (Stage A); plain "/dev/nvmeXn1" paths keep the bio backend. */
		if (ndev && !strncmp(paths[0], "pci:", 4))
			co_backend = nvmev_co_backend_direct();
		else
			co_backend = nvmev_co_backend_nvme();
		co_backend_is_real = true;
		g_ndev = ndev ? ndev : 1;	/* stripe across the opened devices */
		rc = co_backend->init(ndev, paths, g_pend_cap, coalesce_page);
		kfree(dup);	/* backend_nvme opened the bdevs; paths no longer needed */
	} else {
		co_backend = nvmev_co_backend_model();
		co_backend_is_real = false;
		g_ndev = 1;
		rc = co_backend->init(1, NULL, g_pend_cap, coalesce_page);
	}
	if (rc)
		return rc;

	g_nshards = clamp_val(coalesce_cores, 1, CO_MAX_CORES);
	g_shard_bytes = PAGE_ALIGN(hdr + seqb + rqb + cqb);
	g_region_bytes = (size_t)g_nshards * g_shard_bytes;
	/* Allocate the window to the full ring capacity so a live `coalesce_window`
	 * raise (sweeps) is always both safe and real (window can't exceed the ring). */
	g_wcap = ring;
	g_warena_cap = g_wcap * 2;	/* waiters can span up to ~2 flushes in flight */
	htab_sz = roundup_pow_of_two(g_wcap * 2);

	g_region = vmalloc_user(g_region_bytes);
	if (!g_region) {
		co_backend->exit();
		return -ENOMEM;
	}

	/* One contiguous, page-granular, mmap-able staging region for all shards. */
	g_staging_bytes = (size_t)g_nshards * g_pend_cap * coalesce_page;
	g_staging = vmalloc_user(g_staging_bytes);
	if (!g_staging) {
		vfree(g_region);
		g_region = NULL;
		co_backend->exit();
		return -ENOMEM;
	}

	/* GPU-pull descriptor rings (b3): per-shard SPSC ring sized to the L1 ring
	 * depth (bounds outstanding reaped-but-undelivered rows). */
	g_desc_entries = ring;
	g_desc_shard_bytes = PAGE_ALIGN(round_up(sizeof(struct l1_gring), 64) +
			(size_t)g_desc_entries * sizeof(struct l1_gdesc));
	g_desc_bytes = (size_t)g_nshards * g_desc_shard_bytes;
	g_desc = vmalloc_user(g_desc_bytes);
	if (!g_desc) {
		vfree(g_staging); g_staging = NULL;
		vfree(g_region); g_region = NULL;
		co_backend->exit();
		return -ENOMEM;
	}
	co_deliver = coalesce_deliver ? &co_deliver_gpupull : &co_deliver_cpu;
	g_gpupull = coalesce_deliver != 0;

	for (k = 0; k < g_nshards; k++) {
		struct co_shard *s = &g_shard[k];
		struct l1_gring *gr = (struct l1_gring *)((u8 *)g_desc +
					(size_t)k * g_desc_shard_bytes);
		struct l1_ring *r = (struct l1_ring *)((u8 *)g_region + (size_t)k * g_shard_bytes);

		memset(r, 0, sizeof(*r));
		r->magic = NVMEV_L1_MAGIC;
		r->version = NVMEV_L1_VERSION;
		r->shard = k;
		r->nr_shards = g_nshards;
		r->rq_entries = ring;
		r->cq_entries = ring;
		r->rq_seq_off = hdr;
		r->rq_off = hdr + seqb;
		r->cq_off = hdr + seqb + rqb;
		r->shard_bytes = g_shard_bytes;

		s->ring = r;
		s->rq_seq = (u32 *)((u8 *)r + r->rq_seq_off);
		s->rq = (struct l1_req *)((u8 *)r + r->rq_off);
		s->cq = (struct l1_cqe *)((u8 *)r + r->cq_off);
		s->rq_mask = ring - 1;
		s->cq_mask = ring - 1;
		s->id = k;
		s->win_n = 0;

		memset(gr, 0, sizeof(*gr));		/* GPU-pull descriptor ring (b3) */
		gr->magic = NVMEV_L1_GRING_MAGIC;
		gr->entries = g_desc_entries;
		gr->desc_off = round_up(sizeof(struct l1_gring), 64);
		gr->shard = k;
		gr->shard_bytes = g_desc_shard_bytes;
		s->gring = gr;
		s->gdesc = (struct l1_gdesc *)((u8 *)gr + gr->desc_off);

		{ u32 i; for (i = 0; i < ring; i++) s->rq_seq[i] = i; } /* MPSC init */
		s->win = kvmalloc_array(g_wcap, sizeof(*s->win), GFP_KERNEL);
		s->scr = kvmalloc_array(g_wcap, sizeof(*s->scr), GFP_KERNEL);
		s->htab = kvmalloc_array(htab_sz, sizeof(u64), GFP_KERNEL);
		if (!s->win || !s->scr || !s->htab) {
			rc = -ENOMEM;
			goto err_free;
		}
		s->htab_mask = htab_sz - 1;

		rc = co_shard_pools_alloc(s);	/* pending / staging / waiter pools */
		if (rc)
			goto err_free;
	}

	rc = misc_register(&co_misc);
	if (rc)
		goto err_free;

	g_running = true;
	for (k = 0; k < g_nshards; k++) {
		struct co_shard *s = &g_shard[k];
		u32 cpu = coalesce_cpu_base + k;

		s->task = kthread_create(co_consumer, s, "nvmev_l1_%u", k);
		if (IS_ERR(s->task)) {
			rc = PTR_ERR(s->task);
			s->task = NULL;
			goto err_stop;
		}
		if (cpu < nr_cpu_ids && cpu_online(cpu))
			kthread_bind(s->task, cpu);

		/* DD4-phase-2: reap+deliver on the shard core's SMT sibling, so the
		 * stall-heavy WC-store delivery overlaps the issuer's sort/issue.
		 * Create+bind before waking the issuer so it sees s->reaper. */
		if (coalesce_split) {
			u32 sib = cpu + coalesce_sibling_off;

			s->reaper = kthread_create(co_reaper, s, "nvmev_l1r_%u", k);
			if (IS_ERR(s->reaper)) {
				rc = PTR_ERR(s->reaper);
				s->reaper = NULL;
				goto err_stop;
			}
			if (sib < nr_cpu_ids && cpu_online(sib))
				kthread_bind(s->reaper, sib);
			wake_up_process(s->reaper);
		}
		wake_up_process(s->task);
	}

	proc_create("coalesce", 0664, proc_root, &co_proc_fops);
	NVMEV_INFO("coalesce: %u cores, ring=%u, window=%u, /dev/nvmev_l1 %lu bytes\n",
		   g_nshards, ring, coalesce_window, (unsigned long)g_region_bytes);
	return 0;

err_stop:
	for (k = 0; k < g_nshards; k++)
		if (g_shard[k].task)
			kthread_stop(g_shard[k].task);
	for (k = 0; k < g_nshards; k++)
		if (g_shard[k].reaper)
			kthread_stop(g_shard[k].reaper);
	misc_deregister(&co_misc);
err_free:
	for (k = 0; k < g_nshards; k++) {
		kvfree(g_shard[k].win);
		kvfree(g_shard[k].scr);
		kvfree(g_shard[k].htab);
		co_shard_pools_free(&g_shard[k]);
	}
	vfree(g_desc);
	g_desc = NULL;
	vfree(g_staging);
	g_staging = NULL;
	vfree(g_region);
	g_region = NULL;
	if (co_backend)
		co_backend->exit();
	return rc;
}

void nvmev_coalesce_exit(struct proc_dir_entry *proc_root)
{
	u32 k;

	if (!g_running)
		return;
	/* Stop the issuers first (they finish issuing the resident window), then
	 * the reapers (they drain all outstanding reads). */
	for (k = 0; k < g_nshards; k++)
		if (g_shard[k].task)
			kthread_stop(g_shard[k].task);
	for (k = 0; k < g_nshards; k++)
		if (g_shard[k].reaper)
			kthread_stop(g_shard[k].reaper);
	remove_proc_entry("coalesce", proc_root);
	misc_deregister(&co_misc);
	/* Quiesce the backend (wait out in-flight DMAs) BEFORE freeing staging
	 * pages, or a still-running bio would write freed memory. */
	if (co_backend)
		co_backend->exit();
	for (k = 0; k < g_nshards; k++) {
		kvfree(g_shard[k].win);
		kvfree(g_shard[k].scr);
		kvfree(g_shard[k].htab);
		co_shard_pools_free(&g_shard[k]);
	}
	vfree(g_desc);
	g_desc = NULL;
	vfree(g_staging);
	g_staging = NULL;
	vfree(g_region);
	g_region = NULL;
	g_running = false;
}

#endif /* CONFIG_NVMEVIRT_MIDLAYER */
