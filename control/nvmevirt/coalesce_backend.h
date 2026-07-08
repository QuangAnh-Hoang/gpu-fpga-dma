/* SPDX-License-Identifier: GPL-2.0 */
/*
 * coalesce_backend — the two kernel-internal vtables that decouple the middle
 * layer's three planes (docs/middle-layer-assessment.md DD3, §3.3.1-§3.3.2).
 * NOT part of any cross-boundary ABI; no BUILD_BUG_ON size freeze.
 *
 *   GPU app  ⇄  [L1 ring ABI (frozen)]  ⇄  co_deliver_ops   ← front-end delivery
 *                    middle: window/sort/coalesce/fan-out    ← pure logic
 *                                    co_backend_ops          ← back end
 *
 * The middle logic (coalesce.c) reaches the GPU only through co_deliver_ops and
 * storage only through co_backend_ops, so either end is swappable (CPU WC-copy
 * → FPGA DMA; emulated DRAM → real NVMe → FPGA-issued) without touching the L1
 * ring ABI or the window/coalesce code.
 */
#ifndef NVMEV_COALESCE_BACKEND_H
#define NVMEV_COALESCE_BACKEND_H

#ifdef CONFIG_NVMEVIRT_MIDLAYER
#include <linux/types.h>
#include <linux/llist.h>

struct co_shard;   /* opaque here; defined in coalesce.c */
struct l1_cqe;     /* defined in coalesce.h */
struct page;

/* Upper bound on consumer shards (matches CO_MAX_CORES in coalesce.c); lets an
 * async backend size per-shard completion state without a cross-file constant. */
#define CO_BACKEND_MAX_SHARDS 32
#define CO_MAX_DEVS 8        /* backend devices (real SSDs) for striping */
#define CO_NIL 0xFFFFFFFFu   /* freelist / chain terminator */

/* ---- back end: one outstanding read of one backend page into one staging
 * buffer. `dev` is the backend device index (== nsid for the emulated backend;
 * a real device slot once multi-SSD striping lands, §3.3.6). `cookie` routes
 * the async completion back to its pending slot (shard<<32 | slot). ---- */
struct co_bread {
	u64		line_tag;	/* page-aligned byte offset in feature space */
	u32		dev;		/* backend device index                      */
	u32		len;		/* == coalesce_page                          */
	void		*staging;	/* kernel-virtual staging buffer             */
	struct page	*spage;		/* same buffer as struct page (bio path)     */
	u64		cookie;		/* shard<<32 | pending-slot index            */
};

struct co_backend_ops {
	const char *name;
	/* true  = issue_read fills staging and completes synchronously; the caller
	 *         may deliver the rows inline on return (backend_sync).
	 * false = issue_read only starts the read; staging is NOT ready on return.
	 *         Completion arrives later via nvmev_co_backend_done() and the rows
	 *         are delivered by co_reap (backend_model, backend_nvme). This is
	 *         independent of ->poll: a real device (poll==NULL) still completes
	 *         asynchronously from its IRQ. */
	bool sync;
	/* paths: nr_devs backend device paths (NULL for the emulated backend). */
	int  (*init)(u32 nr_devs, const char *const *paths,
		     u32 inflight_cap, u32 page_bytes);
	void (*exit)(void);
	/* M0 (backend_sync): synchronous — on return 0, rd->staging holds the
	 *   page and the caller may deliver immediately; <0 = error.
	 * M1+ (async providers): 0 = accepted, completion will follow via
	 *   nvmev_co_backend_done(); -EAGAIN = QD full (retry after reaping);
	 *   <0 = hard error. */
	int  (*issue_read)(struct co_bread *rd);
	/* Optional: drive deferred completions for one shard (called each lap of
	 * that shard's consumer loop). NULL when completions are interrupt- or
	 * inline-driven (backend_sync, real NVMe). backend_model uses it to fire
	 * its per-shard timer ring. */
	void (*poll)(u32 shard);
};

/* Async completion upcall (M1+). May run in any context including hardirq:
 * it only validates the cookie (gen + state) and llist_add()s onto the owning
 * shard's done_list — no alloc, no sleep, no GPU mapping (liveness rule L5).
 * A stale/duplicate completion (slot timed-out and reused) is silently dropped.
 * cookie layout: [39:32]=shard  [31:16]=gen  [15:0]=pending-slot. */
void nvmev_co_backend_done(u64 cookie, int status);

/* Cookie encode/decode — shared so a backend provider can recover the owning
 * shard (e.g. to route a completion to that shard's I/O queue-pair). */
#define CO_COOKIE(shard, gen, slot) \
	(((u64)(shard) << 32) | ((u64)(u16)(gen) << 16) | (u64)(u16)(slot))
#define CO_CK_SLOT(c)	((u32)((c) & 0xffff))
#define CO_CK_GEN(c)	((u32)(((c) >> 16) & 0xffff))
#define CO_CK_SHARD(c)	((u32)(((c) >> 32) & 0xff))

/* Backend providers: emulated DRAM (backend_model.c, async or sync per
 * emul_async) and real NVMe via the block layer (backend_nvme.c). coalesce.c
 * picks based on the coalesce_backend device-path list. */
const struct co_backend_ops *nvmev_co_backend_model(void);
const struct co_backend_ops *nvmev_co_backend_nvme(void);
/* Middle-layer-owned NVMe I/O QP (Stage A): selected when a coalesce_backend
 * path is prefixed "pci:" (a PCIe BDF). Owns the SSD directly, poll-mode. */
const struct co_backend_ops *nvmev_co_backend_direct(void);

/* ---- front end: how one finished feature row reaches the GPU. ---- */
struct co_waiter;	/* defined below; the GPU-pull deliver op takes one */

struct co_deliver_ops {
	const char *name;
	/* Copy/DMA one row to a GPU-physical destination. dst_phys == 0 =
	 * measurement-only (no copy). Returns 0 on success, 2 if dst_phys is
	 * not in a registered GPU region. May sleep. */
	int  (*row)(u64 dst_phys, const void *src, u32 len);
	/* Post one completion to shard s's L1 completion ring. */
	void (*cqe)(struct co_shard *s, const struct l1_cqe *c);
	/* Optional: deliver one finished row as a single unit. If set, used
	 * instead of row()+cqe(). GPU-pull enqueues a gather descriptor carrying
	 * req_id+dst+staging_off and the GPU posts the CQE after delivery, so the
	 * two-call row/cqe split (which exposes req_id only to cqe) does not fit.
	 * page_src is the kernel VA of the staged page; the waiter carries the
	 * within-page offset, dst, and req_id. May sleep (bounded ring wait). */
	void (*deliver)(struct co_shard *s, const struct co_waiter *w,
			const void *page_src, u16 status);
};

/* ---- fan-out state carried across the split phase (issue → reap). Populated
 * at M1; the types are defined now so M1 does not touch this header. ---- */
struct co_waiter {		/* copied from the sorted window at issue */
	u64	req_id;
	u64	dst_phys;
	u16	off_in_page;	/* (node_id*row) & (page-1)               */
	u16	_pad;
	u32	next;		/* per-shard freelist / per-page chain; CO_NIL = end */
};

struct co_pend_page {
	u64		line_tag;
	void		*staging;
	struct page	*spage;
	u32		first_waiter;	/* index into the per-shard waiter arena */
	u16		n_waiters;
	u8		dev;
	u8		_pad;
	/* GPU-pull (b4): a page's staging must not be reused until the GPU has
	 * gathered every descriptor that references it. gpu_wm = the descriptor
	 * ring head just after this page's descriptors were enqueued; the page is
	 * held on gpu_held_next until tail >= gpu_wm. Unused in CPU WC-store mode. */
	u32		gpu_wm;
	u32		gpu_held_next;	/* reaper-local held list; CO_NIL = end */
	/* (gen<<8)|state, transitioned by a single cmpxchg so a completion from
	 * an interrupt (M3 bio bi_end_io) can't race the consumer's timeout sweep
	 * or a slot reuse. gen (16b, carried in the cookie) rejects a late/dup
	 * completion of a reused slot; state is the lifecycle. */
	u32		genstate;
	u64		issue_ns;
	struct llist_node done_node;	/* completion → shard done-list          */
};

enum co_pend_state { CO_PEND_FREE = 0, CO_PEND_ISSUED, CO_PEND_DONE, CO_PEND_ERROR };

#endif /* CONFIG_NVMEVIRT_MIDLAYER */
#endif /* NVMEV_COALESCE_BACKEND_H */
