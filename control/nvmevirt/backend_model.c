// SPDX-License-Identifier: GPL-2.0
/*
 * backend_model — emulated (DRAM-backed) providers for co_backend_ops.
 *
 *  backend_sync  (M0): synchronous copy of one backend page from the DRAM
 *      namespace into staging; the caller delivers inline. Behaviour-preserving
 *      reference (QD-1). Selected with emul_async=0.
 *
 *  backend_model (M1, default): the same copy, but the completion is *deferred*
 *      to coalesce_read_ns later via a per-shard FIFO timer ring drained from
 *      the shard's consumer loop (->poll). Concurrent in-flight pages model a
 *      QD-N device (assessment §5.1: this is the QD-1→QD-N semantic change that
 *      replaces the old inline latency spin). Fires nvmev_co_backend_done().
 *
 * Read latency is coalesce_read_ns (owned by coalesce.c, shared here); with a
 * constant latency the ring's due times are monotonic, so a FIFO drains exactly
 * in completion order. issue_read copies DRAM→staging exactly like backend_sync,
 * so delivered rows are identical to M0 — only the *timing* differs.
 */
#ifdef CONFIG_NVMEVIRT_MIDLAYER

#include <linux/errno.h>
#include <linux/moduleparam.h>
#include <linux/random.h>
#include <linux/sched/clock.h>
#include <linux/slab.h>
#include <linux/string.h>

#include "nvmev.h"
#include "coalesce_backend.h"

extern unsigned int coalesce_read_ns;	/* defined in coalesce.c */

static unsigned int emul_async = 1;	/* 1 = backend_model (async), 0 = backend_sync */
module_param(emul_async, uint, 0444);
MODULE_PARM_DESC(emul_async, "emulated backend: 1=async timer model (default), 0=synchronous");

/* Fault injection (parts-per-million of completions) for M2 liveness testing:
 * drop = never complete (exercises the coalesce timeout sweep), err = complete
 * with error status (exercises error CQE delivery). */
static unsigned int emul_drop_ppm;
module_param(emul_drop_ppm, uint, 0644);
MODULE_PARM_DESC(emul_drop_ppm, "backend_model: drop this many completions per million (0=off)");
static unsigned int emul_err_ppm;
module_param(emul_err_ppm, uint, 0644);
MODULE_PARM_DESC(emul_err_ppm, "backend_model: error this many completions per million (0=off)");

/* ---- shared: copy one backend page from the DRAM namespace into staging ---- */
static int emul_fill(struct co_bread *rd)
{
	u32 nsid = rd->dev;

	if (nsid >= nvmev_vdev->nr_ns || !nvmev_vdev->ns[nsid].mapped)
		return -EIO;
	memcpy(rd->staging, (const u8 *)nvmev_vdev->ns[nsid].mapped + rd->line_tag,
	       rd->len);
	return 0;
}

/* ---- backend_sync (M0): synchronous ---- */
static int backend_sync_init(u32 nr_devs, const char *const *paths,
			     u32 inflight_cap, u32 page_bytes)
{
	(void)nr_devs; (void)paths; (void)inflight_cap; (void)page_bytes;
	return 0;
}
static void backend_sync_exit(void) {}
static int backend_sync_issue_read(struct co_bread *rd)
{
	return emul_fill(rd);	/* staging ready on return; caller delivers inline */
}
static const struct co_backend_ops backend_sync = {
	.name = "sync-dram",
	.sync = true,		/* fills staging + completes on issue_read return */
	.init = backend_sync_init,
	.exit = backend_sync_exit,
	.issue_read = backend_sync_issue_read,
	.poll = NULL,
};

/* ---- backend_model (M1): async per-shard FIFO timer ring ---- */
struct bm_ent { u64 due_ns; u64 cookie; };
struct bm_ring {
	struct bm_ent	*e;
	u32		cap;		/* power of two                       */
	u32		head, tail;	/* FIFO: monotonic due => in order    */
};
static struct bm_ring bm_ring[CO_BACKEND_MAX_SHARDS];

static int backend_model_init(u32 nr_devs, const char *const *paths,
			      u32 inflight_cap, u32 page_bytes)
{
	u32 cap = roundup_pow_of_two(inflight_cap ? inflight_cap : 64);
	u32 k;

	(void)nr_devs; (void)paths; (void)page_bytes;
	for (k = 0; k < CO_BACKEND_MAX_SHARDS; k++) {
		bm_ring[k].e = kcalloc(cap, sizeof(struct bm_ent), GFP_KERNEL);
		if (!bm_ring[k].e) {
			while (k--)
				kfree(bm_ring[k].e);
			return -ENOMEM;
		}
		bm_ring[k].cap = cap;
		bm_ring[k].head = bm_ring[k].tail = 0;
	}
	return 0;
}

static void backend_model_exit(void)
{
	u32 k;

	for (k = 0; k < CO_BACKEND_MAX_SHARDS; k++) {
		kfree(bm_ring[k].e);
		bm_ring[k].e = NULL;
	}
}

static int backend_model_issue_read(struct co_bread *rd)
{
	u32 shard = (u32)(rd->cookie >> 32);
	struct bm_ring *r;
	int rc;

	if (shard >= CO_BACKEND_MAX_SHARDS)
		return -EINVAL;
	r = &bm_ring[shard];
	if (r->tail - r->head >= r->cap)	/* ring full (== inflight cap) */
		return -EAGAIN;
	rc = emul_fill(rd);			/* device "DMA" into staging */
	if (rc)
		return rc;
	r->e[r->tail & (r->cap - 1)] = (struct bm_ent){
		.due_ns = local_clock() + (u64)coalesce_read_ns,
		.cookie = rd->cookie,
	};
	r->tail++;
	return 0;	/* completion follows via ->poll -> nvmev_co_backend_done */
}

static void backend_model_poll(u32 shard)
{
	struct bm_ring *r;
	u64 now;

	if (shard >= CO_BACKEND_MAX_SHARDS)
		return;
	r = &bm_ring[shard];
	now = local_clock();
	while (r->head != r->tail) {
		struct bm_ent *e = &r->e[r->head & (r->cap - 1)];
		u64 cookie;

		if (e->due_ns > now)
			break;			/* FIFO: nothing later is due yet */
		cookie = e->cookie;
		r->head++;

		if (emul_drop_ppm || emul_err_ppm) {
			u32 roll = prandom_u32_max(1000000u);

			if (roll < emul_drop_ppm)
				continue;	/* dropped: the slot will time out */
			if (roll < emul_drop_ppm + emul_err_ppm) {
				nvmev_co_backend_done(cookie, -EIO);
				continue;
			}
		}
		nvmev_co_backend_done(cookie, 0);
	}
}

static const struct co_backend_ops backend_model = {
	.name = "model-dram-async",
	.sync = false,		/* deferred completion via ->poll timer ring */
	.init = backend_model_init,
	.exit = backend_model_exit,
	.issue_read = backend_model_issue_read,
	.poll = backend_model_poll,
};

const struct co_backend_ops *nvmev_co_backend_model(void)
{
	return emul_async ? &backend_model : &backend_sync;
}

#endif /* CONFIG_NVMEVIRT_MIDLAYER */
