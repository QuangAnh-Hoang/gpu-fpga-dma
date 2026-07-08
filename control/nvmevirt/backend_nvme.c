// SPDX-License-Identifier: GPL-2.0
/*
 * backend_nvme — real NVMe backend for co_backend_ops via the block layer.
 *
 * Each unique coalesced page becomes one REQ_OP_READ bio of the staging page,
 * submitted to a backing block device (a real SSD namespace). Completion is
 * asynchronous: bi_end_io fires from the block layer's IRQ/softirq context and
 * calls nvmev_co_backend_done(), which is hardirq-safe (cmpxchg on the pending
 * slot). No ->poll — the consumer's co_reap() drains the llist the completion
 * pushes to.
 *
 * line_tag is the page-aligned byte offset of the feature page; the feature
 * store is assumed to begin at LBA 0, so bi_sector = line_tag >> 9 (512 B
 * logical sectors). Multi-device striping (co_map_node, M5) selects rd->dev.
 * v1 is read-only (feature gather); the write-fold path is M8.
 */
#ifdef CONFIG_NVMEVIRT_MIDLAYER

#include <linux/atomic.h>
#include <linux/bio.h>
#include <linux/blkdev.h>
#include <linux/blk_types.h>
#include <linux/delay.h>
#include <linux/errno.h>
#include <linux/mm.h>		/* vmalloc_to_page: staging is a vmalloc slice (b2) */
#include <linux/vmalloc.h>

#include "nvmev.h"
#include "coalesce_backend.h"

static struct block_device *nv_bdev[CO_MAX_DEVS];
static u32 nv_ndev;
/* Outstanding bios. exit() must wait these out before the caller frees the
 * staging pages, or a still-running DMA would write freed memory. */
static atomic_t nv_inflight = ATOMIC_INIT(0);

static int backend_nvme_init(u32 nr_devs, const char *const *paths,
			     u32 inflight_cap, u32 page_bytes)
{
	u32 i;

	(void)inflight_cap; (void)page_bytes;
	nv_ndev = 0;
	for (i = 0; i < nr_devs && i < CO_MAX_DEVS; i++) {
		struct block_device *bdev =
			blkdev_get_by_path(paths[i], FMODE_READ, NULL);

		if (IS_ERR(bdev)) {
			NVMEV_ERROR("backend_nvme: open %s failed (%ld)\n",
				    paths[i], PTR_ERR(bdev));
			goto err;
		}
		nv_bdev[i] = bdev;
		nv_ndev++;
		NVMEV_INFO("backend_nvme: dev%u = %s\n", i, paths[i]);
	}
	if (!nv_ndev)
		return -EINVAL;
	return 0;
err:
	while (i--) {
		blkdev_put(nv_bdev[i], FMODE_READ);
		nv_bdev[i] = NULL;
	}
	nv_ndev = 0;
	return -ENODEV;
}

static void backend_nvme_exit(void)
{
	int guard = 10000;	/* ~10 s @ 1 ms */
	u32 i;

	/* Wait for outstanding DMAs to finish before the caller frees staging. */
	while (atomic_read(&nv_inflight) > 0 && guard-- > 0)
		msleep(1);
	if (atomic_read(&nv_inflight) > 0)
		NVMEV_ERROR("backend_nvme: %d bios still in flight at exit\n",
			    atomic_read(&nv_inflight));

	for (i = 0; i < nv_ndev; i++) {
		if (nv_bdev[i])
			blkdev_put(nv_bdev[i], FMODE_READ);
		nv_bdev[i] = NULL;
	}
	nv_ndev = 0;
}

/* Block-layer completion — may run in hardirq/softirq. */
static void co_bio_end(struct bio *bio)
{
	u64 cookie = (u64)(uintptr_t)bio->bi_private;

	nvmev_co_backend_done(cookie, blk_status_to_errno(bio->bi_status));
	bio_put(bio);
	atomic_dec(&nv_inflight);
}

static int backend_nvme_issue_read(struct co_bread *rd)
{
	/* Staging is a slice of the shared g_staging vmalloc region (b2), so its
	 * physical pages are NOT contiguous: add the buffer to the bio page by page
	 * via vmalloc_to_page (npg == 1 for the default coalesce_page == PAGE_SIZE). */
	u32 npg = (rd->len + PAGE_SIZE - 1) >> PAGE_SHIFT;
	u8 *kva = (u8 *)rd->staging;
	struct bio *bio;
	u32 off = 0;

	if (rd->dev >= nv_ndev)
		return -EIO;

	/* GFP_NOIO: on the IO submission path — must not recurse into the block
	 * layer via reclaim. One bio per coalesced read (npg bvecs). */
	bio = bio_alloc(GFP_NOIO, npg);
	if (!bio)
		return -EAGAIN;			/* transient: caller retries next lap */

	bio_set_dev(bio, nv_bdev[rd->dev]);
	bio->bi_iter.bi_sector = rd->line_tag >> 9;	/* 512 B logical sectors */
	bio->bi_opf = REQ_OP_READ;
	bio->bi_end_io = co_bio_end;
	bio->bi_private = (void *)(uintptr_t)rd->cookie;

	while (off < rd->len) {
		u32 chunk = min_t(u32, PAGE_SIZE, rd->len - off);

		if (bio_add_page(bio, vmalloc_to_page(kva + off), chunk, 0) != chunk) {
			bio_put(bio);
			return -EIO;
		}
		off += chunk;
	}
	atomic_inc(&nv_inflight);	/* balanced in co_bio_end */
	submit_bio(bio);
	return 0;	/* completion follows via co_bio_end -> nvmev_co_backend_done */
}

static const struct co_backend_ops backend_nvme = {
	.name = "nvme-bio",
	.sync = false,		/* async: staging filled by DMA, done via bi_end_io */
	.init = backend_nvme_init,
	.exit = backend_nvme_exit,
	.issue_read = backend_nvme_issue_read,
	.poll = NULL,	/* completions arrive via bi_end_io, not polling */
};

const struct co_backend_ops *nvmev_co_backend_nvme(void)
{
	return &backend_nvme;
}

#endif /* CONFIG_NVMEVIRT_MIDLAYER */
