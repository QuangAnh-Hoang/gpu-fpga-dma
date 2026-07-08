// SPDX-License-Identifier: GPL-2.0
/*
 * backend_direct — a middle-layer-OWNED NVMe I/O queue-pair for co_backend_ops
 * (docs/south-side-direct-nvme-plan.md, Stage A).
 *
 * Instead of going through the stock nvme driver + block layer (backend_nvme.c),
 * this provider owns the target SSD's PCIe function directly: it brings the
 * controller up (reset + admin queue), creates one poll-mode I/O SQ/CQ pair per
 * consumer shard, submits each coalesced page read as a raw NVMe READ SQE, and
 * reaps completions by polling the CQ — no bio, no per-I/O interrupt. The SSD
 * DMAs each page into the shard's staging buffer (host memory); GPU-pull then
 * delivers only rows (off-GPU staging, DS2). The CPU keeps only the lightweight
 * control half (build SQE / ring doorbell / poll CQ); the Stage-B FPGA later
 * deletes even that. This is the executable reference for the FPGA initiator FSM.
 *
 * The NVMe protocol logic is standard (cf. drivers/nvme/host/pci.c and the
 * vendored libnvm at control/vendor/bam as sequence references); it reuses the
 * in-tree command/register definitions in nvme.h / pci.h. It does NOT link
 * libnvm (whose queue layer is userspace/CUDA).
 *
 * Requirements / v1 assumptions:
 *   - The target SSD must be UNBOUND from the stock nvme driver before load
 *     (echo <BDF> > /sys/bus/pci/drivers/nvme/unbind). One controller owner.
 *   - amd_iommu=off (platform invariant): PRP = page_to_phys(staging page); the
 *     DMA/bus address equals the physical address. If an IOMMU is ever enabled,
 *     switch to dma_map_page() per staging page.
 *   - coalesce_page == PAGE_SIZE (one PRP entry / read). Larger pages need a PRP
 *     list (deferred).
 *   - namespace id 1, 512 B logical blocks (matches backend_nvme.c); the feature
 *     store begins at LBA 0 so slba = line_tag >> 9.
 *
 * Selected by prefixing a coalesce_backend path with "pci:", e.g.
 *   coalesce_backend=pci:0000:c1:00.0
 */
#ifdef CONFIG_NVMEVIRT_MIDLAYER

#include <linux/atomic.h>
#include <linux/delay.h>
#include <linux/dma-mapping.h>
#include <linux/errno.h>
#include <linux/io.h>
#include <linux/io-64-nonatomic-lo-hi.h>
#include <linux/iopoll.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/slab.h>
#include <linux/spinlock.h>
#include <linux/vmalloc.h>

#include "nvme.h"
#include "nvmev.h"
#include "coalesce_backend.h"

/* NVMe controller register offsets (BAR0). */
#define NVME_REG_CAP	0x00	/* 64-bit */
#define NVME_REG_CC	0x14
#define NVME_REG_CSTS	0x1c
#define NVME_REG_AQA	0x24
#define NVME_REG_ASQ	0x28	/* 64-bit */
#define NVME_REG_ACQ	0x30	/* 64-bit */

#define DIRECT_ADMIN_QD	32
#define DIRECT_BAR_MAP	0x4000	/* enough to reach I/O doorbells for CO_BACKEND_MAX_SHARDS */

/* A0a self-test: if >= 0, read this LBA once at init and dump it to dmesg. */
static int direct_test_lba = -1;
module_param(direct_test_lba, int, 0444);
MODULE_PARM_DESC(direct_test_lba, "backend_direct bring-up self-test LBA (-1=off)");

static void direct_poll(u32 sh);	/* reaper-side CQ drain (defined below) */

/* One poll-mode I/O queue-pair, owned by one shard on one controller. */
struct dqp {
	struct nvme_command	*sq;		/* dma-coherent, io_qd entries    */
	struct nvme_completion	*cq;
	dma_addr_t		sq_dma, cq_dma;
	void __iomem		*sq_db, *cq_db;	/* doorbell registers             */
	u16			qid;
	u16			sq_tail;	/* issuer-owned                   */
	u16			cq_head;	/* reaper-owned                   */
	u8			cq_phase;	/* expected phase bit (starts 1)  */
	bool			created;	/* published to the reaper        */
	/* cid pool: cid ∈ [0, io_qd-1); free stack + cid→cookie, both guarded
	 * by lock (issuer allocates, reaper frees — different threads when
	 * coalesce_split=1). in-flight is capped at io_qd-1 so the SQ ring and
	 * the CQ never overflow. */
	spinlock_t		lock;
	u16			*free;
	u16			free_top;
	u64			*cid_cookie;
	dma_addr_t		*cid_dma;	/* streaming DMA map of each in-flight
						 * staging page (IOMMU-safe; unmapped
						 * on completion — dma_unmap syncs it
						 * for the CPU). */
};

/* One owned NVMe controller. */
struct dctrl {
	struct pci_dev	*pdev;
	void __iomem	*bar;
	struct nvme_command	*asq;		/* admin SQ (dma-coherent)        */
	struct nvme_completion	*acq;		/* admin CQ                       */
	dma_addr_t	asq_dma, acq_dma;
	void __iomem	*asq_db, *acq_db;
	u16		asq_tail, acq_head;
	u8		acq_phase;
	u32		dstrd;			/* doorbell stride (CAP.DSTRD)    */
	u32		io_qd;			/* I/O queue depth                */
	struct mutex	admin_lock;		/* serialize admin commands       */
	struct dqp	qp[CO_BACKEND_MAX_SHARDS];	/* one per shard, lazily created  */
	bool		up;
};

static struct dctrl dctrls[CO_MAX_DEVS];
static u32 nv_ndev;
static u32 nv_page_bytes = PAGE_SIZE;	/* read len (== coalesce_page); unmap size */
/* Total outstanding reads across all QPs; exit() drains this before the caller
 * frees the staging pool (a still-running DMA would write freed memory). */
static atomic_t nv_inflight = ATOMIC_INIT(0);

static inline void __iomem *db_ptr(struct dctrl *c, u16 qid, bool cq)
{
	return c->bar + 0x1000 + (2 * qid + (cq ? 1 : 0)) * (4u << c->dstrd);
}

/* Submit one admin command and poll the admin CQ for its completion.
 * Returns the NVMe status code (0 = success) or a negative errno on timeout. */
static int admin_sync(struct dctrl *c, struct nvme_command *cmd, u32 *result)
{
	struct nvme_completion *cqe;
	u16 cid, want_head;
	u16 status = 0;
	int loops;

	mutex_lock(&c->admin_lock);
	cid = c->asq_tail;
	cmd->common.command_id = cpu_to_le16(cid);
	c->asq[c->asq_tail] = *cmd;
	c->asq_tail = (c->asq_tail + 1) % DIRECT_ADMIN_QD;
	wmb();
	writel(c->asq_tail, c->asq_db);

	want_head = c->acq_head;
	cqe = &c->acq[want_head];
	for (loops = 0; loops < 500000; loops++) {	/* ~5 s @ 10 us */
		if ((le16_to_cpu(READ_ONCE(cqe->status)) & 1) == c->acq_phase)
			goto done;
		udelay(10);
	}
	mutex_unlock(&c->admin_lock);
	NVMEV_ERROR("backend_direct: admin cmd opc=0x%x timed out\n",
		    cmd->common.opcode);
	return -ETIMEDOUT;
done:
	rmb();
	status = le16_to_cpu(cqe->status) >> 1;
	if (result)
		*result = le32_to_cpu(cqe->result0);
	c->acq_head = (c->acq_head + 1) % DIRECT_ADMIN_QD;
	if (c->acq_head == 0)
		c->acq_phase ^= 1;
	writel(c->acq_head, c->acq_db);
	mutex_unlock(&c->admin_lock);
	return status & 0x7ff;		/* SC | SCT, DNR bit masked off */
}

/* Controller reset + admin queue bring-up. */
static int ctrl_bringup(struct dctrl *c)
{
	u64 cap;
	u32 cc, csts;
	int rc;

	cap = lo_hi_readq(c->bar + NVME_REG_CAP);
	c->dstrd = (cap >> 32) & 0xf;
	c->io_qd = min_t(u32, c->io_qd, (u32)(cap & 0xffff) + 1);
	c->asq_db = db_ptr(c, 0, false);
	c->acq_db = db_ptr(c, 0, true);

	/* Disable, wait not-ready. */
	writel(0, c->bar + NVME_REG_CC);
	rc = readl_poll_timeout(c->bar + NVME_REG_CSTS, csts,
				!(csts & NVME_CSTS_RDY), 1000, 2000000);
	if (rc) {
		NVMEV_ERROR("backend_direct: ctrl stuck ready on disable\n");
		return rc;
	}

	/* Admin queues (physically-contiguous coherent memory). */
	c->asq = dma_alloc_coherent(&c->pdev->dev,
				    DIRECT_ADMIN_QD * sizeof(*c->asq),
				    &c->asq_dma, GFP_KERNEL);
	c->acq = dma_alloc_coherent(&c->pdev->dev,
				    DIRECT_ADMIN_QD * sizeof(*c->acq),
				    &c->acq_dma, GFP_KERNEL);
	if (!c->asq || !c->acq)
		return -ENOMEM;
	c->asq_tail = c->acq_head = 0;
	c->acq_phase = 1;			/* phase toggles to 1 on first CQE */

	writel(((DIRECT_ADMIN_QD - 1) << 16) | (DIRECT_ADMIN_QD - 1),
	       c->bar + NVME_REG_AQA);
	lo_hi_writeq(c->asq_dma, c->bar + NVME_REG_ASQ);
	lo_hi_writeq(c->acq_dma, c->bar + NVME_REG_ACQ);

	/* Enable: 4 KB pages (MPS=0), NVM command set, RR arbitration,
	 * 64 B SQE / 16 B CQE. */
	cc = NVME_CC_ENABLE | NVME_CC_CSS_NVM | (0 << NVME_CC_MPS_SHIFT) |
	     NVME_CC_ARB_RR | NVME_CC_IOSQES | NVME_CC_IOCQES;
	writel(cc, c->bar + NVME_REG_CC);
	rc = readl_poll_timeout(c->bar + NVME_REG_CSTS, csts,
				(csts & NVME_CSTS_RDY), 1000, 2000000);
	if (rc) {
		NVMEV_ERROR("backend_direct: ctrl not ready after enable\n");
		return rc;
	}

	/* Ask for enough I/O queues (0-based counts in dword11). One per shard;
	 * request the max and let lazy creation use what we need. */
	{
		struct nvme_command f = {};
		u32 nq = CO_BACKEND_MAX_SHARDS;

		f.features.opcode = nvme_admin_set_features;
		f.features.fid = cpu_to_le32(NVME_FEAT_NUM_QUEUES);
		f.features.dword11 = cpu_to_le32(((nq - 1) << 16) | (nq - 1));
		rc = admin_sync(c, &f, NULL);
		if (rc)		/* non-fatal: controller reports its own max */
			NVMEV_INFO("backend_direct: SET num-queues status 0x%x\n", rc);
	}

	c->up = true;
	NVMEV_INFO("backend_direct: %s up (dstrd=%u io_qd=%u)\n",
		   pci_name(c->pdev), c->dstrd, c->io_qd);
	return 0;
}

/* Create shard `sh`'s poll-mode I/O SQ/CQ on demand (issuer thread). */
static int qp_create(struct dctrl *c, u32 sh)
{
	struct dqp *q = &c->qp[sh];
	struct nvme_command cmd;
	u16 i, qid = sh + 1;
	int rc;

	q->sq = dma_alloc_coherent(&c->pdev->dev,
				   c->io_qd * sizeof(*q->sq),
				   &q->sq_dma, GFP_KERNEL);
	q->cq = dma_alloc_coherent(&c->pdev->dev,
				   c->io_qd * sizeof(*q->cq),
				   &q->cq_dma, GFP_KERNEL);
	q->free = kmalloc_array(c->io_qd, sizeof(*q->free), GFP_KERNEL);
	q->cid_cookie = kmalloc_array(c->io_qd, sizeof(*q->cid_cookie),
				      GFP_KERNEL);
	q->cid_dma = kmalloc_array(c->io_qd, sizeof(*q->cid_dma), GFP_KERNEL);
	if (!q->sq || !q->cq || !q->free || !q->cid_cookie || !q->cid_dma) {
		rc = -ENOMEM;
		goto fail;
	}
	q->qid = qid;
	q->sq_tail = q->cq_head = 0;
	q->cq_phase = 1;
	q->sq_db = db_ptr(c, qid, false);
	q->cq_db = db_ptr(c, qid, true);
	spin_lock_init(&q->lock);
	/* Reserve one slot: max in-flight = io_qd-1 so SQ/CQ never fill. */
	for (i = 0; i < c->io_qd - 1; i++)
		q->free[i] = i;
	q->free_top = c->io_qd - 1;

	/* CREATE_CQ then CREATE_SQ (poll-mode: no IRQ_ENABLED). */
	memset(&cmd, 0, sizeof(cmd));
	cmd.create_cq.opcode = nvme_admin_create_cq;
	cmd.create_cq.prp1 = cpu_to_le64(q->cq_dma);
	cmd.create_cq.cqid = cpu_to_le16(qid);
	cmd.create_cq.qsize = cpu_to_le16(c->io_qd - 1);
	cmd.create_cq.cq_flags = cpu_to_le16(NVME_QUEUE_PHYS_CONTIG);
	rc = admin_sync(c, &cmd, NULL);
	if (rc) {
		NVMEV_ERROR("backend_direct: CREATE_CQ qid=%u status 0x%x\n",
			    qid, rc);
		rc = -EIO;
		goto fail;
	}

	memset(&cmd, 0, sizeof(cmd));
	cmd.create_sq.opcode = nvme_admin_create_sq;
	cmd.create_sq.prp1 = cpu_to_le64(q->sq_dma);
	cmd.create_sq.sqid = cpu_to_le16(qid);
	cmd.create_sq.qsize = cpu_to_le16(c->io_qd - 1);
	cmd.create_sq.sq_flags = cpu_to_le16(NVME_QUEUE_PHYS_CONTIG);
	cmd.create_sq.cqid = cpu_to_le16(qid);
	rc = admin_sync(c, &cmd, NULL);
	if (rc) {
		NVMEV_ERROR("backend_direct: CREATE_SQ qid=%u status 0x%x\n",
			    qid, rc);
		rc = -EIO;
		goto fail_delcq;
	}

	smp_store_release(&q->created, true);	/* publish to the reaper */
	return 0;

fail_delcq:
	memset(&cmd, 0, sizeof(cmd));
	cmd.delete_queue.opcode = nvme_admin_delete_cq;
	cmd.delete_queue.qid = cpu_to_le16(qid);
	admin_sync(c, &cmd, NULL);
fail:
	if (q->sq)
		dma_free_coherent(&c->pdev->dev, c->io_qd * sizeof(*q->sq),
				  q->sq, q->sq_dma);
	if (q->cq)
		dma_free_coherent(&c->pdev->dev, c->io_qd * sizeof(*q->cq),
				  q->cq, q->cq_dma);
	kfree(q->free);
	kfree(q->cid_cookie);
	kfree(q->cid_dma);
	memset(q, 0, sizeof(*q));
	return rc;
}

static void qp_destroy(struct dctrl *c, u32 sh)
{
	struct dqp *q = &c->qp[sh];
	struct nvme_command cmd;

	if (!q->created)
		return;
	memset(&cmd, 0, sizeof(cmd));
	cmd.delete_queue.opcode = nvme_admin_delete_sq;
	cmd.delete_queue.qid = cpu_to_le16(q->qid);
	admin_sync(c, &cmd, NULL);
	memset(&cmd, 0, sizeof(cmd));
	cmd.delete_queue.opcode = nvme_admin_delete_cq;
	cmd.delete_queue.qid = cpu_to_le16(q->qid);
	admin_sync(c, &cmd, NULL);

	dma_free_coherent(&c->pdev->dev, c->io_qd * sizeof(*q->sq),
			  q->sq, q->sq_dma);
	dma_free_coherent(&c->pdev->dev, c->io_qd * sizeof(*q->cq),
			  q->cq, q->cq_dma);
	kfree(q->free);
	kfree(q->cid_cookie);
	kfree(q->cid_dma);
	memset(q, 0, sizeof(*q));
}

/* A0a self-test: read one LBA into a scratch buffer through a temp QP and dump
 * the first bytes to dmesg — validates bring-up without the coalesce path. */
static void direct_selftest(struct dctrl *c)
{
	struct dqp *q;
	void *buf;
	dma_addr_t buf_dma;
	struct nvme_command *sqe;
	struct nvme_completion *cqe;
	int loops;

	if (direct_test_lba < 0)
		return;
	if (qp_create(c, 0)) {
		NVMEV_ERROR("backend_direct: selftest qp_create failed\n");
		return;
	}
	q = &c->qp[0];
	buf = dma_alloc_coherent(&c->pdev->dev, PAGE_SIZE, &buf_dma, GFP_KERNEL);
	if (!buf)
		return;

	sqe = &q->sq[q->sq_tail];
	memset(sqe, 0, sizeof(*sqe));
	sqe->rw.opcode = nvme_cmd_read;
	sqe->rw.command_id = cpu_to_le16(0);
	sqe->rw.nsid = cpu_to_le32(1);
	sqe->rw.prp1 = cpu_to_le64(buf_dma);
	sqe->rw.slba = cpu_to_le64(direct_test_lba);
	sqe->rw.length = cpu_to_le16((PAGE_SIZE >> 9) - 1);
	q->sq_tail = (q->sq_tail + 1) % c->io_qd;
	wmb();
	writel(q->sq_tail, q->sq_db);

	cqe = &q->cq[q->cq_head];
	for (loops = 0; loops < 500000; loops++) {
		if ((le16_to_cpu(READ_ONCE(cqe->status)) & 1) == q->cq_phase)
			break;
		udelay(10);
	}
	rmb();
	NVMEV_INFO("backend_direct: selftest LBA %d status=0x%x bytes=%*ph\n",
		   direct_test_lba, le16_to_cpu(cqe->status) >> 1, 16, buf);
	q->cq_head = (q->cq_head + 1) % c->io_qd;
	if (q->cq_head == 0)
		q->cq_phase ^= 1;
	writel(q->cq_head, q->cq_db);

	dma_free_coherent(&c->pdev->dev, PAGE_SIZE, buf, buf_dma);
	qp_destroy(c, 0);
}

/* Parse "pci:DDDD:BB:DD.F" (or "DDDD:BB:DD.F") → open + bring up the controller. */
static int ctrl_open(struct dctrl *c, const char *path, u32 io_qd)
{
	unsigned int dom, bus, dev, fn;
	struct pci_dev *pdev;
	int rc;

	if (!strncmp(path, "pci:", 4))
		path += 4;
	if (sscanf(path, "%x:%x:%x.%x", &dom, &bus, &dev, &fn) != 4) {
		NVMEV_ERROR("backend_direct: bad BDF '%s' (want DDDD:BB:DD.F)\n",
			    path);
		return -EINVAL;
	}
	pdev = pci_get_domain_bus_and_slot(dom, bus, PCI_DEVFN(dev, fn));
	if (!pdev) {
		NVMEV_ERROR("backend_direct: %04x:%02x:%02x.%x not found\n",
			    dom, bus, dev, fn);
		return -ENODEV;
	}
	c->pdev = pdev;
	mutex_init(&c->admin_lock);
	c->io_qd = io_qd;

	rc = pci_enable_device(pdev);
	if (rc) {
		NVMEV_ERROR("backend_direct: enable %s failed\n", pci_name(pdev));
		goto put;
	}
	pci_set_master(pdev);
	pci_intx(pdev, 0);	/* poll-mode: no legacy INTx, no MSI-X armed */
	rc = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(64));
	if (rc)
		goto disable;
	rc = pci_request_region(pdev, 0, "nvmev_direct");
	if (rc) {
		NVMEV_ERROR("backend_direct: BAR0 busy on %s (unbind stock nvme?)\n",
			    pci_name(pdev));
		goto disable;
	}
	c->bar = ioremap(pci_resource_start(pdev, 0), DIRECT_BAR_MAP);
	if (!c->bar) {
		rc = -ENOMEM;
		goto release;
	}
	rc = ctrl_bringup(c);
	if (rc)
		goto unmap;

	direct_selftest(c);
	return 0;

unmap:
	iounmap(c->bar);
	c->bar = NULL;
release:
	pci_release_region(pdev, 0);
disable:
	pci_disable_device(pdev);
put:
	pci_dev_put(pdev);
	c->pdev = NULL;
	return rc;
}

static void ctrl_close(struct dctrl *c)
{
	u32 sh;

	if (!c->pdev)
		return;
	for (sh = 0; sh < CO_BACKEND_MAX_SHARDS; sh++)
		qp_destroy(c, sh);
	if (c->up)
		writel(0, c->bar + NVME_REG_CC);	/* disable controller */
	if (c->asq)
		dma_free_coherent(&c->pdev->dev,
				  DIRECT_ADMIN_QD * sizeof(*c->asq),
				  c->asq, c->asq_dma);
	if (c->acq)
		dma_free_coherent(&c->pdev->dev,
				  DIRECT_ADMIN_QD * sizeof(*c->acq),
				  c->acq, c->acq_dma);
	if (c->bar)
		iounmap(c->bar);
	pci_release_region(c->pdev, 0);
	pci_disable_device(c->pdev);
	pci_dev_put(c->pdev);
	memset(c, 0, sizeof(*c));
}

/* ---- co_backend_ops ---- */

static int backend_direct_init(u32 nr_devs, const char *const *paths,
			       u32 inflight_cap, u32 page_bytes)
{
	u32 io_qd = roundup_pow_of_two(max_t(u32, inflight_cap, 4));
	u32 i;
	int rc;

	if (page_bytes != PAGE_SIZE) {
		NVMEV_ERROR("backend_direct: coalesce_page %u != PAGE_SIZE (PRP-list not supported)\n",
			    page_bytes);
		return -EINVAL;
	}
	nv_page_bytes = page_bytes;	/* read/unmap size for the streaming DMA maps */
	nv_ndev = 0;
	for (i = 0; i < nr_devs && i < CO_MAX_DEVS; i++) {
		rc = ctrl_open(&dctrls[i], paths[i], io_qd);
		if (rc)
			goto err;
		nv_ndev++;
	}
	if (!nv_ndev)
		return -EINVAL;
	return 0;
err:
	while (i--)
		ctrl_close(&dctrls[i]);
	nv_ndev = 0;
	return rc;
}

static void backend_direct_exit(void)
{
	int guard = 10000;	/* ~10 s @ 1 ms */
	u32 i;

	/* Drain outstanding reads before the caller frees staging. The reaper
	 * threads are already stopped at this point (coalesce teardown order),
	 * so poll() won't run — drain by reaping here. direct_poll(sh) sweeps
	 * every controller's shard-sh QP, so call it once per shard. */
	while (atomic_read(&nv_inflight) > 0 && guard-- > 0) {
		u32 sh;

		for (sh = 0; sh < CO_BACKEND_MAX_SHARDS; sh++)
			direct_poll(sh);
		msleep(1);
	}
	if (atomic_read(&nv_inflight) > 0)
		NVMEV_ERROR("backend_direct: %d reads still in flight at exit\n",
			    atomic_read(&nv_inflight));
	for (i = 0; i < nv_ndev; i++)
		ctrl_close(&dctrls[i]);
	nv_ndev = 0;
}

static int backend_direct_issue_read(struct co_bread *rd)
{
	u32 sh = CO_CK_SHARD(rd->cookie);
	struct dctrl *c;
	struct dqp *q;
	struct nvme_command *sqe;
	unsigned long flags;
	dma_addr_t dma;
	u16 cid;

	if (rd->dev >= nv_ndev)
		return -EIO;
	c = &dctrls[rd->dev];
	q = &c->qp[sh];

	if (unlikely(!q->created)) {		/* lazy create on first use */
		int rc = qp_create(c, sh);

		if (rc)
			return rc;
	}

	spin_lock_irqsave(&q->lock, flags);
	if (!q->free_top) {			/* QD full: retry next lap */
		spin_unlock_irqrestore(&q->lock, flags);
		return -EAGAIN;
	}
	cid = q->free[--q->free_top];
	q->cid_cookie[cid] = rd->cookie;
	spin_unlock_irqrestore(&q->lock, flags);

	/* Streaming DMA map of the staging page (proper DMA-API; IOMMU-safe, and
	 * == page_to_phys under amd_iommu=off). dma_unmap in direct_poll() syncs it
	 * for the CPU on completion. */
	dma = dma_map_page(&c->pdev->dev, rd->spage, 0, rd->len, DMA_FROM_DEVICE);
	if (dma_mapping_error(&c->pdev->dev, dma)) {
		spin_lock_irqsave(&q->lock, flags);
		q->free[q->free_top++] = cid;		/* return the cid */
		spin_unlock_irqrestore(&q->lock, flags);
		return -EIO;
	}
	q->cid_dma[cid] = dma;

	sqe = &q->sq[q->sq_tail];		/* issuer-owned tail */
	memset(sqe, 0, sizeof(*sqe));
	sqe->rw.opcode = nvme_cmd_read;
	sqe->rw.command_id = cpu_to_le16(cid);
	sqe->rw.nsid = cpu_to_le32(1);
	sqe->rw.prp1 = cpu_to_le64(dma);
	sqe->rw.slba = cpu_to_le64(rd->line_tag >> 9);	/* 512 B LBAs */
	sqe->rw.length = cpu_to_le16((rd->len >> 9) - 1);
	q->sq_tail = (q->sq_tail + 1) % c->io_qd;
	/* Count before the doorbell: the SSD may complete and the reaper may
	 * decrement before this returns, so incrementing after would go negative. */
	atomic_inc(&nv_inflight);
	wmb();
	writel(q->sq_tail, q->sq_db);
	return 0;	/* completion follows via poll() -> nvmev_co_backend_done */
}

/* Reap shard `sh`'s completions across all controllers (reaper thread). */
static void direct_poll(u32 sh)
{
	u32 i;

	for (i = 0; i < nv_ndev; i++) {
		struct dqp *q = &dctrls[i].qp[sh];
		bool advanced = false;

		if (!smp_load_acquire(&q->created))
			continue;
		for (;;) {
			struct nvme_completion *cqe = &q->cq[q->cq_head];
			u16 st = le16_to_cpu(READ_ONCE(cqe->status));
			unsigned long flags;
			u64 cookie;
			u16 cid;

			if ((st & 1) != q->cq_phase)
				break;		/* no new completion */
			rmb();
			cid = le16_to_cpu(cqe->command_id);
			cookie = q->cid_cookie[cid];
			/* End the streaming map (syncs the DMA'd page for the CPU
			 * before the reaper/GPU consumes staging). */
			dma_unmap_page(&dctrls[i].pdev->dev, q->cid_dma[cid],
				       nv_page_bytes, DMA_FROM_DEVICE);

			q->cq_head++;
			if (q->cq_head == dctrls[i].io_qd) {
				q->cq_head = 0;
				q->cq_phase ^= 1;
			}
			advanced = true;

			spin_lock_irqsave(&q->lock, flags);
			q->free[q->free_top++] = cid;
			spin_unlock_irqrestore(&q->lock, flags);

			atomic_dec(&nv_inflight);
			nvmev_co_backend_done(cookie, (st >> 1) ? -EIO : 0);
		}
		if (advanced)
			writel(q->cq_head, q->cq_db);
	}
}

static const struct co_backend_ops backend_direct = {
	.name = "direct",
	.sync = false,		/* async: SSD DMAs staging, reaped by poll() */
	.init = backend_direct_init,
	.exit = backend_direct_exit,
	.issue_read = backend_direct_issue_read,
	.poll = direct_poll,
};

const struct co_backend_ops *nvmev_co_backend_direct(void)
{
	return &backend_direct;
}

#endif /* CONFIG_NVMEVIRT_MIDLAYER */
