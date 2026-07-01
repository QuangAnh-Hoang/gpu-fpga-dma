// SPDX-License-Identifier: GPL-2.0
/*
 * Milestone F1b — NVMeVirt -> userspace FPGA daemon forward ring.
 *
 * Allocates a single vmalloc_user region laid out per the FQ/FCQ ABI, exposes it
 * mmap-able via /dev/nvmev_fpga, and provides SPSC push/pop with kernel
 * acquire/release barriers that interoperate with the daemon's __atomic
 * acquire/release accessors (gpu-fpga-dma/src/ise_fpga_ring.h). CPU<->CPU shared
 * memory only — no device DMA, no nvidia_p2p.
 */
#ifdef CONFIG_NVMEVIRT_FPGA

#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/miscdevice.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/spinlock.h>
#include <linux/string.h>
#include <linux/vmalloc.h>

#include "fpga_fwd.h"

static struct nvmev_fpga g_fpga;

/* Pending NVMe commands awaiting FPGA completion, indexed by req_id & mask. The
 * monotonic req_id never reuses a live slot because in-flight forwards are
 * bounded by the FQ depth (<< FPGA_PEND). Guarded by g_fwd_lock (single producer
 * = the NVMeVirt dispatcher; drain runs on the same thread). */
#define FPGA_PEND (1u << 16)
#define FPGA_PEND_MASK (FPGA_PEND - 1)
static struct fpga_pend g_pend[FPGA_PEND];
static u64 g_req_seq;
static DEFINE_SPINLOCK(g_fwd_lock);

static inline void *fq_base(struct ise_fwd_ring *r)
{
	return (char *)r + r->fq_off;
}
static inline void *fcq_base(struct ise_fwd_ring *r)
{
	return (char *)r + r->fcq_off;
}

bool nvmev_fpga_ready(void)
{
	return g_fpga.ready;
}

/* producer side (NVMeVirt): only writer of fq_head */
bool nvmev_fpga_fq_push(const struct ise_fwd_desc *d)
{
	struct ise_fwd_ring *r = g_fpga.ring;
	u32 h, t;

	if (unlikely(!g_fpga.ready))
		return false;
	h = READ_ONCE(r->fq_head);
	t = smp_load_acquire(&r->fq_tail);
	if (h - t >= r->fq_entries)
		return false; /* full */
	memcpy((char *)fq_base(r) + (size_t)(h & (r->fq_entries - 1)) * sizeof(*d),
	       d, sizeof(*d));
	smp_store_release(&r->fq_head, h + 1);
	return true;
}

/* consumer side (NVMeVirt): only writer of fcq_tail */
bool nvmev_fpga_fcq_pop(struct ise_fwd_cqe *c)
{
	struct ise_fwd_ring *r = g_fpga.ring;
	u32 h, t;

	if (unlikely(!g_fpga.ready))
		return false;
	t = READ_ONCE(r->fcq_tail);
	h = smp_load_acquire(&r->fcq_head);
	if (h == t)
		return false; /* empty */
	memcpy(c, (char *)fcq_base(r) + (size_t)(t & (r->fcq_entries - 1)) * sizeof(*c),
	       sizeof(*c));
	smp_store_release(&r->fcq_tail, t + 1);
	return true;
}

static void ring_layout(struct ise_fwd_ring *r, size_t bytes, u32 fq, u32 fcq)
{
	u32 hdr = (sizeof(*r) + 4095u) & ~4095u; /* page-align header */

	memset(r, 0, bytes);
	r->fq_entries = fq;
	r->fcq_entries = fcq;
	r->desc_size = sizeof(struct ise_fwd_desc);
	r->cqe_size = sizeof(struct ise_fwd_cqe);
	r->fq_off = hdr;
	r->fcq_off = hdr + fq * (u32)sizeof(struct ise_fwd_desc);
	r->region_bytes = (u32)bytes;
	r->version = ISF_VERSION;
	smp_store_release(&r->magic, ISF_MAGIC); /* publish last */
}

static int fpga_mmap(struct file *f, struct vm_area_struct *vma)
{
	unsigned long sz = vma->vm_end - vma->vm_start;

	if (!g_fpga.region)
		return -ENODEV;
	if (vma->vm_pgoff)
		return -EINVAL;
	if (sz > g_fpga.region_bytes)
		return -EINVAL;
	return remap_vmalloc_range(vma, g_fpga.region, 0);
}

static const struct file_operations fpga_fops = {
	.owner = THIS_MODULE,
	.mmap = fpga_mmap,
	.llseek = noop_llseek,
};

static struct miscdevice fpga_misc = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = "nvmev_fpga",
	.fops = &fpga_fops,
	.mode = 0666,
};

int nvmev_fpga_init(u32 fq, u32 fcq)
{
	size_t hdr = (sizeof(struct ise_fwd_ring) + 4095u) & ~4095UL;
	size_t bytes;
	int ret;

	BUILD_BUG_ON(sizeof(struct ise_fwd_desc) != 40);
	BUILD_BUG_ON(sizeof(struct ise_fwd_cqe) != 16);

	if (!fq || !fcq || (fq & (fq - 1)) || (fcq & (fcq - 1)))
		return -EINVAL; /* capacities must be non-zero powers of two */

	bytes = hdr + (size_t)fq * sizeof(struct ise_fwd_desc) +
		(size_t)fcq * sizeof(struct ise_fwd_cqe);

	g_fpga.region = vmalloc_user(bytes);
	if (!g_fpga.region)
		return -ENOMEM;
	g_fpga.region_bytes = bytes;
	g_fpga.ring = (struct ise_fwd_ring *)g_fpga.region;
	ring_layout(g_fpga.ring, bytes, fq, fcq);

	ret = misc_register(&fpga_misc);
	if (ret) {
		vfree(g_fpga.region);
		g_fpga.region = NULL;
		g_fpga.ring = NULL;
		return ret;
	}
	g_fpga.ready = true;
	pr_info("nvmev_fpga: /dev/nvmev_fpga ready (fq=%u fcq=%u, %zu bytes)\n", fq,
		fcq, bytes);
	return 0;
}

int nvmev_fpga_forward(u16 sqid, u16 cqid, u32 sq_entry, u16 cmd_id, u64 node_id,
		       u32 n_rows, u32 row_bytes, u64 gpu_dst, u16 flags)
{
	struct ise_fwd_desc d;
	unsigned long irqfl;
	u64 req;
	u32 slot;
	int ret = 0;

	if (!g_fpga.ready)
		return -ENODEV;

	spin_lock_irqsave(&g_fwd_lock, irqfl);
	req = g_req_seq;
	slot = (u32)req & FPGA_PEND_MASK;
	if (g_pend[slot].busy) { /* completions lagging badly */
		ret = -EBUSY;
		goto out;
	}
	d.req_id = req;
	d.arg0 = node_id;
	d.count = n_rows;
	d.row_bytes = row_bytes;
	d.gpu_dst = gpu_dst;
	d.sqid = sqid;
	d.cmd_id = cmd_id;
	d.flags = flags;
	if (!nvmev_fpga_fq_push(&d)) { /* FQ full */
		ret = -EAGAIN;
		goto out;
	}
	g_pend[slot].req_id = req;
	g_pend[slot].sq_entry = sq_entry;
	g_pend[slot].sqid = sqid;
	g_pend[slot].cqid = cqid;
	g_pend[slot].cmd_id = cmd_id;
	g_pend[slot].busy = 1;
	g_req_seq = req + 1;
out:
	spin_unlock_irqrestore(&g_fwd_lock, irqfl);
	return ret;
}

bool nvmev_fpga_pend_take(u64 req_id, struct fpga_pend *out)
{
	u32 slot = (u32)req_id & FPGA_PEND_MASK;
	unsigned long irqfl;
	bool ok = false;

	spin_lock_irqsave(&g_fwd_lock, irqfl);
	if (g_pend[slot].busy && g_pend[slot].req_id == req_id) {
		*out = g_pend[slot];
		g_pend[slot].busy = 0;
		ok = true;
	}
	spin_unlock_irqrestore(&g_fwd_lock, irqfl);
	return ok;
}

void nvmev_fpga_exit(void)
{
	if (!g_fpga.region)
		return;
	g_fpga.ready = false;
	misc_deregister(&fpga_misc);
	vfree(g_fpga.region);
	g_fpga.region = NULL;
	g_fpga.ring = NULL;
}

#endif /* CONFIG_NVMEVIRT_FPGA */
