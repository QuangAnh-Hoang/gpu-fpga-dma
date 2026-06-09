// SPDX-License-Identifier: GPL-2.0
/*
 * fpga_gdr — FPGA GPUDirect-RDMA helper driver (Milestone B, spike).
 *
 * Pins a cudaMalloc'd GPU range via the legacy nvidia_p2p API, maps it for the
 * Alveo (peer) to obtain PCIe bus addresses, and exposes raw read/write of the
 * XDMA shell's address_translator (HMSS / slave-bridge) registers so userspace
 * can point the bridge's translation table at those GPU bus addresses. A kernel
 * AXI master wired to HOST[0] then writes straight into GPU VRAM.
 *
 * Single pinned mapping at a time (sufficient for the de-risking spike).
 *
 * References: nv-p2p.h (NVIDIA driver src), gdrcopy gdrdrv.c, and XRT
 * xocl address_translator.c (register layout).
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/miscdevice.h>
#include <linux/uaccess.h>
#include <linux/mutex.h>
#include <linux/io.h>
#include <linux/pci.h>
#include <linux/slab.h>

#include "nv-p2p.h"
#include "fpga_gdr.h"

#define DRV "fpga_gdr"
#define GPU_PAGE_SHIFT 16
#define GPU_PAGE_SIZE  (1UL << GPU_PAGE_SHIFT)   /* 64 KB */
#define GPU_PAGE_MASK  (GPU_PAGE_SIZE - 1)

/* Alveo U55C user function and the address_translator BAR window (this host). */
static uint alveo_domain = 0x0000;
static uint alveo_bus = 0x21;
static uint alveo_devfn = PCI_DEVFN(0x00, 1);
static ulong hmss_phys = 0x13080081000UL;   /* address_translator IO start */
static ulong hmss_size = 0x1000;
module_param(alveo_bus, uint, 0444);
module_param(hmss_phys, ulong, 0444);
MODULE_PARM_DESC(hmss_phys, "physical base of the address_translator registers");

struct fpga_gdr_dev {
	struct mutex lock;
	void __iomem *hmss;                /* mapped address_translator regs   */
	struct pci_dev *peer;              /* Alveo user PF                    */

	/* current pin */
	bool pinned;
	u64 va, len;
	struct nvidia_p2p_page_table *pt;
	struct nvidia_p2p_dma_mapping *dm;
};

static struct fpga_gdr_dev *gdev;

/* ---- nvidia_p2p free callback: GPU memory went away underneath us ---- */
static void gdr_free_callback(void *data)
{
	struct fpga_gdr_dev *d = data;

	/* Called by the NVIDIA driver; this IS the teardown path. Do NOT
	 * call put_pages here — just release our mapping/table structs. */
	if (d->dm) {
		nvidia_p2p_free_dma_mapping(d->dm);
		d->dm = NULL;
	}
	if (d->pt) {
		nvidia_p2p_free_page_table(d->pt);
		d->pt = NULL;
	}
	d->pinned = false;
	pr_info(DRV ": free_callback fired; pin invalidated\n");
}

static void gdr_unpin_locked(struct fpga_gdr_dev *d)
{
	if (!d->pinned)
		return;
	if (d->dm) {
		nvidia_p2p_dma_unmap_pages(d->peer, d->pt, d->dm);
		d->dm = NULL;
	}
	if (d->pt) {
		/* put_pages also frees the page table */
		nvidia_p2p_put_pages(0, 0, d->va, d->pt);
		d->pt = NULL;
	}
	d->pinned = false;
}

static long gdr_do_pin(struct fpga_gdr_dev *d, struct fpga_gdr_pin __user *uarg)
{
	struct fpga_gdr_pin *p;
	long ret;
	u32 i;

	p = kzalloc(sizeof(*p), GFP_KERNEL);
	if (!p)
		return -ENOMEM;
	if (copy_from_user(p, uarg, sizeof(*p))) {
		ret = -EFAULT;
		goto out;
	}

	if ((p->va & GPU_PAGE_MASK) || (p->len & GPU_PAGE_MASK) || !p->len) {
		pr_err(DRV ": va/len must be 64KB-aligned (va=0x%llx len=0x%llx)\n",
		       p->va, p->len);
		ret = -EINVAL;
		goto out;
	}

	mutex_lock(&d->lock);
	gdr_unpin_locked(d);

	ret = nvidia_p2p_get_pages(0, 0, p->va, p->len, &d->pt,
				   gdr_free_callback, d);
	if (ret) {
		pr_err(DRV ": nvidia_p2p_get_pages failed: %ld\n", ret);
		goto unlock;
	}
	ret = nvidia_p2p_dma_map_pages(d->peer, d->pt, &d->dm);
	if (ret) {
		pr_err(DRV ": nvidia_p2p_dma_map_pages failed: %ld\n", ret);
		nvidia_p2p_put_pages(0, 0, p->va, d->pt);
		d->pt = NULL;
		goto unlock;
	}

	d->va = p->va;
	d->len = p->len;
	d->pinned = true;

	p->n_pages = d->dm->entries;
	p->page_size = GPU_PAGE_SIZE;
	if (p->n_pages > FPGA_GDR_MAX_PAGES)
		p->n_pages = FPGA_GDR_MAX_PAGES;
	for (i = 0; i < p->n_pages; i++)
		p->dma_addrs[i] = d->dm->dma_addresses[i];

	pr_info(DRV ": pinned va=0x%llx len=0x%llx pages=%u dma[0]=0x%llx\n",
		d->va, d->len, d->dm->entries, d->dm->dma_addresses[0]);
	ret = 0;
unlock:
	mutex_unlock(&d->lock);
out:
	if (!ret && copy_to_user(uarg, p, sizeof(*p)))
		ret = -EFAULT;
	kfree(p);
	return ret;
}

static long gdr_ioctl(struct file *f, unsigned int cmd, unsigned long arg)
{
	struct fpga_gdr_dev *d = gdev;
	struct fpga_gdr_reg reg;

	switch (cmd) {
	case FPGA_GDR_PIN:
		return gdr_do_pin(d, (struct fpga_gdr_pin __user *)arg);

	case FPGA_GDR_UNPIN:
		mutex_lock(&d->lock);
		gdr_unpin_locked(d);
		mutex_unlock(&d->lock);
		return 0;

	case FPGA_GDR_HMSS_WRITE:
		if (copy_from_user(&reg, (void __user *)arg, sizeof(reg)))
			return -EFAULT;
		if (reg.offset >= hmss_size - 3)
			return -EINVAL;
		writel(reg.value, d->hmss + reg.offset);
		return 0;

	case FPGA_GDR_HMSS_READ:
		if (copy_from_user(&reg, (void __user *)arg, sizeof(reg)))
			return -EFAULT;
		if (reg.offset >= hmss_size - 3)
			return -EINVAL;
		reg.value = readl(d->hmss + reg.offset);
		if (copy_to_user((void __user *)arg, &reg, sizeof(reg)))
			return -EFAULT;
		return 0;
	}
	return -ENOTTY;
}

static const struct file_operations gdr_fops = {
	.owner = THIS_MODULE,
	.unlocked_ioctl = gdr_ioctl,
};

static struct miscdevice gdr_misc = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = DRV,
	.fops = &gdr_fops,
};

static int __init gdr_init(void)
{
	int ret;

	gdev = kzalloc(sizeof(*gdev), GFP_KERNEL);
	if (!gdev)
		return -ENOMEM;
	mutex_init(&gdev->lock);

	gdev->peer = pci_get_domain_bus_and_slot(alveo_domain, alveo_bus,
						 alveo_devfn);
	if (!gdev->peer) {
		pr_err(DRV ": Alveo peer %04x:%02x.%x not found\n",
		       alveo_domain, alveo_bus, alveo_devfn);
		ret = -ENODEV;
		goto err_free;
	}

	gdev->hmss = ioremap(hmss_phys, hmss_size);
	if (!gdev->hmss) {
		pr_err(DRV ": ioremap(0x%lx) failed\n", hmss_phys);
		ret = -ENOMEM;
		goto err_pci;
	}

	ret = misc_register(&gdr_misc);
	if (ret)
		goto err_unmap;

	pr_info(DRV ": loaded. peer=%s hmss=0x%lx\n",
		pci_name(gdev->peer), hmss_phys);
	return 0;

err_unmap:
	iounmap(gdev->hmss);
err_pci:
	pci_dev_put(gdev->peer);
err_free:
	kfree(gdev);
	gdev = NULL;
	return ret;
}

static void __exit gdr_exit(void)
{
	mutex_lock(&gdev->lock);
	gdr_unpin_locked(gdev);
	mutex_unlock(&gdev->lock);

	misc_deregister(&gdr_misc);
	iounmap(gdev->hmss);
	pci_dev_put(gdev->peer);
	kfree(gdev);
	gdev = NULL;
	pr_info(DRV ": unloaded\n");
}

module_init(gdr_init);
module_exit(gdr_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("gpu-fpga-dma");
MODULE_DESCRIPTION("FPGA GPUDirect-RDMA helper: nvidia_p2p pin + HMSS poke");
