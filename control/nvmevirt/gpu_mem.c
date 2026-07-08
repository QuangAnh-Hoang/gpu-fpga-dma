// SPDX-License-Identifier: GPL-2.0-only

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/io.h>
#include <linux/sort.h>
#include <linux/seq_file.h>
#include <linux/uaccess.h>
#include <linux/version.h>

#include <nv-p2p.h>

#include "nvmev.h"
#include "gpu_mem.h"

static struct nvmev_gpu_mem_mgr gpu_mgr;

/* ========================================================================
 * Sorted lookup table with RCU-protected reads and binary search
 * ======================================================================== */

static int __entry_cmp(const void *a, const void *b)
{
	const struct nvmev_gpu_map_entry *ea = a;
	const struct nvmev_gpu_map_entry *eb = b;

	if (ea->phys_addr < eb->phys_addr)
		return -1;
	if (ea->phys_addr > eb->phys_addr)
		return 1;
	return 0;
}

static int __bsearch_page(const struct nvmev_gpu_lookup_table *table,
			  u64 phys_addr)
{
	int lo = 0, hi = (int)table->nr_entries - 1;

	while (lo <= hi) {
		int mid = lo + (hi - lo) / 2;
		u64 start = table->entries[mid].phys_addr;

		if (phys_addr < start)
			hi = mid - 1;
		else if (phys_addr >= start + NVMEV_GPU_PAGE_SIZE)
			lo = mid + 1;
		else
			return mid;
	}
	return -1;
}

/*
 * Rebuild the sorted lookup table from all valid regions.
 * Must be called with gpu_mgr.lock held.
 */
static void __rebuild_lookup_table(void)
{
	struct nvmev_gpu_lookup_table *old_table, *new_table;
	struct nvmev_gpu_region *region;
	unsigned int total_pages = 0;
	unsigned int idx = 0;

	list_for_each_entry(region, &gpu_mgr.regions, list) {
		if (region->valid)
			total_pages += region->nr_pages;
	}

	if (total_pages == 0) {
		old_table = rcu_dereference_protected(gpu_mgr.lookup,
				lockdep_is_held(&gpu_mgr.lock));
		rcu_assign_pointer(gpu_mgr.lookup, NULL);
		if (old_table)
			kfree_rcu(old_table, rcu);
		return;
	}

	new_table = kmalloc(struct_size(new_table, entries, total_pages),
			    GFP_KERNEL);
	if (!new_table) {
		NVMEV_ERROR("Failed to allocate GPU lookup table (%u entries)\n",
			    total_pages);
		return;
	}

	list_for_each_entry(region, &gpu_mgr.regions, list) {
		unsigned int i;

		if (!region->valid)
			continue;
		for (i = 0; i < region->nr_pages; i++) {
			new_table->entries[idx] = region->pages[i];
			idx++;
		}
	}
	new_table->nr_entries = idx;

	sort(new_table->entries, new_table->nr_entries,
	     sizeof(struct nvmev_gpu_map_entry), __entry_cmp, NULL);

	old_table = rcu_dereference_protected(gpu_mgr.lookup,
			lockdep_is_held(&gpu_mgr.lock));
	rcu_assign_pointer(gpu_mgr.lookup, new_table);
	if (old_table)
		kfree_rcu(old_table, rcu);
}

/* ========================================================================
 * GPU page mapping / unmapping
 * ======================================================================== */

static void __unmap_region_pages(struct nvmev_gpu_region *region)
{
	unsigned int i;

	if (!region->pages)
		return;

	for (i = 0; i < region->nr_pages; i++) {
		if (region->pages[i].kaddr) {
			iounmap(region->pages[i].kaddr);
			region->pages[i].kaddr = NULL;
		}
	}
}

static void __release_nvidia_pages(struct nvmev_gpu_region *region)
{
	if (!region->page_table)
		return;

	if (region->callback_fired) {
		nvidia_p2p_free_page_table(region->page_table);
	} else {
		nvidia_p2p_put_pages(0, 0, region->gpu_va, region->page_table);
	}
	region->page_table = NULL;
}

static void __free_region(struct nvmev_gpu_region *region)
{
	__unmap_region_pages(region);
	__release_nvidia_pages(region);
	kfree(region->pages);
	kfree(region);
}

/*
 * NVIDIA driver calls this when GPU memory is forcefully reclaimed (e.g.,
 * the owning CUDA context is destroyed). After this callback, the physical
 * pages are invalid but the page_table metadata must be freed by us.
 */
static void nvmev_gpu_free_callback(void *data)
{
	struct nvmev_gpu_region *region = data;

	NVMEV_INFO("GPU memory reclaimed by driver: va=0x%llx\n",
		   region->gpu_va);

	region->valid = false;
	region->callback_fired = true;

	/*
	 * We cannot modify the lookup table here (might be in interrupt context
	 * and we need the mutex). Mark invalid and let lookups check the valid
	 * flag on the entries. The explicit unregister or module exit will do
	 * the actual cleanup.
	 */
}

/* ========================================================================
 * Public API
 * ======================================================================== */

void __iomem *nvmev_gpu_kmap(u64 phys_addr)
{
	const struct nvmev_gpu_lookup_table *table;
	int idx;
	u64 offset_in_gpu_page;
	u64 aligned_offset;

	atomic64_inc(&gpu_mgr.nr_lookups);

	table = rcu_dereference(gpu_mgr.lookup);
	if (!table)
		return NULL;

	idx = __bsearch_page(table, phys_addr);
	if (idx < 0)
		return NULL;

	atomic64_inc(&gpu_mgr.nr_hits);

	offset_in_gpu_page = phys_addr - table->entries[idx].phys_addr;
	aligned_offset = offset_in_gpu_page & PAGE_MASK;

	return table->entries[idx].kaddr + aligned_offset;
}

bool nvmev_is_gpu_addr(u64 phys_addr)
{
	const struct nvmev_gpu_lookup_table *table;
	int idx;

	rcu_read_lock();
	table = rcu_dereference(gpu_mgr.lookup);
	idx = table ? __bsearch_page(table, phys_addr) : -1;
	rcu_read_unlock();

	return idx >= 0;
}

int nvmev_gpu_region_pages(u64 gpu_va, u64 len, u64 *out, u32 max, u32 *n_out)
{
	struct nvmev_gpu_region *region;
	int ret = -ENOENT;

	mutex_lock(&gpu_mgr.lock);
	list_for_each_entry(region, &gpu_mgr.regions, list) {
		u64 aligned_va;
		u32 start_pg, npg, i;

		if (!region->valid)
			continue;
		if (gpu_va < region->gpu_va ||
		    gpu_va >= region->gpu_va + region->size)
			continue;

		aligned_va = gpu_va & NVMEV_GPU_PAGE_MASK;
		start_pg = (u32)((aligned_va - region->gpu_va) >> NVMEV_GPU_PAGE_SHIFT);
		npg = (u32)DIV_ROUND_UP((gpu_va - aligned_va) + len,
					NVMEV_GPU_PAGE_SIZE);
		if (start_pg + npg > region->nr_pages)
			npg = region->nr_pages - start_pg;
		if (npg > max)
			npg = max;
		for (i = 0; i < npg; i++)
			out[i] = region->pages[start_pg + i].phys_addr;
		*n_out = npg;
		ret = 0;
		break;
	}
	mutex_unlock(&gpu_mgr.lock);
	return ret;
}

int nvmev_gpu_mem_register(u64 gpu_va, u64 size)
{
	struct nvmev_gpu_region *region;
	struct nvidia_p2p_page_table *page_table = NULL;
	u64 aligned_va, aligned_size;
	unsigned int i;
	int ret;

	aligned_va = gpu_va & NVMEV_GPU_PAGE_MASK;
	aligned_size = ALIGN(gpu_va + size, NVMEV_GPU_PAGE_SIZE) - aligned_va;

	region = kzalloc(sizeof(*region), GFP_KERNEL);
	if (!region)
		return -ENOMEM;

	region->gpu_va = aligned_va;
	region->size = aligned_size;
	region->valid = true;
	region->callback_fired = false;
	INIT_LIST_HEAD(&region->list);

	ret = nvidia_p2p_get_pages(0, 0, aligned_va, aligned_size,
				   &page_table, nvmev_gpu_free_callback,
				   region);
	if (ret) {
		NVMEV_ERROR("nvidia_p2p_get_pages failed: %d "
			    "(va=0x%llx size=%llu)\n",
			    ret, aligned_va, aligned_size);
		kfree(region);
		return ret;
	}

	if (!NVIDIA_P2P_PAGE_TABLE_VERSION_COMPATIBLE(page_table)) {
		NVMEV_ERROR("Incompatible NVIDIA P2P page table version\n");
		nvidia_p2p_put_pages(0, 0, aligned_va, page_table);
		kfree(region);
		return -EINVAL;
	}

	if (page_table->page_size != NVIDIA_P2P_PAGE_SIZE_64KB) {
		NVMEV_ERROR("Expected 64KB GPU pages, got page_size id=%u\n",
			    page_table->page_size);
		nvidia_p2p_put_pages(0, 0, aligned_va, page_table);
		kfree(region);
		return -EINVAL;
	}

	region->page_table = page_table;
	region->nr_pages = page_table->entries;
	region->pages = kcalloc(region->nr_pages,
				sizeof(struct nvmev_gpu_map_entry), GFP_KERNEL);
	if (!region->pages) {
		__release_nvidia_pages(region);
		kfree(region);
		return -ENOMEM;
	}

	for (i = 0; i < region->nr_pages; i++) {
		u64 pa = page_table->pages[i]->physical_address;

		region->pages[i].phys_addr = pa;
		region->pages[i].kaddr = ioremap_wc(pa, NVMEV_GPU_PAGE_SIZE);
		if (!region->pages[i].kaddr) {
			NVMEV_ERROR("ioremap_wc failed for GPU page %u "
				    "(pa=0x%llx)\n", i, pa);
			__unmap_region_pages(region);
			__release_nvidia_pages(region);
			kfree(region->pages);
			kfree(region);
			return -ENOMEM;
		}
	}

	mutex_lock(&gpu_mgr.lock);

	if (gpu_mgr.nr_regions >= NVMEV_GPU_MAX_REGIONS) {
		mutex_unlock(&gpu_mgr.lock);
		NVMEV_ERROR("Maximum GPU regions (%d) reached\n",
			    NVMEV_GPU_MAX_REGIONS);
		__free_region(region);
		return -ENOSPC;
	}

	list_add_tail(&region->list, &gpu_mgr.regions);
	gpu_mgr.nr_regions++;
	gpu_mgr.nr_pages_mapped += region->nr_pages;

	__rebuild_lookup_table();

	mutex_unlock(&gpu_mgr.lock);

	NVMEV_INFO("GPU memory registered: va=0x%llx size=%llu pages=%u\n",
		   aligned_va, aligned_size, region->nr_pages);

	return 0;
}

int nvmev_gpu_mem_unregister(u64 gpu_va)
{
	struct nvmev_gpu_region *region, *tmp;
	struct nvmev_gpu_region *found = NULL;

	gpu_va &= NVMEV_GPU_PAGE_MASK;

	mutex_lock(&gpu_mgr.lock);
	list_for_each_entry_safe(region, tmp, &gpu_mgr.regions, list) {
		if (region->gpu_va == gpu_va) {
			found = region;
			list_del(&region->list);
			gpu_mgr.nr_regions--;
			gpu_mgr.nr_pages_mapped -= region->nr_pages;
			break;
		}
	}

	if (found)
		__rebuild_lookup_table();

	mutex_unlock(&gpu_mgr.lock);

	if (!found)
		return -ENOENT;

	/*
	 * After rebuilding the lookup table, new lookups won't find this
	 * region. Wait for all existing RCU readers to finish before freeing
	 * the ioremap mappings they might be using.
	 */
	synchronize_rcu();
	__free_region(found);

	NVMEV_INFO("GPU memory unregistered: va=0x%llx\n", gpu_va);
	return 0;
}

/* ========================================================================
 * Auto-mapping cache for unregistered MMIO addresses
 *
 * When libnvm allocates GPU DMA buffers, the resulting PRP addresses
 * point to GPU BAR1. If the user didn't manually register these via
 * procfs, the I/O path would fall back to per-page memremap/memunmap
 * which is extremely slow (~100µs per 4KB page due to ioremap + TLB
 * invalidation on every call).
 *
 * This cache creates persistent ioremap_wc mappings on first access
 * and reuses them for all subsequent I/O to the same 256MB region.
 * 64 entries × 256MB = 16GB coverage; linear scan stays trivially fast.
 * ======================================================================== */

#define NVMEV_AUTOMAP_MAX	64

struct nvmev_automap_entry {
	u64		phys_base;	/* 256MB-aligned, 0 = empty */
	void __iomem	*kaddr;
};

static struct nvmev_automap_entry automap_cache[NVMEV_AUTOMAP_MAX];
static int automap_count;
static DEFINE_MUTEX(automap_mutex);
static atomic64_t automap_hits = ATOMIC_INIT(0);

void __iomem *nvmev_gpu_auto_map(u64 phys_addr)
{
	u64 phys_base = phys_addr & NVMEV_AUTOMAP_MASK;
	u64 offset = phys_addr - phys_base;
	void __iomem *kaddr;
	int i, count;

	count = READ_ONCE(automap_count);
	for (i = 0; i < count; i++) {
		if (READ_ONCE(automap_cache[i].phys_base) == phys_base) {
			kaddr = READ_ONCE(automap_cache[i].kaddr);
			if (kaddr) {
				atomic64_inc(&automap_hits);
				return kaddr + offset;
			}
		}
	}

	mutex_lock(&automap_mutex);

	for (i = 0; i < automap_count; i++) {
		if (automap_cache[i].phys_base == phys_base &&
		    automap_cache[i].kaddr) {
			kaddr = automap_cache[i].kaddr;
			mutex_unlock(&automap_mutex);
			return kaddr + offset;
		}
	}

	if (automap_count >= NVMEV_AUTOMAP_MAX) {
		mutex_unlock(&automap_mutex);
		NVMEV_ERROR("Auto-map cache full (%d entries)\n",
			    NVMEV_AUTOMAP_MAX);
		return NULL;
	}

	kaddr = ioremap_wc(phys_base, NVMEV_AUTOMAP_REGION);
	if (!kaddr) {
		mutex_unlock(&automap_mutex);
		NVMEV_ERROR("ioremap_wc failed for auto-map "
			    "(pa=0x%llx, size=%lu MB)\n",
			    phys_base, NVMEV_AUTOMAP_REGION >> 20);
		return NULL;
	}

	i = automap_count;
	WRITE_ONCE(automap_cache[i].phys_base, phys_base);
	WRITE_ONCE(automap_cache[i].kaddr, kaddr);
	smp_wmb();
	WRITE_ONCE(automap_count, i + 1);

	mutex_unlock(&automap_mutex);

	NVMEV_INFO("Auto-mapped MMIO region: phys=0x%llx size=%lu MB → %px (slot %d)\n",
		   phys_base, NVMEV_AUTOMAP_REGION >> 20, kaddr, i);

	return kaddr + offset;
}

static void __automap_cleanup(void)
{
	int i;

	for (i = 0; i < automap_count; i++) {
		if (automap_cache[i].kaddr) {
			iounmap(automap_cache[i].kaddr);
			automap_cache[i].kaddr = NULL;
		}
	}
	NVMEV_INFO("Auto-map cache cleaned up (%d entries, %lld hits)\n",
		   automap_count, atomic64_read(&automap_hits));
	automap_count = 0;
}

int nvmev_gpu_mem_init(void)
{
	INIT_LIST_HEAD(&gpu_mgr.regions);
	mutex_init(&gpu_mgr.lock);
	RCU_INIT_POINTER(gpu_mgr.lookup, NULL);
	gpu_mgr.nr_regions = 0;
	gpu_mgr.nr_pages_mapped = 0;
	atomic64_set(&gpu_mgr.nr_lookups, 0);
	atomic64_set(&gpu_mgr.nr_hits, 0);

	mutex_init(&automap_mutex);
	automap_count = 0;
	atomic64_set(&automap_hits, 0);

	NVMEV_INFO("GPU Direct I/O support initialized\n");
	return 0;
}

void nvmev_gpu_mem_exit(void)
{
	struct nvmev_gpu_region *region, *tmp;
	struct nvmev_gpu_lookup_table *table;
	LIST_HEAD(free_list);

	mutex_lock(&gpu_mgr.lock);

	table = rcu_dereference_protected(gpu_mgr.lookup,
			lockdep_is_held(&gpu_mgr.lock));
	rcu_assign_pointer(gpu_mgr.lookup, NULL);

	list_splice_init(&gpu_mgr.regions, &free_list);

	mutex_unlock(&gpu_mgr.lock);

	synchronize_rcu();

	if (table)
		kfree(table);

	list_for_each_entry_safe(region, tmp, &free_list, list) {
		list_del(&region->list);
		__free_region(region);
	}

	__automap_cleanup();

	NVMEV_INFO("GPU Direct I/O support cleaned up "
		   "(lookups=%lld hits=%lld)\n",
		   atomic64_read(&gpu_mgr.nr_lookups),
		   atomic64_read(&gpu_mgr.nr_hits));
}

/* ========================================================================
 * Procfs interface
 * ======================================================================== */

static int __gpu_mem_show(struct seq_file *m, void *v)
{
	struct nvmev_gpu_region *region;

	seq_printf(m, "# GPU Direct I/O Memory Manager\n");
	seq_printf(m, "# regions: %lu  pages_mapped: %lu  "
		   "lookups: %lld  hits: %lld\n",
		   gpu_mgr.nr_regions, gpu_mgr.nr_pages_mapped,
		   atomic64_read(&gpu_mgr.nr_lookups),
		   atomic64_read(&gpu_mgr.nr_hits));
	seq_printf(m, "#\n");
	seq_printf(m, "# Usage:\n");
	seq_printf(m, "#   echo 'pin <cuda_va_hex> <size>' > /proc/nvmev/gpu_mem\n");
	seq_printf(m, "#   echo 'unpin <cuda_va_hex>' > /proc/nvmev/gpu_mem\n");
	seq_printf(m, "#\n");

	mutex_lock(&gpu_mgr.lock);
	list_for_each_entry(region, &gpu_mgr.regions, list) {
		unsigned int i;

		seq_printf(m, "region gpu_va=0x%llx size=%llu pages=%u "
			   "valid=%d callback=%d\n",
			   region->gpu_va, region->size, region->nr_pages,
			   region->valid, region->callback_fired);

		for (i = 0; i < region->nr_pages; i++) {
			seq_printf(m, "  page[%u] phys=0x%llx kaddr=%px\n",
				   i, region->pages[i].phys_addr,
				   region->pages[i].kaddr);
		}
	}
	mutex_unlock(&gpu_mgr.lock);

	return 0;
}

static ssize_t __gpu_mem_write(struct file *file, const char __user *buf,
			       size_t len, loff_t *offp)
{
	char input[128];
	u64 va, size;
	int ret;
	size_t nr;

	if (len >= sizeof(input))
		return -EINVAL;

	nr = copy_from_user(input, buf, len);
	if (nr)
		return -EFAULT;
	input[len] = '\0';

	if (strncmp(input, "pin ", 4) == 0) {
		char *p = input + 4;

		va = simple_strtoull(p, &p, 0);
		while (*p == ' ' || *p == '\t')
			p++;
		size = simple_strtoull(p, &p, 0);

		if (!va || !size)
			return -EINVAL;

		ret = nvmev_gpu_mem_register(va, size);
		if (ret)
			return ret;
	} else if (strncmp(input, "unpin ", 6) == 0) {
		va = simple_strtoull(input + 6, NULL, 0);
		if (!va)
			return -EINVAL;

		ret = nvmev_gpu_mem_unregister(va);
		if (ret)
			return ret;
	} else {
		return -EINVAL;
	}

	return len;
}

static int __gpu_mem_open(struct inode *inode, struct file *file)
{
	return single_open(file, __gpu_mem_show, NULL);
}

#if LINUX_VERSION_CODE > KERNEL_VERSION(5, 0, 0)
static const struct proc_ops gpu_mem_proc_ops = {
	.proc_open	= __gpu_mem_open,
	.proc_read	= seq_read,
	.proc_write	= __gpu_mem_write,
	.proc_lseek	= seq_lseek,
	.proc_release	= single_release,
};
#else
static const struct file_operations gpu_mem_proc_ops = {
	.open		= __gpu_mem_open,
	.read		= seq_read,
	.write		= __gpu_mem_write,
	.llseek		= seq_lseek,
	.release	= single_release,
};
#endif

/* Phase 3: Export BAR0 physical address for GPU doorbell registration */
static int __bar_info_show(struct seq_file *m, void *v)
{
	if (!nvmev_vdev) {
		seq_printf(m, "# NVMeVirt device not initialized\n");
		return 0;
	}

	seq_printf(m, "# NVMeVirt BAR0 Information for GPU Doorbell Access\n");
	seq_printf(m, "#\n");
	seq_printf(m, "# To enable GPU-initiated doorbell writes:\n");
	seq_printf(m, "#   1. mmap bar0_phys via /dev/mem or libnvm helper\n");
	seq_printf(m, "#   2. cudaHostRegister(ptr, bar0_size, "
		   "cudaHostRegisterIoMemory)\n");
	seq_printf(m, "#   3. cudaHostGetDevicePointer(&devptr, ptr, 0)\n");
	seq_printf(m, "#   4. SQ doorbell[n] is at devptr + doorbell_offset "
		   "+ n*8\n");
	seq_printf(m, "#   5. CQ doorbell[n] is at devptr + doorbell_offset "
		   "+ n*8 + 4\n");
	seq_printf(m, "#\n");
	seq_printf(m, "bar0_phys=0x%lx\n", nvmev_vdev->config.memmap_start);
	seq_printf(m, "bar0_size=%lu\n", PAGE_SIZE * 2);
	seq_printf(m, "doorbell_offset=%u\n", (unsigned int)PAGE_SIZE);
	seq_printf(m, "nr_sq=%u\n", nvmev_vdev->nr_sq);
	seq_printf(m, "nr_cq=%u\n", nvmev_vdev->nr_cq);

	return 0;
}

static int __bar_info_open(struct inode *inode, struct file *file)
{
	return single_open(file, __bar_info_show, NULL);
}

#if LINUX_VERSION_CODE > KERNEL_VERSION(5, 0, 0)
static const struct proc_ops bar_info_proc_ops = {
	.proc_open	= __bar_info_open,
	.proc_read	= seq_read,
	.proc_lseek	= seq_lseek,
	.proc_release	= single_release,
};
#else
static const struct file_operations bar_info_proc_ops = {
	.open		= __bar_info_open,
	.read		= seq_read,
	.llseek		= seq_lseek,
	.release	= single_release,
};
#endif

static struct proc_dir_entry *gpu_mem_entry;
static struct proc_dir_entry *bar_info_entry;

int nvmev_gpu_procfs_init(struct proc_dir_entry *parent)
{
	gpu_mem_entry = proc_create("gpu_mem", 0664, parent,
				    &gpu_mem_proc_ops);
	if (!gpu_mem_entry)
		return -ENOMEM;

	bar_info_entry = proc_create("bar_info", 0444, parent,
				     &bar_info_proc_ops);
	if (!bar_info_entry) {
		remove_proc_entry("gpu_mem", parent);
		return -ENOMEM;
	}

	return 0;
}

void nvmev_gpu_procfs_exit(struct proc_dir_entry *parent)
{
	if (bar_info_entry)
		remove_proc_entry("bar_info", parent);
	if (gpu_mem_entry)
		remove_proc_entry("gpu_mem", parent);
}
