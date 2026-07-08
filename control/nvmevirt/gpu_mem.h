// SPDX-License-Identifier: GPL-2.0-only

#ifndef _NVMEV_GPU_MEM_H
#define _NVMEV_GPU_MEM_H

#include <linux/types.h>

#ifdef CONFIG_NVMEVIRT_GPU_DIRECT

#include <linux/list.h>
#include <linux/mutex.h>
#include <linux/rcupdate.h>
#include <linux/proc_fs.h>
#include <linux/atomic.h>

#define NVMEV_GPU_PAGE_SHIFT	16
#define NVMEV_GPU_PAGE_SIZE	(1UL << NVMEV_GPU_PAGE_SHIFT)
#define NVMEV_GPU_PAGE_MASK	(~(NVMEV_GPU_PAGE_SIZE - 1))
#define NVMEV_GPU_PAGE_OFFSET	(NVMEV_GPU_PAGE_SIZE - 1)

#define NVMEV_AUTOMAP_SHIFT	28			/* 256 MB per entry */
#define NVMEV_AUTOMAP_REGION	(1UL << NVMEV_AUTOMAP_SHIFT)
#define NVMEV_AUTOMAP_MASK	(~(NVMEV_AUTOMAP_REGION - 1))

#define NVMEV_GPU_MAX_REGIONS	64

struct nvidia_p2p_page_table;

struct nvmev_gpu_map_entry {
	u64		phys_addr;
	void __iomem	*kaddr;
};

struct nvmev_gpu_lookup_table {
	struct rcu_head		rcu;
	unsigned int		nr_entries;
	struct nvmev_gpu_map_entry entries[];
};

struct nvmev_gpu_region {
	struct list_head	list;
	u64			gpu_va;
	u64			size;

	struct nvidia_p2p_page_table *page_table;
	unsigned int		nr_pages;
	struct nvmev_gpu_map_entry *pages;
	bool			valid;
	bool			callback_fired;
};

struct nvmev_gpu_mem_mgr {
	struct nvmev_gpu_lookup_table __rcu *lookup;
	struct list_head	regions;
	struct mutex		lock;

	unsigned long		nr_regions;
	unsigned long		nr_pages_mapped;
	atomic64_t		nr_lookups;
	atomic64_t		nr_hits;
};

int nvmev_gpu_mem_init(void);
void nvmev_gpu_mem_exit(void);

int nvmev_gpu_mem_register(u64 gpu_va, u64 size);
int nvmev_gpu_mem_unregister(u64 gpu_va);

/*
 * Map a physical address that may reside in GPU BAR1 to a kernel pointer.
 * Returns the base of the 4KB-aligned host page within the GPU ioremap,
 * or NULL if the address is not in registered GPU memory.
 *
 * MUST be called inside rcu_read_lock(). The returned pointer is only valid
 * until the corresponding rcu_read_unlock().
 */
void __iomem *nvmev_gpu_kmap(u64 phys_addr);

bool nvmev_is_gpu_addr(u64 phys_addr);

/*
 * Fill @out with the GPU-page physical/bus addresses covering
 * [gpu_va, gpu_va+len) from the registered region containing gpu_va (up to
 * @max entries; count in *@n_out). GPU pages are NVMEV_GPU_PAGE_SIZE (64 KB).
 * Lets userspace resolve a per-row dst_phys without parsing /proc text.
 * Returns 0, -ENOENT if gpu_va is not in a registered region.
 */
int nvmev_gpu_region_pages(u64 gpu_va, u64 len, u64 *out, u32 max, u32 *n_out);

/*
 * Auto-map an MMIO physical address with ioremap_wc and cache the result.
 *
 * On first call for a given 256MB-aligned region, creates a persistent
 * ioremap_wc mapping covering the entire 256MB region. Subsequent calls
 * for addresses within the same region return the cached mapping with
 * the appropriate offset. The mapping persists until module unload.
 *
 * 64 entries × 256MB = 16GB maximum coverage.
 *
 * NOT protected by RCU; the returned pointer is valid for the module
 * lifetime. No unlock/unmap needed by the caller.
 */
void __iomem *nvmev_gpu_auto_map(u64 phys_addr);

int nvmev_gpu_procfs_init(struct proc_dir_entry *parent);
void nvmev_gpu_procfs_exit(struct proc_dir_entry *parent);

#else /* !CONFIG_NVMEVIRT_GPU_DIRECT */

static inline int nvmev_gpu_mem_init(void) { return 0; }
static inline void nvmev_gpu_mem_exit(void) {}

static inline void __iomem *nvmev_gpu_kmap(u64 phys_addr)
{
	return NULL;
}

static inline bool nvmev_is_gpu_addr(u64 phys_addr)
{
	return false;
}

static inline int nvmev_gpu_region_pages(u64 gpu_va, u64 len, u64 *out,
					 u32 max, u32 *n_out)
{
	return -ENODEV;
}

static inline void __iomem *nvmev_gpu_auto_map(u64 phys_addr)
{
	return NULL;
}

static inline int nvmev_gpu_procfs_init(struct proc_dir_entry *p) { return 0; }
static inline void nvmev_gpu_procfs_exit(struct proc_dir_entry *p) {}

#endif /* CONFIG_NVMEVIRT_GPU_DIRECT */
#endif /* _NVMEV_GPU_MEM_H */
