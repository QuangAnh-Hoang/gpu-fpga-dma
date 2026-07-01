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
#ifdef CONFIG_NVMEVIRT_FPGA

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

#ifdef CONFIG_NVMEVIRT_GPU_DIRECT
#include <asm/fpu/api.h>
#endif

/* ---- tunables (swept) ---- */
static unsigned int coalesce_enable;      /* 0 = subsystem idle             */
module_param(coalesce_enable, uint, 0644);
static unsigned int coalesce_cores = 4;   /* N dedicated consumer cores     */
module_param(coalesce_cores, uint, 0444);
static unsigned int coalesce_cpu_base = 17;/* bind cores to base..base+N-1  */
module_param(coalesce_cpu_base, uint, 0444);
static unsigned int coalesce_ring = 65536;/* per-shard ring depth (pow2)    */
module_param(coalesce_ring, uint, 0444);
static unsigned int coalesce_window = 4096;/* W: flush depth                */
module_param(coalesce_window, uint, 0644);
static unsigned int coalesce_flush_us = 200;/* T: max age before flush      */
module_param(coalesce_flush_us, uint, 0644);
static unsigned int coalesce_page = 4096;
module_param(coalesce_page, uint, 0644);
static unsigned int coalesce_row = 512;   /* feature row bytes             */
module_param(coalesce_row, uint, 0644);
static unsigned int coalesce_nsid = 1;    /* 1-based backing namespace     */
module_param(coalesce_nsid, uint, 0644);
static unsigned int coalesce_sort;        /* enum co_sort (0=radix default)*/
module_param(coalesce_sort, uint, 0644);
static unsigned int coalesce_read_ns;     /* modeled backend latency per page read */
module_param(coalesce_read_ns, uint, 0644);
static unsigned int coalesce_no_merge;    /* 1 = charge per request (baseline, no coalescing) */
module_param(coalesce_no_merge, uint, 0644);

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
	struct task_struct *task;
	struct co_ent *win, *scr;
	u64 *htab;		/* hash-mode unique set */
	u32 htab_mask;
	u32 rcnt[256];		/* radix histogram (off-stack) */
	u32 win_n;
	u64 oldest_ns;
	int id;
} ____cacheline_aligned;

static void *g_region;
static size_t g_region_bytes;
static u32 g_shard_bytes;
static u32 g_nshards;
static u32 g_wcap;	/* allocated window capacity; caps the live coalesce_window */
static struct co_shard g_shard[CO_MAX_CORES];
static bool g_running;

/* ---- counters ---- */
static struct {
	atomic64_t flushes, reqs_in, unique_pages, page_merges;
	atomic64_t bytes_useful, bytes_read, max_window, sort_ns;
} g_stats;

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

/* Deliver one waiter's feature row from the DRAM backing into VRAM (P2P). */
static void deliver(struct co_shard *s, const struct co_ent *e)
{
	u32 nsid = coalesce_nsid ? coalesce_nsid - 1 : 0;
	u64 src_off = (u64)e->node_id * coalesce_row;
	const u8 *src;
	struct l1_cqe c = { .req_id = e->req_id, .status = 0 };

	if (nsid >= nvmev_vdev->nr_ns || !nvmev_vdev->ns[nsid].mapped) {
		c.status = 1;
		push_cqe(s, &c);
		return;
	}
	src = (const u8 *)nvmev_vdev->ns[nsid].mapped + src_off;

#ifdef CONFIG_NVMEVIRT_GPU_DIRECT
	if (e->dst_phys) {
		void __iomem *dst;

		rcu_read_lock();
		dst = nvmev_gpu_kmap(e->dst_phys);
		if (dst) {
			/* nvmev_gpu_kmap() rounds down to a PAGE_SIZE boundary and
			 * discards the low bits; add them back like __do_perform_io. */
			dst = (u8 __iomem *)dst + (e->dst_phys & (PAGE_SIZE - 1));
			kernel_fpu_begin();
			nvmev_memcpy_to_gpu(dst, src, coalesce_row);
			kernel_fpu_end();
			rcu_read_unlock();
		} else {
			rcu_read_unlock();
			c.status = 2; /* dst_phys not in a registered region */
		}
	}
	/* dst_phys==0: measurement-only (ring+coalesce), no copy */
#endif
	push_cqe(s, &c);
}

static void flush_shard(struct co_shard *s)
{
	u32 n = s->win_n, uniq;
	u64 t0, i;
	u32 page_shift;

	if (!n)
		return;
	page_shift = ilog2(coalesce_page);

	t0 = local_clock();
	switch (coalesce_sort) {
	case CO_SORT_COMPARISON:
		sort(s->win, n, sizeof(*s->win), cmp_tag, swp_ent);
		uniq = count_adjacent(s->win, n);
		break;
	case CO_SORT_HASH:
		uniq = hash_unique(s, n);
		break;
	case CO_SORT_RADIX:
	default:
		radix_by_page(s->win, s->scr, n, page_shift, s->rcnt);
		uniq = count_adjacent(s->win, n);
		break;
	}
	atomic64_add(local_clock() - t0, &g_stats.sort_ns);

	/* Modeled backend read cost: charge per unique page (coalesced) or per
	 * request (no_merge baseline). This is what turns the coalescing factor
	 * into a measurable throughput delta on the DRAM-backed emulator. */
	if (coalesce_read_ns) {
		u64 npages = coalesce_no_merge ? n : uniq;
		u64 remain = npages * (u64)coalesce_read_ns;

		/* Model the backend delay in slices, yielding between them: a long
		 * uninterrupted spin (large window x per-page latency) would otherwise
		 * hog the CPU and trip the soft-lockup watchdog. */
		while (remain) {
			u64 chunk = min_t(u64, remain, 200000); /* 200us slices */
			u64 t = local_clock() + chunk;

			while (local_clock() < t)
				cpu_relax();
			remain -= chunk;
			cond_resched();
		}
	}

	for (i = 0; i < n; i++)
		deliver(s, &s->win[i]);

	atomic64_inc(&g_stats.flushes);
	atomic64_add(n, &g_stats.reqs_in);
	atomic64_add(uniq, &g_stats.unique_pages);
	atomic64_add(n - uniq, &g_stats.page_merges);
	atomic64_add((u64)n * coalesce_row, &g_stats.bytes_useful);
	atomic64_add((u64)uniq * coalesce_page, &g_stats.bytes_read);
	if (n > (u32)atomic64_read(&g_stats.max_window))
		atomic64_set(&g_stats.max_window, n);

	s->win_n = 0;
}

static bool aged(struct co_shard *s)
{
	return local_clock() - s->oldest_ns >= (u64)coalesce_flush_us * 1000;
}

static int co_consumer(void *arg)
{
	struct co_shard *s = arg;

	while (!kthread_should_stop()) {
		struct l1_req req;
		bool got = false;

		u32 W = min_t(u32, coalesce_window, g_wcap); /* cap to allocation */

		if (!coalesce_enable) {
			schedule_timeout_interruptible(1);
			continue;
		}
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
			got = true;
		}
		if (s->win_n && (s->win_n >= W || rq_empty(s) || aged(s)))
			flush_shard(s);
		cond_resched();		/* always yield: never hog the CPU under load */
		if (!got)
			cpu_relax();
	}
	flush_shard(s);
	return 0;
}

/* ---- region + device ---- */
static int co_mmap(struct file *file, struct vm_area_struct *vma)
{
	unsigned long sz = vma->vm_end - vma->vm_start;

	if (sz > g_region_bytes)
		return -EINVAL;
	return remap_vmalloc_range(vma, g_region, 0);
}

static const struct file_operations co_dev_fops = {
	.owner = THIS_MODULE,
	.mmap = co_mmap,
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
	seq_printf(m, "cores           %u\n", g_nshards);
	seq_printf(m, "window          %u\n", coalesce_window);
	seq_printf(m, "flush_us        %u\n", coalesce_flush_us);
	seq_printf(m, "page            %u\n", coalesce_page);
	seq_printf(m, "row             %u\n", coalesce_row);
	seq_printf(m, "sort            %u\n", coalesce_sort);
	seq_printf(m, "read_ns         %u\n", coalesce_read_ns);
	seq_printf(m, "no_merge        %u\n", coalesce_no_merge);
	seq_printf(m, "flushes         %llu\n", atomic64_read(&g_stats.flushes));
	seq_printf(m, "reqs_in         %llu\n", reqs);
	seq_printf(m, "unique_pages    %llu\n", up);
	seq_printf(m, "backend_reads   %llu\n", up);
	seq_printf(m, "page_merges     %llu\n", atomic64_read(&g_stats.page_merges));
	seq_printf(m, "bytes_useful    %llu\n", useful);
	seq_printf(m, "bytes_read      %llu\n", read);
	seq_printf(m, "max_window      %llu\n", atomic64_read(&g_stats.max_window));
	seq_printf(m, "sort_ns         %llu\n", atomic64_read(&g_stats.sort_ns));
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

	g_nshards = clamp_val(coalesce_cores, 1, CO_MAX_CORES);
	g_shard_bytes = PAGE_ALIGN(hdr + seqb + rqb + cqb);
	g_region_bytes = (size_t)g_nshards * g_shard_bytes;
	/* Allocate the window to the full ring capacity so a live `coalesce_window`
	 * raise (sweeps) is always both safe and real (window can't exceed the ring). */
	g_wcap = ring;
	htab_sz = roundup_pow_of_two(g_wcap * 2);

	g_region = vmalloc_user(g_region_bytes);
	if (!g_region)
		return -ENOMEM;

	for (k = 0; k < g_nshards; k++) {
		struct co_shard *s = &g_shard[k];
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
		{ u32 i; for (i = 0; i < ring; i++) s->rq_seq[i] = i; } /* MPSC init */
		s->win = kvmalloc_array(g_wcap, sizeof(*s->win), GFP_KERNEL);
		s->scr = kvmalloc_array(g_wcap, sizeof(*s->scr), GFP_KERNEL);
		s->htab = kvmalloc_array(htab_sz, sizeof(u64), GFP_KERNEL);
		if (!s->win || !s->scr || !s->htab) {
			rc = -ENOMEM;
			goto err_free;
		}
		s->htab_mask = htab_sz - 1;
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
	misc_deregister(&co_misc);
err_free:
	for (k = 0; k < g_nshards; k++) {
		kvfree(g_shard[k].win);
		kvfree(g_shard[k].scr);
		kvfree(g_shard[k].htab);
	}
	vfree(g_region);
	g_region = NULL;
	return rc;
}

void nvmev_coalesce_exit(struct proc_dir_entry *proc_root)
{
	u32 k;

	if (!g_running)
		return;
	for (k = 0; k < g_nshards; k++)
		if (g_shard[k].task)
			kthread_stop(g_shard[k].task);
	remove_proc_entry("coalesce", proc_root);
	misc_deregister(&co_misc);
	for (k = 0; k < g_nshards; k++) {
		kvfree(g_shard[k].win);
		kvfree(g_shard[k].scr);
		kvfree(g_shard[k].htab);
	}
	vfree(g_region);
	g_region = NULL;
	g_running = false;
}

#endif /* CONFIG_NVMEVIRT_FPGA */
