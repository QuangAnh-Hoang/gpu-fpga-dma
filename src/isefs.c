// SPDX-License-Identifier: GPL-2.0
/* libisefs — see isefs.h. Host (CPU) producer path over the L1 ring ABI. */
#define _GNU_SOURCE
#include "isefs.h"
#include "nvmev_l1.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

struct isefs_dst {
	uint64_t	va;
	uint64_t	*pg_phys;	/* per GPU-page bus addr */
	unsigned	npages;
	unsigned	page_shift;	/* 16 => 64 KB */
	size_t		cap_rows;
};

struct isefs {
	int		fd;
	void		*region;
	size_t		region_bytes;
	struct l1_ring	**shard;	/* nr_shards ring pointers into region */
	unsigned	nr_shards;
	unsigned	row, page;

	struct isefs_dst dst[ISEFS_MAX_DST];	/* destination-buffer slots */
};

static int write_proc(const char *path, const char *s)
{
	FILE *f = fopen(path, "w");

	if (!f)
		return -1;
	if (fputs(s, f) < 0) {
		fclose(f);
		return -1;
	}
	return fclose(f) ? -1 : 0;
}

isefs_t *isefs_open(const char *l1_dev, unsigned row_bytes, unsigned page_bytes)
{
	isefs_t *h;
	struct l1_ring hdr;
	void *map;
	unsigned k;

	h = calloc(1, sizeof(*h));
	if (!h)
		return NULL;
	h->row = row_bytes;
	h->page = page_bytes;
	h->fd = open(l1_dev, O_RDWR);
	if (h->fd < 0)
		goto err;

	/* mmap one page to read shard geometry, then the whole region. */
	map = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, h->fd, 0);
	if (map == MAP_FAILED)
		goto err;
	memcpy(&hdr, map, sizeof(hdr));
	munmap(map, 4096);
	if (hdr.magic != NVMEV_L1_MAGIC || !hdr.nr_shards) {
		errno = EPROTO;
		goto err;
	}
	h->nr_shards = hdr.nr_shards;
	h->region_bytes = (size_t)hdr.nr_shards * hdr.shard_bytes;
	h->region = mmap(NULL, h->region_bytes, PROT_READ | PROT_WRITE,
			 MAP_SHARED, h->fd, 0);
	if (h->region == MAP_FAILED)
		goto err;
	h->shard = calloc(h->nr_shards, sizeof(*h->shard));
	if (!h->shard)
		goto err;
	for (k = 0; k < h->nr_shards; k++)
		h->shard[k] = l1_shard(h->region, k);
	return h;
err:
	if (h->fd >= 0)
		close(h->fd);
	free(h);
	return NULL;
}

void isefs_close(isefs_t *h)
{
	char buf[64];
	unsigned s;

	if (!h)
		return;
	for (s = 0; s < ISEFS_MAX_DST; s++) {
		if (h->dst[s].va) {
			snprintf(buf, sizeof(buf), "unpin 0x%llx\n",
				 (unsigned long long)h->dst[s].va);
			write_proc("/proc/nvmev/gpu_mem", buf);
		}
		free(h->dst[s].pg_phys);
	}
	free(h->shard);
	if (h->region && h->region != MAP_FAILED)
		munmap(h->region, h->region_bytes);
	if (h->fd >= 0)
		close(h->fd);
	free(h);
}

int isefs_set_dst(isefs_t *h, unsigned slot, uint64_t dst_gpu_va,
		  size_t capacity_rows)
{
	struct isefs_dst *d;
	struct l1_translate t;
	char buf[96];
	size_t bytes = capacity_rows * h->row;
	unsigned cap_pages;

	if (!h || slot >= ISEFS_MAX_DST || !dst_gpu_va || !bytes)
		return -1;
	d = &h->dst[slot];

	/* pin the VRAM range for GPUDirect delivery */
	snprintf(buf, sizeof(buf), "pin 0x%llx %zu\n",
		 (unsigned long long)dst_gpu_va, bytes);
	if (write_proc("/proc/nvmev/gpu_mem", buf))
		return -1;

	cap_pages = (unsigned)(bytes / 4096) + 2; /* upper bound on 64 KB pages */
	free(d->pg_phys);
	d->pg_phys = calloc(cap_pages, sizeof(uint64_t));
	if (!d->pg_phys)
		return -1;

	memset(&t, 0, sizeof(t));
	t.gpu_va = dst_gpu_va;
	t.len = bytes;
	t.phys = (uint64_t)(uintptr_t)d->pg_phys;
	t.n_pages = cap_pages;
	if (ioctl(h->fd, NVMEV_L1_IOC_TRANSLATE, &t)) {
		perror("ioctl NVMEV_L1_IOC_TRANSLATE");
		return -1;
	}
	d->va = dst_gpu_va;
	d->npages = t.n_pages;
	d->page_shift = t.page_shift;
	d->cap_rows = capacity_rows;
	return 0;
}

/* GPU bus address of row i within slot's buffer. The translate table is indexed
 * from the buffer's page-aligned base, so add the buffer's offset within its
 * first GPU page (va need not be 64 KB-aligned). */
static uint64_t dst_phys_of(const struct isefs_dst *d, unsigned row, size_t i)
{
	uint64_t pg_mask = ((uint64_t)1 << d->page_shift) - 1;
	uint64_t off = (d->va & pg_mask) + (uint64_t)i * row;
	unsigned pg = (unsigned)(off >> d->page_shift);

	if (pg >= d->npages)
		return 0;
	return d->pg_phys[pg] + (off & pg_mask);
}

int isefs_fetch(isefs_t *h, unsigned slot, const uint32_t *node_ids, size_t n,
		uint64_t req_base, size_t dst_off)
{
	const struct isefs_dst *d;
	size_t i;

	if (!h || slot >= ISEFS_MAX_DST)
		return -1;
	d = &h->dst[slot];
	if (!d->va || dst_off + n > d->cap_rows)
		return -1;
	for (i = 0; i < n; i++) {
		unsigned k = l1_shard_of(node_ids[i], h->row, h->page,
					 h->nr_shards);
		struct l1_req r = {
			.req_id = req_base + i,
			.node_id = node_ids[i],
			.dst_phys = dst_phys_of(d, h->row, dst_off + i),
			.flags = 0,
		};
		l1_push_req(h->shard[k], &r);	/* blocks only if a ring is full */
	}
	return 0;
}

int isefs_poll(isefs_t *h, uint64_t *done, size_t max)
{
	unsigned k;
	int got = 0;

	for (k = 0; k < h->nr_shards && (size_t)got < max; k++) {
		struct l1_cqe c;

		while ((size_t)got < max && l1_pop_cqe(h->shard[k], &c) == 0)
			done[got++] = c.req_id;
	}
	return got;
}

int isefs_fetch_wait(isefs_t *h, const uint32_t *node_ids, size_t n)
{
	/* Bounded push/drain: never leave more than LIMIT requests outstanding, so
	 * the per-shard completion ring can't overflow (which would wedge the
	 * consumer's push_cqe and deadlock the whole loop). LIMIT must be < a
	 * shard's cq depth even if every request skews to one shard. */
	const size_t LIMIT = 16384, CHUNK = 4096;
	uint64_t *scratch;
	size_t pushed = 0, done = 0;

	if (!h)
		return -1;
	scratch = malloc(CHUNK * sizeof(uint64_t));
	if (!scratch)
		return -1;
	while (done < n) {
		while (pushed < n && (pushed - done) < LIMIT) {
			size_t c = n - pushed;

			if (c > CHUNK)
				c = CHUNK;
			if (c > LIMIT - (pushed - done))
				c = LIMIT - (pushed - done);
			if (isefs_fetch(h, 0, node_ids + pushed, c, pushed, pushed)) {
				free(scratch);
				return -1;
			}
			pushed += c;
		}
		done += isefs_poll(h, scratch, CHUNK);
	}
	free(scratch);
	return 0;
}

unsigned isefs_nr_shards(const isefs_t *h) { return h ? h->nr_shards : 0; }
