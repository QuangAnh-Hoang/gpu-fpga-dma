// SPDX-License-Identifier: GPL-2.0
/*
 * libisefs_gpupull — GPU-pull delivery backend for libisefs (b4). Where the
 * WC-store libisefs.so has the CPU copy each feature row to VRAM and poll CQEs,
 * this launches a persistent GPU kernel that drains the kernel's per-shard
 * descriptor rings and gathers each row from mmap'd staging straight into the
 * caller's VRAM output buffer — the CPU never touches the bytes. Completion is
 * the sum of per-shard ring tails (the GPU advances tail as it delivers); the
 * L1 CQ is unused in this mode.
 *
 * ctypes C API (mirrors what isefs.py's GPU-pull path needs):
 *   igp_open(l1_dev, row, page, wps)   open + mmap + register + build ring views
 *   igp_set_out(h, out_gpu_va, rows)   set the output buffer + launch the kernel
 *   igp_fetch(h, node_ids, n)          push n L1 requests (dst = row i) + wait
 *   igp_nr_shards / igp_close
 *
 * Requires the module loaded with coalesce_deliver=1 (GPU-pull plane). The
 * caller's coalesce_row/page must match. Build: make -C sw isefs_gpupull.
 */
#include <cuda_runtime.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#include "nvmev_l1.h"

// per-shard device-side ring view (matches gpupull_probe.cu)
struct GRingDev {
	const volatile uint32_t *head;
	volatile uint32_t *tail;
	const struct l1_gdesc *gdesc;	// read via __ldcv (fresh; DMA-written)
	uint32_t entries;
};

// Per-fetch kernel (NOT persistent — a forever-running kernel hangs any torch
// cudaDeviceSynchronize / allocator sync). One block per shard, W warps. Resumes
// from the ring tail, drains this fetch's descriptors as the module produces them
// (overlap), counting each delivered row into a shared global g_done; ALL blocks
// exit once g_done reaches `target` (= this fetch's row count), so the launch
// completes and torch can sync. Reads descriptors + staging fresh (__ldcv, they
// are recycled DMA-updated host memory); tail published via __threadfence_system.
__global__ void igp_puller(const GRingDev *rings, const uint8_t *staging,
			   uint8_t *out, unsigned *g_done, unsigned target, unsigned nshards)
{
	unsigned shard = blockIdx.x;
	if (shard >= nshards) return;
	GRingDev g = rings[shard];
	unsigned W = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
	__shared__ uint32_t s_tail, s_head, s_avail, s_done;
	if (threadIdx.x == 0) { s_tail = *g.tail; s_head = 0; }	// resume from ring tail
	__syncthreads();
	uint32_t pub = s_tail;

	for (;;) {
		if (threadIdx.x == 0) {
			s_done = atomicAdd(g_done, 0u);		// rows delivered this fetch
			if (s_tail >= s_head) s_head = *g.head;	// re-read head (idle only)
			uint32_t av = s_head - s_tail;
			s_avail = av < W ? av : W;
		}
		__syncthreads();
		if (s_done >= target) break;			// whole fetch delivered: exit
		uint32_t avail = s_avail;
		if (avail == 0) continue;			// wait for the module to produce
		if (warp < avail) {
			unsigned long long d_off = 0, d_dst = 0;
			unsigned d_len = 0, d_status = 1;
			if (lane == 0) {		// read 32 B descriptor as 2 coalesced uint4
				const uint4 *dq = reinterpret_cast<const uint4 *>(
					&g.gdesc[(s_tail + warp) & (g.entries - 1)]);
				uint4 a = __ldcv(&dq[0]), b = __ldcv(&dq[1]);
				d_off = ((unsigned long long)a.y << 32) | a.x;
				d_dst = ((unsigned long long)a.w << 32) | a.z;
				d_len = b.y & 0xffff; d_status = b.y >> 16;
			}
			d_off    = __shfl_sync(0xffffffffu, d_off, 0);
			d_dst    = __shfl_sync(0xffffffffu, d_dst, 0);
			d_len    = __shfl_sync(0xffffffffu, d_len, 0);
			d_status = __shfl_sync(0xffffffffu, d_status, 0);
			if (d_status == 0) {
				const uint4 *sp = reinterpret_cast<const uint4 *>(staging + d_off);
				uint4 *o = reinterpret_cast<uint4 *>(out + d_dst);
				for (unsigned k = lane; k < (d_len >> 4); k += 32)
					o[k] = __ldcv(&sp[k]);
			}
		}
		__syncthreads();
		if (threadIdx.x == 0) {
			s_tail += avail;
			atomicAdd(g_done, avail);		// count this fetch's deliveries
			if (s_tail - pub >= 64) {
				__threadfence_system(); *g.tail = s_tail; pub = s_tail;
			}
		}
		__syncthreads();
	}
	if (threadIdx.x == 0 && pub != s_tail) { __threadfence_system(); *g.tail = s_tail; }
}

struct igp {
	int fd;
	unsigned row, page, wps, ns;
	void *region;			// L1 rings (CPU push)
	void *region_d; size_t region_bytes;	// same rings mapped for GPU push
	cudaStream_t st_push;		// push kernel runs concurrently with the puller
	void *stg_h; uint8_t *stg_d;	// staging
	void *desc_h, *desc_d;		// descriptor rings
	struct l1_descinfo di;
	GRingDev *rings_d;
	uint8_t *out_d;			// caller's VRAM output buffer (base of the mega-buffer)
	unsigned *g_done;		// device: rows delivered this fetch (reset each fetch)
	cudaStream_t st;
	int out_set;
	/* cross-batch pipeline (b5): D output slots in one mega-buffer so D batches'
	 * requests co-reside in the module's coalesce windows (cross-batch dedup). */
	unsigned depth;			// D output slots (1 = legacy single-buffer)
	unsigned slot_rows;		// rows per slot (== max_batch)
	unsigned long long slot_bytes;	// slot_rows * row (dst stride between slots)
	unsigned *g_done_slot;		// device: rows delivered per slot (persists across drains)
};

extern "C" {

igp *igp_open(const char *l1_dev, unsigned row, unsigned page, unsigned wps)
{
	if (wps < 1) wps = 1; if (wps > 32) wps = 32;
	igp *h = (igp *)calloc(1, sizeof(*h));
	if (!h) return NULL;
	h->row = row; h->page = page; h->wps = wps;
	h->fd = open(l1_dev, O_RDWR);
	if (h->fd < 0) { free(h); return NULL; }

	struct l1_ring hdr;
	void *m0 = mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_SHARED, h->fd, 0);
	if (m0 == MAP_FAILED) goto fail;
	memcpy(&hdr, m0, sizeof hdr); munmap(m0, 4096);
	h->ns = hdr.nr_shards;
	h->region_bytes = (size_t)h->ns * hdr.shard_bytes;
	h->region = mmap(NULL, h->region_bytes,
			 PROT_READ|PROT_WRITE, MAP_SHARED, h->fd, 0);
	/* Register the L1 rings for GPU access so a push kernel (l1_push_req_dev)
	 * can produce requests with no CPU in the loop (GPU-push). */
	if (h->region == MAP_FAILED ||
	    cudaHostRegister(h->region, h->region_bytes, cudaHostRegisterMapped)) goto fail;
	if (cudaHostGetDevicePointer(&h->region_d, h->region, 0) || !h->region_d)
		goto fail;			/* pusher would fault on a bad device alias */
	cudaStreamCreate(&h->st_push);
	{
		struct l1_staging sg;
		if (ioctl(h->fd, NVMEV_L1_IOC_STGINFO, &sg) ||
		    ioctl(h->fd, NVMEV_L1_IOC_DESCINFO, &h->di)) goto fail;
		h->stg_h  = mmap(NULL, sg.bytes, PROT_READ|PROT_WRITE, MAP_SHARED, h->fd, sg.mmap_off);
		h->desc_h = mmap(NULL, h->di.bytes, PROT_READ|PROT_WRITE, MAP_SHARED, h->fd, h->di.mmap_off);
		if (h->stg_h == MAP_FAILED || h->desc_h == MAP_FAILED) goto fail;
		if (cudaHostRegister(h->stg_h, sg.bytes, cudaHostRegisterMapped) ||
		    cudaHostRegister(h->desc_h, h->di.bytes, cudaHostRegisterMapped)) goto fail;
		cudaHostGetDevicePointer(&h->stg_d, h->stg_h, 0);
		cudaHostGetDevicePointer(&h->desc_d, h->desc_h, 0);

		GRingDev *rings_h = (GRingDev *)calloc(h->ns, sizeof(GRingDev));
		for (unsigned k = 0; k < h->ns; k++) {
			struct l1_gring *gh = l1_gshard(h->desc_h, k, h->di.shard_bytes);
			uint8_t *base = (uint8_t *)h->desc_d + (size_t)k * h->di.shard_bytes;
			rings_h[k].head  = (uint32_t *)(base + offsetof(struct l1_gring, head));
			rings_h[k].tail  = (uint32_t *)(base + offsetof(struct l1_gring, tail));
			rings_h[k].gdesc = (struct l1_gdesc *)(base + gh->desc_off);
			rings_h[k].entries = gh->entries;
		}
		cudaMalloc(&h->rings_d, h->ns * sizeof(GRingDev));
		cudaMemcpy(h->rings_d, rings_h, h->ns * sizeof(GRingDev), cudaMemcpyHostToDevice);
		free(rings_h);
	}
	cudaMalloc(&h->g_done, sizeof(unsigned));
	cudaStreamCreate(&h->st);
	return h;
fail:
	if (h->fd >= 0) close(h->fd);
	free(h);
	return NULL;
}

/* Wait for a stream to finish, but give up after `deadline_s` wall-clock seconds
 * (returns -1) instead of hanging forever. A genuine wedge (module consumer stuck,
 * a request that never produces a descriptor) then surfaces as an error the Python
 * layer can report + reload on, rather than a silent hang. */
static int stream_wait(cudaStream_t st, double deadline_s)
{
	struct timespec t0;
	clock_gettime(CLOCK_MONOTONIC, &t0);
	for (;;) {
		cudaError_t e = cudaStreamQuery(st);
		if (e == cudaSuccess) return 0;
		if (e != cudaErrorNotReady) return -1;	// real CUDA error (fault)
		struct timespec now;
		clock_gettime(CLOCK_MONOTONIC, &now);
		double dt = (now.tv_sec - t0.tv_sec) + (now.tv_nsec - t0.tv_nsec) * 1e-9;
		if (dt > deadline_s) return -1;		// wedged: bail instead of hang
		struct timespec s = { 0, 50 * 1000 };	// 50us backoff (don't busy-spin)
		nanosleep(&s, NULL);
	}
}

// Set the caller's VRAM output buffer (the per-fetch kernel writes rows here).
int igp_set_out(igp *h, uint64_t out_gpu_va, size_t capacity_rows)
{
	(void)capacity_rows;
	h->out_d = (uint8_t *)(uintptr_t)out_gpu_va;
	h->out_set = 1;
	return 0;
}

// Push n requests (feature[node_ids[i]] -> output row i) and run a per-fetch GPU
// kernel that gathers them into the output buffer, returning once all n have
// landed. The kernel resumes from the ring tail, overlaps with the module
// producing descriptors, and self-terminates at g_done==n (so torch can sync).
// dst = i*row is a byte offset into the output buffer, reused every fetch.
int igp_fetch(igp *h, const uint32_t *node_ids, size_t n)
{
	if (!h->out_set) return -1;
	if (n == 0) return 0;

	cudaMemsetAsync(h->g_done, 0, sizeof(unsigned), h->st);	// reset per-fetch counter
	igp_puller<<<h->ns, h->wps * 32, 0, h->st>>>(
		h->rings_d, h->stg_d, h->out_d, h->g_done, (unsigned)n, h->ns);
	if (cudaGetLastError()) return -1;

	for (size_t i = 0; i < n; i++) {			// produce (module reaps -> descriptors)
		struct l1_req r = { .req_id = (uint64_t)i, .node_id = node_ids[i],
				    .dst_phys = (uint64_t)i * h->row };
		l1_push_req(l1_shard(h->region,
			    l1_shard_of(node_ids[i], h->row, h->page, h->ns)), &r);
	}
	if (stream_wait(h->st, 30.0))				// kernel exits at g_done==n
		return -2;
	return 0;
}

/* GPU producer: each thread pushes one L1 request (feature[ids[i]] -> row i) via
 * the MPSC ring's device push. Mirrors l1_gnn_probe's push_kernel. dst_base is the
 * byte offset of this batch's output slot in the mega-buffer (0 for single-buffer). */
__global__ void igp_pusher(void *region, const uint32_t *ids, unsigned n,
			   unsigned row, unsigned page, unsigned ns,
			   unsigned long long dst_base)
{
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n) return;
	uint32_t node = ids[i];
	struct l1_req r;
	r.req_id = i; r.node_id = node; r._pad0 = 0;
	r.dst_phys = dst_base + (uint64_t)i * row;	/* byte offset into out buffer */
	r.flags = 0; r._pad1 = 0; r._pad2 = 0;
	l1_push_req_dev(l1_shard(region, l1_shard_of(node, row, page, ns)), &r);
}

/* Pipelined puller: like igp_puller, but descriptors from D co-resident batches are
 * intermixed in each shard's ring (the module coalesces across batches, reordering).
 * Each delivered row is counted into g_done_slot[slot] where slot = dst / slot_bytes,
 * so a per-batch drain terminates on ITS slot's count while still delivering (and
 * crediting) rows belonging to the other in-flight batches. Counters persist across
 * launches (reset by the host when a slot is reused), so a later drain of batch k+1
 * finishes off whatever this drain already delivered for it. */
__global__ void igp_puller_pipe(const GRingDev *rings, const uint8_t *staging,
				uint8_t *out, unsigned *g_done_slot,
				unsigned long long slot_bytes, unsigned target_slot,
				unsigned target_n, unsigned nshards)
{
	unsigned shard = blockIdx.x;
	if (shard >= nshards) return;
	GRingDev g = rings[shard];
	unsigned W = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
	__shared__ uint32_t s_tail, s_head, s_avail, s_done;
	if (threadIdx.x == 0) { s_tail = *g.tail; s_head = 0; }
	__syncthreads();
	uint32_t pub = s_tail;

	for (;;) {
		if (threadIdx.x == 0) {
			s_done = atomicAdd(&g_done_slot[target_slot], 0u);
			if (s_tail >= s_head) s_head = *g.head;
			uint32_t av = s_head - s_tail;
			s_avail = av < W ? av : W;
		}
		__syncthreads();
		if (s_done >= target_n) break;			// this batch fully landed
		uint32_t avail = s_avail;
		if (avail == 0) continue;
		if (warp < avail) {
			unsigned long long d_off = 0, d_dst = 0;
			unsigned d_len = 0, d_status = 1;
			if (lane == 0) {
				const uint4 *dq = reinterpret_cast<const uint4 *>(
					&g.gdesc[(s_tail + warp) & (g.entries - 1)]);
				uint4 a = __ldcv(&dq[0]), b = __ldcv(&dq[1]);
				d_off = ((unsigned long long)a.y << 32) | a.x;
				d_dst = ((unsigned long long)a.w << 32) | a.z;
				d_len = b.y & 0xffff; d_status = b.y >> 16;
			}
			d_off    = __shfl_sync(0xffffffffu, d_off, 0);
			d_dst    = __shfl_sync(0xffffffffu, d_dst, 0);
			d_len    = __shfl_sync(0xffffffffu, d_len, 0);
			d_status = __shfl_sync(0xffffffffu, d_status, 0);
			if (d_status == 0) {
				const uint4 *sp = reinterpret_cast<const uint4 *>(staging + d_off);
				uint4 *o = reinterpret_cast<uint4 *>(out + d_dst);
				for (unsigned k = lane; k < (d_len >> 4); k += 32)
					o[k] = __ldcv(&sp[k]);
			}
			if (lane == 0)		// credit this row to its batch's slot
				atomicAdd(&g_done_slot[(unsigned)(d_dst / slot_bytes)], 1u);
		}
		__syncthreads();
		if (threadIdx.x == 0) {
			s_tail += avail;
			if (s_tail - pub >= 64) {
				__threadfence_system(); *g.tail = s_tail; pub = s_tail;
			}
		}
		__syncthreads();
	}
	if (threadIdx.x == 0 && pub != s_tail) { __threadfence_system(); *g.tail = s_tail; }
}

/* GPU-PUSH variant of igp_fetch: node_ids_dev is a DEVICE pointer (GPU-resident,
 * e.g. straight from GPU neighbor sampling) so the CPU is out of the producer.
 *
 * The push and the pull are SEQUENCED, not raced on two streams: the puller busy-
 * spins on host-mapped ring state over PCIe, and the module's window only flushes
 * after coalesce_flush_us, so a concurrent puller would hammer the PCIe link (reads
 * + a global atomic) for the whole pre-flush gap and starve the pusher's system
 * atomics. Push first (module reaps + issues reads meanwhile), then drain — the
 * module's read latency is what we overlap with, and it dwarfs the gather. */
int igp_fetch_dev(igp *h, const uint32_t *node_ids_dev, size_t n)
{
	unsigned T = 256, B;

	if (!h->out_set) return -1;
	if (n == 0) return 0;
	B = (unsigned)((n + T - 1) / T);

	cudaMemsetAsync(h->g_done, 0, sizeof(unsigned), h->st_push);	// reset counter
	igp_pusher<<<B, T, 0, h->st_push>>>(			// produce all n requests
		h->region_d, node_ids_dev, (unsigned)n, h->row, h->page, h->ns, 0ull);
	if (cudaGetLastError()) return -1;
	if (stream_wait(h->st_push, 30.0)) return -2;		// all requests pushed
	igp_puller<<<h->ns, h->wps * 32, 0, h->st>>>(		// then drain descriptors
		h->rings_d, h->stg_d, h->out_d, h->g_done, (unsigned)n, h->ns);
	if (cudaGetLastError()) return -1;
	if (stream_wait(h->st, 30.0)) return -2;		// all rows delivered
	return 0;
}

/* ---- cross-batch pipeline (b5) ----
 * The single-buffer fetch drains each batch before the next is pushed, so a batch's
 * requests never co-reside with another's in the module's coalesce window — all the
 * cross-batch page reuse is lost. The pipeline instead PUSHES D batches into D output
 * slots of one mega-buffer before draining any, so up to D batches' requests sit in
 * the windows together and coalesce across batches (fewer backend reads). Push and
 * pull stay sequenced (no two-stream race); the co-residency is on the push side, and
 * the module's reads overlap the pushes. Raise coalesce_flush_min/flush_us so a window
 * waits for ~D batches instead of sealing on the gap between two pushes. */

/* Configure D output slots of `slot_rows` rows each. out (from igp_set_out) must be a
 * mega-buffer of at least depth*slot_rows rows. Idempotent-ish: reallocs the counters. */
int igp_pipe_config(igp *h, unsigned depth, unsigned slot_rows)
{
	if (depth < 1) depth = 1;
	h->depth = depth;
	h->slot_rows = slot_rows;
	h->slot_bytes = (unsigned long long)slot_rows * h->row;
	if (h->g_done_slot) { cudaFree(h->g_done_slot); h->g_done_slot = NULL; }
	if (cudaMalloc(&h->g_done_slot, (size_t)depth * sizeof(unsigned))) return -1;
	if (cudaMemset(h->g_done_slot, 0, (size_t)depth * sizeof(unsigned))) return -1;
	return 0;
}

/* Push batch `slot`'s n requests (GPU-resident ids) into output slot `slot`, resetting
 * that slot's completion counter first. Does NOT drain — call igp_drain_slot later.
 * `slot` must already be fully drained+consumed (host enforces the D-deep pipeline). */
int igp_push_slot_dev(igp *h, const uint32_t *node_ids_dev, size_t n, unsigned slot)
{
	unsigned T = 256, B;
	if (!h->out_set || !h->g_done_slot || slot >= h->depth) return -1;
	if (n == 0) return 0;
	B = (unsigned)((n + T - 1) / T);
	cudaMemsetAsync(&h->g_done_slot[slot], 0, sizeof(unsigned), h->st_push);
	igp_pusher<<<B, T, 0, h->st_push>>>(
		h->region_d, node_ids_dev, (unsigned)n, h->row, h->page, h->ns,
		(unsigned long long)slot * h->slot_bytes);
	if (cudaGetLastError()) return -1;
	if (stream_wait(h->st_push, 30.0)) return -2;
	return 0;
}

/* CPU-push variant of igp_push_slot_dev (node_ids is a HOST pointer). */
int igp_push_slot(igp *h, const uint32_t *node_ids, size_t n, unsigned slot)
{
	if (!h->out_set || !h->g_done_slot || slot >= h->depth) return -1;
	if (n == 0) return 0;
	cudaMemset(&h->g_done_slot[slot], 0, sizeof(unsigned));
	unsigned long long base = (unsigned long long)slot * h->slot_bytes;
	for (size_t i = 0; i < n; i++) {
		struct l1_req r = { .req_id = (uint64_t)i, .node_id = node_ids[i],
				    .dst_phys = base + (uint64_t)i * h->row };
		l1_push_req(l1_shard(h->region,
			    l1_shard_of(node_ids[i], h->row, h->page, h->ns)), &r);
	}
	return 0;
}

/* Drain until batch `slot`'s n rows have landed. Delivers (and credits) rows for the
 * other in-flight batches it encounters too; their counters persist so their own
 * drain finishes them off. Blocks (with watchdog) until g_done_slot[slot] >= n. */
int igp_drain_slot(igp *h, unsigned slot, size_t n)
{
	if (!h->out_set || !h->g_done_slot || slot >= h->depth) return -1;
	if (n == 0) return 0;
	igp_puller_pipe<<<h->ns, h->wps * 32, 0, h->st>>>(
		h->rings_d, h->stg_d, h->out_d, h->g_done_slot, h->slot_bytes,
		slot, (unsigned)n, h->ns);
	if (cudaGetLastError()) return -1;
	if (stream_wait(h->st, 30.0)) return -2;
	return 0;
}

unsigned igp_nr_shards(igp *h) { return h ? h->ns : 0; }

void igp_close(igp *h)
{
	if (!h) return;
	cudaStreamSynchronize(h->st);
	cudaStreamSynchronize(h->st_push);
	if (h->g_done_slot) cudaFree(h->g_done_slot);
	if (h->region && h->region != MAP_FAILED) cudaHostUnregister(h->region);
	if (h->fd >= 0) close(h->fd);
	free(h);
}

}  // extern "C"
