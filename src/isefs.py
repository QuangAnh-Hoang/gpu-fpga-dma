# SPDX-License-Identifier: GPL-2.0
"""
isefs.py — Python binding for libisefs (the NVMeVirt L1-ring feature-store
transport) + a DGL dataloader adapter, so a DGL GNN training loop fetches node
features through the ISE middle layer instead of a resident tensor
(assessment DD1 / §3.3.7). DGL is used so ISE plugs in at the SAME point as GIDS
(GIDS_DGLDataLoader) for an apples-to-apples head-to-head — same graph, sampler,
and model, only the feature backend differs.

Two ways to use it:

  1. ISE_DGLDataLoader — drop-in parallel to GIDS_DGLDataLoader: wraps a DGL
     DataLoader and appends an ISEFS-fetched feature tensor to each batch, so
     the training loop is identical to GIDS:
         fs = ISEFS(feat_dim=dim, max_batch=BIG)
         loader = ISE_DGLDataLoader(g, train_nid, sampler, batch_size, dim, fs)
         for input_nodes, output_nodes, blocks, feat in loader:
             out = model(blocks, feat); ...

  2. Direct — call fs.fetch(input_nodes) yourself anywhere a feature gather
     happens:  x = fs.fetch(input_nodes)   # [n, feat_dim] CUDA tensor

Requires: nvmev.ko loaded with CONFIG_NVMEVIRT_GPU_DIRECT, obj/libisefs.so built
(`make -C sw isefs`), write access to /proc/nvmev/gpu_mem (run as root), and the
module's coalesce_row == feat_dim*dtype.itemsize, coalesce_page matching.
Feature n's bytes must be seeded on the backend at byte offset n*row.
"""
import ctypes
import os

import torch

_LIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "obj", "libisefs.so")
_LIB_GP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "obj",
                       "libisefs_gpupull.so")


def _load_gpupull():
    lib = ctypes.CDLL(_LIB_GP)
    lib.igp_open.restype = ctypes.c_void_p
    lib.igp_open.argtypes = [ctypes.c_char_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint]
    lib.igp_set_out.restype = ctypes.c_int
    lib.igp_set_out.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
    lib.igp_fetch.restype = ctypes.c_int
    lib.igp_fetch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.igp_fetch_dev.restype = ctypes.c_int   # GPU-push: node_ids is a DEVICE ptr
    lib.igp_fetch_dev.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    # cross-batch pipeline (b5): push D batches into D slots, drain per-slot
    lib.igp_pipe_config.restype = ctypes.c_int
    lib.igp_pipe_config.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint]
    lib.igp_push_slot_dev.restype = ctypes.c_int    # GPU-push into a slot (DEVICE ptr)
    lib.igp_push_slot_dev.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                      ctypes.c_uint]
    lib.igp_push_slot.restype = ctypes.c_int        # CPU-push into a slot (HOST ptr)
    lib.igp_push_slot.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                  ctypes.c_uint]
    lib.igp_drain_slot.restype = ctypes.c_int
    lib.igp_drain_slot.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t]
    lib.igp_nr_shards.restype = ctypes.c_uint
    lib.igp_nr_shards.argtypes = [ctypes.c_void_p]
    lib.igp_close.argtypes = [ctypes.c_void_p]
    return lib


def _load():
    lib = ctypes.CDLL(_LIB)
    lib.isefs_open.restype = ctypes.c_void_p
    lib.isefs_open.argtypes = [ctypes.c_char_p, ctypes.c_uint, ctypes.c_uint]
    lib.isefs_close.argtypes = [ctypes.c_void_p]
    lib.isefs_set_dst.restype = ctypes.c_int
    lib.isefs_set_dst.argtypes = [ctypes.c_void_p, ctypes.c_uint,
                                  ctypes.c_uint64, ctypes.c_size_t]
    lib.isefs_fetch.restype = ctypes.c_int
    lib.isefs_fetch.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p,
                                ctypes.c_size_t, ctypes.c_uint64, ctypes.c_size_t]
    lib.isefs_poll.restype = ctypes.c_int
    lib.isefs_poll.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.isefs_fetch_wait.restype = ctypes.c_int
    lib.isefs_fetch_wait.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.isefs_nr_shards.restype = ctypes.c_uint
    lib.isefs_nr_shards.argtypes = [ctypes.c_void_p]
    return lib


class ISEFS:
    """Fetch feature rows into VRAM through /dev/nvmev_l1.

    Pins `ndst` reusable destination buffers of `max_batch` rows (ndst>1 enables
    prefetch pipelining — fetch batch k+1 into one buffer while k is consumed
    from another). fetch() is the simple blocking single-buffer path; the
    async fetch_gen/wait_gen/view_gen trio drives the pipeline.
    """

    def __init__(self, feat_dim, dtype=torch.float32, max_batch=4096, ndst=1,
                 page_bytes=4096, dev="/dev/nvmev_l1", limit=None):
        import numpy as np
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required (features are delivered to VRAM)")
        self.lib = _load()
        self.feat_dim = feat_dim
        self.dtype = dtype
        self.row = feat_dim * torch.empty(0, dtype=dtype).element_size()
        self.max_batch = max_batch
        self.ndst = ndst
        self.h = self.lib.isefs_open(dev.encode(), self.row, page_bytes)
        if not self.h:
            raise RuntimeError(f"isefs_open({dev}) failed — module loaded? "
                               "GPU_DIRECT built? row/page match?")
        self._bufs, self._views = [], []
        for slot in range(ndst):
            buf = torch.empty(max_batch * self.row, dtype=torch.uint8, device="cuda")
            rc = self.lib.isefs_set_dst(self.h, slot,
                                        ctypes.c_uint64(buf.data_ptr()), max_batch)
            if rc != 0:
                raise RuntimeError("isefs_set_dst failed (pin/translate) — root?")
            self._bufs.append(buf)
            self._views.append(buf.view(dtype).view(max_batch, feat_dim))
        # completion bookkeeping for the async path
        self._done = np.empty(max_batch, dtype=np.uint64)
        self._done_ptr = self._done.ctypes.data
        self._count = {}          # gen -> completions seen so far
        self._pushed = 0          # running totals for flow control
        self._completed = 0
        # Flow-control limit: keep outstanding (pushed-not-completed) below a
        # shard's completion-ring depth, else a large batch overflows the CQ,
        # wedges the consumer's push_cqe, and deadlocks. A LARGER limit lets the
        # coalesce window fill deeper (more dedup) — raise it toward the ring size
        # to exploit a bigger coalesce_window, but it MUST stay < ring so an
        # all-skew-to-one-shard batch can't overflow that shard's CQ.
        # Precedence: explicit `limit=` arg > $ISEFS_LIMIT env > default ring//4.
        try:
            ring = int(open("/sys/module/nvmev/parameters/coalesce_ring").read())
        except OSError:
            ring = 65536
        env = os.environ.get("ISEFS_LIMIT")
        want = limit if limit is not None else (int(env) if env else ring // 4)
        self.limit = max(1024, min(int(want), ring - 1))     # < ring: CQ safety
        if self.limit > ring // 2:
            print(f"# ISEFS: push limit {self.limit} > ring/2 ({ring // 2}) — "
                  f"CQ-overflow risk if a batch all skews to one shard", flush=True)
        self._chunk = min(4096, self.limit)

    def nr_shards(self):
        return int(self.lib.isefs_nr_shards(self.h))

    # ---- simple blocking single-buffer path (slot 0, bounded push/drain) ----
    def fetch(self, node_ids, stall_s=5.0):
        """Blocking. Returns a [n, feat_dim] CUDA view of buffer 0 (valid until
        the next fetch into slot 0). Bounded push/drain (outstanding < a shard's
        CQ) so a large/skewed batch can't overflow the CQ and deadlock. A stall
        watchdog prints if no completions arrive for `stall_s` (=> the kernel
        consumer is wedged, usually from a prior crashed/Ctrl-C'd run — reload
        the module) instead of hanging silently."""
        import time
        n = int(node_ids.numel())
        if n > self.max_batch:
            raise ValueError(
                f"batch {n} > max_batch {self.max_batch} — this is the sampler's "
                f"receptive field (input_nodes), not --batch. Raise --max-batch "
                f"(ISEFS max_batch, VRAM = max_batch*row*ndst) or reduce --batch/--fanout.")
        ids = node_ids.to(torch.int32).cpu().contiguous()
        base, elt = ids.data_ptr(), 4
        pushed = done = 0
        last = time.time()
        while done < n:
            while pushed < n and (pushed - done) < self.limit:
                c = min(self._chunk, n - pushed, self.limit - (pushed - done))
                if self.lib.isefs_fetch(self.h, 0,
                                        ctypes.c_void_p(base + pushed * elt),
                                        c, ctypes.c_uint64(pushed), pushed):
                    raise RuntimeError("isefs_fetch failed")
                pushed += c
            got = self.lib.isefs_poll(self.h, ctypes.c_void_p(self._done_ptr),
                                      self.max_batch)
            done += got
            now = time.time()
            if got:
                last = now
            elif now - last > stall_s:
                print(f"  [ise] STALL: pushed={pushed} done={done}/{n}, no "
                      f"completions for {now-last:.0f}s — kernel consumer wedged? "
                      f"reload the module (a prior deadlocked run leaves it stuck).",
                      flush=True)
                last = now
        return self._views[0][:n]

    # ---- async pipeline path (bounded so big batches can't overflow the CQ) ----
    def fetch_gen(self, gen, node_ids):
        """Issue the fetch for batch `gen` into slot gen%ndst. Pushes in bounded
        chunks, draining completions when outstanding hits the limit, so it never
        deadlocks on a large batch. Returns once all requests are pushed (some may
        still be in flight); wait_gen() finishes them."""
        n = int(node_ids.numel())
        if n > self.max_batch:
            raise ValueError(
                f"batch {n} > max_batch {self.max_batch} — this is the sampler's "
                f"receptive field (input_nodes), not --batch. Raise --max-batch "
                f"(ISEFS max_batch, VRAM = max_batch*row*ndst) or reduce --batch/--fanout.")
        ids = node_ids.to(torch.int32).cpu().contiguous()  # consumed by each push
        self._ids_hold = ids            # keep alive until fully pushed
        base = ids.data_ptr()
        slot = gen % self.ndst
        elt = 4  # int32 element size
        off = 0
        while off < n:
            while self._pushed - self._completed >= self.limit:
                self._drain()
            c = min(self._chunk, n - off,
                    self.limit - (self._pushed - self._completed))
            if self.lib.isefs_fetch(self.h, slot,
                                    ctypes.c_void_p(base + off * elt), c,
                                    ctypes.c_uint64(gen * self.max_batch + off),
                                    off):
                raise RuntimeError("isefs_fetch failed")
            self._pushed += c
            off += c

    def _drain(self):
        got = self.lib.isefs_poll(self.h, ctypes.c_void_p(self._done_ptr),
                                  self.max_batch)
        self._completed += got
        for k in range(got):
            g = int(self._done[k]) // self.max_batch
            self._count[g] = self._count.get(g, 0) + 1

    def wait_gen(self, gen, n):
        """Block until all n rows of batch `gen` have landed."""
        while self._count.get(gen, 0) < n:
            self._drain()
        self._count.pop(gen, None)

    def view_gen(self, gen, n):
        return self._views[gen % self.ndst][:n]

    def close(self):
        if getattr(self, "h", None):
            self.lib.isefs_close(self.h)
            self.h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class ISEFS_GPUPull:
    """GPU-pull feature store: same fetch/pipeline interface as ISEFS, but the
    delivery is done by a GPU kernel (libisefs_gpupull.so) draining the module's
    descriptor rings straight into VRAM — the CPU never copies the bytes.
    Requires the module loaded with coalesce_deliver=1 (GPU-pull plane).

    Cross-batch pipeline (depth>1): a mega-buffer of `depth` slots lets the loader
    PUSH depth batches into the module's coalesce windows before draining any, so up
    to depth batches' requests co-reside and coalesce across batches (fewer backend
    reads). fetch_gen(gen) pushes into slot gen%depth (no drain); wait_gen(gen) drains
    that slot; view_gen(gen) returns its rows. ISE_DGLDataLoader (depth = fs.ndst)
    clones each batch before its slot is reused. depth=1 is the legacy single-buffer
    synchronous path. RAISE coalesce_flush_min/flush_us so a window waits for ~depth
    batches (else it seals on the gap between two pushes and loses the co-residency).
    `wps` = gather warps per shard (throughput knob)."""

    def __init__(self, feat_dim, dtype=torch.float32, max_batch=4096, wps=32,
                 page_bytes=4096, dev="/dev/nvmev_l1", gpu_push=False, depth=1):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required (features are delivered to VRAM)")
        self.lib = _load_gpupull()
        self.feat_dim = feat_dim
        self.dtype = dtype
        self.row = feat_dim * torch.empty(0, dtype=dtype).element_size()
        self.max_batch = max_batch
        self.gpu_push = gpu_push            # push requests from a GPU kernel (no CPU)
        self.depth = max(1, int(depth))
        self.ndst = self.depth             # ISE_DGLDataLoader pipelines this deep
        self.h = self.lib.igp_open(dev.encode(), self.row, page_bytes, wps)
        if not self.h:
            raise RuntimeError(f"igp_open({dev}) failed — module loaded with "
                               "coalesce_deliver=1? row/page match? built .so?")
        # one mega-buffer of `depth` slots; slot s holds batch gen where gen%depth==s
        self.buf = torch.empty(self.depth * max_batch * self.row,
                               dtype=torch.uint8, device="cuda")
        self.view = self.buf.view(dtype).view(self.depth * max_batch, feat_dim)
        if self.lib.igp_set_out(self.h, ctypes.c_uint64(self.buf.data_ptr()),
                                self.depth * max_batch):
            raise RuntimeError("igp_set_out failed")
        if self.lib.igp_pipe_config(self.h, self.depth, max_batch):
            raise RuntimeError("igp_pipe_config failed (counter alloc)")
        self._stash = {}
        self._ids_hold = {}                # keep pushed id tensors alive until drained
        self._pending = {}                 # slot -> n: pushed but not yet wait_gen'd
        # Deadlock guard: the pipeline pushes `depth` batches before draining, so up
        # to depth*per_shard requests+descriptors sit un-drained in the per-shard rings
        # (rq and g_desc are both `coalesce_ring` deep). If that exceeds the ring, the
        # module's descriptor producer blocks -> stops reaping rq -> the GPU pusher
        # spins on the full ring -> stall. So require depth*ceil(n/nshards) < ring.
        try:
            self._ring = int(open("/sys/module/nvmev/parameters/coalesce_ring").read())
        except OSError:
            self._ring = 262144
        self._ns = self.nr_shards()
        self._warned_depth = False

    def nr_shards(self):
        return int(self.lib.igp_nr_shards(self.h))

    def _slot_view(self, slot, n):
        base = slot * self.max_batch
        return self.view[base:base + n]

    def fetch(self, node_ids, stall_s=5.0):
        """Blocking single-batch fetch into slot 0. Returns a [n, feat_dim] CUDA view
        (valid until the next fetch/push touches slot 0)."""
        n = int(node_ids.numel())
        self._check_n(n)
        if self.gpu_push:
            ids = node_ids.to(torch.int32).contiguous()
            if not ids.is_cuda:
                ids = ids.cuda()
            self._ids_hold[0] = ids
            rc = self.lib.igp_push_slot_dev(self.h, ctypes.c_void_p(ids.data_ptr()), n, 0)
        else:
            ids = node_ids.to(torch.int32).cpu().contiguous()
            self._ids_hold[0] = ids
            rc = self.lib.igp_push_slot(self.h, ctypes.c_void_p(ids.data_ptr()), n, 0)
        if rc:
            self._raise(rc)
        rc = self.lib.igp_drain_slot(self.h, 0, n)
        if rc:
            self._raise(rc)
        return self._slot_view(0, n)

    def _check_n(self, n):
        if n > self.max_batch:
            raise ValueError(
                f"batch {n} > max_batch {self.max_batch} — this is the sampler's "
                f"receptive field (input_nodes), not --batch. Raise --max-batch "
                f"(ISEFS max_batch, VRAM = depth*max_batch*row) or reduce --batch/--fanout.")

    def _raise(self, rc):
        if rc == -2:
            raise RuntimeError("igp GPU-pull stalled — GPU/consumer wedged; reload the "
                               "module (coalesce_enable=1 coalesce_deliver=1).")
        raise RuntimeError("igp GPU-pull failed")

    def _pipe_guard(self, n):
        """Ensure depth*per_shard stays under a ring so the un-drained descriptors of
        the D in-flight batches can't overflow the descriptor ring and deadlock the
        module's producer. per_shard ~= n/nshards; keep 15% headroom for shard skew."""
        per_shard = -(-n // max(1, self._ns))          # ceil
        outstanding = self.depth * per_shard
        if outstanding > int(0.85 * self._ring):
            d_max = max(1, int(0.85 * self._ring) // max(1, per_shard))
            raise RuntimeError(
                f"pipeline would deadlock: depth {self.depth} x per-shard ~{per_shard} "
                f"(batch {n} / {self._ns} shards) = {outstanding} un-drained reqs > "
                f"coalesce_ring {self._ring}. The pipeline pushes depth batches before "
                f"draining, so their descriptors must fit the per-shard ring. Fix: use "
                f"--pipe-depth <= {d_max}, OR raise coalesce_ring (reload RING=), OR "
                f"reduce --batch/--fanout (smaller receptive field), OR add shards "
                f"(per_shard = input_nodes/nshards).")

    # cross-batch pipeline: push into slot gen%depth WITHOUT draining, so depth
    # batches accumulate in the module's windows before wait_gen() drains them.
    def fetch_gen(self, gen, node_ids):
        slot = gen % self.depth
        n = int(node_ids.numel())
        self._check_n(n)
        self._pipe_guard(n)
        if self.gpu_push:
            ids = node_ids.to(torch.int32).contiguous()
            if not ids.is_cuda:
                ids = ids.cuda()
            self._ids_hold[slot] = ids     # alive until this slot is drained/reused
            rc = self.lib.igp_push_slot_dev(self.h, ctypes.c_void_p(ids.data_ptr()),
                                            n, slot)
        else:
            ids = node_ids.to(torch.int32).cpu().contiguous()
            self._ids_hold[slot] = ids
            rc = self.lib.igp_push_slot(self.h, ctypes.c_void_p(ids.data_ptr()), n, slot)
        if rc:
            self._raise(rc)
        self._pending[slot] = n            # in-flight until wait_gen drains it

    def wait_gen(self, gen, n):
        rc = self.lib.igp_drain_slot(self.h, gen % self.depth, n)
        self._pending.pop(gen % self.depth, None)
        if rc:
            self._raise(rc)

    def view_gen(self, gen, n):
        return self._slot_view(gen % self.depth, n)

    def close(self):
        if getattr(self, "h", None):
            # Drain any batch the pipeline pushed but the caller never wait_gen'd
            # (e.g. the loop stopped mid-stream while `depth` batches were primed).
            # Leaving them un-drained wedges the module rings for the NEXT open on
            # the same device — the stale head/tail desyncs the puller counter.
            for slot, n in list(self._pending.items()):
                try:
                    self.lib.igp_drain_slot(self.h, slot, n)   # best-effort
                except Exception:
                    break
            self._pending.clear()
            self.lib.igp_close(self.h)
            self.h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ---- DGL dataloader adapter (drop-in parallel to GIDS_DGLDataLoader) ----
try:
    import dgl  # noqa: F401

    class ISE_DGLDataLoader:
        """Wrap a DGL DataLoader and append an ISEFS-fetched feature tensor to
        each batch, so a training loop `for input_nodes, output_nodes, blocks,
        feat in loader:` is identical to GIDS_DGLDataLoader — only the feature
        backend (ISE vs BaM) differs. This is the apples-to-apples head-to-head
        integration point (same graph/sampler/model).

        Prefetch pipeline (depth = isefs.ndst): the fetch for batch k+D is issued
        while batches k..k+D-1 are in flight, so feature delivery overlaps sampling
        and compute — parallel to GIDS's window buffering. Set ISEFS(ndst=2..3).
        With ndst=1 it degrades to synchronous fetch-then-yield.

        `dim` must equal the ISEFS feat_dim; `isefs.max_batch` >= the largest
        input-node set (a k-hop neighborhood is far larger than batch_size)."""

        def __init__(self, graph, indices, graph_sampler, batch_size, dim,
                     isefs, device="cuda", **kwargs):
            self.dl = dgl.dataloading.DataLoader(
                graph, indices, graph_sampler, batch_size=batch_size,
                device=device, **kwargs)
            self.fs = isefs
            self.dim = dim

        def __iter__(self):
            from collections import deque
            fs = self.fs
            depth = max(1, fs.ndst)
            it = iter(self.dl)
            q = deque()
            gen = [0]

            def issue():
                try:
                    input_nodes, output_nodes, blocks = next(it)
                except StopIteration:
                    return False
                g = gen[0]; gen[0] += 1
                fs.fetch_gen(g, input_nodes)              # async, into slot g%ndst
                q.append((g, int(input_nodes.numel()),
                          input_nodes, output_nodes, blocks))
                return True

            for _ in range(depth):                        # prime the pipeline
                issue()
            while q:
                g, n, in_n, out_n, blocks = q.popleft()
                fs.wait_gen(g, n)                         # block on this batch only
                feat = fs.view_gen(g, n).clone()          # clone BEFORE slot reuse
                issue()                                   # keep the pipeline full
                yield in_n, out_n, blocks, feat

        def __len__(self):
            return len(self.dl)

except Exception:  # DGL not installed — the direct ISEFS API still works
    ISE_DGLDataLoader = None


if __name__ == "__main__":
    # Smoke test: open, one fetch of a few nodes, print shape. Needs the module
    # loaded + backend seeded. feat_dim*dtype must equal coalesce_row.
    import sys
    dim = int(sys.argv[1]) if len(sys.argv) > 1 else 128  # 128*4B = 512B row
    fs = ISEFS(feat_dim=dim, dtype=torch.float32, max_batch=1024)
    print("shards:", fs.nr_shards())
    ids = torch.randint(0, 8192, (256,))
    x = fs.fetch(ids)
    print("fetched", tuple(x.shape), x.dtype, "on", x.device)
    fs.close()
