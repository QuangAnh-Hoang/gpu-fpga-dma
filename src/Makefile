# Host-side build for the NVMe HW data-plane bring-up.
# (Gather-era XRT host `main.cpp` and the CUDA probes now live in ../attic/gather-sw.)

.PHONY: all
all: nvme_probe

include Makefile.local

# libisefs: userspace feature-store transport over /dev/nvmev_l1 (M6, DD1).
# Shared lib for the PyG adapter (ctypes), plus the C verification probe.
.PHONY: isefs
isefs:
	mkdir -p obj/
	cc -O2 -Wall -fPIC -shared isefs.c -o ./obj/libisefs.so
	cc -O2 -Wall -c isefs.c -o ./obj/isefs.o

# GPU-pull backend for libisefs: persistent-kernel delivery as a shared lib
# for the isefs.py DGL adapter. Needs module coalesce_deliver=1.
.PHONY: isefs_gpupull
isefs_gpupull:
	mkdir -p obj/
	$(NVCC) -O2 -std=c++14 -arch=sm_86 --compiler-options "-fPIC" -shared \
		isefs_gpupull.cu -o ./obj/libisefs_gpupull.so -lcuda -lcudart

clean:
	rm -rf obj/
