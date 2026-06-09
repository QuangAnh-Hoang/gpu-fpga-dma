/* SPDX-License-Identifier: GPL-2.0 */
/*
 * fpga_gdr — userspace ABI for the FPGA-GPUDirect helper driver (Milestone B).
 *
 * PIN:        pin a cudaMalloc'd GPU range via nvidia_p2p and return the PCIe
 *             bus (DMA) addresses the Alveo must target to reach that VRAM.
 * HMSS_READ/  raw 32-bit read/write of the XDMA shell's address_translator
 * HMSS_WRITE: (slave-bridge / HMSS) registers, so userspace can point the
 *             translation page table at the GPU bus addresses returned by PIN.
 * UNPIN:      release the current pin.
 */
#ifndef _FPGA_GDR_H_
#define _FPGA_GDR_H_

#include <linux/types.h>

#define FPGA_GDR_MAX_PAGES 256

struct fpga_gdr_pin {
	__u64 va;        /* in:  GPU virtual address, 64KB-aligned            */
	__u64 len;       /* in:  length in bytes, multiple of 64KB           */
	__u32 n_pages;   /* out: number of GPU pages pinned                  */
	__u32 page_size; /* out: GPU page size in bytes (e.g. 65536)         */
	__u64 dma_addrs[FPGA_GDR_MAX_PAGES]; /* out: PCIe bus addr per page  */
};

struct fpga_gdr_reg {
	__u32 offset;    /* in:  byte offset into the address_translator BAR */
	__u32 value;     /* in (write) / out (read)                          */
};

#define FPGA_GDR_MAGIC 'G'
#define FPGA_GDR_PIN        _IOWR(FPGA_GDR_MAGIC, 1, struct fpga_gdr_pin)
#define FPGA_GDR_UNPIN      _IO(FPGA_GDR_MAGIC, 2)
#define FPGA_GDR_HMSS_WRITE _IOW(FPGA_GDR_MAGIC, 3, struct fpga_gdr_reg)
#define FPGA_GDR_HMSS_READ  _IOWR(FPGA_GDR_MAGIC, 4, struct fpga_gdr_reg)

/* address_translator register offsets (from XRT xocl subdev/address_translator.c) */
#define HMSS_REG_ENTRY_NUM   0x08
#define HMSS_REG_BASE_LO     0x10
#define HMSS_REG_BASE_HI     0x14
#define HMSS_REG_ADDR_RANGE  0x18
#define HMSS_REG_PTABLE      0x800  /* page_table_phys[i].lo = 0x800 + i*8 */

#endif /* _FPGA_GDR_H_ */
