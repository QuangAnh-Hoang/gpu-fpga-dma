// SPDX-License-Identifier: GPL-2.0-only
/*
 * High-performance streaming memcpy for GPU write-combining (WC) memory.
 *
 * Ported from GDRCopy's userspace SIMD routines to kernel space. Uses
 * non-temporal AVX/SSE streaming instructions to maximize throughput
 * over PCIe to GPU BAR1 ioremap_wc mappings.
 *
 * All public functions here require kernel_fpu_begin()/kernel_fpu_end()
 * to be called by the caller.
 */

#ifndef _NVMEV_GPU_MEMCPY_H
#define _NVMEV_GPU_MEMCPY_H

#ifdef CONFIG_NVMEVIRT_GPU_DIRECT

#include <linux/types.h>

/*
 * Copy from host RAM to GPU WC-mapped memory using non-temporal stores.
 * Uses AVX vmovntpd (256-bit) with 4x unroll if available, otherwise
 * falls back to SSE movntdq (128-bit).
 *
 * Must be called between kernel_fpu_begin() and kernel_fpu_end().
 */
void nvmev_memcpy_to_gpu(void __iomem *dest, const void *src, size_t n);

/*
 * Copy from GPU WC-mapped memory to host RAM using SSE4.1 streaming loads.
 * Uses movntdqa (128-bit non-temporal load) with 8x unroll.
 *
 * Must be called between kernel_fpu_begin() and kernel_fpu_end().
 */
void nvmev_memcpy_from_gpu(void *dest, const void __iomem *src, size_t n);

#endif /* CONFIG_NVMEVIRT_GPU_DIRECT */
#endif /* _NVMEV_GPU_MEMCPY_H */
