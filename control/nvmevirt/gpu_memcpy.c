// SPDX-License-Identifier: GPL-2.0-only
/*
 * High-performance streaming memcpy for GPU write-combining memory.
 *
 * Adapted from GDRCopy (Copyright (c) 2014-2021 NVIDIA CORPORATION, MIT license).
 * Ported to kernel space with kernel_fpu_begin/end wrapping handled by callers.
 *
 * Write path: AVX _mm256_stream_pd (non-temporal 256-bit stores)
 * Read path:  SSE4.1 _mm_stream_load_si128 (non-temporal 128-bit loads)
 */

#include <linux/kernel.h>
#include <linux/string.h>
#include <linux/io.h>

#include <asm/fpu/api.h>
#include <asm/simd.h>

#pragma GCC push_options

/*
 * Prevent immintrin.h from pulling in mm_malloc.h → stdlib.h which
 * does not exist in kernel build environments.
 */
#define _MM_MALLOC_H_INCLUDED
#define __MM_MALLOC_H

/*
 * Write: host RAM → GPU WC memory (non-temporal stores)
 *
 * Uses AVX 256-bit streaming stores with 4x unroll for maximum WC buffer
 * utilization. Each _mm256_stream_pd bypasses the CPU cache and writes
 * directly to the WC buffer, which coalesces into 64-byte PCIe TLPs.
 */
#pragma GCC target("avx,sse4.1")
#include <immintrin.h>

void nvmev_memcpy_to_gpu(void __iomem *dest, const void *src, size_t n)
{
	char *d = (char *)dest;
	uintptr_t d_int = (uintptr_t)d;
	const char *s = (const char *)src;
	uintptr_t s_int = (uintptr_t)s;

	/* Align dest to 32-byte boundary */
	if (d_int & 0x1f) {
		size_t head = min_t(size_t, 0x20 - (d_int & 0x1f), n);

		__builtin_memcpy(d, s, head);
		d += head; d_int += head;
		s += head; s_int += head;
		n -= head;
	}

	if (s_int & 0x1f) {
		/* Source unaligned: use unaligned loads */
		__m256d r0, r1, r2, r3;

		while (n >= 4 * sizeof(__m256d)) {
			r0 = _mm256_loadu_pd((const double *)(s + 0 * sizeof(__m256d)));
			r1 = _mm256_loadu_pd((const double *)(s + 1 * sizeof(__m256d)));
			r2 = _mm256_loadu_pd((const double *)(s + 2 * sizeof(__m256d)));
			r3 = _mm256_loadu_pd((const double *)(s + 3 * sizeof(__m256d)));
			_mm256_stream_pd((double *)(d + 0 * sizeof(__m256d)), r0);
			_mm256_stream_pd((double *)(d + 1 * sizeof(__m256d)), r1);
			_mm256_stream_pd((double *)(d + 2 * sizeof(__m256d)), r2);
			_mm256_stream_pd((double *)(d + 3 * sizeof(__m256d)), r3);
			s += 4 * sizeof(__m256d);
			d += 4 * sizeof(__m256d);
			n -= 4 * sizeof(__m256d);
		}
		while (n >= sizeof(__m256d)) {
			r0 = _mm256_loadu_pd((const double *)s);
			_mm256_stream_pd((double *)d, r0);
			s += sizeof(__m256d);
			d += sizeof(__m256d);
			n -= sizeof(__m256d);
		}
	} else {
		/* Both aligned: use aligned loads + 8x unroll */
		__m256d r0, r1, r2, r3, r4, r5, r6, r7;

		while (n >= 8 * sizeof(__m256d)) {
			r0 = _mm256_load_pd((const double *)(s + 0 * sizeof(__m256d)));
			r1 = _mm256_load_pd((const double *)(s + 1 * sizeof(__m256d)));
			r2 = _mm256_load_pd((const double *)(s + 2 * sizeof(__m256d)));
			r3 = _mm256_load_pd((const double *)(s + 3 * sizeof(__m256d)));
			r4 = _mm256_load_pd((const double *)(s + 4 * sizeof(__m256d)));
			r5 = _mm256_load_pd((const double *)(s + 5 * sizeof(__m256d)));
			r6 = _mm256_load_pd((const double *)(s + 6 * sizeof(__m256d)));
			r7 = _mm256_load_pd((const double *)(s + 7 * sizeof(__m256d)));
			_mm256_stream_pd((double *)(d + 0 * sizeof(__m256d)), r0);
			_mm256_stream_pd((double *)(d + 1 * sizeof(__m256d)), r1);
			_mm256_stream_pd((double *)(d + 2 * sizeof(__m256d)), r2);
			_mm256_stream_pd((double *)(d + 3 * sizeof(__m256d)), r3);
			_mm256_stream_pd((double *)(d + 4 * sizeof(__m256d)), r4);
			_mm256_stream_pd((double *)(d + 5 * sizeof(__m256d)), r5);
			_mm256_stream_pd((double *)(d + 6 * sizeof(__m256d)), r6);
			_mm256_stream_pd((double *)(d + 7 * sizeof(__m256d)), r7);
			s += 8 * sizeof(__m256d);
			d += 8 * sizeof(__m256d);
			n -= 8 * sizeof(__m256d);
		}
		while (n >= sizeof(__m256d)) {
			r0 = _mm256_load_pd((const double *)s);
			_mm256_stream_pd((double *)d, r0);
			s += sizeof(__m256d);
			d += sizeof(__m256d);
			n -= sizeof(__m256d);
		}
	}

	/* Handle remainder (< 32 bytes) */
	if (n)
		__builtin_memcpy(d, s, n);

	_mm_sfence();
}

/*
 * Read: GPU WC memory → host RAM (non-temporal loads)
 *
 * Uses SSE4.1 MOVNTDQA streaming loads with 8x unroll. This instruction
 * is specifically designed for reading from WC/uncacheable memory and
 * avoids polluting CPU caches.
 */
void nvmev_memcpy_from_gpu(void *dest, const void __iomem *src, size_t n)
{
	char *d = (char *)dest;
	uintptr_t d_int = (uintptr_t)d;
	const char *s = (const char *)src;
	uintptr_t s_int = (uintptr_t)s;

	/* Align src to 16-byte boundary for streaming loads */
	if (s_int & 0xf) {
		size_t head = min_t(size_t, 0x10 - (s_int & 0x0f), n);

		__builtin_memcpy(d, s, head);
		d += head; d_int += head;
		s += head; s_int += head;
		n -= head;
	}

	if (d_int & 0xf) {
		/* Dest unaligned: streaming loads + unaligned stores */
		__m128i r0, r1, r2, r3, r4, r5, r6, r7;

		while (n >= 8 * sizeof(__m128i)) {
			r0 = _mm_stream_load_si128((__m128i *)(s + 0 * sizeof(__m128i)));
			r1 = _mm_stream_load_si128((__m128i *)(s + 1 * sizeof(__m128i)));
			r2 = _mm_stream_load_si128((__m128i *)(s + 2 * sizeof(__m128i)));
			r3 = _mm_stream_load_si128((__m128i *)(s + 3 * sizeof(__m128i)));
			r4 = _mm_stream_load_si128((__m128i *)(s + 4 * sizeof(__m128i)));
			r5 = _mm_stream_load_si128((__m128i *)(s + 5 * sizeof(__m128i)));
			r6 = _mm_stream_load_si128((__m128i *)(s + 6 * sizeof(__m128i)));
			r7 = _mm_stream_load_si128((__m128i *)(s + 7 * sizeof(__m128i)));
			_mm_storeu_si128((__m128i *)(d + 0 * sizeof(__m128i)), r0);
			_mm_storeu_si128((__m128i *)(d + 1 * sizeof(__m128i)), r1);
			_mm_storeu_si128((__m128i *)(d + 2 * sizeof(__m128i)), r2);
			_mm_storeu_si128((__m128i *)(d + 3 * sizeof(__m128i)), r3);
			_mm_storeu_si128((__m128i *)(d + 4 * sizeof(__m128i)), r4);
			_mm_storeu_si128((__m128i *)(d + 5 * sizeof(__m128i)), r5);
			_mm_storeu_si128((__m128i *)(d + 6 * sizeof(__m128i)), r6);
			_mm_storeu_si128((__m128i *)(d + 7 * sizeof(__m128i)), r7);
			s += 8 * sizeof(__m128i);
			d += 8 * sizeof(__m128i);
			n -= 8 * sizeof(__m128i);
		}
		while (n >= sizeof(__m128i)) {
			r0 = _mm_stream_load_si128((__m128i *)s);
			_mm_storeu_si128((__m128i *)d, r0);
			s += sizeof(__m128i);
			d += sizeof(__m128i);
			n -= sizeof(__m128i);
		}
	} else {
		/* Both aligned: streaming loads + streaming stores */
		__m128i r0, r1, r2, r3, r4, r5, r6, r7;

		while (n >= 8 * sizeof(__m128i)) {
			r0 = _mm_stream_load_si128((__m128i *)(s + 0 * sizeof(__m128i)));
			r1 = _mm_stream_load_si128((__m128i *)(s + 1 * sizeof(__m128i)));
			r2 = _mm_stream_load_si128((__m128i *)(s + 2 * sizeof(__m128i)));
			r3 = _mm_stream_load_si128((__m128i *)(s + 3 * sizeof(__m128i)));
			r4 = _mm_stream_load_si128((__m128i *)(s + 4 * sizeof(__m128i)));
			r5 = _mm_stream_load_si128((__m128i *)(s + 5 * sizeof(__m128i)));
			r6 = _mm_stream_load_si128((__m128i *)(s + 6 * sizeof(__m128i)));
			r7 = _mm_stream_load_si128((__m128i *)(s + 7 * sizeof(__m128i)));
			_mm_stream_si128((__m128i *)(d + 0 * sizeof(__m128i)), r0);
			_mm_stream_si128((__m128i *)(d + 1 * sizeof(__m128i)), r1);
			_mm_stream_si128((__m128i *)(d + 2 * sizeof(__m128i)), r2);
			_mm_stream_si128((__m128i *)(d + 3 * sizeof(__m128i)), r3);
			_mm_stream_si128((__m128i *)(d + 4 * sizeof(__m128i)), r4);
			_mm_stream_si128((__m128i *)(d + 5 * sizeof(__m128i)), r5);
			_mm_stream_si128((__m128i *)(d + 6 * sizeof(__m128i)), r6);
			_mm_stream_si128((__m128i *)(d + 7 * sizeof(__m128i)), r7);
			s += 8 * sizeof(__m128i);
			d += 8 * sizeof(__m128i);
			n -= 8 * sizeof(__m128i);
		}
		while (n >= sizeof(__m128i)) {
			r0 = _mm_stream_load_si128((__m128i *)s);
			_mm_stream_si128((__m128i *)d, r0);
			s += sizeof(__m128i);
			d += sizeof(__m128i);
			n -= sizeof(__m128i);
		}
	}

	if (n)
		__builtin_memcpy(d, s, n);

	_mm_sfence();
}

#pragma GCC pop_options
