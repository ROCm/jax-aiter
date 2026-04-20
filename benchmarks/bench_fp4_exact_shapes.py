#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmark FP4 GEMM at production Llama shapes.

Compares AITER FP4 ASM forward kernel throughput against:
  - AITER BF16 ASM (via gemm_fwd)
  - hipBLASLt BF16 (via jnp.matmul / lax.dot_general)

Usage (inside container, single GPU):
  HIP_VISIBLE_DEVICES=0 python3 benchmarks/bench_fp4_exact_shapes.py
"""

import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
import jax.numpy as jnp

from jax_aiter.gemm_fp4 import gemm_fp4
from jax_aiter.gemm_fp4.fp4_utils import (
    bf16_to_mxfp4, mxfp4_to_bf16, e8m0_shuffle, shuffle_weight,
)

WARMUP = 5
ITERS = 20

SHAPES = [
    ("attn_qo_fwd",     73728,  4096,  4096),
    ("mlp_gate_up_fwd", 73728, 14336,  4096),
    ("mlp_down_fwd",    73728,  4096, 14336),
    ("attn_qo_dB",       4096,  4096, 73728),
    ("mlp_gate_up_dB",   4096, 14336, 73728),
    ("mlp_down_dB",     14336,  4096, 73728),
    ("small_sq",         4096,  4096,  4096),
]


def tflops(M, N, K, elapsed_s):
    return 2.0 * M * N * K / elapsed_s / 1e12


def bench_fp4(M, N, K):
    """Benchmark FP4 GEMM: quantize + shuffle + kernel."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    a_bf16 = jax.random.normal(k1, (M, K), dtype=jnp.bfloat16)
    b_bf16 = jax.random.normal(k2, (N, K), dtype=jnp.bfloat16)

    a_p, a_s = bf16_to_mxfp4(a_bf16)
    b_p, b_s = bf16_to_mxfp4(b_bf16)
    b_p_sh = shuffle_weight(b_p)
    a_s_sh = e8m0_shuffle(a_s)
    b_s_sh = e8m0_shuffle(b_s)

    for _ in range(WARMUP):
        out = gemm_fp4(a_p, b_p_sh, a_s_sh, b_s_sh)
        out.block_until_ready()

    jax.effects_barrier()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        out = gemm_fp4(a_p, b_p_sh, a_s_sh, b_s_sh)
        out.block_until_ready()
    elapsed = (time.perf_counter() - t0) / ITERS

    out_f32 = out.astype(jnp.float32)
    has_nan = bool(jnp.any(jnp.isnan(out_f32)))
    has_inf = bool(jnp.any(jnp.isinf(out_f32)))

    ref = jnp.matmul(
        mxfp4_to_bf16(a_p, a_s).astype(jnp.float32),
        mxfp4_to_bf16(b_p, b_s).astype(jnp.float32).T,
    )
    abs_err = jnp.abs(out_f32 - ref)
    scale = jnp.maximum(jnp.abs(ref), 1.0)
    mean_rel = float(jnp.mean(abs_err / scale))

    return elapsed, tflops(M, N, K, elapsed), has_nan, has_inf, mean_rel


def bench_hipblaslt_bf16(M, N, K):
    """Benchmark hipBLASLt BF16 via jnp.matmul."""
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    a = jax.random.normal(k1, (M, K), dtype=jnp.bfloat16)
    b = jax.random.normal(k2, (N, K), dtype=jnp.bfloat16)

    for _ in range(WARMUP):
        out = jnp.matmul(a, b.T)
        out.block_until_ready()

    jax.effects_barrier()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        out = jnp.matmul(a, b.T)
        out.block_until_ready()
    elapsed = (time.perf_counter() - t0) / ITERS
    return elapsed, tflops(M, N, K, elapsed)


def bench_aiter_bf16(M, N, K):
    """Benchmark AITER BF16 ASM via gemm()."""
    try:
        from jax_aiter.gemm import gemm
    except ImportError:
        return None, None

    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    a = jax.random.normal(k1, (M, K), dtype=jnp.bfloat16)
    b = jax.random.normal(k2, (N, K), dtype=jnp.bfloat16)

    for _ in range(WARMUP):
        out = gemm(a, b)
        out.block_until_ready()

    jax.effects_barrier()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        out = gemm(a, b)
        out.block_until_ready()
    elapsed = (time.perf_counter() - t0) / ITERS
    return elapsed, tflops(M, N, K, elapsed)


def main():
    print(f"FP4 GEMM Benchmark — {jax.devices()[0].device_kind}")
    print(f"  Warmup: {WARMUP}, Iterations: {ITERS}")
    print()
    print(f"{'Shape':<20s} {'M':>6s} {'N':>6s} {'K':>6s} | "
          f"{'FP4 T/s':>8s} {'FP4 ms':>7s} {'RelErr':>8s} {'NaN':>4s} | "
          f"{'BF16h T/s':>9s} {'BF16a T/s':>9s} | "
          f"{'FP4/BF16h':>9s} {'FP4/BF16a':>9s}")
    print("-" * 130)

    for name, M, N, K in SHAPES:
        try:
            fp4_time, fp4_tflops, has_nan, has_inf, mean_rel = bench_fp4(M, N, K)
        except Exception as e:
            print(f"{name:<20s} {M:>6d} {N:>6d} {K:>6d} | FP4 FAILED: {e}")
            continue

        hipblas_time, hipblas_tflops = bench_hipblaslt_bf16(M, N, K)
        aiter_time, aiter_tflops = bench_aiter_bf16(M, N, K)

        fp4_vs_hipblas = f"{fp4_tflops / hipblas_tflops:.2f}x" if hipblas_tflops else "N/A"
        fp4_vs_aiter = f"{fp4_tflops / aiter_tflops:.2f}x" if aiter_tflops else "N/A"
        nan_str = "YES" if has_nan else "no"
        aiter_str = f"{aiter_tflops:>8.0f}" if aiter_tflops else "N/A"

        print(f"{name:<20s} {M:>6d} {N:>6d} {K:>6d} | "
              f"{fp4_tflops:>8.0f} {fp4_time*1000:>6.2f}ms {mean_rel:>8.4f} {nan_str:>4s} | "
              f"{hipblas_tflops:>9.0f} {aiter_str:>9s} | "
              f"{fp4_vs_hipblas:>9s} {fp4_vs_aiter:>9s}")

    print()


if __name__ == "__main__":
    main()
