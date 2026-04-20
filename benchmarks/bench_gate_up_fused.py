#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Kernel microbench: fused gate+up vs separate gate / up at MLP shapes.

Measures the wall-clock time for a full forward + backward step of a gated
MLP block when the gate and up projections are called separately versus
fused via ``gemm_fp4_gate_up_bf16`` (concat-GEMM-split).

Usage (inside container, single GPU):
    cd /ruvaidya/aiter_proj/jax-aiter && \\
      JA_ROOT_DIR=$PWD AITER_ASM_DIR=$PWD/third_party/aiter/hsa/ \\
      AITER_SYMBOL_VISIBLE=1 GPU_ARCHS=gfx950 \\
      HIP_VISIBLE_DEVICES=0 \\
      python3 benchmarks/bench_gate_up_fused.py [--size 8b|70b|both]
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np


SHAPES_8B = {
    "Llama3-8B (M=73728, N=14336, K=4096)": (73728, 14336, 4096),
}

SHAPES_70B = {
    "Llama3.3-70B (M=81920, N=28672, K=8192)": (81920, 28672, 8192),
}

WARMUP = 5
ITERS = 20


def bench(fn, *args):
    for _ in range(WARMUP):
        out = fn(*args)
        if isinstance(out, tuple):
            out[0].block_until_ready()
        else:
            out.block_until_ready()
    jax.effects_barrier()
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        out = fn(*args)
        if isinstance(out, tuple):
            out[0].block_until_ready()
        else:
            out.block_until_ready()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def count_ffi(fn, *args):
    hlo = fn.lower(*args).compile().as_text()
    return hlo.count("custom_call_target")


def run_one(name, M, N, K):
    from jax_aiter.gemm_fp4 import gemm_fp4_bf16, gemm_fp4_gate_up_bf16

    key = jax.random.PRNGKey(0)
    kx, kg, ku = jax.random.split(key, 3)
    x = jax.random.normal(kx, (M, K), dtype=jnp.bfloat16) * 0.1
    wg = jax.random.normal(kg, (N, K), dtype=jnp.bfloat16) * 0.1
    wu = jax.random.normal(ku, (N, K), dtype=jnp.bfloat16) * 0.1

    def swiglu_sep(x, wg, wu):
        g = gemm_fp4_bf16(x, wg)
        u = gemm_fp4_bf16(x, wu)
        return jnp.mean(jax.nn.silu(g) * u)

    def swiglu_fused(x, wg, wu):
        g, u = gemm_fp4_gate_up_bf16(x, wg, wu)
        return jnp.mean(jax.nn.silu(g) * u)

    sep_fn = jax.jit(jax.value_and_grad(swiglu_sep, argnums=(0, 1, 2)))
    fused_fn = jax.jit(jax.value_and_grad(swiglu_fused, argnums=(0, 1, 2)))

    t_sep = bench(sep_fn, x, wg, wu)
    t_fused = bench(fused_fn, x, wg, wu)

    ffi_sep = count_ffi(sep_fn, x, wg, wu)
    ffi_fused = count_ffi(fused_fn, x, wg, wu)

    # Total FLOPs: 2 projections × (fwd + dA + dB) × 2*M*N*K.
    flops_total = 2 * 3 * (2 * M * N * K)

    print(f"  {name}")
    print(f"    separate:  {t_sep*1000:8.2f} ms   {flops_total/t_sep/1e12:7.1f} TF/s   "
          f"{ffi_sep:3d} FFI")
    print(f"    fused:     {t_fused*1000:8.2f} ms   {flops_total/t_fused/1e12:7.1f} TF/s   "
          f"{ffi_fused:3d} FFI")
    print(f"    speedup:   {t_sep/t_fused:.2f}x   FFI savings: {ffi_sep-ffi_fused:+d}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=("8b", "70b", "both"), default="both")
    args = parser.parse_args()

    print(f"JAX {jax.__version__}  |  Devices: {jax.devices()}")
    print("=" * 80)
    print("Gate+Up fusion: single GPU kernel benchmark")
    print("=" * 80)

    if args.size in ("8b", "both"):
        for name, (M, N, K) in SHAPES_8B.items():
            run_one(name, M, N, K)

    if args.size in ("70b", "both"):
        for name, (M, N, K) in SHAPES_70B.items():
            run_one(name, M, N, K)


if __name__ == "__main__":
    main()
