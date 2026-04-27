#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Single-GPU kernel microbenchmark for the FP4 MLP backward path.

Measures the per-projection cost of fwd + dA + dB at 8B/70B production shapes
under three configurations:
  - Hybrid (current production): FP4 fwd + FP4 dA + native FP8 ``dot_general`` dB.
  - FP4 (default recipe):        FP4 fwd + FP4 dA + FP4 wgrad ``GemmFp4FwdJA``
                                  via ``_fp4_ffi_partitioned_wgrad``.
  - BF16 reference:              plain ``lax.dot_general`` for all three GEMMs.

Usage (inside container):
    cd /ruvaidya/aiter_proj/jax-aiter && \\
      JA_ROOT_DIR=$PWD AITER_ASM_DIR=$PWD/third_party/aiter/hsa/ \\
      AITER_SYMBOL_VISIBLE=1 GPU_ARCHS=gfx950 \\
      HIP_VISIBLE_DEVICES=0 \\
      python3 benchmarks/bench_fp4_mlp.py [--size 8b|70b|both]
"""

from __future__ import annotations

import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np


SHAPES_8B = {
    "gate_up_proj": (73728, 14336, 4096),
    "down_proj":    (73728, 4096,  14336),
    "attn_qo":      (73728, 4096,  4096),
}

SHAPES_70B = {
    "gate_up_proj": (81920, 28672, 8192),
    "down_proj":    (81920, 8192,  28672),
    "attn_qo":      (81920, 8192,  8192),
}

WARMUP = 5
ITERS = 20


def tflops(M, N, K, seconds):
    return 2 * M * N * K / seconds / 1e12


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


def run_one(name, M, N, K):
    key = jax.random.PRNGKey(0)
    kx, kw = jax.random.split(key)
    x = jax.random.normal(kx, (M, K), dtype=jnp.bfloat16) * 0.1
    w = jax.random.normal(kw, (N, K), dtype=jnp.bfloat16) * 0.1

    from jax_aiter.gemm_fp4 import gemm_fp4_bf16

    total_flops = 2 * M * N * K

    def mlp(x, w):
        y = gemm_fp4_bf16(x, w)
        return jnp.mean(y.astype(jnp.float32) ** 2)

    g_fn = jax.jit(jax.value_and_grad(mlp, argnums=(0, 1)))
    t_fp4 = bench(g_fn, x, w)

    def bf16_mlp(x, w):
        y = jax.lax.dot_general(x, w, (((1,), (1,)), ((), ())))
        return jnp.mean(y.astype(jnp.float32) ** 2)

    bf16_fn = jax.jit(jax.value_and_grad(bf16_mlp, argnums=(0, 1)))
    t_bf16 = bench(bf16_fn, x, w)

    # Total FLOPs for fwd + dA + dB ~= 3 * 2 * M * N * K.
    fwd_da_db_flops = 3 * total_flops

    def fmt(t):
        return f"{t*1000:7.2f} ms  {fwd_da_db_flops/t/1e12:7.1f} TF/s"

    print(f"  {name:<15s}  fp4={fmt(t_fp4)}  bf16={fmt(t_bf16)}  "
          f"fp4/bf16={t_bf16/t_fp4:.2f}x speedup")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=("8b", "70b", "both"), default="both")
    args = parser.parse_args()

    print(f"JAX {jax.__version__}  |  Devices: {jax.devices()}")
    print("=" * 120)

    if args.size in ("8b", "both"):
        print("8B production shapes (M=73728, hidden=4096, intermediate=14336):")
        for name, (M, N, K) in SHAPES_8B.items():
            run_one(name, M, N, K)
        print()

    if args.size in ("70b", "both"):
        print("70B production shapes (M=81920, hidden=8192, intermediate=28672):")
        for name, (M, N, K) in SHAPES_70B.items():
            run_one(name, M, N, K)


if __name__ == "__main__":
    main()
