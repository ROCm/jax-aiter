#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Level 2 holistic bench: one full MLP block fwd+bwd on a single GPU.

No FSDP, no scan, no remat. Isolates per-layer compute for three recipes and
two sizes.

Recipes (selected by env when launching, this script just uses gemm_fp4_bf16
for aiter paths and native jax ops for FP8 / BF16):
  hybrid  : AITER_ALL_FP4=0 ... (picks up env at import time)
  all_fp4 : AITER_ALL_FP4=1
  fp8     : native hipBLASLt FP8 via lax.dot_general with per-tensor amax scaling
  bf16    : plain lax.dot_general (reference)

Shapes:
  8B  : M=73728 (batch=9 * seq=8192), hidden=4096, intermediate=14336
  70B : M=81920 (batch=10 * seq=8192) OR scaled per-GPU 10240, hidden=8192,
        intermediate=28672

Default 70B M is 81920 matching other kernel-level benches. Use --scaled-70b
to drop to 10240 (per-FSDP-shard M) if the full shape OOMs.

Usage (inside container, single GPU):
  cd /ruvaidya/aiter_proj/jax-aiter && \\
    JA_ROOT_DIR=$PWD AITER_ASM_DIR=$PWD/third_party/aiter/hsa/ \\
    AITER_SYMBOL_VISIBLE=1 GPU_ARCHS=gfx950 \\
    HIP_VISIBLE_DEVICES=0 \\
    AITER_ALL_FP4=<0|1> AITER_FP4_DA=1 AITER_FUSED_QUANT=1 \\
    python3 benchmarks/bench_mlp_layer_holistic.py --recipe <hybrid|all_fp4|fp8|bf16> --size <8b|70b>
"""

from __future__ import annotations

import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np


SHAPES = {
    "8b":  dict(M=73728, K=4096, N=14336),
    "70b": dict(M=81920, K=8192, N=28672),
}

WARMUP = 3
ITERS = 8

FP8_MAX = 448.0


def build_mlp(recipe: str):
    """Return a loss function that runs one MLP block.

    block: h = silu(x @ W_gate^T) * (x @ W_up^T); out = h @ W_down^T; loss = mean(out^2)

    Weight shapes:
      W_gate, W_up: [N, K]  (rows along output dim N)
      W_down:       [K, N]
    """
    if recipe == "hybrid" or recipe == "all_fp4":
        from jax_aiter.gemm_fp4 import gemm_fp4_bf16
        gemm = gemm_fp4_bf16
    elif recipe == "fp8":
        def gemm(a_bf16, b_bf16):
            # a_bf16: [M, K], b_bf16: [N, K]. Compute a @ b.T via f8e4m3fn.
            eps = jnp.finfo(jnp.float32).tiny
            a_scale = jnp.float32(FP8_MAX) / (jnp.max(jnp.abs(a_bf16)) + eps)
            b_scale = jnp.float32(FP8_MAX) / (jnp.max(jnp.abs(b_bf16)) + eps)
            a_fp8 = (a_bf16.astype(jnp.float32) * a_scale).astype(jnp.float8_e4m3fn)
            b_fp8 = (b_bf16.astype(jnp.float32) * b_scale).astype(jnp.float8_e4m3fn)
            out = jax.lax.dot_general(
                a_fp8, b_fp8,
                (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.bfloat16,
            )
            return (out * (jnp.float32(1.0) / (a_scale * b_scale))).astype(jnp.bfloat16)
    elif recipe == "bf16":
        def gemm(a_bf16, b_bf16):
            return jax.lax.dot_general(
                a_bf16, b_bf16,
                (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.bfloat16,
            )
    else:
        raise ValueError(f"unknown recipe: {recipe}")

    def loss_fn(x, w_gate, w_up, w_down, target):
        gate = gemm(x, w_gate)
        up   = gemm(x, w_up)
        h    = jax.nn.silu(gate) * up
        out  = gemm(h, w_down)
        return jnp.mean((out - target) ** 2)

    return loss_fn


def bench_fn(fn, *args, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        out = fn(*args)
        jax.block_until_ready(out)
    jax.effects_barrier()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return np.median(times), np.min(times)


def tflops_3gemm(M, K, N, seconds):
    """3 GEMM forward FLOPs only: 2*M*K*N + 2*M*K*N + 2*M*N*K = 6*M*K*N."""
    return 6 * M * K * N / seconds / 1e12


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--recipe", choices=["hybrid", "all_fp4", "fp8", "bf16"], required=True)
    p.add_argument("--size", choices=["8b", "70b"], required=True)
    p.add_argument("--scaled-70b", action="store_true",
                   help="Use per-GPU-shard M=10240 for 70B instead of full 81920.")
    args = p.parse_args()

    shape = SHAPES[args.size]
    if args.size == "70b" and args.scaled_70b:
        shape = dict(shape, M=10240)

    M, K, N = shape["M"], shape["K"], shape["N"]
    print(f"=== Level 2: MLP block fwd+bwd (recipe={args.recipe}, size={args.size}, "
          f"M={M}, K={K}, N={N}) ===", flush=True)
    print(f"JAX {jax.__version__}  |  Devices: {jax.devices()}", flush=True)

    # Relevant env for aiter recipes
    print(f"AITER_ALL_FP4={os.environ.get('AITER_ALL_FP4', 'unset')}, "
          f"AITER_FP4_DA={os.environ.get('AITER_FP4_DA', 'unset')}, "
          f"AITER_FUSED_QUANT={os.environ.get('AITER_FUSED_QUANT', 'unset')}",
          flush=True)

    key = jax.random.PRNGKey(0)
    kx, kg, ku, kd, kt = jax.random.split(key, 5)
    x      = jax.random.normal(kx, (M, K), dtype=jnp.bfloat16)
    w_gate = jax.random.normal(kg, (N, K), dtype=jnp.bfloat16)
    w_up   = jax.random.normal(ku, (N, K), dtype=jnp.bfloat16)
    w_down = jax.random.normal(kd, (K, N), dtype=jnp.bfloat16)
    target = jax.random.normal(kt, (M, K), dtype=jnp.bfloat16)

    loss_fn = build_mlp(args.recipe)

    # JIT both forward-only and value_and_grad so compile doesn't pollute timing.
    fwd_only = jax.jit(loss_fn)
    vgrad    = jax.jit(jax.value_and_grad(loss_fn, argnums=(1, 2, 3)))

    # Warmup compile
    fwd_only(x, w_gate, w_up, w_down, target).block_until_ready()
    out = vgrad(x, w_gate, w_up, w_down, target)
    out[0].block_until_ready()

    fwd_med, fwd_min = bench_fn(fwd_only, x, w_gate, w_up, w_down, target)

    def _vgrad_wrapped(*a):
        v, (dg, du, dd) = vgrad(*a)
        return v, dg, du, dd
    vg_med, vg_min = bench_fn(_vgrad_wrapped, x, w_gate, w_up, w_down, target)

    print(f"\n-- Results --", flush=True)
    print(f"fwd only  :  median {fwd_med*1000:8.3f} ms  min {fwd_min*1000:8.3f} ms  "
          f"fwd-only TFLOP/s (3 GEMM): {tflops_3gemm(M, K, N, fwd_med):7.1f}", flush=True)
    print(f"fwd+bwd   :  median {vg_med*1000:8.3f} ms  min {vg_min*1000:8.3f} ms", flush=True)
    bwd_ms = (vg_med - fwd_med) * 1000
    print(f"bwd-only  :  {bwd_ms:8.3f} ms  (fwd+bwd - fwd)", flush=True)
    print(f"bwd/fwd   :  {bwd_ms/1000/fwd_med:5.2f}x", flush=True)
    print(flush=True)


if __name__ == "__main__":
    main()
