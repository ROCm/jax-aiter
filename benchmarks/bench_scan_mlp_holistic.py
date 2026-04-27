#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Level 3 holistic bench: K MLP layers chained via jax.lax.scan on a single GPU.

Isolates scan overhead and remat tax. Compares ``minimal_flash``-style remat
(checkpoint attention only, which for our MLP-only micro means NO extra remat)
vs ``full`` remat (checkpoint everything, forward runs twice in backward).

This mirrors the asymmetric choices in MaxText:
  8B  uses remat_policy=minimal_flash
  70B uses remat_policy=full + param_scan_axis=1

The "remat tax" at 70B is the factor by which full remat slows bwd, which
should be ~= 1 + fwd/bwd ratio (since the forward runs a second time inside
backward).

Usage (inside container, single GPU):
  cd /ruvaidya/aiter_proj/jax-aiter && \\
    JA_ROOT_DIR=$PWD AITER_ASM_DIR=$PWD/third_party/aiter/hsa/ \\
    AITER_SYMBOL_VISIBLE=1 GPU_ARCHS=gfx950 \\
    HIP_VISIBLE_DEVICES=0 \\
    AITER_ALL_FP4=<0|1> AITER_FP4_DA=1 AITER_FUSED_QUANT=1 \\
    python3 benchmarks/bench_scan_mlp_holistic.py --recipe <hybrid|all_fp4|fp8|bf16> \\
        --size <8b|70b> --layers <K> --remat <none|full>
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
    # 70B: scale M down to per-GPU shard (full M=81920 needs too much memory with residuals
    # for a K-layer scan on 1 GPU).
    "70b": dict(M=10240, K=8192, N=28672),
}

WARMUP = 3
ITERS = 5

FP8_MAX = 448.0


def make_gemm(recipe: str):
    if recipe in ("hybrid", "all_fp4"):
        from jax_aiter.gemm_fp4 import gemm_fp4_bf16
        return gemm_fp4_bf16
    if recipe == "fp8":
        def gemm(a_bf16, b_bf16):
            eps = jnp.finfo(jnp.float32).tiny
            a_scale = jnp.float32(FP8_MAX) / (jnp.max(jnp.abs(a_bf16)) + eps)
            b_scale = jnp.float32(FP8_MAX) / (jnp.max(jnp.abs(b_bf16)) + eps)
            a_fp8 = (a_bf16.astype(jnp.float32) * a_scale).astype(jnp.float8_e4m3fn)
            b_fp8 = (b_bf16.astype(jnp.float32) * b_scale).astype(jnp.float8_e4m3fn)
            out = jax.lax.dot_general(
                a_fp8, b_fp8, (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.bfloat16,
            )
            return (out * (jnp.float32(1.0) / (a_scale * b_scale))).astype(jnp.bfloat16)
        return gemm
    if recipe == "bf16":
        def gemm(a_bf16, b_bf16):
            return jax.lax.dot_general(
                a_bf16, b_bf16, (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.bfloat16,
            )
        return gemm
    raise ValueError(f"unknown recipe: {recipe}")


def build_scan_loss(recipe: str, remat: str):
    gemm = make_gemm(recipe)

    def mlp_block(x, params):
        w_gate, w_up, w_down = params
        gate = gemm(x, w_gate)
        up = gemm(x, w_up)
        h = jax.nn.silu(gate) * up
        return gemm(h, w_down)

    if remat == "full":
        mlp_block = jax.checkpoint(mlp_block, policy=None)  # None = checkpoint everything

    def scan_body(x, params):
        return mlp_block(x, params), None

    def loss_fn(x, all_params, target):
        final, _ = jax.lax.scan(scan_body, x, all_params)
        return jnp.mean((final - target) ** 2)

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--recipe", choices=["hybrid", "all_fp4", "fp8", "bf16"], required=True)
    p.add_argument("--size", choices=["8b", "70b"], required=True)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--remat", choices=["none", "full"], default="none")
    args = p.parse_args()

    K_LAYERS = args.layers
    shape = SHAPES[args.size]
    M, K, N = shape["M"], shape["K"], shape["N"]

    print(f"=== Level 3: scan({K_LAYERS} MLP) fwd+bwd  recipe={args.recipe}  "
          f"size={args.size}  M={M}  K={K}  N={N}  remat={args.remat} ===", flush=True)
    print(f"AITER_ALL_FP4={os.environ.get('AITER_ALL_FP4', 'unset')}", flush=True)

    key = jax.random.PRNGKey(0)
    kx, kwg, kwu, kwd, kt = jax.random.split(key, 5)
    x      = jax.random.normal(kx, (M, K), dtype=jnp.bfloat16)
    w_gate = jax.random.normal(kwg, (K_LAYERS, N, K), dtype=jnp.bfloat16)
    w_up   = jax.random.normal(kwu, (K_LAYERS, N, K), dtype=jnp.bfloat16)
    w_down = jax.random.normal(kwd, (K_LAYERS, K, N), dtype=jnp.bfloat16)
    target = jax.random.normal(kt, (M, K), dtype=jnp.bfloat16)

    loss_fn = build_scan_loss(args.recipe, args.remat)

    fwd = jax.jit(loss_fn)
    vg  = jax.jit(jax.value_and_grad(loss_fn, argnums=1))

    fwd(x, (w_gate, w_up, w_down), target).block_until_ready()
    out = vg(x, (w_gate, w_up, w_down), target)
    jax.block_until_ready(out)

    fwd_med, fwd_min = bench_fn(fwd, x, (w_gate, w_up, w_down), target)
    def _vg_wrap(*a):
        v, g = vg(*a)
        return (v, g[0], g[1], g[2])
    vg_med, vg_min = bench_fn(_vg_wrap, x, (w_gate, w_up, w_down), target)

    per_layer_fwd_ms = fwd_med * 1000 / K_LAYERS
    per_layer_total_ms = vg_med * 1000 / K_LAYERS
    per_layer_bwd_ms = per_layer_total_ms - per_layer_fwd_ms

    print(f"\n-- Results ({K_LAYERS} layers) --", flush=True)
    print(f"fwd only   : median {fwd_med*1000:9.3f} ms   per-layer {per_layer_fwd_ms:7.3f} ms",
          flush=True)
    print(f"fwd+bwd    : median {vg_med*1000:9.3f} ms   per-layer {per_layer_total_ms:7.3f} ms",
          flush=True)
    print(f"bwd alone  : {(vg_med-fwd_med)*1000:9.3f} ms   per-layer {per_layer_bwd_ms:7.3f} ms",
          flush=True)
    print(f"bwd/fwd    : {(vg_med-fwd_med)/fwd_med:5.2f}x   "
          f"(remat=full expected ~2.0-3.0x; remat=none expected ~1.0-1.5x)", flush=True)
    print(flush=True)


if __name__ == "__main__":
    main()
