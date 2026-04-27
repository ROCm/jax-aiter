#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Level 4 holistic bench: one MLP block fwd+bwd under 8-way FSDP.

Activations are M-sharded across the FSDP mesh axis. Weights are FSDP-sharded
on their first (output/N) dim, matching MaxText's default FSDP layout. This
exercises:

- all-gather on weights before each forward GEMM
- reduce-scatter or all-reduce on the wgrad output (all-FP4 only)
- the asymmetric sharding that the hybrid path never exercises as cleanly

We measure total step time plus count collective ops in the compiled HLO.

Usage (inside container, 8 GPUs):
  cd /ruvaidya/aiter_proj/jax-aiter && \\
    JA_ROOT_DIR=$PWD AITER_ASM_DIR=$PWD/third_party/aiter/hsa/ \\
    AITER_SYMBOL_VISIBLE=1 GPU_ARCHS=gfx950 \\
    HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\
    AITER_ALL_FP4=<0|1> AITER_FP4_DA=1 AITER_FUSED_QUANT=1 \\
    python3 benchmarks/bench_fsdp_mlp_holistic.py --recipe <hybrid|all_fp4|fp8|bf16> \\
        --size <8b|70b>
"""

from __future__ import annotations

import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


SHAPES = {
    "8b":  dict(M=73728, K=4096, N=14336),
    "70b": dict(M=81920, K=8192, N=28672),
}

WARMUP = 3
ITERS = 8

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--recipe", choices=["hybrid", "all_fp4", "fp8", "bf16"], required=True)
    p.add_argument("--size", choices=["8b", "70b"], required=True)
    p.add_argument("--dump-hlo", action="store_true",
                   help="Also write compiled HLO to docs/logs/session22/")
    args = p.parse_args()

    shape = SHAPES[args.size]
    M, K, N = shape["M"], shape["K"], shape["N"]

    devices = jax.devices()
    n_dev = len(devices)
    if n_dev < 8:
        print(f"WARNING: expected 8 devices, got {n_dev}", flush=True)

    mesh = Mesh(devices, axis_names=("fsdp",))
    # Activations M-sharded (matches MaxText FSDP activation layout).
    x_spec = NamedSharding(mesh, P("fsdp", None))
    # Weights: FSDP-shard on first (output) dim, matching gate/up layout.
    w_out_spec = NamedSharding(mesh, P("fsdp", None))
    # Down projection has [K, N] -> shard on K makes the wgrad reduction axis
    # the FSDP axis. MaxText does this too.
    w_down_spec = NamedSharding(mesh, P("fsdp", None))

    print(f"=== Level 4 FSDP ({n_dev}x): MLP fwd+bwd  recipe={args.recipe}  "
          f"size={args.size}  M={M} K={K} N={N} ===", flush=True)
    print(f"AITER_ALL_FP4={os.environ.get('AITER_ALL_FP4', 'unset')}", flush=True)

    gemm = make_gemm(args.recipe)

    def loss_fn(x, w_gate, w_up, w_down, target):
        gate = gemm(x, w_gate)
        up = gemm(x, w_up)
        h = jax.nn.silu(gate) * up
        out = gemm(h, w_down)
        return jnp.mean((out - target) ** 2)

    vg = jax.jit(
        jax.value_and_grad(loss_fn, argnums=(1, 2, 3)),
        in_shardings=(x_spec, w_out_spec, w_out_spec, w_down_spec, x_spec),
        out_shardings=(NamedSharding(mesh, P()),
                       (w_out_spec, w_out_spec, w_down_spec)),
    )

    key = jax.random.PRNGKey(0)
    kx, kg, ku, kd, kt = jax.random.split(key, 5)
    x      = jax.device_put(jax.random.normal(kx, (M, K), dtype=jnp.bfloat16), x_spec)
    w_gate = jax.device_put(jax.random.normal(kg, (N, K), dtype=jnp.bfloat16), w_out_spec)
    w_up   = jax.device_put(jax.random.normal(ku, (N, K), dtype=jnp.bfloat16), w_out_spec)
    w_down = jax.device_put(jax.random.normal(kd, (K, N), dtype=jnp.bfloat16), w_down_spec)
    target = jax.device_put(jax.random.normal(kt, (M, K), dtype=jnp.bfloat16), x_spec)

    compiled = vg.lower(x, w_gate, w_up, w_down, target).compile()

    # HLO collective counts
    hlo = compiled.as_text()
    counts = {
        "all-gather": hlo.count("all-gather"),
        "reduce-scatter": hlo.count("reduce-scatter"),
        "all-reduce": hlo.count("all-reduce"),
        "collective-permute": hlo.count("collective-permute"),
        "custom_call_target": hlo.count("custom_call_target="),
        "__cublas$lt$matmul$f8": hlo.count("__cublas$lt$matmul$f8"),
        "GemmFp4FwdJA": hlo.count("GemmFp4FwdJA"),
        "CastMxfp4DualJA": hlo.count("CastMxfp4DualJA"),
        "CastMxfp4JA-only": hlo.count('custom_call_target="CastMxfp4JA"'),
    }

    if args.dump_hlo:
        tag = "all_fp4" if os.environ.get("AITER_ALL_FP4") == "1" else args.recipe
        outpath = f"/ruvaidya/aiter_proj/docs/logs/session22/hlo_fsdp_{tag}_{args.size}.hlo"
        try:
            with open(outpath, "w") as f:
                f.write(hlo)
            print(f"HLO written to {outpath}", flush=True)
        except OSError as e:
            print(f"WARN: could not write HLO: {e}", flush=True)

    def _vg_wrap(*a):
        v, (dg, du, dd) = vg(*a)
        return v, dg, du, dd

    for _ in range(WARMUP):
        out = _vg_wrap(x, w_gate, w_up, w_down, target)
        jax.block_until_ready(out)
    jax.effects_barrier()

    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        out = _vg_wrap(x, w_gate, w_up, w_down, target)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)

    med = float(np.median(times))
    mn = float(np.min(times))

    print(f"\n-- HLO collective counts --", flush=True)
    for k, v in counts.items():
        print(f"  {k:>25s}: {v}", flush=True)

    print(f"\n-- Timing --", flush=True)
    print(f"  fwd+bwd step   : median {med*1000:9.3f} ms   min {mn*1000:9.3f} ms", flush=True)
    fwd_fl = 6 * M * K * N  # 3 MLP GEMMs FLOPs (fwd only)
    total_fl = fwd_fl * 3   # fwd + dA + dB roughly
    print(f"  fwd-only TFLOP/s (3 GEMM): {fwd_fl / med / 1e12:7.1f}", flush=True)
    print(f"  total  TFLOP/s (fwd+dA+dB): {total_fl / med / 1e12:7.1f}", flush=True)
    print(flush=True)


if __name__ == "__main__":
    main()
