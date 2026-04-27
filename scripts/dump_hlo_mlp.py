#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Dump HLO of a single MLP block at production shapes for collective inspection.

Used to verify reduce-scatter / all-gather placement around FP4 backward GEMMs
after sharding changes. Run on a single node with 8-way FSDP emulation.

Usage (inside container):
    cd /ruvaidya/aiter_proj/jax-aiter && \\
      JA_ROOT_DIR=$PWD AITER_ASM_DIR=$PWD/third_party/aiter/hsa/ \\
      AITER_SYMBOL_VISIBLE=1 GPU_ARCHS=gfx950 \\
      HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\
      python3 scripts/dump_hlo_mlp.py --size 8b --out docs/logs/hlo_hybrid_8b.hlo

    # Dump 70B MLP block HLO:
    python3 scripts/dump_hlo_mlp.py --size 70b --out docs/logs/hlo_70b.hlo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


SHAPES = {
    "8b": dict(M=73728, hidden=4096, intermediate=14336),
    "70b": dict(M=81920, hidden=8192, intermediate=28672),
}


def _build_mlp_step(gemm_fp4_bf16):
    """One MLP block: gate = x @ W_gate, up = x @ W_up, silu(gate) * up @ W_down."""

    def mlp_loss(x, w_gate, w_up, w_down, target):
        gate = gemm_fp4_bf16(x, w_gate)
        up = gemm_fp4_bf16(x, w_up)
        hidden = jax.nn.silu(gate) * up
        out = gemm_fp4_bf16(hidden, w_down)
        return jnp.mean((out - target) ** 2)

    return jax.value_and_grad(mlp_loss, argnums=(1, 2, 3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=list(SHAPES.keys()), default="8b")
    parser.add_argument("--out", type=str, required=True,
                        help="Output HLO path (directory or file).")
    parser.add_argument("--stage", choices=["optimized", "unoptimized"],
                        default="optimized",
                        help="'optimized' runs the full XLA pipeline (best for "
                             "collective inspection); 'unoptimized' shows raw lowering.")
    args = parser.parse_args()

    devices = jax.devices()
    print(f"Devices: {devices}", flush=True)
    if len(devices) < 8:
        print(f"WARNING: expected 8 devices for FSDP emulation, got {len(devices)}",
              file=sys.stderr)

    shape = SHAPES[args.size]
    M = shape["M"]
    K = shape["hidden"]
    N = shape["intermediate"]

    from jax_aiter.gemm_fp4 import gemm_fp4_bf16

    mesh = Mesh(devices, axis_names=("fsdp",))
    x_spec = NamedSharding(mesh, P("fsdp", None))
    w_spec = NamedSharding(mesh, P(None, None))

    key = jax.random.PRNGKey(0)
    k_x, k_wg, k_wu, k_wd, k_t = jax.random.split(key, 5)
    x = jax.device_put(
        jax.random.normal(k_x, (M, K), dtype=jnp.bfloat16), x_spec)
    w_gate = jax.device_put(
        jax.random.normal(k_wg, (N, K), dtype=jnp.bfloat16), w_spec)
    w_up = jax.device_put(
        jax.random.normal(k_wu, (N, K), dtype=jnp.bfloat16), w_spec)
    w_down = jax.device_put(
        jax.random.normal(k_wd, (K, N), dtype=jnp.bfloat16), w_spec)
    target = jax.device_put(
        jax.random.normal(k_t, (M, K), dtype=jnp.bfloat16), x_spec)

    value_and_grad_fn = _build_mlp_step(gemm_fp4_bf16)
    jitted = jax.jit(value_and_grad_fn,
                     in_shardings=(x_spec, w_spec, w_spec, w_spec, x_spec),
                     out_shardings=(NamedSharding(mesh, P()),
                                    (w_spec, w_spec, w_spec)))

    lowered = jitted.lower(x, w_gate, w_up, w_down, target)

    out_path = Path(args.out)
    if out_path.is_dir() or args.out.endswith("/"):
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / f"hlo_fp4_{args.size}.hlo"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_file = out_path

    if args.stage == "optimized":
        compiled = lowered.compile()
        text = compiled.as_text()
    else:
        text = lowered.as_text()
    out_file.write_text(text)
    print(f"Wrote {len(text)} bytes of HLO to {out_file}", flush=True)

    collectives = {
        "all-gather": text.count("all-gather"),
        "reduce-scatter": text.count("reduce-scatter"),
        "all-reduce": text.count("all-reduce"),
        "collective-permute": text.count("collective-permute"),
    }
    ffi_calls = text.count("custom_call_target=")
    fp8_dot = text.count("__cublas$lt$matmul$f8")

    print("Summary:", flush=True)
    for k, v in collectives.items():
        print(f"  {k:>20s}: {v}")
    print(f"  {'custom_call_target':>20s}: {ffi_calls}")
    print(f"  {'__cublas$lt$matmul$f8':>20s}: {fp8_dot}")


if __name__ == "__main__":
    main()
