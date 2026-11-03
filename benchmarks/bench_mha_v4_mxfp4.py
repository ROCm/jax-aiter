# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Time dense MXFP4 or MXFP6 FMHA v4 in JAX and, when aiter is importable, in PyTorch.

The timed call is the whole recipe: Hadamard Q, coalesced K, FP6-P V, then the
dense code object for the requested format and head dim. MXFP6 is HD256 only.
TFLOPS use 4*B*H*Sq*Sk*head_dim. PyTorch time is a CUDA graph of the same call.

    cd /dockerx/jax-aiter
    PYTHONPATH=$PWD:$PWD/third_party/aiter \
      JA_ROOT_DIR=$PWD AITER_ASM_DIR=$PWD/third_party/aiter/hsa/ \
      GPU_ARCHS=gfx950 AITER_SYMBOL_VISIBLE=1 \
      python3 benchmarks/bench_mha_v4_mxfp4.py --format mxfp6
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Running ``python benchmarks/<script>.py`` puts only ``benchmarks`` on
# sys.path. Use this checkout when it has been built, and load the vendored
# aiter tree for the PyTorch comparison when that tree is present. A checkout
# with no build must not hide an installed wheel.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_AITER_ROOT = os.path.join(_REPO_ROOT, "third_party", "aiter")
_checkout_built = any(
    os.path.isfile(os.path.join(_REPO_ROOT, *parts))
    for parts in (
        ("build", "jax_aiter_build", "libjax_aiter.so"),
        ("jax_aiter", "_lib", "jax_aiter_build", "libjax_aiter.so"),
    )
)
if _checkout_built and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if os.path.isdir(os.path.join(_AITER_ROOT, "aiter")) and _AITER_ROOT not in sys.path:
    sys.path.insert(0, _AITER_ROOT)

import jax
import jax.numpy as jnp
import numpy as np

from jax_aiter.ja_compat.chip_info import get_gfx
from jax_aiter.ops.mha_v4 import mha_v4_mxfp4, mha_v4_mxfp6


def _flops(batch, sequence_q, sequence_k, heads, head_dim):
    return 4 * batch * heads * sequence_q * sequence_k * head_dim


def _jax_loop(fn, reps):
    compiled = jax.jit(fn)
    compiled().block_until_ready()

    @jax.jit
    def many():
        def body(_, acc):
            out = fn()
            return acc + out[0, 0, 0, 0].astype(jnp.float32)

        return jax.lax.fori_loop(0, reps, body, jnp.float32(0))

    many().block_until_ready()
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        many().block_until_ready()
        samples.append((time.perf_counter() - start) * 1e6)
    return sorted(samples)[len(samples) // 2] / reps


def _torch_times(q, k, v, scale, warmup, iters, format_name):
    """Replay a captured graph so the quant kernels and the ASM launch stay on device."""
    import torch
    from aiter.ops.mha_v4 import AttentionFormat, mha_v4

    fmt = AttentionFormat.MXFP6 if format_name == "mxfp6" else AttentionFormat.MXFP4
    host_q = np.asarray(q).view(np.uint16).copy()
    host_k = np.asarray(k).view(np.uint16).copy()
    host_v = np.asarray(v).view(np.uint16).copy()
    tq = torch.as_tensor(host_q, device="cuda").view(torch.bfloat16)
    tk = torch.as_tensor(host_k, device="cuda").view(torch.bfloat16)
    tv = torch.as_tensor(host_v, device="cuda").view(torch.bfloat16)

    def call():
        return mha_v4(tq, tk, tv, fmt, fmt, fmt, softmax_scale=scale)

    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def main():
    parser = argparse.ArgumentParser(description="Benchmark dense MXFP4 or MXFP6 FMHA v4")
    parser.add_argument("--skip-torch", action="store_true")
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--format", choices=("mxfp4", "mxfp6"), default="mxfp4")
    parser.add_argument("--head-dim", type=int, choices=(128, 256), default=None)
    args = parser.parse_args()
    if get_gfx() != "gfx950":
        raise SystemExit(f"{args.format} fmha_v4_fwd is for gfx950, got {get_gfx()}")
    if args.format == "mxfp6":
        head_dim = 256 if args.head_dim is None else args.head_dim
        if head_dim != 256:
            raise SystemExit("MXFP6 fmha_v4_fwd is head dim 256 only")
        launch = mha_v4_mxfp6
    else:
        head_dim = 128 if args.head_dim is None else args.head_dim
        launch = mha_v4_mxfp4

    shapes = [
        (1, 1024, 1024, 8),
        (1, 1024, 4096, 8),
        (1, 4096, 4096, 8),
        (4, 4096, 4096, 16),
        (4, 4096, 4096, 64),
        (8, 8192, 8192, 16),
    ]
    print(f"gfx={get_gfx()} format={args.format} head_dim={head_dim} devices={jax.devices()}")
    header = f"{'B':>3} {'Sq':>5} {'Sk':>5} {'H':>3} | {'jax us':>8} {'jax TF':>8}"
    if not args.skip_torch:
        header += f" | {'pt us':>8} {'pt TF':>8}"
    print(header)
    scale = head_dim ** -0.5
    for batch, sequence_q, sequence_k, heads in shapes:
        key = jax.random.PRNGKey(batch + sequence_q + sequence_k + heads)
        q = jax.random.normal(key, (batch, sequence_q, heads, head_dim), dtype=jnp.bfloat16)
        k = jax.random.normal(jax.random.fold_in(key, 1), (batch, sequence_k, heads, head_dim), dtype=jnp.bfloat16)
        v = jax.random.normal(jax.random.fold_in(key, 2), k.shape, dtype=jnp.bfloat16)
        micros = _jax_loop(lambda: launch(q, k, v, softmax_scale=scale), args.reps)
        work = _flops(batch, sequence_q, sequence_k, heads, head_dim)
        row = f"{batch:3d} {sequence_q:5d} {sequence_k:5d} {heads:3d} | {micros:8.1f} {work / micros / 1e6:8.1f}"
        if not args.skip_torch:
            event = _torch_times(q, k, v, scale, warmup=3, iters=10, format_name=args.format)
            row += f" | {event:8.1f} {work / event / 1e6:8.1f}"
        print(row, flush=True)


if __name__ == "__main__":
    main()
