# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmark the JAX BF16 fmha_v4_fwd ASM kernels.

Dense shapes follow aiter op_tests/test_mha_v4.py. Sparse shapes and densities
follow op_tests/test_mha_v4_sparse.py. Quantized recipes are not timed: jax-aiter
launches the BF16 code objects only.

Usage (inside the container):
  cd /dockerx/jax-aiter
  JA_ROOT_DIR=$PWD AITER_ASM_DIR=$PWD/third_party/aiter/hsa/ \\
    GPU_ARCHS=gfx950 AITER_SYMBOL_VISIBLE=1 \\
    python3 benchmarks/bench_mha_v4.py
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from jax_aiter.ja_compat.chip_info import get_gfx
from jax_aiter.ops.mha_v4 import mha_v4, mha_v4_kv_tile, mha_v4_q_tile


def _cosine(actual, reference):
    got = np.asarray(actual, dtype=np.float32).ravel()
    ref = np.asarray(reference, dtype=np.float32).ravel()
    return float(np.dot(got, ref) / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-12))


def _dense_reference(q, k, v, scale):
    qf = np.asarray(q, dtype=np.float32)
    kf = np.asarray(k, dtype=np.float32)
    vf = np.asarray(v, dtype=np.float32)
    scores = np.einsum("bqhd,bkhd->bhqk", qf, kf) * scale
    peak = np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores - peak)
    weights /= np.sum(weights, axis=-1, keepdims=True)
    return np.einsum("bhqk,bkhd->bqhd", weights, vf)


def _sparse_reference(q, k, v, block_mask, scale, q_tile, kv_tile):
    qf = np.asarray(q, dtype=np.float32)
    kf = np.asarray(k, dtype=np.float32)
    vf = np.asarray(v, dtype=np.float32)
    scores = np.einsum("bqhd,bkhd->bhqk", qf, kf) * scale
    mask = np.asarray(block_mask)
    keep = np.repeat(mask, q_tile, axis=2)[:, :, : qf.shape[1], :]
    keep = np.repeat(keep, kv_tile, axis=3)[:, :, :, : kf.shape[1]]
    scores = np.where(keep, scores, -np.inf)
    finite = np.isfinite(scores)
    shifted = np.where(finite, scores, 0)
    peak = np.max(np.where(finite, shifted, -np.inf), axis=-1, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0)
    weights = np.exp(shifted - peak) * finite
    denom = np.sum(weights, axis=-1, keepdims=True)
    weights = np.divide(weights, denom, out=np.zeros_like(weights), where=denom > 0)
    return np.einsum("bhqk,bkhd->bqhd", weights, vf)


def _time(fn, warmup, iters):
    """Time a jitted call so the LUT build and the ASM launch share one executable."""
    compiled = jax.jit(fn)
    for _ in range(warmup):
        out = compiled()
        out.block_until_ready()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        out = compiled()
        out.block_until_ready()
        samples.append((time.perf_counter() - start) * 1e6)
    return out, float(np.median(samples))


def bench_dense(batch, sequence_q, sequence_k, heads, warmup, iters):
    scale = 128 ** -0.5
    key = jax.random.PRNGKey(batch + sequence_q + sequence_k + heads)
    q = jax.random.normal(key, (batch, sequence_q, heads, 128), dtype=jnp.bfloat16)
    k = jax.random.normal(jax.random.fold_in(key, 1), (batch, sequence_k, heads, 128), dtype=jnp.bfloat16)
    v = jax.random.normal(jax.random.fold_in(key, 2), k.shape, dtype=jnp.bfloat16)
    reference = _dense_reference(q, k, v, scale)
    output, micros = _time(lambda: mha_v4(q, k, v, softmax_scale=scale), warmup, iters)
    flops = 4 * batch * heads * sequence_q * sequence_k * 128
    nbytes = batch * heads * 128 * (sequence_q * 2 + sequence_k * 2) * 2
    return {
        "kind": "dense",
        "B": batch,
        "Sq": sequence_q,
        "Sk": sequence_k,
        "H": heads,
        "us": micros,
        "TFLOPS": flops / micros / 1e6,
        "TB/s": nbytes / micros / 1e6,
        "cosine": _cosine(output, reference),
    }


def bench_sparse(batch, sequence_q, sequence_k, heads, density, warmup, iters):
    scale = 128 ** -0.5
    q_tile = mha_v4_q_tile()
    kv_tile = mha_v4_kv_tile()
    if sequence_k % kv_tile != 0:
        raise ValueError(f"sequence_k must be a multiple of {kv_tile}")
    key = jax.random.PRNGKey(batch + sequence_q + sequence_k + heads)
    q = jax.random.normal(key, (batch, sequence_q, heads, 128), dtype=jnp.bfloat16)
    k = jax.random.normal(jax.random.fold_in(key, 1), (batch, sequence_k, heads, 128), dtype=jnp.bfloat16)
    v = jax.random.normal(jax.random.fold_in(key, 2), k.shape, dtype=jnp.bfloat16)
    q_tiles = (sequence_q + q_tile - 1) // q_tile
    kv_tiles = sequence_k // kv_tile
    keep = max(1, int(round(kv_tiles * density)))
    mask = jnp.zeros((batch, heads, q_tiles, kv_tiles), dtype=jnp.bool_)
    mask = mask.at[..., :keep].set(True)
    reference = _sparse_reference(q, k, v, mask, scale, q_tile, kv_tile)
    output, micros = _time(
        lambda: mha_v4(q, k, v, softmax_scale=scale, block_mask=mask), warmup, iters
    )
    visited_k = keep * kv_tile
    flops = 4 * batch * heads * sequence_q * visited_k * 128
    nbytes = batch * heads * 128 * (sequence_q * 2 + visited_k * 2) * 2
    row = {
        "kind": "sparse",
        "B": batch,
        "Sq": sequence_q,
        "Sk": sequence_k,
        "H": heads,
        "density": density,
        "kept": keep,
        "us": micros,
        "TFLOPS": flops / micros / 1e6,
        "TB/s": nbytes / micros / 1e6,
        "cosine": _cosine(output, reference),
    }
    if keep == kv_tiles:
        dense_out, dense_us = _time(lambda: mha_v4(q, k, v, softmax_scale=scale), warmup, iters)
        row["dense_us"] = dense_us
        row["dense_cosine"] = _cosine(dense_out, reference)
    return row


def _print(rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    for extra in rows:
        for key in extra:
            if key not in keys:
                keys.append(key)
    header = " | ".join(f"{key:>10}" for key in keys)
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = []
        for key in keys:
            value = row.get(key, "")
            if isinstance(value, float):
                cells.append(f"{value:10.3f}")
            else:
                cells.append(f"{value!s:>10}")
        print(" | ".join(cells))


def main():
    parser = argparse.ArgumentParser(description="Benchmark JAX BF16 FMHA v4 ASM")
    parser.add_argument("--batch", type=int, nargs="*", default=[1])
    parser.add_argument("--sequence-q", type=int, nargs="*", default=[1024])
    parser.add_argument("--sequence-k", type=int, nargs="*", default=[1024, 4096])
    parser.add_argument("--heads", type=int, nargs="*", default=[8])
    parser.add_argument("--sparse-sequence-q", type=int, nargs="*", default=[256, 512])
    parser.add_argument("--sparse-sequence-k", type=int, nargs="*", default=[1024, 2048])
    parser.add_argument("--density", type=float, nargs="*", default=[0.25, 0.5, 1.0])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--skip-sparse", action="store_true")
    args = parser.parse_args()

    gfx = get_gfx()
    if gfx != "gfx950":
        raise SystemExit(f"BF16 fmha_v4_fwd benchmark is for gfx950, got {gfx}")
    print(f"gfx={gfx} devices={jax.devices()}")

    rows = []
    if not args.skip_dense:
        for batch in args.batch:
            for sequence_q in args.sequence_q:
                for sequence_k in args.sequence_k:
                    for heads in args.heads:
                        rows.append(
                            bench_dense(batch, sequence_q, sequence_k, heads, args.warmup, args.iters)
                        )
        print("\nBF16 dense fmha_v4_fwd")
        _print([row for row in rows if row["kind"] == "dense"])

    sparse_rows = []
    if not args.skip_sparse:
        for batch in args.batch:
            for sequence_q in args.sparse_sequence_q:
                for sequence_k in args.sparse_sequence_k:
                    for heads in args.heads:
                        for density in args.density:
                            sparse_rows.append(
                                bench_sparse(
                                    batch, sequence_q, sequence_k, heads, density, args.warmup, args.iters
                                )
                            )
        print("\nBF16 sorted-sparse fmha_v4_fwd")
        _print(sparse_rows)


if __name__ == "__main__":
    main()
