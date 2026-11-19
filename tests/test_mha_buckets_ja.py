# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Tests for bounded group-mode calls sharing one output (flash_attn_varlen_buckets)."""

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

mha = pytest.importorskip("jax_aiter.mha", reason="AITER MHA libraries missing")
from jax_aiter.mha.mha import _resolve_fwd_dispatch

ROW = 128
# Per-row document lengths; every row but the third ends in padding.
ROWS = [[5, 12, 30, 3, 50], [90, 20], [128], [40, 7, 60]]
BOUNDS = (16, 32, 64, 128)
SLOTS = (6, 3, 4, 3)


def _documents():
    starts, lengths = [], []
    for r, segs in enumerate(ROWS):
        offset = r * ROW
        for length in segs:
            starts.append(offset)
            lengths.append(length)
            offset += length
    return np.asarray(starts), np.asarray(lengths), ROW * len(ROWS)


def _bucket_layout():
    """Per-bucket (seqstart, cu_seqlen) and token owners, as MaxText builds them."""
    starts, lengths, total = _documents()
    buckets, owners, lo = [], [], 0
    for hi, slots in zip(BOUNDS, SLOTS):
        keep = (lengths > lo) & (lengths <= hi)
        assert keep.sum() <= slots
        s = np.concatenate([starts[keep], np.full(slots - keep.sum(), total)])
        n = np.concatenate([lengths[keep], np.zeros(slots - keep.sum(), np.int64)])
        seqstart = np.append(s, total).astype(np.int32)
        cu = np.concatenate([[0], np.cumsum(n)]).astype(np.int32)
        owner = np.zeros(total, bool)
        for a, m in zip(starts[keep], lengths[keep]):
            owner[a:a + m] = True
        buckets.append((jnp.asarray(seqstart), jnp.asarray(cu)))
        owners.append(jnp.asarray(owner))
        lo = hi
    seg_id = np.zeros(total, np.int32)
    for i, (a, m) in enumerate(zip(starts, lengths)):
        seg_id[a:a + m] = i + 1
    return tuple(buckets), tuple(owners), seg_id, total


def _inputs(hq, d_qk, d_v, total, seg_id, seed=0):
    rng = np.random.default_rng(seed)
    q, k = (rng.normal(size=(total, hq, d_qk)).astype(np.float32) for _ in range(2))
    v = rng.normal(size=(total, hq, d_v)).astype(np.float32)
    dout = rng.normal(size=(total, hq, d_v)).astype(np.float32)
    dout[seg_id == 0] = 0.0
    bf = lambda x: jnp.asarray(x, jnp.bfloat16)
    return bf(q), bf(k), bf(v), bf(dout), 1.0 / np.sqrt(d_qk)


def _oracle(q, k, v, dout, seg_id, scale):
    """FP32 segment-masked causal attention and its gradients, on CPU."""
    cpu = jax.devices("cpu")[0]
    q, k, v, dout = (jax.device_put(np.asarray(x, np.float32), cpu) for x in (q, k, v, dout))
    seg = jax.device_put(jnp.asarray(seg_id), cpu)

    def attend(q, k, v):
        s = jnp.einsum("thd,shd->hts", q, k, precision="highest") * scale
        same = (seg[:, None] == seg[None, :]) & (seg[:, None] > 0)
        causal = jnp.arange(q.shape[0])[:, None] >= jnp.arange(q.shape[0])[None, :]
        mask = same & causal
        s = jnp.where(mask[None], s, -jnp.inf)
        p = jnp.where(mask[None], jax.nn.softmax(s, axis=-1), 0.0)
        out = jnp.einsum("hts,shd->thd", p, v, precision="highest")
        return jnp.where((seg > 0)[:, None, None], out, 0.0)

    out, vjp = jax.vjp(attend, q, k, v)
    return (np.asarray(out),) + tuple(np.asarray(g) for g in vjp(dout))


def _rel(got, ref, rows):
    got = np.asarray(got, np.float32)[rows]
    ref = np.asarray(ref, np.float32)[rows]
    return float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-12))


def _grads(attend, q, k, v, dout):
    def loss(q_, k_, v_):
        return jnp.sum(attend(q_, k_, v_).astype(jnp.float32) * dout.astype(jnp.float32))

    out = jax.jit(attend)(q, k, v)
    return (out,) + tuple(jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(q, k, v))


def _shared(buckets, owners, scale):
    return lambda q, k, v: mha.flash_attn_varlen_buckets(
        q, k, v, buckets, owners, BOUNDS, scale, True)


def _merged(buckets, owners, scale):
    """The per-call route: one full-size result per bucket, merged by owner."""
    def attend(q, k, v):
        out = jnp.zeros(q.shape[:2] + v.shape[-1:], q.dtype)
        for (seqstart, cu), owner, bound in zip(buckets, owners, BOUNDS):
            out_i = mha.flash_attn_varlen_auto(
                q, k, v, seqstart, seqstart, cu, cu, bound, bound, 0.0, scale, True, (-1, -1))
            out = jnp.where(owner[:, None, None], out_i, out)
        return out

    return attend


def _gpu_ready():
    try:
        return bool(jax.devices("gpu"))
    except Exception:
        return False


gpu = pytest.mark.skipif(not _gpu_ready(), reason="GPU missing")


def test_padded_logical_metadata_guards_asm_forward(monkeypatch):
    monkeypatch.delenv("JA_MHA_FWD_USE_ASM_V3", raising=False)
    monkeypatch.delenv("JA_MHA_FWD_FORCE_ASM_V3", raising=False)

    assert _resolve_fwd_dispatch(None)
    assert _resolve_fwd_dispatch(jnp.zeros((0,), jnp.int32))
    assert not _resolve_fwd_dispatch(jnp.asarray([0, 3, 3], jnp.int32))


def test_padded_asm_forward_guard_is_diagnostically_overridable(monkeypatch):
    logical = jnp.asarray([0, 3, 3], jnp.int32)
    monkeypatch.setenv("JA_MHA_FWD_FORCE_ASM_V3", "1")
    assert _resolve_fwd_dispatch(logical)

    monkeypatch.setenv("JA_MHA_FWD_USE_ASM_V3", "0")
    assert not _resolve_fwd_dispatch(logical)


@gpu
@pytest.mark.parametrize("hq,d_qk,d_v", [(2, 192, 128), (2, 128, 128)])
def test_shared_buckets_match_oracle_and_merged_calls(monkeypatch, hq, d_qk, d_v):
    monkeypatch.delenv("JA_MHA_VARLEN_BWD_BACKEND", raising=False)
    buckets, owners, seg_id, total = _bucket_layout()
    q, k, v, dout, scale = _inputs(hq, d_qk, d_v, total, seg_id)

    shared = _grads(_shared(buckets, owners, scale), q, k, v, dout)
    merged = _grads(_merged(buckets, owners, scale), q, k, v, dout)
    oracle = _oracle(q, k, v, dout, seg_id, scale)

    real = seg_id > 0
    for name, got, ref, alt in zip(("out", "dq", "dk", "dv"), shared, oracle, merged):
        got_np = np.asarray(got, np.float32)
        assert np.isfinite(got_np).all(), name
        assert not got_np[~real].any(), f"{name} padding rows nonzero"
        assert _rel(got, ref, real) < 3e-2, (name, _rel(got, ref, real))
        # Same kernels and metadata; only dQ's fp32 atomics may reorder.
        assert _rel(got, alt, real) < 1e-2, (name, _rel(got, alt, real))
    np.testing.assert_array_equal(np.asarray(shared[0]), np.asarray(merged[0]))


@gpu
def test_shared_buckets_reject_forced_triton(monkeypatch):
    monkeypatch.setenv("JA_MHA_VARLEN_BWD_BACKEND", "triton")
    buckets, owners, seg_id, total = _bucket_layout()
    q, k, v, _dout, scale = _inputs(2, 192, 128, total, seg_id)
    with pytest.raises(ValueError, match="AITER backward"):
        jax.jit(_shared(buckets, owners, scale))(q, k, v)
