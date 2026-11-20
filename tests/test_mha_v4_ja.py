# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""BF16 dense fmha_v4_fwd ASM coverage, ported from aiter op_tests/test_mha_v4.py.

Quantized recipes in that file stay on the Torch producers. These cases cover
the BF16 dense code object: reference match, LSE, GQA, and per-batch key lengths.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from jax_aiter.ja_compat.chip_info import get_gfx
from jax_aiter.ops.mha_v4 import mha_v4


pytestmark = pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 BF16 fmha_v4_fwd")


def _cosine(a, b):
    af = np.asarray(a, dtype=np.float32).ravel()
    bf = np.asarray(b, dtype=np.float32).ravel()
    return float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))


def _attention(q, k, v, scale, seqlens=None):
    qf = np.asarray(q, dtype=np.float32)
    kf = np.asarray(k, dtype=np.float32)
    vf = np.asarray(v, dtype=np.float32)
    hq, hk = qf.shape[2], kf.shape[2]
    if hq != hk:
        rep = hq // hk
        kf = np.repeat(kf, rep, axis=2)
        vf = np.repeat(vf, rep, axis=2)
    scores = np.einsum("bqhd,bkhd->bhqk", qf, kf) * scale
    if seqlens is not None:
        pos = np.arange(kf.shape[1])[None, :]
        keep = pos < np.asarray(seqlens, dtype=np.int32)[:, None]
        scores = np.where(keep[:, None, None, :], scores, -np.inf)
    peak = np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores - peak)
    weights /= np.sum(weights, axis=-1, keepdims=True)
    return np.einsum("bhqk,bkhd->bqhd", weights, vf)


def _logsumexp(q, k, scale):
    qf = np.asarray(q, dtype=np.float32)
    kf = np.asarray(k, dtype=np.float32)
    scores = np.einsum("bqhd,bkhd->bhqk", qf, kf) * scale
    peak = np.max(scores, axis=-1, keepdims=True)
    return (peak + np.log(np.sum(np.exp(scores - peak), axis=-1, keepdims=True))).squeeze(-1)


def _rand(key, shape):
    return jax.random.normal(key, shape, dtype=jnp.bfloat16)


@pytest.mark.parametrize(
    ("b", "sq", "sk", "hq", "hk"),
    [
        (1, 128, 128, 4, 4),
        (1, 129, 257, 4, 4),
        (1, 512, 512, 5, 5),
        (2, 256, 256, 8, 2),
        (1, 257, 193, 1, 1),
    ],
)
def test_mha_v4_bf16_matches_reference(b, sq, sk, hq, hk):
    key = jax.random.PRNGKey(b + sq + sk + hq)
    q = _rand(key, (b, sq, hq, 128))
    k = _rand(jax.random.fold_in(key, 1), (b, sk, hk, 128))
    v = _rand(jax.random.fold_in(key, 2), (b, sk, hk, 128))
    scale = 128 ** -0.5
    out = mha_v4(q, k, v, softmax_scale=scale)
    ref = _attention(q, k, v, scale)
    assert out.dtype == jnp.bfloat16
    assert out.shape == q.shape
    assert np.isfinite(np.asarray(out, dtype=np.float32)).all()
    assert _cosine(out, ref) > 0.99
    np.testing.assert_allclose(np.asarray(out, dtype=np.float32), ref, rtol=2e-2, atol=2e-2)


def test_mha_v4_zero_inputs_are_finite():
    q = jnp.zeros((1, 64, 2, 128), dtype=jnp.bfloat16)
    out = mha_v4(q, q, q)
    got = np.asarray(out, dtype=np.float32)
    assert np.isfinite(got).all()
    assert np.max(np.abs(got)) < 1e-3


def test_mha_v4_dense_lse_matches_reference():
    key = jax.random.PRNGKey(31)
    q = _rand(key, (1, 512, 5, 128))
    k = _rand(jax.random.fold_in(key, 1), q.shape)
    v = _rand(jax.random.fold_in(key, 2), q.shape)
    scale = 128 ** -0.5
    out_only = mha_v4(q, k, v, softmax_scale=scale)
    out, lse = mha_v4(q, k, v, softmax_scale=scale, return_lse=True)
    ref = _logsumexp(q, k, scale)
    assert lse.shape == ref.shape
    assert lse.dtype == jnp.float32
    assert np.array_equal(np.asarray(out_only), np.asarray(out))
    err = np.abs(np.asarray(lse, dtype=np.float32) - ref)
    assert err.max() < 0.25, err.max()
    assert err.mean() < 0.05, err.mean()


def test_mha_v4_lse_bias_is_stable_across_key_chunks():
    key = jax.random.PRNGKey(31)
    sequence, heads, chunks = 1024, 4, 4
    q = _rand(key, (1, sequence, heads, 128))
    ramp = np.linspace(1.0, 6.0, sequence, dtype=np.float32)
    k = _rand(jax.random.fold_in(key, 1), q.shape).astype(jnp.float32) * ramp.reshape(1, -1, 1, 1)
    k = k.astype(jnp.bfloat16)
    v = _rand(jax.random.fold_in(key, 2), q.shape)
    scale = 128 ** -0.5
    span = sequence // chunks
    biases = []
    for start in range(0, sequence, span):
        key_chunk = k[:, start : start + span]
        _, lse = mha_v4(q, key_chunk, v[:, start : start + span], softmax_scale=scale, return_lse=True)
        ref = _logsumexp(q, key_chunk, scale)
        biases.append(np.asarray(lse, dtype=np.float32) - ref)
    spread = np.max(np.stack(biases), axis=0) - np.min(np.stack(biases), axis=0)
    assert spread.max() < 0.05, spread.max()


@pytest.mark.parametrize("lengths", [(300, 137), (512, 64), (1, 511), (65, 65)])
def test_mha_v4_seqlens_k_attends_over_each_batch_length(lengths):
    batch, sequence, heads = len(lengths), 512, 4
    key = jax.random.PRNGKey(41)
    q = _rand(key, (batch, sequence, heads, 128))
    k = _rand(jax.random.fold_in(key, 1), q.shape)
    v = _rand(jax.random.fold_in(key, 2), q.shape)
    seqlens = jnp.asarray(lengths, dtype=jnp.int32)
    out = mha_v4(q, k, v, seqlens_k=seqlens)
    scale = 128 ** -0.5
    for index, valid in enumerate(lengths):
        ref = _attention(q[index : index + 1], k[index : index + 1], v[index : index + 1], scale, seqlens=[valid])
        assert _cosine(out[index : index + 1], ref) > 0.99, lengths


def test_mha_v4_seqlens_k_at_full_length_is_the_dense_result():
    key = jax.random.PRNGKey(41)
    q = _rand(key, (2, 512, 4, 128))
    k = _rand(jax.random.fold_in(key, 1), q.shape)
    v = _rand(jax.random.fold_in(key, 2), q.shape)
    dense = mha_v4(q, k, v)
    full = mha_v4(q, k, v, seqlens_k=jnp.full((2,), 512, dtype=jnp.int32))
    assert np.array_equal(np.asarray(dense), np.asarray(full))


def test_mha_v4_rejects_unsupported_contracts():
    q = jnp.zeros((1, 32, 2, 64), dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="head dim"):
        mha_v4(q, q, q)
    q = jnp.zeros((1, 32, 4, 128), dtype=jnp.float16)
    with pytest.raises(ValueError, match="bfloat16"):
        mha_v4(q, q, q)
    q = jnp.zeros((1, 32, 6, 128), dtype=jnp.bfloat16)
    k = jnp.zeros((1, 32, 2, 128), dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="power of two"):
        mha_v4(q, k, k)
    q = jnp.zeros((2, 64, 2, 128), dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="int32"):
        mha_v4(q, q, q, seqlens_k=jnp.asarray([1.0, 2.0]))
    mask = jnp.ones((2, 2, 1, 1), dtype=jnp.bool_)
    with pytest.raises(NotImplementedError, match="key lengths"):
        mha_v4(q, q, q, block_mask=mask, seqlens_k=jnp.asarray([64, 64], dtype=jnp.int32))
