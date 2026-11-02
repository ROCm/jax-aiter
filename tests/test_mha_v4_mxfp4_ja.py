# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MXFP4/MXFP6 FMHA v4 producers and dense ASM launches, checked against aiter."""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_aiter.ops.mha_v4 import (
    mha_v4_mxfp4,
    mha_v4_mxfp4_quantize_k,
    mha_v4_mxfp4_quantize_q,
    mha_v4_mxfp4_quantize_v,
    mha_v4_mxfp6,
)

_AITER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "third_party", "aiter"))
if _AITER_ROOT not in sys.path:
    sys.path.insert(0, _AITER_ROOT)

_LOG2E = 1.4426950408889634


def _torch():
    torch = pytest.importorskip("torch")
    pytest.importorskip("aiter")
    return torch


def _pair(batch, sequence_q, sequence_k, heads_q, heads_k, seed):
    key = jax.random.PRNGKey(seed)
    q = jax.random.normal(key, (batch, sequence_q, heads_q, 128), dtype=jnp.bfloat16)
    k = jax.random.normal(jax.random.fold_in(key, 1), (batch, sequence_k, heads_k, 128), dtype=jnp.bfloat16)
    v = jax.random.normal(jax.random.fold_in(key, 2), k.shape, dtype=jnp.bfloat16)
    return q, k, v


def _to_torch(torch, array):
    host = np.asarray(array)
    return torch.from_numpy(host.view(np.int16).astype(np.uint16)).view(torch.bfloat16).cuda()


def _zero_unwritten_k(raw, batch, sequence, heads):
    """Clear K-tile bytes the producer never stores.

    The coalesced kernel writes one 64-byte row per real token and leaves the
    rest of the final 128-token tile untouched. Those bytes are allocator
    garbage, so a full-buffer compare is only stable when the sequence fills
    the tile.
    """
    out = np.array(raw, copy=True)
    tail = sequence % 128
    if tail == 0:
        return out
    tiles = (sequence + 127) // 128
    for index in range(batch * heads):
        base = (index * tiles + tiles - 1) * 8192
        for token in range(tail, 128):
            for chunk in range(4):
                offset = base + chunk * 2048 + token * 16
                out[offset : offset + 16] = 0
    return out


@pytest.mark.parametrize(
    ("batch", "sequence", "heads"),
    [(1, 256, 4), (1, 300, 2), (2, 128, 2)],
)
def test_mxfp4_producers_match_aiter(batch, sequence, heads):
    torch = _torch()
    from aiter.ops.mha_v4_quant import (
        quantize_mxfp4_k,
        quantize_mxfp4_q,
        quantize_v_mxfp4_fp6_p,
    )

    q, k, v = _pair(batch, sequence, sequence, heads, heads, 7)
    multiplier = (128 ** -0.5) * _LOG2E
    jq, jqs = mha_v4_mxfp4_quantize_q(q, multiplier)
    jk, jks = mha_v4_mxfp4_quantize_k(k)
    jv, jvs = mha_v4_mxfp4_quantize_v(v)
    tq, tk, tv = _to_torch(torch, q), _to_torch(torch, k), _to_torch(torch, v)
    pq, pqs = quantize_mxfp4_q(tq, multiplier)
    pk, pks = quantize_mxfp4_k(tk)
    pv, pvs = quantize_v_mxfp4_fp6_p(tv)
    torch.cuda.synchronize()

    assert np.array_equal(np.asarray(jq), pq.cpu().numpy())
    assert np.array_equal(np.asarray(jqs)[: pqs.numel()], pqs.detach().cpu().numpy().reshape(-1))
    assert np.array_equal(
        _zero_unwritten_k(np.asarray(jk), batch, sequence, heads),
        _zero_unwritten_k(pk.cpu().numpy(), batch, sequence, heads),
    )
    assert np.array_equal(np.asarray(jks)[: pks.numel()], pks.detach().cpu().numpy().reshape(-1))
    assert np.array_equal(np.asarray(jv), pv.cpu().numpy())
    assert np.array_equal(np.asarray(jvs)[: pvs.numel()], pvs.detach().cpu().numpy().reshape(-1))


@pytest.mark.parametrize(
    ("sequence_q", "sequence_k", "heads_q", "heads_k"),
    [(256, 256, 4, 4), (256, 384, 8, 2), (129, 200, 4, 4)],
)
def test_mxfp4_dense_matches_aiter(sequence_q, sequence_k, heads_q, heads_k):
    torch = _torch()
    from aiter.ops.mha_v4 import AttentionFormat, mha_v4

    q, k, v = _pair(1, sequence_q, sequence_k, heads_q, heads_k, 11)
    got = np.asarray(mha_v4_mxfp4(q, k, v), dtype=np.float32)
    fmt = AttentionFormat.MXFP4
    ref = mha_v4(_to_torch(torch, q), _to_torch(torch, k), _to_torch(torch, v), fmt, fmt, fmt)
    torch.cuda.synchronize()
    ref_np = ref.float().cpu().numpy()
    cosine = float(np.dot(got.ravel(), ref_np.ravel()) / (np.linalg.norm(got) * np.linalg.norm(ref_np) + 1e-12))
    assert cosine > 0.999
    assert np.allclose(got, ref_np, rtol=2e-2, atol=2e-2)


def _pair_hd(batch, sequence_q, sequence_k, heads_q, heads_k, head_dim, seed):
    key = jax.random.PRNGKey(seed)
    q = jax.random.normal(key, (batch, sequence_q, heads_q, head_dim), dtype=jnp.bfloat16)
    k = jax.random.normal(
        jax.random.fold_in(key, 1), (batch, sequence_k, heads_k, head_dim), dtype=jnp.bfloat16
    )
    v = jax.random.normal(jax.random.fold_in(key, 2), k.shape, dtype=jnp.bfloat16)
    return q, k, v


@pytest.mark.parametrize(
    ("batch", "sequence", "heads"),
    [(1, 256, 1), (2, 512, 2)],
)
def test_mxfp4_hd256_producers_match_aiter(batch, sequence, heads):
    torch = _torch()
    from aiter.ops.mha_v4_quant import (
        quantize_mxfp4_k_hd256,
        quantize_mxfp4_q_hd256,
        quantize_v_mxfp4_fp6_p_hd256,
    )

    q, k, v = _pair_hd(batch, sequence, sequence, heads, heads, 256, 19)
    multiplier = (256 ** -0.5) * _LOG2E
    jq, jqs = mha_v4_mxfp4_quantize_q(q, multiplier)
    jk, jks = mha_v4_mxfp4_quantize_k(k)
    jv, jvs = mha_v4_mxfp4_quantize_v(v)
    tq, tk, tv = _to_torch(torch, q), _to_torch(torch, k), _to_torch(torch, v)
    pq, pqs = quantize_mxfp4_q_hd256(tq, multiplier)
    pk, pks = quantize_mxfp4_k_hd256(tk)
    pv, pvs = quantize_v_mxfp4_fp6_p_hd256(tv)
    torch.cuda.synchronize()

    assert np.array_equal(np.asarray(jq), pq.cpu().numpy())
    assert np.array_equal(np.asarray(jqs)[: pqs.numel()], pqs.detach().cpu().numpy().reshape(-1))
    assert np.array_equal(np.asarray(jk), pk.cpu().numpy())
    assert np.array_equal(np.asarray(jks)[: pks.numel()], pks.detach().cpu().numpy().reshape(-1))
    assert np.array_equal(np.asarray(jv), pv.cpu().numpy())
    assert np.array_equal(np.asarray(jvs)[: pvs.numel()], pvs.detach().cpu().numpy().reshape(-1))


@pytest.mark.parametrize(
    ("batch", "sequence_q", "sequence_k", "heads_q", "heads_k"),
    [(1, 256, 256, 1, 1), (2, 512, 512, 2, 2), (1, 768, 576, 2, 2), (1, 257, 320, 16, 4)],
)
def test_mxfp4_hd256_dense_matches_aiter(batch, sequence_q, sequence_k, heads_q, heads_k):
    torch = _torch()
    from aiter.ops.mha_v4 import AttentionFormat, mha_v4

    q, k, v = _pair_hd(batch, sequence_q, sequence_k, heads_q, heads_k, 256, 23)
    got = np.asarray(mha_v4_mxfp4(q, k, v), dtype=np.float32)
    fmt = AttentionFormat.MXFP4
    ref = mha_v4(_to_torch(torch, q), _to_torch(torch, k), _to_torch(torch, v), fmt, fmt, fmt)
    torch.cuda.synchronize()
    ref_np = ref.float().cpu().numpy()
    cosine = float(
        np.dot(got.ravel(), ref_np.ravel()) / (np.linalg.norm(got) * np.linalg.norm(ref_np) + 1e-12)
    )
    assert got.shape[-1] == 256
    assert cosine > 0.999
    assert np.allclose(got, ref_np, rtol=2e-2, atol=2e-2)


def _cosine(a, b):
    af = np.asarray(a, dtype=np.float32).ravel()
    bf = np.asarray(b, dtype=np.float32).ravel()
    return float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))


@pytest.mark.parametrize(
    ("head_dim", "lengths", "padded", "heads"),
    [
        (128, (96, 40), 128, 4),
        (256, (192, 128), 256, 4),
        (256, (80, 160, 40), 192, 2),
    ],
)
def test_mxfp4_seqlens_k_matches_truncated_batches(head_dim, lengths, padded, heads):
    """Per-batch key lengths match the same kernel on physically truncated K/V.

    A float32 reference sits near 0.98 cosine for MXFP4, which is looser than a
    broken scale-stride. A truncated tail is quantized separately from the padded
    final tile, so it is an accuracy control rather than a bitwise control.
    """
    batch = len(lengths)
    sequence_q = 128 if head_dim == 128 else 256
    q, k, v = _pair_hd(batch, sequence_q, padded, heads, heads, head_dim, 41)
    seqlens = jnp.asarray(lengths, dtype=jnp.int32)
    got = np.asarray(mha_v4_mxfp4(q, k, v, seqlens_k=seqlens))
    dense = np.asarray(mha_v4_mxfp4(q, k, v))
    assert not np.array_equal(got, dense)
    for index, valid in enumerate(lengths):
        truncated = np.asarray(
            mha_v4_mxfp4(
                q[index : index + 1],
                k[index : index + 1, :valid],
                v[index : index + 1, :valid],
            )
        )
        assert _cosine(got[index : index + 1], truncated) > 0.999
        np.testing.assert_allclose(got[index : index + 1], truncated, rtol=1e-1, atol=8e-2)


@pytest.mark.parametrize("head_dim", [128, 256])
def test_mxfp4_seqlens_k_full_length_matches_dense(head_dim):
    q, k, v = _pair_hd(2, 128, 128, 2, 2, head_dim, 43)
    dense = mha_v4_mxfp4(q, k, v)
    full = mha_v4_mxfp4(q, k, v, seqlens_k=jnp.full((2,), 128, dtype=jnp.int32))
    np.testing.assert_array_equal(np.asarray(dense), np.asarray(full))


def test_mxfp4_rejects_bad_seqlens_k():
    q, k, v = _pair_hd(2, 32, 32, 2, 2, 128, 1)
    with pytest.raises(ValueError, match="int32"):
        mha_v4_mxfp4(q, k, v, seqlens_k=jnp.asarray([1.0, 2.0]))


@pytest.mark.parametrize(
    ("head_dim", "heads_q", "heads_k", "bias"),
    [(128, 4, 4, 0.0), (256, 4, 2, 0.0), (128, 8, 2, 3.0), (256, 8, 2, 3.0)],
)
def test_mxfp4_lse_matches_aiter(head_dim, heads_q, heads_k, bias):
    """Log-sum-exp matches aiter, including the GQA K-mean restoration.

    ``bias`` opens the K-smoothing gate. The exported value is the quantized-score
    log-sum-exp plus ``softmax_scale * q · k_mean``, so the reference is aiter
    rather than ``torch.logsumexp`` of the exact scores.
    """
    torch = _torch()
    from aiter.ops.mha_v4 import AttentionFormat, mha_v4

    q, k, v = _pair_hd(1, 128, 160, heads_q, heads_k, head_dim, 47)
    if bias:
        k = (k.astype(jnp.float32) * 0.05 + bias).astype(jnp.bfloat16)
    got_out, got_lse = mha_v4_mxfp4(q, k, v, return_lse=True)
    out_only = mha_v4_mxfp4(q, k, v)
    fmt = AttentionFormat.MXFP4
    ref_out, ref_lse = mha_v4(
        _to_torch(torch, q),
        _to_torch(torch, k),
        _to_torch(torch, v),
        fmt,
        fmt,
        fmt,
        return_lse=True,
    )
    torch.cuda.synchronize()
    got_lse_np = np.asarray(got_lse, dtype=np.float32)
    ref_lse_np = ref_lse.float().cpu().numpy()
    err = np.max(np.abs(got_lse_np - ref_lse_np))
    assert got_lse.shape == ref_lse_np.shape
    assert np.isfinite(got_lse_np).all()
    assert np.array_equal(np.asarray(out_only), np.asarray(got_out))
    assert _cosine(got_out, ref_out.float().cpu().numpy()) > 0.999
    assert err < 0.05, err


def test_mxfp4_hd256_seqlens_k_and_lse_match_aiter():
    torch = _torch()
    from aiter.ops.mha_v4 import AttentionFormat, mha_v4

    lengths = (96, 160, 48)
    q, k, v = _pair_hd(3, 128, 192, 4, 2, 256, 53)
    seqlens = jnp.asarray(lengths, dtype=jnp.int32)
    got_out, got_lse = mha_v4_mxfp4(q, k, v, return_lse=True, seqlens_k=seqlens)
    fmt = AttentionFormat.MXFP4
    ref_out, ref_lse = mha_v4(
        _to_torch(torch, q),
        _to_torch(torch, k),
        _to_torch(torch, v),
        fmt,
        fmt,
        fmt,
        return_lse=True,
        seqlens_k=torch.tensor(lengths, device="cuda", dtype=torch.int32),
    )
    torch.cuda.synchronize()
    got_np = np.asarray(got_out, dtype=np.float32)
    ref_np = ref_out.float().cpu().numpy()
    lse_err = np.max(np.abs(np.asarray(got_lse, dtype=np.float32) - ref_lse.float().cpu().numpy()))
    assert _cosine(got_np, ref_np) > 0.999
    assert np.allclose(got_np, ref_np, rtol=2e-2, atol=2e-2)
    assert lse_err < 0.05, lse_err
    for index, valid in enumerate(lengths):
        truncated = mha_v4(
            _to_torch(torch, q[index : index + 1]),
            _to_torch(torch, k[index : index + 1, :valid]),
            _to_torch(torch, v[index : index + 1, :valid]),
            fmt,
            fmt,
            fmt,
        )
        torch.cuda.synchronize()
        assert _cosine(got_np[index : index + 1], truncated.float().cpu().numpy()) > 0.999


@pytest.mark.parametrize(
    ("batch", "sequence_q", "sequence_k", "heads_q", "heads_k"),
    [(1, 256, 256, 1, 1), (2, 512, 384, 4, 2), (1, 257, 320, 16, 4)],
)
def test_mxfp6_hd256_dense_matches_aiter(batch, sequence_q, sequence_k, heads_q, heads_k):
    """The HD256 MXFP6 JAX FFI matches AITER's new native HD256 row."""
    torch = _torch()
    from aiter.ops.mha_v4 import AttentionFormat, mha_v4

    q, k, v = _pair_hd(batch, sequence_q, sequence_k, heads_q, heads_k, 256, 61)
    got = np.asarray(mha_v4_mxfp6(q, k, v), dtype=np.float32)
    fmt = AttentionFormat.MXFP6
    ref = mha_v4(_to_torch(torch, q), _to_torch(torch, k), _to_torch(torch, v), fmt, fmt, fmt)
    torch.cuda.synchronize()
    ref_np = ref.float().cpu().numpy()
    assert got.shape == q.shape
    assert _cosine(got, ref_np) > 0.995
    assert np.allclose(got, ref_np, rtol=3e-2, atol=3e-2)


def test_mxfp6_hd256_jit_and_seqlens_k():
    """The custom-call producers trace and the ASM row reads per-batch lengths."""
    torch = _torch()
    from aiter.ops.mha_v4 import AttentionFormat, mha_v4

    lengths = (96, 160)
    q, k, v = _pair_hd(2, 256, 192, 4, 2, 256, 67)
    seqlens = jnp.asarray(lengths, dtype=jnp.int32)
    launch = jax.jit(lambda q, k, v, seqlens: mha_v4_mxfp6(q, k, v, seqlens_k=seqlens))
    got = np.asarray(launch(q, k, v, seqlens), dtype=np.float32)
    fmt = AttentionFormat.MXFP6
    ref = mha_v4(
        _to_torch(torch, q),
        _to_torch(torch, k),
        _to_torch(torch, v),
        fmt,
        fmt,
        fmt,
        seqlens_k=torch.tensor(lengths, device="cuda", dtype=torch.int32),
    )
    torch.cuda.synchronize()
    assert _cosine(got, ref.float().cpu().numpy()) > 0.995
