# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Comprehensive tests for unified MHA handlers (mha_v2).

Test strategy:
- Dtype-based tolerances (eps^(2/3))
- Dropout: shape/crash check only (no numeric comparison)
- Covers all head dims, seqlen combos, MHA/MQA/GQA, features, edge cases
- Regression guards for every historical bug found during AITER bump
"""

import math
import pytest
import jax
import jax.numpy as jnp
import numpy as np

from jax_aiter.mha import flash_attn_func, flash_attn_varlen


# ---------------------------------------------------------------------------
# Tolerances (matching TE: eps^(2/3))
# ---------------------------------------------------------------------------

def get_tolerances(dtype):
    if dtype == jnp.float16:
        eps = jnp.finfo(jnp.float16).eps
        tol = float(eps ** (2.0 / 3.0))
        return tol, tol
    elif dtype == jnp.bfloat16:
        eps = jnp.finfo(jnp.bfloat16).eps
        tol = float(eps ** (2.0 / 3.0))
        return tol, tol
    return 1e-5, 1e-5


def assert_close(actual, expected, dtype, name="", bwd_factor=1):
    atol, rtol = get_tolerances(dtype)
    atol *= bwd_factor
    a32 = actual.astype(jnp.float32)
    e32 = expected.astype(jnp.float32)
    max_diff = float(jnp.max(jnp.abs(a32 - e32)))
    max_ref = float(jnp.max(jnp.abs(e32)))
    rel_diff = max_diff / max(max_ref, 1e-6)
    assert max_diff < atol or rel_diff < rtol, \
        f"{name}: max_diff={max_diff:.6f} (atol={atol:.6f}), rel={rel_diff:.6f} (rtol={rtol:.6f})"


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

def attention_ref(q, k, v, causal=False, scale=None, window_size=(-1, -1)):
    q, k, v = q.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32)
    scale = scale or 1.0 / math.sqrt(q.shape[-1])
    sq, sk = q.shape[1], k.shape[1]
    attn = jnp.einsum("bshd,bthd->bhst", q, k) * scale
    if causal:
        mask = jnp.tril(jnp.ones((sq, sk), dtype=bool), k=sk - sq)
        attn = jnp.where(mask[None, None, :, :], attn, float("-inf"))
    if window_size != (-1, -1):
        wl, wr = window_size
        row_idx = jnp.arange(sq)[:, None] + (sk - sq)
        col_idx = jnp.arange(sk)[None, :]
        swa_mask = jnp.ones((sq, sk), dtype=bool)
        if wl >= 0:
            swa_mask = swa_mask & (row_idx - col_idx <= wl)
        if wr >= 0:
            swa_mask = swa_mask & (col_idx - row_idx <= wr)
        attn = jnp.where(swa_mask[None, None, :, :], attn, float("-inf"))
    attn = jax.nn.softmax(attn, axis=-1)
    return jnp.einsum("bhst,bthd->bshd", attn, v)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_qkv(b, sq, sk, hq, hk, d, dtype, seed=0):
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    q = jax.random.normal(k1, (b, sq, hq, d), dtype=dtype)
    k_t = jax.random.normal(k2, (b, sk, hk, d), dtype=dtype)
    v = jax.random.normal(k3, (b, sk, hk, d), dtype=dtype)
    return q, k_t, v


def make_varlen(batch, max_sq, max_sk, hq, hk, d, dtype, seed=0):
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    sq = jax.random.randint(k4, (batch,), 1, max_sq + 1)
    sk = jax.random.randint(k5, (batch,), 1, max_sk + 1)
    tq, tk = int(jnp.sum(sq)), int(jnp.sum(sk))
    cu_sq = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(sq).astype(jnp.int32)])
    cu_sk = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(sk).astype(jnp.int32)])
    q = jax.random.normal(k1, (tq, hq, d), dtype=dtype)
    k_t = jax.random.normal(k2, (tk, hk, d), dtype=dtype)
    v = jax.random.normal(k3, (tk, hk, d), dtype=dtype)
    return q, k_t, v, cu_sq, cu_sk, max_sq, max_sk


def run_fwd(q, k, v, **kw):
    result = flash_attn_func(q, k, v, **kw)
    return result[0] if isinstance(result, tuple) else result


def run_bwd(q, k, v, **kw):
    def loss(q, k, v):
        return jnp.sum(run_fwd(q, k, v, **kw))
    return jax.grad(loss, argnums=(0, 1, 2))(q, k, v)


def run_varlen_fwd(q, k, v, cu_sq, cu_sk, msq, msk, **kw):
    result = flash_attn_varlen(q, k, v, cu_sq, cu_sk,
                               max_seqlen_q=msq, max_seqlen_k=msk, **kw)
    return result[0] if isinstance(result, tuple) else result


def run_varlen_bwd(q, k, v, cu_sq, cu_sk, msq, msk, **kw):
    def loss(q, k, v):
        return jnp.sum(run_varlen_fwd(q, k, v, cu_sq, cu_sk, msq, msk, **kw))
    return jax.grad(loss, argnums=(0, 1, 2))(q, k, v)


# ===========================================================================
# BATCH CONFIGS
# ===========================================================================

# Core configs: cover all head dims, seqlens, MHA types
BATCH_CORE = [
    # Self-attention configs
    pytest.param(2, 128, 128, 4, 4, 32, jnp.bfloat16, id="d32_bf16"),
    pytest.param(2, 128, 128, 4, 4, 40, jnp.float16, id="d40_fp16"),
    pytest.param(2, 128, 128, 4, 4, 59, jnp.bfloat16, id="d59_bf16"),
    pytest.param(2, 128, 128, 4, 4, 64, jnp.bfloat16, id="d64_bf16"),
    pytest.param(2, 128, 128, 4, 4, 64, jnp.float16, id="d64_fp16"),
    pytest.param(2, 64, 64, 4, 4, 96, jnp.bfloat16, id="d96_bf16"),
    pytest.param(2, 64, 64, 4, 4, 96, jnp.float16, id="d96_fp16"),
    pytest.param(2, 64, 64, 4, 4, 111, jnp.float16, id="d111_fp16"),
    pytest.param(2, 128, 128, 4, 4, 128, jnp.bfloat16, id="d128_bf16"),
    pytest.param(2, 128, 128, 4, 4, 128, jnp.float16, id="d128_fp16"),
    pytest.param(2, 64, 64, 4, 4, 160, jnp.bfloat16, id="d160_bf16"),
    pytest.param(2, 64, 64, 4, 4, 256, jnp.bfloat16, id="d256_bf16"),
    # Cross-attention (sq != sk)
    pytest.param(2, 512, 256, 4, 4, 64, jnp.bfloat16, id="cross_sq_gt_sk"),
    pytest.param(2, 256, 512, 4, 4, 64, jnp.bfloat16, id="cross_sq_lt_sk"),
    pytest.param(2, 1024, 1023, 4, 4, 128, jnp.bfloat16, id="off_by_one"),
    pytest.param(2, 1023, 1024, 4, 4, 128, jnp.bfloat16, id="off_by_one_rev"),
    # GQA / MQA
    pytest.param(2, 128, 128, 6, 3, 64, jnp.bfloat16, id="gqa_bf16"),
    pytest.param(2, 128, 128, 6, 3, 64, jnp.float16, id="gqa_fp16"),
    pytest.param(2, 128, 128, 6, 1, 64, jnp.float16, id="mqa_fp16"),
    pytest.param(2, 128, 128, 8, 2, 128, jnp.bfloat16, id="gqa4_d128"),
    # Large seqlens
    pytest.param(2, 1024, 1024, 4, 4, 128, jnp.bfloat16, id="large_1024"),
    pytest.param(1, 2048, 2048, 4, 4, 64, jnp.bfloat16, id="large_2048"),
    # Decode (sq=1)
    pytest.param(2, 1, 128, 4, 4, 64, jnp.bfloat16, id="decode_sq1"),
    pytest.param(2, 1, 512, 4, 4, 128, jnp.bfloat16, id="decode_sq1_long"),
    # Larger batch
    pytest.param(8, 64, 64, 4, 4, 64, jnp.bfloat16, id="batch8"),
]

# Configs suitable for accuracy check (MHA, sq==sk, reasonable size)
BATCH_ACCURACY = [c for c in BATCH_CORE
                  if c.values[1] == c.values[2] and c.values[3] == c.values[4]
                  and c.values[1] <= 256 and c.values[5] <= 128]

# Varlen configs
VARLEN_CONFIGS = [
    pytest.param(4, 128, 128, 4, 4, 64, jnp.bfloat16, id="vl_self_bf16"),
    pytest.param(4, 128, 128, 4, 4, 64, jnp.float16, id="vl_self_fp16"),
    pytest.param(4, 256, 128, 4, 4, 128, jnp.bfloat16, id="vl_cross_bf16"),
    pytest.param(4, 128, 128, 6, 3, 64, jnp.bfloat16, id="vl_gqa_bf16"),
    pytest.param(4, 128, 128, 6, 1, 64, jnp.float16, id="vl_mqa_fp16"),
    pytest.param(4, 64, 64, 4, 4, 96, jnp.bfloat16, id="vl_d96_bf16"),
    pytest.param(4, 64, 64, 4, 4, 128, jnp.float16, id="vl_d128_fp16"),
    pytest.param(4, 64, 64, 4, 4, 32, jnp.bfloat16, id="vl_d32_bf16"),
    pytest.param(2, 512, 512, 4, 4, 64, jnp.bfloat16, id="vl_large_bf16"),
]


# ===========================================================================
# BATCH FORWARD TESTS
# ===========================================================================

@pytest.mark.parametrize("b,sq,sk,hq,hk,d,dtype", BATCH_CORE)
@pytest.mark.parametrize("causal", [False, True], ids=["nomask", "causal"])
def test_batch_fwd_shape(b, sq, sk, hq, hk, d, dtype, causal):
    """Forward: correct shape, dtype, finite values for all configs."""
    q, k_t, v = make_qkv(b, sq, sk, hq, hk, d, dtype)
    out = run_fwd(q, k_t, v, causal=causal)
    assert out.shape == (b, sq, hq, d)
    assert out.dtype == dtype
    assert jnp.all(jnp.isfinite(out)), f"NaN/Inf in output for d={d}"


@pytest.mark.parametrize("b,sq,sk,hq,hk,d,dtype", BATCH_ACCURACY)
@pytest.mark.parametrize("causal", [False, True], ids=["nomask", "causal"])
def test_batch_fwd_accuracy(b, sq, sk, hq, hk, d, dtype, causal):
    """Forward accuracy vs JAX reference."""
    q, k_t, v = make_qkv(b, sq, sk, hq, hk, d, dtype, seed=42)
    scale = d ** (-0.5)
    out = run_fwd(q, k_t, v, causal=causal, softmax_scale=scale)
    ref = attention_ref(q, k_t, v, causal=causal, scale=scale).astype(dtype)
    assert_close(out, ref, dtype, "fwd_out")


# ===========================================================================
# BATCH BACKWARD TESTS
# ===========================================================================

@pytest.mark.parametrize("b,sq,sk,hq,hk,d,dtype", BATCH_CORE)
@pytest.mark.parametrize("causal", [False, True], ids=["nomask", "causal"])
def test_batch_bwd_shape(b, sq, sk, hq, hk, d, dtype, causal):
    """Backward: gradient shapes, dtypes, finiteness."""
    q, k_t, v = make_qkv(b, sq, sk, hq, hk, d, dtype, seed=1)
    dq, dk, dv = run_bwd(q, k_t, v, causal=causal)
    assert dq.shape == q.shape, f"dq {dq.shape} != {q.shape}"
    assert dk.shape == k_t.shape, f"dk {dk.shape} != {k_t.shape}"
    assert dv.shape == v.shape, f"dv {dv.shape} != {v.shape}"
    assert dq.dtype == dtype
    assert jnp.all(jnp.isfinite(dq)), "dq NaN/Inf"
    assert jnp.all(jnp.isfinite(dk)), "dk NaN/Inf"
    assert jnp.all(jnp.isfinite(dv)), "dv NaN/Inf"


@pytest.mark.parametrize("b,sq,sk,hq,hk,d,dtype", BATCH_ACCURACY)
def test_batch_bwd_accuracy(b, sq, sk, hq, hk, d, dtype):
    """Backward accuracy: gradients vs JAX reference (10x relaxed tolerance).

    Head dims 96 / 111 / 128 used to be xfailed here for a CK/ASM backward
    accuracy limitation on gfx950, via an imperative pytest.xfail() that skipped
    the body outright. With the body actually running, all of them pass -- 5/5
    XPASS, stable over three repeats despite the backward's atomics -- so the
    limitation no longer holds on this stack and the exemption is gone. The
    varlen max_sk>256 xfails in TestRegressions are a different code path and
    still stand.
    """
    q, k_t, v = make_qkv(b, sq, sk, hq, hk, d, dtype, seed=2)
    scale = d ** (-0.5)

    def aiter_loss(q, k, v):
        return jnp.sum(run_fwd(q, k, v, softmax_scale=scale))
    def ref_loss(q, k, v):
        return jnp.sum(attention_ref(q, k, v, scale=scale).astype(dtype))

    dq, dk, dv = jax.grad(aiter_loss, argnums=(0, 1, 2))(q, k_t, v)
    dq_r, dk_r, dv_r = jax.grad(ref_loss, argnums=(0, 1, 2))(q, k_t, v)

    for n, g, r in [("dq", dq, dq_r), ("dk", dk, dk_r), ("dv", dv, dv_r)]:
        assert_close(g, r, dtype, n, bwd_factor=10)


# ===========================================================================
# DROPOUT TESTS (shape/crash only, no numeric comparison)
# ===========================================================================

@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("causal", [False, True], ids=["nomask", "causal"])
def test_dropout_fwd(dtype, causal):
    """Dropout forward: no crash, finite output."""
    q, k_t, v = make_qkv(2, 128, 128, 4, 4, 64, dtype, seed=10)
    out = run_fwd(q, k_t, v, dropout_p=0.1, causal=causal)
    assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16], ids=["fp16", "bf16"])
def test_dropout_bwd(dtype):
    """Dropout backward: no crash, finite gradients."""
    q, k_t, v = make_qkv(2, 64, 64, 4, 4, 64, dtype, seed=10)
    dq, dk, dv = run_bwd(q, k_t, v, dropout_p=0.1)
    assert jnp.all(jnp.isfinite(dq))
    assert jnp.all(jnp.isfinite(dk))
    assert jnp.all(jnp.isfinite(dv))


# ===========================================================================
# SWA (SLIDING WINDOW ATTENTION)
# ===========================================================================

@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16], ids=["fp16", "bf16"])
def test_swa_fwd(dtype):
    """SWA forward."""
    q, k_t, v = make_qkv(2, 256, 256, 4, 4, 64, dtype, seed=11)
    out = run_fwd(q, k_t, v, causal=True, window_size=(64, 0))
    assert out.shape == (2, 256, 4, 64)
    assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16], ids=["fp16", "bf16"])
def test_swa_bwd(dtype):
    """SWA backward."""
    q, k_t, v = make_qkv(2, 128, 128, 4, 4, 64, dtype, seed=11)
    dq, dk, dv = run_bwd(q, k_t, v, causal=True, window_size=(32, 0))
    assert jnp.all(jnp.isfinite(dq))
    assert jnp.all(jnp.isfinite(dk))


# ===========================================================================
# BIAS AND ALIBI
# ===========================================================================

@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16], ids=["fp16", "bf16"])
def test_bias_fwd(dtype):
    """Attention bias forward."""
    b, sq, sk, h, d = 2, 128, 128, 4, 64
    q, k_t, v = make_qkv(b, sq, sk, h, h, d, dtype, seed=15)
    bias = jax.random.normal(jax.random.PRNGKey(99), (sq, sk), dtype=dtype) * 0.1
    out = run_fwd(q, k_t, v, bias=bias)
    assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16], ids=["fp16", "bf16"])
def test_bias_bwd(dtype):
    """Attention bias backward: dbias gradient flows."""
    b, sq, sk, h, d = 2, 64, 64, 4, 64
    q, k_t, v = make_qkv(b, sq, sk, h, h, d, dtype, seed=15)
    bias = jax.random.normal(jax.random.PRNGKey(99), (sq, sk), dtype=dtype) * 0.1

    def loss(q, k, v, bias):
        return jnp.sum(run_fwd(q, k, v, bias=bias))
    dq, dk, dv, dbias = jax.grad(loss, argnums=(0, 1, 2, 3))(q, k_t, v, bias)
    assert jnp.all(jnp.isfinite(dq))
    assert jnp.all(jnp.isfinite(dk))


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("alibi_shape", ["1d", "2d"], ids=["alibi1d", "alibi2d"])
def test_alibi_fwd(dtype, alibi_shape):
    """ALiBi forward with 1D and 2D slopes."""
    b, sq, sk, h, d = 2, 128, 128, 4, 64
    q, k_t, v = make_qkv(b, sq, sk, h, h, d, dtype, seed=16)
    if alibi_shape == "1d":
        alibi = jnp.linspace(0.1, 0.5, h, dtype=jnp.float32)
    else:
        alibi = jnp.broadcast_to(jnp.linspace(0.1, 0.5, h), (b, h)).astype(jnp.float32)
    out = run_fwd(q, k_t, v, alibi_slopes=alibi)
    assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16], ids=["fp16", "bf16"])
def test_alibi_causal(dtype):
    """ALiBi with causal mask."""
    b, sq, sk, h, d = 2, 128, 128, 4, 64
    q, k_t, v = make_qkv(b, sq, sk, h, h, d, dtype, seed=16)
    alibi = jnp.linspace(0.1, 0.5, h, dtype=jnp.float32)
    out = run_fwd(q, k_t, v, alibi_slopes=alibi, causal=True)
    assert jnp.all(jnp.isfinite(out))


# ===========================================================================
# RETURN VALUES
# ===========================================================================

def test_return_lse():
    """Return log-sum-exp."""
    b, sq, sk, h, d = 2, 128, 128, 4, 64
    q, k_t, v = make_qkv(b, sq, sk, h, h, d, jnp.bfloat16, seed=12)
    result = flash_attn_func(q, k_t, v, return_lse=True)
    assert isinstance(result, tuple) and len(result) >= 2
    assert result[0].shape == (b, sq, h, d)
    assert result[1].shape == (b, h, sq)
    assert jnp.all(jnp.isfinite(result[1]))


def test_return_attn_probs_with_dropout():
    """Return attention probs (S_dmask) with dropout."""
    b, sq, sk, h, d = 2, 64, 64, 4, 64
    q, k_t, v = make_qkv(b, sq, sk, h, h, d, jnp.bfloat16, seed=12)
    result = flash_attn_func(q, k_t, v, dropout_p=0.1, return_attn_probs=True)
    assert isinstance(result, tuple) and len(result) >= 2


# ===========================================================================
# PADDED HEAD DIMENSIONS
# ===========================================================================

@pytest.mark.parametrize("d", [32, 40, 59, 64, 96, 111, 128, 160, 256],
                         ids=[f"d{d}" for d in [32, 40, 59, 64, 96, 111, 128, 160, 256]])
def test_padded_head_dim_fwd(d):
    """All head dims produce correct output shape."""
    q, k_t, v = make_qkv(2, 64, 64, 4, 4, d, jnp.bfloat16, seed=13)
    out = run_fwd(q, k_t, v)
    assert out.shape == (2, 64, 4, d)
    assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("d", [59, 111], ids=["d59", "d111"])
def test_padded_head_dim_bwd(d):
    """Non-multiple-of-8 head dims: backward produces finite gradients."""
    q, k_t, v = make_qkv(2, 64, 64, 4, 4, d, jnp.bfloat16, seed=13)
    dq, dk, dv = run_bwd(q, k_t, v)
    assert dq.shape == (2, 64, 4, d)
    assert jnp.all(jnp.isfinite(dq))


# ===========================================================================
# DETERMINISTIC
# ===========================================================================

@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16], ids=["fp16", "bf16"])
def test_deterministic_consistency(dtype):
    """Deterministic: identical results across calls."""
    q, k_t, v = make_qkv(2, 128, 128, 4, 4, 64, dtype, seed=14)
    o1 = run_fwd(q, k_t, v, deterministic=True)
    o2 = run_fwd(q, k_t, v, deterministic=True)
    assert jnp.allclose(o1, o2, atol=0)


def test_deterministic_bwd():
    """Deterministic backward: identical gradients across calls."""
    q, k_t, v = make_qkv(2, 64, 64, 4, 4, 64, jnp.bfloat16, seed=14)
    dq1, _, _ = run_bwd(q, k_t, v, deterministic=True)
    dq2, _, _ = run_bwd(q, k_t, v, deterministic=True)
    assert jnp.allclose(dq1, dq2, atol=0)


# ===========================================================================
# VARLEN TESTS
# ===========================================================================

@pytest.mark.parametrize("batch,max_sq,max_sk,hq,hk,d,dtype", VARLEN_CONFIGS)
@pytest.mark.parametrize("causal", [False, True], ids=["nomask", "causal"])
def test_varlen_fwd(batch, max_sq, max_sk, hq, hk, d, dtype, causal):
    """Varlen forward: shape, dtype, finite."""
    q, k_t, v, cu_sq, cu_sk, msq, msk = make_varlen(batch, max_sq, max_sk, hq, hk, d, dtype, seed=20)
    out = run_varlen_fwd(q, k_t, v, cu_sq, cu_sk, msq, msk, causal=causal)
    assert out.shape == (q.shape[0], hq, d)
    assert out.dtype == dtype
    assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("batch,max_sq,max_sk,hq,hk,d,dtype", VARLEN_CONFIGS)
def test_varlen_bwd(batch, max_sq, max_sk, hq, hk, d, dtype):
    """Varlen backward: gradient shapes and finiteness."""
    q, k_t, v, cu_sq, cu_sk, msq, msk = make_varlen(batch, max_sq, max_sk, hq, hk, d, dtype, seed=21)
    dq, dk, dv = run_varlen_bwd(q, k_t, v, cu_sq, cu_sk, msq, msk)
    assert dq.shape == q.shape
    assert dk.shape == k_t.shape
    assert dv.shape == v.shape
    assert jnp.all(jnp.isfinite(dq))
    assert jnp.all(jnp.isfinite(dk))
    assert jnp.all(jnp.isfinite(dv))


# ===========================================================================
# EDGE CASES
# ===========================================================================

def test_decode_sq1_fwd_bwd():
    """Decode: sq=1, forward + backward."""
    q, k_t, v = make_qkv(2, 1, 256, 4, 4, 64, jnp.bfloat16, seed=30)
    out = run_fwd(q, k_t, v)
    assert out.shape == (2, 1, 4, 64)
    dq, dk, dv = run_bwd(q, k_t, v)
    assert jnp.all(jnp.isfinite(dq))


def test_sq_gt_sk_nomask():
    """sq > sk without causal mask."""
    q, k_t, v = make_qkv(2, 256, 128, 4, 4, 64, jnp.bfloat16, seed=31)
    out = run_fwd(q, k_t, v)
    assert out.shape == (2, 256, 4, 64)
    dq, dk, dv = run_bwd(q, k_t, v)
    assert jnp.all(jnp.isfinite(dq))


def test_sq_gt_sk_causal():
    """sq > sk with causal mask."""
    q, k_t, v = make_qkv(2, 256, 128, 4, 4, 64, jnp.bfloat16, seed=31)
    out = run_fwd(q, k_t, v, causal=True)
    assert jnp.all(jnp.isfinite(out))
    dq, dk, dv = run_bwd(q, k_t, v, causal=True)
    assert jnp.all(jnp.isfinite(dq))


def test_large_batch():
    """Batch=16."""
    q, k_t, v = make_qkv(16, 64, 64, 4, 4, 64, jnp.bfloat16, seed=32)
    out = run_fwd(q, k_t, v, causal=True)
    assert jnp.all(jnp.isfinite(out))


def test_single_head():
    """Single head (nheads=1)."""
    q, k_t, v = make_qkv(2, 64, 64, 1, 1, 64, jnp.bfloat16, seed=33)
    out = run_fwd(q, k_t, v)
    assert out.shape == (2, 64, 1, 64)
    dq, dk, dv = run_bwd(q, k_t, v)
    assert jnp.all(jnp.isfinite(dq))


def test_many_heads():
    """Many heads (nheads=16)."""
    q, k_t, v = make_qkv(2, 64, 64, 16, 16, 64, jnp.bfloat16, seed=34)
    out = run_fwd(q, k_t, v)
    assert out.shape == (2, 64, 16, 64)


# ===========================================================================
# PADDED GROUP MODE (physical seqstart + logical cu_seqlen)
# ===========================================================================
#
# AITER's group mode takes two arrays when the packing contains padding
# (mha_fwd.h / mha_bwd.h sequence-pointer notes): seqstart_* are cumulative
# PHYSICAL offsets and cu_seqlen_* are cumulative LOGICAL lengths. Supplying
# only the physical array claims every physical token is real, so a framework
# that keeps its segments at fixed offsets would attend into the padding. TE
# passes both; these tests pin the same behaviour for jax-aiter.

def _padded_layout(rows, row_len, seg_lens):
    """Segments at fixed row offsets, each followed by padding.

    Returns ``(seqstart, cu_seqlen, real_index)`` where ``real_index`` lists the
    token positions that carry actual data.
    """
    total = rows * row_len
    seqstart, lengths, real_index = [], [], []
    for r in range(rows):
        off = r * row_len
        for ln in seg_lens[r]:
            seqstart.append(off)
            lengths.append(ln)
            real_index.extend(range(off, off + ln))
            off += ln
        # Remaining tokens in the row are padding, owned by the last segment's
        # physical span.
    seqstart.append(total)
    cu = [0]
    for ln in lengths:
        cu.append(cu[-1] + ln)
    return (
        jnp.array(seqstart, jnp.int32),
        jnp.array(cu, jnp.int32),
        jnp.array(real_index, jnp.int32),
    )


def _segment_ref(q, k, v, seqstart, cu_seqlen, scale):
    """FP32 per-segment causal attention over the logical tokens only."""
    seqstart = np.asarray(seqstart)
    cu = np.asarray(cu_seqlen)
    out = np.zeros(q.shape[:1] + q.shape[1:2] + v.shape[2:], dtype=np.float32)
    q32 = np.asarray(q, dtype=np.float32)
    k32 = np.asarray(k, dtype=np.float32)
    v32 = np.asarray(v, dtype=np.float32)
    hq, hk = q.shape[1], k.shape[1]
    ratio = hq // hk
    for i in range(len(cu) - 1):
        ln = int(cu[i + 1] - cu[i])
        if ln == 0:
            continue
        beg = int(seqstart[i])
        sl = slice(beg, beg + ln)
        for h in range(hq):
            logits = q32[sl, h] @ k32[sl, h // ratio].T * scale
            mask = np.tril(np.ones((ln, ln), dtype=bool))
            logits = np.where(mask, logits, -np.inf)
            probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs /= probs.sum(axis=-1, keepdims=True)
            out[sl, h] = probs @ v32[sl, h // ratio]
    return out


class TestPaddedGroupMode:
    """Physical/logical metadata pair for packed sequences with padding."""

    ROWS, ROW_LEN, HQ, HK, D = 2, 256, 4, 4, 64
    SEG_LENS = [[100, 56], [200]]  # each row keeps trailing padding

    def _inputs(self, seed=50):
        n = self.ROWS * self.ROW_LEN
        key = jax.random.PRNGKey(seed)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        q = jax.random.normal(k1, (n, self.HQ, self.D), jnp.float32).astype(jnp.bfloat16)
        k_t = jax.random.normal(k2, (n, self.HK, self.D), jnp.float32).astype(jnp.bfloat16)
        v = jax.random.normal(k3, (n, self.HK, self.D), jnp.float32).astype(jnp.bfloat16)
        dout = jax.random.normal(k4, (n, self.HQ, self.D), jnp.float32).astype(jnp.bfloat16)
        return q, k_t, v, dout

    def test_padding_is_not_attended(self):
        """Logical lengths must exclude the padding tail from the softmax."""
        q, k_t, v, _ = self._inputs()
        seqstart, cu, real = _padded_layout(self.ROWS, self.ROW_LEN, self.SEG_LENS)
        scale = self.D ** -0.5

        out = flash_attn_varlen(
            q, k_t, v, seqstart, seqstart,
            cu_seqlens_q_logical=cu, cu_seqlens_k_logical=cu,
            max_seqlen_q=self.ROW_LEN, max_seqlen_k=self.ROW_LEN,
            softmax_scale=scale, causal=True,
        )[0]

        ref = _segment_ref(q, k_t, v, seqstart, cu, scale)
        got = np.asarray(out.astype(jnp.float32))[np.asarray(real)]
        exp = ref[np.asarray(real)]
        rel = np.linalg.norm(got - exp) / max(np.linalg.norm(exp), 1e-30)
        assert rel < 2e-2, f"padded group-mode output rel_l2={rel:.6f}"

    def test_padded_matches_tight_packing(self):
        """Same logical tokens, tight vs padded layout: same attention result."""
        q, k_t, v, _ = self._inputs()
        seqstart, cu, real = _padded_layout(self.ROWS, self.ROW_LEN, self.SEG_LENS)
        scale = self.D ** -0.5

        padded = flash_attn_varlen(
            q, k_t, v, seqstart, seqstart,
            cu_seqlens_q_logical=cu, cu_seqlens_k_logical=cu,
            max_seqlen_q=self.ROW_LEN, max_seqlen_k=self.ROW_LEN,
            softmax_scale=scale, causal=True,
        )[0]

        # Gather the real tokens into a contiguous buffer and describe it without
        # any padding, which needs the physical array only.
        real_np = np.asarray(real)
        qt, kt, vt = q[real_np], k_t[real_np], v[real_np]
        tight = flash_attn_varlen(
            qt, kt, vt, cu, cu,
            max_seqlen_q=self.ROW_LEN, max_seqlen_k=self.ROW_LEN,
            softmax_scale=scale, causal=True,
        )[0]

        a = np.asarray(padded.astype(jnp.float32))[real_np]
        b = np.asarray(tight.astype(jnp.float32))
        rel = np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)
        assert rel < 1e-2, f"padded vs tight rel_l2={rel:.6f}"

    def test_padded_backward_is_finite_and_padding_free(self):
        """Backward runs in padded group mode and leaves padding gradients at 0."""
        q, k_t, v, dout = self._inputs(seed=51)
        seqstart, cu, real = _padded_layout(self.ROWS, self.ROW_LEN, self.SEG_LENS)
        scale = self.D ** -0.5

        def loss(q_, k_, v_):
            out = flash_attn_varlen(
                q_, k_, v_, seqstart, seqstart,
                cu_seqlens_q_logical=cu, cu_seqlens_k_logical=cu,
                max_seqlen_q=self.ROW_LEN, max_seqlen_k=self.ROW_LEN,
                softmax_scale=scale, causal=True,
            )[0]
            return jnp.sum(out.astype(jnp.float32) * dout.astype(jnp.float32))

        dq, dk, dv = jax.grad(loss, argnums=(0, 1, 2))(q, k_t, v)
        for name, g in (("dq", dq), ("dk", dk), ("dv", dv)):
            assert jnp.all(jnp.isfinite(g)), f"{name} not finite"

        # Padding rows contribute to no segment, so their gradient stays zero.
        n = self.ROWS * self.ROW_LEN
        pad = np.setdiff1d(np.arange(n), np.asarray(real))
        assert pad.size > 0
        for name, g in (("dq", dq), ("dk", dk), ("dv", dv)):
            pad_max = float(jnp.max(jnp.abs(g.astype(jnp.float32)[pad])))
            assert pad_max == 0.0, f"{name} nonzero on padding: {pad_max}"


# ===========================================================================
# FUSED GQA dK/dV REDUCTION
# ===========================================================================

class TestFusedGqaReduction:
    """One-pass dK+dV reduction must match the prior two-launch path."""

    @staticmethod
    def _inputs(seed=70):
        n, hq, hk, d = 512, 8, 2, 64
        key = jax.random.PRNGKey(seed)
        kq, kk, kv, kd = jax.random.split(key, 4)
        q = jax.random.normal(kq, (n, hq, d), jnp.float32).astype(jnp.bfloat16)
        k_t = jax.random.normal(kk, (n, hk, d), jnp.float32).astype(jnp.bfloat16)
        v = jax.random.normal(kv, (n, hk, d), jnp.float32).astype(jnp.bfloat16)
        dout = jax.random.normal(kd, (n, hq, d), jnp.float32).astype(jnp.bfloat16)
        return q, k_t, v, dout

    @staticmethod
    def _grad_fn(seqstart, cu_logical=None):
        d = 64

        def loss(q, k_t, v, dout):
            out = flash_attn_varlen(
                q, k_t, v, seqstart, seqstart,
                cu_seqlens_q_logical=cu_logical,
                cu_seqlens_k_logical=cu_logical,
                max_seqlen_q=256, max_seqlen_k=256,
                softmax_scale=d ** -0.5, causal=True,
            )[0]
            return jnp.sum(out.astype(jnp.float32) * dout.astype(jnp.float32))

        return jax.jit(jax.grad(loss, argnums=(0, 1, 2)))

    def test_fused_pair_is_bitwise_equal_to_separate_reductions(self, monkeypatch):
        q, k_t, v, dout = self._inputs()
        cu = jnp.array([0, 256, 512], jnp.int32)
        grad = self._grad_fn(cu)

        monkeypatch.setenv("JA_MHA_FUSE_GQA_REDUCE", "0")
        separate = jax.block_until_ready(grad(q, k_t, v, dout))
        monkeypatch.setenv("JA_MHA_FUSE_GQA_REDUCE", "1")
        fused = jax.block_until_ready(grad(q, k_t, v, dout))

        for name, actual, expected in zip(("dq", "dk", "dv"), fused, separate):
            np.testing.assert_array_equal(
                np.asarray(actual), np.asarray(expected),
                err_msg=f"{name}: fused reduction differs from separate path",
            )

    def test_fused_pair_preserves_zero_padding_gradients(self, monkeypatch):
        q, k_t, v, dout = self._inputs(seed=71)
        # Two physical rows of 256 tokens, with logical lengths 128 and 192.
        seqstart = jnp.array([0, 256, 512], jnp.int32)
        cu_logical = jnp.array([0, 128, 320], jnp.int32)
        grad = self._grad_fn(seqstart, cu_logical)
        monkeypatch.setenv("JA_MHA_FUSE_GQA_REDUCE", "1")
        dq, dk, dv = jax.block_until_ready(grad(q, k_t, v, dout))

        pad = np.concatenate([np.arange(128, 256), np.arange(448, 512)])
        for name, g in (("dq", dq), ("dk", dk), ("dv", dv)):
            assert jnp.all(jnp.isfinite(g)), f"{name} not finite"
            pad_max = float(jnp.max(jnp.abs(g.astype(jnp.float32)[pad])))
            assert pad_max == 0.0, f"{name} nonzero on padding: {pad_max}"


# ===========================================================================
# REMAT TAGGING
# ===========================================================================
#
# MaxText's `minimal_with_context` / `minimal_flash` /
# `minimal_flash_save_fp4col` policies save the checkpoint name "context", which
# is what TE tags its attention output, LSE and RNG state with. We tag the same
# values so the two backends describe their residuals identically to the same
# policy.
#
# The tag is load-bearing, not cosmetic. When the residual is saveable the
# gradient contains one attention forward; when it is not, the forward is traced
# a second time. So `JA_MHA_REMAT_CONTEXT=1` is worth about one attention forward
# per layer under MaxText's policy, and these cases assert that difference rather
# than a fixed count.

class TestRematContextTag:
    """`JA_MHA_REMAT_CONTEXT` decides whether the attention forward is recomputed."""

    @staticmethod
    def _forward_calls(policy):
        """Attention forwards traced into the gradient under `policy`."""
        n, hq, hk, d = 256, 4, 4, 64
        key = jax.random.PRNGKey(60)
        k1, k2, k3 = jax.random.split(key, 3)
        q = jax.random.normal(k1, (n, hq, d), jnp.float32).astype(jnp.bfloat16)
        k_t = jax.random.normal(k2, (n, hk, d), jnp.float32).astype(jnp.bfloat16)
        v = jax.random.normal(k3, (n, hk, d), jnp.float32).astype(jnp.bfloat16)
        cu = jnp.array([0, 128, n], jnp.int32)

        def f(q_, k_, v_):
            out = flash_attn_varlen(
                q_, k_, v_, cu, cu,
                max_seqlen_q=128, max_seqlen_k=128,
                softmax_scale=d ** -0.5, causal=True,
            )[0]
            return jnp.sum(out.astype(jnp.float32))

        wrapped = jax.checkpoint(f, policy=policy)
        jaxpr = str(jax.make_jaxpr(jax.grad(wrapped, argnums=(0, 1, 2)))(q, k_t, v))
        return jaxpr.count("MhaFwdUnifiedJA"), jaxpr.count("MhaBwdUnifiedJA")

    def test_tag_saves_the_forward_under_maxtext_policy(self, monkeypatch):
        monkeypatch.setenv("JA_MHA_REMAT_CONTEXT", "1")
        policy = jax.checkpoint_policies.save_only_these_names("context")
        assert self._forward_calls(policy) == (1, 1)

    def test_without_the_tag_the_policy_matches_nothing_and_the_forward_repeats(
        self, monkeypatch
    ):
        """This is the whole point of the tag: no name to match, no saved residual."""
        monkeypatch.setenv("JA_MHA_REMAT_CONTEXT", "0")
        policy = jax.checkpoint_policies.save_only_these_names("context")
        assert self._forward_calls(policy) == (2, 1)

    def test_nothing_saveable_recomputes_the_forward_even_when_tagged(self, monkeypatch):
        """A custom_vjp forward IS re-traced when the policy saves nothing."""
        monkeypatch.setenv("JA_MHA_REMAT_CONTEXT", "1")
        assert self._forward_calls(jax.checkpoint_policies.nothing_saveable) == (2, 1)

    def test_everything_saveable_keeps_a_single_forward(self, monkeypatch):
        monkeypatch.setenv("JA_MHA_REMAT_CONTEXT", "1")
        assert self._forward_calls(jax.checkpoint_policies.everything_saveable) == (1, 1)


# ===========================================================================
# REGRESSION GUARDS — every historical bug from AITER bump
# ===========================================================================

class TestRegressions:
    """Configs that caught historical bugs. Must never regress."""

    def test_v3_bwd_sq_gt_sk_causal(self):
        """c5bc2e2: ASM v3 bwd wrong gradients for causal sq > sk on gfx950."""
        for d in [96, 128]:
            q, k_t, v = make_qkv(2, 128, 64, 4, 4, d, jnp.bfloat16, seed=40)
            dq, dk, dv = run_bwd(q, k_t, v, causal=True)
            assert jnp.all(jnp.isfinite(dq)), f"d={d}: dq NaN"

    @pytest.mark.parametrize("d", [96, 111, 128], ids=["d96", "d111", "d128"])
    @pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16], ids=["fp16", "bf16"])
    def test_1024_1023_causal(self, d, dtype):
        """c5bc2e2: seqlen=(1024,1023) causal d>=96 on gfx950."""
        q, k_t, v = make_qkv(2, 1024, 1023, 4, 4, d, dtype, seed=41)
        dq, dk, dv = run_bwd(q, k_t, v, causal=True)
        assert jnp.all(jnp.isfinite(dq)), f"d={d}: dq NaN"

    def test_mqa_gqa_bwd_routing(self):
        """e53633c: MQA/GQA backward routing (nhead_q != nhead_k guard)."""
        for hq, hk in [(6, 3), (6, 1), (8, 2)]:
            q, k_t, v = make_qkv(2, 128, 128, hq, hk, 64, jnp.bfloat16, seed=42)
            dq, dk, dv = run_bwd(q, k_t, v, causal=True)
            assert dq.shape == (2, 128, hq, 64)
            assert dk.shape == (2, 128, hk, 64)
            assert jnp.all(jnp.isfinite(dq)), f"hq={hq},hk={hk}: dq NaN"
            assert jnp.all(jnp.isfinite(dk))
            assert jnp.all(jnp.isfinite(dv))

    @pytest.mark.parametrize("d", [
        pytest.param(96, marks=pytest.mark.xfail(reason="gfx950 CK varlen bwd causal max_sk>256 kernel issue")),
        pytest.param(128, marks=pytest.mark.xfail(reason="gfx950 CK varlen bwd causal max_sk>256 kernel issue")),
    ], ids=["d96", "d128"])
    def test_varlen_large_sk_causal(self, d):
        """c5bc2e2: varlen max_sk>256 causal d>=96 on gfx950."""
        q, k_t, v, cu_sq, cu_sk, msq, msk = make_varlen(4, 512, 512, 4, 4, d, jnp.bfloat16, seed=43)
        dq, dk, dv = run_varlen_bwd(q, k_t, v, cu_sq, cu_sk, msq, msk, causal=True)
        assert jnp.all(jnp.isfinite(dq)), f"d={d}: varlen dq NaN"

    def test_gfx950_1block_override(self):
        """is_950_1block: sk<=256 with hd in (64,128] forces deterministic=False."""
        q, k_t, v = make_qkv(2, 128, 128, 4, 4, 96, jnp.bfloat16, seed=44)
        dq, dk, dv = run_bwd(q, k_t, v, deterministic=True)
        assert jnp.all(jnp.isfinite(dq))

    def test_swa_not_v3_bwd(self):
        """SWA excluded from ASM v3 backward (wrong gradients on gfx950)."""
        q, k_t, v = make_qkv(2, 128, 128, 4, 4, 128, jnp.bfloat16, seed=45)
        dq, dk, dv = run_bwd(q, k_t, v, causal=True, window_size=(32, 0))
        assert jnp.all(jnp.isfinite(dq))

    @pytest.mark.parametrize("d", [32, 64, 96, 128], ids=["d32", "d64", "d96", "d128"])
    def test_all_head_dims_bwd(self, d):
        """All common head dims produce finite backward gradients."""
        q, k_t, v = make_qkv(2, 64, 64, 4, 4, d, jnp.bfloat16, seed=46)
        dq, dk, dv = run_bwd(q, k_t, v)
        assert jnp.all(jnp.isfinite(dq)), f"d={d}: dq NaN"
        assert jnp.all(jnp.isfinite(dk))
        assert jnp.all(jnp.isfinite(dv))
