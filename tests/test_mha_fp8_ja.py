# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""FP8 MHA tests for unified MHA handlers (mha_v2).

gfx950 ASM v3 kernels: head dims 128 and 256.
"""

import math

import pytest
import jax
import jax.numpy as jnp

from jax_aiter.mha import flash_attn_func, flash_attn_varlen
from jax_aiter.ja_compat.dtypes import get_dtype_fp8


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

def _repeat_kv(x, n_rep):
    """Expand [b, s, hk, d] to [b, s, hq, d] for GQA/MQA (hq = hk * n_rep)."""
    if n_rep == 1:
        return x
    return jnp.repeat(x, n_rep, axis=2)


def attention_ref(q, k, v, causal=False, scale=None, window_size=(-1, -1)):
    q, k, v = q.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32)
    hq, hk = q.shape[2], k.shape[2]
    assert hq % hk == 0, f"query heads {hq} must be a multiple of kv heads {hk}"
    n_rep = hq // hk
    k = _repeat_kv(k, n_rep)
    v = _repeat_kv(v, n_rep)
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


def run_fwd(q, k, v, **kw):
    result = flash_attn_func(q, k, v, **kw)
    return result[0] if isinstance(result, tuple) else result


# ---------------------------------------------------------------------------
# FP8 helpers
# ---------------------------------------------------------------------------

def _fp8_dtype():
    """Returns the FP8 dtype for the current GPU (e4m3fn on gfx950, e4m3fnuz on gfx942)."""
    return get_dtype_fp8()


def _make_fp8_qkv(b, sq, sk, hq, hk, d, seed=0):
    """Make quantised FP8 q/k/v with per-tensor descales.

    We generate float32 data, clamp to a sane range, cast to FP8, and set
    the per-tensor descale so the effective fp32 value is reconstructed as
    fp8_val * descale.  Using scale=1.0 keeps the math simple: the FP8
    attention output should match a bf16 reference run on the same values.
    """
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    fp8 = _fp8_dtype()
    # Small values → no saturation in FP8
    q32 = jax.random.normal(k1, (b, sq, hq, d)) * 0.1
    k32 = jax.random.normal(k2, (b, sk, hk, d)) * 0.1
    v32 = jax.random.normal(k3, (b, sk, hk, d)) * 0.1
    q_fp8 = q32.astype(fp8)
    k_fp8 = k32.astype(fp8)
    v_fp8 = v32.astype(fp8)
    descale = jnp.ones((1,), dtype=jnp.float32)
    return q_fp8, k_fp8, v_fp8, descale, descale, descale


def _make_varlen_fp8(batch, max_sq, max_sk, hq, hk, d, seed=0):
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    fp8 = _fp8_dtype()
    sq = jax.random.randint(k4, (batch,), 1, max_sq + 1)
    sk = jax.random.randint(k5, (batch,), 1, max_sk + 1)
    tq, tk = int(jnp.sum(sq)), int(jnp.sum(sk))
    cu_sq = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(sq).astype(jnp.int32)])
    cu_sk = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(sk).astype(jnp.int32)])
    q32 = jax.random.normal(k1, (tq, hq, d)) * 0.1
    k32 = jax.random.normal(k2, (tk, hk, d)) * 0.1
    v32 = jax.random.normal(k3, (tk, hk, d)) * 0.1
    q_fp8 = q32.astype(fp8)
    k_fp8 = k32.astype(fp8)
    v_fp8 = v32.astype(fp8)
    descale = jnp.ones((1,), dtype=jnp.float32)
    return q_fp8, k_fp8, v_fp8, cu_sq, cu_sk, max_sq, max_sk, descale


# FP8 batch configs: only hdim 128 and 256 are supported by ASM v3 FP8 kernels.
# GQA ratio must be power-of-2 (1, 2, 4, 8, 16).
FP8_BATCH_CONFIGS = [
    # (b, sq, sk, hq, hk, d)
    pytest.param(2, 128, 128, 4, 4, 128, id="fp8_d128_mha"),
    pytest.param(2, 256, 256, 4, 4, 128, id="fp8_d128_sq256"),
    pytest.param(2, 512, 512, 4, 4, 128, id="fp8_d128_sq512"),
    pytest.param(2, 128, 128, 8, 4, 128, id="fp8_d128_gqa2"),
    pytest.param(2, 128, 128, 8, 2, 128, id="fp8_d128_gqa4"),
    pytest.param(2, 128, 128, 4, 1, 128, id="fp8_d128_mqa"),
    pytest.param(2, 128, 128, 4, 4, 256, id="fp8_d256_mha"),
    pytest.param(2, 256, 256, 4, 4, 256, id="fp8_d256_sq256"),
    pytest.param(2, 512, 512, 4, 4, 256, id="fp8_d256_sq512"),
    pytest.param(1, 1024, 1024, 4, 4, 256, id="fp8_d256_sq1024"),
    pytest.param(2, 128, 128, 8, 4, 256, id="fp8_d256_gqa2"),
    pytest.param(2, 128, 128, 8, 2, 256, id="fp8_d256_gqa4"),
    pytest.param(2, 128, 128, 4, 1, 256, id="fp8_d256_mqa"),
]

FP8_VARLEN_CONFIGS = [
    # (batch, max_sq, max_sk, hq, hk, d)
    pytest.param(4, 128, 128, 4, 4, 128, id="fp8_vl_d128_mha"),
    pytest.param(4, 256, 256, 4, 4, 128, id="fp8_vl_d128_sq256"),
    pytest.param(4, 128, 128, 8, 2, 128, id="fp8_vl_d128_gqa4"),
    pytest.param(4, 128, 128, 4, 4, 256, id="fp8_vl_d256_mha"),
    pytest.param(4, 256, 256, 4, 4, 256, id="fp8_vl_d256_sq256"),
    pytest.param(4, 128, 128, 8, 2, 256, id="fp8_vl_d256_gqa4"),
]


@pytest.mark.parametrize("b,sq,sk,hq,hk,d", FP8_BATCH_CONFIGS)
@pytest.mark.parametrize("causal", [False, True], ids=["nomask", "causal"])
def test_fp8_batch_fwd_shape(b, sq, sk, hq, hk, d, causal):
    """FP8 batch forward: output is bf16, correct shape, finite values."""
    q, k_t, v, qd, kd, vd = _make_fp8_qkv(b, sq, sk, hq, hk, d, seed=50)
    out = run_fwd(q, k_t, v, causal=causal,
                  q_descale=qd, k_descale=kd, v_descale=vd)
    assert out.shape == (b, sq, hq, d), f"shape mismatch: {out.shape}"
    assert out.dtype == jnp.bfloat16, f"expected bf16 output, got {out.dtype}"
    assert jnp.all(jnp.isfinite(out)), "NaN/Inf in FP8 output"


@pytest.mark.parametrize("b,sq,sk,hq,hk,d", FP8_BATCH_CONFIGS)
def test_fp8_batch_fwd_accuracy(b, sq, sk, hq, hk, d):
    """FP8 batch forward: output close to bf16 reference (scale=1, small values).

    Because all descales are 1.0 and values are small (no FP8 saturation),
    the FP8 kernel output should match bf16 attention within bf16 tolerance.
    """
    q, k_t, v, qd, kd, vd = _make_fp8_qkv(b, sq, sk, hq, hk, d, seed=51)
    scale = d ** (-0.5)
    out = run_fwd(q, k_t, v, softmax_scale=scale,
                  q_descale=qd, k_descale=kd, v_descale=vd)
    # Reference: same data in bf16
    q_ref = q.astype(jnp.bfloat16)
    k_ref = k_t.astype(jnp.bfloat16)
    v_ref = v.astype(jnp.bfloat16)
    ref = attention_ref(q_ref, k_ref, v_ref, scale=scale).astype(jnp.bfloat16)
    assert_close(out, ref, jnp.bfloat16, f"fp8_fwd_d{d}", bwd_factor=4)


@pytest.mark.parametrize("d", [128, 256], ids=["d128", "d256"])
@pytest.mark.parametrize("causal", [False, True], ids=["nomask", "causal"])
def test_fp8_batch_fwd_per_head_descale(d, causal):
    """FP8 forward with per-head descale tensors [batch, nheads_k]."""
    b, sq, sk, hq, hk = 2, 128, 128, 4, 4
    fp8 = _fp8_dtype()
    key = jax.random.PRNGKey(60)
    k1, k2, k3 = jax.random.split(key, 3)
    q32 = jax.random.normal(k1, (b, sq, hq, d)) * 0.1
    k32 = jax.random.normal(k2, (b, sk, hk, d)) * 0.1
    v32 = jax.random.normal(k3, (b, sk, hk, d)) * 0.1
    q_fp8 = q32.astype(fp8)
    k_fp8 = k32.astype(fp8)
    v_fp8 = v32.astype(fp8)
    # Per-head descales [b, hk]
    descale = jnp.ones((b, hk), dtype=jnp.float32)
    out = run_fwd(q_fp8, k_fp8, v_fp8, causal=causal,
                  q_descale=descale, k_descale=descale, v_descale=descale)
    assert out.shape == (b, sq, hq, d)
    assert out.dtype == jnp.bfloat16
    assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("d", [128, 256], ids=["d128", "d256"])
def test_fp8_batch_fwd_return_lse(d):
    """FP8 forward: return_lse gives finite fp32 LSE values."""
    b, sq, sk, hq, hk = 2, 128, 128, 4, 4
    q, k_t, v, qd, kd, vd = _make_fp8_qkv(b, sq, sk, hq, hk, d, seed=61)
    result = flash_attn_func(q, k_t, v, return_lse=True,
                             q_descale=qd, k_descale=kd, v_descale=vd)
    assert isinstance(result, tuple) and len(result) >= 2
    out, lse = result[0], result[1]
    assert out.shape == (b, sq, hq, d)
    assert out.dtype == jnp.bfloat16
    assert lse.shape == (b, hq, sq)
    assert lse.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(lse))


@pytest.mark.parametrize("b,sq,sk,hq,hk,d", FP8_BATCH_CONFIGS)
@pytest.mark.parametrize("causal", [False, True], ids=["nomask", "causal"])
def test_fp8_varlen_fwd_shape(b, sq, sk, hq, hk, d, causal):
    """FP8 varlen forward: output is bf16, correct shape, finite values."""
    q, k_t, v, cu_sq, cu_sk, msq, msk, desc = _make_varlen_fp8(
        b, sq, sk, hq, hk, d, seed=70)
    result = flash_attn_varlen(q, k_t, v, cu_sq, cu_sk,
                               max_seqlen_q=msq, max_seqlen_k=msk,
                               causal=causal,
                               q_descale=desc, k_descale=desc, v_descale=desc)
    out = result[0] if isinstance(result, tuple) else result
    assert out.shape == (q.shape[0], hq, d)
    assert out.dtype == jnp.bfloat16
    assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("d", [128, 256], ids=["d128", "d256"])
def test_fp8_varlen_fwd_accuracy(d):
    """FP8 varlen forward: close to bf16 reference (scale=1, small values)."""
    batch, max_sq, max_sk, hq, hk = 4, 128, 128, 4, 4
    q, k_t, v, cu_sq, cu_sk, msq, msk, desc = _make_varlen_fp8(
        batch, max_sq, max_sk, hq, hk, d, seed=71)
    scale = d ** (-0.5)
    result = flash_attn_varlen(q, k_t, v, cu_sq, cu_sk,
                               max_seqlen_q=msq, max_seqlen_k=msk,
                               softmax_scale=scale,
                               q_descale=desc, k_descale=desc, v_descale=desc)
    out_fp8 = result[0] if isinstance(result, tuple) else result

    # Build per-sample bf16 references and compare packed.
    out_ref_list = []
    for i in range(batch):
        qi = q[int(cu_sq[i]):int(cu_sq[i + 1])].astype(jnp.bfloat16)  # [sq_i, hq, d]
        ki = k_t[int(cu_sk[i]):int(cu_sk[i + 1])].astype(jnp.bfloat16)
        vi = v[int(cu_sk[i]):int(cu_sk[i + 1])].astype(jnp.bfloat16)
        # attention_ref expects [b, s, h, d]
        ref_i = attention_ref(qi[None], ki[None], vi[None], scale=scale)
        out_ref_list.append(ref_i[0])  # [sq_i, hq, d]
    out_ref = jnp.concatenate(out_ref_list, axis=0).astype(jnp.bfloat16)
    assert_close(out_fp8, out_ref, jnp.bfloat16, f"fp8_varlen_d{d}", bwd_factor=4)


@pytest.mark.parametrize("d", [128, 256], ids=["d128", "d256"])
def test_fp8_varlen_fwd_return_lse(d):
    """FP8 varlen forward: return_lse gives finite fp32 LSE values."""
    batch, max_sq, max_sk, hq, hk = 4, 128, 128, 4, 4
    q, k_t, v, cu_sq, cu_sk, msq, msk, desc = _make_varlen_fp8(
        batch, max_sq, max_sk, hq, hk, d, seed=72)
    result = flash_attn_varlen(q, k_t, v, cu_sq, cu_sk,
                               max_seqlen_q=msq, max_seqlen_k=msk,
                               return_lse=True,
                               q_descale=desc, k_descale=desc, v_descale=desc)
    assert isinstance(result, tuple) and len(result) == 2
    out, lse = result
    assert out.dtype == jnp.bfloat16
    assert lse.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(lse))


class TestFp8Regressions:
    """Guard against FP8-specific failure modes."""

    def test_fp8_missing_descale_raises(self):
        """Calling FP8 forward without descales must fail (C++ validation)."""
        fp8 = _fp8_dtype()
        q = jnp.ones((2, 128, 4, 128), dtype=fp8)
        k_t = jnp.ones((2, 128, 4, 128), dtype=fp8)
        v = jnp.ones((2, 128, 4, 128), dtype=fp8)
        with pytest.raises(Exception):
            # No descales → C++ bridge returns error → JAX raises
            _ = run_fwd(q, k_t, v)

    def test_fp8_non_power_of_2_gqa_raises(self):
        """FP8 GQA ratio 3 (non-power-of-2) must be rejected."""
        fp8 = _fp8_dtype()
        q = jnp.ones((2, 128, 6, 128), dtype=fp8)
        k_t = jnp.ones((2, 128, 2, 128), dtype=fp8)  # ratio=3, not power-of-2
        v = jnp.ones((2, 128, 2, 128), dtype=fp8)
        desc = jnp.ones((1,), dtype=jnp.float32)
        with pytest.raises(Exception):
            _ = run_fwd(q, k_t, v, q_descale=desc, k_descale=desc, v_descale=desc)

    @pytest.mark.parametrize("d", [128, 256], ids=["d128", "d256"])
    def test_fp8_output_is_bf16_not_fp8(self, d):
        """Output dtype for FP8 input must be bfloat16, never fp8."""
        q, k_t, v, qd, kd, vd = _make_fp8_qkv(2, 128, 128, 4, 4, d, seed=80)
        out = run_fwd(q, k_t, v, q_descale=qd, k_descale=kd, v_descale=vd)
        assert out.dtype == jnp.bfloat16, \
            f"Expected bfloat16 output for fp8 input, got {out.dtype}"

    @pytest.mark.parametrize("d", [128, 256], ids=["d128", "d256"])
    def test_fp8_causal_and_non_causal_differ(self, d):
        """Causal and non-causal FP8 outputs should differ for sq>1."""
        q, k_t, v, qd, kd, vd = _make_fp8_qkv(2, 128, 128, 4, 4, d, seed=81)
        out_nc = run_fwd(q, k_t, v, causal=False,
                         q_descale=qd, k_descale=kd, v_descale=vd)
        out_c = run_fwd(q, k_t, v, causal=True,
                        q_descale=qd, k_descale=kd, v_descale=vd)
        assert not jnp.allclose(out_nc, out_c, atol=1e-3), \
            "Causal and non-causal outputs should differ"

    @pytest.mark.parametrize("d", [128, 256], ids=["d128", "d256"])
    def test_fp8_batch_and_varlen_agree(self, d):
        """Batch-mode and varlen FP8 outputs agree for uniform sequence lengths."""
        b, sq, hq, hk = 2, 64, 4, 4
        fp8 = _fp8_dtype()
        key = jax.random.PRNGKey(82)
        k1, k2, k3 = jax.random.split(key, 3)
        q32 = jax.random.normal(k1, (b, sq, hq, d)) * 0.1
        k32 = jax.random.normal(k2, (b, sq, hk, d)) * 0.1
        v32 = jax.random.normal(k3, (b, sq, hk, d)) * 0.1
        q_fp8 = q32.astype(fp8)
        k_fp8 = k32.astype(fp8)
        v_fp8 = v32.astype(fp8)
        desc = jnp.ones((1,), dtype=jnp.float32)

        # Batch forward
        out_batch = run_fwd(q_fp8, k_fp8, v_fp8, causal=True,
                            q_descale=desc, k_descale=desc, v_descale=desc)

        # Varlen forward (pack [b, sq, h, d] → [b*sq, h, d])
        q_vl = q_fp8.reshape(b * sq, hq, d)
        k_vl = k_fp8.reshape(b * sq, hk, d)
        v_vl = v_fp8.reshape(b * sq, hk, d)
        seqlens = jnp.array([sq] * b, dtype=jnp.int32)
        cu_sq_vl = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(seqlens)])
        cu_sk_vl = cu_sq_vl
        result_vl = flash_attn_varlen(q_vl, k_vl, v_vl, cu_sq_vl, cu_sk_vl,
                                      max_seqlen_q=sq, max_seqlen_k=sq,
                                      causal=True,
                                      q_descale=desc, k_descale=desc, v_descale=desc)
        out_vl = result_vl[0].reshape(b, sq, hq, d)

        assert_close(out_batch, out_vl, jnp.bfloat16,
                     f"fp8_batch_vs_varlen_d{d}", bwd_factor=2)
