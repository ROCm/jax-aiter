# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for AITER Triton varlen MHA through TritonDispatchJA."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_aiter.ja_compat import config as ja_config


def _bridge_present() -> bool:
    try:
        return (ja_config.get_jax_aiter_lib_dir() / "triton_bridge_ja.so").is_file()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _bridge_present(),
    reason="triton_bridge_ja.so missing; run make -f Makefile.triton bridge",
)


def _cu_seqlens(lengths):
    offsets = np.zeros((len(lengths) + 1,), dtype=np.int32)
    np.cumsum(lengths, out=offsets[1:])
    return jnp.asarray(offsets)


def _oracle(q, k, v, cu, scale):
    """FP32 packed causal attention with optional QK PE dims."""
    q = np.asarray(q, np.float32)
    k = np.asarray(k, np.float32)
    v = np.asarray(v, np.float32)
    cu = np.asarray(cu)
    total, hq, d_qk = q.shape
    hk = k.shape[1]
    dv = v.shape[-1]
    out = np.zeros((total, hq, dv), np.float32)
    group = hq // hk
    for seq in range(len(cu) - 1):
        start, end = int(cu[seq]), int(cu[seq + 1])
        qs = q[start:end]
        ks = k[start:end]
        vs = v[start:end]
        sl = end - start
        q_nope, q_pe = qs[..., :dv], qs[..., dv:]
        k_nope, k_pe = ks[..., :dv], ks[..., dv:]
        for h in range(hq):
            hk_i = h // group
            scores = np.einsum("td,sd->ts", q_nope[:, h], k_nope[:, hk_i])
            if q_pe.shape[-1]:
                scores = scores + np.einsum(
                    "td,sd->ts", q_pe[:, h], k_pe[:, hk_i]
                )
            scores = scores * scale
            mask = np.triu(np.ones((sl, sl), np.bool_), 1)
            scores = np.where(mask, np.finfo(np.float32).min / 4, scores)
            weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
            weights = weights / np.clip(weights.sum(axis=-1, keepdims=True), 1e-6, None)
            out[start:end, h] = weights @ vs[:, hk_i]
    return out


def test_hlo_uses_triton_dispatch():
    from jax_aiter.triton.attention.varlen import flash_attn_varlen_triton

    lengths = (8, 8)
    total = sum(lengths)
    cu = _cu_seqlens(lengths)
    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.normal(size=(total, 2, 128)).astype(np.float32)).astype(
        jnp.bfloat16
    )
    k = jnp.asarray(rng.normal(size=(total, 1, 128)).astype(np.float32)).astype(
        jnp.bfloat16
    )
    v = k

    def fn(q, k, v, cu):
        return flash_attn_varlen_triton(q, k, v, cu, cu, 8, 8, 0.0, 1.0, True)

    lowered = jax.jit(fn).lower(q, k, v, cu)
    text = str(lowered.compiler_ir())
    assert "TritonDispatchJA" in text
    assert "MhaBwdUnifiedJA" not in text


def test_remat_context_saves_forward_residuals():
    from jax_aiter.triton.attention.varlen import flash_attn_varlen_triton

    cu = _cu_seqlens((8, 8))
    q = jnp.ones((16, 2, 128), jnp.bfloat16)
    k = jnp.ones((16, 2, 128), jnp.bfloat16)
    v = jnp.ones((16, 2, 128), jnp.bfloat16)

    def loss(q_, k_, v_):
        out = flash_attn_varlen_triton(q_, k_, v_, cu, cu, 8, 8, 0.0, 1.0, True)
        return jnp.sum(out.astype(jnp.float32))

    wrapped = jax.checkpoint(
        loss, policy=jax.checkpoint_policies.save_only_these_names("context")
    )
    hlo = (
        jax.jit(jax.grad(wrapped, argnums=(0, 1, 2)))
        .lower(q, k, v)
        .compile()
        .as_text()
    )
    # One forward, one backward preprocess, one one-kernel backward.
    assert hlo.count('custom_call_target="TritonDispatchJA"') == 3


def test_varlen_equal_head_vs_oracle():
    from jax_aiter.triton.attention.varlen import flash_attn_varlen_triton

    lengths = (16, 16)
    total = sum(lengths)
    cu = _cu_seqlens(lengths)
    rng = np.random.default_rng(1)
    q = rng.normal(size=(total, 4, 128)).astype(np.float32)
    k = rng.normal(size=(total, 2, 128)).astype(np.float32)
    v = rng.normal(size=(total, 2, 128)).astype(np.float32)
    scale = 1.0 / np.sqrt(128)
    out = flash_attn_varlen_triton(
        jnp.asarray(q).astype(jnp.bfloat16),
        jnp.asarray(k).astype(jnp.bfloat16),
        jnp.asarray(v).astype(jnp.bfloat16),
        cu,
        cu,
        16,
        16,
        0.0,
        float(scale),
        True,
    )
    got = np.asarray(out, np.float32)
    ref = _oracle(q, k, v, cu, scale)
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    assert rel < 0.08, rel


def test_varlen_qk192_v128_vs_oracle():
    from jax_aiter.triton.attention.varlen import flash_attn_varlen_triton

    lengths = (16, 16)
    total = sum(lengths)
    cu = _cu_seqlens(lengths)
    rng = np.random.default_rng(2)
    q = rng.normal(size=(total, 2, 192)).astype(np.float32)
    k = rng.normal(size=(total, 1, 192)).astype(np.float32)
    v = rng.normal(size=(total, 1, 128)).astype(np.float32)
    scale = 1.0 / np.sqrt(192)
    out = flash_attn_varlen_triton(
        jnp.asarray(q).astype(jnp.bfloat16),
        jnp.asarray(k).astype(jnp.bfloat16),
        jnp.asarray(v).astype(jnp.bfloat16),
        cu,
        cu,
        16,
        16,
        0.0,
        float(scale),
        True,
    )
    got = np.asarray(out, np.float32)
    ref = _oracle(q, k, v, cu, scale)
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    assert rel < 0.08, rel


def test_varlen_backward_finite():
    from jax_aiter.triton.attention.varlen import flash_attn_varlen_triton

    lengths = (16, 16)
    total = sum(lengths)
    cu = _cu_seqlens(lengths)
    rng = np.random.default_rng(3)

    def loss(q, k, v):
        out = flash_attn_varlen_triton(
            q, k, v, cu, cu, 16, 16, 0.0, 1.0 / np.sqrt(192), True
        )
        return jnp.square(out.astype(jnp.float32)).mean()

    q = jnp.asarray(rng.normal(size=(total, 2, 192)).astype(np.float32)).astype(
        jnp.bfloat16
    )
    k = jnp.asarray(rng.normal(size=(total, 1, 192)).astype(np.float32)).astype(
        jnp.bfloat16
    )
    v = jnp.asarray(rng.normal(size=(total, 1, 128)).astype(np.float32)).astype(
        jnp.bfloat16
    )
    dq, dk, dv = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)
    assert np.isfinite(np.asarray(dq, np.float32)).all()
    assert np.isfinite(np.asarray(dk, np.float32)).all()
    assert np.isfinite(np.asarray(dv, np.float32)).all()


def test_equal_head_matches_existing_mha_if_present():
    try:
        from jax_aiter.mha import flash_attn_varlen_raw
    except Exception:
        pytest.skip("CK/ASM MHA libraries are not present")

    from jax_aiter.triton.attention.varlen import flash_attn_varlen_triton

    lengths = (16, 16)
    total = sum(lengths)
    cu = _cu_seqlens(lengths)
    rng = np.random.default_rng(4)
    q = jnp.asarray(rng.normal(size=(total, 4, 128)).astype(np.float32)).astype(
        jnp.bfloat16
    )
    k = jnp.asarray(rng.normal(size=(total, 2, 128)).astype(np.float32)).astype(
        jnp.bfloat16
    )
    v = jnp.asarray(rng.normal(size=(total, 2, 128)).astype(np.float32)).astype(
        jnp.bfloat16
    )
    scale = 1.0 / np.sqrt(128)
    tri = flash_attn_varlen_triton(q, k, v, cu, cu, 16, 16, 0.0, scale, True)
    ref = flash_attn_varlen_raw(
        q, k, v, cu, cu, None, None, 16, 16, 0.0, scale, True, (-1, -1)
    )
    rel = float(
        np.linalg.norm(np.asarray(tri, np.float32) - np.asarray(ref, np.float32))
        / np.linalg.norm(np.asarray(ref, np.float32))
    )
    assert rel < 0.08, rel
