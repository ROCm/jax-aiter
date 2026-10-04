# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Tests for per-direction MHA backend selection (jax_aiter.mha.select)."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import jax
import jax.numpy as jnp

select = pytest.importorskip("jax_aiter.mha.select", reason="AITER MHA libraries missing")

MLA = dict(hd_qk=192, hd_v=128)
POLICY_OK = dict(q_dtype=jnp.bfloat16, causal=True, dropout_p=0.0, window_size=(-1, -1))


def _layout(row_len, rows, slots):
    """AITER physical/logical metadata for rows of real segments plus tail padding.

    ``rows`` lists each row's segment lengths. ``slots`` is the fixed slot count;
    unused slots are empty and start at ``total``, as MaxText builds them.
    """
    starts, lengths = [], []
    for r, segs in enumerate(rows):
        offset = r * row_len
        for length in segs:
            starts.append(offset)
            lengths.append(length)
            offset += length
        assert offset <= (r + 1) * row_len
    total = row_len * len(rows)
    starts += [total] * (slots - len(starts))
    lengths += [0] * (slots - len(lengths))
    seqstart = np.asarray(starts + [total], np.int32)
    cu_logical = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    seg_id = np.zeros(total, np.int32)
    for i, (s, n) in enumerate(zip(starts, lengths)):
        seg_id[s:s + n] = i + 1
    return seqstart, cu_logical, seg_id, total


# ---------------------------------------------------------------------------
# Layout mapping and policy (no kernels)
# ---------------------------------------------------------------------------

def test_layout_tight_is_passthrough():
    seqstart = jnp.asarray([0, 3, 8], jnp.int32)
    cu, valid = select.triton_varlen_layout(seqstart, None, 8)
    np.testing.assert_array_equal(np.asarray(cu), [0, 3, 8])
    assert valid is None


def test_layout_marks_holes_and_leading_gap():
    # Physical [pad pad | a a a | pad | b b | pad pad], plus one unused slot.
    seqstart = jnp.asarray([2, 6, 10, 10], jnp.int32)
    cu_logical = jnp.asarray([0, 3, 5, 5], jnp.int32)
    cu, valid = select.triton_varlen_layout(seqstart, cu_logical, 10)
    np.testing.assert_array_equal(np.asarray(cu), [2, 5, 6, 8, 10, 10, 10])
    np.testing.assert_array_equal(
        np.asarray(valid), [0, 0, 1, 1, 1, 0, 1, 1, 0, 0]
    )


@pytest.mark.parametrize("heads", [MLA, dict(hd_qk=128, hd_v=128)])
@pytest.mark.parametrize("env", [None, "", "auto"], ids=["unset", "empty", "auto"])
def test_policy_defaults_to_aiter(monkeypatch, heads, env):
    if env is None:
        monkeypatch.delenv("JA_MHA_VARLEN_BWD_BACKEND", raising=False)
    else:
        monkeypatch.setenv("JA_MHA_VARLEN_BWD_BACKEND", env)
    assert select.varlen_bwd_backend(**POLICY_OK, **heads) == "aiter"


def test_policy_forced_backends(monkeypatch):
    monkeypatch.setenv("JA_MHA_VARLEN_BWD_BACKEND", "triton")
    assert select.varlen_bwd_backend(**POLICY_OK, **MLA) == "triton"
    monkeypatch.setenv("JA_MHA_VARLEN_BWD_BACKEND", "aiter")
    assert select.varlen_bwd_backend(**POLICY_OK, **MLA) == "aiter"


@pytest.mark.parametrize(
    "override",
    [
        dict(causal=False),
        dict(dropout_p=0.1),
        dict(window_size=(128, 0)),
        dict(q_dtype=jnp.float32),
        dict(hd_qk=128, hd_v=192),
        dict(hd_qk=160, hd_v=96),
    ],
    ids=["noncausal", "dropout", "window", "fp32", "v-wider", "non-pow2-split"],
)
def test_forced_triton_rejects_unsupported(monkeypatch, override):
    monkeypatch.setenv("JA_MHA_VARLEN_BWD_BACKEND", "triton")
    with pytest.raises(ValueError, match="cannot serve"):
        select.varlen_bwd_backend(**{**POLICY_OK, **MLA, **override})


def test_full_rows_use_batch_unless_triton_backward_is_forced(monkeypatch):
    monkeypatch.delenv("JA_MHA_VARLEN_BWD_BACKEND", raising=False)
    assert select.full_row_layout(**POLICY_OK, **MLA) == "batch"
    monkeypatch.setenv("JA_MHA_VARLEN_BWD_BACKEND", "triton")
    assert select.full_row_layout(**POLICY_OK, **MLA) == "varlen"


def test_policy_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("JA_MHA_VARLEN_BWD_BACKEND", "ck")
    with pytest.raises(ValueError, match="expected one of"):
        select.varlen_bwd_backend(**POLICY_OK, **MLA)


# ---------------------------------------------------------------------------
# Kernels on GPU
# ---------------------------------------------------------------------------

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


def _gpu_ready():
    try:
        from jax_aiter.ja_compat import config as ja_config

        return bool(jax.devices("gpu")) and (
            ja_config.get_jax_aiter_lib_dir() / "triton_bridge_ja.so"
        ).is_file()
    except Exception:
        return False


gpu = pytest.mark.skipif(not _gpu_ready(), reason="GPU or Triton bridge missing")

ROW = 128
ROWS = [[37, 50, 21], [90], [128]]


def _inputs(hq, d_qk, d_v, seed=0):
    seqstart, cu_logical, seg_id, total = _layout(ROW, ROWS, slots=8)
    rng = np.random.default_rng(seed)
    q, k = (rng.normal(size=(total, hq, d_qk)).astype(np.float32) for _ in range(2))
    v = rng.normal(size=(total, hq, d_v)).astype(np.float32)
    dout = rng.normal(size=(total, hq, d_v)).astype(np.float32)
    dout[seg_id == 0] = 0.0
    bf = lambda x: jnp.asarray(x, jnp.bfloat16)
    return (bf(q), bf(k), bf(v), bf(dout), jnp.asarray(seqstart),
            jnp.asarray(cu_logical), seg_id, 1.0 / np.sqrt(d_qk))


def _auto_grads(q, k, v, dout, seqstart, cu, scale):
    def loss(q_, k_, v_):
        out = select.flash_attn_varlen_auto(
            q_, k_, v_, seqstart, seqstart, cu, cu, ROW, ROW, 0.0, scale, True, (-1, -1))
        return jnp.sum(out.astype(jnp.float32) * dout.astype(jnp.float32))

    return jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(q, k, v)


@gpu
@pytest.mark.parametrize("hq,d_qk,d_v", [(2, 192, 128), (2, 128, 128)])
def test_triton_backward_from_aiter_forward_matches_oracle(hq, d_qk, d_v):
    from jax_aiter.mha.mha import _favr_fwd, _favr_bwd

    q, k, v, dout, seqstart, cu, seg_id, scale = _inputs(hq, d_qk, d_v)
    args = (q, k, v, seqstart, seqstart, cu, cu, ROW, ROW, 0.0, scale, True, (-1, -1))
    out, res = jax.jit(_favr_fwd, static_argnums=(7, 8, 9, 10, 11, 12))(*args)
    _, _, _, out_res, lse, *_ = res
    aiter = jax.jit(lambda r, d: _favr_bwd(ROW, ROW, 0.0, scale, True, (-1, -1), r, d))(res, dout)
    hybrid = jax.jit(
        lambda d, q_, k_, v_, o_, l_: select.triton_bwd_from_aiter_forward(
            d, q_, k_, v_, o_, l_, seqstart, cu, ROW, ROW, scale)
    )(dout, q, k, v, out_res, lse)

    ref_out, *ref_grads = _oracle(q, k, v, dout, seg_id, scale)
    real = seg_id > 0
    assert _rel(out, ref_out, real) < 2e-2
    for name, got_h, got_a, ref in zip(("dq", "dk", "dv"), hybrid, aiter[:3], ref_grads):
        assert np.isfinite(np.asarray(got_h, np.float32)).all(), name
        assert _rel(got_h, ref, real) < 3e-2, (name, _rel(got_h, ref, real))
        assert _rel(got_h, got_a, real) < 3e-2, (name, _rel(got_h, got_a, real))
        assert not np.asarray(got_h, np.float32)[~real].any(), f"{name} padding rows nonzero"


@gpu
def test_varlen_auto_default_is_the_aiter_path(monkeypatch):
    from jax_aiter.mha import flash_attn_varlen_raw

    monkeypatch.delenv("JA_MHA_VARLEN_BWD_BACKEND", raising=False)
    q, k, v, dout, seqstart, cu, _seg, scale = _inputs(2, 192, 128)

    def raw_loss(q_, k_, v_):
        out = flash_attn_varlen_raw(
            q_, k_, v_, seqstart, seqstart, cu, cu, ROW, ROW, 0.0, scale, True, (-1, -1))
        return jnp.sum(out.astype(jnp.float32) * dout.astype(jnp.float32))

    raw = jax.jit(jax.grad(raw_loss, argnums=(0, 1, 2)))(q, k, v)
    auto = _auto_grads(q, k, v, dout, seqstart, cu, scale)
    for a, b in zip(auto, raw):
        np.testing.assert_array_equal(np.asarray(a, np.float32), np.asarray(b, np.float32))


@gpu
@pytest.mark.parametrize("fwd_asm", ["1", "0"], ids=["asm-fwd", "ck-fwd"])
def test_varlen_auto_forward_knob_matches_oracle(monkeypatch, fwd_asm):
    monkeypatch.setenv("JA_MHA_FWD_USE_ASM_V3", fwd_asm)
    monkeypatch.delenv("JA_MHA_VARLEN_BWD_BACKEND", raising=False)
    q, k, v, dout, seqstart, cu, seg_id, scale = _inputs(2, 192, 128, seed=3)
    out = jax.jit(lambda q_, k_, v_: select.flash_attn_varlen_auto(
        q_, k_, v_, seqstart, seqstart, cu, cu, ROW, ROW, 0.0, scale, True, (-1, -1)))(q, k, v)
    ref_out, *_ = _oracle(q, k, v, dout, seg_id, scale)
    assert _rel(out, ref_out, seg_id > 0) < 2e-2


@gpu
def test_varlen_auto_forced_triton_matches_oracle(monkeypatch):
    monkeypatch.setenv("JA_MHA_VARLEN_BWD_BACKEND", "triton")
    q, k, v, dout, seqstart, cu, seg_id, scale = _inputs(2, 192, 128, seed=1)
    grads = _auto_grads(q, k, v, dout, seqstart, cu, scale)
    _, *ref_grads = _oracle(q, k, v, dout, seg_id, scale)
    real = seg_id > 0
    for name, got, ref in zip(("dq", "dk", "dv"), grads, ref_grads):
        assert _rel(got, ref, real) < 3e-2, (name, _rel(got, ref, real))
        assert not np.asarray(got, np.float32)[~real].any(), f"{name} padding rows nonzero"


def _lowered_targets(policy_env, monkeypatch, remat_policy=None):
    if policy_env is None:
        monkeypatch.delenv("JA_MHA_VARLEN_BWD_BACKEND", raising=False)
    else:
        monkeypatch.setenv("JA_MHA_VARLEN_BWD_BACKEND", policy_env)
    q, k, v, dout, seqstart, cu, _seg, scale = _inputs(2, 192, 128)

    def loss(q_, k_, v_):
        out = select.flash_attn_varlen_auto(
            q_, k_, v_, seqstart, seqstart, cu, cu, ROW, ROW, 0.0, scale, True, (-1, -1))
        return jnp.sum(out.astype(jnp.float32) * dout.astype(jnp.float32))

    if remat_policy is not None:
        loss = jax.checkpoint(loss, policy=remat_policy)
    # Optimized HLO inlines every call; lowered text can share one callee.
    text = jax.jit(jax.grad(loss, argnums=(0, 1, 2))).lower(q, k, v).compile().as_text()
    return {
        t: text.count(f'custom_call_target="{t}"')
        for t in ("MhaFwdUnifiedJA", "MhaBwdUnifiedJA", "TritonDispatchJA")
    }


@gpu
@pytest.mark.parametrize(
    "policy_env,expected",
    [
        (None, {"MhaFwdUnifiedJA": 1, "MhaBwdUnifiedJA": 1, "TritonDispatchJA": 0}),
        ("triton", {"MhaFwdUnifiedJA": 1, "MhaBwdUnifiedJA": 0, "TritonDispatchJA": 2}),
    ],
    ids=["auto-aiter", "forced-triton"],
)
def test_varlen_auto_dispatch_and_single_forward_under_remat(monkeypatch, policy_env, expected):
    save_context = jax.checkpoint_policies.save_only_these_names("context")
    assert _lowered_targets(policy_env, monkeypatch) == expected
    assert _lowered_targets(policy_env, monkeypatch, save_context) == expected


# ---------------------------------------------------------------------------
# Padded-V group-mode backward (QK=192 / V=128 reaches the 192/192 ASM chain)
# ---------------------------------------------------------------------------

def _is_gfx950():
    try:
        from jax_aiter.ja_compat.chip_info import get_gfx

        return get_gfx() == "gfx950"
    except Exception:
        return False


on_gfx950 = pytest.mark.skipif(not _is_gfx950(), reason="padded-V targets gfx950 kernels")
PAD_V_ENV = "JA_MHA_VARLEN_BWD_PAD_V"


def _pad_v_to(q_shape=(8, 2, 192), v_shape=(8, 2, 128), dtype=jnp.bfloat16, **overrides):
    from jax_aiter.mha.mha import _varlen_bwd_pad_v_to

    kw = dict(use_v3=True, atomic_fp32=True, hard_block=False, deterministic=False)
    kw.update(overrides)
    return _varlen_bwd_pad_v_to(
        jax.ShapeDtypeStruct(q_shape, dtype), jax.ShapeDtypeStruct(v_shape, dtype), **kw)


@on_gfx950
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16], ids=["bf16", "fp16"])
def test_pad_v_applies_to_mla_group_backward(monkeypatch, dtype):
    monkeypatch.delenv(PAD_V_ENV, raising=False)
    assert _pad_v_to(dtype=dtype) == 192


@on_gfx950
@pytest.mark.parametrize(
    "override",
    [dict(use_v3=False), dict(atomic_fp32=False), dict(hard_block=True), dict(deterministic=True),
     dict(q_shape=(8, 2, 192), v_shape=(8, 2, 192)), dict(q_shape=(8, 2, 128), v_shape=(8, 2, 128)),
     dict(dtype=jnp.float32)],
    ids=["ck-forced", "a16", "dropout-or-window", "deterministic", "equal-192", "equal-128", "fp32"],
)
def test_pad_v_declines_other_configs(monkeypatch, override):
    monkeypatch.delenv(PAD_V_ENV, raising=False)
    assert _pad_v_to(**override) is None


@on_gfx950
def test_pad_v_env_keeps_ck(monkeypatch):
    monkeypatch.setenv(PAD_V_ENV, "0")
    assert _pad_v_to() is None


def _raw_grads(q, k, v, dout, seqstart, cu, scale):
    from jax_aiter.mha import flash_attn_varlen_raw

    def loss(q_, k_, v_):
        out = flash_attn_varlen_raw(
            q_, k_, v_, seqstart, seqstart, cu, cu, ROW, ROW, 0.0, scale, True, (-1, -1))
        return jnp.sum(out.astype(jnp.float32) * dout.astype(jnp.float32))

    return jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(q, k, v)


def _partitioned_grads(q, k, v, dout, seqstart, cu, scale):
    from jax_aiter.mha.mha import flash_attn_varlen

    def loss(q_, k_, v_):
        out = flash_attn_varlen(
            q_, k_, v_, seqstart, seqstart,
            cu_seqlens_q_logical=cu, cu_seqlens_k_logical=cu,
            max_seqlen_q=ROW, max_seqlen_k=ROW, softmax_scale=scale, causal=True)[0]
        return jnp.sum(out.astype(jnp.float32) * dout.astype(jnp.float32))

    return jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(q, k, v)


@gpu
@on_gfx950
@pytest.mark.parametrize("grads_fn", [_raw_grads, _partitioned_grads], ids=["raw", "partitioned"])
def test_padded_v_backward_matches_ck_and_oracle(monkeypatch, grads_fn):
    q, k, v, dout, seqstart, cu, seg_id, scale = _inputs(2, 192, 128, seed=4)
    monkeypatch.delenv(PAD_V_ENV, raising=False)
    padded = grads_fn(q, k, v, dout, seqstart, cu, scale)
    monkeypatch.setenv(PAD_V_ENV, "0")
    ck = grads_fn(q, k, v, dout, seqstart, cu, scale)
    _, *ref_grads = _oracle(q, k, v, dout, seg_id, scale)
    real = seg_id > 0
    for name, got, base, ref in zip(("dq", "dk", "dv"), padded, ck, ref_grads):
        assert got.shape == base.shape, name
        assert np.isfinite(np.asarray(got, np.float32)).all(), name
        assert _rel(got, ref, real) < 3e-2, (name, _rel(got, ref, real))
        assert _rel(got, base, real) < 1e-2, (name, _rel(got, base, real))
        assert not np.asarray(got, np.float32)[~real].any(), f"{name} padding rows nonzero"


_DISPATCH_PROBE = textwrap.dedent("""
    import numpy as np, jax, jax.numpy as jnp
    from jax_aiter.mha import flash_attn_varlen_raw

    total, hq = 256, 2
    seqstart = jnp.asarray([0, 100, 128, 256, 256], jnp.int32)
    cu = jnp.asarray([0, 100, 120, 248, 248], jnp.int32)
    rng = np.random.default_rng(0)
    q, k = (jnp.asarray(rng.normal(size=(total, hq, 192)), jnp.bfloat16) for _ in range(2))
    v, dout = (jnp.asarray(rng.normal(size=(total, hq, 128)), jnp.bfloat16) for _ in range(2))

    def loss(q_, k_, v_):
        out = flash_attn_varlen_raw(q_, k_, v_, seqstart, seqstart, cu, cu, 128, 128,
                                    0.0, 192 ** -0.5, True, (-1, -1))
        return jnp.sum(out.astype(jnp.float32) * dout.astype(jnp.float32))

    jax.block_until_ready(jax.jit(jax.grad(loss, argnums=(0, 1, 2)))(q, k, v))
""")


@gpu
@on_gfx950
@pytest.mark.parametrize("pad_v,expect_asm", [(None, True), ("0", False)], ids=["default", "pad-v-off"])
def test_padded_v_dispatches_group_asm_backward(pad_v, expect_asm):
    env = {k: v for k, v in os.environ.items() if k != PAD_V_ENV}
    if pad_v is not None:
        env[PAD_V_ENV] = pad_v
    proc = subprocess.run([sys.executable, "-c", _DISPATCH_PROBE], env=env,
                          capture_output=True, text=True, timeout=900)
    log = proc.stdout + proc.stderr
    assert proc.returncode == 0, log[-4000:]
    asm_group_bwd = "bwd_hd192_bf16_causal" in log and "psskddv_group" in log
    assert asm_group_bwd == expect_asm, log[-4000:]


@gpu
def test_flash_attn_auto_full_rows_matches_oracle():
    b, s, h, d_qk, d_v = 2, 128, 2, 192, 128
    rng = np.random.default_rng(2)
    shape = lambda d: (b, s, h, d)
    q, k = (rng.normal(size=shape(d_qk)).astype(np.float32) for _ in range(2))
    v, dout = (rng.normal(size=shape(d_v)).astype(np.float32) for _ in range(2))
    scale = 1.0 / np.sqrt(d_qk)
    bf = lambda x: jnp.asarray(x, jnp.bfloat16)

    def loss(q_, k_, v_):
        out = select.flash_attn_auto(q_, k_, v_, softmax_scale=scale)
        return jnp.sum(out.astype(jnp.float32) * bf(dout).astype(jnp.float32)), out

    (_, out), grads = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2), has_aux=True))(
        bf(q), bf(k), bf(v))
    flat = lambda x: np.asarray(x, np.float32).reshape(b * s, h, -1)
    seg_id = np.repeat(np.arange(1, b + 1), s)
    ref_out, *ref_grads = _oracle(*(flat(bf(x)) for x in (q, k, v, dout)), seg_id, scale)
    real = seg_id > 0
    assert _rel(flat(out), ref_out, real) < 2e-2
    for name, got, ref in zip(("dq", "dk", "dv"), grads, ref_grads):
        assert _rel(flat(got), ref, real) < 3e-2, (name, _rel(flat(got), ref, real))
