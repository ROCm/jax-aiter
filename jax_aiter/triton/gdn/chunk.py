# SPDX-License-Identifier: Apache-2.0
"""Chunked Gated DeltaNet (gated delta rule) with FLA Triton kernels.

Forward and backward follow FLA's ``chunk_gated_delta_rule_fwd/bwd`` for the
training case: equal-length sequences, no initial or final recurrent state,
and no context parallelism. JAX owns the VJP; FLA's Torch autograd wrappers
are not used. The chunk-local cumulative sums run in JAX.
"""

from __future__ import annotations

import functools
import json
import os
from typing import Any

import jax
import jax.numpy as jnp

from ..call import triton_call
from .fla import fla_kernels

CHUNK_SIZE = 64
# FLA evaluates gates with exp2, so cumulative gates are kept in log2 units.
_RCP_LN2 = 1.4426950216

# Launch configs for MI355X (gfx950). Override per kernel with
# JA_FLA_GDN_CONFIG='{"fwd_h": {"BV": 64, "num_warps": 4, "num_stages": 1}}'.
_DEFAULT_CONFIGS: dict[str, dict[str, Any]] = {
    "kkt_solve": {"BK": 64, "num_warps": 4},
    "recompute_w_u": {"num_warps": 4, "num_stages": 2},
    "fwd_h": {"BV": 32, "num_warps": 4, "num_stages": 2},
    "fwd_o": {"BK": 64, "BV": 64, "num_warps": 4, "num_stages": 2},
    "bwd_dv_local": {"num_warps": 4, "num_stages": 2},
    "bwd_dhu": {"BV": 32, "num_warps": 4, "num_stages": 1},
    "bwd_dqkwg": {"num_warps": 4, "num_stages": 2},
    "wy_bwd": {"num_warps": 4, "num_stages": 2},
}
_OPTION_KEYS = ("num_warps", "num_stages", "waves_per_eu")
# Block sizes FLA's launchers pick for K = V = 128 when the device has at least
# 101376 B of shared memory (gfx950 has 160 KiB).
_TILE = 64


@functools.lru_cache(maxsize=None)
def _config_overrides() -> dict[str, dict[str, Any]]:
    raw = os.environ.get("JA_FLA_GDN_CONFIG")
    return json.loads(raw) if raw else {}


def _config(name: str) -> dict[str, Any]:
    return {**_DEFAULT_CONFIGS[name], **_config_overrides().get(name, {})}


def _unwrap(kernel):
    """Return the ``@triton.jit`` function and FLA's heuristics rules."""
    rules: dict[str, Any] = {}
    fn = kernel
    while not (hasattr(fn, "arg_names") and hasattr(fn, "constexprs")):
        values = getattr(fn, "values", None)
        if isinstance(values, dict):
            rules.update(values)
        fn = fn.fn
    return fn, rules


def _pointer_type(dtype) -> str:
    dtype = jnp.dtype(dtype)
    if dtype == jnp.bfloat16:
        return "*bf16"
    if dtype == jnp.float16:
        return "*fp16"
    if dtype == jnp.float32:
        return "*fp32"
    if dtype == jnp.int32:
        return "*i32"
    raise TypeError(f"Unsupported FLA buffer dtype {dtype}")


def _launch(
    name: str,
    *,
    grid,
    tensors: dict[str, jax.Array | None],
    outputs: dict[str, jax.ShapeDtypeStruct],
    scalars: dict[str, tuple[Any, str]],
    constexprs: dict[str, Any],
    zero_outputs: tuple[str, ...] = (),
    config: dict[str, Any] | None = None,
):
    """Launch one FLA kernel, filling heuristics the way FLA's launcher would."""
    fn, rules = _unwrap(getattr(fla_kernels(), name))
    config = dict(_config(name) if config is None else config)
    options = {key: config.pop(key) for key in _OPTION_KEYS if key in config}
    cx = {**constexprs, **config}
    probe = {**tensors, **outputs, **cx}
    for rule_name, rule in rules.items():
        if rule_name not in cx:
            cx[rule_name] = rule(probe)

    output_names = tuple(outputs)
    runtime_types: dict[str, str] = {}
    kinds: list[str] = []
    arrays: list[jax.Array] = []
    aliases: dict[int, int] = {}
    scalar_values: dict[str, Any] = {}
    for name_ in fn.arg_names:
        if name_ in cx:
            continue
        if name_ in outputs:
            runtime_types[name_] = _pointer_type(outputs[name_].dtype)
            kinds.append("buffer")
            if name_ in zero_outputs:
                spec = outputs[name_]
                aliases[len(arrays)] = output_names.index(name_)
                arrays.append(jnp.zeros(spec.shape, spec.dtype))
        elif name_ in tensors:
            value = tensors[name_]
            if value is None:
                cx[name_] = None
                continue
            runtime_types[name_] = _pointer_type(value.dtype)
            kinds.append("buffer")
            arrays.append(value)
        elif name_ in scalars:
            value, triton_type = scalars[name_]
            runtime_types[name_] = triton_type
            kinds.append("scalar")
            scalar_values[name_] = value
        else:
            raise ValueError(f"FLA kernel {name}: no value for argument {name_!r}")

    results = triton_call(
        *arrays,
        fn=fn,
        runtime_types=runtime_types,
        constexprs=cx,
        kinds=kinds,
        scalars=scalar_values,
        out_shape=tuple(outputs[n] for n in output_names),
        output_names=output_names,
        grid=tuple(grid) + (1,) * (3 - len(grid)),
        options=options,
        input_output_aliases=aliases,
    )
    return results if len(results) > 1 else results[0]


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _sds(shape, dtype) -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct(tuple(shape), dtype)


def _chunk_cumsum(x: jax.Array, reverse: bool = False) -> jax.Array:
    b, t, hv = x.shape
    chunks = x.reshape(b, t // CHUNK_SIZE, CHUNK_SIZE, hv)
    if reverse:
        out = jnp.flip(jnp.cumsum(jnp.flip(chunks, 2), axis=2), 2)
    else:
        out = jnp.cumsum(chunks, axis=2)
    return out.reshape(b, t, hv)


def _dims(q, v):
    b, t, h, k = q.shape
    hv, dv = v.shape[2], v.shape[3]
    return b, t, h, hv, k, dv, t // CHUNK_SIZE


def _recompute_w_u(k, v, beta, a, g):
    b, t, h, hv, dk, dv, nt = _dims(k, v)
    return _launch(
        "recompute_w_u",
        grid=(nt, b * hv),
        tensors=dict(k=k, v=v, beta=beta, A=a, g=g, cu_seqlens=None, chunk_indices=None),
        outputs=dict(w=_sds((b, t, hv, dk), k.dtype), u=_sds(v.shape, v.dtype)),
        scalars=dict(T=(t, "i32")),
        constexprs=dict(H=h, HV=hv, K=dk, V=dv, BT=CHUNK_SIZE, BK=_TILE, BV=_TILE),
    )


def _fwd_h(k, u, w, g):
    b, t, h, hv, dk, dv, nt = _dims(k, u)
    config = _config("fwd_h")
    return _launch(
        "fwd_h",
        grid=(_cdiv(dv, config["BV"]) * b * hv,),
        tensors=dict(k=k, v=u, w=w, g=g, gk=None, h0=None, ht=None,
                     cu_seqlens=None, chunk_offsets=None),
        outputs=dict(h=_sds((b, nt, hv, dk, dv), k.dtype),
                     v_new=_sds(u.shape, u.dtype)),
        scalars=dict(T=(t, "i32")),
        constexprs=dict(H=h, HV=hv, K=dk, V=dv, BT=CHUNK_SIZE, STATE_V_FIRST=False),
        config=config,
    )


def _forward(q, k, v, g, beta, scale):
    b, t, h, hv, dk, dv, nt = _dims(q, v)
    gc = _chunk_cumsum(g.astype(jnp.float32)) * _RCP_LN2
    a = _launch(
        "kkt_solve",
        grid=(nt, b * hv),
        tensors=dict(k=k, g=gc, beta=beta, cu_seqlens=None, chunk_indices=None),
        outputs=dict(A=_sds((b, t, hv, CHUNK_SIZE), k.dtype)),
        scalars=dict(T=(t, "i32")),
        constexprs=dict(H=h, HV=hv, K=dk, BT=CHUNK_SIZE, BC=16),
        zero_outputs=("A",),
    )
    w, u = _recompute_w_u(k, v, beta, a, gc)
    hs, v_new = _fwd_h(k, u, w, gc)
    config = _config("fwd_o")
    o = _launch(
        "fwd_o",
        grid=(_cdiv(dv, config["BV"]) * nt, b * hv),
        tensors=dict(q=q, k=k, v=v_new, h=hs, g=gc, g_gamma=None,
                     cu_seqlens=None, chunk_indices=None),
        outputs=dict(o=_sds(v.shape, v.dtype)),
        scalars=dict(scale=(scale, "fp32"), T=(t, "i32")),
        constexprs=dict(H=h, HV=hv, K=dk, V=dv, BT=CHUNK_SIZE, STATE_V_FIRST=False),
        config=config,
    )
    return o, (q, k, v, gc, beta, a)


def _backward(scale, residuals, do):
    q, k, v, gc, beta, a = residuals
    b, t, h, hv, dk, dv, nt = _dims(q, v)
    do = do.astype(v.dtype)
    w, u = _recompute_w_u(k, v, beta, a, gc)
    hs, v_new = _fwd_h(k, u, w, gc)
    shape_cx = dict(H=h, HV=hv, K=dk, V=dv, BT=CHUNK_SIZE)

    dv_local = _launch(
        "bwd_dv_local",
        grid=(nt, b * hv),
        tensors=dict(q=q, k=k, g=gc, g_gamma=None, A=None, do=do,
                     cu_seqlens=None, chunk_indices=None),
        outputs=dict(dv=_sds(do.shape, do.dtype)),
        scalars=dict(scale=(scale, "fp32"), T=(t, "i32")),
        constexprs=dict(**shape_cx, BK=_TILE, BV=_TILE),
    )
    config = _config("bwd_dhu")
    dh, dv2 = _launch(
        "bwd_dhu",
        grid=(_cdiv(dv, config["BV"]) * b * hv,),
        tensors=dict(q=q, k=k, w=w, g=gc, gk=None, dht=None, dh0=None, do=do,
                     dv=dv_local, cu_seqlens=None, chunk_offsets=None),
        outputs=dict(dh=_sds((b, nt, hv, dk, dv), q.dtype), dv2=_sds(do.shape, do.dtype)),
        scalars=dict(scale=(scale, "fp32"), T=(t, "i32")),
        constexprs=dict(**shape_cx, STATE_V_FIRST=False),
        config=config,
    )
    nk = _cdiv(dk, _TILE)
    dq, dk_, dw, dg = _launch(
        "bwd_dqkwg",
        grid=(nk * nt, b * hv),
        tensors=dict(q=q, k=k, v=v_new, g=gc, g_gamma=None, h=hs, do=do, dh=dh,
                     dv=dv2, cu_seqlens=None, chunk_indices=None),
        outputs=dict(dq=_sds((b, t, hv, dk), q.dtype), dk=_sds((b, t, hv, dk), k.dtype),
                     dw=_sds(w.shape, w.dtype), dg=_sds((nk, b, t, hv), jnp.float32)),
        scalars=dict(scale=(scale, "fp32"), T=(t, "i32")),
        constexprs=dict(**shape_cx, B=b, BK=_TILE, BV=_TILE, STATE_V_FIRST=False),
    )
    dk2, dv_out, db, dg2 = _launch(
        "wy_bwd",
        grid=(nt, b * hv),
        tensors=dict(k=k, v=v, beta=beta, g=gc, A=a, dw=dw, du=dv2,
                     cu_seqlens=None, chunk_indices=None),
        outputs=dict(dk=_sds((b, t, hv, dk), k.dtype), dv=_sds(v.shape, v.dtype),
                     db=_sds(beta.shape, beta.dtype), dg=_sds(gc.shape, jnp.float32)),
        scalars=dict(T=(t, "i32")),
        constexprs=dict(**shape_cx, BK=_TILE, BV=_TILE),
    )
    if h != hv:
        groups = hv // h
        dq = dq.astype(jnp.float32).reshape(b, t, h, groups, dk).sum(3)
        dk_ = dk_.astype(jnp.float32).reshape(b, t, h, groups, dk).sum(3)
        dk2 = dk2.astype(jnp.float32).reshape(b, t, h, groups, dk).sum(3)
    dk_total = dk_.astype(jnp.float32) + dk2.astype(jnp.float32)
    dg_total = _chunk_cumsum(dg.sum(0) + dg2, reverse=True)
    return dq.astype(q.dtype), dk_total.astype(k.dtype), dv_out, dg_total, db


@functools.partial(jax.custom_vjp, nondiff_argnums=(5,))
def _gdn(q, k, v, g, beta, scale):
    return _forward(q, k, v, g, beta, scale)[0]


def _gdn_fwd(q, k, v, g, beta, scale):
    return _forward(q, k, v, g, beta, scale)


def _gdn_bwd(scale, residuals, do):
    return _backward(scale, residuals, do)


_gdn.defvjp(_gdn_fwd, _gdn_bwd)


def chunk_gated_delta_rule_fla(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    scale: float | None = None,
) -> jax.Array:
    """Gated delta rule over equal-length sequences, FLA chunk algorithm.

    Args:
      q, k: ``[B, T, H, K]`` (already L2-normalized if the model wants it).
      v: ``[B, T, HV, V]``; ``HV`` must be a multiple of ``H`` (grouped values).
      g: ``[B, T, HV]`` per-token log decay (natural log, <= 0).
      beta: ``[B, T, HV]``.
      scale: query scale; defaults to ``K ** -0.5``.

    Returns:
      ``[B, T, HV, V]`` in ``v.dtype``. No recurrent state is returned.
    """
    b, t, h, dk = q.shape
    if k.shape != q.shape:
        raise ValueError(f"q {q.shape} and k {k.shape} must match")
    if v.shape[:2] != (b, t) or v.shape[2] % h:
        raise ValueError(f"v {v.shape} must be [B, T, HV, V] with HV % H == 0")
    hv = v.shape[2]
    if g.shape != (b, t, hv) or beta.shape != (b, t, hv):
        raise ValueError(f"g {g.shape} and beta {beta.shape} must be [B, T, HV]")
    if t % CHUNK_SIZE:
        raise ValueError(f"sequence length {t} must be a multiple of {CHUNK_SIZE}")
    if dk > 256 or v.shape[3] > 256:
        raise ValueError("FLA chunk kernels support head dims up to 256")
    if scale is None:
        scale = dk ** -0.5
    return _gdn(q, k, v, g.astype(jnp.float32), beta, float(scale))
