# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Raw MHA forward/backward FFI wrappers via unified AITER entry point.

Single-kernel ops with no custom_vjp or custom_partitioning.
CK vs ASM v3 dispatch is handled internally by AITER.
"""

from __future__ import annotations
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np

from ..ffi.registry import register_ffi_target

MhaFwdConfig = namedtuple("MhaFwdConfig", [
    "dropout_p", "softmax_scale", "is_causal", "wl", "wr",
    "return_lse", "return_randval", "use_asm_v3", "how_v3_bf16_cvt",
    "max_seqlen_q", "max_seqlen_k", "min_seqlen_q",
    "logits_soft_cap", "zero_tensors",
    "cp_axis", "cp_size", "cp_load_balanced",
])

MhaBwdConfig = namedtuple("MhaBwdConfig", [
    "dropout_p", "softmax_scale", "is_causal", "wl", "wr",
    "deterministic", "use_asm_v3", "is_v3_atomic_fp32", "how_v3_bf16_cvt",
    "max_seqlen_q", "max_seqlen_k", "zero_tensors",
    "cp_axis", "cp_size", "cp_load_balanced",
])


def _ensure_registered(target: str):
    register_ffi_target(target, "ROCM")


def _empty(dtype):
    return jnp.zeros((0,), dtype=dtype)


def _sf(x) -> np.float32:
    return np.float32(x)


def _si(x) -> np.int32:
    return np.int32(x)


def _cached_unified_fwd_call(out_shape, lse_shape, p_shape, rng_shape, dtype):
    call = jax.ffi.ffi_call(
        "MhaFwdUnifiedJA",
        (
            jax.ShapeDtypeStruct(out_shape, dtype),
            jax.ShapeDtypeStruct(lse_shape, jnp.float32),
            jax.ShapeDtypeStruct(p_shape, jnp.uint8),
            jax.ShapeDtypeStruct(rng_shape, jnp.int64),
        ),
        vmap_method="broadcast_all",
        input_layouts=[None] * 11,
        output_layouts=[None] * 4,
        has_side_effect=False,
    )

    def _invoke(q, k, v, cu_sq, cu_skv, out_prov, bias, alibi, gen,
                cu_sq_log, cu_skv_log, *,
                dropout_p, softmax_scale, is_causal, wl, wr,
                return_lse, return_randval, use_asm_v3, how_v3_bf16_cvt,
                max_seqlen_q_attr, max_seqlen_k_attr, min_seqlen_q,
                logits_soft_cap, zero_tensors):
        return call(q, k, v, cu_sq, cu_skv, out_prov, bias, alibi, gen,
                    cu_sq_log, cu_skv_log,
                    dropout_p=dropout_p, softmax_scale=softmax_scale,
                    is_causal=is_causal, window_size_left=wl, window_size_right=wr,
                    return_softmax_lse=return_lse,
                    return_dropout_randval=return_randval,
                    use_asm_v3=use_asm_v3, how_v3_bf16_cvt=how_v3_bf16_cvt,
                    max_seqlen_q_attr=max_seqlen_q_attr,
                    max_seqlen_k_attr=max_seqlen_k_attr,
                    min_seqlen_q=min_seqlen_q,
                    logits_soft_cap=logits_soft_cap,
                    zero_tensors=zero_tensors)

    return jax.jit(_invoke, static_argnames=(
        "dropout_p", "softmax_scale", "is_causal", "wl", "wr",
        "return_lse", "return_randval", "use_asm_v3", "how_v3_bf16_cvt",
        "max_seqlen_q_attr", "max_seqlen_k_attr", "min_seqlen_q",
        "logits_soft_cap", "zero_tensors"))


def _cached_unified_bwd_call(dq_shape, dk_shape, dv_shape, sd_shape, dbias_shape, dtype):
    call = jax.ffi.ffi_call(
        "MhaBwdUnifiedJA",
        (
            jax.ShapeDtypeStruct(dq_shape, dtype),
            jax.ShapeDtypeStruct(dk_shape, dtype),
            jax.ShapeDtypeStruct(dv_shape, dtype),
            jax.ShapeDtypeStruct(sd_shape, jnp.float32),
            jax.ShapeDtypeStruct(dbias_shape, dtype),
        ),
        vmap_method="broadcast_all",
        input_layouts=[None] * 17,
        output_layouts=[None] * 5,
        has_side_effect=False,
    )

    def _invoke(dout, q, k, v, out, lse, cu_sq, cu_sk,
                dq, dk, dv, bias, alibi, rng, gen,
                cu_sq_log, cu_sk_log, *,
                dropout_p, softmax_scale, is_causal, wl, wr,
                deterministic, use_asm_v3, is_v3_atomic_fp32, how_v3_bf16_cvt,
                max_seqlen_q_attr, max_seqlen_k_attr, zero_tensors):
        return call(dout, q, k, v, out, lse, cu_sq, cu_sk,
                    dq, dk, dv, bias, alibi, rng, gen,
                    cu_sq_log, cu_sk_log,
                    dropout_p=dropout_p, softmax_scale=softmax_scale,
                    is_causal=is_causal, window_size_left=wl, window_size_right=wr,
                    deterministic=deterministic, use_asm_v3=use_asm_v3,
                    is_v3_atomic_fp32=is_v3_atomic_fp32,
                    how_v3_bf16_cvt=how_v3_bf16_cvt,
                    max_seqlen_q_attr=max_seqlen_q_attr,
                    max_seqlen_k_attr=max_seqlen_k_attr,
                    zero_tensors=zero_tensors)

    return jax.jit(_invoke, static_argnames=(
        "dropout_p", "softmax_scale", "is_causal", "wl", "wr",
        "deterministic", "use_asm_v3", "is_v3_atomic_fp32", "how_v3_bf16_cvt",
        "max_seqlen_q_attr", "max_seqlen_k_attr", "zero_tensors"))


def mha_fwd(q, k, v, cu_sq, cu_skv, out_prov, bias, alibi, gen,
            cu_sq_log=None, cu_skv_log=None, config=None):
    """Raw MHA forward FFI call. Derives output shapes from per-shard Q.

    Args:
        q: [B, Sq, Hq, D] (batch) or [total_q, Hq, D] (varlen).
        k: [B, Sk, Hk, D] or [total_k, Hk, D].
        v: [B, Sk, Hk, Dv] or [total_k, Hk, Dv].
        cu_sq: Cumulative physical query offsets (varlen) or empty.
        cu_skv: Cumulative physical KV offsets (varlen) or empty.
        out_prov: Provisioning tensor (empty).
        bias: Attention bias or empty.
        alibi: ALiBi slopes or empty.
        gen: RNG generator state or empty.
        cu_sq_log: Cumulative logical query lengths excluding padding, or empty.
        cu_skv_log: Cumulative logical KV lengths excluding padding, or empty.
        config: MhaFwdConfig namedtuple.

    Returns:
        (out, lse, p, rng_state) tuple.
    """
    _ensure_registered("MhaFwdUnifiedJA")
    if cu_sq_log is None:
        cu_sq_log = _empty(jnp.int32)
    if cu_skv_log is None:
        cu_skv_log = _empty(jnp.int32)
    is_varlen = (q.ndim == 3)
    if is_varlen:
        total_q, hq, dq = q.shape
        _, hk, dv = v.shape
        out_shape = (total_q, hq, dv)
        # Varlen LSE spans ALL packed tokens per head: the FFI handler sets
        # nhead_stride_lse = stride(lse_dims, 0), so the buffer must be
        # (hq, total_q). Using max_seqlen_q here undersizes it whenever
        # total_q > max_seqlen_q (multi-segment packing) -> OOB write -> GPU
        # memory-access fault. (was: (hq, max_seqlen_q))
        lse_shape = (hq, total_q) if config.return_lse else (0,)
        p_shape = (0,)
    else:
        b, sq, hq, dq = q.shape
        _, sk, hk, dv = v.shape
        out_shape = (b, sq, hq, dv)
        lse_shape = (b, hq, sq) if config.return_lse else (0,)
        p_shape = (b, hq, sq, sk) if config.return_randval else (0,)
    rng_shape = (2,)
    fn = _cached_unified_fwd_call(out_shape, lse_shape, p_shape,
                                  rng_shape, q.dtype)
    return fn(q, k, v, cu_sq, cu_skv, out_prov, bias, alibi, gen,
              cu_sq_log, cu_skv_log,
              dropout_p=_sf(config.dropout_p),
              softmax_scale=_sf(config.softmax_scale),
              is_causal=config.is_causal,
              wl=_si(config.wl), wr=_si(config.wr),
              return_lse=config.return_lse,
              return_randval=config.return_randval,
              use_asm_v3=config.use_asm_v3,
              how_v3_bf16_cvt=_si(config.how_v3_bf16_cvt),
              max_seqlen_q_attr=_si(config.max_seqlen_q),
              max_seqlen_k_attr=_si(config.max_seqlen_k),
              min_seqlen_q=_si(config.min_seqlen_q),
              logits_soft_cap=_sf(config.logits_soft_cap),
              zero_tensors=config.zero_tensors)


def mha_bwd(dout, q, k, v, out, lse, cu_sq, cu_sk,
            dq_ws, dk_ws, dv_ws, bias, alibi, rng, gen,
            cu_sq_log=None, cu_sk_log=None, config=None):
    """Raw MHA backward FFI call. Derives output shapes from per-shard Q.

    Args:
        dout: Gradient of output, same shape as forward out.
        q, k, v: Same as forward.
        out: Forward output.
        lse: Log-sum-exp from forward.
        cu_sq, cu_sk: Cumulative physical offsets (varlen) or empty.
        dq_ws, dk_ws, dv_ws: Workspace tensors (empty).
        bias, alibi, rng, gen: Same as forward.
        cu_sq_log, cu_sk_log: Cumulative logical lengths excluding padding, or
            empty.
        config: MhaBwdConfig namedtuple.

    Returns:
        (dq, dk, dv, softmax_d, dbias) tuple.
    """
    _ensure_registered("MhaBwdUnifiedJA")
    if cu_sq_log is None:
        cu_sq_log = _empty(jnp.int32)
    if cu_sk_log is None:
        cu_sk_log = _empty(jnp.int32)
    is_varlen = (q.ndim == 3)
    if is_varlen:
        total_q, hq, dq_dim = q.shape
        _, hk, _ = k.shape
        dv_dim = v.shape[-1]
        total_k = k.shape[0]
        dq_shape = (total_q, hq, dq_dim)
        dk_shape = (total_k, hk, dq_dim)
        dv_shape = (total_k, hk, dv_dim)
        # softmax_d spans all packed tokens per head (matches LSE layout above).
        sd_shape = (hq, total_q)
        dbias_shape = (0,)
    else:
        b, sq, hq, dq_dim = q.shape
        _, sk, hk, _ = k.shape
        dv_dim = v.shape[-1]
        dq_shape = (b, sq, hq, dq_dim)
        dk_shape = (b, sk, hk, dq_dim)
        dv_shape = (b, sk, hk, dv_dim)
        sd_shape = (b, hq, sq)
        dbias_shape = (b, sq, hq, sk) if (bias.size > 0) else (0,)
    fn = _cached_unified_bwd_call(dq_shape, dk_shape, dv_shape,
                                  sd_shape, dbias_shape, q.dtype)
    return fn(dout, q, k, v, out, lse, cu_sq, cu_sk,
              dq_ws, dk_ws, dv_ws, bias, alibi, rng, gen,
              cu_sq_log, cu_sk_log,
              dropout_p=_sf(config.dropout_p),
              softmax_scale=_sf(config.softmax_scale),
              is_causal=config.is_causal,
              wl=_si(config.wl), wr=_si(config.wr),
              deterministic=config.deterministic,
              use_asm_v3=config.use_asm_v3,
              is_v3_atomic_fp32=config.is_v3_atomic_fp32,
              how_v3_bf16_cvt=_si(config.how_v3_bf16_cvt),
              max_seqlen_q_attr=_si(config.max_seqlen_q),
              max_seqlen_k_attr=_si(config.max_seqlen_k),
              zero_tensors=config.zero_tensors)
