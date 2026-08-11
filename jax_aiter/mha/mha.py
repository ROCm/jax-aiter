# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Simplified MHA using unified AITER entry point.

Calls aiter::mha_fwd / aiter::mha_bwd through a single FFI handler per
direction. CK vs ASM v3 dispatch is handled internally by AITER based on
the use_asm_v3 flag. No Python-side dispatch logic.

GSPMD sharding: custom_partitioning tells XLA how to partition the FFI
calls for multi-GPU FSDP.  For batch-mode attention every dimension
except the batch axis is replicated, so each device runs independently
on its local batch shard (output sharding = Q input sharding, no
collectives).  custom_partitioning wraps the raw FFI calls;
custom_vjp sits on the outer public API -- they compose because they
are on different levels of the call stack.
"""

from __future__ import annotations
import logging
import os
from typing import Tuple, Optional
from functools import partial

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.experimental.custom_partitioning import custom_partitioning, SdyShardingRule
from jax.sharding import NamedSharding, PartitionSpec as P

from ..ja_compat.chip_info import get_gfx
from ..ops.mha import (
    mha_fwd as _mha_fwd_raw,
    mha_bwd as _mha_bwd_raw,
    MhaFwdConfig, MhaBwdConfig,
    _empty,
)

log = logging.getLogger("jax-aiter.mha_v2")


# ---------------------------------------------------------------------------
# Backward dispatch resolution
# ---------------------------------------------------------------------------
#
# Two classes of reason to decline the ASM v3 backward, kept apart because they
# carry very different confidence:
#
#   hard    -- no ASM binary exists for the config (dropout, bias, sliding
#              window). Never overridable.
#   suspect -- ``c5bc2e2`` (2026-03-11) measured wrong gradients for causal
#              gfx950 and fell back to CK: batched ``seqlen_q > seqlen_k``, and
#              varlen ``max_seqlen_k > 256`` in group mode. That measurement was
#              taken against AITER ``v0.1.11.post1-14-g3baf198aa``; 76 backward
#              ASM binaries changed between that pin and the v0.1.19 we now ship.
#              Overridable via ``JA_MHA_BWD_FORCE_ASM_V3`` so it stays testable.
#
# The varlen half of the suspect set was retested on v0.1.19 and does not
# reproduce, so it no longer blocks. 30 cells at the llama3-8b shape (hd128,
# bf16, 32 query / 8 KV heads, causal group) across seqlen 256/512/2048/8192,
# single-segment and 32-segment packing, against an independent FP32 oracle:
# fp32 atomics match the CK backward to six decimal places everywhere, and
# 16-bit atomics stay within 1.5x of CK on dQ at worst (single 8192 segment;
# identical to CK at production packing). Evidence, including the loaded code
# objects proving ASM really dispatched:
# docs/runs/llama3_8b/kernels/20260730_8b_bf16_mha_asm_accuracy_097_i1/
#
# The batched ``seqlen_q > seqlen_k`` half was NOT retested and still blocks.
# AITER's own smoke_test_bwd_v3.sh skips atomic16 whenever ``sq != sk``, so at
# least the a16 form of it is a documented upstream restriction.
#
# Environment knobs, all defaulting to historical behaviour:
#
#   JA_MHA_BWD_USE_ASM_V3=0    force the CK/v2 backward (FAv3-vs-FAv2 A/B)
#   JA_MHA_BWD_FORCE_ASM_V3=1  bypass the *suspect* guard only
#   JA_MHA_BWD_ATOMIC_FP32=0|1 dQ atomic width, independent of the ASM decision.
#                              Unset keeps the historical tie to use_asm_v3, so
#                              ASM implies fp32 atomics. TE runs a16 for the same
#                              shape, which this makes reachable.
#   JA_MHA_BWD_BF16_CVT=0|1|2  0=RTNE, 1=RTNA, 2=RTZ. Unset keeps the arch
#                              default (0 on gfx950). TE uses 2.

# ---------------------------------------------------------------------------
# Rematerialisation tagging
# ---------------------------------------------------------------------------
#
# A framework layer-remat policy can only *save* a value it can name. TE names
# its attention output, softmax LSE and RNG state ``context``
# (transformer_engine/jax/attention.py, ``context_checkpoint_name``), which is
# exactly what MaxText's ``minimal_with_context`` / ``minimal_flash`` /
# ``minimal_flash_save_fp4col`` policies list
# (maxtext/src/maxtext/layers/decoders.py, ``_minimal_names``). Leaving our
# residuals untagged makes the same policy recompute the whole attention
# forward inside the backward pass, which is why the reroute lost time while
# *reducing* compiled memory. Tagging costs nothing when the enclosing graph is
# not rematerialised.
_CONTEXT_CKPT_NAME = "context"


def _has_padding(cu_logical) -> bool:
    """True when the caller described padding via cumulative logical lengths."""
    return cu_logical is not None and getattr(cu_logical, "size", 0) > 0


def _zero_pad(cu_logical) -> bool:
    """Whether to clear output/gradient buffers before the kernel runs.

    AITER writes only the logical tokens of each segment, so in a padded layout
    the padding rows of o/lse/dq/dk/dv keep whatever the buffer already held.
    Those rows are still part of the tensor MaxText hands to the next op, so they
    must start at a defined value. TE solves this the same way and gates it on
    ``NVTE_CK_ZERO_OUT_PAD`` (default on); ``JA_MHA_ZERO_PAD=0`` is the matching
    escape for measuring what the clear costs. Tight packings have no padding
    rows and never pay for it.
    """
    if not _has_padding(cu_logical):
        return False
    return os.environ.get("JA_MHA_ZERO_PAD", "1") != "0"


def _tag_context(*vals):
    """Name attention residuals so a layer-remat policy can save them."""
    if os.environ.get("JA_MHA_REMAT_CONTEXT", "1") == "0":
        return vals if len(vals) > 1 else vals[0]
    name = os.environ.get("JA_MHA_CONTEXT_CKPT_NAME", _CONTEXT_CKPT_NAME)
    tagged = tuple(
        v if v is None else checkpoint_name(v, name) for v in vals
    )
    return tagged if len(tagged) > 1 else tagged[0]


def _resolve_bwd_dispatch(hard_block: bool, suspect_block: bool):
    """Return ``(use_asm_v3, is_v3_atomic_fp32, how_v3_bf16_cvt)``."""
    if os.environ.get("JA_MHA_BWD_USE_ASM_V3", "1") == "0" or hard_block:
        use_v3 = False
    elif suspect_block:
        use_v3 = os.environ.get("JA_MHA_BWD_FORCE_ASM_V3", "0") == "1"
    else:
        use_v3 = True

    atomic_env = os.environ.get("JA_MHA_BWD_ATOMIC_FP32")
    atomic = use_v3 if atomic_env is None else atomic_env == "1"

    cvt_env = os.environ.get("JA_MHA_BWD_BF16_CVT")
    bf16_cvt = (0 if get_gfx() == "gfx950" else 1) if cvt_env is None else int(cvt_env)

    return use_v3, atomic, bf16_cvt


# ---------------------------------------------------------------------------
# Sharding helpers
# ---------------------------------------------------------------------------

def _get_padded_spec(arg_info):
    """Pad a PartitionSpec to match ndim, filling with None."""
    if arg_info.sharding is None:
        return (None,) * arg_info.ndim
    spec = arg_info.sharding.spec
    return spec + (None,) * (arg_info.ndim - len(spec))


def _get_rank(t):
    """Get tensor rank from either JAX ShapeDtypeStruct or MLIR RankedTensorType."""
    if hasattr(t, 'ndim'):
        return t.ndim
    if hasattr(t, 'rank'):
        return t.rank
    return len(t.shape)


# ---------------------------------------------------------------------------
# custom_partitioning: forward
# ---------------------------------------------------------------------------

@partial(custom_partitioning, static_argnums=(11,))
def _mha_fwd_partitioned(q, k, v, cu_sq, cu_skv, out_prov,
                         bias, alibi, gen, cu_sq_log, cu_skv_log, config):
    return _mha_fwd_raw(q, k, v, cu_sq, cu_skv, out_prov,
                        bias, alibi, gen, cu_sq_log, cu_skv_log, config)


def _mha_fwd_infer_sharding(config, mesh, arg_shapes, result_shapes):
    q_spec = _get_padded_spec(arg_shapes[0])
    is_varlen = (arg_shapes[0].ndim == 3)

    out_sharding = NamedSharding(mesh, P(*q_spec))

    if is_varlen:
        # lse is [hq, total_q]: shard the token dim (dim1) like q's token dim
        # (q dim0) and the head dim (dim0) like q's head dim (q dim1), so the
        # partitioner-declared lse shape matches the per-shard FFI output.
        lse_sh = NamedSharding(mesh, P(q_spec[1], q_spec[0]))
    else:
        if result_shapes[1].ndim == 3:
            lse_sh = NamedSharding(mesh, P(q_spec[0], q_spec[2], q_spec[1]))
        else:
            lse_sh = NamedSharding(mesh, P(None))

    p_sh = NamedSharding(mesh, P(*((None,) * result_shapes[2].ndim)))
    rng_sh = NamedSharding(mesh, P(*((None,) * result_shapes[3].ndim)))
    return (out_sharding, lse_sh, p_sh, rng_sh)


def _mha_fwd_partition(config, mesh, arg_shapes, result_shapes):
    out_shardings = _mha_fwd_infer_sharding(config, mesh,
                                            arg_shapes, result_shapes)
    q_spec = _get_padded_spec(arg_shapes[0])
    cp_axis = config.cp_axis
    cp_active = cp_axis and config.cp_size > 1

    shardings = []
    for i, a in enumerate(arg_shapes):
        if a.shape[0] == 0:
            shardings.append(NamedSharding(mesh, P(*((None,) * a.ndim))))
        elif i == 6 and a.ndim == 2:
            shardings.append(NamedSharding(mesh, P(q_spec[1], None)))
        elif cp_active and i in (1, 2) and a.ndim == 4:
            s = _get_padded_spec(a)
            shardings.append(NamedSharding(mesh, P(s[0], None, s[2], s[3])))
        else:
            shardings.append(a.sharding)
    arg_shardings = tuple(shardings)

    def _lowered(q, k, v, cu_sq, cu_skv, out_prov, bias, alibi, gen,
                 cu_sq_log, cu_skv_log):
        return _mha_fwd_raw(q, k, v, cu_sq, cu_skv, out_prov,
                            bias, alibi, gen, cu_sq_log, cu_skv_log, config)

    return mesh, _lowered, out_shardings, arg_shardings


def _mha_fwd_shardy_rule(config, mesh, in_types, out_types):
    """Shardy sharding rule: batch dims passthrough, rest placeholders."""
    is_4d = (_get_rank(in_types[0]) == 4)
    if is_4d:
        q_spec = ("…0", "sq", "hq", "dq")
        k_spec = ("…0", "sk", "hk", "dq")
        v_spec = ("…0", "sk", "hk", "dv")
    else:
        q_spec = ("…0", "hq", "dq")
        k_spec = ("…1", "hk", "dq")
        v_spec = ("…1", "hk", "dv")
    in_spec = [q_spec, k_spec, v_spec]
    fid = 10
    for i in range(3, len(in_types)):
        if i == 6 and _get_rank(in_types[i]) == 2:
            in_spec.append(("sq", "sk"))
        else:
            in_spec.append((f"…{fid}",))
            fid += 1

    out_spec = []
    if is_4d:
        out_spec.append(("…0", "sq", "hq", "dv"))
        out_spec.append(("…0", "hq", "sq") if _get_rank(out_types[1]) == 3
                        else (f"…{fid}",))
    else:
        out_spec.append(("…0", "hq", "dv"))
        out_spec.append((f"…{fid}",))
    fid += 1
    for j in range(2, len(out_types)):
        out_spec.append((f"…{fid}",))
        fid += 1
    return SdyShardingRule(tuple(in_spec), tuple(out_spec))


_mha_fwd_partitioned.def_partition(
    _mha_fwd_partition,
    infer_sharding_from_operands=_mha_fwd_infer_sharding,
    sharding_rule=_mha_fwd_shardy_rule,
)


# ---------------------------------------------------------------------------
# custom_partitioning: backward
# ---------------------------------------------------------------------------

@partial(custom_partitioning, static_argnums=(17,))
def _mha_bwd_partitioned(dout, q, k, v, out, lse, cu_sq, cu_sk,
                         dq_ws, dk_ws, dv_ws, bias, alibi, rng, gen,
                         cu_sq_log, cu_sk_log, config):
    return _mha_bwd_raw(dout, q, k, v, out, lse, cu_sq, cu_sk,
                        dq_ws, dk_ws, dv_ws, bias, alibi, rng, gen,
                        cu_sq_log, cu_sk_log, config)


def _mha_bwd_infer_sharding(config, mesh, arg_shapes, result_shapes):
    q_spec = _get_padded_spec(arg_shapes[1])
    k_spec = _get_padded_spec(arg_shapes[2])
    v_spec = _get_padded_spec(arg_shapes[3])

    dq_sh = NamedSharding(mesh, P(*q_spec))
    dk_sh = NamedSharding(mesh, P(*k_spec))
    dv_sh = NamedSharding(mesh, P(*v_spec))
    sd_sh = NamedSharding(mesh, P(*((None,) * result_shapes[3].ndim)))
    dbias_sh = NamedSharding(mesh, P(*((None,) * result_shapes[4].ndim)))

    if result_shapes[3].ndim == 3:
        sd_sh = NamedSharding(mesh, P(q_spec[0], q_spec[2], q_spec[1]))
    elif result_shapes[3].ndim == 2:
        # varlen softmax_d [hq, total_q]: shard token dim like q dim0, head like q dim1.
        sd_sh = NamedSharding(mesh, P(q_spec[1], q_spec[0]))
    if result_shapes[4].ndim == 4:
        dbias_sh = NamedSharding(mesh, P(q_spec[0], q_spec[1], q_spec[2], None))

    return (dq_sh, dk_sh, dv_sh, sd_sh, dbias_sh)


def _mha_bwd_partition(config, mesh, arg_shapes, result_shapes):
    out_shardings = _mha_bwd_infer_sharding(config, mesh,
                                            arg_shapes, result_shapes)
    q_spec = _get_padded_spec(arg_shapes[1])
    cp_axis = config.cp_axis
    cp_active = cp_axis and config.cp_size > 1

    shardings = []
    for i, a in enumerate(arg_shapes):
        if a.shape[0] == 0:
            shardings.append(NamedSharding(mesh, P(*((None,) * a.ndim))))
        elif i == 11 and a.ndim == 2:
            shardings.append(NamedSharding(mesh, P(q_spec[1], None)))
        elif cp_active and i in (2, 3) and a.ndim == 4:
            s = _get_padded_spec(a)
            shardings.append(NamedSharding(mesh, P(s[0], None, s[2], s[3])))
        else:
            shardings.append(a.sharding)
    arg_shardings = tuple(shardings)

    def _lowered(dout, q, k, v, out, lse, cu_sq, cu_sk,
                 dq_ws, dk_ws, dv_ws, bias, alibi, rng, gen,
                 cu_sq_log, cu_sk_log):
        return _mha_bwd_raw(dout, q, k, v, out, lse, cu_sq, cu_sk,
                            dq_ws, dk_ws, dv_ws, bias, alibi, rng, gen,
                            cu_sq_log, cu_sk_log, config)

    return mesh, _lowered, out_shardings, arg_shardings


def _mha_bwd_shardy_rule(config, mesh, in_types, out_types):
    """Shardy sharding rule for backward: all independent placeholders."""
    fid = 0
    in_spec = []
    for i in range(len(in_types)):
        in_spec.append((f"…{fid}",))
        fid += 1
    out_spec = []
    for i in range(len(out_types)):
        out_spec.append((f"…{fid}",))
        fid += 1
    return SdyShardingRule(tuple(in_spec), tuple(out_spec))


_mha_bwd_partitioned.def_partition(
    _mha_bwd_partition,
    infer_sharding_from_operands=_mha_bwd_infer_sharding,
    sharding_rule=_mha_bwd_shardy_rule,
)


# ---------------------------------------------------------------------------
# Forward: single call to aiter::mha_fwd (AITER handles CK vs ASM)
# ---------------------------------------------------------------------------

def mha_fwd_unified(q, k, v, dropout_p, softmax_scale, causal,
                    wl, wr, return_lse, return_softmax,
                    bias=None, alibi_slopes=None,
                    cu_seqlens_q=None, cu_seqlens_kv=None, gen=None,
                    max_seqlen_q=-1, max_seqlen_k=-1, min_seqlen_q=0,
                    logits_soft_cap=0.0, zero_tensors=False,
                    cp_axis=None, cp_size=1, cp_load_balanced=True,
                    cu_seqlens_q_logical=None, cu_seqlens_kv_logical=None):
    """Unified forward for both batch (4D q) and varlen (3D q).

    In varlen/group mode ``cu_seqlens_*`` are cumulative *physical* offsets and
    ``cu_seqlens_*_logical`` are cumulative *logical* lengths excluding padding.
    """
    if cu_seqlens_q is None:
        cu_seqlens_q = _empty(jnp.int32)
    if cu_seqlens_kv is None:
        cu_seqlens_kv = _empty(jnp.int32)
    if cu_seqlens_q_logical is None:
        cu_seqlens_q_logical = _empty(jnp.int32)
    if cu_seqlens_kv_logical is None:
        cu_seqlens_kv_logical = _empty(jnp.int32)
    if bias is None:
        bias = _empty(q.dtype)
    if alibi_slopes is None:
        alibi_slopes = _empty(jnp.float32)
    if gen is None:
        gen = _empty(jnp.int64)

    bf16_cvt = 0 if get_gfx() == "gfx950" else 1

    # Forward uses the CK FA v3 ASM kernel by default (AITER falls back to v2 CK
    # when use_asm_v3=False or the shape is unsupported). JA_MHA_FWD_USE_ASM_V3=0
    # forces the v2 forward — used by the FAv3-vs-FAv2 forward numeric A/B
    # (AIMA-164). Default 1 preserves existing behavior.
    _fwd_use_asm_v3 = os.environ.get("JA_MHA_FWD_USE_ASM_V3", "1") != "0"

    config = MhaFwdConfig(
        dropout_p=float(dropout_p),
        softmax_scale=float(softmax_scale),
        is_causal=causal,
        wl=int(wl), wr=int(wr),
        return_lse=return_lse,
        return_randval=bool(return_softmax and dropout_p > 0),
        use_asm_v3=_fwd_use_asm_v3,
        how_v3_bf16_cvt=int(bf16_cvt),
        max_seqlen_q=int(max_seqlen_q),
        max_seqlen_k=int(max_seqlen_k),
        min_seqlen_q=int(min_seqlen_q),
        logits_soft_cap=float(logits_soft_cap),
        zero_tensors=zero_tensors,
        cp_axis=cp_axis,
        cp_size=int(cp_size) if cp_size else 1,
        cp_load_balanced=cp_load_balanced,
    )
    return _mha_fwd_partitioned(q, k, v, cu_seqlens_q, cu_seqlens_kv,
                                _empty(q.dtype), bias, alibi_slopes, gen,
                                cu_seqlens_q_logical, cu_seqlens_kv_logical,
                                config)


def mha_bwd_unified(dout, q, k, v, out, lse, dropout_p, softmax_scale,
                    causal, wl, wr, deterministic,
                    use_asm_v3, is_v3_atomic_fp32, how_v3_bf16_cvt,
                    bias=None, alibi_slopes=None, rng_state=None,
                    cu_seqlens_q=None, cu_seqlens_k=None,
                    max_seqlen_q=-1, max_seqlen_k=-1, zero_tensors=False,
                    cp_axis=None, cp_size=1, cp_load_balanced=True,
                    cu_seqlens_q_logical=None, cu_seqlens_k_logical=None):
    """Unified backward for both batch (4D q) and varlen (3D q)."""
    if cu_seqlens_q is None:
        cu_seqlens_q = _empty(jnp.int32)
    if cu_seqlens_k is None:
        cu_seqlens_k = _empty(jnp.int32)
    if cu_seqlens_q_logical is None:
        cu_seqlens_q_logical = _empty(jnp.int32)
    if cu_seqlens_k_logical is None:
        cu_seqlens_k_logical = _empty(jnp.int32)
    if bias is None:
        bias = _empty(q.dtype)
    if alibi_slopes is None:
        alibi_slopes = _empty(jnp.float32)
    if rng_state is None:
        rng_state = _empty(jnp.int64)

    config = MhaBwdConfig(
        dropout_p=float(dropout_p),
        softmax_scale=float(softmax_scale),
        is_causal=causal,
        wl=int(wl), wr=int(wr),
        deterministic=deterministic,
        use_asm_v3=use_asm_v3,
        is_v3_atomic_fp32=is_v3_atomic_fp32,
        how_v3_bf16_cvt=int(how_v3_bf16_cvt),
        max_seqlen_q=int(max_seqlen_q),
        max_seqlen_k=int(max_seqlen_k),
        zero_tensors=zero_tensors,
        cp_axis=cp_axis,
        cp_size=int(cp_size) if cp_size else 1,
        cp_load_balanced=cp_load_balanced,
    )
    results = _mha_bwd_partitioned(
        dout, q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k,
        _empty(q.dtype), _empty(q.dtype), _empty(q.dtype),
        bias, alibi_slopes, rng_state, _empty(jnp.int64),
        cu_seqlens_q_logical, cu_seqlens_k_logical,
        config)

    dq_out, dk_out, dv_out, sd_out, dbias_expanded = results
    is_varlen = (q.ndim == 3)
    if not is_varlen and bias.size > 0:
        dbias_out = jnp.sum(dbias_expanded, axis=(0, 2))
    else:
        dbias_out = dbias_expanded
    return [dq_out, dk_out, dv_out, sd_out, dbias_out]


# ---------------------------------------------------------------------------
# Simplified forward/backward dispatch (no can_impl_* logic)
# ---------------------------------------------------------------------------

def _flash_attn_forward(q, k, v, dropout_p, softmax_scale, causal,
                        wl, wr, bias, alibi_slopes,
                        return_lse, return_softmax,
                        cu_seqlens_q=None, cu_seqlens_kv=None):
    _, sk, _, _ = v.shape
    if wl >= sk: wl = -1
    if wr >= sk: wr = -1

    result = mha_fwd_unified(
        q, k, v, dropout_p, softmax_scale, causal, wl, wr,
        return_lse, return_softmax,
        bias=bias, alibi_slopes=alibi_slopes,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv)
    return result


def _flash_attn_backward(dout, q, k, v, out, lse,
                         dropout_p, softmax_scale, causal, wl, wr,
                         bias, alibi_slopes, deterministic,
                         rng_state=None):
    _, sq, hq, dq = q.shape
    _, sk, hk, _ = k.shape

    swa = (wl > 0) or (wr >= 0 and wr != -1)
    # gfx950 1-block override: sk<=256 with hd in (64,128]
    is_950_1block = (
        get_gfx() == "gfx950" and sk <= 256
        and dq > 64 and dq <= 128 and dq % 8 == 0
    )
    hard_block = (
        dropout_p > 0
        or (bias is not None and bias.size > 0)
        or swa
        or is_950_1block
    )
    suspect_block = causal and get_gfx() == "gfx950" and sq > sk

    bwd_det = False if is_950_1block else deterministic
    use_v3_bwd, bwd_atomic, bf16_cvt = _resolve_bwd_dispatch(hard_block, suspect_block)

    results = mha_bwd_unified(
        dout, q, k, v, out, lse,
        dropout_p, softmax_scale, causal, wl, wr,
        bwd_det, use_v3_bwd, bwd_atomic, bf16_cvt,
        bias=bias, alibi_slopes=alibi_slopes, rng_state=rng_state)

    return results[0], results[1], results[2], results[3], results[4]


# ---------------------------------------------------------------------------
# Public API: flash_attn_func with custom_vjp
# ---------------------------------------------------------------------------

@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 9, 10, 11, 12, 13))
def flash_attn_func(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    bias: Optional[jnp.ndarray] = None,
    alibi_slopes: Optional[jnp.ndarray] = None,
    deterministic: bool = True,
    return_lse: bool = False,
    return_attn_probs: bool = False,
    cu_seqlens_q: Optional[jnp.ndarray] = None,
    cu_seqlens_kv: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Flash attention with automatic CK/ASM v3 dispatch via AITER.

    Args:
        q: (batch, seqlen_q, nheads, headdim_q)
        k: (batch, seqlen_k, nheads_k, headdim_q)
        v: (batch, seqlen_k, nheads_k, headdim_v)
        dropout_p: Dropout probability (0.0 during eval).
        softmax_scale: Scaling factor (default: 1/sqrt(headdim_q)).
        causal: Apply causal mask (bottom-right aligned).
        window_size: (left, right) for sliding window attention.
        bias: (seqlen_q, seqlen_k) attention bias.
        alibi_slopes: (nheads,) or (batch, nheads) ALiBi slopes.
        deterministic: Use deterministic backward (slower, more memory).
        return_lse: Return log-sum-exp values.
        return_attn_probs: Return attention probabilities (testing only).
    Returns:
        out: (batch, seqlen_q, nheads, headdim_v), or tuple if return_lse/return_attn_probs.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    hd_q_og = q.shape[3]
    hd_v_og = v.shape[3]

    q_p, k_p, v_p = q, k, v
    if hd_q_og % 8 != 0:
        pad = 8 - hd_q_og % 8
        q_p = jnp.pad(q, ((0, 0), (0, 0), (0, 0), (0, pad)))
        k_p = jnp.pad(k, ((0, 0), (0, 0), (0, 0), (0, pad)))
    if hd_v_og % 8 != 0:
        pad = 8 - hd_v_og % 8
        v_p = jnp.pad(v, ((0, 0), (0, 0), (0, 0), (0, pad)))

    sk = k_p.shape[1]
    wl = -1 if window_size[0] >= sk else window_size[0]
    wr = -1 if window_size[1] >= sk else window_size[1]

    out_p, lse, s_dmask, _ = _flash_attn_forward(
        q_p, k_p, v_p, dropout_p, softmax_scale,
        causal=causal, wl=wl, wr=wr,
        bias=bias, alibi_slopes=alibi_slopes,
        return_lse=return_lse,
        return_softmax=return_attn_probs and dropout_p > 0,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv)

    out = out_p[..., :hd_v_og]
    result = [out]
    if return_lse:
        result.append(lse)
    if return_attn_probs:
        result.append(s_dmask)
    return tuple(result)


def _flash_attn_func_fwd(q, k, v,
                         dropout_p=0.0, softmax_scale=None, causal=False,
                         window_size=(-1, -1), bias=None, alibi_slopes=None,
                         deterministic=True, return_lse=False,
                         return_attn_probs=False,
                         cu_seqlens_q=None, cu_seqlens_kv=None):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    hd_q_og = q.shape[3]
    hd_v_og = v.shape[3]

    q_p, k_p, v_p = q, k, v
    if hd_q_og % 8 != 0:
        pad = 8 - hd_q_og % 8
        q_p = jnp.pad(q, ((0, 0), (0, 0), (0, 0), (0, pad)))
        k_p = jnp.pad(k, ((0, 0), (0, 0), (0, 0), (0, pad)))
    if hd_v_og % 8 != 0:
        pad = 8 - hd_v_og % 8
        v_p = jnp.pad(v, ((0, 0), (0, 0), (0, 0), (0, pad)))

    sk = k_p.shape[1]
    wl = -1 if window_size[0] >= sk else window_size[0]
    wr = -1 if window_size[1] >= sk else window_size[1]

    out_p, lse, s_dmask, rng_state = _flash_attn_forward(
        q_p, k_p, v_p, dropout_p, softmax_scale,
        causal=causal, wl=wl, wr=wr,
        bias=bias, alibi_slopes=alibi_slopes,
        return_lse=True, return_softmax=return_attn_probs and dropout_p > 0,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv)

    out_p, lse, rng_state = _tag_context(out_p, lse, rng_state)
    out = out_p[..., :hd_v_og]
    result = [out]
    if return_lse:
        result.append(lse)
    if return_attn_probs:
        result.append(s_dmask)
    result = tuple(result)

    residuals = (q_p, k_p, v_p, out_p, lse, rng_state,
                 dropout_p, softmax_scale, causal, (wl, wr),
                 bias, alibi_slopes, deterministic, hd_q_og, hd_v_og)
    return result, residuals


def _flash_attn_func_bwd(dropout_p, softmax_scale, causal, window_size,
                         deterministic, return_lse, return_attn_probs,
                         cu_seqlens_q, cu_seqlens_kv,
                         residuals, grad_outputs):
    (q_p, k_p, v_p, out_p, lse, rng_state,
     res_dp, res_scale, res_causal, res_ws,
     res_bias, res_alibi, res_det, hd_q_og, hd_v_og) = residuals

    dout = grad_outputs[0] if isinstance(grad_outputs, tuple) else grad_outputs
    if dout.shape[-1] != out_p.shape[-1]:
        pad = out_p.shape[-1] - dout.shape[-1]
        dout = jnp.pad(dout, ((0, 0), (0, 0), (0, 0), (0, pad)))

    dq_p, dk_p, dv_p, _, dbias = _flash_attn_backward(
        dout, q_p, k_p, v_p, out_p, lse,
        res_dp, res_scale, res_causal, res_ws[0], res_ws[1],
        res_bias, res_alibi, res_det, rng_state)

    dq = dq_p[..., :hd_q_og]
    dk = dk_p[..., :hd_q_og]
    dv = dv_p[..., :hd_v_og]

    return (dq, dk, dv, dbias, None)


flash_attn_func.defvjp(_flash_attn_func_fwd, _flash_attn_func_bwd)


# ---------------------------------------------------------------------------
# Varlen public API: flash_attn_varlen with custom_vjp
# ---------------------------------------------------------------------------

@partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11, 12, 13, 14))
def flash_attn_varlen(
    q: jnp.ndarray,              # [total_q, nheads, headdim]
    k: jnp.ndarray,              # [total_k, nheads_k, headdim]
    v: jnp.ndarray,              # [total_k, nheads_k, headdim_v]
    cu_seqlens_q: jnp.ndarray,   # [batch_size + 1]
    cu_seqlens_k: jnp.ndarray,   # [batch_size + 1]
    cu_seqlens_q_logical: Optional[jnp.ndarray] = None,
    cu_seqlens_k_logical: Optional[jnp.ndarray] = None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    deterministic: bool = False,
    return_lse: bool = False,
) -> jnp.ndarray:
    """Variable-length flash attention using packed sequences.

    Args:
        q: [total_q, nheads, headdim] packed query tokens.
        k: [total_k, nheads_k, headdim] packed key tokens.
        v: [total_k, nheads_k, headdim_v] packed value tokens.
        cu_seqlens_q: [batch_size+1] cumulative *physical* Q offsets, i.e.
            including any inter-segment padding (AITER ``seqstart_q_ptr``).
        cu_seqlens_k: [batch_size+1] cumulative physical KV offsets.
        cu_seqlens_q_logical: optional [batch_size+1] cumulative *logical* Q
            lengths, excluding padding (AITER ``cu_seqlen_q_ptr``). Required
            whenever the physical spans contain padding; leave ``None`` for
            tightly packed inputs.
        cu_seqlens_k_logical: optional [batch_size+1] cumulative logical KV
            lengths.
        max_seqlen_q: Maximum query sequence length.
        max_seqlen_k: Maximum key sequence length.
    Returns:
        out: [total_q, nheads, headdim_v].
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    hd_q_og = q.shape[-1]
    hd_v_og = v.shape[-1]

    q_p, k_p, v_p = q, k, v
    if hd_q_og % 8 != 0:
        pad = 8 - hd_q_og % 8
        q_p = jnp.pad(q, ((0, 0), (0, 0), (0, pad)))
        k_p = jnp.pad(k, ((0, 0), (0, 0), (0, pad)))
    if hd_v_og % 8 != 0:
        pad = 8 - hd_v_og % 8
        v_p = jnp.pad(v, ((0, 0), (0, 0), (0, pad)))

    wl = window_size[0]
    wr = window_size[1]

    out_p, lse, _, _ = mha_fwd_unified(
        q_p, k_p, v_p, dropout_p, softmax_scale, causal, wl, wr,
        return_lse=return_lse, return_softmax=False,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        zero_tensors=_zero_pad(cu_seqlens_q_logical),
        cu_seqlens_q_logical=cu_seqlens_q_logical,
        cu_seqlens_kv_logical=cu_seqlens_k_logical)

    out = out_p[..., :hd_v_og]
    if return_lse:
        return (out, lse)
    return (out,)


def _flash_attn_varlen_fwd(q, k, v, cu_seqlens_q, cu_seqlens_k,
                           cu_seqlens_q_logical, cu_seqlens_k_logical,
                           max_seqlen_q, max_seqlen_k, dropout_p,
                           softmax_scale, causal, window_size,
                           deterministic, return_lse):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    hd_q_og = q.shape[-1]
    hd_v_og = v.shape[-1]

    q_p, k_p, v_p = q, k, v
    if hd_q_og % 8 != 0:
        pad = 8 - hd_q_og % 8
        q_p = jnp.pad(q, ((0, 0), (0, 0), (0, pad)))
        k_p = jnp.pad(k, ((0, 0), (0, 0), (0, pad)))
    if hd_v_og % 8 != 0:
        pad = 8 - hd_v_og % 8
        v_p = jnp.pad(v, ((0, 0), (0, 0), (0, pad)))

    wl, wr = window_size

    out_p, lse, _, rng_state = mha_fwd_unified(
        q_p, k_p, v_p, dropout_p, softmax_scale, causal, wl, wr,
        return_lse=True, return_softmax=False,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        zero_tensors=_zero_pad(cu_seqlens_q_logical),
        cu_seqlens_q_logical=cu_seqlens_q_logical,
        cu_seqlens_kv_logical=cu_seqlens_k_logical)

    out_p, lse, rng_state = _tag_context(out_p, lse, rng_state)
    out = out_p[..., :hd_v_og]
    result = (out, lse) if return_lse else (out,)

    residuals = (q_p, k_p, v_p, out_p, lse, rng_state,
                 cu_seqlens_q, cu_seqlens_k,
                 cu_seqlens_q_logical, cu_seqlens_k_logical,
                 dropout_p, softmax_scale, causal, (wl, wr),
                 deterministic, hd_q_og, hd_v_og,
                 max_seqlen_q, max_seqlen_k)
    return result, residuals


def _flash_attn_varlen_bwd(max_seqlen_q, max_seqlen_k, dropout_p,
                           softmax_scale, causal, window_size,
                           deterministic, return_lse,
                           residuals, grad_outputs):
    (q_p, k_p, v_p, out_p, lse, rng_state,
     cu_sq, cu_sk, cu_sq_log, cu_sk_log,
     res_dp, res_scale, res_causal, res_ws,
     res_det, hd_q_og, hd_v_og,
     res_max_sq, res_max_sk) = residuals

    dout = grad_outputs[0] if isinstance(grad_outputs, tuple) else grad_outputs
    if dout.shape[-1] != out_p.shape[-1]:
        pad = out_p.shape[-1] - dout.shape[-1]
        dout = jnp.pad(dout, ((0, 0), (0, 0), (0, pad)))

    _, _, hq, dq = q_p.shape if q_p.ndim == 4 else (None, None, q_p.shape[1], q_p.shape[2])
    hk = k_p.shape[1] if k_p.ndim == 3 else k_p.shape[2]

    swa = (window_size[0] > 0) or (window_size[1] >= 0 and window_size[1] != -1)
    hard_block = res_dp > 0 or swa
    # The causal/gfx950/max_seqlen_k>256 block was retested on v0.1.19 and
    # cleared; see the dispatch notes above.
    use_v3, bwd_atomic, bf16_cvt = _resolve_bwd_dispatch(hard_block, suspect_block=False)

    results = mha_bwd_unified(
        dout, q_p, k_p, v_p, out_p, lse,
        res_dp, res_scale, res_causal, res_ws[0], res_ws[1],
        res_det, use_v3, bwd_atomic, bf16_cvt,
        rng_state=rng_state,
        cu_seqlens_q=cu_sq, cu_seqlens_k=cu_sk,
        max_seqlen_q=res_max_sq, max_seqlen_k=res_max_sk,
        zero_tensors=_zero_pad(cu_sq_log),
        cu_seqlens_q_logical=cu_sq_log, cu_seqlens_k_logical=cu_sk_log)

    dq = results[0][..., :hd_q_og]
    dk = results[1][..., :hd_q_og]
    dv = results[2][..., :hd_v_og]

    return (dq, dk, dv, None, None, None, None)


flash_attn_varlen.defvjp(_flash_attn_varlen_fwd, _flash_attn_varlen_bwd)


# ---------------------------------------------------------------------------
# Raw varlen (NO custom_partitioning) for use INSIDE shard_map.
# custom_partitioning + shard_map (manual mode) conflict; under shard_map each
# device already holds a fully-local shard, so we call the raw FFI ops directly
# (ops.mha_fwd/mha_bwd = aiter::mha_fwd/bwd) with the device-LOCAL cu_seqlens.
# Same kernel as flash_attn_varlen, just without the global partitioner.
# ---------------------------------------------------------------------------

@partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11, 12))
def flash_attn_varlen_raw(q, k, v, cu_seqlens_q, cu_seqlens_k,
                          cu_seqlens_q_logical, cu_seqlens_k_logical,
                          max_seqlen_q, max_seqlen_k,
                          dropout_p, softmax_scale, causal, window_size):
    """Varlen attention over a device-local shard, without custom_partitioning.

    ``cu_seqlens_*`` are cumulative **physical** offsets (AITER
    ``seqstart_*_ptr``); ``cu_seqlens_*_logical`` are cumulative **logical**
    lengths excluding padding (AITER ``cu_seqlen_*_ptr``). Pass ``None`` for the
    logical pair when the packing carries no padding, which is AITER's
    "group mode without padding" contract (``mha_fwd.h`` sequence-pointer
    notes).
    """
    out, _ = _favr_fwd(q, k, v, cu_seqlens_q, cu_seqlens_k,
                       cu_seqlens_q_logical, cu_seqlens_k_logical,
                       max_seqlen_q, max_seqlen_k,
                       dropout_p, softmax_scale, causal, window_size)
    return out


def _favr_fwd(q, k, v, cu_q, cu_k, cu_q_log, cu_k_log, max_sq, max_sk,
              dropout_p, softmax_scale, causal, window_size):
    bf16_cvt = 0 if get_gfx() == "gfx950" else 1
    wl, wr = window_size
    cfg = MhaFwdConfig(
        dropout_p=float(dropout_p), softmax_scale=float(softmax_scale),
        is_causal=causal, wl=int(wl), wr=int(wr),
        return_lse=True, return_randval=False,
        use_asm_v3=True, how_v3_bf16_cvt=int(bf16_cvt),
        max_seqlen_q=int(max_sq), max_seqlen_k=int(max_sk), min_seqlen_q=0,
        logits_soft_cap=0.0, zero_tensors=_zero_pad(cu_q_log),
        cp_axis=None, cp_size=1, cp_load_balanced=True)
    cu_q_log = _empty(jnp.int32) if cu_q_log is None else cu_q_log
    cu_k_log = _empty(jnp.int32) if cu_k_log is None else cu_k_log
    out, lse, _p, rng = _mha_fwd_raw(
        q, k, v, cu_q, cu_k, _empty(q.dtype), _empty(q.dtype),
        _empty(jnp.float32), _empty(jnp.int64), cu_q_log, cu_k_log, cfg)
    out, lse, rng = _tag_context(out, lse, rng)
    return out, (q, k, v, out, lse, rng, cu_q, cu_k, cu_q_log, cu_k_log)


def _favr_bwd(max_sq, max_sk, dropout_p, softmax_scale, causal, window_size,
              res, dout):
    q, k, v, out, lse, rng, cu_q, cu_k, cu_q_log, cu_k_log = res
    wl, wr = window_size
    swa = (wl > 0) or (wr >= 0 and wr != -1)
    hard_block = dropout_p > 0 or swa
    # The causal/gfx950/max_seqlen_k>256 block was retested on v0.1.19 and
    # cleared; see the dispatch notes above.
    use_v3, bwd_atomic, bf16_cvt = _resolve_bwd_dispatch(hard_block, suspect_block=False)
    cfg = MhaBwdConfig(
        dropout_p=float(dropout_p), softmax_scale=float(softmax_scale),
        is_causal=causal, wl=int(wl), wr=int(wr), deterministic=False,
        use_asm_v3=use_v3, is_v3_atomic_fp32=bwd_atomic, how_v3_bf16_cvt=int(bf16_cvt),
        max_seqlen_q=int(max_sq), max_seqlen_k=int(max_sk),
        zero_tensors=_zero_pad(cu_q_log),
        cp_axis=None, cp_size=1, cp_load_balanced=True)
    dq, dk, dv, _sd, _db = _mha_bwd_raw(
        dout, q, k, v, out, lse, cu_q, cu_k,
        _empty(q.dtype), _empty(q.dtype), _empty(q.dtype),
        _empty(q.dtype), _empty(jnp.float32), rng, _empty(jnp.int64),
        cu_q_log, cu_k_log, cfg)
    return (dq, dk, dv, None, None, None, None)


flash_attn_varlen_raw.defvjp(_favr_fwd, _favr_bwd)
