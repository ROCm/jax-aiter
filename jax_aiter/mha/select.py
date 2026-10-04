# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Per-direction backend selection for causal MHA.

The AITER unified entry point chooses ASM v3 or CK inside AITER. Where AITER
has no ASM backward for a configuration it runs CK, and for some of those
configurations the AITER Triton one-kernel backward is faster. The functions
here keep the AITER forward, so its ASM kernels stay in use, and choose the
backward implementation per call.

Measured on gfx950 at the DeepSeek 671B-6L per-device attention shape
(docs/runs/deepseek_v3/20260924_mha_backend_crossover/):

* Full-length rows reach the ASM forward and backward in batch mode with
  ``deterministic=False``; deterministic batch mode runs CK.
* AITER has no group-mode ASM backward for QK=192 / V=128. The group-mode
  backward in ``mha.py`` zero-extends V, O and dO to 192 to reach the 192/192
  ASM chain (``JA_MHA_VARLEN_BWD_PAD_V=0`` keeps CK).
* In group mode AITER's backward beat the Triton one-kernel backward on every
  layout except many short segments (about 16 or more per 4096-token row).
  Segment counts are data, not shape, so the trace-time default stays on AITER
  and Triton is an explicit choice.
"""

from __future__ import annotations

import os
from functools import partial

import jax
import jax.numpy as jnp

from ..ja_compat.chip_info import get_gfx

_BACKEND_ENV = "JA_MHA_VARLEN_BWD_BACKEND"
_BACKENDS = ("auto", "aiter", "triton")


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def triton_varlen_bwd_unsupported_reason(
    *, q_dtype, hd_qk, hd_v, causal, dropout_p, window_size
):
    """Why the Triton backward cannot serve this call, or ``None`` if it can."""
    if get_gfx() != "gfx950":
        return f"compiled for gfx950 only, running on {get_gfx()}"
    if q_dtype not in (jnp.bfloat16, jnp.float16):
        return f"dtype {q_dtype} is not bf16/fp16"
    if not causal:
        return "causal attention only"
    if dropout_p:
        return "no dropout"
    if tuple(window_size) != (-1, -1):
        return "no sliding window"
    pe = hd_qk - hd_v
    if pe < 0:
        return "QK head dim must be >= V head dim"
    if pe and not (_is_pow2(hd_v) and _is_pow2(pe)):
        return "QK/V head-dim split must be power-of-two NOPE and PE dims"
    return None


def varlen_bwd_backend(*, q_dtype, hd_qk, hd_v, causal, dropout_p, window_size):
    """``"aiter"`` (ASM, else CK) or ``"triton"`` for a group-mode backward.

    ``JA_MHA_VARLEN_BWD_BACKEND=aiter|triton`` forces a backend; ``auto`` (the
    default) applies the measured policy described in the module docstring.
    """
    choice = (os.environ.get(_BACKEND_ENV) or "auto").strip().lower()
    if choice not in _BACKENDS:
        raise ValueError(f"{_BACKEND_ENV}={choice!r}; expected one of {_BACKENDS}")
    if choice != "triton":
        return "aiter"
    reason = triton_varlen_bwd_unsupported_reason(
        q_dtype=q_dtype, hd_qk=hd_qk, hd_v=hd_v, causal=causal,
        dropout_p=dropout_p, window_size=window_size,
    )
    if reason:
        raise ValueError(f"{_BACKEND_ENV}=triton cannot serve this call: {reason}")
    return "triton"


def full_row_layout(*, q_dtype, hd_qk, hd_v, causal, dropout_p=0.0, window_size=(-1, -1)):
    """``"batch"`` or ``"varlen"`` for full-length rows without padding.

    Batch mode reaches AITER's ASM kernels in both directions. The Triton
    backward exists only in group mode, so forcing it sends full rows through
    one tight varlen segment per row.
    """
    backend = varlen_bwd_backend(
        q_dtype=q_dtype, hd_qk=hd_qk, hd_v=hd_v, causal=causal,
        dropout_p=dropout_p, window_size=window_size,
    )
    return "varlen" if backend == "triton" else "batch"


def triton_varlen_layout(seqstart, cu_seqlens_logical, total):
    """Map an AITER group-mode layout onto Triton's single offset array.

    AITER describes a padded packing with physical segment starts
    (``seqstart``) plus cumulative logical lengths, so it can skip padding in
    place. Triton's varlen kernels take one cumulative array whose segments
    tile a contiguous range, so every padding run becomes its own segment.
    Those segments are isolated from the real ones and their rows are masked
    out, which costs compute proportional to the square of each hole.

    Returns ``(cu_seqlens, valid)``. ``valid`` is ``None`` for tight input,
    where every physical token is real.
    """
    seqstart = seqstart.astype(jnp.int32)
    if cu_seqlens_logical is None or cu_seqlens_logical.size == 0:
        return seqstart, None

    lengths = jnp.diff(cu_seqlens_logical.astype(jnp.int32))
    starts = seqstart[:-1]
    ends = starts + lengths
    cu = jnp.concatenate(
        [jnp.stack([starts, ends], axis=1).reshape(-1), seqstart[-1:]]
    )

    # Even regions [cu[2i], cu[2i+1]) are real tokens. Tokens before cu[0] or
    # at/after cu[-1] are outside every Triton segment and never written.
    pos = jnp.arange(total, dtype=jnp.int32)
    region = jnp.searchsorted(cu, pos, side="right") - 1
    valid = (region >= 0) & (region < cu.shape[0] - 1) & (region % 2 == 0)
    return cu, valid


def triton_bwd_from_aiter_forward(
    dout, q, k, v, out, lse, seqstart, cu_seqlens_logical,
    max_seqlen_q, max_seqlen_k, softmax_scale,
):
    """Triton one-kernel backward fed by an AITER unified varlen forward.

    ``lse`` is AITER's head-major ``[Hq, total_q]`` natural-log LSE. Gradients
    of padding rows are zero, matching AITER's ``zero_tensors`` contract.
    """
    from ..triton.attention.varlen import _bwd

    cu, valid = triton_varlen_layout(seqstart, cu_seqlens_logical, q.shape[0])
    dq, dk, dv = _bwd(
        q, k, v, out, lse, cu, cu, dout,
        max_seqlen_q, max_seqlen_k, softmax_scale,
        lse_head_major=True,
    )
    if valid is not None:
        mask = valid[:, None, None]
        dq = jnp.where(mask, dq, jnp.zeros_like(dq))
        dk = jnp.where(mask, dk, jnp.zeros_like(dk))
        dv = jnp.where(mask, dv, jnp.zeros_like(dv))
    return dq, dk, dv


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def flash_attn_auto(q, k, v, *, softmax_scale, causal=True):
    """Attention over full-length rows ``[B, S, H, D]``, no padding.

    Batch mode with ``deterministic=False`` is the layout that reaches AITER's
    ASM backward for these rows (see the module docstring).
    """
    from .mha import flash_attn_func

    return flash_attn_func(
        q, k, v, softmax_scale=softmax_scale, causal=causal, deterministic=False
    )[0]


def _backend_for(q, v, dropout_p, causal, window_size):
    return varlen_bwd_backend(
        q_dtype=q.dtype, hd_qk=q.shape[-1], hd_v=v.shape[-1], causal=causal,
        dropout_p=dropout_p, window_size=window_size,
    )


@partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11, 12))
def flash_attn_varlen_auto(
    q, k, v, seqstart_q, seqstart_k, cu_seqlens_q_logical, cu_seqlens_k_logical,
    max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal, window_size,
):
    """Device-local group-mode attention with a per-direction backend.

    Same contract as :func:`jax_aiter.mha.flash_attn_varlen_raw`: call it
    inside ``shard_map`` with device-local metadata. ``seqstart_*`` are
    cumulative physical offsets and ``cu_seqlens_*_logical`` cumulative
    logical lengths, or ``None`` for tight packing. The forward always runs
    AITER unified; the backward follows :func:`varlen_bwd_backend`. The Triton
    backward requires self-attention with identical Q and K layouts.
    """
    out, _ = _fava_fwd(
        q, k, v, seqstart_q, seqstart_k, cu_seqlens_q_logical, cu_seqlens_k_logical,
        max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal, window_size,
    )
    return out


def _fava_fwd(q, k, v, seqstart_q, seqstart_k, cu_q_log, cu_k_log,
              max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal, window_size):
    from .mha import _favr_fwd

    if _backend_for(q, v, dropout_p, causal, window_size) == "triton":
        if seqstart_q.shape != seqstart_k.shape or q.shape[0] != k.shape[0]:
            raise ValueError("Triton varlen backward requires identical Q/K layouts")
    return _favr_fwd(
        q, k, v, seqstart_q, seqstart_k, cu_q_log, cu_k_log,
        max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal, window_size,
    )


def _fava_bwd(max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal,
              window_size, res, dout):
    from .mha import _favr_bwd

    q, k, v, out, lse, _rng, seqstart_q, _seqstart_k, cu_q_log, _cu_k_log = res
    if _backend_for(q, v, dropout_p, causal, window_size) == "aiter":
        return _favr_bwd(max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale,
                         causal, window_size, res, dout)
    dq, dk, dv = triton_bwd_from_aiter_forward(
        dout, q, k, v, out, lse, seqstart_q, cu_q_log,
        max_seqlen_q, max_seqlen_k, softmax_scale,
    )
    return (dq, dk, dv, None, None, None, None)


flash_attn_varlen_auto.defvjp(_fava_fwd, _fava_bwd)
