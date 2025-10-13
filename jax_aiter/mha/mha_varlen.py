# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Variable-length Multi-Head Attention for JAX with automatic kernel dispatch."""

from __future__ import annotations
import logging
from typing import Tuple, Optional, List
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ..ja_compat import dtypes
from ..ffi.registry import register_ffi_target

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("jax-aiter.mha_varlen")

# Debug mode controlled by environment variable.
DEBUG_MHA = os.getenv("DEBUG_MHA", "0") == "1"


def debug_kernel_params(kernel_name: str, tensors: dict, scalars: dict):
    """Print kernel parameters for debugging."""
    if DEBUG_MHA:
        print(f"\n[MHA-VARLEN-PARAMS] {kernel_name} - Parameter Debug:")
        print("=== TENSOR PARAMETERS ===")
        for name, tensor in tensors.items():
            if tensor is not None and hasattr(tensor, "shape"):
                print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}")
            else:
                print(f"  {name}: {tensor} (type: {type(tensor)})")

        print("=== SCALAR PARAMETERS ===")
        for name, value in scalars.items():
            print(f"  {name}: {value} (type: {type(value)})")
        print("=" * 50)


def _ensure_ffi_target_registered(target_name: str):
    """Register FFI target for ROCM backend."""
    register_ffi_target(target_name, "ROCM")


def _empty_tensor(dtype):
    """Create empty tensor with valid data pointer."""
    return jnp.zeros((0,), dtype=dtype)


def _static_float(x) -> float:
    """Convert to float for static arguments."""
    return np.float32(x)


def _static_int(x) -> int:
    """Convert to int32 for static arguments."""
    return np.int32(x)


def _normalize_window_size(
    window_size_left: int, window_size_right: int, max_seqlen_k: int
):
    """Normalize window sizes: set to -1 if >= max sequence length."""
    wl = -1 if window_size_left >= max_seqlen_k else window_size_left
    wr = -1 if window_size_right >= max_seqlen_k else window_size_right
    return wl, wr


def _cached_mha_varlen_fwd_call(
    out_shape,
    softmax_lse_shape,
    p_shape,
    rng_state_shape,
    out_dtype,
):
    """Create JIT-compiled FFI call for MHA varlen forward kernel."""
    call = jax.ffi.ffi_call(
        "MhaVarlenFwdJA",
        (
            jax.ShapeDtypeStruct(out_shape, out_dtype),
            jax.ShapeDtypeStruct(softmax_lse_shape, jnp.float32),
            jax.ShapeDtypeStruct(p_shape, jnp.uint8),
            jax.ShapeDtypeStruct(rng_state_shape, jnp.int64),
        ),
        vmap_method="broadcast_all",
    )

    def _invoke(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        out_provided,
        block_table,
        bias,
        alibi_slopes,
        gen,
        *,
        max_seqlen_q,
        max_seqlen_k,
        min_seqlen_q,
        p_dropout,
        softmax_scale,
        logits_soft_cap,
        zero_tensors,
        is_causal,
        window_size_left,
        window_size_right,
        return_softmax_lse,
        return_dropout_randval,
    ):
        result = call(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            out_provided,
            block_table,
            bias,
            alibi_slopes,
            gen,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            min_seqlen_q=min_seqlen_q,
            p_dropout=p_dropout,
            softmax_scale=softmax_scale,
            logits_soft_cap=logits_soft_cap,
            zero_tensors=zero_tensors,
            is_causal=is_causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            return_softmax_lse=return_softmax_lse,
            return_dropout_randval=return_dropout_randval,
        )

        return result

    return jax.jit(
        _invoke,
        static_argnames=(
            "max_seqlen_q",
            "max_seqlen_k",
            "min_seqlen_q",
            "p_dropout",
            "softmax_scale",
            "logits_soft_cap",
            "zero_tensors",
            "is_causal",
            "window_size_left",
            "window_size_right",
            "return_softmax_lse",
            "return_dropout_randval",
        ),
    )


def _cached_mha_varlen_bwd_call(
    dq_shape,
    dk_shape,
    dv_shape,
    softmax_d_shape,
    grad_dtype,
):
    """Create JIT-compiled FFI call for MHA varlen backward kernel."""
    call = jax.ffi.ffi_call(
        "MhaVarlenBwdJA",
        (
            jax.ShapeDtypeStruct(dq_shape, grad_dtype),
            jax.ShapeDtypeStruct(dk_shape, grad_dtype),
            jax.ShapeDtypeStruct(dv_shape, grad_dtype),
            jax.ShapeDtypeStruct(softmax_d_shape, jnp.float32),
        ),
        vmap_method="broadcast_all",
    )

    def _invoke(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        cu_seqlens_q,
        cu_seqlens_k,
        dq,
        dk,
        dv,
        alibi_slopes,
        rng_state,
        gen,
        *,
        max_seqlen_q,
        max_seqlen_k,
        p_dropout,
        softmax_scale,
        zero_tensors,
        is_causal,
        window_size_left,
        window_size_right,
        deterministic,
    ):
        result = call(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            dq,
            dk,
            dv,
            alibi_slopes,
            rng_state,
            gen,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            p_dropout=p_dropout,
            softmax_scale=softmax_scale,
            zero_tensors=zero_tensors,
            is_causal=is_causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            deterministic=deterministic,
        )

        return result

    return jax.jit(
        _invoke,
        static_argnames=(
            "max_seqlen_q",
            "max_seqlen_k",
            "p_dropout",
            "softmax_scale",
            "zero_tensors",
            "is_causal",
            "window_size_left",
            "window_size_right",
            "deterministic",
        ),
    )


def mha_varlen_fwd(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    cu_seqlens_q: jnp.ndarray,
    cu_seqlens_k: Optional[jnp.ndarray],
    max_seqlen_q: int,
    max_seqlen_k: int,
    min_seqlen_q: int,
    p_dropout: float,
    softmax_scale: float,
    logits_soft_cap: float,
    zero_tensors: bool,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    return_softmax_lse: bool,
    return_dropout_randval: bool,
    out_provided: Optional[jnp.ndarray] = None,
    block_table: Optional[jnp.ndarray] = None,
    bias: Optional[jnp.ndarray] = None,
    alibi_slopes: Optional[jnp.ndarray] = None,
    gen: Optional[jnp.ndarray] = None,
) -> List[jnp.ndarray]:
    """MHA varlen forward kernel."""
    _ensure_ffi_target_registered("MhaVarlenFwdJA")

    total_q, num_heads_q, head_size_q = q.shape
    total_k, _, head_size_v = v.shape

    # Handle optional tensors.
    if cu_seqlens_k is None:
        cu_seqlens_k = _empty_tensor(jnp.int32)
    if out_provided is None:
        out_provided = _empty_tensor(v.dtype)
    if block_table is None:
        block_table = _empty_tensor(jnp.int32)
    if bias is None:
        bias = _empty_tensor(jnp.float32)
    if alibi_slopes is None:
        alibi_slopes = _empty_tensor(jnp.float32)
    if gen is None:
        gen = _empty_tensor(jnp.int64)

    out_shape = (total_q, num_heads_q, head_size_v)
    softmax_lse_shape = (num_heads_q, total_q)
    p_shape = (num_heads_q, total_q, max_seqlen_k)
    rng_state_shape = (2,)

    fn = _cached_mha_varlen_fwd_call(
        out_shape,
        softmax_lse_shape,
        p_shape,
        rng_state_shape,
        v.dtype,  # Use v.dtype for output
    )

    results = fn(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        out_provided,
        block_table,
        bias,
        alibi_slopes,
        gen,
        max_seqlen_q=_static_int(max_seqlen_q),
        max_seqlen_k=_static_int(max_seqlen_k),
        min_seqlen_q=_static_int(min_seqlen_q),
        p_dropout=_static_float(p_dropout),
        softmax_scale=_static_float(softmax_scale),
        logits_soft_cap=_static_float(logits_soft_cap),
        zero_tensors=zero_tensors,
        is_causal=is_causal,
        window_size_left=_static_int(window_size_left),
        window_size_right=_static_int(window_size_right),
        return_softmax_lse=return_softmax_lse,
        return_dropout_randval=return_dropout_randval,
    )

    return list(results)


def mha_varlen_bwd(
    dout: jnp.ndarray,
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    out: jnp.ndarray,
    softmax_lse: jnp.ndarray,  # [batch, nheads, max_seqlen_q]
    cu_seqlens_q: jnp.ndarray,
    cu_seqlens_k: jnp.ndarray,
    max_seqlen_q: int,
    max_seqlen_k: int,
    p_dropout: float,
    softmax_scale: float,
    zero_tensors: bool,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool,
    dq: Optional[jnp.ndarray] = None,
    dk: Optional[jnp.ndarray] = None,
    dv: Optional[jnp.ndarray] = None,
    alibi_slopes: Optional[jnp.ndarray] = None,
    rng_state: Optional[jnp.ndarray] = None,
    gen: Optional[jnp.ndarray] = None,
) -> List[jnp.ndarray]:
    """MHA varlen backward kernel."""
    _ensure_ffi_target_registered("MhaVarlenBwdJA")

    total_q, num_heads, head_size_q = q.shape
    total_k, num_heads_k, _ = k.shape
    head_size_v = v.shape[-1]
    batch_size = cu_seqlens_q.shape[0] - 1

    # Handle optional tensors
    if dq is None:
        dq = _empty_tensor(q.dtype)
    if dk is None:
        dk = _empty_tensor(k.dtype)
    if dv is None:
        dv = _empty_tensor(v.dtype)
    if alibi_slopes is None:
        alibi_slopes = _empty_tensor(jnp.float32)
    if rng_state is None:
        rng_state = _empty_tensor(jnp.int64)
    if gen is None:
        gen = _empty_tensor(jnp.int64)

    # Output shapes
    dq_shape = (total_q, num_heads, head_size_q)
    dk_shape = (total_k, num_heads_k, head_size_q)
    dv_shape = (total_k, num_heads_k, head_size_v)
    softmax_d_shape = (batch_size, num_heads, max_seqlen_q)

    fn = _cached_mha_varlen_bwd_call(
        dq_shape,
        dk_shape,
        dv_shape,
        softmax_d_shape,
        q.dtype,
    )

    results = fn(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        cu_seqlens_q,
        cu_seqlens_k,
        dq,
        dk,
        dv,
        alibi_slopes,
        rng_state,
        gen,
        max_seqlen_q=_static_int(max_seqlen_q),
        max_seqlen_k=_static_int(max_seqlen_k),
        p_dropout=_static_float(p_dropout),
        softmax_scale=_static_float(softmax_scale),
        zero_tensors=zero_tensors,
        is_causal=is_causal,
        window_size_left=_static_int(window_size_left),
        window_size_right=_static_int(window_size_right),
        deterministic=deterministic,
    )

    return results


def _flash_attn_varlen_forward(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    cu_seqlens_q: jnp.ndarray,
    cu_seqlens_k: Optional[jnp.ndarray],
    max_seqlen_q: int,
    max_seqlen_k: int,
    min_seqlen_q: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    logits_soft_cap: float = 0.0,
    window_size_left: int = -1,
    window_size_right: int = -1,
    bias: Optional[jnp.ndarray] = None,
    alibi_slopes: Optional[jnp.ndarray] = None,
    return_lse: bool = False,
    return_softmax: bool = False,
    how_v3_bf16_cvt: Optional[int] = 1,
    block_table: Optional[jnp.ndarray] = None,
    out: Optional[jnp.ndarray] = None,
    zero_tensors: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Forward pass for varlen attention."""
    (_, nhead_q, hdim_q) = q.shape

    nhead_k = v.shape[-2]
    hdim_v = v.shape[-1]

    mask = causal == True and window_size_left == -1  # causal mask
    nmask = (
        causal == False and window_size_left == -1 and window_size_right == -1
    )  # no mask
    swa = (window_size_left > 0) or (window_size_right > 0)

    def can_impl_fmha_v3_fwd():
        ret = alibi_slopes is None
        ret = ret and (bias is None)
        ret = ret and (dropout_p == 0.0)
        ret = ret and (hdim_q == hdim_v)
        ret = ret and (hdim_q == 128)
        ret = ret and (nhead_q % nhead_k == 0)
        ret = ret and (not swa)
        ret = ret and (q.dtype == dtypes.bf16)
        return ret

    # if can_impl_fmha_v3_fwd():
    result = mha_varlen_fwd(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        min_seqlen_q,
        dropout_p,
        softmax_scale,
        logits_soft_cap,
        zero_tensors,
        causal,
        window_size_left,
        window_size_right,
        return_lse,
        return_softmax,
        out,
        block_table,
        bias,
        alibi_slopes,
        None,
    )

    return list(result)


def _flash_attn_varlen_backward(
    dout: jnp.ndarray,
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    out: jnp.ndarray,
    softmax_lse_3d: jnp.ndarray,  # [batch, nheads, max_seqlen_q] from forward (3D format)
    cu_seqlens_q: jnp.ndarray,
    cu_seqlens_k: jnp.ndarray,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    alibi_slopes: Optional[jnp.ndarray],
    deterministic: bool,
    rng_state: Optional[jnp.ndarray] = None,
    is_v3_atomic_fp32: Optional[bool] = True,
    how_v3_bf16_cvt: Optional[int] = 1,
    zero_tensors: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Backward pass for varlen attention."""
    # LSE is already in 3D format from forward, no conversion needed
    results = mha_varlen_bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse_3d,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        zero_tensors,
        causal,
        window_size_left,
        window_size_right,
        deterministic,
        None,  # dq
        None,  # dk
        None,  # dv
        alibi_slopes,
        rng_state,
        None,  # gen
    )

    dq_grad, dk_grad, dv_grad, softmax_d = results
    return dq_grad, dk_grad, dv_grad, softmax_d


# Public API functions with custom VJP for autodiff.
@partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17))
def flash_attn_varlen(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    cu_seqlens_q: jnp.ndarray,
    cu_seqlens_k: Optional[jnp.ndarray] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    min_seqlen_q: int = 0,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    logits_soft_cap: float = 0.0,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    bias: Optional[jnp.ndarray] = None,
    alibi_slopes: Optional[jnp.ndarray] = None,
    deterministic: bool = False,
    return_lse: bool = False,
    return_attn_probs: bool = False,
    block_table: Optional[jnp.ndarray] = None,
    out: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Flash Attention function for variable-length sequences.

    Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Args:
        q: Query tensor [total_q, nheads, headdim_q], where total_q = total number of query tokens in the batch
        k: Key tensor [total_k, nheads_k, headdim_q], where total_k = total number of key tokens in the batch
        v: Value tensor [total_k, nheads_k, headdim_v], where total_k = total number of key tokens in the batch
        cu_seqlens_q: Cumulative sequence lengths for queries [batch_size + 1], dtype int32
        cu_seqlens_k: Cumulative sequence lengths for keys [batch_size + 1], dtype int32. If None, uses cu_seqlens_q
        max_seqlen_q: Maximum query sequence length in the batch (computed if None)
        max_seqlen_k: Maximum key sequence length in the batch (computed if None)
        min_seqlen_q: Minimum query sequence length for chunked prefill (default: 0)
        dropout_p: Dropout probability (default: 0.0)
        softmax_scale: Scaling factor for softmax. Default to 1 / sqrt(headdim_q)
        logits_soft_cap: Soft capping value for logits (default: 0.0)
        causal: Whether to apply causal attention mask (e.g., for auto-regressive modeling)
        window_size: (left, right). If not (-1, -1), implements sliding window local attention
        bias: Optional bias tensor [seqlen_q, seqlen_k]
        alibi_slopes: Optional ALiBi slopes [nheads] or [batch_size, nheads], fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|) is added to the attention score of query i and key j
        deterministic: Whether to use deterministic algorithms. The forward pass is always deterministic
        return_lse: Whether to return the log-sum-exp values
        return_attn_probs: Whether to return the attention probabilities. This option is for
            testing only. The returned probabilities are not guaranteed to be correct
            (they might not have the right scaling)
        block_table: Optional block table for paged attention
        out: Optional output tensor. If provided, the output will be written to this tensor.
            Note: This is provided for API compatibility but is not commonly used in JAX

    Returns:
        out: [total, nheads, headdim_v]
        softmax_lse [optional, if return_lse=True]: [nheads, total_q]. The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor)
        S_dmask [optional, if return_attn_probs=True]: [nheads, total_q, max_seqlen_k].
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
            Only returned when dropout_p > 0
    """
    # Set default softmax scale.
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    head_size_q_og = q.shape[2]
    head_size_v_og = v.shape[2]

    # Pad head dimensions to multiples of 8.
    q_padded, k_padded, v_padded = q, k, v
    if head_size_q_og % 8 != 0:
        pad_q = 8 - head_size_q_og % 8
        q_padded = jnp.pad(q, ((0, 0), (0, 0), (0, pad_q)))
        k_padded = jnp.pad(k, ((0, 0), (0, 0), (0, pad_q)))
    if head_size_v_og % 8 != 0:
        pad_v = 8 - head_size_v_og % 8
        v_padded = jnp.pad(v, ((0, 0), (0, 0), (0, pad_v)))

    wl_norm, wr_norm = _normalize_window_size(
        window_size[0], window_size[1], max_seqlen_k
    )

    # Call forward kernel.
    out_padded, softmax_lse_3d, S_dmask, rng_state = _flash_attn_varlen_forward(
        q_padded,
        k_padded,
        v_padded,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        min_seqlen_q,
        dropout_p,
        softmax_scale,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
        window_size_left=wl_norm,
        window_size_right=wr_norm,
        bias=bias,
        alibi_slopes=alibi_slopes,
        return_lse=return_lse,
        return_softmax=return_attn_probs and dropout_p > 0,
        how_v3_bf16_cvt=1,  # Default value, matching AITER
        block_table=block_table,
        out=out,
        zero_tensors=False,  # Keep internal, use default value
    )

    # Unpad output to original dimensions.
    out = out_padded[..., :head_size_v_og]

    # Always return a fixed triple; let caller ignore Nones.
    lse_ret = softmax_lse_3d if return_lse else None
    s_ret = S_dmask if (return_attn_probs and dropout_p > 0) else None
    return (out, lse_ret, s_ret)


def _flash_attn_varlen_fwd(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    cu_seqlens_q: jnp.ndarray,
    cu_seqlens_k: Optional[jnp.ndarray] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    min_seqlen_q: int = 0,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    logits_soft_cap: float = 0.0,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    bias: Optional[jnp.ndarray] = None,
    alibi_slopes: Optional[jnp.ndarray] = None,
    deterministic: bool = False,
    return_lse: bool = False,
    return_attn_probs: bool = False,
    block_table: Optional[jnp.ndarray] = None,
    out: Optional[jnp.ndarray] = None,
    is_v3_atomic_fp32: Optional[bool] = True,
    how_v3_bf16_cvt: Optional[int] = 1,
    zero_tensors: bool = False,
):
    """Forward pass that returns both output and residuals for backward pass."""
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    head_size_q_og = q.shape[2]
    head_size_v_og = v.shape[2]

    # Pad head dimensions to multiples of 8
    q_padded, k_padded, v_padded = q, k, v
    if head_size_q_og % 8 != 0:
        pad_q = 8 - head_size_q_og % 8
        q_padded = jnp.pad(q, ((0, 0), (0, 0), (0, pad_q)))
        k_padded = jnp.pad(k, ((0, 0), (0, 0), (0, pad_q)))
    if head_size_v_og % 8 != 0:
        pad_v = 8 - head_size_v_og % 8
        v_padded = jnp.pad(v, ((0, 0), (0, 0), (0, pad_v)))

    # Parent-level normalization consistent with mha.py
    wl_norm, wr_norm = _normalize_window_size(
        window_size[0], window_size[1], max_seqlen_k
    )

    # Pass normalized concrete values to forward
    out_padded, softmax_lse_3d, S_dmask, rng_state = _flash_attn_varlen_forward(
        q_padded,
        k_padded,
        v_padded,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        min_seqlen_q,
        dropout_p,
        softmax_scale,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
        window_size_left=wl_norm,
        window_size_right=wr_norm,
        bias=bias,
        alibi_slopes=alibi_slopes,
        return_lse=True,  # Always return for backward
        return_softmax=return_attn_probs and dropout_p > 0,
        how_v3_bf16_cvt=how_v3_bf16_cvt,
        block_table=block_table,
        out=out,
        zero_tensors=zero_tensors,
    )

    out = out_padded[..., :head_size_v_og]

    lse_ret = softmax_lse_3d if return_lse else None
    s_ret = S_dmask if (return_attn_probs and dropout_p > 0) else None
    result = (out, lse_ret, s_ret)

    # Residuals needed for backward pass
    residuals = (
        q_padded,
        k_padded,
        v_padded,
        out_padded,
        softmax_lse_3d,  # Keep 3D format
        rng_state,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        logits_soft_cap,
        zero_tensors,
        causal,
        (wl_norm, wr_norm),
        bias,
        alibi_slopes,
        block_table,
        deterministic,
        head_size_q_og,
        head_size_v_og,
        is_v3_atomic_fp32,
        how_v3_bf16_cvt,
    )

    return result, residuals


def _flash_attn_varlen_bwd(
    max_seqlen_q,
    max_seqlen_k,
    min_seqlen_q,
    dropout_p,
    softmax_scale,
    logits_soft_cap,
    causal,
    window_size,
    deterministic,
    return_lse,
    return_attn_probs,
    residuals,
    grad_outputs,
):
    """Backward pass using residuals and output gradients."""
    # Unpack grad_outputs (cotangent of the output)
    if isinstance(grad_outputs, (tuple, list)):
        dout = grad_outputs[0]
    else:
        dout = grad_outputs

    # Unpack residuals
    (
        q_padded,
        k_padded,
        v_padded,
        out_padded,
        softmax_lse_3d,
        rng_state,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        res_dropout_p,
        res_softmax_scale,
        res_logits_soft_cap,
        res_zero_tensors,
        res_causal,
        res_window_size,
        res_bias,
        res_alibi_slopes,
        res_block_table,
        res_deterministic,
        head_size_q_og,
        head_size_v_og,
        is_v3_atomic_fp32,
        how_v3_bf16_cvt,
    ) = residuals

    dout_padded = dout
    if head_size_v_og % 8 != 0:
        pad_v = 8 - head_size_v_og % 8
        dout_padded = jnp.pad(dout, ((0, 0), (0, 0), (0, pad_v)))

    # Call backward function
    dq_padded, dk_padded, dv_padded, softmax_d = _flash_attn_varlen_backward(
        dout=dout_padded,
        q=q_padded,
        k=k_padded,
        v=v_padded,
        out=out_padded,
        softmax_lse_3d=softmax_lse_3d,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=res_dropout_p,
        softmax_scale=res_softmax_scale,
        causal=res_causal,
        window_size_left=res_window_size[0],
        window_size_right=res_window_size[1],
        alibi_slopes=res_alibi_slopes,
        deterministic=res_deterministic,
        rng_state=rng_state,
        is_v3_atomic_fp32=is_v3_atomic_fp32,
        how_v3_bf16_cvt=how_v3_bf16_cvt,
        zero_tensors=res_zero_tensors,
    )

    # Unpad gradients to match original input dimensions
    dq = dq_padded[..., :head_size_q_og]
    dk = dk_padded[..., :head_size_q_og]
    dv = dv_padded[..., :head_size_v_og]

    # Return gradients for all inputs (None for non-differentiable params)
    return (
        dq,  # q
        dk,  # k
        dv,  # v
        None,  # cu_seqlens_q
        None,  # cu_seqlens_k
        None,  # bias (no gradient computation for varlen)
        None,  # alibi_slopes
        None,  # block_table
        None,  # out
    )


# Register the custom VJP
flash_attn_varlen.defvjp(_flash_attn_varlen_fwd, _flash_attn_varlen_bwd)
