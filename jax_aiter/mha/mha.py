# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Multi-Head Attention with automatic kernel dispatch for optimal performance."""

from __future__ import annotations
import logging
from typing import Tuple, Optional, List
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import core

from ..ja_compat import dtypes
from ..ja_compat.chip_info import get_gfx
from ..ffi.registry import register_ffi_target

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("jax-aiter.mha")

# Debug mode controlled by environment variable.
DEBUG_MHA = os.getenv("DEBUG_MHA", "0") == "1"


def debug_kernel_params(kernel_name: str, tensors: dict, scalars: dict):
    """Print kernel parameters for debugging."""
    if DEBUG_MHA:
        print(f"\n[MHA-PARAMS] {kernel_name} - Parameter Debug:")
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


def _static_is_zero(x) -> bool:
    """Convert to Python bool for (x == 0) without tracer issues."""
    return np.float32(x) == 0.0


def _static_float(x) -> np.float32:
    """Convert to float for static arguments."""
    return np.float32(x)


def _static_int(x) -> np.int32:
    """Convert to int32 for static arguments."""
    return np.int32(x)


def _normalize_window_size(
    window_size_left: int, window_size_right: int, seqlen_k: int
):
    """Normalize window sizes: set to -1 if >= sequence length."""
    wl = -1 if window_size_left >= seqlen_k else window_size_left
    wr = -1 if window_size_right >= seqlen_k else window_size_right
    return wl, wr


def _create_generator_tensor(dropout_p: float) -> Optional[jnp.ndarray]:
    """Create RNG generator tensor for dropout with fixed seed."""
    if dropout_p <= 0.0:
        return None
    return jnp.array([123, np.int64(0)], dtype=jnp.int64)


def _can_impl_fmha_v3_fwd(
    q,
    k,
    v,
    dropout_p,
    seqlen_q,
    seqlen_k,
    hdim_q,
    hdim_v,
    nhead_q,
    nhead_k,
    alibi_slopes,
    bias,
    window_size_left,
    window_size_right,
    return_lse,
):
    """Check if FMHA v3 forward kernel can be used."""
    # Check for sliding window attention.
    concrete_wl = _static_int(window_size_left)
    concrete_wr = _static_int(window_size_right)
    swa = (concrete_wl > 0) or (concrete_wr > 0)

    # Basic constraints (matching aiter exactly)
    gfx = get_gfx()
    ret = alibi_slopes is None
    ret = ret and (bias is None)
    ret = ret and _static_is_zero(dropout_p)
    ret = ret and (hdim_q == hdim_v)
    ret = ret and (hdim_q == 128)
    ret = ret and (nhead_q % nhead_k == 0)
    ret = ret and (not swa)
    ret = ret and (q.dtype == dtypes.bf16)
    return ret


def _can_impl_fmha_v3_bwd(
    dout,
    q,
    k,
    v,
    dk,
    dv,
    dbias,
    dropout_p,
    causal,
    window_size_left,
    window_size_right,
    bias,
    alibi_slopes,
    deterministic,
    is_v3_atomic_fp32=True,
):
    """Check if FMHA v3 backward kernel can be used.

    Note: v3 backward does NOT support bias gradient computation.
    """
    _, seqlen_q, nhead_q, hdim_q = q.shape
    _, seqlen_k, nhead_k, hdim_v = v.shape

    # Check for sliding window attention.
    concrete_wl = _static_int(window_size_left)
    concrete_wr = _static_int(window_size_right)
    swa = (concrete_wl > 0) or (concrete_wr > 0)

    # Same constraints as forward for consistency.
    gfx = get_gfx()
    ret = alibi_slopes is None
    # Reject v3 backward if bias is present since v3 cannot compute bias gradients.
    ret = ret and (bias is None)
    ret = ret and _static_is_zero(dropout_p)
    ret = ret and (hdim_q == hdim_v)
    ret = ret and (nhead_q % nhead_k == 0)
    ret = ret and (not deterministic)

    return ret


def _cached_mha_fwd_call(
    out_shape,
    softmax_lse_shape,
    p_shape,
    rng_state_shape,
    out_dtype,
):
    """Create JIT-compiled FFI call for standard MHA forward kernel."""
    call = jax.ffi.ffi_call(
        "MhaFwdJA",
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
        out_provided,
        bias,
        alibi_slopes,
        gen,
        *,
        dropout_p,
        softmax_scale,
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
            out_provided,
            bias,
            alibi_slopes,
            gen,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
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
            "dropout_p",
            "softmax_scale",
            "is_causal",
            "window_size_left",
            "window_size_right",
            "return_softmax_lse",
            "return_dropout_randval",
        ),
    )


def _cached_fmha_v3_fwd_call(
    out_shape,
    softmax_lse_shape,
    p_shape,
    rng_state_shape,
    out_dtype,
):
    """Create JIT-compiled FFI call for FMHA v3 forward kernel."""
    call = jax.ffi.ffi_call(
        "FmhaV3FwdJA",
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
        out_provided,
        bias,
        alibi_slopes,
        gen,
        *,
        dropout_p,
        softmax_scale,
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
            out_provided,
            bias,
            alibi_slopes,
            gen,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
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
            "dropout_p",
            "softmax_scale",
            "is_causal",
            "window_size_left",
            "window_size_right",
            "return_softmax_lse",
            "return_dropout_randval",
        ),
    )


def _cached_mha_bwd_call(
    dq_shape,
    dk_shape,
    dv_shape,
    softmax_d_shape,
    grad_dtype,
):
    """Create JIT-compiled FFI call for standard MHA backward kernel."""
    call = jax.ffi.ffi_call(
        "MhaBwdJA",
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
        dq,
        dk,
        dv,
        dbias,
        bias,
        alibi_slopes,
        rng_state,
        gen,
        *,
        dropout_p,
        softmax_scale,
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
            dq,
            dk,
            dv,
            dbias,
            bias,
            alibi_slopes,
            rng_state,
            gen,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            is_causal=is_causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            deterministic=deterministic,
        )

        return result

    return jax.jit(
        _invoke,
        static_argnames=(
            "dropout_p",
            "softmax_scale",
            "is_causal",
            "window_size_left",
            "window_size_right",
            "deterministic",
        ),
    )


def _cached_fmha_v3_bwd_call(
    dq_shape,
    dk_shape,
    dv_shape,
    softmax_d_shape,
    grad_dtype,
):
    """Create JIT-compiled FFI call for FMHA v3 backward kernel."""
    call = jax.ffi.ffi_call(
        "FmhaV3BwdJA",
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
        dq,
        dk,
        dv,
        alibi_slopes,
        rng_state,
        gen,
        *,
        dropout_p,
        softmax_scale,
        is_causal,
        window_size_left,
        window_size_right,
        deterministic,
        is_v3_atomic_fp32,
        how_v3_bf16_cvt,
    ):
        result = call(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            alibi_slopes,
            rng_state,
            gen,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            is_causal=is_causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            deterministic=deterministic,
            is_v3_atomic_fp32=is_v3_atomic_fp32,
            how_v3_bf16_cvt=how_v3_bf16_cvt,
        )

        return result

    return jax.jit(
        _invoke,
        static_argnames=(
            "dropout_p",
            "softmax_scale",
            "is_causal",
            "window_size_left",
            "window_size_right",
            "deterministic",
            "is_v3_atomic_fp32",
            "how_v3_bf16_cvt",
        ),
    )


def mha_fwd(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    dropout_p: float,
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    return_softmax_lse: bool,
    return_dropout_randval: bool,
    out: Optional[jnp.ndarray] = None,
    bias: Optional[jnp.ndarray] = None,
    alibi_slopes: Optional[jnp.ndarray] = None,
    gen: Optional[jnp.ndarray] = None,
) -> List[jnp.ndarray]:
    """Standard MHA forward kernel with support for bias and dropout."""
    _ensure_ffi_target_registered("MhaFwdJA")

    batch_size, seqlen_q, num_heads, head_size_q = q.shape
    _, seqlen_k, num_heads_k, head_size_v = (
        k.shape[0],
        k.shape[1],
        k.shape[2],
        v.shape[3],
    )

    # Handle optional tensors
    if out is None:
        out = _empty_tensor(q.dtype)
    if bias is None:
        bias = _empty_tensor(jnp.float32)
    if alibi_slopes is None:
        alibi_slopes = _empty_tensor(jnp.float32)
    if gen is None:
        gen = _empty_tensor(jnp.int64)

    # Output shapes for MhaFwdJA.
    out_shape = (batch_size, seqlen_q, num_heads, head_size_v)
    softmax_lse_shape = (batch_size, num_heads, seqlen_q)
    p_shape = (batch_size, num_heads, seqlen_q, seqlen_k)
    rng_state_shape = (2,)

    fn = _cached_mha_fwd_call(
        out_shape,
        softmax_lse_shape,
        p_shape,
        rng_state_shape,
        q.dtype,
    )

    results = fn(
        q,
        k,
        v,
        out,
        bias,
        alibi_slopes,
        gen,
        dropout_p=_static_float(dropout_p),
        softmax_scale=_static_float(softmax_scale),
        is_causal=is_causal,
        window_size_left=_static_int(window_size_left),
        window_size_right=_static_int(window_size_right),
        return_softmax_lse=return_softmax_lse,
        return_dropout_randval=return_dropout_randval,
    )

    # Convert tuple to list for consistency.
    return list(results)


def fmha_v3_fwd(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    dropout_p: float,
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    return_softmax_lse: bool,
    return_dropout_randval: bool,
    out: Optional[jnp.ndarray] = None,
    bias: Optional[jnp.ndarray] = None,
    alibi_slopes: Optional[jnp.ndarray] = None,
    gen: Optional[jnp.ndarray] = None,
) -> List[jnp.ndarray]:
    """FMHA v3 forward kernel optimized for bf16, head_dim=128, no bias/dropout."""
    _ensure_ffi_target_registered("FmhaV3FwdJA")

    batch_size, seqlen_q, num_heads_q, head_size_q = q.shape
    _, seqlen_k, num_heads_v, head_size_v = v.shape
    _, seqlen_k, num_heads_k, _ = k.shape

    # Handle optional tensors
    if out is None:
        out = _empty_tensor(q.dtype)
    if bias is None:
        bias = _empty_tensor(jnp.float32)
    if alibi_slopes is None:
        alibi_slopes = _empty_tensor(jnp.float32)
    if gen is None:
        gen = _empty_tensor(jnp.int64)

    # Output shapes - always allocate full tensors to avoid null buffer issues.
    out_shape = (batch_size, seqlen_q, num_heads_q, head_size_v)
    softmax_lse_shape = (batch_size, num_heads_q, seqlen_q)
    p_shape = (batch_size, num_heads_q, seqlen_q, seqlen_k)
    rng_state_shape = (2,)

    fn = _cached_fmha_v3_fwd_call(
        out_shape,
        softmax_lse_shape,
        p_shape,
        rng_state_shape,
        q.dtype,
    )

    results = fn(
        q,
        k,
        v,
        out,
        bias,
        alibi_slopes,
        gen,
        dropout_p=_static_float(dropout_p),
        softmax_scale=_static_float(softmax_scale),
        is_causal=is_causal,
        window_size_left=_static_int(window_size_left),
        window_size_right=_static_int(window_size_right),
        return_softmax_lse=return_softmax_lse,
        return_dropout_randval=return_dropout_randval,
    )

    return list(results)


def mha_bwd(
    dout: jnp.ndarray,
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    out: jnp.ndarray,
    softmax_lse: jnp.ndarray,
    dropout_p: float,
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool,
    dq: Optional[jnp.ndarray] = None,
    dk: Optional[jnp.ndarray] = None,
    dv: Optional[jnp.ndarray] = None,
    dbias: Optional[jnp.ndarray] = None,
    bias: Optional[jnp.ndarray] = None,
    alibi_slopes: Optional[jnp.ndarray] = None,
    rng_state: Optional[jnp.ndarray] = None,
    gen: Optional[jnp.ndarray] = None,
) -> List[jnp.ndarray]:
    """Standard MHA backward kernel with support for bias gradients."""
    _ensure_ffi_target_registered("MhaBwdJA")

    batch_size, seqlen_q, num_heads, head_size_q = q.shape
    _, seqlen_k, num_heads_k, _ = k.shape
    head_size_v = v.shape[-1]

    # Handle optional tensors
    if dq is None:
        dq = _empty_tensor(q.dtype)
    if dk is None:
        dk = _empty_tensor(k.dtype)
    if dv is None:
        dv = _empty_tensor(v.dtype)
    if dbias is None:
        dbias = _empty_tensor(jnp.float32)
    if bias is None:
        bias = _empty_tensor(jnp.float32)
    if alibi_slopes is None:
        alibi_slopes = _empty_tensor(jnp.float32)
    if rng_state is None:
        rng_state = _empty_tensor(jnp.int64)
    if gen is None:
        gen = _empty_tensor(jnp.int64)

    # Output shapes.
    dq_shape = (batch_size, seqlen_q, num_heads, head_size_q)
    dk_shape = (batch_size, seqlen_k, num_heads_k, head_size_q)
    dv_shape = (batch_size, seqlen_k, num_heads_k, head_size_v)
    softmax_d_shape = (batch_size, num_heads, seqlen_q)

    fn = _cached_mha_bwd_call(
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
        dq,
        dk,
        dv,
        dbias,
        bias,
        alibi_slopes,
        rng_state,
        gen,
        dropout_p=_static_float(dropout_p),
        softmax_scale=_static_float(softmax_scale),
        is_causal=is_causal,
        window_size_left=_static_int(window_size_left),
        window_size_right=_static_int(window_size_right),
        deterministic=deterministic,
    )

    return results


def fmha_v3_bwd(
    dout: jnp.ndarray,
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    out: jnp.ndarray,
    softmax_lse: jnp.ndarray,
    dropout_p: float,
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool,
    is_v3_atomic_fp32: bool,
    how_v3_bf16_cvt: int,
    dq: Optional[jnp.ndarray] = None,
    dk: Optional[jnp.ndarray] = None,
    dv: Optional[jnp.ndarray] = None,
    dbias: Optional[jnp.ndarray] = None,
    bias: Optional[jnp.ndarray] = None,
    alibi_slopes: Optional[jnp.ndarray] = None,
    rng_state: Optional[jnp.ndarray] = None,
    gen: Optional[jnp.ndarray] = None,
) -> List[jnp.ndarray]:
    """FMHA v3 backward kernel optimized for bf16, no bias gradient support."""
    _ensure_ffi_target_registered("FmhaV3BwdJA")

    batch_size, seqlen_q, num_heads, head_size_q = q.shape
    _, seqlen_k, num_heads_k, _ = k.shape
    head_size_v = v.shape[-1]

    # Handle optional tensors.
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

    # Output shapes - always allocate full tensors to avoid null buffer issues.
    dq_shape = (batch_size, seqlen_q, num_heads, head_size_q)
    dk_shape = (batch_size, seqlen_k, num_heads_k, head_size_q)
    dv_shape = (batch_size, seqlen_k, num_heads_k, head_size_v)
    softmax_d_shape = (batch_size, num_heads, seqlen_q)

    fn = _cached_fmha_v3_bwd_call(
        dq_shape,
        dk_shape,
        dv_shape,
        softmax_d_shape,
        q.dtype,
    )

    return fn(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        alibi_slopes,
        rng_state,
        gen,
        dropout_p=_static_float(dropout_p),
        softmax_scale=_static_float(softmax_scale),
        is_causal=is_causal,
        window_size_left=_static_int(window_size_left),
        window_size_right=_static_int(window_size_right),
        deterministic=deterministic,
        is_v3_atomic_fp32=is_v3_atomic_fp32,
        how_v3_bf16_cvt=_static_int(how_v3_bf16_cvt),
    )


def _flash_attn_forward(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    bias: Optional[jnp.ndarray],
    alibi_slopes: Optional[jnp.ndarray],
    return_lse: bool,
    return_softmax: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Forward pass with automatic kernel selection and window size normalization."""
    _, seqlen_q, nhead_q, hdim_q = q.shape
    _, seqlen_k, nhead_k, hdim_v = v.shape

    # Normalize window sizes for compatibility.
    window_size_left, window_size_right = _normalize_window_size(
        window_size_left, window_size_right, seqlen_k
    )

    # Select optimal kernel based on input constraints.
    can_use_v3 = _can_impl_fmha_v3_fwd(
        q,
        k,
        v,
        dropout_p,
        seqlen_q,
        seqlen_k,
        hdim_q,
        hdim_v,
        nhead_q,
        nhead_k,
        alibi_slopes,
        bias,
        window_size_left,
        window_size_right,
        return_lse,
    )

    # Create generator for dropout.
    gen_tensor = _create_generator_tensor(dropout_p)

    if can_use_v3:
        result = fmha_v3_fwd(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal,
            window_size_left,
            window_size_right,
            return_lse,
            return_softmax,
            None,
            bias,
            alibi_slopes,
            gen_tensor,
        )
    else:
        result = mha_fwd(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal,
            window_size_left,
            window_size_right,
            return_lse,
            return_softmax,
            None,
            bias,
            alibi_slopes,
            gen_tensor,
        )

    return result


def _flash_attn_backward(
    dout: jnp.ndarray,
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    out: jnp.ndarray,
    softmax_lse: jnp.ndarray,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    bias: Optional[jnp.ndarray],
    alibi_slopes: Optional[jnp.ndarray],
    deterministic: bool,
    rng_state: Optional[jnp.ndarray] = None,
    is_v3_atomic_fp32: Optional[bool] = True,
    how_v3_bf16_cvt: Optional[int] = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
    """Backward pass with automatic kernel selection and bias gradient support.

    Returns: (dq, dk, dv, softmax_d, dbias) - dbias is None if no bias provided.
    """
    # Select optimal backward kernel.
    can_use_v3_bwd = _can_impl_fmha_v3_bwd(
        dout,
        q,
        k,
        v,
        None,  # dk
        None,  # dv
        None,  # dbias
        dropout_p,
        causal,
        window_size_left,
        window_size_right,
        bias,
        alibi_slopes,
        deterministic,
        is_v3_atomic_fp32,
    )

    # Allocate bias gradient buffer if needed.
    dbias_buffer = None
    if bias is not None and not can_use_v3_bwd:
        seqlen_q, seqlen_k = q.shape[1], k.shape[1]
        dbias_buffer = jnp.zeros((seqlen_q, seqlen_k), dtype=jnp.float32)

    try:
        if can_use_v3_bwd:
            log.info("Using FMHA v3 backward kernel (no bias gradient support)")
            results = fmha_v3_bwd(
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                dropout_p,
                softmax_scale,
                causal,
                window_size_left,
                window_size_right,
                deterministic,
                is_v3_atomic_fp32,
                how_v3_bf16_cvt,
                None,  # dq
                None,  # dk
                None,  # dv
                None,  # dbias - v3 doesn't support it
                bias,
                alibi_slopes,
                rng_state,
                None,  # gen
            )
            dbias_result = None
        else:
            log.info("Using MHA backward kernel (supports bias gradients)")
            results = mha_bwd(
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                dropout_p,
                softmax_scale,
                causal,
                window_size_left,
                window_size_right,
                deterministic,
                None,  # dq
                None,  # dk
                None,  # dv
                dbias_buffer,
                bias,
                alibi_slopes,
                rng_state,
                None,  # gen
            )
            dbias_result = dbias_buffer

        # Validate gradient results.
        if results is None or len(results) != 4:
            raise RuntimeError(
                f"Backward kernel returned invalid results: {results}. "
                f"Expected 4 gradients, got {len(results) if results else 0}"
            )

        dq_grad, dk_grad, dv_grad, softmax_d = results

        if dq_grad is None or dk_grad is None or dv_grad is None:
            raise RuntimeError(
                f"Backward kernel returned None gradients: "
                f"dq={dq_grad is not None}, dk={dk_grad is not None}, dv={dv_grad is not None}"
            )

        return dq_grad, dk_grad, dv_grad, softmax_d, dbias_result

    except (RuntimeError, ValueError) as e:
        # Re-raise known error types with context
        kernel_type = "FMHA v3" if can_use_v3_bwd else "MHA"
        raise RuntimeError(
            f"Flash attention backward pass failed with {kernel_type} kernel: {e}. "
            f"Input shapes: q={q.shape}, k={k.shape}, v={v.shape}, "
            f"dout={dout.shape}, dropout_p={dropout_p}, causal={causal}"
        ) from e

    except Exception as e:
        # Log unexpected errors but don't mask them
        kernel_type = "FMHA v3" if can_use_v3_bwd else "MHA"
        # Re-raise instead of silently returning zeros
        raise RuntimeError(
            f"Unexpected error in flash attention backward pass: {e}. "
            f"This indicates a bug that should be reported."
        ) from e


# Public API functions with custom VJP for autodiff.
@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 9, 10, 11))
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
) -> jnp.ndarray:
    """
    Flash Attention function following canonical JAX custom_vjp pattern.

    Args:
        q: Query tensor [batch_size, seqlen, nheads, headdim_q]
        k: Key tensor [batch_size, seqlen, nheads_k, headdim_q]
        v: Value tensor [batch_size, seqlen, nheads_k, headdim_v]
        dropout_p: Dropout probability
        softmax_scale: Scaling factor for softmax (default: 1/sqrt(headdim_q))
        causal: Whether to apply causal masking
        window_size: (left, right) window sizes for sliding window attention
        bias: Optional bias tensor
        alibi_slopes: Optional ALiBi slopes
        deterministic: Whether to use deterministic algorithms
        return_lse: Whether to return log-sum-exp values
        return_attn_probs: Whether to return attention probabilities

    Returns:
        Always returns (output, softmax_lse, S_dmask, rng_state) - user extracts what they need.
    """
    # Set default softmax scale.
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    head_size_q_og = q.shape[3]
    head_size_v_og = v.shape[3]

    # Pad head dimensions to multiples of 8
    q_padded, k_padded, v_padded = q, k, v
    if head_size_q_og % 8 != 0:
        pad_q = 8 - head_size_q_og % 8
        q_padded = jnp.pad(q, ((0, 0), (0, 0), (0, 0), (0, pad_q)))
        k_padded = jnp.pad(k, ((0, 0), (0, 0), (0, 0), (0, pad_q)))
    if head_size_v_og % 8 != 0:
        pad_v = 8 - head_size_v_og % 8
        v_padded = jnp.pad(v, ((0, 0), (0, 0), (0, 0), (0, pad_v)))

    # Normalize window sizes once here using concrete values.
    # This ensures consistent behavior between forward and backward passes.
    seqlen_k = k_padded.shape[1]
    wl_norm, wr_norm = _normalize_window_size(window_size[0], window_size[1], seqlen_k)

    # Call forward kernel with normalized window sizes for consistent behavior
    out_padded, softmax_lse, S_dmask, _ = _flash_attn_forward(
        q_padded,
        k_padded,
        v_padded,
        dropout_p,
        softmax_scale,
        causal=causal,
        window_size_left=wl_norm,  # Use normalized values
        window_size_right=wr_norm,  # Use normalized values
        bias=bias,
        alibi_slopes=alibi_slopes,
        return_lse=return_lse,
        return_softmax=return_attn_probs and dropout_p > 0,
    )

    # Unpad output to original dimensions
    out = out_padded[..., :head_size_v_og]

    return (out, softmax_lse, S_dmask)


def _flash_attn_func_fwd(
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
):
    """Forward pass that returns both output and residuals for backward pass."""
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    head_size_q_og = q.shape[3]
    head_size_v_og = v.shape[3]

    # Pad head dimensions to multiples of 8
    q_padded, k_padded, v_padded = q, k, v
    if head_size_q_og % 8 != 0:
        pad_q = 8 - head_size_q_og % 8
        q_padded = jnp.pad(q, ((0, 0), (0, 0), (0, 0), (0, pad_q)))
        k_padded = jnp.pad(k, ((0, 0), (0, 0), (0, 0), (0, pad_q)))
    if head_size_v_og % 8 != 0:
        pad_v = 8 - head_size_v_og % 8
        v_padded = jnp.pad(v, ((0, 0), (0, 0), (0, 0), (0, pad_v)))

    seqlen_k = k_padded.shape[1]
    wl_norm, wr_norm = _normalize_window_size(window_size[0], window_size[1], seqlen_k)

    # Pass normalized concrete values to forward - ensures consistent behavior
    out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
        q_padded,
        k_padded,
        v_padded,
        dropout_p,
        softmax_scale,
        causal=causal,
        window_size_left=wl_norm,  # Use normalized values
        window_size_right=wr_norm,  # Use normalized values
        bias=bias,
        alibi_slopes=alibi_slopes,
        return_lse=True,  # Always return for backward
        return_softmax=return_attn_probs and dropout_p > 0,
    )

    out = out_padded[..., :head_size_v_og]
    result = (out, softmax_lse, S_dmask)

    # Residuals needed for backward pass.
    residuals = (
        q_padded,
        k_padded,
        v_padded,
        out_padded,
        softmax_lse,
        rng_state,
        dropout_p,
        softmax_scale,
        causal,
        (wl_norm, wr_norm),
        bias,
        alibi_slopes,
        deterministic,
        head_size_q_og,  # Store original Q head dimension
        head_size_v_og,  # Store original V head dimension
    )

    return result, residuals


def _flash_attn_func_bwd(
    dropout_p,
    softmax_scale,
    causal,
    window_size,
    deterministic,
    return_lse,
    return_attn_probs,
    residuals,
    grad_outputs,
):
    """Backward pass using residuals and output gradients.

    Note: With nondiff_argnums, non-differentiable arguments come first in the signature.
    """
    (
        q_padded,
        k_padded,
        v_padded,
        out_padded,
        softmax_lse,
        rng_state,
        res_dropout_p,
        res_softmax_scale,
        res_causal,
        res_window_size,
        res_bias,
        res_alibi_slopes,
        res_deterministic,
        head_size_q_og,
        head_size_v_og,
    ) = residuals

    # Handle different output formats - extract gradient w.r.t. main output
    if isinstance(grad_outputs, tuple):
        dout = grad_outputs[0]  # Gradient w.r.t. main output tensor
    else:
        dout = grad_outputs

    # Pad gradient to match padded dimensions
    if dout.shape[-1] != out_padded.shape[-1]:
        pad_v = out_padded.shape[-1] - dout.shape[-1]
        dout_padded = jnp.pad(dout, ((0, 0), (0, 0), (0, 0), (0, pad_v)))
    else:
        dout_padded = dout

    # Call unified backward function that handles kernel dispatch
    # Use normalized window sizes from forward pass for consistency
    dq_padded, dk_padded, dv_padded, softmax_d, dbias_grad = _flash_attn_backward(
        dout_padded,
        q_padded,
        k_padded,
        v_padded,
        out_padded,
        softmax_lse,
        res_dropout_p,  # Use same conversion as forward
        res_softmax_scale,  # Use same conversion as forward
        res_causal,
        res_window_size[0],
        res_window_size[1],
        res_bias,
        res_alibi_slopes,
        res_deterministic,  # Use same deterministic setting as forward pass
        rng_state,
        True,  # is_v3_atomic_fp32
        1,  # how_v3_bf16_cvt
    )

    # Unpad gradients to match original input dimensions
    # Both Q and K have the same head dimension (head_size_q_og)
    # V has its own dimension (head_size_v_og)
    dq = dq_padded[..., :head_size_q_og]
    dk = dk_padded[..., :head_size_q_og]
    dv = dv_padded[..., :head_size_v_og]

    # Handle bias gradient - now properly extracted from backward kernel
    dbias = dbias_grad  # Already computed by the appropriate kernel

    # Return gradients for differentiable inputs only: q, k, v, softmax_scale, bias, alibi_slopes
    return (
        dq,  # q (index 0)
        dk,  # k (index 1)
        dv,  # v (index 2)
        dbias,  # bias (index 7)
        None,  # alibi_slopes (index 8) - typically no gradient needed
    )


# Register the custom VJP
flash_attn_func.defvjp(_flash_attn_func_fwd, _flash_attn_func_bwd)
