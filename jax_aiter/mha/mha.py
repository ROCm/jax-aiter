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
    swa = (window_size_left > 0) or (window_size_right > 0)

    ret = alibi_slopes is None
    ret = ret and (bias is None)
    ret = ret and (dropout_p == 0.0)
    ret = ret and (hdim_q == hdim_v)
    ret = ret and (hdim_q == 128)
    ret = ret and (nhead_q % nhead_k == 0)
    ret = ret and (not swa)
    ret = ret and (q.dtype == dtypes.bf16)
    return ret


def can_impl_fmha_v3_bwd(
    dout: jnp.ndarray,
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    dk: Optional[jnp.ndarray],
    dv: Optional[jnp.ndarray],
    dbias: Optional[jnp.ndarray],
    dropout_p: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    bias: Optional[jnp.ndarray],
    alibi_slopes: Optional[jnp.ndarray],
    deterministic: bool,
    is_v3_atomic_fp32: bool = True,
) -> bool:
    """Check if FMHA v3 backward kernel can be used.

    This function determines kernel compatibility based on tensor shapes, strides,
    and hardware constraints. It mirrors the logic from the PyTorch/Aiter implementation.
    """
    # Extract shapes.
    _, seqlen_q, nhead_q, hdim_q = q.shape
    _, seqlen_k, nhead_k, hdim_v = v.shape

    # Create placeholder tensors for stride checking if not provided.
    if dk is None:
        dk = jnp.empty_like(k)
    if dv is None:
        dv = jnp.empty_like(v)

    # Calculate strides (JAX arrays have byte strides, convert to element strides)
    # For JAX arrays: element_stride = byte_stride / itemsize.
    def _get_strides(arr):
        """Get element strides from JAX array."""
        byte_strides = arr.strides if hasattr(arr, "strides") else None
        if byte_strides is None:
            # Compute strides for a C-contiguous array.
            itemsize = arr.dtype.itemsize
            shape = arr.shape
            strides = []
            stride = 1
            for dim in reversed(shape):
                strides.insert(0, stride)
                stride *= dim
            return tuple(strides)
        return tuple(s // arr.dtype.itemsize for s in byte_strides)

    q_strides = _get_strides(q)
    k_strides = _get_strides(k)
    v_strides = _get_strides(v)
    dout_strides = _get_strides(dout)
    dk_strides = _get_strides(dk)
    dv_strides = _get_strides(dv)

    batch_stride_q = q_strides[0]
    stride_q = q_strides[1]
    nhead_stride_q = q_strides[2]

    batch_stride_k = k_strides[0]
    stride_k = k_strides[1]
    nhead_stride_k = k_strides[2]

    batch_stride_v = v_strides[0]
    stride_v = v_strides[1]
    nhead_stride_v = v_strides[2]

    batch_stride_do = dout_strides[0]
    stride_do = dout_strides[1]
    nhead_stride_do = dout_strides[2]

    batch_stride_dk = dk_strides[0]
    nhead_stride_dk = dk_strides[2]

    batch_stride_dv = dv_strides[0]
    nhead_stride_dv = dv_strides[2]

    # Normalize window sizes.
    window_size_left = -1 if window_size_left >= seqlen_k else window_size_left
    window_size_right = -1 if window_size_right >= seqlen_k else window_size_right
    mask = causal and window_size_left == -1  # causal mask
    nmask = not causal and window_size_left == -1 and window_size_right == -1  # no mask
    swa = (window_size_left > 0) or (window_size_right > 0)  # sliding window attention

    def np():
        """Check non-padded cases with specific stride requirements."""
        npssk = seqlen_q == seqlen_k
        npssk &= seqlen_k % 64 == 0
        npssk &= stride_q == stride_do
        npssk &= nhead_stride_q == nhead_stride_do
        npssk &= batch_stride_q == batch_stride_do
        npssk &= stride_k == stride_v
        npssk &= nhead_stride_k == nhead_stride_v
        npssk &= batch_stride_k == batch_stride_v
        npssk &= nhead_stride_k == nhead_stride_dk
        npssk &= nhead_stride_v == nhead_stride_dv
        npssk &= (batch_stride_dk / batch_stride_k) == (nhead_q / nhead_k)
        npssk &= (batch_stride_dv / batch_stride_v) == (nhead_q / nhead_k)

        hd128_case = (hdim_q == 128) and npssk
        hd64_case = (hdim_q == 64 and is_v3_atomic_fp32 == False) and npssk

        ret = hd128_case or hd64_case
        return ret

    def pssk():
        """Check for specific head dimensions with atomic fp32 operations."""
        gfx = get_gfx()
        # nhead_stride_dq_acc >= stride_dq_acc must be guaranteed.
        ret = (hdim_q == 64 and gfx == "gfx942" and is_v3_atomic_fp32 == True) or (
            hdim_q == 128 and gfx == "gfx950"
        )
        ret &= nmask or (
            mask and seqlen_q == seqlen_k
        )  # TODO: or (seqlen_q != seqlen_k and mask_type == top_left)

        return ret

    def pddv():
        """Check for padded dimensions between 64 and 128."""
        ret = is_v3_atomic_fp32 == False
        ret &= hdim_q > 64 and hdim_q < 128
        ret &= seqlen_q == seqlen_k
        ret &= seqlen_k % 64 == 0
        ret &= stride_q == stride_do
        ret &= nhead_stride_q == nhead_stride_do
        ret &= batch_stride_q == batch_stride_do
        ret &= stride_k == stride_v
        ret &= nhead_stride_k == nhead_stride_v
        ret &= batch_stride_k == batch_stride_v
        ret &= nhead_stride_k == nhead_stride_dk
        ret &= nhead_stride_v == nhead_stride_dv
        ret &= (batch_stride_dk / batch_stride_k) == (nhead_q / nhead_k)
        ret &= (batch_stride_dv / batch_stride_v) == (nhead_q / nhead_k)

        return ret

    def psskddv():
        """Check for padded cases with sliding window attention support."""
        ret = is_v3_atomic_fp32 == True
        ret &= hdim_q > 64 and hdim_q <= 192
        ret &= (
            nmask
            or (mask and seqlen_q == seqlen_k)
            or (swa and hdim_q > 64 and hdim_q <= 128)
        )  # TODO: or (seqlen_q != seqlen_k and mask_type == top_left)

        return ret

    # Basic constraints
    ret = alibi_slopes is None
    ret &= bias is None
    ret &= dropout_p == 0.0
    ret &= not deterministic
    ret &= hdim_q == hdim_v
    ret &= nhead_q % nhead_k == 0
    ret &= hdim_q >= 64 and hdim_q <= 192 and hdim_q % 8 == 0
    ret &= np() or pssk() or pddv() or psskddv()

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
    dbias_shape,
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
            jax.ShapeDtypeStruct(dbias_shape, grad_dtype),
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
    dbias_shape = bias.shape if bias is not None else (0,)

    fn = _cached_mha_bwd_call(
        dq_shape,
        dk_shape,
        dv_shape,
        softmax_d_shape,
        dbias_shape,
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
            None,  # gen
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
            None,  # gen
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
    can_impl_fmha_v3_bwd_ = can_impl_fmha_v3_bwd(
        dout,
        q,
        k,
        v,
        jnp.empty_like(k),  # To maintain parity with aiter
        jnp.empty_like(v),  # To maintain parity with aiter
        (
            jnp.empty_like(bias) if bias is not None else None
        ),  # To maintain parity with aiter
        dropout_p,
        causal,
        window_size_left,
        window_size_right,
        bias,
        alibi_slopes,
        deterministic,
        is_v3_atomic_fp32,
    )

    try:
        if can_impl_fmha_v3_bwd_:
            log.info("Using FMHA v3 backward kernel")
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
            return results[0], results[1], results[2], results[3], None
        else:
            log.info("Using MHA backward kernel")
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
                None,  # dbias - ck supports it
                bias,
                alibi_slopes,
                rng_state,
                None,  # gen
            )
        return results

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
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
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

    Arguments:
        q: (batch_size, seqlen, nheads, headdim_q)
        k: (batch_size, seqlen, nheads_k, headdim_q)
        v: (batch_size, seqlen, nheads_k, headdim_v)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim_q).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        bias: (seqlen_q, seqlen_k)
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim_v).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
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

    result = [out]
    if return_lse:
        result.append(softmax_lse)
    if return_attn_probs:
        result.append(S_dmask)

    return tuple(result)


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

    result = [out]
    if return_lse:
        result.append(softmax_lse)
    if return_attn_probs:
        result.append(S_dmask)

    result = tuple(result)

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

    # Pad gradient to match padded dimensions.
    if dout.shape[-1] != out_padded.shape[-1]:
        pad_v = out_padded.shape[-1] - dout.shape[-1]
        dout_padded = jnp.pad(dout, ((0, 0), (0, 0), (0, 0), (0, pad_v)))
    else:
        dout_padded = dout

    # Call unified backward function that handles kernel dispatch.
    dq_padded, dk_padded, dv_padded, softmax_d, dbias_grad = _flash_attn_backward(
        dout_padded,
        q_padded,
        k_padded,
        v_padded,
        out_padded,
        softmax_lse,
        res_dropout_p,
        res_softmax_scale,
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
