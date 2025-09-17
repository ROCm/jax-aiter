# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""FFI wrappers for all MHA kernels with kernel dispatch logic."""

from __future__ import annotations
import logging
from functools import lru_cache
from typing import Tuple, Optional, List
import os

import jax
import jax.numpy as jnp
import numpy as np

from ..ja_compat import dtypes
from ..ja_compat.chip_info import get_gfx
from ..ffi.registry import register_ffi_target

log = logging.getLogger("aiter.mha")

# Debug control
DEBUG_MHA = os.getenv("DEBUG_MHA", "0") == "1"


def debug_print(msg: str, level: str = "INFO"):
    """Print debug message if DEBUG_MHA is enabled."""
    if DEBUG_MHA:
        print(f"[MHA-{level}] {msg}")


def debug_func_entry(func_name: str, **kwargs):
    """Debug function entry with parameters."""
    if DEBUG_MHA:
        print(f"[MHA-ENTRY] {func_name}")
        for key, value in kwargs.items():
            if hasattr(value, "shape") and hasattr(value, "dtype"):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {value} (type: {type(value)})")


def debug_kernel_dispatch(kernel_name: str, can_use: bool, reason: str = ""):
    """Debug kernel dispatch decisions."""
    if DEBUG_MHA:
        status = "SELECTED" if can_use else "REJECTED"
        print(f"[MHA-DISPATCH] {kernel_name}: {status}")
        if reason:
            print(f"  Reason: {reason}")


def debug_ffi_call(target_name: str, **params):
    """Debug FFI call parameters."""
    if DEBUG_MHA:
        print(f"[MHA-FFI] Calling {target_name}")
        for key, value in params.items():
            print(f"  {key}: {value} (type: {type(value)})")


def _ensure_ffi_target_registered(target_name: str):
    """Lazily register a specific FFI target when first needed."""
    register_ffi_target(target_name, "ROCM")


# Helper functions for optional tensors.
def _empty_tensor(dtype):
    """Create a valid empty tensor that won't cause null buffer issues."""
    # Create a zero-sized tensor with valid data pointer (not null)
    return jnp.zeros((0,), dtype=dtype)


def _maybe_contiguous(x):
    """Ensure tensor is contiguous (JAX equivalent)."""
    return x if x is not None else None


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
    """Check if we can use FMHA v3 forward kernel (exact aiter logic)."""
    # Sliding window check
    swa = (window_size_left > 0) or (window_size_right > 0)

    # Basic constraints (matching aiter exactly)
    gfx = get_gfx()
    ret = alibi_slopes is None
    ret = ret and (bias is None)
    ret = ret and (dropout_p == 0.0)
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
    """Check if we can use FMHA v3 backward kernel."""
    _, seqlen_q, nhead_q, hdim_q = q.shape
    _, seqlen_k, nhead_k, hdim_v = v.shape

    # Basic constraints
    ret = alibi_slopes is None
    ret = ret and bias is None
    ret = ret and dbias is None
    ret = ret and dropout_p == 0.0
    ret = ret and not deterministic
    ret = ret and hdim_q == hdim_v
    ret = ret and nhead_q % nhead_k == 0
    ret = ret and hdim_q >= 64 and hdim_q <= 192 and hdim_q % 8 == 0

    # Additional v3-specific constraints
    gfx = get_gfx()
    if hdim_q == 64:
        ret = ret and ((gfx == "gfx942" and is_v3_atomic_fp32) or (gfx == "gfx950"))
    elif hdim_q == 128:
        ret = ret and (gfx == "gfx950")

    return ret


def _can_impl_fmha_v3_varlen_bwd(
    dout,
    q,
    k,
    v,
    dropout_p,
    causal,
    window_size_left,
    window_size_right,
    alibi_slopes,
    deterministic,
    is_v3_atomic_fp32=True,
):
    """Check if we can use FMHA v3 varlen backward kernel."""
    _, nhead_q, hdim_q = q.shape
    _, nhead_k, hdim_v = v.shape

    # Basic constraints
    ret = alibi_slopes is None
    ret = ret and dropout_p == 0.0
    ret = ret and not deterministic
    ret = ret and hdim_q == hdim_v
    ret = ret and nhead_q % nhead_k == 0
    ret = ret and hdim_q >= 64 and hdim_q <= 128 and hdim_q % 8 == 0

    # Mask constraints
    nmask = not causal and window_size_left == -1 and window_size_right == -1
    mask = causal and window_size_left == -1
    ret = ret and (mask or nmask)

    # Architecture-specific constraints
    if hdim_q in [64, 128]:
        ret = ret and is_v3_atomic_fp32

    return ret


# Cached FFI call implementations
@lru_cache(maxsize=None)
def _cached_mha_fwd_call(
    out_shape,
    softmax_lse_shape,
    p_shape,
    rng_state_shape,
    dropout_p,
    softmax_scale,
    is_causal,
    window_size_left,
    window_size_right,
    return_softmax_lse,
    return_dropout_randval,
):
    """Create cached JIT call for MHA forward."""
    call = jax.ffi.ffi_call(
        "MhaFwdJA",
        (
            jax.ShapeDtypeStruct(out_shape, jnp.bfloat16),
            jax.ShapeDtypeStruct(softmax_lse_shape, jnp.float32),
            jax.ShapeDtypeStruct(p_shape, jnp.uint8),
            jax.ShapeDtypeStruct(rng_state_shape, jnp.int64),
        ),
        vmap_method="broadcast_all",
    )

    def _invoke(q, k, v, out_provided, bias, alibi_slopes, gen):
        debug_ffi_call(
            "FmhaV3FwdJA",
            q_shape=q.shape,
            k_shape=k.shape,
            v_shape=v.shape,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            is_causal=is_causal,
            return_softmax_lse=return_softmax_lse,
            return_dropout_randval=return_dropout_randval,
        )
        
        result = call(
            q,
            k,
            v,
            out_provided,
            bias,
            alibi_slopes,
            gen,
            dropout_p=np.float32(dropout_p),
            softmax_scale=np.float32(softmax_scale),
            is_causal=is_causal,
            window_size_left=np.int32(window_size_left),
            window_size_right=np.int32(window_size_right),
            return_softmax_lse=return_softmax_lse,
            return_dropout_randval=return_dropout_randval,
        )
        
        # Debug the outputs
        if DEBUG_MHA:
            print(f"[MHA-OUTPUT] FmhaV3FwdJA returned {len(result)} outputs")
            for i, output in enumerate(result):
                if output is not None and hasattr(output, 'shape'):
                    print(f"  Output[{i}]: shape={output.shape}, dtype={output.dtype}")
                else:
                    print(f"  Output[{i}]: {output} (type: {type(output)})")
        
        return result

    return jax.jit(_invoke)


@lru_cache(maxsize=None)
def _cached_fmha_v3_fwd_call(
    out_shape,
    softmax_lse_shape,
    p_shape,
    rng_state_shape,
    dropout_p,
    softmax_scale,
    is_causal,
    window_size_left,
    window_size_right,
    return_softmax_lse,
    return_dropout_randval,
):
    """Create cached JIT call for FMHA v3 forward."""
    call = jax.ffi.ffi_call(
        "FmhaV3FwdJA",
        (
            jax.ShapeDtypeStruct(out_shape, jnp.bfloat16),
            jax.ShapeDtypeStruct(softmax_lse_shape, jnp.float32),
            jax.ShapeDtypeStruct(p_shape, jnp.uint8),
            jax.ShapeDtypeStruct(rng_state_shape, jnp.int64),
        ),
        vmap_method="broadcast_all",
    )

    def _invoke(q, k, v, out_provided, bias, alibi_slopes, gen):
        debug_ffi_call(
            "FmhaV3FwdJA",
            q_shape=q.shape,
            k_shape=k.shape,
            v_shape=v.shape,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            is_causal=is_causal,
            return_softmax_lse=return_softmax_lse,
            return_dropout_randval=return_dropout_randval,
        )
        
        result = call(
            q,
            k,
            v,
            out_provided,
            bias,
            alibi_slopes,
            gen,
            dropout_p=np.float32(dropout_p),
            softmax_scale=np.float32(softmax_scale),
            is_causal=is_causal,
            window_size_left=np.int32(window_size_left),
            window_size_right=np.int32(window_size_right),
            return_softmax_lse=return_softmax_lse,
            return_dropout_randval=return_dropout_randval,
        )
        
        # Debug the outputs
        if DEBUG_MHA:
            print(f"[MHA-OUTPUT] FmhaV3FwdJA returned {len(result)} outputs")
            for i, output in enumerate(result):
                if output is not None and hasattr(output, 'shape'):
                    print(f"  Output[{i}]: shape={output.shape}, dtype={output.dtype}")
                else:
                    print(f"  Output[{i}]: {output} (type: {type(output)})")
        
        return result

    return jax.jit(_invoke)


@lru_cache(maxsize=None)
def _cached_mha_bwd_call(
    dq_shape,
    dk_shape,
    dv_shape,
    softmax_d_shape,
    dropout_p,
    softmax_scale,
    is_causal,
    window_size_left,
    window_size_right,
    deterministic,
):
    """Create cached JIT call for MHA backward."""
    call = jax.ffi.ffi_call(
        "MhaBwdJA",
        (
            jax.ShapeDtypeStruct(dq_shape, jnp.bfloat16),
            jax.ShapeDtypeStruct(dk_shape, jnp.bfloat16),
            jax.ShapeDtypeStruct(dv_shape, jnp.bfloat16),
            jax.ShapeDtypeStruct(softmax_d_shape, jnp.float32),
        ),
        vmap_method="broadcast_all",
    )

    def _invoke(dout, q, k, v, out, softmax_lse, dq, dk, dv, dbias, bias, alibi_slopes, rng_state, gen):
        debug_ffi_call(
            "MhaBwdJA",
            dout_shape=dout.shape,
            q_shape=q.shape,
            k_shape=k.shape,
            v_shape=v.shape,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            is_causal=is_causal,
            deterministic=deterministic,
        )
        
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
            p_dropout=np.float32(dropout_p),
            softmax_scale=np.float32(softmax_scale),
            is_causal=is_causal,
            window_size_left=np.int32(window_size_left),
            window_size_right=np.int32(window_size_right),
            deterministic=deterministic,
        )
        
        # Debug the outputs
        if DEBUG_MHA:
            print(f"[MHA-OUTPUT] MhaBwdJA returned {len(result)} outputs")
            for i, output in enumerate(result):
                if output is not None and hasattr(output, 'shape'):
                    print(f"  Output[{i}]: shape={output.shape}, dtype={output.dtype}")
                else:
                    print(f"  Output[{i}]: {output} (type: {type(output)})")
        
        return result

    return jax.jit(_invoke)


@lru_cache(maxsize=None)
def _cached_fmha_v3_bwd_call(
    dq_shape,
    dk_shape,
    dv_shape,
    softmax_d_shape,
    dropout_p,
    softmax_scale,
    is_causal,
    window_size_left,
    window_size_right,
    deterministic,
    is_v3_atomic_fp32,
    how_v3_bf16_cvt,
):
    """Create cached JIT call for FMHA v3 backward."""
    call = jax.ffi.ffi_call(
        "FmhaV3BwdJA",
        (
            jax.ShapeDtypeStruct(dq_shape, jnp.bfloat16),
            jax.ShapeDtypeStruct(dk_shape, jnp.bfloat16),
            jax.ShapeDtypeStruct(dv_shape, jnp.bfloat16),
            jax.ShapeDtypeStruct(softmax_d_shape, jnp.float32),
        ),
        vmap_method="broadcast_all",
    )

    def _invoke(
        dout, q, k, v, out, softmax_lse, dq, dk, dv, alibi_slopes, rng_state, gen
    ):
        debug_ffi_call(
            "FmhaV3BwdJA",
            dout_shape=dout.shape,
            q_shape=q.shape,
            k_shape=k.shape,
            v_shape=v.shape,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            is_causal=is_causal,
            deterministic=deterministic,
            is_v3_atomic_fp32=is_v3_atomic_fp32,
            how_v3_bf16_cvt=how_v3_bf16_cvt,
        )
        
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
            dropout_p=np.float32(dropout_p),
            softmax_scale=np.float32(softmax_scale),
            is_causal=is_causal,
            window_size_left=np.int32(window_size_left),
            window_size_right=np.int32(window_size_right),
            deterministic=deterministic,
            is_v3_atomic_fp32=is_v3_atomic_fp32,
            how_v3_bf16_cvt=np.int32(how_v3_bf16_cvt),
        )
        
        # Debug the outputs
        if DEBUG_MHA:
            print(f"[MHA-OUTPUT] FmhaV3BwdJA returned {len(result)} outputs")
            for i, output in enumerate(result):
                if output is not None and hasattr(output, 'shape'):
                    print(f"  Output[{i}]: shape={output.shape}, dtype={output.dtype}")
                else:
                    print(f"  Output[{i}]: {output} (type: {type(output)})")
        
        return result

    return jax.jit(_invoke)


@lru_cache(maxsize=None)
def _cached_mha_varlen_fwd_call(out_shape, softmax_lse_shape, p_shape, rng_state_shape):
    """Create cached JIT call for MHA varlen forward."""
    call = jax.ffi.ffi_call(
        "MhaVarlenFwdJA",
        (
            jax.ShapeDtypeStruct(out_shape, jnp.bfloat16),
            jax.ShapeDtypeStruct(softmax_lse_shape, jnp.float32),
            jax.ShapeDtypeStruct(p_shape, jnp.uint8),
            jax.ShapeDtypeStruct(rng_state_shape, jnp.int32),
        ),
        vmap_method="broadcast_all",
    )

    def _invoke(*args):
        return call(*args)

    return jax.jit(_invoke)


@lru_cache(maxsize=None)
def _cached_mha_varlen_bwd_call(dq_shape, dk_shape, dv_shape, softmax_d_shape):
    """Create cached JIT call for MHA varlen backward."""
    call = jax.ffi.ffi_call(
        "MhaVarlenBwdJA",
        (
            jax.ShapeDtypeStruct(dq_shape, jnp.bfloat16),
            jax.ShapeDtypeStruct(dk_shape, jnp.bfloat16),
            jax.ShapeDtypeStruct(dv_shape, jnp.bfloat16),
            jax.ShapeDtypeStruct(softmax_d_shape, jnp.float32),
        ),
        vmap_method="broadcast_all",
    )

    def _invoke(*args):
        return call(*args)

    return jax.jit(_invoke)


@lru_cache(maxsize=None)
def _cached_fmha_v3_varlen_bwd_call(dq_shape, dk_shape, dv_shape, softmax_d_shape):
    """Create cached JIT call for FMHA v3 varlen backward."""
    call = jax.ffi.ffi_call(
        "FmhaV3VarlenBwdJA",
        (
            jax.ShapeDtypeStruct(dq_shape, jnp.bfloat16),
            jax.ShapeDtypeStruct(dk_shape, jnp.bfloat16),
            jax.ShapeDtypeStruct(dv_shape, jnp.bfloat16),
            jax.ShapeDtypeStruct(softmax_d_shape, jnp.float32),
        ),
        vmap_method="broadcast_all",
    )

    def _invoke(*args):
        return call(*args)

    return jax.jit(_invoke)


@lru_cache(maxsize=None)
def _cached_mha_batch_prefill_call(
    out_shape, softmax_lse_shape, p_shape, rng_state_shape
):
    """Create cached JIT call for MHA batch prefill."""
    call = jax.ffi.ffi_call(
        "MhaBatchPrefillJA",
        (
            jax.ShapeDtypeStruct(out_shape, jnp.bfloat16),
            jax.ShapeDtypeStruct(softmax_lse_shape, jnp.float32),
            jax.ShapeDtypeStruct(p_shape, jnp.uint8),
            jax.ShapeDtypeStruct(rng_state_shape, jnp.int32),
        ),
        vmap_method="broadcast_all",
    )

    def _invoke(*args):
        return call(*args)

    return jax.jit(_invoke)


# Low-level kernel functions
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
    """Multi-head attention forward pass."""
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
        gen = _empty_tensor(jnp.int32)

    # Output shapes for MhaFwdJA (now 4 outputs to match PyTorch interface)
    out_shape = (batch_size, seqlen_q, num_heads, head_size_v)
    softmax_lse_shape = (
        (batch_size, num_heads, seqlen_q) if return_softmax_lse else None
    )
    p_shape = (
        (batch_size, num_heads, seqlen_q, seqlen_k) if return_dropout_randval else None
    )
    rng_state_shape = (2,)

    fn = _cached_mha_fwd_call(
        out_shape,
        softmax_lse_shape,
        p_shape,
        rng_state_shape,
        dropout_p,
        softmax_scale,
        is_causal,
        window_size_left,
        window_size_right,
        return_softmax_lse,
        return_dropout_randval,
    )

    # Call with PyTorch-compatible signature and ensure list return
    results = fn(q, k, v, out, bias, alibi_slopes, gen)

    # Convert tuple to list for consistency
    return list(results)


@jax.custom_vjp
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
    """FMHA v3 forward pass (FFI) — returns [out, softmax_lse, p, rng_state]."""
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
        gen = _empty_tensor(jnp.int32)

    # Output shapes - always allocate full tensors to avoid null buffer issues
    out_shape = (batch_size, seqlen_q, num_heads_q, head_size_v)
    # Always allocate softmax_lse for potential backward pass, even if not returned to user
    softmax_lse_shape = (batch_size, num_heads_q, seqlen_q)
    p_shape = (
        (batch_size, num_heads_q, seqlen_q, seqlen_k)
        if return_dropout_randval
        else (0,)
    )
    rng_state_shape = (2,)

    fn = _cached_fmha_v3_fwd_call(
        out_shape,
        softmax_lse_shape,
        p_shape,
        rng_state_shape,
        dropout_p,
        softmax_scale,
        is_causal,
        window_size_left,
        window_size_right,
        return_softmax_lse,
        return_dropout_randval,
    )

    return fn(q, k, v, out, bias, alibi_slopes, gen)


def _fmha_v3_fwd_fwd(
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
):
    """Forward pass for VJP."""
    results = fmha_v3_fwd(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        is_causal,
        window_size_left,
        window_size_right,
        True,
        return_dropout_randval,
        out,
        bias,
        alibi_slopes,
        gen,  # Always return LSE for backward
    )

    # Store auxiliary data needed for backward pass
    out_tensor, softmax_lse, p_tensor, rng_state = results
    aux_data = (
        q,
        k,
        v,
        out_tensor,
        softmax_lse,
        p_tensor,
        rng_state,
        dropout_p,
        softmax_scale,
        is_causal,
        window_size_left,
        window_size_right,
        bias,
        alibi_slopes,
        return_softmax_lse,
        return_dropout_randval,
    )

    # Always return 4 values to match the FFI call interface
    # The calling code (_flash_attn_forward) expects consistent 4-tuple output
    return (out_tensor, softmax_lse, p_tensor, rng_state), aux_data


def _fmha_v3_fwd_bwd(aux_data, g):
    """Backward pass for VJP."""
    (
        q,
        k,
        v,
        out,
        softmax_lse,
        p_tensor,
        rng_state,
        dropout_p,
        softmax_scale,
        is_causal,
        window_size_left,
        window_size_right,
        bias,
        alibi_slopes,
        return_softmax_lse,
        return_dropout_randval,
    ) = aux_data

    # Extract gradient for output tensor.
    if isinstance(g, tuple):
        dout = g[0]  # Gradient w.r.t. output tensor.
    else:
        dout = g

    # Call backward kernel.
    _ensure_ffi_target_registered("FmhaV3BwdJA")

    try:
        grads = fmha_v3_bwd(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dropout_p,
            softmax_scale,
            is_causal,
            window_size_left,
            window_size_right,
            deterministic=False,
            is_v3_atomic_fp32=True,
            how_v3_bf16_cvt=1,
            dq=None,
            dk=None,
            dv=None,
            alibi_slopes=alibi_slopes,
            rng_state=rng_state,
            gen=None,
        )
        dq, dk, dv, _ = grads
    except Exception as e:
        # Fallback to zeros if backward fails.
        debug_print(f"FMHA v3 backward failed: {e}, using zero gradients", "WARNING")
        dq = jnp.zeros_like(q)
        dk = jnp.zeros_like(k)
        dv = jnp.zeros_like(v)

    # Return gradients for all inputs (None for non-differentiable arguments).
    return (
        dq,
        dk,
        dv,  # q, k, v gradients
        None,  # dropout_p
        None,  # softmax_scale
        None,  # is_causal
        None,  # window_size_left
        None,  # window_size_right
        None,  # return_softmax_lse
        None,  # return_dropout_randval
        None,  # out
        None,  # bias
        None,  # alibi_slopes
        None,  # gen
    )


# Register the VJP
fmha_v3_fwd.defvjp(_fmha_v3_fwd_fwd, _fmha_v3_fwd_bwd)


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
    """Multi-head attention backward pass."""
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
        gen = _empty_tensor(jnp.int32)

    # Output shapes for the FFI call
    dq_shape = (batch_size, seqlen_q, num_heads, head_size_q)
    dk_shape = (batch_size, seqlen_k, num_heads_k, head_size_q)
    dv_shape = (batch_size, seqlen_k, num_heads_k, head_size_v)
    softmax_d_shape = (batch_size, num_heads, seqlen_q)

    fn = _cached_mha_bwd_call(
        dq_shape,
        dk_shape,
        dv_shape,
        softmax_d_shape,
        dropout_p,
        softmax_scale,
        is_causal,
        window_size_left,
        window_size_right,
        deterministic,
    )

    return fn(dout, q, k, v, out, softmax_lse, dq, dk, dv, dbias, bias, alibi_slopes, rng_state, gen)


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
    alibi_slopes: Optional[jnp.ndarray] = None,
    rng_state: Optional[jnp.ndarray] = None,
    gen: Optional[jnp.ndarray] = None,
) -> List[jnp.ndarray]:
    """FMHA v3 backward pass."""
    _ensure_ffi_target_registered("FmhaV3BwdJA")

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
    if alibi_slopes is None:
        alibi_slopes = _empty_tensor(jnp.float32)
    if rng_state is None:
        rng_state = _empty_tensor(jnp.int64)
    if gen is None:
        gen = _empty_tensor(jnp.int32)

    # Output shapes
    dq_shape = (batch_size, seqlen_q, num_heads, head_size_q)
    dk_shape = (batch_size, seqlen_k, num_heads_k, head_size_q)
    dv_shape = (batch_size, seqlen_k, num_heads_k, head_size_v)
    softmax_d_shape = (batch_size, num_heads, seqlen_q)

    fn = _cached_fmha_v3_bwd_call(
        dq_shape,
        dk_shape,
        dv_shape,
        softmax_d_shape,
        dropout_p,
        softmax_scale,
        is_causal,
        window_size_left,
        window_size_right,
        deterministic,
        is_v3_atomic_fp32,
        how_v3_bf16_cvt,
    )

    return fn(dout, q, k, v, out, softmax_lse, dq, dk, dv, alibi_slopes, rng_state, gen)


def mha_varlen_fwd(
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
    logits_soft_cap: float,
    zero_tensors: bool,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    return_softmax_lse: bool,
    return_dropout_randval: bool,
    out: Optional[jnp.ndarray] = None,
    block_table: Optional[jnp.ndarray] = None,
    bias: Optional[jnp.ndarray] = None,
    alibi_slopes: Optional[jnp.ndarray] = None,
    gen: Optional[jnp.ndarray] = None,
) -> List[jnp.ndarray]:
    """Multi-head attention varlen forward pass."""
    _ensure_ffi_target_registered("MhaVarlenFwdJA")

    total_q, num_heads, head_size_v = q.shape
    total_k, num_heads_k, _ = k.shape

    # Handle optional tensors
    if cu_seqlens_k is None:
        cu_seqlens_k = cu_seqlens_q
    if out is None:
        out = _empty_tensor(q.dtype)
    if block_table is None:
        block_table = _empty_tensor(jnp.int32)
    if bias is None:
        bias = _empty_tensor(jnp.float32)
    if alibi_slopes is None:
        alibi_slopes = _empty_tensor(jnp.float32)
    if gen is None:
        gen = _empty_tensor(jnp.int32)

    # Output shapes
    out_shape = (total_q, num_heads, head_size_v)
    softmax_lse_shape = (num_heads, total_q) if return_softmax_lse else (0,)
    p_shape = (num_heads, total_q, max_seqlen_k) if return_dropout_randval else (0,)
    rng_state_shape = (2,)

    fn = _cached_mha_varlen_fwd_call(
        out_shape, softmax_lse_shape, p_shape, rng_state_shape
    )

    # Prepare arguments for FFI call
    args = [
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        out,
        block_table,
        bias,
        alibi_slopes,
        gen,
        np.int32(max_seqlen_q),
        np.int32(max_seqlen_k),
        np.int32(min_seqlen_q),
        np.float32(dropout_p),
        np.float32(softmax_scale),
        np.float32(logits_soft_cap),
        np.bool_(zero_tensors),
        np.bool_(is_causal),
        np.int32(window_size_left),
        np.int32(window_size_right),
        np.bool_(return_softmax_lse),
        np.bool_(return_dropout_randval),
    ]

    return fn(*args)


def mha_varlen_bwd(
    dout: jnp.ndarray,
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    out: jnp.ndarray,
    softmax_lse: jnp.ndarray,
    cu_seqlens_q: jnp.ndarray,
    cu_seqlens_k: jnp.ndarray,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
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
    """Multi-head attention varlen backward pass."""
    _ensure_ffi_target_registered("MhaVarlenBwdJA")

    total_q, num_heads, head_size_q = q.shape
    total_k, num_heads_k, _ = k.shape
    head_size_v = v.shape[-1]

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
        rng_state = _empty_tensor(jnp.int32)
    if gen is None:
        gen = _empty_tensor(jnp.int32)

    # Output shapes
    dq_shape = (total_q, num_heads, head_size_q)
    dk_shape = (total_k, num_heads_k, head_size_q)
    dv_shape = (total_k, num_heads_k, head_size_v)
    softmax_d_shape = (num_heads, total_q)

    fn = _cached_mha_varlen_bwd_call(dq_shape, dk_shape, dv_shape, softmax_d_shape)

    # Prepare arguments for FFI call
    args = [
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
        np.int32(max_seqlen_q),
        np.int32(max_seqlen_k),
        np.float32(dropout_p),
        np.float32(softmax_scale),
        np.bool_(zero_tensors),
        np.bool_(is_causal),
        np.int32(window_size_left),
        np.int32(window_size_right),
        np.bool_(deterministic),
    ]

    return fn(*args)


def fmha_v3_varlen_bwd(
    dout: jnp.ndarray,
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    out: jnp.ndarray,
    softmax_lse: jnp.ndarray,
    cu_seqlens_q: jnp.ndarray,
    cu_seqlens_k: jnp.ndarray,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    zero_tensors: bool,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool,
    is_v3_atomic_fp32: bool,
    how_v3_bf16_cvt: int,
    dq: Optional[jnp.ndarray] = None,
    dk: Optional[jnp.ndarray] = None,
    dv: Optional[jnp.ndarray] = None,
    alibi_slopes: Optional[jnp.ndarray] = None,
    rng_state: Optional[jnp.ndarray] = None,
    gen: Optional[jnp.ndarray] = None,
) -> List[jnp.ndarray]:
    """FMHA v3 varlen backward pass."""
    _ensure_ffi_target_registered("FmhaV3VarlenBwdJA")

    total_q, num_heads, head_size_q = q.shape
    total_k, num_heads_k, _ = k.shape
    head_size_v = v.shape[-1]
    b = cu_seqlens_q.shape[0] - 1

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
        rng_state = _empty_tensor(jnp.int32)
    if gen is None:
        gen = _empty_tensor(jnp.int32)

    # Output shapes
    dq_shape = (total_q, num_heads, head_size_q)
    dk_shape = (total_k, num_heads_k, head_size_q)
    dv_shape = (total_k, num_heads_k, head_size_v)
    softmax_d_shape = (b, num_heads, max_seqlen_q)

    fn = _cached_fmha_v3_varlen_bwd_call(dq_shape, dk_shape, dv_shape, softmax_d_shape)

    # Prepare arguments for FFI call
    args = [
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
        np.int32(max_seqlen_q),
        np.int32(max_seqlen_k),
        np.float32(dropout_p),
        np.float32(softmax_scale),
        np.bool_(zero_tensors),
        np.bool_(is_causal),
        np.int32(window_size_left),
        np.int32(window_size_right),
        np.bool_(deterministic),
        np.bool_(is_v3_atomic_fp32),
        np.int32(how_v3_bf16_cvt),
    ]

    return fn(*args)


def mha_batch_prefill(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    cu_seqlens_q: jnp.ndarray,
    kv_indptr: jnp.ndarray,
    kv_page_indices: jnp.ndarray,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    logits_soft_cap: float,
    zero_tensors: bool,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    return_softmax_lse: bool,
    return_dropout_randval: bool,
    out: Optional[jnp.ndarray] = None,
    alibi_slopes: Optional[jnp.ndarray] = None,
    gen: Optional[jnp.ndarray] = None,
) -> List[jnp.ndarray]:
    """Multi-head attention batch prefill."""
    _ensure_ffi_target_registered("MhaBatchPrefillJA")

    total_q, num_heads, head_size_v = q.shape

    # Handle optional tensors
    if out is None:
        out = _empty_tensor(q.dtype)
    if alibi_slopes is None:
        alibi_slopes = _empty_tensor(jnp.float32)
    if gen is None:
        gen = _empty_tensor(jnp.int32)

    # Output shapes
    out_shape = (total_q, num_heads, head_size_v)
    softmax_lse_shape = (num_heads, total_q) if return_softmax_lse else (0,)
    p_shape = (num_heads, total_q, max_seqlen_k) if return_dropout_randval else (0,)
    rng_state_shape = (2,)

    fn = _cached_mha_batch_prefill_call(
        out_shape, softmax_lse_shape, p_shape, rng_state_shape
    )

    # Prepare arguments for FFI call
    args = [
        q,
        k,
        v,
        cu_seqlens_q,
        kv_indptr,
        kv_page_indices,
        out,
        alibi_slopes,
        gen,
        np.int32(max_seqlen_q),
        np.int32(max_seqlen_k),
        np.float32(dropout_p),
        np.float32(softmax_scale),
        np.float32(logits_soft_cap),
        np.bool_(zero_tensors),
        np.bool_(is_causal),
        np.int32(window_size_left),
        np.int32(window_size_right),
        np.bool_(return_softmax_lse),
        np.bool_(return_dropout_randval),
    ]

    return fn(*args)


# High-level Flash Attention API functions
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
    """Flash attention forward with kernel dispatch (exact aiter logic)."""

    debug_func_entry(
        "_flash_attn_forward",
        q=q,
        k=k,
        v=v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        bias=bias,
        alibi_slopes=alibi_slopes,
        return_lse=return_lse,
        return_softmax=return_softmax,
    )

    _, seqlen_q, nhead_q, hdim_q = q.shape
    _, seqlen_k, nhead_k, hdim_v = v.shape

    # causal=true is the same as causal=false in this case (aiter optimization)
    if seqlen_q == 1 and alibi_slopes is None:
        causal = False

    # Mask processing
    window_size_left = -1 if window_size_left >= seqlen_k else window_size_left
    window_size_right = -1 if window_size_right >= seqlen_k else window_size_right
    mask = causal and window_size_left == -1  # causal mask
    nmask = not causal and window_size_left == -1 and window_size_right == -1  # no mask
    swa = (window_size_left > 0) or (window_size_right > 0)  # sliding window

    # Kernel dispatch logic
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

    debug_kernel_dispatch(
        "FMHA_V3_FWD",
        can_use_v3,
        f"mask={mask}, nmask={nmask}, swa={swa}, hdim_q={hdim_q}, dtype={q.dtype}",
    )

    q, k, v = [_maybe_contiguous(x) for x in (q, k, v)]

    if can_use_v3 and seqlen_q > 128:
        debug_print("Using FMHA v3 forward kernel")
        return fmha_v3_fwd(
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
            None,
        )
    debug_print("Using MHA forward kernel")
    return mha_fwd(
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
        None,
    )


def _flash_attn_backward(
    dout: jnp.ndarray,
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    out: jnp.ndarray,
    softmax_lse: jnp.ndarray,
    dq: Optional[jnp.ndarray],
    dk: Optional[jnp.ndarray],
    dv: Optional[jnp.ndarray],
    dbias: Optional[jnp.ndarray],
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
) -> jnp.ndarray:
    """Flash attention backward with kernel dispatch."""

    debug_func_entry(
        "_flash_attn_backward",
        dout=dout,
        q=q,
        k=k,
        v=v,
        out=out,
        softmax_lse=softmax_lse,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        deterministic=deterministic,
        is_v3_atomic_fp32=is_v3_atomic_fp32,
        how_v3_bf16_cvt=how_v3_bf16_cvt,
    )

    # Check if we can use v3 backward
    can_use_v3_bwd = _can_impl_fmha_v3_bwd(
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
        is_v3_atomic_fp32,
    )

    debug_kernel_dispatch(
        "FMHA_V3_BWD",
        can_use_v3_bwd,
        f"causal={causal}, deterministic={deterministic}, dropout_p={dropout_p}, "
        f"hdim={q.shape[-1]}, is_v3_atomic_fp32={is_v3_atomic_fp32}",
    )

    # Ensure tensors are contiguous
    dout, q, k, v, out = [_maybe_contiguous(x) for x in (dout, q, k, v, out)]

    if can_use_v3_bwd:
        debug_print("Using FMHA v3 backward kernel")
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
            dq,
            dk,
            dv,
            alibi_slopes,
            rng_state,
            None,
        )
    else:
        debug_print("Using MHA backward kernel")
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
            dq,
            dk,
            dv,
            dbias,
            bias,
            alibi_slopes,
            rng_state,
            None,
        )

    return results[3]  # Return softmax_d


def _flash_attn_varlen_forward(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    cu_seqlens_q: jnp.ndarray,
    cu_seqlens_k: jnp.ndarray,
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
    block_table: Optional[jnp.ndarray] = None,
    out: Optional[jnp.ndarray] = None,
    zero_tensors: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Flash attention varlen forward."""

    q, k, v = [_maybe_contiguous(x) for x in (q, k, v)]
    return mha_varlen_fwd(
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


def _flash_attn_varlen_backward(
    dout: jnp.ndarray,
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    out: jnp.ndarray,
    softmax_lse: jnp.ndarray,
    dq: Optional[jnp.ndarray],
    dk: Optional[jnp.ndarray],
    dv: Optional[jnp.ndarray],
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
) -> jnp.ndarray:
    """Flash attention varlen backward with kernel dispatch."""

    # Check if we can use v3 varlen backward
    can_use_v3_varlen_bwd = _can_impl_fmha_v3_varlen_bwd(
        dout,
        q,
        k,
        v,
        dropout_p,
        causal,
        window_size_left,
        window_size_right,
        alibi_slopes,
        deterministic,
        is_v3_atomic_fp32,
    )

    # Ensure tensors are contiguous
    dout, q, k, v, out = [_maybe_contiguous(x) for x in (dout, q, k, v, out)]

    if can_use_v3_varlen_bwd:
        results = fmha_v3_varlen_bwd(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
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
            is_v3_atomic_fp32,
            how_v3_bf16_cvt,
            dq,
            dk,
            dv,
            alibi_slopes,
            rng_state,
            None,
        )
    else:
        results = mha_varlen_bwd(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
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
            dq,
            dk,
            dv,
            alibi_slopes,
            rng_state,
            None,
        )

    return results[3]  # Return softmax_d


def _mha_batch_prefill(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    cu_seqlens_q: jnp.ndarray,
    kv_indptr: jnp.ndarray,
    kv_page_indices: jnp.ndarray,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    logits_soft_cap: float = 0.0,
    window_size_left: int = -1,
    window_size_right: int = -1,
    alibi_slopes: Optional[jnp.ndarray] = None,
    return_lse: bool = False,
    return_softmax: bool = False,
    zero_tensors: bool = False,
    out: jnp.ndarray = None,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """MHA batch prefill implementation."""

    q, k, v = [_maybe_contiguous(x) for x in (q, k, v)]
    return mha_batch_prefill(
        q,
        k,
        v,
        cu_seqlens_q,
        kv_indptr,
        kv_page_indices,
        max_seqlen_q,
        max_seqlen_k,
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
        alibi_slopes,
        None,
    )


# JAX Autograd Function for Flash Attention
class FlashAttnFunc:
    @staticmethod
    def forward(
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
        window_size: Tuple[int, int],
        bias: Optional[jnp.ndarray],
        alibi_slopes: Optional[jnp.ndarray],
        deterministic: bool,
        return_lse: bool,
        return_softmax: bool,
        is_v3_atomic_fp32: Optional[bool] = True,
        how_v3_bf16_cvt: Optional[int] = 1,
    ):
        """Flash attention forward pass."""
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        head_size_q_og = q.shape[3]
        head_size_v_og = v.shape[3]

        # Pad head dimensions to multiples of 8
        if head_size_q_og % 8 != 0:
            pad_q = 8 - head_size_q_og % 8
            q = jnp.pad(q, ((0, 0), (0, 0), (0, 0), (0, pad_q)))
            k = jnp.pad(k, ((0, 0), (0, 0), (0, 0), (0, pad_q)))
        if head_size_v_og % 8 != 0:
            pad_v = 8 - head_size_v_og % 8
            v = jnp.pad(v, ((0, 0), (0, 0), (0, 0), (0, pad_v)))

        out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=int(window_size[0]),
            window_size_right=int(window_size[1]),
            bias=bias,
            alibi_slopes=alibi_slopes,
            return_lse=return_lse,
            return_softmax=return_softmax and dropout_p > 0,
        )

        out = out_padded[..., :head_size_v_og]

        result = [out]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(S_dmask)

        return result[0] if len(result) == 1 else tuple(result)


class FlashAttnVarlenFunc:
    @staticmethod
    def forward(
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        cu_seqlens_q: jnp.ndarray,
        cu_seqlens_k: jnp.ndarray,
        max_seqlen_q: int,
        max_seqlen_k: int,
        min_seqlen_q: int,
        dropout_p: float,
        softmax_scale: float,
        logits_soft_cap: float,
        causal: bool,
        window_size: Tuple[int, int],
        bias: Optional[jnp.ndarray],
        alibi_slopes: Optional[jnp.ndarray],
        deterministic: bool,
        return_lse: bool,
        return_softmax: bool,
        block_table: Optional[jnp.ndarray],
        out: Optional[jnp.ndarray],
        is_v3_atomic_fp32: Optional[bool] = True,
        how_v3_bf16_cvt: Optional[int] = 1,
    ):
        """Flash attention varlen forward pass."""
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        head_size_q_og = q.shape[-1]
        head_size_v_og = v.shape[-1]

        # Pad head dimensions to multiples of 8
        if head_size_q_og % 8 != 0:
            pad_q = 8 - head_size_q_og % 8
            q = jnp.pad(q, ((0, 0), (0, 0), (0, pad_q)))
            k = jnp.pad(k, ((0, 0), (0, 0), (0, pad_q)))
        if head_size_v_og % 8 != 0:
            pad_v = 8 - head_size_v_og % 8
            v = jnp.pad(v, ((0, 0), (0, 0), (0, pad_v)))

        out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_varlen_forward(
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
            causal=causal,
            logits_soft_cap=logits_soft_cap,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            bias=bias,
            alibi_slopes=alibi_slopes,
            return_lse=return_lse,
            return_softmax=return_softmax and dropout_p > 0,
            block_table=block_table,
            out=out,
        )

        out = out_padded[..., :head_size_v_og]

        result = [out]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(S_dmask)

        return result[0] if len(result) == 1 else tuple(result)


# Public API functions with custom VJP for autodiff
@jax.custom_vjp
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
    Flash Attention function compatible with standard API.

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
        Output tensor, optionally with lse and attention probabilities
    """
    return FlashAttnFunc.forward(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        deterministic,
        return_lse,
        return_attn_probs,
    )


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

    # Always get LSE for backward pass, regardless of return_lse
    out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
        q_padded,
        k_padded,
        v_padded,
        dropout_p,
        softmax_scale,
        causal=causal,
        window_size_left=int(window_size[0]),
        window_size_right=int(window_size[1]),
        bias=bias,
        alibi_slopes=alibi_slopes,
        return_lse=True,  # Always return for backward
        return_softmax=return_attn_probs and dropout_p > 0,
    )

    out = out_padded[..., :head_size_v_og]

    # Prepare return value based on user requirements
    if return_lse and return_attn_probs:
        result = (out, softmax_lse, S_dmask)
    elif return_lse:
        result = (out, softmax_lse)
    elif return_attn_probs:
        result = (out, S_dmask)
    else:
        result = out

    # Residuals needed for backward pass
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
        window_size,
        bias,
        alibi_slopes,
        deterministic,
        head_size_q_og,  # Store original Q head dimension
        head_size_v_og,  # Store original V head dimension
    )

    return result, residuals


def _flash_attn_func_bwd(residuals, grad_outputs):
    """Backward pass using residuals and output gradients."""
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

    # Determine which backward kernel to use (same logic as forward)
    _, seqlen_q, nhead_q, hdim_q = q_padded.shape
    _, seqlen_k, nhead_k, hdim_v = v_padded.shape

    can_use_v3_bwd = _can_impl_fmha_v3_bwd(
        dout_padded,
        q_padded,
        k_padded,
        v_padded,
        None,  # dk
        None,  # dv
        None,  # dbias
        res_dropout_p,
        res_causal,
        res_window_size[0],
        res_window_size[1],
        res_bias,
        res_alibi_slopes,
        res_deterministic,
        True,  # is_v3_atomic_fp32
    )

    # Call the appropriate backward kernel
    try:
        if can_use_v3_bwd:
            grads = fmha_v3_bwd(
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
                res_deterministic,
                True,  # is_v3_atomic_fp32
                1,  # how_v3_bf16_cvt
                None,  # dq
                None,  # dk
                None,  # dv
                res_alibi_slopes,
                rng_state,
                None,  # gen
            )
        else:
            grads = mha_bwd(
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
                res_deterministic,
                None,  # dq
                None,  # dk
                None,  # dv
                None,  # dbias
                res_bias,
                res_alibi_slopes,
                rng_state,
                None,  # gen
            )

        # Extract gradients from results
        dq_padded, dk_padded, dv_padded, softmax_d = grads

    except Exception as e:
        debug_print(
            f"Flash attention backward failed: {e}, using zero gradients", "WARNING"
        )
        exit()
        # Fallback to zero gradients
        dq_padded = jnp.zeros_like(q_padded)
        dk_padded = jnp.zeros_like(k_padded)
        dv_padded = jnp.zeros_like(v_padded)

    # Unpad gradients to match original input dimensions
    # We need to get the original head dimensions from the residuals
    q_original, k_original, v_original = residuals[:3]
    head_size_q_og = q_original.shape[-1]
    head_size_k_og = k_original.shape[-1]

    # Correctly unpad the gradients to match original input shapes
    dq = dq_padded[..., :head_size_q_og]
    dk = dk_padded[..., :head_size_k_og]
    dv = dv_padded[..., :head_size_v_og]

    # Handle bias gradient if needed
    dbias = None
    if res_bias is not None:
        # TODO: Extract dbias from backward kernel results if supported
        dbias = None

    # Return gradients for all inputs (None for non-differentiable params)
    return (
        dq,  # q
        dk,  # k
        dv,  # v
        None,  # dropout_p
        None,  # softmax_scale
        None,  # causal
        None,  # window_size
        dbias,  # bias
        None,  # alibi_slopes
        None,  # deterministic
        None,  # return_lse
        None,  # return_attn_probs
    )


# Register the custom VJP
flash_attn_func.defvjp(_flash_attn_func_fwd, _flash_attn_func_bwd)


def flash_attn_varlen_func(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    cu_seqlens_q: jnp.ndarray,
    cu_seqlens_k: jnp.ndarray,
    max_seqlen_q: int,
    max_seqlen_k: int,
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
    Flash Attention variable-length function.

    Args:
        q: Query tensor [total_q, nheads, headdim_q]
        k: Key tensor [total_k, nheads_k, headdim_q]
        v: Value tensor [total_k, nheads_k, headdim_v]
        cu_seqlens_q: Cumulative sequence lengths for queries [batch_size + 1]
        cu_seqlens_k: Cumulative sequence lengths for keys [batch_size + 1]
        max_seqlen_q: Maximum query sequence length
        max_seqlen_k: Maximum key sequence length
        min_seqlen_q: Minimum query sequence length
        dropout_p: Dropout probability
        softmax_scale: Scaling factor for softmax (default: 1/sqrt(headdim_q))
        logits_soft_cap: Soft cap for logits
        causal: Whether to apply causal masking
        window_size: (left, right) window sizes for sliding window attention
        bias: Optional bias tensor
        alibi_slopes: Optional ALiBi slopes
        deterministic: Whether to use deterministic algorithms
        return_lse: Whether to return log-sum-exp values
        return_attn_probs: Whether to return attention probabilities
        block_table: Optional block table for paged attention
        out: Optional pre-allocated output tensor

    Returns:
        Output tensor, optionally with lse and attention probabilities
    """
    return FlashAttnVarlenFunc.forward(
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
        causal,
        window_size,
        bias,
        alibi_slopes,
        deterministic,
        return_lse,
        return_attn_probs,
        block_table,
        out,
    )


def mha_batch_prefill_func(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    cu_seqlens_q: jnp.ndarray,
    kv_indptr: jnp.ndarray,
    kv_page_indices: jnp.ndarray,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    logits_soft_cap: float = 0.0,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[jnp.ndarray] = None,
    deterministic: bool = False,
    return_lse: bool = False,
    return_attn_probs: bool = False,
    out: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    MHA batch prefill function for paged attention.

    Args:
        q: Query tensor [total_q, nheads, headdim_q]
        k: Key tensor [total_k, nheads_k, headdim_q]
        v: Value tensor [total_k, nheads_k, headdim_v]
        cu_seqlens_q: Cumulative sequence lengths for queries [batch_size + 1]
        kv_indptr: KV page pointers
        kv_page_indices: KV page indices
        max_seqlen_q: Maximum query sequence length
        max_seqlen_k: Maximum key sequence length
        dropout_p: Dropout probability
        softmax_scale: Scaling factor for softmax (default: 1/sqrt(headdim_q))
        logits_soft_cap: Soft cap for logits
        causal: Whether to apply causal masking
        window_size: (left, right) window sizes for sliding window attention
        alibi_slopes: Optional ALiBi slopes
        deterministic: Whether to use deterministic algorithms
        return_lse: Whether to return log-sum-exp values
        return_attn_probs: Whether to return attention probabilities
        out: Optional pre-allocated output tensor

    Returns:
        Output tensor, optionally with lse and attention probabilities
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    head_size_q_og = q.shape[2]
    head_size_v_og = v.shape[2]

    # Pad head dimensions to multiples of 8
    if head_size_q_og % 8 != 0:
        pad_q = 8 - head_size_q_og % 8
        q = jnp.pad(q, ((0, 0), (0, 0), (0, pad_q)))
        k = jnp.pad(k, ((0, 0), (0, 0), (0, pad_q)))
    if head_size_v_og % 8 != 0:
        pad_v = 8 - head_size_v_og % 8
        v = jnp.pad(v, ((0, 0), (0, 0), (0, pad_v)))

    out_padded, softmax_lse, S_dmask, rng_state = _mha_batch_prefill(
        q,
        k,
        v,
        cu_seqlens_q,
        kv_indptr,
        kv_page_indices,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        alibi_slopes=alibi_slopes,
        return_lse=return_lse,
        return_softmax=return_attn_probs and dropout_p > 0,
        out=out,
    )

    out = out_padded[..., :head_size_v_og]

    result = [out]
    if return_lse:
        result.append(softmax_lse)
    if return_attn_probs:
        result.append(S_dmask)

    return result[0] if len(result) == 1 else tuple(result)
