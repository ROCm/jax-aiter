# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Raw NEOX RoPE forward FFI wrapper (self-contained HIP kernel).

Single-kernel op with no custom_vjp -- use jax_aiter.rope.rope for training.
Matches MaxText RotaryEmbedding.apply_rotary:
    out = x * cos + rotate_half(x) * sin
with full-width cos/sin (concat of the half-width cos/sin).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..ffi.registry import register_ffi_target


def _ensure_registered():
    register_ffi_target("RopeFwdJA", "ROCM")


def _rope_fwd_call(out_shape, dtype):
    call = jax.ffi.ffi_call(
        "RopeFwdJA",
        jax.ShapeDtypeStruct(out_shape, dtype),
        vmap_method="broadcast_all",
    )

    def _invoke(x, cos, sin):
        return call(x, cos, sin)

    return jax.jit(_invoke)


def rope_fwd(x, cos, sin):
    """NEOX RoPE forward via the self-contained AITER-style HIP kernel.

    Args:
        x:   [B, S, N, D] bf16 (query/key after projection, BSHD).
        cos: [B, S, D] bf16 (full-width, shared across the N heads).
        sin: [B, S, D] bf16.

    Returns:
        out: [B, S, N, D] bf16.
    """
    _ensure_registered()
    if x.ndim != 4:
        raise ValueError(f"rope_fwd expects x rank 4 [B,S,N,D], got {x.shape}")
    if cos.ndim != 3 or sin.ndim != 3:
        raise ValueError("rope_fwd expects cos/sin [B,S,D]")
    fn = _rope_fwd_call(x.shape, x.dtype)
    return fn(x, cos.astype(x.dtype), sin.astype(x.dtype))
