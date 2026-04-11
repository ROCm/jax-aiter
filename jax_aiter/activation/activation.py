# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Fused SiLU-and-Mul activation via AITER CK kernel, backward via JAX.

Forward: AITER CK fused kernel — silu(gate) * up in one pass.
Backward: JAX-computed (no AITER backward kernel exists).

Usage:
    from jax_aiter.activation import silu_and_mul
    result = silu_and_mul(gate, up)  # equivalent to jax.nn.silu(gate) * up
"""

from __future__ import annotations
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ..ffi.registry import register_ffi_target


def _ensure_registered():
    register_ffi_target("SiluAndMulJA", "ROCM")


# ---------------------------------------------------------------------------
# FFI call wrapper
# ---------------------------------------------------------------------------

def _silu_and_mul_ffi_call(out_shape, dtype):
    """Create a JIT-compiled FFI call for SiluAndMul."""
    call = jax.ffi.ffi_call(
        "SiluAndMulJA",
        jax.ShapeDtypeStruct(out_shape, dtype),
        vmap_method="broadcast_all",
    )
    return jax.jit(call)


def _silu_and_mul_ffi(gate: jnp.ndarray, up: jnp.ndarray) -> jnp.ndarray:
    """Low-level FFI call: concatenate gate/up → [M, 2*D], call kernel → [M, D]."""
    _ensure_registered()

    # Concatenate gate and up along last axis: [*, D] + [*, D] → [*, 2*D]
    combined = jnp.concatenate([gate, up], axis=-1)

    # Output shape is gate's shape (= up's shape).
    out_shape = gate.shape
    fn = _silu_and_mul_ffi_call(out_shape, gate.dtype)
    return fn(combined)


# ---------------------------------------------------------------------------
# silu_and_mul: fused silu(gate) * up with custom_vjp
# ---------------------------------------------------------------------------

@partial(jax.custom_vjp, nondiff_argnums=())
def silu_and_mul(gate: jnp.ndarray, up: jnp.ndarray) -> jnp.ndarray:
    """Fused SiLU-and-Mul: silu(gate) * up.

    Equivalent to jax.nn.silu(gate) * up, but fused into a single AITER
    kernel launch with MI350-optimized v_pk_mul_f32 ASM.

    Args:
        gate: [..., D] tensor (SiLU activation applied).
        up: [..., D] tensor (linear, multiplied elementwise).

    Returns:
        [..., D] tensor: silu(gate) * up.
    """
    return _silu_and_mul_ffi(gate, up)


def _silu_and_mul_fwd(gate, up):
    """Forward pass: compute silu(gate)*up, save residuals for backward."""
    result = _silu_and_mul_ffi(gate, up)
    return result, (gate, up)


def _silu_and_mul_bwd(residuals, grad_output):
    """Backward pass: compute gradients w.r.t. gate and up.

    Given y = silu(gate) * up:
        dy/d_gate = up * silu'(gate)
                  = up * sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
        dy/d_up   = silu(gate)
    """
    gate, up = residuals
    g = grad_output

    # Compute in float32 for numerical stability.
    gate_f32 = gate.astype(jnp.float32)
    up_f32 = up.astype(jnp.float32)
    g_f32 = g.astype(jnp.float32)

    # sigmoid(gate)
    sig = jax.nn.sigmoid(gate_f32)
    # silu(gate) = gate * sigmoid(gate)
    silu_gate = gate_f32 * sig
    # silu'(gate) = sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))
    #             = sigmoid(gate) + gate * sigmoid(gate) * (1 - sigmoid(gate))
    #             = sig + silu_gate * (1 - sig)
    silu_prime = sig + silu_gate * (1.0 - sig)

    # Gradients.
    grad_gate = g_f32 * up_f32 * silu_prime
    grad_up = g_f32 * silu_gate

    return grad_gate.astype(gate.dtype), grad_up.astype(up.dtype)


silu_and_mul.defvjp(_silu_and_mul_fwd, _silu_and_mul_bwd)
