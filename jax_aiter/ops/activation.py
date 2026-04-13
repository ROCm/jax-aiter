# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Raw fused SiLU-and-Mul activation FFI wrapper.

Single-kernel op with no custom_vjp -- use jax_aiter.activation.silu_and_mul for training.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..ffi.registry import register_ffi_target


def _ensure_registered():
    register_ffi_target("SiluAndMulJA", "ROCM")


def _silu_and_mul_ffi_call(out_shape, dtype):
    call = jax.ffi.ffi_call(
        "SiluAndMulJA",
        jax.ShapeDtypeStruct(out_shape, dtype),
        vmap_method="broadcast_all",
    )
    return jax.jit(call)


def silu_and_mul(gate, up):
    """Fused SiLU-and-Mul: silu(gate) * up via AITER CK kernel (single FFI call).

    Args:
        gate: [..., D] tensor (SiLU activation applied).
        up: [..., D] tensor (multiplied elementwise).

    Returns:
        [..., D] tensor: silu(gate) * up.
    """
    _ensure_registered()
    combined = jnp.concatenate([gate, up], axis=-1)
    out_shape = gate.shape
    fn = _silu_and_mul_ffi_call(out_shape, gate.dtype)
    return fn(combined)
