# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Raw RMSNorm forward FFI wrapper via AITER CK kernel.

Single-kernel op with no custom_vjp -- use jax_aiter.rmsnorm.rms_norm for training.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..ffi.registry import register_ffi_target


def _ensure_registered():
    register_ffi_target("RmsnormFwdJA", "ROCM")


def _empty(dtype):
    return jnp.zeros((0,), dtype=dtype)


def _rmsnorm_fwd_call(y_shape, residual_out_shape, inv_rms_shape, dtype):
    call = jax.ffi.ffi_call(
        "RmsnormFwdJA",
        (
            jax.ShapeDtypeStruct(y_shape, dtype),
            jax.ShapeDtypeStruct(residual_out_shape, dtype),
            jax.ShapeDtypeStruct(inv_rms_shape, dtype),
        ),
        vmap_method="broadcast_all",
    )

    def _invoke(x, gamma, residual, *, epsilon, save_rms, fused_add):
        return call(x, gamma, residual,
                    epsilon=np.float32(epsilon),
                    save_rms=save_rms,
                    fused_add=np.int32(fused_add))

    return jax.jit(_invoke, static_argnames=("epsilon", "save_rms", "fused_add"))


def rmsnorm_fwd(x, gamma, residual=None, *, epsilon=1e-6, save_rms=False, fused_add=False):
    """RMSNorm forward via AITER CK kernel (single FFI call).

    Args:
        x: [..., D] input tensor.
        gamma: [D] scale parameter.
        residual: Optional [..., D] tensor for fused add (x + residual before norm).
        epsilon: RMSNorm epsilon.
        save_rms: Whether to save inverse RMS for backward.
        fused_add: Whether to compute fused add+norm.

    Returns:
        (y, residual_out, inv_rms) tuple:
            y: [..., D] normalized output.
            residual_out: [..., D] (x + residual) if fused_add else empty.
            inv_rms: [...] inverse RMS if save_rms else empty.
    """
    _ensure_registered()
    if residual is None:
        residual = _empty(x.dtype)

    residual_out_shape = x.shape if fused_add else (0,)
    inv_rms_shape = x.shape[:-1] if save_rms else (0,)

    fn = _rmsnorm_fwd_call(x.shape, residual_out_shape, inv_rms_shape, x.dtype)
    return fn(x, gamma, residual,
              epsilon=epsilon, save_rms=save_rms, fused_add=int(fused_add))
