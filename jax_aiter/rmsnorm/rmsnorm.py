# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""RMSNorm forward via AITER CK kernel, backward via JAX.

Forward: CK rmsnorm2d_fwd (fused square, mean, rsqrt, scale).
Backward: JAX-computed (no CK backward kernel exists yet).
"""

from __future__ import annotations
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ..ffi.registry import register_ffi_target


def _ensure_registered():
    register_ffi_target("RmsnormFwdJA", "ROCM")


def _rmsnorm_fwd_call(y_shape, inv_rms_shape, dtype):
    call = jax.ffi.ffi_call(
        "RmsnormFwdJA",
        (
            jax.ShapeDtypeStruct(y_shape, dtype),
            jax.ShapeDtypeStruct(inv_rms_shape, dtype),
        ),
        vmap_method="broadcast_all",
    )

    def _invoke(x, gamma, *, epsilon, save_rms):
        return call(x, gamma,
                    epsilon=np.float32(epsilon),
                    save_rms=save_rms)

    return jax.jit(_invoke, static_argnames=("epsilon", "save_rms"))


@partial(jax.custom_vjp, nondiff_argnums=(2,))
def rms_norm(
    x: jnp.ndarray,
    gamma: jnp.ndarray,
    epsilon: float = 1e-6,
) -> jnp.ndarray:
    """RMSNorm: y = x / sqrt(mean(x^2) + eps) * gamma."""
    _ensure_registered()

    y_shape = x.shape
    inv_rms_shape = x.shape[:-1]

    fn = _rmsnorm_fwd_call(y_shape, inv_rms_shape, x.dtype)
    y, _ = fn(x, gamma, epsilon=epsilon, save_rms=False)
    return y


def _rms_norm_fwd(x, gamma, epsilon):
    _ensure_registered()

    y_shape = x.shape
    inv_rms_shape = x.shape[:-1]

    fn = _rmsnorm_fwd_call(y_shape, inv_rms_shape, x.dtype)
    y, _ = fn(x, gamma, epsilon=epsilon, save_rms=False)

    return y, (x, gamma)


def _rms_norm_bwd(epsilon, residuals, grad_y):
    x, gamma = residuals

    x_f32 = x.astype(jnp.float32)
    gamma_f32 = gamma.astype(jnp.float32)
    grad_y_f32 = grad_y.astype(jnp.float32)

    mean2 = jnp.mean(x_f32 ** 2, axis=-1, keepdims=True)
    inv_rms_f32 = jax.lax.rsqrt(mean2 + epsilon)

    x_hat = x_f32 * inv_rms_f32

    grad_gamma = jnp.sum(grad_y_f32 * x_hat,
                         axis=tuple(range(len(x.shape) - 1)))

    grad_x_hat = grad_y_f32 * gamma_f32
    mean_term = jnp.mean(grad_x_hat * x_hat, axis=-1, keepdims=True)
    grad_x = inv_rms_f32 * (grad_x_hat - x_hat * mean_term)

    return grad_x.astype(x.dtype), grad_gamma.astype(gamma.dtype)


rms_norm.defvjp(_rms_norm_fwd, _rms_norm_bwd)
