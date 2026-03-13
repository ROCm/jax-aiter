# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""RMSNorm forward via AITER CK kernel, backward via JAX.

Forward: CK rmsnorm2d_fwd (fused square, mean, rsqrt, scale).
Backward: JAX-computed (no CK backward kernel exists yet).
Fused add variant: y = rms_norm(x + residual) * gamma, in one kernel call.
"""

from __future__ import annotations
from functools import partial

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


# ---------------------------------------------------------------------------
# rms_norm: y = rms_norm(x) * gamma.
# ---------------------------------------------------------------------------

@partial(jax.custom_vjp, nondiff_argnums=(2,))
def rms_norm(
    x: jnp.ndarray,
    gamma: jnp.ndarray,
    epsilon: float = 1e-6,
) -> jnp.ndarray:
    """RMSNorm: y = x / sqrt(mean(x^2) + eps) * gamma."""
    _ensure_registered()

    fn = _rmsnorm_fwd_call(x.shape, (0,), x.shape[:-1], x.dtype)
    y, _, _ = fn(x, gamma, _empty(x.dtype),
                 epsilon=epsilon, save_rms=False, fused_add=0)
    return y


def _rms_norm_fwd(x, gamma, epsilon):
    _ensure_registered()

    fn = _rmsnorm_fwd_call(x.shape, (0,), x.shape[:-1], x.dtype)
    y, _, _ = fn(x, gamma, _empty(x.dtype),
                 epsilon=epsilon, save_rms=False, fused_add=0)
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


# ---------------------------------------------------------------------------
# rms_norm_with_add: y = rms_norm(x + residual) * gamma.
# Also returns x + residual as residual_out (one kernel, one memory pass).
# ---------------------------------------------------------------------------

@partial(jax.custom_vjp, nondiff_argnums=(3,))
def rms_norm_with_add(
    x: jnp.ndarray,
    residual: jnp.ndarray,
    gamma: jnp.ndarray,
    epsilon: float = 1e-6,
) -> tuple:
    """Fused add + RMSNorm: y = rms_norm(x + residual) * gamma.

    Returns (y, residual_out) where residual_out = x + residual.
    """
    _ensure_registered()

    fn = _rmsnorm_fwd_call(x.shape, x.shape, x.shape[:-1], x.dtype)
    y, residual_out, _ = fn(x, gamma, residual,
                            epsilon=epsilon, save_rms=False, fused_add=1)
    return y, residual_out


def _rms_norm_with_add_fwd(x, residual, gamma, epsilon):
    _ensure_registered()

    fn = _rmsnorm_fwd_call(x.shape, x.shape, x.shape[:-1], x.dtype)
    y, residual_out, _ = fn(x, gamma, residual,
                            epsilon=epsilon, save_rms=False, fused_add=1)
    return (y, residual_out), (residual_out, gamma)


def _rms_norm_with_add_bwd(epsilon, residuals, grad_outputs):
    # residual_out = x + residual (saved from forward).
    x_plus_res, gamma = residuals
    grad_y, grad_res_out = grad_outputs

    x_f32 = x_plus_res.astype(jnp.float32)
    gamma_f32 = gamma.astype(jnp.float32)
    grad_y_f32 = grad_y.astype(jnp.float32)

    mean2 = jnp.mean(x_f32 ** 2, axis=-1, keepdims=True)
    inv_rms_f32 = jax.lax.rsqrt(mean2 + epsilon)

    x_hat = x_f32 * inv_rms_f32

    grad_gamma = jnp.sum(grad_y_f32 * x_hat,
                         axis=tuple(range(len(x_plus_res.shape) - 1)))

    grad_x_hat = grad_y_f32 * gamma_f32
    mean_term = jnp.mean(grad_x_hat * x_hat, axis=-1, keepdims=True)
    # Gradient w.r.t. (x + residual).
    grad_sum = inv_rms_f32 * (grad_x_hat - x_hat * mean_term)
    grad_sum = grad_sum.astype(x_plus_res.dtype)

    # grad_res_out flows through to both x and residual.
    grad_x = grad_sum + grad_res_out
    grad_residual = grad_sum + grad_res_out

    return grad_x, grad_residual, grad_gamma.astype(gamma.dtype)


rms_norm_with_add.defvjp(_rms_norm_with_add_fwd, _rms_norm_with_add_bwd)
