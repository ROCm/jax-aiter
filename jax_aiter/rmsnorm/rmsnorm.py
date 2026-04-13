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

from ..ops.rmsnorm import rmsnorm_fwd as _rmsnorm_fwd_op


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
    y, _, _ = _rmsnorm_fwd_op(x, gamma, epsilon=epsilon)
    return y


def _rms_norm_fwd(x, gamma, epsilon):
    y, _, _ = _rmsnorm_fwd_op(x, gamma, epsilon=epsilon)
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
    y, residual_out, _ = _rmsnorm_fwd_op(x, gamma, residual,
                                         epsilon=epsilon, fused_add=True)
    return y, residual_out


def _rms_norm_with_add_fwd(x, residual, gamma, epsilon):
    y, residual_out, _ = _rmsnorm_fwd_op(x, gamma, residual,
                                         epsilon=epsilon, fused_add=True)
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
