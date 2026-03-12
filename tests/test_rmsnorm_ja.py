# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Tests for AITER RMSNorm via jax_aiter.rmsnorm.

- Forward accuracy vs JAX reference (mean-square, rsqrt, scale).
- Backward accuracy via custom_vjp (JAX-computed grad).
- Covers bf16/fp16, typical MaxText hidden dims (4096, 8192, 14336).
- TE-style tolerances (eps^(2/3)).
"""

import pytest
import jax
import jax.numpy as jnp

from jax_aiter.rmsnorm import rms_norm


def rms_norm_ref(x, gamma, epsilon=1e-6):
    """JAX reference RMSNorm."""
    x_f32 = x.astype(jnp.float32)
    mean2 = jnp.mean(x_f32 ** 2, axis=-1, keepdims=True)
    inv_rms = jax.lax.rsqrt(mean2 + epsilon)
    y = (x_f32 * inv_rms * gamma.astype(jnp.float32))
    return y.astype(x.dtype)


def get_tolerances(dtype):
    eps = jnp.finfo(dtype).eps
    tol = float(eps ** (2.0 / 3.0))
    return tol, tol


def assert_close(actual, expected, dtype, name=""):
    atol, rtol = get_tolerances(dtype)
    a32 = actual.astype(jnp.float32)
    e32 = expected.astype(jnp.float32)
    max_diff = float(jnp.max(jnp.abs(a32 - e32)))
    max_ref = float(jnp.max(jnp.abs(e32)))
    rel_diff = max_diff / max(max_ref, 1e-6)
    assert max_diff < atol or rel_diff < rtol, \
        f"{name}: max_diff={max_diff:.6f} (atol={atol:.6f}), rel={rel_diff:.6f} (rtol={rtol:.6f})"


FWD_CONFIGS = [
    pytest.param(1, 128, jnp.bfloat16, id="1x128_bf16"),
    pytest.param(1, 4096, jnp.bfloat16, id="1x4096_bf16"),
    pytest.param(1, 4096, jnp.float16, id="1x4096_fp16"),
    pytest.param(4, 4096, jnp.bfloat16, id="4x4096_bf16"),
    pytest.param(32, 4096, jnp.bfloat16, id="32x4096_bf16"),
    pytest.param(32, 4096, jnp.float16, id="32x4096_fp16"),
    pytest.param(128, 4096, jnp.bfloat16, id="128x4096_bf16"),
    pytest.param(64, 8192, jnp.bfloat16, id="64x8192_bf16"),
    pytest.param(16, 8192, jnp.float16, id="16x8192_fp16"),
    pytest.param(1, 11008, jnp.bfloat16, id="1x11008_bf16"),
    pytest.param(8, 14336, jnp.bfloat16, id="8x14336_bf16"),
]

BWD_CONFIGS = [
    pytest.param(4, 4096, jnp.bfloat16, id="4x4096_bf16"),
    pytest.param(4, 4096, jnp.float16, id="4x4096_fp16"),
    pytest.param(32, 4096, jnp.bfloat16, id="32x4096_bf16"),
    pytest.param(16, 8192, jnp.bfloat16, id="16x8192_bf16"),
]


def make_inputs(m, n, dtype, seed=0):
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (m, n), dtype=dtype)
    gamma = jnp.ones((n,), dtype=dtype)
    return x, gamma


@pytest.mark.parametrize("m,n,dtype", FWD_CONFIGS)
def test_rmsnorm_fwd_shape(m, n, dtype):
    """Forward shape, dtype, finiteness."""
    x, gamma = make_inputs(m, n, dtype)
    y = rms_norm(x, gamma, epsilon=1e-6)
    assert y.shape == (m, n)
    assert y.dtype == dtype
    assert jnp.all(jnp.isfinite(y)), "NaN/Inf in output"


@pytest.mark.parametrize("m,n,dtype", FWD_CONFIGS)
def test_rmsnorm_fwd_accuracy(m, n, dtype):
    """Forward accuracy vs JAX reference."""
    x, gamma = make_inputs(m, n, dtype, seed=42)
    y = rms_norm(x, gamma, epsilon=1e-6)
    ref = rms_norm_ref(x, gamma, epsilon=1e-6)
    assert_close(y, ref, dtype, "fwd")


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16], ids=["fp16", "bf16"])
def test_rmsnorm_fwd_with_scale(dtype):
    """Forward with non-trivial gamma."""
    key = jax.random.PRNGKey(7)
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (16, 4096), dtype=dtype)
    gamma = jax.random.uniform(k2, (4096,), dtype=dtype, minval=0.5, maxval=2.0)
    y = rms_norm(x, gamma, epsilon=1e-6)
    ref = rms_norm_ref(x, gamma, epsilon=1e-6)
    assert_close(y, ref, dtype, "fwd_scale")


def test_rmsnorm_fwd_epsilon():
    """Different epsilon values."""
    x, gamma = make_inputs(8, 4096, jnp.bfloat16, seed=3)
    for eps in [1e-5, 1e-6, 1e-8]:
        y = rms_norm(x, gamma, epsilon=eps)
        ref = rms_norm_ref(x, gamma, epsilon=eps)
        assert_close(y, ref, jnp.bfloat16, f"eps={eps}")


def test_rmsnorm_fwd_deterministic():
    """Deterministic across calls."""
    x, gamma = make_inputs(16, 4096, jnp.bfloat16, seed=5)
    y1 = rms_norm(x, gamma)
    y2 = rms_norm(x, gamma)
    assert jnp.allclose(y1, y2, atol=0)


@pytest.mark.parametrize("m,n,dtype", BWD_CONFIGS)
def test_rmsnorm_bwd_shape(m, n, dtype):
    """Backward gradient shapes, dtypes, finiteness."""
    x, gamma = make_inputs(m, n, dtype, seed=10)

    def loss(x, gamma):
        return jnp.sum(rms_norm(x, gamma, epsilon=1e-6))

    dx, dgamma = jax.grad(loss, argnums=(0, 1))(x, gamma)
    assert dx.shape == x.shape, f"dx {dx.shape} != {x.shape}"
    assert dgamma.shape == gamma.shape, f"dgamma {dgamma.shape} != {gamma.shape}"
    assert dx.dtype == dtype
    assert dgamma.dtype == dtype
    assert jnp.all(jnp.isfinite(dx)), "dx NaN/Inf"
    assert jnp.all(jnp.isfinite(dgamma)), "dgamma NaN/Inf"


@pytest.mark.parametrize("m,n,dtype", BWD_CONFIGS)
def test_rmsnorm_bwd_accuracy(m, n, dtype):
    """Backward accuracy vs JAX reference."""
    x, gamma = make_inputs(m, n, dtype, seed=11)

    def aiter_loss(x, gamma):
        return jnp.sum(rms_norm(x, gamma, epsilon=1e-6))

    def ref_loss(x, gamma):
        return jnp.sum(rms_norm_ref(x, gamma, epsilon=1e-6))

    dx, dgamma = jax.grad(aiter_loss, argnums=(0, 1))(x, gamma)
    dx_r, dgamma_r = jax.grad(ref_loss, argnums=(0, 1))(x, gamma)

    for name, g, r in [("dx", dx, dx_r), ("dgamma", dgamma, dgamma_r)]:
        assert_close(g, r, dtype, name)
