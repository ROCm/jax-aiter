# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Tests for AITER fused silu_and_mul FFI kernel."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np


def _ref_silu_and_mul(gate, up):
    """Reference JAX implementation."""
    return jax.nn.silu(gate) * up


@pytest.fixture(autouse=True)
def _ensure_gpu():
    devs = jax.devices("gpu")
    if not devs:
        pytest.skip("No GPU devices available")


class TestSiluAndMulForward:
    """Test forward pass correctness."""

    @pytest.mark.parametrize("M,D", [
        (1, 128),
        (4, 256),
        (32, 1024),
        (64, 4096),
        (128, 11008),   # Llama2-7B intermediate_dim
        (256, 14336),   # Llama3-8B intermediate_dim
    ])
    def test_shapes(self, M, D):
        from jax_aiter.activation import silu_and_mul

        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        gate = jax.random.normal(k1, (M, D), dtype=jnp.bfloat16)
        up = jax.random.normal(k2, (M, D), dtype=jnp.bfloat16)

        result = silu_and_mul(gate, up)
        ref = _ref_silu_and_mul(gate, up)

        assert result.shape == (M, D)
        assert result.dtype == jnp.bfloat16
        np.testing.assert_allclose(
            result.astype(jnp.float32),
            ref.astype(jnp.float32),
            atol=1e-2, rtol=1e-2,
        )

    def test_3d_input(self):
        """Test with 3D input [batch, seq, D] (typical transformer shape)."""
        from jax_aiter.activation import silu_and_mul

        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        gate = jax.random.normal(k1, (2, 128, 4096), dtype=jnp.bfloat16)
        up = jax.random.normal(k2, (2, 128, 4096), dtype=jnp.bfloat16)

        result = silu_and_mul(gate, up)
        ref = _ref_silu_and_mul(gate, up)

        assert result.shape == (2, 128, 4096)
        np.testing.assert_allclose(
            result.astype(jnp.float32),
            ref.astype(jnp.float32),
            atol=1e-2, rtol=1e-2,
        )

    def test_fp16(self):
        """Test with fp16 input."""
        from jax_aiter.activation import silu_and_mul

        key = jax.random.PRNGKey(1)
        k1, k2 = jax.random.split(key)
        gate = jax.random.normal(k1, (32, 1024), dtype=jnp.float16)
        up = jax.random.normal(k2, (32, 1024), dtype=jnp.float16)

        result = silu_and_mul(gate, up)
        ref = _ref_silu_and_mul(gate, up)

        assert result.dtype == jnp.float16
        np.testing.assert_allclose(
            result.astype(jnp.float32),
            ref.astype(jnp.float32),
            atol=1e-2, rtol=1e-2,
        )


class TestSiluAndMulBackward:
    """Test backward pass (custom_vjp) correctness."""

    @pytest.mark.parametrize("M,D", [
        (4, 256),
        (32, 1024),
        (64, 4096),
    ])
    def test_grad_gate(self, M, D):
        """Test gradient w.r.t. gate."""
        from jax_aiter.activation import silu_and_mul

        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        gate = jax.random.normal(k1, (M, D), dtype=jnp.bfloat16)
        up = jax.random.normal(k2, (M, D), dtype=jnp.bfloat16)

        def aiter_fn(g):
            return jnp.sum(silu_and_mul(g, up))

        def ref_fn(g):
            return jnp.sum(_ref_silu_and_mul(g, up))

        grad_aiter = jax.grad(aiter_fn)(gate)
        grad_ref = jax.grad(ref_fn)(gate)

        np.testing.assert_allclose(
            grad_aiter.astype(jnp.float32),
            grad_ref.astype(jnp.float32),
            atol=5e-2, rtol=5e-2,
        )

    @pytest.mark.parametrize("M,D", [
        (4, 256),
        (32, 1024),
        (64, 4096),
    ])
    def test_grad_up(self, M, D):
        """Test gradient w.r.t. up."""
        from jax_aiter.activation import silu_and_mul

        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        gate = jax.random.normal(k1, (M, D), dtype=jnp.bfloat16)
        up = jax.random.normal(k2, (M, D), dtype=jnp.bfloat16)

        def aiter_fn(u):
            return jnp.sum(silu_and_mul(gate, u))

        def ref_fn(u):
            return jnp.sum(_ref_silu_and_mul(gate, u))

        grad_aiter = jax.grad(aiter_fn)(up)
        grad_ref = jax.grad(ref_fn)(up)

        np.testing.assert_allclose(
            grad_aiter.astype(jnp.float32),
            grad_ref.astype(jnp.float32),
            atol=5e-2, rtol=5e-2,
        )

    def test_grad_both(self):
        """Test gradients w.r.t. both gate and up simultaneously."""
        from jax_aiter.activation import silu_and_mul

        M, D = 32, 1024
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        gate = jax.random.normal(k1, (M, D), dtype=jnp.bfloat16)
        up = jax.random.normal(k2, (M, D), dtype=jnp.bfloat16)

        def aiter_fn(g, u):
            return jnp.sum(silu_and_mul(g, u))

        def ref_fn(g, u):
            return jnp.sum(_ref_silu_and_mul(g, u))

        grad_aiter = jax.grad(aiter_fn, argnums=(0, 1))(gate, up)
        grad_ref = jax.grad(ref_fn, argnums=(0, 1))(gate, up)

        for ga, gr in zip(grad_aiter, grad_ref):
            np.testing.assert_allclose(
                ga.astype(jnp.float32),
                gr.astype(jnp.float32),
                atol=5e-2, rtol=5e-2,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
