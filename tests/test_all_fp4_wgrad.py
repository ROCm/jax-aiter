# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Integration tests for AITER_ALL_FP4=1 training recipe.

Verifies:
  1. The all-FP4 backward produces finite dA/dB with shapes matching dot_general.
  2. Under an FSDP-like sharded mesh, the wgrad custom_partitioning emits a
     psum (observable via HLO inspection) and produces a replicated output.
  3. Loss value and per-layer gradients match the hybrid (FP4 fwd + FP8 dB)
     path within FP4 numeric noise.
"""

import os
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from jax_aiter.gemm_fp4.quantizer import _fused_quant_available


pytestmark = pytest.mark.skipif(
    not _fused_quant_available(),
    reason="CastMxfp4DualJA / GemmFp4FwdJA FFI not available (run 'make ja_mods')"
)


def _set_all_fp4(value: bool):
    """Toggle AITER_ALL_FP4 and reset all gemm_fp4 module env caches."""
    os.environ["AITER_ALL_FP4"] = "1" if value else "0"
    from jax_aiter.gemm_fp4 import gemm_fp4 as _m
    _m._ALL_FP4_CACHE = None
    # Also reset FP4_DA cache so first lookup picks up the current env.
    _m._FP4_DA_CACHE = None
    _m._FP4_DB_CACHE = None
    _m._FP8_DB_CACHE = None
    _m._FUSED_QUANT_CACHE = None


def _loss_fn(gemm_fn, x, w, target):
    out = gemm_fn(x, w)
    return jnp.mean((out - target) ** 2)


def test_all_fp4_backward_shapes_finite():
    """All-FP4 backward returns finite dA/dB with the expected shapes."""
    _set_all_fp4(True)
    try:
        from jax_aiter.gemm_fp4 import gemm_fp4_bf16

        key = jax.random.PRNGKey(17)
        kx, kw, kt = jax.random.split(key, 3)
        M, N, K = 256, 512, 1024
        x = jax.random.normal(kx, (M, K), dtype=jnp.bfloat16)
        w = jax.random.normal(kw, (N, K), dtype=jnp.bfloat16)
        t = jax.random.normal(kt, (M, N), dtype=jnp.bfloat16)

        fn = jax.value_and_grad(
            lambda x, w: _loss_fn(gemm_fp4_bf16, x, w, t),
            argnums=(0, 1),
        )
        loss, (dx, dw) = jax.jit(fn)(x, w)

        assert jnp.isfinite(loss)
        assert dx.shape == x.shape
        assert dw.shape == w.shape
        assert jnp.all(jnp.isfinite(dx))
        assert jnp.all(jnp.isfinite(dw))
    finally:
        _set_all_fp4(False)


@pytest.mark.skipif(len(jax.devices()) < 2,
                    reason="wgrad sharding test needs >= 2 devices")
def test_all_fp4_wgrad_under_fsdp_mesh():
    """Under FSDP sharding of M (batch), wgrad emits psum and returns full dW."""
    _set_all_fp4(True)
    try:
        from jax_aiter.gemm_fp4 import gemm_fp4_bf16

        devices = jax.devices()
        mesh = Mesh(devices, axis_names=("fsdp",))
        x_spec = NamedSharding(mesh, P("fsdp", None))
        w_spec = NamedSharding(mesh, P(None, None))
        t_spec = NamedSharding(mesh, P("fsdp", None))

        key = jax.random.PRNGKey(23)
        kx, kw, kt = jax.random.split(key, 3)
        M = 512 * len(devices)
        N, K = 512, 1024
        x = jax.device_put(jax.random.normal(kx, (M, K), dtype=jnp.bfloat16), x_spec)
        w = jax.device_put(jax.random.normal(kw, (N, K), dtype=jnp.bfloat16), w_spec)
        t = jax.device_put(jax.random.normal(kt, (M, N), dtype=jnp.bfloat16), t_spec)

        fn = jax.value_and_grad(
            lambda x, w: _loss_fn(gemm_fp4_bf16, x, w, t),
            argnums=(0, 1),
        )
        jitted = jax.jit(
            fn,
            in_shardings=(x_spec, w_spec),
            out_shardings=(NamedSharding(mesh, P()),
                           (x_spec, w_spec)),
        )
        loss, (dx, dw) = jitted(x, w)

        assert jnp.isfinite(loss)
        assert jnp.all(jnp.isfinite(dx))
        assert jnp.all(jnp.isfinite(dw))

        # HLO should contain a psum/all-reduce over the fsdp axis for wgrad.
        compiled = jitted.lower(x, w).compile()
        hlo = compiled.as_text()
        has_reduction = (
            "all-reduce" in hlo
            or "reduce-scatter" in hlo
            or "psum" in hlo
        )
        assert has_reduction, "Expected wgrad to emit an FSDP-axis reduction in HLO"
    finally:
        _set_all_fp4(False)


def test_all_fp4_matches_hybrid_within_fp4_noise():
    """Loss under all-FP4 is close to hybrid (FP4 fwd + FP8 dB) within FP4 noise."""
    from jax_aiter.gemm_fp4 import gemm_fp4_bf16

    key = jax.random.PRNGKey(42)
    kx, kw, kt = jax.random.split(key, 3)
    M, N, K = 512, 256, 512
    x = jax.random.normal(kx, (M, K), dtype=jnp.bfloat16) * 0.1
    w = jax.random.normal(kw, (N, K), dtype=jnp.bfloat16) * 0.1
    t = jax.random.normal(kt, (M, N), dtype=jnp.bfloat16) * 0.1

    def loss(x, w):
        return _loss_fn(gemm_fp4_bf16, x, w, t)

    # Hybrid path (current production default).
    _set_all_fp4(False)
    loss_hybrid, (dw_hybrid,) = jax.jit(jax.value_and_grad(loss, argnums=(1,)))(x, w)

    # All-FP4 path (new recipe).
    _set_all_fp4(True)
    try:
        loss_all_fp4, (dw_all_fp4,) = (
            jax.jit(jax.value_and_grad(loss, argnums=(1,)))(x, w))
    finally:
        _set_all_fp4(False)

    # Forward was identical in both paths; loss should match exactly modulo
    # jit caching.
    assert jnp.allclose(loss_hybrid, loss_all_fp4, rtol=1e-3), (
        f"Loss mismatch: hybrid {loss_hybrid} vs all-FP4 {loss_all_fp4}")

    # dW differs because dB is FP4 (noisy) vs FP8 (less noisy) but the bulk
    # should agree in direction.
    dot = float(jnp.sum(dw_hybrid.astype(jnp.float32) * dw_all_fp4.astype(jnp.float32)))
    norm_h = float(jnp.linalg.norm(dw_hybrid.astype(jnp.float32)))
    norm_f = float(jnp.linalg.norm(dw_all_fp4.astype(jnp.float32)))
    cos_sim = dot / (norm_h * norm_f + 1e-8)
    assert cos_sim > 0.95, (
        f"dW direction drift too large: cos similarity {cos_sim:.3f}")
