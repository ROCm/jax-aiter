# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Backward / wgrad tests for the canonical FP4 (MXFP4) recipe.

Verifies:
  1. The FP4 backward produces finite dA/dB with shapes matching dot_general.
  2. Under an FSDP-like sharded mesh, the wgrad custom_partitioning emits a
     psum (observable via HLO inspection) and produces a replicated output.

Single-recipe FP4 path. No env-flag toggling -- session-23 cleanup made
``gemm_fp4_bf16``'s backward always run the all-FP4 NT-layout wgrad.
"""

import importlib

import pytest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from jax_aiter.gemm_fp4.quantizer import _fused_quant_available


pytestmark = pytest.mark.skipif(
    not _fused_quant_available(),
    reason="CastMxfp4DualJA / GemmFp4FwdJA FFI not available (run 'make ja_mods')"
)


def _loss_fn(gemm_fn, x, w, target):
    out = gemm_fn(x, w)
    return jnp.mean((out - target) ** 2)


def test_fp4_backward_shapes_finite():
    """FP4 backward returns finite dA/dB with the expected shapes."""
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


@pytest.mark.multigpu
@pytest.mark.skipif(len(jax.devices()) < 2,
                    reason="wgrad sharding test needs >= 2 devices")
def test_fp4_wgrad_under_fsdp_mesh():
    """Under FSDP sharding of M (batch), wgrad emits psum and returns full dW."""
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

    compiled = jitted.lower(x, w).compile()
    hlo = compiled.as_text()
    has_reduction = (
        "all-reduce" in hlo
        or "reduce-scatter" in hlo
        or "psum" in hlo
    )
    assert has_reduction, "Expected wgrad to emit an FSDP-axis reduction in HLO"


@pytest.mark.multigpu
@pytest.mark.skipif(len(jax.devices()) < 2,
                    reason="keyed wgrad sharding test needs >= 2 devices")
def test_keyed_column_sr_wgrad_under_fsdp_mesh(monkeypatch):
    """The runtime SR key stays replicated while wgrad retains its reduction."""
    module = importlib.import_module("jax_aiter.gemm_fp4.gemm_fp4")
    monkeypatch.setattr(module, "_SR_GRAD", False)
    monkeypatch.setattr(module, "_SR_DGRAD_ROW", False)
    monkeypatch.setattr(module, "_SR_WGRAD_COL", True)
    monkeypatch.setattr(module, "_SR_ACT", False)
    monkeypatch.setattr(module, "_SR_WT", False)
    monkeypatch.setattr(module, "_SR_ANY", True)

    devices = jax.devices()
    mesh = Mesh(devices, axis_names=("fsdp",))
    x_spec = NamedSharding(mesh, P("fsdp", None))
    w_spec = NamedSharding(mesh, P(None, None))
    key_spec = NamedSharding(mesh, P(None))
    M = 256 * len(devices)
    N = K = 256
    root = jax.random.PRNGKey(71)
    kx, kw = jax.random.split(root)
    x = jax.device_put(
        jax.random.normal(kx, (M, K), dtype=jnp.bfloat16), x_spec
    )
    w = jax.device_put(
        jax.random.normal(kw, (N, K), dtype=jnp.bfloat16), w_spec
    )
    sr_key = jax.device_put(
        jnp.array([7, 11, 13, 17], dtype=jnp.uint32), key_spec
    )

    def loss_fn(x_, w_, key_):
        return jnp.mean(module.gemm_fp4_bf16(x_, w_, sr_key=key_))

    jitted = jax.jit(
        jax.value_and_grad(loss_fn, argnums=(0, 1)),
        in_shardings=(x_spec, w_spec, key_spec),
        out_shardings=(
            NamedSharding(mesh, P()),
            (x_spec, w_spec),
        ),
    )
    loss, (dx, dw) = jitted(x, w, sr_key)
    assert jnp.isfinite(loss)
    assert jnp.all(jnp.isfinite(dx))
    assert jnp.all(jnp.isfinite(dw))

    hlo = jitted.lower(x, w, sr_key).compile().as_text()
    assert "CastMxfp4DualKeyedSrJA" in hlo
    assert "all-reduce" in hlo or "reduce-scatter" in hlo or "psum" in hlo
