# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Tests for the fused gate+up FP4 projection."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from jax_aiter.gemm_fp4 import (
    gemm_fp4_bf16,
    gemm_fp4_gate_up_bf16,
    gemm_fp4_gate_up_raw,
)
from jax_aiter.gemm_fp4.quantizer import _fused_quant_available


pytestmark = pytest.mark.skipif(
    not _fused_quant_available(),
    reason="FP4 FFI not available (build with 'make ja_mods')"
)


def _rel_err(a, b):
    """Mean relative error (tolerant of FP4 quant noise)."""
    a = a.astype(jnp.float32)
    b = b.astype(jnp.float32)
    return float(jnp.mean(jnp.abs(a - b) / (jnp.abs(a) + 1e-3)))


@pytest.mark.parametrize("M,N,K", [
    (256, 512, 1024),
    (512, 1024, 2048),
])
def test_fused_forward_matches_separate(M, N, K):
    """Fused gate+up output is within FP4 quant noise of two separate calls.

    Exact bit-equality is not guaranteed because concatenating weights along
    N and passing them through a single cast may result in slightly different
    B-preshuffle row reorderings than casting the two halves independently.
    The numerical output is still equivalent within FP4 block quantization
    noise.
    """
    key = jax.random.PRNGKey(0)
    kx, kg, ku = jax.random.split(key, 3)
    x = jax.random.normal(kx, (M, K), dtype=jnp.bfloat16) * 0.1
    wg = jax.random.normal(kg, (N, K), dtype=jnp.bfloat16) * 0.1
    wu = jax.random.normal(ku, (N, K), dtype=jnp.bfloat16) * 0.1

    gate_sep = gemm_fp4_bf16(x, wg)
    up_sep = gemm_fp4_bf16(x, wu)
    gate_fused, up_fused = gemm_fp4_gate_up_raw(x, wg, wu)

    # Outputs agree within FP4 quantization noise.
    assert _rel_err(gate_sep, gate_fused) < 0.05
    assert _rel_err(up_sep, up_fused) < 0.05


def test_fused_backward_direction_matches_separate():
    """Gradients from fused path agree in direction with separate backward."""
    key = jax.random.PRNGKey(1)
    kx, kg, ku = jax.random.split(key, 3)
    M, N, K = 256, 512, 1024
    x = jax.random.normal(kx, (M, K), dtype=jnp.bfloat16) * 0.1
    wg = jax.random.normal(kg, (N, K), dtype=jnp.bfloat16) * 0.1
    wu = jax.random.normal(ku, (N, K), dtype=jnp.bfloat16) * 0.1

    def loss_separate(x, wg, wu):
        g = gemm_fp4_bf16(x, wg)
        u = gemm_fp4_bf16(x, wu)
        return jnp.mean(jax.nn.silu(g) * u)

    def loss_fused(x, wg, wu):
        g, u = gemm_fp4_gate_up_bf16(x, wg, wu)
        return jnp.mean(jax.nn.silu(g) * u)

    l_sep, grads_sep = jax.value_and_grad(loss_separate, argnums=(0, 1, 2))(x, wg, wu)
    l_fus, grads_fus = jax.value_and_grad(loss_fused, argnums=(0, 1, 2))(x, wg, wu)

    # Loss values agree (forward is identical modulo concat + split).
    assert float(jnp.abs(l_sep - l_fus)) < 1e-3, (
        f"Loss diverges: {float(l_sep)} vs {float(l_fus)}")

    # Gradients agree in direction — same FP4 kernel, same operands modulo concat.
    for name, gs, gf in zip(("dx", "dw_gate", "dw_up"), grads_sep, grads_fus):
        dot = float(jnp.sum(gs.astype(jnp.float32) * gf.astype(jnp.float32)))
        ns = float(jnp.linalg.norm(gs.astype(jnp.float32)))
        nf = float(jnp.linalg.norm(gf.astype(jnp.float32)))
        cos = dot / (ns * nf + 1e-8)
        assert cos > 0.98, f"{name} cos similarity too low: {cos:.4f}"


def test_fused_ffi_call_count_lower():
    """HLO of fused path uses fewer FFI calls than separate gate + up."""
    key = jax.random.PRNGKey(2)
    kx, kg, ku = jax.random.split(key, 3)
    M, N, K = 256, 512, 1024
    x = jax.random.normal(kx, (M, K), dtype=jnp.bfloat16)
    wg = jax.random.normal(kg, (N, K), dtype=jnp.bfloat16)
    wu = jax.random.normal(ku, (N, K), dtype=jnp.bfloat16)

    def mlp_separate(x, wg, wu):
        g = gemm_fp4_bf16(x, wg)
        u = gemm_fp4_bf16(x, wu)
        return jnp.mean(g + u)

    def mlp_fused(x, wg, wu):
        g, u = gemm_fp4_gate_up_bf16(x, wg, wu)
        return jnp.mean(g + u)

    sep_fn = jax.jit(jax.value_and_grad(mlp_separate, argnums=(0, 1, 2)))
    fused_fn = jax.jit(jax.value_and_grad(mlp_fused, argnums=(0, 1, 2)))

    hlo_sep = sep_fn.lower(x, wg, wu).compile().as_text()
    hlo_fused = fused_fn.lower(x, wg, wu).compile().as_text()

    ffi_sep = hlo_sep.count("custom_call_target")
    ffi_fused = hlo_fused.count("custom_call_target")
    # Fused must save at least 1 FFI call.
    assert ffi_fused < ffi_sep, (
        f"Fused ({ffi_fused}) should have fewer FFI calls than separate ({ffi_sep})")


def test_fused_shapes():
    key = jax.random.PRNGKey(3)
    kx, kg, ku = jax.random.split(key, 3)
    M, N, K = 128, 256, 512
    x = jax.random.normal(kx, (M, K), dtype=jnp.bfloat16)
    wg = jax.random.normal(kg, (N, K), dtype=jnp.bfloat16)
    wu = jax.random.normal(ku, (N, K), dtype=jnp.bfloat16)

    g, u = gemm_fp4_gate_up_bf16(x, wg, wu)
    assert g.shape == (M, N)
    assert u.shape == (M, N)
    assert g.dtype == jnp.bfloat16


def test_fused_rejects_shape_mismatch():
    key = jax.random.PRNGKey(4)
    kx, kg, ku = jax.random.split(key, 3)
    x = jax.random.normal(kx, (32, 256), dtype=jnp.bfloat16)
    wg = jax.random.normal(kg, (128, 256), dtype=jnp.bfloat16)
    wu_bad = jax.random.normal(ku, (64, 256), dtype=jnp.bfloat16)
    with pytest.raises(ValueError):
        gemm_fp4_gate_up_raw(x, wg, wu_bad)
