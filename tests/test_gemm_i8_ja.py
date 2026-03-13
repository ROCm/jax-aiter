# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Tests for INT8 GEMM via AITER ASM kernels (gfx942/MI300 only).

Skipped on gfx950 -- .co files only available for gfx942.
Tests FFI wrapper: shape, dtype, edge cases.
"""

import pytest
import jax
import jax.numpy as jnp

from jax_aiter.gemm_i8 import gemm_i8

import subprocess
def _get_arch():
    try:
        r = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=10)
        for line in r.stdout.split("\n"):
            if "gfx9" in line and "Name:" in line:
                return line.split(":")[-1].strip()
    except Exception:
        pass
    return "unknown"

pytestmark = pytest.mark.skipif(
    _get_arch() != "gfx942",
    reason="Requires gfx942 (MI300) -- INT8 GEMM .co only available for gfx942"
)


def make_i8_inputs(M, N, K, seed=0):
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    a = jax.random.randint(k1, (M, K), -128, 127, dtype=jnp.int8)
    b = jax.random.randint(k2, (N, K), -128, 127, dtype=jnp.int8)
    a_scale = jnp.ones((M, 1), dtype=jnp.float32) * 0.01
    b_scale = jnp.ones((1, N), dtype=jnp.float32) * 0.01
    bias = jnp.zeros((1, N), dtype=jnp.float32)
    return a, b, a_scale, b_scale, bias



@pytest.mark.parametrize("M,N,K", [
    pytest.param(128, 128, 128, id="128sq"),
    pytest.param(256, 256, 256, id="256sq"),
    pytest.param(128, 256, 128, id="rect"),
    pytest.param(1024, 1024, 512, id="1k"),
    pytest.param(64, 256, 128, id="small_m"),
])
def test_i8_gemm_fwd_shape(M, N, K):
    a, b, a_scale, b_scale, bias = make_i8_inputs(M, N, K)
    out = gemm_i8(a, b, a_scale, b_scale, bias)
    assert out.shape == (M, N)
    assert out.dtype == jnp.bfloat16
    assert jnp.all(jnp.isfinite(out))



@pytest.mark.parametrize("M,N,K", [
    pytest.param(128, 128, 128, id="128sq"),
    pytest.param(256, 256, 256, id="256sq"),
])
def test_i8_gemm_fwd_sanity(M, N, K):
    a, b, a_scale, b_scale, bias = make_i8_inputs(M, N, K, seed=42)
    out = gemm_i8(a, b, a_scale, b_scale, bias)
    out_mean = float(jnp.mean(jnp.abs(out.astype(jnp.float32))))
    assert out_mean > 0, "Output is all zeros"



def test_i8_gemm_with_bias():
    M, N, K = 128, 128, 128
    a, b, a_scale, b_scale, _ = make_i8_inputs(M, N, K)
    bias = jnp.ones((1, N), dtype=jnp.float32) * 0.5
    out = gemm_i8(a, b, a_scale, b_scale, bias)
    assert jnp.all(jnp.isfinite(out))



def test_i8_gemm_zeros():
    M, N, K = 64, 128, 128
    a = jnp.zeros((M, K), dtype=jnp.int8)
    b = jnp.zeros((N, K), dtype=jnp.int8)
    a_scale = jnp.ones((M, 1), dtype=jnp.float32)
    b_scale = jnp.ones((1, N), dtype=jnp.float32)
    bias = jnp.zeros((1, N), dtype=jnp.float32)
    out = gemm_i8(a, b, a_scale, b_scale, bias)
    assert jnp.all(out == 0)
