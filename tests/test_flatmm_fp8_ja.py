# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Tests for FP8 flat matmul (gfx942/MI300 only).

Skipped on gfx950 -- .co file only available for gfx942.
"""

import pytest
import jax
import jax.numpy as jnp

from jax_aiter.flatmm_fp8 import flatmm_fp8

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
    reason="Requires gfx942 (MI300) -- flatmm .co only available for gfx942"
)


def make_fp8_inputs(M, N, K, seed=0):
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    xq = (jax.random.normal(k1, (M, K), dtype=jnp.float32) * 0.1).astype(jnp.float8_e4m3fn)
    wq = (jax.random.normal(k2, (N, K), dtype=jnp.float32) * 0.1).astype(jnp.float8_e4m3fn)
    x_scale = jnp.ones((K // 128, M), dtype=jnp.float32)
    w_scale = jnp.ones((K // 128, N // 128), dtype=jnp.float32)
    return xq, wq, x_scale, w_scale



@pytest.mark.parametrize("M,N,K", [
    pytest.param(128, 256, 128, id="128x256x128"),
    pytest.param(256, 256, 128, id="256x256x128"),
    pytest.param(256, 512, 256, id="256x512x256"),
    pytest.param(1024, 1024, 256, id="1k_sq"),
])
def test_flatmm_fwd_shape(M, N, K):
    xq, wq, x_scale, w_scale = make_fp8_inputs(M, N, K)
    out = flatmm_fp8(xq, wq, x_scale, w_scale)
    assert out.shape == (M, N)
    assert out.dtype == jnp.float16
    assert jnp.all(jnp.isfinite(out))



@pytest.mark.parametrize("M,N,K", [
    pytest.param(128, 256, 128, id="128x256x128"),
    pytest.param(256, 512, 256, id="256x512x256"),
])
def test_flatmm_fwd_sanity(M, N, K):
    xq, wq, x_scale, w_scale = make_fp8_inputs(M, N, K, seed=42)
    out = flatmm_fp8(xq, wq, x_scale, w_scale)
    out_mean = float(jnp.mean(jnp.abs(out.astype(jnp.float32))))
    assert out_mean > 0, "Output is all zeros"



def test_flatmm_zeros():
    M, N, K = 128, 256, 128
    xq = jnp.zeros((M, K), dtype=jnp.float8_e4m3fn)
    wq = jnp.zeros((N, K), dtype=jnp.float8_e4m3fn)
    x_scale = jnp.ones((K // 128, M), dtype=jnp.float32)
    w_scale = jnp.ones((K // 128, N // 128), dtype=jnp.float32)
    out = flatmm_fp8(xq, wq, x_scale, w_scale)
    assert jnp.all(out == 0)



def test_flatmm_large_m():
    xq, wq, x_scale, w_scale = make_fp8_inputs(4096, 256, 128)
    out = flatmm_fp8(xq, wq, x_scale, w_scale)
    assert out.shape == (4096, 256)
    assert jnp.all(jnp.isfinite(out))
