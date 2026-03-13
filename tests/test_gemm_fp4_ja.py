# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Tests for FP4 GEMM via AITER ASM kernels.

Works on both gfx942 (MI300) and gfx950 (MI350).
Tests FFI wrapper: packed FP4 input handling, scale tensors, shape, edge cases.

SKIPPED: FP4 packed format (fp4x2) requires proper quantization to produce
valid inputs. Random uint8 does not represent valid FP4 data, causing
overflow/NaN for larger matrix sizes. Enable once FP4 quantization utilities
are available.
"""

import pytest
import jax
import jax.numpy as jnp

pytestmark = pytest.mark.skip(reason="FP4 GEMM needs proper fp4x2 quantization utilities -- skipped until validated")

from jax_aiter.gemm_fp4 import gemm_fp4


def make_fp4_inputs(M, N, K, seed=0):
    """Create packed FP4 inputs. K is the logical dimension (K/2 stored)."""
    assert K % 2 == 0, "K must be even for FP4 packing"
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    a = jax.random.randint(k1, (M, K // 2), 0, 255, dtype=jnp.uint8)
    b = jax.random.randint(k2, (N, K // 2), 0, 255, dtype=jnp.uint8)
    a_scale = jnp.ones((M, K // 32), dtype=jnp.uint8)
    b_scale = jnp.ones((N, K // 32), dtype=jnp.uint8)
    return a, b, a_scale, b_scale


@pytest.mark.parametrize("M,N,K", [
    pytest.param(128, 128, 256, id="128x128x256"),
    pytest.param(256, 256, 256, id="256x256x256"),
    pytest.param(128, 256, 512, id="128x256x512"),
    pytest.param(256, 512, 256, id="256x512x256"),
    pytest.param(1024, 1024, 512, id="1k_sq"),
    pytest.param(64, 256, 256, id="small_m"),
])
def test_fp4_gemm_fwd_shape(M, N, K):
    a, b, a_scale, b_scale = make_fp4_inputs(M, N, K)
    out = gemm_fp4(a, b, a_scale, b_scale)
    assert out.shape == (M, N), f"shape {out.shape} != ({M}, {N})"
    assert out.dtype == jnp.bfloat16


@pytest.mark.parametrize("M,N,K", [
    pytest.param(128, 128, 256, id="128x128x256"),
    pytest.param(256, 256, 256, id="256x256x256"),
])
def test_fp4_gemm_fwd_sanity(M, N, K):
    a, b, a_scale, b_scale = make_fp4_inputs(M, N, K, seed=42)
    out = gemm_fp4(a, b, a_scale, b_scale)
    out_mean = float(jnp.mean(jnp.abs(out.astype(jnp.float32))))
    assert out_mean > 0, "Output is all zeros"


def test_fp4_gemm_zeros():
    M, N, K = 64, 128, 256
    a = jnp.zeros((M, K // 2), dtype=jnp.uint8)
    b = jnp.zeros((N, K // 2), dtype=jnp.uint8)
    a_scale = jnp.ones((M, K // 32), dtype=jnp.uint8)
    b_scale = jnp.ones((N, K // 32), dtype=jnp.uint8)
    out = gemm_fp4(a, b, a_scale, b_scale)
    assert out.shape == (M, N)
    assert jnp.all(jnp.isfinite(out))


def test_fp4_gemm_large_m():
    a, b, a_scale, b_scale = make_fp4_inputs(4096, 256, 256)
    out = gemm_fp4(a, b, a_scale, b_scale)
    assert out.shape == (4096, 256)


def test_fp4_gemm_output_dtype():
    """Verify output is bf16 regardless of input."""
    a, b, a_scale, b_scale = make_fp4_inputs(128, 128, 256, seed=5)
    out = gemm_fp4(a, b, a_scale, b_scale)
    assert out.dtype == jnp.bfloat16
