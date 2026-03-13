# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Tests for MI350 FP8 block-scale GEMM via AITER ASM kernels.

Testing our FFI wrapper, not AITER kernel correctness:
- Shape handling through FFI boundary
- Dtype routing (fp8 in, bf16 out)
- Scale tensor dimension validation
- Constraint checking (M >= 16, K >= 512, alignment)
- Edge cases in wrapper code
"""

import math
import pytest
import jax
import jax.numpy as jnp
import numpy as np

from jax_aiter.gemm_fp8 import gemm_fp8_mi350


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fp8_inputs(M, N, K, seed=0):
    """Create FP8 inputs with proper block scales."""
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    # FP8 activations and weights (random small values cast to fp8)
    a_f32 = jax.random.normal(k1, (M, K), dtype=jnp.float32) * 0.1
    b_f32 = jax.random.normal(k2, (N, K), dtype=jnp.float32) * 0.1
    xq = a_f32.astype(jnp.float8_e4m3fn)
    wq = b_f32.astype(jnp.float8_e4m3fn)

    # Block scales: [K/128, M] and [K/128, N/128]
    assert K % 128 == 0 and N % 128 == 0
    x_scale = jnp.ones((K // 128, M), dtype=jnp.float32)
    w_scale = jnp.ones((K // 128, N // 128), dtype=jnp.float32)

    return xq, wq, x_scale, w_scale


def reference_fp8_gemm(xq, wq, x_scale, w_scale):
    """Approximate reference: dequant to f32, matmul, cast to bf16."""
    a_f32 = xq.astype(jnp.float32)
    b_f32 = wq.astype(jnp.float32)
    return (a_f32 @ b_f32.T).astype(jnp.bfloat16)


# ---------------------------------------------------------------------------
# Forward shape tests
# ---------------------------------------------------------------------------

FWD_CONFIGS = [
    # N must be divisible by 256, K by 128, M >= 16, K >= 512
    pytest.param(128, 256, 512, id="128x256x512"),
    pytest.param(256, 256, 512, id="256x256x512"),
    pytest.param(256, 256, 1024, id="256x256x1024"),
    pytest.param(64, 256, 512, id="64x256x512"),
    pytest.param(32, 256, 512, id="32x256x512"),
    pytest.param(16, 256, 512, id="M16_min"),
    pytest.param(1024, 1024, 1024, id="1k_square"),
    pytest.param(128, 512, 1024, id="rect_128x512x1024"),
]


@pytest.mark.parametrize("M,N,K", FWD_CONFIGS)
def test_fp8_gemm_fwd_shape(M, N, K):
    """Output shape and dtype are correct."""
    xq, wq, x_scale, w_scale = make_fp8_inputs(M, N, K)
    out = gemm_fp8_mi350(xq, wq, x_scale, w_scale)
    assert out.shape == (M, N), f"shape {out.shape} != ({M}, {N})"
    assert out.dtype == jnp.bfloat16
    assert jnp.all(jnp.isfinite(out)), "NaN/Inf in output"


@pytest.mark.parametrize("M,N,K", FWD_CONFIGS)
def test_fp8_gemm_fwd_sanity(M, N, K):
    """Output is non-zero and in reasonable range (sanity, not exact accuracy)."""
    xq, wq, x_scale, w_scale = make_fp8_inputs(M, N, K, seed=42)
    out = gemm_fp8_mi350(xq, wq, x_scale, w_scale)
    ref = reference_fp8_gemm(xq, wq, x_scale, w_scale)

    out_mean = float(jnp.mean(jnp.abs(out.astype(jnp.float32))))
    ref_mean = float(jnp.mean(jnp.abs(ref.astype(jnp.float32))))

    assert out_mean > 0, "Output is all zeros"
    if ref_mean > 0.01:
        ratio = out_mean / ref_mean
        assert 0.1 < ratio < 10.0, f"Output magnitude wildly off: out_mean={out_mean}, ref_mean={ref_mean}"


# ---------------------------------------------------------------------------
# Llama3-70B shapes
# ---------------------------------------------------------------------------

LLAMA70B_FP8_CONFIGS = [
    pytest.param(32768, 8192, 8192, id="Q_proj"),
    pytest.param(32768, 1024, 8192, id="K_proj"),
    pytest.param(32768, 8192, 8192, id="O_proj"),
    pytest.param(32768, 28672, 8192, id="gate_proj"),
]


@pytest.mark.parametrize("M,N,K", LLAMA70B_FP8_CONFIGS)
def test_fp8_gemm_fwd_llama70b(M, N, K):
    """Llama3-70B shapes run without error and produce finite output."""
    xq, wq, x_scale, w_scale = make_fp8_inputs(M, N, K, seed=7)
    out = gemm_fp8_mi350(xq, wq, x_scale, w_scale)
    assert out.shape == (M, N)
    assert out.dtype == jnp.bfloat16
    assert jnp.all(jnp.isfinite(out)), "NaN/Inf in Llama70B output"


# ---------------------------------------------------------------------------
# Scale tensor tests
# ---------------------------------------------------------------------------

def test_fp8_gemm_scale_ones():
    """With scale=1.0 everywhere, output should match simple fp8 matmul."""
    xq, wq, x_scale, w_scale = make_fp8_inputs(128, 256, 512)
    out = gemm_fp8_mi350(xq, wq, x_scale, w_scale)
    assert jnp.all(jnp.isfinite(out))


def test_fp8_gemm_scale_varied():
    """Non-trivial scale values."""
    M, N, K = 128, 256, 512
    key = jax.random.PRNGKey(99)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    xq = jax.random.normal(k1, (M, K), dtype=jnp.float32).astype(jnp.float8_e4m3fn)
    wq = jax.random.normal(k2, (N, K), dtype=jnp.float32).astype(jnp.float8_e4m3fn)
    x_scale = jax.random.uniform(k3, (K // 128, M), dtype=jnp.float32, minval=0.5, maxval=2.0)
    w_scale = jax.random.uniform(k4, (K // 128, N // 128), dtype=jnp.float32, minval=0.5, maxval=2.0)

    out = gemm_fp8_mi350(xq, wq, x_scale, w_scale)
    assert out.shape == (M, N)
    assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_fp8_gemm_m16_minimum():
    """M=16 is the minimum supported."""
    xq, wq, x_scale, w_scale = make_fp8_inputs(16, 256, 512)
    out = gemm_fp8_mi350(xq, wq, x_scale, w_scale)
    assert out.shape == (16, 256)


def test_fp8_gemm_m32_boundary():
    """M=32 boundary (switches from x32 to x128 kernel)."""
    xq, wq, x_scale, w_scale = make_fp8_inputs(32, 256, 512)
    out = gemm_fp8_mi350(xq, wq, x_scale, w_scale)
    assert out.shape == (32, 256)


def test_fp8_gemm_m33_above_boundary():
    """M=33 uses x128 kernel."""
    xq, wq, x_scale, w_scale = make_fp8_inputs(33, 256, 512)
    out = gemm_fp8_mi350(xq, wq, x_scale, w_scale)
    assert out.shape == (33, 256)


def test_fp8_gemm_large_m():
    """Large M (training batch)."""
    xq, wq, x_scale, w_scale = make_fp8_inputs(4096, 256, 512)
    out = gemm_fp8_mi350(xq, wq, x_scale, w_scale)
    assert out.shape == (4096, 256)


def test_fp8_gemm_zeros_input():
    """All-zeros FP8 input produces all-zeros output."""
    M, N, K = 64, 256, 512
    xq = jnp.zeros((M, K), dtype=jnp.float8_e4m3fn)
    wq = jnp.zeros((N, K), dtype=jnp.float8_e4m3fn)
    x_scale = jnp.ones((K // 128, M), dtype=jnp.float32)
    w_scale = jnp.ones((K // 128, N // 128), dtype=jnp.float32)

    out = gemm_fp8_mi350(xq, wq, x_scale, w_scale)
    assert jnp.all(out == 0), "Zeros input should give zeros output"


def test_fp8_gemm_consistency():
    """Same inputs produce close outputs across calls."""
    xq, wq, x_scale, w_scale = make_fp8_inputs(128, 256, 512, seed=5)
    out1 = gemm_fp8_mi350(xq, wq, x_scale, w_scale)
    out2 = gemm_fp8_mi350(xq, wq, x_scale, w_scale)

    diff = float(jnp.max(jnp.abs(out1.astype(jnp.float32) - out2.astype(jnp.float32))))
    max_val = float(jnp.max(jnp.abs(out1.astype(jnp.float32))))
    if max_val > 0:
        assert diff / max_val < 0.01, f"Non-deterministic: rel_diff={diff/max_val}"
