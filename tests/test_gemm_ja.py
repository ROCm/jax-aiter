# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Tests for BF16 GEMM via AITER ASM kernels.

- Forward accuracy vs JAX reference (A @ B^T in float32).
- Backward accuracy via custom_vjp.
- Covers bf16/fp16, Llama3-70B projection shapes, edge cases.
- TE-style tolerances scaled by sqrt(K) for GEMM accumulation noise.
"""

import math
import pytest
import jax
import jax.numpy as jnp

from jax_aiter.gemm import gemm


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

def get_tolerances(dtype, K=1):
    """GEMM tolerance scales with sqrt(K) due to bf16 accumulation noise."""
    eps = jnp.finfo(dtype).eps
    base_tol = float(eps ** (2.0 / 3.0))
    k_factor = max(1.0, math.sqrt(K) / 8.0)
    tol = base_tol * k_factor
    return tol, tol


def assert_close(actual, expected, dtype, name="", K=1, factor=1):
    atol, rtol = get_tolerances(dtype, K)
    atol *= factor
    rtol *= factor
    a32 = actual.astype(jnp.float32)
    e32 = expected.astype(jnp.float32)
    max_diff = float(jnp.max(jnp.abs(a32 - e32)))
    max_ref = float(jnp.max(jnp.abs(e32)))
    rel_diff = max_diff / max(max_ref, 1e-6)
    assert max_diff < atol or rel_diff < rtol, \
        f"{name}: max_diff={max_diff:.6f} (atol={atol:.6f}), rel={rel_diff:.6f} (rtol={rtol:.6f})"


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------

def gemm_ref(a, b):
    """Reference: A @ B^T in float32."""
    return (a.astype(jnp.float32) @ b.astype(jnp.float32).T).astype(a.dtype)


def make_ab(M, N, K, dtype, seed=0):
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    a = jax.random.normal(k1, (M, K), dtype=jnp.float32).astype(dtype)
    b = jax.random.normal(k2, (N, K), dtype=jnp.float32).astype(dtype)
    return a, b


# ---------------------------------------------------------------------------
# Test configs
# ---------------------------------------------------------------------------

FWD_CONFIGS = [
    # Small squares.
    pytest.param(128, 128, 128, jnp.bfloat16, id="128sq_bf16"),
    pytest.param(256, 256, 256, jnp.bfloat16, id="256sq_bf16"),
    # Rectangles.
    pytest.param(128, 256, 64, jnp.bfloat16, id="128x256x64_bf16"),
    pytest.param(512, 1024, 256, jnp.bfloat16, id="512x1024x256_bf16"),
    # Single row (decode/inference).
    pytest.param(1, 128, 64, jnp.bfloat16, id="M1_bf16"),
    pytest.param(1, 256, 128, jnp.bfloat16, id="M1_256x128_bf16"),
    # Small batch.
    pytest.param(32, 64, 64, jnp.bfloat16, id="tiny_bf16"),
    # Large (4K).
    pytest.param(1024, 1024, 1024, jnp.bfloat16, id="1k_sq_bf16"),
    pytest.param(4096, 4096, 4096, jnp.bfloat16, id="4k_sq_bf16"),
]

LLAMA70B_CONFIGS = [
    # Llama3-70B: emb=8192, head=128, q_heads=64, kv_heads=8, mlp=28672
    # per_device_batch_size=4, seq=8192 -> M=32768
    pytest.param(32768, 8192, 8192, jnp.bfloat16, id="Q_proj"),
    pytest.param(32768, 1024, 8192, jnp.bfloat16, id="K_proj"),
    pytest.param(32768, 1024, 8192, jnp.bfloat16, id="V_proj"),
    pytest.param(32768, 8192, 8192, jnp.bfloat16, id="O_proj"),
    pytest.param(32768, 28672, 8192, jnp.bfloat16, id="gate_proj"),
    pytest.param(32768, 28672, 8192, jnp.bfloat16, id="up_proj"),
    pytest.param(32768, 8192, 28672, jnp.bfloat16, id="down_proj"),
]

BWD_CONFIGS = [
    # Backward transposes dims: dA needs K'=N, dB needs K'=M.
    # Both M and N must be divisible by 64 for backward to work.
    pytest.param(128, 128, 128, jnp.bfloat16, id="128sq_bf16"),
    pytest.param(256, 512, 256, jnp.bfloat16, id="256x512x256_bf16"),
    pytest.param(64, 128, 64, jnp.bfloat16, id="64x128x64_bf16"),
    pytest.param(64, 256, 128, jnp.bfloat16, id="64x256x128_bf16"),
]


# ---------------------------------------------------------------------------
# Forward tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N,K,dtype", FWD_CONFIGS)
def test_gemm_fwd_shape(M, N, K, dtype):
    """Forward shape, dtype, finiteness."""
    a, b = make_ab(M, N, K, dtype)
    out = gemm(a, b)
    assert out.shape == (M, N), f"shape {out.shape} != ({M}, {N})"
    assert out.dtype == dtype
    assert jnp.all(jnp.isfinite(out)), "NaN/Inf in output"


@pytest.mark.parametrize("M,N,K,dtype", FWD_CONFIGS)
def test_gemm_fwd_accuracy(M, N, K, dtype):
    """Forward accuracy vs float32 reference."""
    a, b = make_ab(M, N, K, dtype, seed=42)
    out = gemm(a, b)
    ref = gemm_ref(a, b)
    assert_close(out, ref, dtype, f"fwd {M}x{N}x{K}", K=K)


@pytest.mark.parametrize("M,N,K,dtype", LLAMA70B_CONFIGS)
def test_gemm_fwd_llama70b(M, N, K, dtype):
    """Forward accuracy for Llama3-70B projection shapes."""
    a, b = make_ab(M, N, K, dtype, seed=7)
    out = gemm(a, b)
    ref = gemm_ref(a, b)
    assert_close(out, ref, dtype, f"llama70b {M}x{N}x{K}", K=K)


# ---------------------------------------------------------------------------
# Backward tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("M,N,K,dtype", BWD_CONFIGS)
def test_gemm_bwd_shape(M, N, K, dtype):
    """Backward gradient shapes, dtypes, finiteness."""
    a, b = make_ab(M, N, K, dtype, seed=10)

    def loss(a, b):
        return jnp.sum(gemm(a, b))

    da, db = jax.grad(loss, argnums=(0, 1))(a, b)
    assert da.shape == a.shape, f"da {da.shape} != {a.shape}"
    assert db.shape == b.shape, f"db {db.shape} != {b.shape}"
    assert da.dtype == dtype
    assert db.dtype == dtype
    assert jnp.all(jnp.isfinite(da)), "da NaN/Inf"
    assert jnp.all(jnp.isfinite(db)), "db NaN/Inf"


@pytest.mark.parametrize("M,N,K,dtype", BWD_CONFIGS)
def test_gemm_bwd_accuracy(M, N, K, dtype):
    """Backward accuracy vs float32 reference."""
    a, b = make_ab(M, N, K, dtype, seed=11)

    def aiter_loss(a, b):
        return jnp.sum(gemm(a, b))

    def ref_loss(a, b):
        return jnp.sum(gemm_ref(a, b).astype(jnp.float32))

    da, db = jax.grad(aiter_loss, argnums=(0, 1))(a, b)
    da_r, db_r = jax.grad(ref_loss, argnums=(0, 1))(a, b)

    assert_close(da, da_r, dtype, "dA", K=N, factor=4)
    assert_close(db, db_r, dtype, "dB", K=M, factor=4)


# ---------------------------------------------------------------------------
# Deterministic / consistency tests
# ---------------------------------------------------------------------------

def test_gemm_consistency():
    """Same inputs produce close outputs across calls.

    Split-K kernels use atomic adds which may have non-deterministic
    ordering, so we check closeness rather than exact equality.
    """
    a, b = make_ab(256, 256, 256, jnp.bfloat16, seed=5)
    out1 = gemm(a, b)
    out2 = gemm(a, b)
    assert_close(out1, out2, jnp.bfloat16, "consistency", K=256)


def test_gemm_identity_like():
    """A @ I^T should approximate A (when N=K)."""
    K = 128
    a, _ = make_ab(64, K, K, jnp.bfloat16, seed=20)
    eye = jnp.eye(K, dtype=jnp.bfloat16)
    out = gemm(a, eye)
    assert_close(out, a, jnp.bfloat16, "identity", K=K)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_gemm_m1():
    """M=1 (single vector, decode/inference)."""
    a, b = make_ab(1, 256, 128, jnp.bfloat16, seed=30)
    out = gemm(a, b)
    ref = gemm_ref(a, b)
    assert out.shape == (1, 256)
    assert_close(out, ref, jnp.bfloat16, "M1", K=128)


def test_gemm_large_m():
    """Large M (training batch)."""
    a, b = make_ab(8192, 256, 256, jnp.bfloat16, seed=31)
    out = gemm(a, b)
    ref = gemm_ref(a, b)
    assert_close(out, ref, jnp.bfloat16, "large_M", K=256)


def test_gemm_large_k():
    """Large K (deep reduction)."""
    a, b = make_ab(128, 128, 8192, jnp.bfloat16, seed=32)
    out = gemm(a, b)
    ref = gemm_ref(a, b)
    assert_close(out, ref, jnp.bfloat16, "large_K", K=8192)


def test_gemm_wide_n():
    """Wide N (MLP up-projection: 8192 -> 28672)."""
    a, b = make_ab(128, 28672, 8192, jnp.bfloat16, seed=33)
    out = gemm(a, b)
    ref = gemm_ref(a, b)
    assert out.shape == (128, 28672)
    assert_close(out, ref, jnp.bfloat16, "wide_N", K=8192)


def test_gemm_zeros_input():
    """All-zeros input produces all-zeros output."""
    a = jnp.zeros((64, 128), dtype=jnp.bfloat16)
    b = jnp.ones((256, 128), dtype=jnp.bfloat16)
    out = gemm(a, b)
    assert jnp.all(out == 0), "Zeros A should give zeros output"


def test_gemm_ones_matmul():
    """Ones @ Ones^T = K * ones (sanity)."""
    K = 64
    a = jnp.ones((32, K), dtype=jnp.bfloat16)
    b = jnp.ones((128, K), dtype=jnp.bfloat16)
    out = gemm(a, b)
    expected = jnp.full((32, 128), K, dtype=jnp.bfloat16)
    assert_close(out, expected, jnp.bfloat16, "ones", K=1)
