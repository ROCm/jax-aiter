# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Tests for MI350 FP8 block-scale GEMM via AITER ASM kernels.

Tests forward correctness, backward (custom_vjp), weight shuffle,
and the delayed scaling infrastructure.
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

def make_fp8_inputs(M, N, K, seed=0, with_shuffle=True):
    """Create FP8 inputs with proper block scales and optional weight shuffle."""
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)

    block_k = 128
    fp8_max = 448.0

    a_bf16 = jax.random.normal(k1, (M, K), dtype=jnp.bfloat16) * 0.1
    b_bf16 = jax.random.normal(k2, (N, K), dtype=jnp.bfloat16) * 0.1

    # Quantize activations: per-row-block (1x128)
    a_blocks = a_bf16.reshape(M, K // block_k, block_k)
    x_amax = jnp.max(jnp.abs(a_blocks), axis=-1, keepdims=True)
    x_scale_3d = jnp.where(x_amax == 0, 1.0, x_amax / fp8_max)
    xq = (a_blocks / x_scale_3d).astype(jnp.float8_e4m3fn).reshape(M, K)
    x_scale = x_scale_3d.squeeze(-1).transpose(1, 0).astype(jnp.float32)

    # Quantize weights: per-2D-block (128x128)
    b_blocks = b_bf16.reshape(N // block_k, block_k, K // block_k, block_k)
    w_amax = jnp.max(jnp.abs(b_blocks), axis=(1, 3), keepdims=True)
    w_scale_4d = jnp.where(w_amax == 0, 1.0, w_amax / fp8_max)
    wq = (b_blocks / w_scale_4d).astype(jnp.float8_e4m3fn).reshape(N, K)
    w_scale = w_scale_4d.squeeze((1, 3)).astype(jnp.float32)

    # Shuffle weight for ASM kernel
    if with_shuffle:
        wq = wq.reshape(N // 16, 16, K // 32, 2, 16).transpose(0, 2, 3, 1, 4).reshape(N, K)

    return xq, wq, x_scale, w_scale, a_bf16, b_bf16


# ---------------------------------------------------------------------------
# Forward shape tests
# ---------------------------------------------------------------------------

FWD_CONFIGS = [
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
    xq, wq, x_scale, w_scale, a_bf16, b_bf16 = make_fp8_inputs(M, N, K)
    out = gemm_fp8_mi350(xq, wq, x_scale, w_scale, a_bf16, b_bf16)
    assert out.shape == (M, N), f"shape {out.shape} != ({M}, {N})"
    assert out.dtype == jnp.bfloat16
    assert jnp.all(jnp.isfinite(out)), "NaN/Inf in output"


@pytest.mark.parametrize("M,N,K", FWD_CONFIGS)
def test_fp8_gemm_fwd_accuracy(M, N, K):
    """Forward accuracy vs BF16 reference (with proper quantization + shuffle)."""
    xq, wq, x_scale, w_scale, a_bf16, b_bf16 = make_fp8_inputs(M, N, K, seed=42)
    out = gemm_fp8_mi350(xq, wq, x_scale, w_scale, a_bf16, b_bf16)
    ref = (a_bf16.astype(jnp.float32) @ b_bf16.astype(jnp.float32).T).astype(jnp.bfloat16)

    diff = float(jnp.max(jnp.abs(out.astype(jnp.float32) - ref.astype(jnp.float32))))
    ref_max = float(jnp.max(jnp.abs(ref.astype(jnp.float32))))
    if ref_max > 0.01:
        rel = diff / ref_max
        assert rel < 0.05, f"FP8 relative error {rel:.4f} > 5% (diff={diff:.4f}, ref_max={ref_max:.4f})"


# ---------------------------------------------------------------------------
# Backward tests (custom_vjp with lax.dot_general)
# ---------------------------------------------------------------------------

BWD_CONFIGS = [
    pytest.param(128, 256, 512, id="128x256x512"),
    pytest.param(256, 256, 512, id="256x256x512"),
    pytest.param(64, 256, 512, id="64x256x512"),
]


@pytest.mark.parametrize("M,N,K", BWD_CONFIGS)
def test_fp8_gemm_bwd_shape(M, N, K):
    """Backward gradient shapes and finiteness."""
    xq, wq, x_scale, w_scale, a_bf16, b_bf16 = make_fp8_inputs(M, N, K, seed=10)

    def loss(a_bf16, b_bf16):
        # Re-quantize inside loss (simulates training step)
        block_k = 128
        fp8_max = 448.0
        a_sg = jax.lax.stop_gradient(a_bf16)
        b_sg = jax.lax.stop_gradient(b_bf16)
        a_blocks = a_sg.reshape(M, K // block_k, block_k)
        x_amax = jnp.max(jnp.abs(a_blocks), axis=-1, keepdims=True)
        x_s = jnp.where(x_amax == 0, 1.0, x_amax / fp8_max)
        xq_ = (a_blocks / x_s).astype(jnp.float8_e4m3fn).reshape(M, K)
        xs_ = x_s.squeeze(-1).transpose(1, 0).astype(jnp.float32)
        b_blocks = b_sg.reshape(N // block_k, block_k, K // block_k, block_k)
        w_amax = jnp.max(jnp.abs(b_blocks), axis=(1, 3), keepdims=True)
        w_s = jnp.where(w_amax == 0, 1.0, w_amax / fp8_max)
        wq_ = (b_blocks / w_s).astype(jnp.float8_e4m3fn).reshape(N, K)
        ws_ = w_s.squeeze((1, 3)).astype(jnp.float32)
        wq_shuf = wq_.reshape(N//16, 16, K//32, 2, 16).transpose(0,2,3,1,4).reshape(N, K)
        return jnp.sum(gemm_fp8_mi350(xq_, wq_shuf, xs_, ws_, a_bf16, b_bf16))

    da, db = jax.grad(loss, argnums=(0, 1))(a_bf16, b_bf16)
    assert da.shape == a_bf16.shape, f"da {da.shape} != {a_bf16.shape}"
    assert db.shape == b_bf16.shape, f"db {db.shape} != {b_bf16.shape}"
    assert jnp.all(jnp.isfinite(da.astype(jnp.float32))), "da NaN/Inf"
    assert jnp.all(jnp.isfinite(db.astype(jnp.float32))), "db NaN/Inf"


# ---------------------------------------------------------------------------
# Llama shapes
# ---------------------------------------------------------------------------

LLAMA_FP8_CONFIGS = [
    pytest.param(4096, 4096, 4096, id="Llama7B_qkvo"),
    pytest.param(4096, 11008, 4096, id="Llama7B_gate_up"),
    pytest.param(4096, 4096, 11008, id="Llama7B_down"),
]


@pytest.mark.parametrize("M,N,K", LLAMA_FP8_CONFIGS)
def test_fp8_gemm_llama_shapes(M, N, K):
    """Llama2-7B shapes with proper quantization + shuffle."""
    xq, wq, x_scale, w_scale, a_bf16, b_bf16 = make_fp8_inputs(M, N, K, seed=7)
    out = gemm_fp8_mi350(xq, wq, x_scale, w_scale, a_bf16, b_bf16)
    assert out.shape == (M, N)
    assert out.dtype == jnp.bfloat16
    assert jnp.all(jnp.isfinite(out)), "NaN/Inf in Llama output"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_fp8_gemm_m16_minimum():
    """M=16 is the minimum supported."""
    xq, wq, x_scale, w_scale, a_bf16, b_bf16 = make_fp8_inputs(16, 256, 512)
    out = gemm_fp8_mi350(xq, wq, x_scale, w_scale, a_bf16, b_bf16)
    assert out.shape == (16, 256)


def test_fp8_gemm_m32_boundary():
    """M=32 boundary (switches from x32 to x128 kernel)."""
    xq, wq, x_scale, w_scale, a_bf16, b_bf16 = make_fp8_inputs(32, 256, 512)
    out = gemm_fp8_mi350(xq, wq, x_scale, w_scale, a_bf16, b_bf16)
    assert out.shape == (32, 256)


def test_fp8_gemm_m33_above_boundary():
    """M=33 uses x128 kernel."""
    xq, wq, x_scale, w_scale, a_bf16, b_bf16 = make_fp8_inputs(33, 256, 512)
    out = gemm_fp8_mi350(xq, wq, x_scale, w_scale, a_bf16, b_bf16)
    assert out.shape == (33, 256)


def test_fp8_gemm_consistency():
    """Same inputs produce close outputs across calls."""
    xq, wq, x_scale, w_scale, a_bf16, b_bf16 = make_fp8_inputs(128, 256, 512, seed=5)
    out1 = gemm_fp8_mi350(xq, wq, x_scale, w_scale, a_bf16, b_bf16)
    out2 = gemm_fp8_mi350(xq, wq, x_scale, w_scale, a_bf16, b_bf16)

    diff = float(jnp.max(jnp.abs(out1.astype(jnp.float32) - out2.astype(jnp.float32))))
    max_val = float(jnp.max(jnp.abs(out1.astype(jnp.float32))))
    if max_val > 0:
        assert diff / max_val < 0.01, f"Non-deterministic: rel_diff={diff/max_val}"


def test_fp8_gemm_zeros_input():
    """All-zeros FP8 input produces all-zeros output."""
    M, N, K = 64, 256, 512
    xq = jnp.zeros((M, K), dtype=jnp.float8_e4m3fn)
    wq = jnp.zeros((N, K), dtype=jnp.float8_e4m3fn)
    x_scale = jnp.ones((K // 128, M), dtype=jnp.float32)
    w_scale = jnp.ones((N // 128, K // 128), dtype=jnp.float32)
    a_bf16 = jnp.zeros((M, K), dtype=jnp.bfloat16)
    b_bf16 = jnp.zeros((N, K), dtype=jnp.bfloat16)

    out = gemm_fp8_mi350(xq, wq, x_scale, w_scale, a_bf16, b_bf16)
    assert jnp.all(out == 0), "Zeros input should give zeros output"
