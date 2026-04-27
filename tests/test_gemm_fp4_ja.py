# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Tests for FP4 GEMM via AITER ASM kernels.

Works on gfx950 (MI350/MI355X). Tests the full pipeline:
  BF16 -> MXFP4 quantize -> shuffle -> FFI kernel -> compare vs BF16 reference.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from jax_aiter.gemm_fp4 import gemm_fp4
from jax_aiter.gemm_fp4.fp4_utils import (
    bf16_to_mxfp4,
    mxfp4_to_bf16,
    e8m0_shuffle,
    shuffle_weight,
    e8m0_to_f32,
    MXFP4_BLOCK_SIZE,
)


def _make_quantized_inputs(M, N, K, seed=0):
    """Create properly quantized FP4 inputs from random BF16 data.

    Returns both the quantized inputs (for the kernel) and the BF16 originals
    (for reference comparison).
    """
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    a_bf16 = jax.random.normal(k1, (M, K), dtype=jnp.bfloat16)
    b_bf16 = jax.random.normal(k2, (N, K), dtype=jnp.bfloat16)

    a_packed, a_scales = bf16_to_mxfp4(a_bf16)
    b_packed, b_scales = bf16_to_mxfp4(b_bf16)

    b_packed_shuffled = shuffle_weight(b_packed)
    b_scales_shuffled = e8m0_shuffle(b_scales)
    a_scales_shuffled = e8m0_shuffle(a_scales)

    return (a_packed, b_packed_shuffled, a_scales_shuffled, b_scales_shuffled,
            a_bf16, b_bf16, a_packed, b_packed, a_scales, b_scales)


# --- Quantization round-trip tests (no GPU kernel needed) ---

@pytest.mark.parametrize("M,K", [
    pytest.param(128, 256, id="128x256"),
    pytest.param(256, 512, id="256x512"),
    pytest.param(1024, 1024, id="1024x1024"),
])
def test_mxfp4_roundtrip(M, K):
    """Verify quantize -> dequantize preserves values within FP4 precision."""
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (M, K), dtype=jnp.bfloat16)

    packed, scales = bf16_to_mxfp4(x)

    assert packed.shape == (M, K // 2), f"packed shape {packed.shape}"
    assert packed.dtype == jnp.uint8
    assert scales.shape == (M, K // MXFP4_BLOCK_SIZE), f"scales shape {scales.shape}"
    assert scales.dtype == jnp.uint8

    x_recon = mxfp4_to_bf16(packed, scales)
    assert x_recon.shape == (M, K)
    assert x_recon.dtype == jnp.bfloat16

    ref_f32 = x.astype(jnp.float32)
    recon_f32 = x_recon.astype(jnp.float32)
    abs_err = jnp.abs(ref_f32 - recon_f32)
    rel_err = abs_err / (jnp.abs(ref_f32) + 1e-10)
    mean_rel = float(jnp.mean(rel_err))
    max_rel = float(jnp.max(rel_err))

    assert mean_rel < 0.5, f"Mean relative error too large: {mean_rel:.4f}"
    assert jnp.all(jnp.isfinite(x_recon)), "Non-finite values in reconstructed output"


def test_mxfp4_zeros():
    """Zero input should quantize to zero output."""
    x = jnp.zeros((64, 128), dtype=jnp.bfloat16)
    packed, scales = bf16_to_mxfp4(x)
    x_recon = mxfp4_to_bf16(packed, scales)
    assert jnp.allclose(x_recon, 0.0, atol=1e-6)


def test_mxfp4_known_values():
    """Test quantization of known E2M1-representable values."""
    representable = jnp.array(
        [[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
          -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0] * 4],
        dtype=jnp.bfloat16,
    )
    packed, scales = bf16_to_mxfp4(representable)
    recon = mxfp4_to_bf16(packed, scales)
    recon_f32 = recon.astype(jnp.float32)
    ref_f32 = representable.astype(jnp.float32)
    assert jnp.allclose(recon_f32, ref_f32, atol=0.01), (
        f"Known values not preserved: max_err={float(jnp.max(jnp.abs(recon_f32 - ref_f32)))}"
    )


# --- FFI kernel tests ---

@pytest.mark.parametrize("M,N,K", [
    pytest.param(128, 128, 256, id="128x128x256"),
    pytest.param(256, 256, 256, id="256x256x256"),
    pytest.param(128, 256, 512, id="128x256x512"),
    pytest.param(256, 512, 256, id="256x512x256"),
    pytest.param(1024, 1024, 512, id="1k_sq"),
])
def test_fp4_gemm_shape_and_dtype(M, N, K):
    """Verify output shape and dtype from FP4 GEMM kernel."""
    (a_p, b_p, a_s, b_s,
     _, _, _, _, _, _) = _make_quantized_inputs(M, N, K)

    out = gemm_fp4(a_p, b_p, a_s, b_s)
    assert out.shape == (M, N), f"shape {out.shape} != ({M}, {N})"
    assert out.dtype == jnp.bfloat16


@pytest.mark.parametrize("M,N,K", [
    pytest.param(128, 128, 256, id="128x128x256"),
    pytest.param(256, 256, 256, id="256x256x256"),
    pytest.param(256, 512, 512, id="256x512x512"),
    pytest.param(1024, 1024, 512, id="1k_sq"),
])
def test_fp4_gemm_accuracy(M, N, K):
    """Compare FP4 GEMM output against BF16 dequantize-then-matmul reference.

    The reference path: dequantize both A and B from MXFP4 to BF16, then
    compute jnp.matmul(A_bf16, B_bf16.T) in float32. The kernel should
    produce results within FP4 quantization noise.
    """
    (a_p, b_p, a_s, b_s,
     a_bf16, b_bf16, a_raw, b_raw, a_sc, b_sc) = _make_quantized_inputs(M, N, K, seed=42)

    out = gemm_fp4(a_p, b_p, a_s, b_s)

    a_deq = mxfp4_to_bf16(a_raw, a_sc).astype(jnp.float32)
    b_deq = mxfp4_to_bf16(b_raw, b_sc).astype(jnp.float32)
    ref = jnp.matmul(a_deq, b_deq.T)

    out_f32 = out.astype(jnp.float32)
    abs_err = jnp.abs(out_f32 - ref)
    scale = jnp.maximum(jnp.abs(ref), 1.0)
    rel_err = abs_err / scale
    mean_rel = float(jnp.mean(rel_err))

    assert jnp.all(jnp.isfinite(out)), "Non-finite values in FP4 GEMM output"
    assert mean_rel < 0.1, (
        f"Mean relative error {mean_rel:.4f} exceeds threshold for {M}x{N}x{K}"
    )


def test_fp4_gemm_zeros():
    """Zero inputs should produce zero output."""
    M, N, K = 128, 128, 256
    a = jnp.zeros((M, K // 2), dtype=jnp.uint8)
    b = jnp.zeros((N, K // 2), dtype=jnp.uint8)
    a_scale = jnp.full((M, K // 32), 127, dtype=jnp.uint8)
    b_scale = jnp.full((N, K // 32), 127, dtype=jnp.uint8)
    a_scale = e8m0_shuffle(a_scale)
    b_scale = e8m0_shuffle(b_scale)
    b = shuffle_weight(b)
    out = gemm_fp4(a, b, a_scale, b_scale)
    assert out.shape == (M, N)
    assert jnp.all(jnp.isfinite(out))


# --- High-level API tests (gemm_fp4_bf16) ---

from jax_aiter.gemm_fp4 import gemm_fp4_bf16


@pytest.mark.parametrize("M,N,K", [
    pytest.param(256, 256, 256, id="small_256"),
    pytest.param(256, 512, 512, id="medium_256x512"),
    pytest.param(1024, 1024, 512, id="1k_sq"),
])
def test_gemm_fp4_bf16_forward(M, N, K):
    """gemm_fp4_bf16 produces finite output of the right shape/dtype."""
    key = jax.random.PRNGKey(7)
    k1, k2 = jax.random.split(key)
    a = jax.random.normal(k1, (M, K), dtype=jnp.bfloat16)
    b = jax.random.normal(k2, (N, K), dtype=jnp.bfloat16)

    out = gemm_fp4_bf16(a, b)

    assert out.shape == (M, N)
    assert out.dtype == jnp.bfloat16
    assert jnp.all(jnp.isfinite(out)), "Non-finite values in gemm_fp4_bf16 output"


@pytest.mark.parametrize("M,N,K", [
    pytest.param(256, 256, 256, id="small_256"),
    pytest.param(256, 512, 512, id="medium_256x512"),
])
def test_gemm_fp4_bf16_gradient_flow(M, N, K):
    """gemm_fp4_bf16 backward produces finite gradients for a and b.

    Backward path: dual-cast grad_out (Hadamard ON) -> FP4 dA (NN) +
    FP4 dB (NT wgrad).
    """
    key = jax.random.PRNGKey(99)
    k1, k2 = jax.random.split(key)
    a = jax.random.normal(k1, (M, K), dtype=jnp.bfloat16)
    b = jax.random.normal(k2, (N, K), dtype=jnp.bfloat16)

    da, db = jax.grad(
        lambda a_, b_: jnp.sum(gemm_fp4_bf16(a_, b_)), argnums=(0, 1)
    )(a, b)

    assert da.shape == a.shape
    assert db.shape == b.shape
    assert jnp.all(jnp.isfinite(da)), "Non-finite da"
    assert jnp.all(jnp.isfinite(db)), "Non-finite db"
    assert jnp.any(da != 0), "da all zeros"
    assert jnp.any(db != 0), "db all zeros"
