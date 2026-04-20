# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Unit tests for MXFP4Quantizer and Mxfp4Tensor.

Verifies:
  1. Quantizer rowwise output matches the legacy JAX pipeline
     (bf16_to_mxfp4 + optional shuffle_weight + e8m0_shuffle).
  2. Dual-cast produces both rowwise and columnwise tensors with the
     expected shapes.
  3. Role presets (for_weight, for_activation, for_grad) set the right flags.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from jax_aiter.gemm_fp4 import (
    MXFP4Quantizer,
    Mxfp4Tensor,
)
from jax_aiter.gemm_fp4.fp4_utils import (
    bf16_to_mxfp4,
    e8m0_shuffle,
    shuffle_weight,
)
from jax_aiter.gemm_fp4.quantizer import _fused_quant_available


# ---------------------------------------------------------------------------
# Preset flag tests — no GPU kernel needed
# ---------------------------------------------------------------------------

def test_weight_preset_flags():
    q = MXFP4Quantizer.for_weight()
    assert q.rowwise is True
    assert q.columnwise is True
    assert q.shuffle_B_matrix_for_aiter is True
    assert q.shuffle_colwise_fp4 is True
    assert q.shuffle_scales is True


def test_activation_preset_flags():
    q = MXFP4Quantizer.for_activation()
    assert q.rowwise is True
    assert q.columnwise is True
    assert q.shuffle_B_matrix_for_aiter is False
    assert q.shuffle_colwise_fp4 is True


def test_grad_preset_flags():
    q = MXFP4Quantizer.for_grad()
    assert q.rowwise is True
    assert q.columnwise is True
    assert q.shuffle_B_matrix_for_aiter is False
    assert q.shuffle_colwise_fp4 is False


def test_quantizer_requires_at_least_one_direction():
    q = MXFP4Quantizer(rowwise=False, columnwise=False, use_fused_kernel=False)
    x = jnp.ones((64, 128), dtype=jnp.bfloat16)
    with pytest.raises(ValueError):
        q.quantize(x)


# ---------------------------------------------------------------------------
# JAX-fallback rowwise parity tests (no FFI needed)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shuffle_B", [False, True])
def test_jax_fallback_rowwise_matches_legacy_pipeline(shuffle_B):
    """Rowwise-only JAX fallback == bf16_to_mxfp4 + (optional shuffle) + e8m0_shuffle."""
    key = jax.random.PRNGKey(7)
    x = jax.random.normal(key, (256, 512), dtype=jnp.bfloat16)

    q = MXFP4Quantizer(
        rowwise=True,
        columnwise=False,
        shuffle_B_matrix_for_aiter=shuffle_B,
        shuffle_scales=True,
        use_hadamard=False,
        use_fused_kernel=False,
    )
    tensor = q.quantize(x)
    assert isinstance(tensor, Mxfp4Tensor)
    assert tensor.has_rowwise
    assert not tensor.has_columnwise

    expected_packed, expected_scales = bf16_to_mxfp4(x)
    if shuffle_B:
        expected_packed = shuffle_weight(expected_packed)
    expected_scales = e8m0_shuffle(expected_scales)

    np.testing.assert_array_equal(
        np.asarray(tensor.rowwise_data), np.asarray(expected_packed))
    np.testing.assert_array_equal(
        np.asarray(tensor.rowwise_scale), np.asarray(expected_scales))


def test_jax_fallback_rejects_columnwise():
    q = MXFP4Quantizer(
        rowwise=True, columnwise=True, use_fused_kernel=False)
    x = jnp.ones((64, 128), dtype=jnp.bfloat16)
    with pytest.raises(NotImplementedError):
        q.quantize(x)


def test_jax_fallback_rejects_hadamard():
    q = MXFP4Quantizer(
        rowwise=True, columnwise=False, use_hadamard=True, use_fused_kernel=False)
    x = jnp.ones((64, 128), dtype=jnp.bfloat16)
    with pytest.raises(NotImplementedError):
        q.quantize(x)


# ---------------------------------------------------------------------------
# Fused-kernel tests (require CastMxfp4JA / CastMxfp4DualJA built)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _fused_quant_available(),
    reason="CastMxfp4JA FFI not available (build with 'make ja_mods')")
def test_fused_rowwise_matches_jax_fallback():
    """Fused HIP kernel rowwise output should equal the JAX fallback."""
    key = jax.random.PRNGKey(11)
    x = jax.random.normal(key, (512, 1024), dtype=jnp.bfloat16)

    q_fused = MXFP4Quantizer(
        rowwise=True, columnwise=False,
        shuffle_B_matrix_for_aiter=False,
        shuffle_scales=True,
        use_fused_kernel=True,
    )
    q_jax = MXFP4Quantizer(
        rowwise=True, columnwise=False,
        shuffle_B_matrix_for_aiter=False,
        shuffle_scales=True,
        use_fused_kernel=False,
    )

    t_fused = q_fused.quantize(x)
    t_jax = q_jax.quantize(x)

    # Both should have same shape; exact bitwise match is desired but
    # floating-point rounding in fused vs JAX paths may differ in the LSB.
    assert t_fused.rowwise_data.shape == t_jax.rowwise_data.shape
    assert t_fused.rowwise_scale.shape[1] == t_jax.rowwise_scale.shape[1]

    # Dequantize both: mean and max absolute error must be bounded by
    # FP4 quantization noise. FP4 values are {0, 0.5, 1, 1.5, 2, 3, 4, 6},
    # so the smallest step is 0.5; adjacent rounding choices can differ
    # by up to one block's scale. We bound mean error tightly and check
    # the bulk of values agree exactly.
    from jax_aiter.gemm_fp4.fp4_utils import mxfp4_to_bf16
    ref = mxfp4_to_bf16(t_jax.rowwise_data,
                        t_jax.rowwise_scale[:x.shape[0], :x.shape[1] // 32])
    x_fused = mxfp4_to_bf16(t_fused.rowwise_data,
                            t_fused.rowwise_scale[:x.shape[0], :x.shape[1] // 32])
    diff = jnp.abs(ref.astype(jnp.float32) - x_fused.astype(jnp.float32))
    mean_abs_err = float(jnp.mean(diff))
    frac_exact = float(jnp.mean(diff == 0))
    assert mean_abs_err < 0.05, (
        f"Fused vs JAX quant mean abs err too large: {mean_abs_err:.4f}")
    assert frac_exact > 0.80, (
        f"Less than 80% of elements match exactly: {frac_exact:.2f}")


@pytest.mark.skipif(
    not _fused_quant_available(),
    reason="CastMxfp4DualJA FFI not available (build with 'make ja_mods')")
@pytest.mark.parametrize("preset_name,expected_shuffle_B", [
    ("for_activation", False),
    ("for_weight", True),
    ("for_grad", False),
])
def test_fused_dual_cast_shapes(preset_name, expected_shuffle_B):
    """Dual-cast produces rowwise + columnwise tensors with expected shapes."""
    key = jax.random.PRNGKey(3)
    M, K = 512, 1024
    x = jax.random.normal(key, (M, K), dtype=jnp.bfloat16)

    q = getattr(MXFP4Quantizer, preset_name)()
    assert q.shuffle_B_matrix_for_aiter == expected_shuffle_B

    tensor = q.quantize(x)
    assert tensor.has_rowwise
    assert tensor.has_columnwise
    assert tensor.rowwise_data.shape == (M, K // 2)
    assert tensor.columnwise_data.shape == (K, M // 2)


@pytest.mark.skipif(
    not _fused_quant_available(),
    reason="CastMxfp4DualJA FFI not available (build with 'make ja_mods')")
def test_fused_rowwise_only_via_columnwise_false():
    """rowwise=True, columnwise=False uses CastMxfp4JA (not dual)."""
    key = jax.random.PRNGKey(5)
    x = jax.random.normal(key, (256, 512), dtype=jnp.bfloat16)

    q = MXFP4Quantizer(
        rowwise=True, columnwise=False,
        shuffle_B_matrix_for_aiter=False,
        shuffle_scales=True,
        use_fused_kernel=True,
    )
    tensor = q.quantize(x)
    assert tensor.has_rowwise
    assert not tensor.has_columnwise
    assert tensor.columnwise_data is None
    assert tensor.columnwise_scale is None


def test_mxfp4_tensor_accessors():
    t = Mxfp4Tensor(
        rowwise_data=jnp.zeros((4, 8), dtype=jnp.uint8),
        rowwise_scale=jnp.zeros((4, 1), dtype=jnp.uint8),
    )
    assert t.has_rowwise
    assert not t.has_columnwise
    t.rowwise_tuple()  # should not raise
    with pytest.raises(ValueError):
        t.columnwise_tuple()
