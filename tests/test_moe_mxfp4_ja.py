# SPDX-License-Identifier: MIT
"""Correctness coverage for collaborator-derived MXFP4 grouped operations."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("flydsl.compiler")

from jax_aiter.ops.moe_mxfp4 import (
    grouped_mxfp4_dw,
    grouped_mxfp4_fwd_da,
    quantize_mxfp4_dim0,
    quantize_mxfp4_dim1,
)

_FP4_VALUES = np.asarray(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32
)
_ONE_SIXTH = np.asarray(0x3E2AAAAB, dtype=np.uint32).view(np.float32)


def _reference_dim0(value):
    value = np.asarray(value, dtype=np.float32)
    rows, cols = value.shape
    blocks = value.reshape(rows, cols // 32, 32)
    maximum = np.max(np.abs(blocks), axis=-1).astype(np.float32)
    working = (maximum * _ONE_SIXTH).view(np.uint32)
    mantissa = working & np.uint32(0x7FFFFF)
    biased_exponent = (working >> np.uint32(23)) & np.uint32(0xFF)
    scales = np.minimum(
        biased_exponent + (mantissa != 0), np.uint32(254)
    ).astype(np.uint8)
    scale_values = (scales.astype(np.uint32) << np.uint32(23)).view(
        np.float32
    )

    normalized = np.divide(
        blocks,
        scale_values[..., None],
        out=np.zeros_like(blocks),
        where=scale_values[..., None] != 0,
    )
    signs = np.signbit(normalized).astype(np.uint8)
    magnitudes = np.abs(normalized)
    positive_codes = np.argmin(
        np.abs(magnitudes[..., None] - _FP4_VALUES), axis=-1
    ).astype(np.uint8)
    codes = positive_codes | (signs << np.uint8(3))
    flat_codes = codes.reshape(rows, cols)
    packed = flat_codes[:, 0::2] | (flat_codes[:, 1::2] << np.uint8(4))
    return packed, scales


def _unpack_fp4(packed):
    packed = np.asarray(packed, dtype=np.uint8)
    codes = np.empty((*packed.shape[:-1], packed.shape[-1] * 2), np.uint8)
    codes[..., 0::2] = packed & np.uint8(0xF)
    codes[..., 1::2] = packed >> np.uint8(4)
    signs = np.where(codes & np.uint8(0x8), -1.0, 1.0)
    return signs * _FP4_VALUES[codes & np.uint8(0x7)]


def _dequantize(packed, scales):
    values = _unpack_fp4(packed)
    scale_values = (
        np.asarray(scales, np.uint8).astype(np.uint32) << np.uint32(23)
    ).view(np.float32)
    return values * np.repeat(scale_values, 32, axis=-1)


def _relative_error(actual, expected):
    actual = np.asarray(actual, np.float32)
    expected = np.asarray(expected, np.float32)
    return np.linalg.norm(actual - expected) / np.linalg.norm(expected)


def test_dim0_matches_rceil_reference():
    positive = _FP4_VALUES
    block = np.concatenate((positive, -positive, positive, -positive))
    row = np.concatenate((block, block * np.float32(2.0)))
    value = np.tile(row, (64, 1)).astype(jnp.bfloat16)
    actual_packed, actual_scales = jax.jit(quantize_mxfp4_dim0)(
        jnp.asarray(value)
    )
    expected_packed, expected_scales = _reference_dim0(value)
    np.testing.assert_array_equal(np.asarray(actual_scales), expected_scales)
    np.testing.assert_array_equal(np.asarray(actual_packed), expected_packed)


def test_dim1_matches_dim0_of_transpose():
    rng = np.random.default_rng(9)
    value = jnp.asarray(
        rng.normal(size=(128, 64)).astype(np.float32), dtype=jnp.bfloat16
    )
    dim1_packed, dim1_scales = jax.jit(quantize_mxfp4_dim1)(value)
    dim0_packed, dim0_scales = jax.jit(quantize_mxfp4_dim0)(value.T)
    np.testing.assert_array_equal(np.asarray(dim1_packed), np.asarray(dim0_packed))
    np.testing.assert_array_equal(np.asarray(dim1_scales), np.asarray(dim0_scales))


def test_dim0_row_chunking_matches_single_launch_bitwise():
    value = jax.random.normal(
        jax.random.key(21), (384, 1024), dtype=jnp.float32
    ).astype(jnp.bfloat16)
    single_launch = jax.jit(quantize_mxfp4_dim0)
    chunked_launch = jax.jit(
        functools.partial(
            quantize_mxfp4_dim0, _max_index_elements=128 * 1024
        )
    )
    chunked_ir = str(chunked_launch.lower(value).compiler_ir()).lower()
    assert "stablehlo.reduce" in chunked_ir
    assert "stablehlo.case" in chunked_ir
    assert "host_callback" not in chunked_ir
    for candidate in (value, value.at[128:].set(0)):
        expected = single_launch(candidate)
        chunked = chunked_launch(candidate)
        for actual, reference in zip(chunked, expected, strict=True):
            np.testing.assert_array_equal(
                np.asarray(actual), np.asarray(reference)
            )


def test_dim1_row_chunking_matches_single_launch_bitwise():
    value = jax.random.normal(
        jax.random.key(23), (384, 1024), dtype=jnp.float32
    ).astype(jnp.bfloat16)
    single_launch = jax.jit(quantize_mxfp4_dim1)
    chunked_launch = jax.jit(
        functools.partial(
            quantize_mxfp4_dim1, _max_index_elements=128 * 1024
        )
    )
    chunked_ir = str(chunked_launch.lower(value).compiler_ir()).lower()
    assert "stablehlo.reduce" in chunked_ir
    assert "stablehlo.case" in chunked_ir
    assert "host_callback" not in chunked_ir
    for candidate in (value, value.at[128:].set(0)):
        expected = single_launch(candidate)
        chunked = chunked_launch(candidate)
        for actual, reference in zip(chunked, expected, strict=True):
            np.testing.assert_array_equal(
                np.asarray(actual), np.asarray(reference)
            )


def test_dim0_dispatches_on_multiple_devices_and_lowers_to_typed_ffi():
    devices = jax.local_devices(backend="gpu")
    if len(devices) < 2:
        pytest.skip("requires at least two local ROCm devices")
    value = jnp.broadcast_to(
        jnp.arange(64 * 64, dtype=jnp.float32).reshape(64, 64),
        (len(devices), 64, 64),
    ).astype(jnp.bfloat16)
    packed, scales = jax.pmap(quantize_mxfp4_dim0, devices=devices)(value)
    np.testing.assert_array_equal(
        np.asarray(packed[1:]),
        np.broadcast_to(np.asarray(packed[0]), np.asarray(packed[1:]).shape),
    )
    np.testing.assert_array_equal(
        np.asarray(scales[1:]),
        np.broadcast_to(np.asarray(scales[0]), np.asarray(scales[1:]).shape),
    )

    lowered = jax.jit(quantize_mxfp4_dim0).lower(value[0])
    assert "FlydslDispatchJA" in str(lowered.compiler_ir())


def test_grouped_forward_matches_dequantized_reference():
    experts, rows, contraction, output = 2, 384, 1024, 256
    key_lhs, key_rhs = jax.random.split(jax.random.key(11))
    lhs = jax.random.normal(
        key_lhs, (rows, contraction), dtype=jnp.float32
    ).astype(jnp.bfloat16)
    rhs = jax.random.normal(
        key_rhs, (experts, output, contraction), dtype=jnp.float32
    ).astype(jnp.bfloat16)
    lhs_packed, lhs_scales = quantize_mxfp4_dim0(lhs)
    rhs_packed, rhs_scales = quantize_mxfp4_dim0(
        rhs.reshape(experts * output, contraction)
    )
    rhs_packed = rhs_packed.reshape(experts, output, contraction // 2)
    rhs_scales = rhs_scales.reshape(experts, output, contraction // 32)
    offsets = jnp.asarray([128, 256], dtype=jnp.int32)
    compiled = jax.jit(grouped_mxfp4_fwd_da).lower(
        lhs_packed, rhs_packed, lhs_scales, rhs_scales, offsets
    ).compile()
    actual = compiled(
        lhs_packed, rhs_packed, lhs_scales, rhs_scales, offsets
    )
    lhs_dequantized = _dequantize(lhs_packed, lhs_scales)
    rhs_dequantized = _dequantize(rhs_packed, rhs_scales)
    expected = np.concatenate(
        (
            lhs_dequantized[:128] @ rhs_dequantized[0].T,
            lhs_dequantized[128:256] @ rhs_dequantized[1].T,
            np.zeros((128, output), dtype=np.float32),
        )
    )
    assert _relative_error(actual, expected) < 0.01

    # Route values change while shapes stay fixed: the same executable must
    # accept an empty first expert and move all valid rows to expert 1.
    changed_offsets = jnp.asarray([0, 256], dtype=jnp.int32)
    changed = compiled(
        lhs_packed,
        rhs_packed,
        lhs_scales,
        rhs_scales,
        changed_offsets,
    )
    changed_expected = np.concatenate(
        (
            lhs_dequantized[:256] @ rhs_dequantized[1].T,
            np.zeros((128, output), dtype=np.float32),
        )
    )
    assert _relative_error(changed, changed_expected) < 0.01


def test_grouped_forward_defines_slack_rows_without_zero_seed():
    experts, rows, contraction, output = 2, 1024, 1024, 256
    key_lhs, key_rhs = jax.random.split(jax.random.key(17))
    lhs = jax.random.normal(
        key_lhs, (rows, contraction), dtype=jnp.float32
    ).astype(jnp.bfloat16)
    # Rows past the last group hold large non-zero values: exact-zero output
    # there must come from the kernel, not from zero inputs.
    lhs = lhs.at[256:].multiply(1000.0)
    rhs = jax.random.normal(
        key_rhs, (experts, output, contraction), dtype=jnp.float32
    ).astype(jnp.bfloat16)
    lhs_packed, lhs_scales = quantize_mxfp4_dim0(lhs)
    rhs_packed, rhs_scales = quantize_mxfp4_dim0(
        rhs.reshape(experts * output, contraction)
    )
    rhs_packed = rhs_packed.reshape(experts, output, contraction // 2)
    rhs_scales = rhs_scales.reshape(experts, output, contraction // 32)
    offsets = jnp.asarray([128, 256], dtype=jnp.int32)

    lowered = jax.jit(grouped_mxfp4_fwd_da).lower(
        lhs_packed, rhs_packed, lhs_scales, rhs_scales, offsets
    )
    assert "output_operand_aliases" not in str(lowered.compiler_ir())
    compiled = lowered.compile()
    chunked = jax.jit(
        functools.partial(
            grouped_mxfp4_fwd_da,
            _max_index_elements=256 * (contraction // 2),
        )
    ).lower(
        lhs_packed, rhs_packed, lhs_scales, rhs_scales, offsets
    )
    chunked_ir = str(chunked.compiler_ir()).lower()
    assert "stablehlo.case" in chunked_ir
    assert "host_callback" not in chunked_ir
    chunked_compiled = chunked.compile()
    actual = np.asarray(
        compiled(lhs_packed, rhs_packed, lhs_scales, rhs_scales, offsets)
    )
    chunked_actual = np.asarray(
        chunked_compiled(
            lhs_packed, rhs_packed, lhs_scales, rhs_scales, offsets
        )
    )

    np.testing.assert_array_equal(actual[256:], 0)
    np.testing.assert_array_equal(chunked_actual, actual)
    lhs_dequantized = _dequantize(lhs_packed, lhs_scales)
    rhs_dequantized = _dequantize(rhs_packed, rhs_scales)
    expected = np.concatenate(
        (
            lhs_dequantized[:128] @ rhs_dequantized[0].T,
            lhs_dequantized[128:256] @ rhs_dequantized[1].T,
        )
    )
    assert _relative_error(actual[:256], expected) < 0.01

    all_empty = np.asarray(
        compiled(
            lhs_packed,
            rhs_packed,
            lhs_scales,
            rhs_scales,
            jnp.asarray([0, 0], dtype=jnp.int32),
        )
    )
    np.testing.assert_array_equal(all_empty, 0)
    chunked_all_empty = np.asarray(
        chunked_compiled(
            lhs_packed,
            rhs_packed,
            lhs_scales,
            rhs_scales,
            jnp.asarray([0, 0], dtype=jnp.int32),
        )
    )
    np.testing.assert_array_equal(chunked_all_empty, all_empty)

    no_slack = np.asarray(
        compiled(
            lhs_packed,
            rhs_packed,
            lhs_scales,
            rhs_scales,
            jnp.asarray([0, rows], dtype=jnp.int32),
        )
    )
    expected_no_slack = lhs_dequantized @ rhs_dequantized[1].T
    assert _relative_error(no_slack, expected_no_slack) < 0.01
    chunked_no_slack = np.asarray(
        chunked_compiled(
            lhs_packed,
            rhs_packed,
            lhs_scales,
            rhs_scales,
            jnp.asarray([0, rows], dtype=jnp.int32),
        )
    )
    np.testing.assert_array_equal(chunked_no_slack, no_slack)


def test_grouped_dw_matches_dequantized_reference_and_zeros_empty_expert():
    experts, rows, contraction, output = 2, 256, 1024, 256
    key_lhs, key_grad = jax.random.split(jax.random.key(13))
    lhs = jax.random.normal(
        key_lhs, (rows, contraction), dtype=jnp.float32
    ).astype(jnp.bfloat16)
    grad = jax.random.normal(
        key_grad, (rows, output), dtype=jnp.float32
    ).astype(jnp.bfloat16)
    grad_packed, grad_scales = quantize_mxfp4_dim1(grad)
    lhs_packed, lhs_scales = quantize_mxfp4_dim1(lhs)
    offsets = jnp.asarray([0, rows], dtype=jnp.int32)

    actual = jax.jit(grouped_mxfp4_dw)(
        grad_packed, grad_scales, lhs_packed, lhs_scales, offsets
    )
    grad_dequantized = _dequantize(grad_packed, grad_scales)
    lhs_dequantized = _dequantize(lhs_packed, lhs_scales)
    expected_nonempty = grad_dequantized @ lhs_dequantized.T

    np.testing.assert_array_equal(np.asarray(actual[0]), 0)
    assert _relative_error(actual[1], expected_nonempty) < 0.01
