# SPDX-License-Identifier: MIT
"""Correctness coverage for the MXFP4 MoE casts."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("flydsl.compiler")

from jax_aiter.ops.moe_mxfp4 import (
    quantize_mxfp4_dim0,
    quantize_mxfp4_dim0_gather,
    quantize_mxfp4_dim1,
    quantize_mxfp4_dim1_gather,
    requantize_mxfp4_dim0_transposed,
)
from jax_aiter.ops.moe_mxfp4_ptquant import quantize_mxfp4_dual

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


@pytest.mark.parametrize("shape", [(128, 64), (160, 96), (96, 384), (384, 1024)])
def test_dim1_matches_dim0_of_transpose(shape):
    rng = np.random.default_rng(9)
    value = jnp.asarray(
        rng.normal(size=shape).astype(np.float32), dtype=jnp.bfloat16
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


@pytest.mark.parametrize("max_index_elements", [None, 128 * 1024])
@pytest.mark.parametrize("limit", [0, 96, 256, 384])
def test_row_limit_matches_unlimited_rows_below_limit(limit, max_index_elements):
    value = jax.random.normal(
        jax.random.key(29), (384, 1024), dtype=jnp.float32
    ).astype(jnp.bfloat16)
    # Tail rows must not reach any output row below the limit.
    poisoned = value.at[limit:].set(jnp.nan)
    extra = {} if max_index_elements is None else {"_max_index_elements": max_index_elements}
    for quantize, axis in ((quantize_mxfp4_dim0, 0), (quantize_mxfp4_dim1, 1)):
        expected = jax.jit(quantize)(value)
        limited = jax.jit(functools.partial(quantize, **extra))(
            poisoned, row_limit=jnp.int32(limit)
        )
        for actual, reference, per_row in zip(limited, expected, (2, 32), strict=True):
            valid = limit // per_row if axis else limit
            index = (slice(None), slice(0, valid)) if axis else slice(0, valid)
            np.testing.assert_array_equal(
                np.asarray(actual)[index], np.asarray(reference)[index]
            )


@pytest.mark.parametrize("cols", [1024, 7168])
@pytest.mark.parametrize("limit", [None, 0, 77, 300])
def test_dim0_gather_matches_materialized_gather(cols, limit):
    in_rows, out_rows = 200, 300
    value = jax.random.normal(
        jax.random.key(31), (in_rows, cols), dtype=jnp.float32
    ).astype(jnp.bfloat16)
    index = jax.random.randint(jax.random.key(37), (out_rows,), -1, in_rows)
    index = index.at[:8].set(jnp.int32(-1)).at[8:16].set(jnp.int32(in_rows - 1))
    index = index.astype(jnp.int32)
    gathered = jnp.where(
        (index >= 0)[:, None], value[jnp.maximum(index, 0)], jnp.zeros((), value.dtype)
    )
    expected = jax.jit(quantize_mxfp4_dim0)(gathered)
    extra = {} if limit is None else {"row_limit": jnp.int32(limit)}
    actual = jax.jit(functools.partial(quantize_mxfp4_dim0_gather, **extra))(
        value, index
    )
    valid = out_rows if limit is None else limit
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(
            np.asarray(result)[:valid], np.asarray(reference)[:valid]
        )


@pytest.mark.parametrize("cols", [1024, 7168, 2080])
@pytest.mark.parametrize("limit", [None, 0, 77, 320])
@pytest.mark.parametrize("out_rows", [384, 288])
def test_dim1_gather_matches_materialized_gather(cols, limit, out_rows):
    in_rows = 200
    value = jax.random.normal(
        jax.random.key(41), (in_rows, cols), dtype=jnp.float32
    ).astype(jnp.bfloat16)
    # Rows no index selects hold NaN: a masked row must not read them.
    value = value.at[in_rows - 8 :].set(jnp.nan)
    index = jax.random.randint(jax.random.key(43), (out_rows,), -1, in_rows - 8)
    index = index.at[:40].set(jnp.int32(-1)).astype(jnp.int32)
    gathered = jnp.where(
        (index >= 0)[:, None], value[jnp.maximum(index, 0)], jnp.zeros((), value.dtype)
    )
    expected = jax.jit(quantize_mxfp4_dim1)(gathered)
    extra = {} if limit is None else {"row_limit": jnp.int32(limit)}
    actual = jax.jit(functools.partial(quantize_mxfp4_dim1_gather, **extra))(
        value, index
    )
    valid = out_rows if limit is None else limit
    for result, reference, per_col in zip(actual, expected, (2, 32), strict=True):
        np.testing.assert_array_equal(
            np.asarray(result)[:, : valid // per_col],
            np.asarray(reference)[:, : valid // per_col],
        )


@pytest.mark.parametrize(
    "quantize",
    [
        quantize_mxfp4_dim0,
        quantize_mxfp4_dim1,
        lambda value: quantize_mxfp4_dim0_gather(value, jnp.arange(256, dtype=jnp.int32)),
        lambda value: quantize_mxfp4_dim1_gather(value, jnp.arange(256, dtype=jnp.int32)),
        quantize_mxfp4_dual,
    ],
    ids=["dim0", "dim1", "dim0_gather", "dim1_gather", "dual"],
)
def test_casts_reject_float32(quantize):
    with pytest.raises(TypeError, match="take bfloat16 input, got float32"):
        quantize(jnp.zeros((256, 256), jnp.float32))


@pytest.mark.parametrize("limit", [None, 0, 77, 200, 320])
@pytest.mark.parametrize(
    "quantize, out_rows, axis",
    [(quantize_mxfp4_dim0_gather, 300, 0), (quantize_mxfp4_dim1_gather, 384, 1)],
)
def test_gather_row_chunking_matches_single_launch_bitwise(quantize, out_rows, axis, limit):
    in_rows, cols = 200, 1024
    value = jax.random.normal(
        jax.random.key(53), (in_rows, cols), dtype=jnp.float32
    ).astype(jnp.bfloat16)
    index = jax.random.randint(jax.random.key(59), (out_rows,), -1, in_rows)
    index = index.at[:40].set(jnp.int32(-1)).astype(jnp.int32)
    extra = {} if limit is None else {"row_limit": jnp.int32(limit)}
    single_launch = jax.jit(functools.partial(quantize, **extra))
    chunked_launch = jax.jit(
        functools.partial(quantize, _max_index_elements=128 * (cols // 2), **extra)
    )
    chunked_ir = str(chunked_launch.lower(value, index).compiler_ir()).lower()
    assert chunked_ir.count("stablehlo.custom_call") == 3
    valid = out_rows if limit is None else limit
    for actual, reference, per_row in zip(
        chunked_launch(value, index), single_launch(value, index), (2, 32), strict=True
    ):
        index_valid = (slice(None), slice(0, valid // per_row)) if axis else slice(0, valid)
        np.testing.assert_array_equal(
            np.asarray(actual)[index_valid], np.asarray(reference)[index_valid]
        )


@pytest.mark.parametrize("shape", [(64, 96), (3, 2048, 224)])
def test_requantize_transposed_matches_dequantized_reference(shape):
    value = jax.random.normal(jax.random.key(47), shape, dtype=jnp.float32)
    value = value.at[..., :32, :32].set(0).astype(jnp.bfloat16)
    *lead, rows, cols = shape
    packed, scales = quantize_mxfp4_dim0(value.reshape(-1, cols))
    packed = packed.reshape(*lead, rows, cols // 2)
    scales = scales.reshape(*lead, rows, cols // 32)
    actual = jax.jit(requantize_mxfp4_dim0_transposed)(packed, scales)
    transposed = np.swapaxes(_dequantize(packed, scales), -1, -2)
    # Requantized values often sit on rounding ties, which the numpy reference
    # breaks differently from the kernel, so the kernel quantizes the exact
    # numpy dequantization.
    exact = jnp.asarray(transposed.reshape(-1, rows), dtype=jnp.bfloat16)
    assert np.array_equal(np.asarray(exact, np.float32), transposed.reshape(-1, rows))
    expected = jax.jit(quantize_mxfp4_dim0)(exact)
    for result, reference, per_row in zip(actual, expected, (2, 32), strict=True):
        np.testing.assert_array_equal(
            np.asarray(result), np.asarray(reference).reshape(*lead, cols, rows // per_row)
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
