# SPDX-License-Identifier: MIT
"""Whole-loop MXFP4 grouped kernels against dequantized references."""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("flydsl.compiler")

from jax_aiter.ops.moe_mxfp4 import (
    grouped_mxfp4_dw_wholeloop,
    grouped_mxfp4_fwd_da_wholeloop,
    quantize_mxfp4_dim0,
    quantize_mxfp4_dim1,
)

from test_moe_mxfp4_ja import _FP4_VALUES, _dequantize, _relative_error

_MOE_EXPERTS = 32
_MOE_ROWS = 16384


def _moe_group_sizes(seed):
    """Skewed 256-aligned sizes with empty experts and a slack tail."""
    return np.random.default_rng(seed).choice([0, 1, 2, 3], _MOE_EXPERTS) * 256


def _bounds(sizes):
    return jnp.asarray(np.concatenate(([0], np.cumsum(sizes))), jnp.int32)


def _bits(value):
    return np.asarray(value).view(np.uint16)


def _dequantize_on_device(packed, scales):
    """:func:`_dequantize` on the GPU, for operands too large for numpy."""
    table = jnp.asarray(np.concatenate((_FP4_VALUES, -_FP4_VALUES)))
    codes = jnp.stack((packed & 0xF, packed >> 4), axis=-1)
    scale = jax.lax.bitcast_convert_type(
        scales.astype(jnp.uint32) << 23, jnp.float32
    )
    values = table[codes.reshape(*packed.shape[:-1], -1)]
    return values * jnp.repeat(scale, 32, axis=-1)


def _group_ranges(sizes):
    ends = np.cumsum(sizes)
    return [(int(end - size), int(end)) for size, end in zip(sizes, ends)]


def _matmul_nt(lhs, rhs):
    return np.asarray(
        jnp.matmul(lhs, rhs.T, precision=jax.lax.Precision.HIGHEST)
    )


def _assert_forward_matches_reference(actual, operands, sizes):
    lhs = _dequantize_on_device(operands[0], operands[2])
    rhs = _dequantize_on_device(operands[1], operands[3])
    for expert, (start, end) in enumerate(_group_ranges(sizes)):
        if end > start:
            expected = _matmul_nt(lhs[start:end], rhs[expert])
            assert _relative_error(actual[start:end], expected) < 0.01, expert


def _assert_dw_matches_reference(actual, operands, sizes):
    grad = _dequantize_on_device(operands[0], operands[1])
    lhs = _dequantize_on_device(operands[2], operands[3])
    for expert, (start, end) in enumerate(_group_ranges(sizes)):
        if end > start:
            expected = _matmul_nt(grad[:, start:end], lhs[:, start:end])
            assert _relative_error(actual[expert], expected) < 0.01, expert
        else:
            np.testing.assert_array_equal(np.asarray(actual[expert]), 0)


def _forward_operands(seed, rows, contraction, output, experts, live_rows):
    key_lhs, key_rhs = jax.random.split(jax.random.key(seed))
    lhs = jax.random.normal(key_lhs, (rows, contraction), jnp.float32)
    # Large slack rows would show up in any live output that read them.
    lhs = lhs.at[live_rows:].multiply(1000.0).astype(jnp.bfloat16)
    rhs = jax.random.normal(
        key_rhs, (experts, output, contraction), jnp.float32
    ).astype(jnp.bfloat16)
    lhs_packed, lhs_scales = quantize_mxfp4_dim0(lhs)
    rhs_packed, rhs_scales = quantize_mxfp4_dim0(
        rhs.reshape(experts * output, contraction)
    )
    return (
        lhs_packed,
        rhs_packed.reshape(experts, output, contraction // 2),
        lhs_scales,
        rhs_scales.reshape(experts, output, contraction // 32),
    )


def _dw_operands(seed, rows, out_rows, out_cols, live_rows):
    key_grad, key_lhs = jax.random.split(jax.random.key(seed))
    grad = jax.random.normal(key_grad, (rows, out_rows), jnp.float32)
    lhs = jax.random.normal(key_lhs, (rows, out_cols), jnp.float32)
    grad = grad.at[live_rows:].multiply(1e4).astype(jnp.bfloat16)
    lhs = lhs.at[live_rows:].multiply(1e4).astype(jnp.bfloat16)
    grad_packed, grad_scales = quantize_mxfp4_dim1(grad)
    lhs_packed, lhs_scales = quantize_mxfp4_dim1(lhs)
    return grad_packed, grad_scales, lhs_packed, lhs_scales


@pytest.mark.parametrize(
    "rows,contraction,output,sizes",
    [
        (4096, 1024, 512, np.asarray([256, 0, 768, 512])),
        (_MOE_ROWS, 7168, 2048, _moe_group_sizes(1)),
        (_MOE_ROWS, 2048, 7168, _moe_group_sizes(2)),
    ],
    ids=["small", "moe_up", "moe_down"],
)
def test_wholeloop_forward_matches_dequantized_reference(
    rows, contraction, output, sizes
):
    bounds = _bounds(sizes)
    live_rows = int(bounds[-1])
    operands = _forward_operands(
        7, rows, contraction, output, len(sizes), live_rows
    )

    actual = jax.jit(grouped_mxfp4_fwd_da_wholeloop)(*operands, bounds)
    unmasked = jax.jit(
        functools.partial(grouped_mxfp4_fwd_da_wholeloop, zero_tail=False)
    )(*operands, bounds)

    _assert_forward_matches_reference(actual, operands, sizes)
    np.testing.assert_array_equal(np.asarray(actual[live_rows:]), 0)
    np.testing.assert_array_equal(
        _bits(unmasked[:live_rows]), _bits(actual[:live_rows])
    )


def test_wholeloop_forward_accepts_unaligned_groups_and_route_changes():
    rows, contraction, output = 1024, 1024, 256
    sizes = np.asarray([100, 0, 413])
    bounds = _bounds(sizes)
    operands = _forward_operands(11, rows, contraction, output, 3, 513)
    compiled = jax.jit(grouped_mxfp4_fwd_da_wholeloop).lower(
        *operands, bounds
    ).compile()

    lhs = _dequantize(operands[0], operands[2])
    rhs = _dequantize(operands[1], operands[3])
    actual = np.asarray(compiled(*operands, bounds), np.float32)
    expected = np.concatenate(
        (lhs[:100] @ rhs[0].T, lhs[100:513] @ rhs[2].T)
    )
    assert _relative_error(actual[:513], expected) < 0.01
    np.testing.assert_array_equal(actual[513:], 0)

    all_first = np.asarray(
        compiled(*operands, _bounds([rows, 0, 0])), np.float32
    )
    assert _relative_error(all_first, lhs @ rhs[0].T) < 0.01
    np.testing.assert_array_equal(
        np.asarray(compiled(*operands, _bounds([0, 0, 0]))), 0
    )


def test_wholeloop_forward_row_chunking_matches_single_launch_bitwise():
    rows, contraction, output = 1024, 1024, 256
    bounds = _bounds([100, 413, 0])
    operands = _forward_operands(19, rows, contraction, output, 3, 513)

    single_launch = jax.jit(grouped_mxfp4_fwd_da_wholeloop).lower(
        *operands, bounds
    ).compile()
    chunked_lowered = jax.jit(
        functools.partial(
            grouped_mxfp4_fwd_da_wholeloop,
            _max_index_elements=256 * (contraction // 2),
        )
    ).lower(*operands, bounds)
    chunked_ir = str(chunked_lowered.compiler_ir()).lower()
    assert "stablehlo.case" in chunked_ir
    assert "host_callback" not in chunked_ir
    chunked_launch = chunked_lowered.compile()
    for candidate_bounds in (bounds, _bounds([0, 0, 0])):
        expected = single_launch(*operands, candidate_bounds)
        actual = chunked_launch(*operands, candidate_bounds)
        np.testing.assert_array_equal(_bits(actual), _bits(expected))
        np.testing.assert_array_equal(np.asarray(actual[513:]), 0)


@pytest.mark.parametrize(
    "rows,out_rows,out_cols,sizes",
    [
        (4096, 512, 1024, np.asarray([256, 0, 768, 512])),
        (_MOE_ROWS, 2048, 7168, _moe_group_sizes(3)),
        (_MOE_ROWS, 7168, 2048, _moe_group_sizes(4)),
    ],
    ids=["small", "moe_up", "moe_down"],
)
def test_wholeloop_dw_matches_dequantized_reference(
    rows, out_rows, out_cols, sizes
):
    bounds = _bounds(sizes)
    operands = _dw_operands(5, rows, out_rows, out_cols, int(bounds[-1]))

    actual = jax.jit(grouped_mxfp4_dw_wholeloop)(*operands, bounds)

    assert actual.shape == (len(sizes), out_rows, out_cols)
    assert actual.dtype == jnp.bfloat16
    _assert_dw_matches_reference(actual, operands, sizes)


def test_wholeloop_dw_schedules_agree_bitwise():
    rows, out_rows, out_cols = 16640, 512, 1024
    sizes = np.asarray([8192, 4096])
    bounds = _bounds(sizes)
    operands = _dw_operands(9, rows, out_rows, out_cols, int(bounds[-1]))

    from jax_aiter.flydsl.kernels.primus_turbo.grouped_gemm import (
        grouped_gemm_mxfp4_kernel,
    )

    long_schedule = grouped_gemm_mxfp4_kernel._select_gmxfp4_wgrad_cfg(
        rows, 2, out_rows, out_cols
    )
    short_schedule = grouped_gemm_mxfp4_kernel._select_gmxfp4_wgrad_cfg(
        8192, 2, out_rows, out_cols
    )
    assert long_schedule != short_schedule

    default = jax.jit(grouped_mxfp4_dw_wholeloop)(*operands, bounds)
    short = jax.jit(
        functools.partial(grouped_mxfp4_dw_wholeloop, schedule_rows=8192)
    )(*operands, bounds)
    np.testing.assert_array_equal(_bits(short), _bits(default))
    _assert_dw_matches_reference(default, operands, sizes)


@pytest.mark.parametrize("out_rows,out_cols", [(1024, 4096), (4096, 1024)])
def test_wholeloop_dw_shape_table_agrees_bitwise(out_rows, out_cols):
    from jax_aiter.flydsl.kernels.primus_turbo.grouped_gemm import (
        grouped_gemm_mxfp4_kernel,
    )
    from jax_aiter.ops import moe_mxfp4

    experts = 64
    rows = experts * moe_mxfp4._DW_SCHEDULE_ROWS[0]
    outside = experts * (moe_mxfp4._DW_SCHEDULE_ROWS[1] + 256)
    table = moe_mxfp4._DW_SCHEDULES[(out_rows, out_cols, experts)]
    assert moe_mxfp4._dw_schedule(rows, experts, out_rows, out_cols) == table
    fallback = moe_mxfp4._dw_schedule(outside, experts, out_rows, out_cols)
    assert fallback == grouped_gemm_mxfp4_kernel._select_gmxfp4_wgrad_cfg(
        outside, experts, out_rows, out_cols
    )
    assert fallback != table
    for shape in ((7168, 2048, 32), (2048, 7168, 32), (4096, 7168, 32)):
        assert moe_mxfp4._dw_schedule(rows, shape[2], *shape[:2]) == (
            grouped_gemm_mxfp4_kernel._select_gmxfp4_wgrad_cfg(
                rows, shape[2], *shape[:2]
            )
        )

    sizes = np.random.default_rng(3).choice([0, 4, 12, 20], experts) * 256
    bounds = _bounds(sizes)
    operands = _dw_operands(11, rows, out_rows, out_cols, int(bounds[-1]))
    tabled = jax.jit(grouped_mxfp4_dw_wholeloop)(*operands, bounds)
    selected = jax.jit(
        functools.partial(grouped_mxfp4_dw_wholeloop, schedule_rows=outside)
    )(*operands, bounds)
    np.testing.assert_array_equal(_bits(tabled), _bits(selected))
    _assert_dw_matches_reference(tabled, operands, sizes)


def test_wholeloop_rejects_unsupported_shapes():
    def forward_operands(rows, contraction, output, experts):
        return (
            jnp.zeros((rows, contraction // 2), jnp.uint8),
            jnp.zeros((experts, output, contraction // 2), jnp.uint8),
            jnp.zeros((rows, contraction // 32), jnp.uint8),
            jnp.zeros((experts, output, contraction // 32), jnp.uint8),
        )

    with pytest.raises(ValueError, match="K % 256"):
        grouped_mxfp4_fwd_da_wholeloop(
            *forward_operands(256, 384, 256, 2), _bounds([256, 0])
        )
    with pytest.raises(ValueError, match="group bounds"):
        grouped_mxfp4_fwd_da_wholeloop(
            *forward_operands(256, 512, 256, 2), _bounds([256])
        )

    tokens = 384
    dw_operands = (
        jnp.zeros((256, tokens // 2), jnp.uint8),
        jnp.zeros((256, tokens // 32), jnp.uint8),
        jnp.zeros((256, tokens // 2), jnp.uint8),
        jnp.zeros((256, tokens // 32), jnp.uint8),
    )
    with pytest.raises(ValueError, match="divisible by 256"):
        grouped_mxfp4_dw_wholeloop(*dw_operands, _bounds([tokens]))
