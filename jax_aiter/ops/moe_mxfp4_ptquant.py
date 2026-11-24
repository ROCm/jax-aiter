# SPDX-License-Identifier: Apache-2.0
"""Primus-Turbo's fused MXFP4 dual cast behind the JAX-AITER cast contract.

One bf16 read gives both :func:`quantize_mxfp4_dim0` and
:func:`quantize_mxfp4_dim1` outputs; :func:`quantize_mxfp4_rows` and
:func:`quantize_mxfp4_cols` give the same bytes for one of them, and their
``_gather`` forms those of :func:`quantize_mxfp4_dim0_gather` and
:func:`quantize_mxfp4_dim1_gather`. Primus-Turbo rounds the E8M0 scale up
when ``amax_bits + bias`` carries into the exponent; ``RCEIL_BIAS`` makes that
``ceil_pow2(amax / 6)``, the scale our casts compute. Primus-Turbo's own
``scale_rounding_mode`` presets 0, 1, 2 are ``1 << 21``, ``1 << 22``, ``3 << 19``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..flydsl import flydsl_call
from .moe_mxfp4 import _row_limit_operands, _validate_cast_input, _validate_gather

RCEIL_BIAS = (1 << 22) - 1
PRESET_BIAS = {0: 1 << 21, 1: 1 << 22, 2: 3 << 19}
_SCALE_BLOCK = 32


def quantize_mxfp4_rows(value, *, row_limit=None, scale_rounding_bias: int = RCEIL_BIAS):
    """``[M, K] -> ([M, K/2] packed, [M, K/32] scales)``, the dim0 half of :func:`quantize_mxfp4_dual`."""
    return _quantize_one_direction(value, True, row_limit, scale_rounding_bias)


def quantize_mxfp4_cols(value, *, row_limit=None, scale_rounding_bias: int = RCEIL_BIAS):
    """``[M, K] -> ([K, M/2] packed, [K, M/32] scales)``, the dim1 half of :func:`quantize_mxfp4_dual`."""
    return _quantize_one_direction(value, False, row_limit, scale_rounding_bias)


def quantize_mxfp4_rows_gather(value, row_index, *, row_limit=None, scale_rounding_bias: int = RCEIL_BIAS):
    """:func:`quantize_mxfp4_rows` of the ``[M, K]`` matrix whose row ``r`` is ``value[row_index[r]]``, or zero for a negative index."""
    return _quantize_one_direction(value, True, row_limit, scale_rounding_bias, row_index)


def quantize_mxfp4_cols_gather(value, row_index, *, row_limit=None, scale_rounding_bias: int = RCEIL_BIAS):
    """:func:`quantize_mxfp4_cols` of the ``[M, K]`` matrix whose row ``r`` is ``value[row_index[r]]``, or zero for a negative index."""
    return _quantize_one_direction(value, False, row_limit, scale_rounding_bias, row_index)


def _quantize_one_direction(value, rowwise: bool, row_limit, scale_rounding_bias: int, row_index=None):
    from ..flydsl.kernels.primus_turbo_mxfp4_quant import compile_mxfp4_gather, compile_mxfp4_one_direction

    value = _validate_cast_input(value)
    if row_index is None:
        rows, cols = value.shape
        launch, grid_x = compile_mxfp4_one_direction(rows, cols, rowwise)
        operands = (value.reshape(-1),)
    else:
        value, row_index = _validate_gather(value, row_index)
        rows, cols = row_index.shape[0], value.shape[1]
        launch, grid_x = compile_mxfp4_gather(value.shape[0], rows, cols, rowwise)
        operands = (value.reshape(-1), row_index)
    packed, scales = flydsl_call(
        *operands,
        *_row_limit_operands(rows if row_limit is None else row_limit, 0, rows),
        kernel=launch,
        out_shape=(
            jax.ShapeDtypeStruct((rows * cols // 2,), jnp.uint8),
            jax.ShapeDtypeStruct((rows * cols // _SCALE_BLOCK,), jnp.uint8),
        ),
        scalars={"R": rows, "C": cols, "SCALE_ROUNDING_BIAS": scale_rounding_bias, "grid_x": grid_x},
        output_positions=(len(operands), len(operands) + 1),
    )
    if rowwise:
        return packed.reshape(rows, cols // 2), scales.reshape(rows, cols // _SCALE_BLOCK)
    return packed.reshape(cols, rows // 2), scales.reshape(cols, rows // _SCALE_BLOCK)


def quantize_mxfp4_dual(value, *, row_limit=None, scale_rounding_bias: int = RCEIL_BIAS):
    """``[M, K] -> (dim0 packed [M, K/2], dim0 scales [M, K/32], dim1 packed [K, M/2], dim1 scales [K, M/32])``.

    Needs ``M % 128 == 0`` and ``K % 256 == 0``. With ``row_limit`` (a traced
    int32 scalar), 128-row tiles starting at or past it are not written: dim0
    rows and dim1 columns there are unspecified, as for the row-limited
    :func:`quantize_mxfp4_dim0` and :func:`quantize_mxfp4_dim1`.
    """
    from ..flydsl.kernels.primus_turbo_mxfp4_quant import compile_mxfp4_dual

    value = _validate_cast_input(value)
    rows, cols = value.shape
    limit_operands = _row_limit_operands(row_limit, 0, rows)
    launch, grid_x = compile_mxfp4_dual(rows, cols, bool(limit_operands))
    # The kernel addresses every buffer in bytes from its base pointer, so the
    # bf16 input and uint8 outputs need no int32 views.
    row_packed, row_scales, col_packed, col_scales = flydsl_call(
        value.reshape(-1),
        *limit_operands,
        kernel=launch,
        out_shape=(
            jax.ShapeDtypeStruct((rows * cols // 2,), jnp.uint8),
            jax.ShapeDtypeStruct((rows * cols // _SCALE_BLOCK,), jnp.uint8),
            jax.ShapeDtypeStruct((cols * rows // 2,), jnp.uint8),
            jax.ShapeDtypeStruct((cols * rows // _SCALE_BLOCK,), jnp.uint8),
        ),
        scalars={
            "R": rows,
            "C": cols,
            "CP": cols,
            "RP": rows,
            "SR_SEED": 0,
            "SCALE_ROUNDING_BIAS": scale_rounding_bias,
            "grid_x": grid_x,
        },
        output_positions=(1, 2, 3, 4),
    )
    return (
        row_packed.reshape(rows, cols // 2),
        row_scales.reshape(rows, cols // _SCALE_BLOCK),
        col_packed.reshape(cols, rows // 2),
        col_scales.reshape(cols, rows // _SCALE_BLOCK),
    )
