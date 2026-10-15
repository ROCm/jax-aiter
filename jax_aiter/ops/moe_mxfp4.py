# SPDX-License-Identifier: MIT
"""Raw FlyDSL MXFP4 operations for ragged grouped expert training.

These functions intentionally define no differentiation rule. MaxText's
grouped-GEMM custom VJP selects the forward/dA and dW bodies explicitly.

The FlyDSL kernel sources these operations launch are license-gated and
distributed separately; they are not part of this repository or its wheels.
Without them, the operations raise ``Mxfp4KernelsUnavailableError`` and
``mxfp4_moe_kernels_available()`` returns ``False``. The rest of JAX-AITER does
not depend on them.
"""

from __future__ import annotations

import functools
import importlib.util

import jax
import jax.numpy as jnp

from ..flydsl import flydsl_call

_SCALE_BLOCK = 32
_GROUP_ALIGNMENT = 128
_WHOLELOOP_ALIGNMENT = 256
_INT32_MAX = (1 << 31) - 1
_KERNEL_PACKAGE = "jax_aiter.flydsl.kernels.mxfp4"


class Mxfp4KernelsUnavailableError(ImportError):
    """The separately distributed MXFP4 MoE kernel sources are not installed."""


def mxfp4_moe_kernels_available() -> bool:
    """Whether the license-gated MXFP4 MoE kernel sources are installed."""
    return importlib.util.find_spec(_KERNEL_PACKAGE) is not None


def _require_kernels() -> None:
    if not mxfp4_moe_kernels_available():
        raise Mxfp4KernelsUnavailableError(
            "JAX-AITER's grouped MXFP4 MoE operations need FlyDSL kernel "
            "sources that are license-gated and distributed separately; they "
            "are not part of this repository or its wheels. Install an "
            "authorized copy at jax_aiter/flydsl/kernels/mxfp4/ "
            "(docker/mxfp4/mxfp4-kernels.sha256 lists the expected files). "
            "The rest of JAX-AITER does not need them."
        )


def _row_chunks(
    rows: int,
    max_elements_per_row: int,
    *,
    alignment: int,
    max_index_elements: int = _INT32_MAX,
) -> tuple[tuple[int, int], ...]:
    """Split rows so every kernel operand stays within int32 indexing."""
    if rows < 0 or max_elements_per_row <= 0:
        raise ValueError(
            f"Invalid row chunk shape rows={rows}, "
            f"max_elements_per_row={max_elements_per_row}"
        )
    if alignment <= 0 or max_index_elements <= 0:
        raise ValueError(
            f"Invalid row chunk limits alignment={alignment}, "
            f"max_index_elements={max_index_elements}"
        )
    if rows * max_elements_per_row <= max_index_elements:
        return ((0, rows),)

    chunk_rows = max_index_elements // max_elements_per_row
    chunk_rows = chunk_rows // alignment * alignment
    if chunk_rows <= 0:
        raise ValueError(
            f"One aligned row needs {max_elements_per_row * alignment} "
            f"indexed elements, exceeding the limit {max_index_elements}"
        )
    return tuple(
        (start, min(start + chunk_rows, rows))
        for start in range(0, rows, chunk_rows)
    )


def _local_group_bounds(
    bounds: jax.Array, start: int, end: int
) -> jax.Array:
    """Clip absolute expert bounds to one static row chunk."""
    return jnp.clip(bounds, start, end) - jnp.int32(start)


def _validate_matrix(name: str, value, *, dtypes) -> jax.Array:
    value = jnp.asarray(value)
    if value.ndim != 2:
        raise ValueError(f"{name} must be rank 2, got shape {value.shape}")
    if value.dtype not in dtypes:
        raise TypeError(f"{name} must have dtype in {dtypes}, got {value.dtype}")
    return value


def _flydsl_dtype_name(dtype) -> str:
    if dtype == jnp.bfloat16:
        return "bfloat16"
    if dtype == jnp.float32:
        return "float32"
    raise TypeError(f"Unsupported MXFP4 quantization input dtype: {dtype}")


def _require_int32_indexing(name: str, element_count: int) -> None:
    if element_count > _INT32_MAX:
        raise ValueError(
            f"{name} has {element_count} flattened elements, exceeding the "
            f"signed-int32 kernel indexing limit {_INT32_MAX}"
        )


def _row_limit_operands(row_limit, start: int, end: int) -> tuple[jax.Array, ...]:
    """Device ``int32[1]`` limit for rows ``[start, end)``, or nothing."""
    if row_limit is None:
        return ()
    limit = jnp.clip(jnp.asarray(row_limit, jnp.int32) - jnp.int32(start), 0, end - start)
    return (limit.reshape(1),)


def _quantize_mxfp4_dim0_single(value, row_limit_operands=()):
    """One int32-addressable dim0 quantization launch."""
    rows, cols = value.shape
    # The kernel bases its input per wave; only the packed output is int32-indexed.
    _require_int32_indexing("dim0 packed output", rows * cols // 2)

    _require_kernels()
    from ..flydsl.kernels.mxfp4.quant_dim0 import (
        GROUPS_PER_WG,
        _compile_quant_dim0,
    )

    launcher = _compile_quant_dim0(
        cols, _flydsl_dtype_name(value.dtype), bool(row_limit_operands)
    )
    num_groups = rows * (cols // _SCALE_BLOCK)
    grid_blocks = (num_groups + GROUPS_PER_WG - 1) // GROUPS_PER_WG
    outputs = flydsl_call(
        value.reshape(-1),
        *row_limit_operands,
        kernel=launcher,
        out_shape=(
            jax.ShapeDtypeStruct((rows * cols // 2,), jnp.uint8),
            jax.ShapeDtypeStruct(
                (rows * cols // _SCALE_BLOCK,), jnp.uint8
            ),
        ),
        scalars={"n_groups": num_groups, "grid_blocks": grid_blocks},
        output_positions=(1, 2),
    )
    packed, scales = outputs
    return (
        packed.reshape(rows, cols // 2),
        scales.reshape(rows, cols // _SCALE_BLOCK),
    )


def _quantize_mxfp4_dim0_maybe_empty(value):
    """Skip a tail launch when masked expert padding made it all zero."""
    rows, cols = value.shape
    return jax.lax.cond(
        jnp.any(value != jnp.zeros((), value.dtype)),
        _quantize_mxfp4_dim0_single,
        lambda _: (
            jnp.zeros((rows, cols // 2), jnp.uint8),
            jnp.zeros((rows, cols // _SCALE_BLOCK), jnp.uint8),
        ),
        value,
    )


def quantize_mxfp4_dim0(
    value, *, row_limit=None, _max_index_elements: int = _INT32_MAX
):
    """Compact RCEIL quantization along the contiguous matrix dimension.

    ``[M, K] -> ([M, K/2] packed FP4, [M, K/32] E8M0)``.

    ``row_limit`` (device int32 scalar) leaves output rows at or past it
    unwritten, for buffers whose tail rows no reader uses.
    """
    value = _validate_matrix(
        "value", value, dtypes=(jnp.bfloat16, jnp.float32)
    )
    rows, cols = value.shape
    if cols % _SCALE_BLOCK:
        raise ValueError(f"dim0 quantization requires K % 32 == 0, got {cols}")
    chunks = _row_chunks(
        rows,
        cols // 2,
        alignment=1,
        max_index_elements=_max_index_elements,
    )
    outputs = []
    for chunk_index, (start, end) in enumerate(chunks):
        chunk = value[start:end]
        if row_limit is not None:
            outputs.append(
                _quantize_mxfp4_dim0_single(
                    chunk, _row_limit_operands(row_limit, start, end)
                )
            )
        else:
            outputs.append(
                _quantize_mxfp4_dim0_single(chunk)
                if chunk_index == 0
                else _quantize_mxfp4_dim0_maybe_empty(chunk)
            )
    if len(outputs) == 1:
        return outputs[0]
    return (
        jnp.concatenate([output[0] for output in outputs], axis=0),
        jnp.concatenate([output[1] for output in outputs], axis=0),
    )


_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def requantize_mxfp4_dim0_transposed(packed, scales):
    """Re-quantize dim0 MXFP4 ``x[..., R, C]`` as dim0 MXFP4 of ``x^T``.

    ``([..., R, C/2], [..., R, C/32]) -> ([..., C, R/2], [..., C, R/32])``.
    The values are dequantized exactly to bfloat16 and quantized again in
    32-blocks along ``R``, so the result carries both roundings and is not
    bit-equal to quantizing the original ``x^T``.
    """
    if packed.dtype != jnp.uint8 or scales.dtype != jnp.uint8:
        raise TypeError(
            f"Expected uint8 packed values and scales, got {packed.dtype}, {scales.dtype}"
        )
    *lead, rows, packed_cols = packed.shape
    cols = packed_cols * 2
    if scales.shape != (*lead, rows, cols // _SCALE_BLOCK):
        raise ValueError(f"Scales {scales.shape} do not match packed {packed.shape}")
    if rows % _SCALE_BLOCK:
        raise ValueError(f"Transposed requantization requires R % 32 == 0, got {rows}")

    table = jnp.asarray(_E2M1_VALUES + tuple(-v for v in _E2M1_VALUES), jnp.float32)
    codes = jnp.stack((packed & 0xF, packed >> 4), axis=-1)
    values = table[codes].reshape(*lead, rows, cols // _SCALE_BLOCK, _SCALE_BLOCK)
    scale = jax.lax.bitcast_convert_type(scales.astype(jnp.uint32) << 23, jnp.float32)
    values = (values * scale[..., None]).astype(jnp.bfloat16)
    transposed = values.reshape(*lead, rows, cols).swapaxes(-1, -2)
    out_packed, out_scales = quantize_mxfp4_dim0(transposed.reshape(-1, rows))
    return (
        out_packed.reshape(*lead, cols, rows // 2),
        out_scales.reshape(*lead, cols, rows // _SCALE_BLOCK),
    )


def _quantize_mxfp4_dim0_gather_single(value, row_index, limit_operand):
    """One int32-addressable gathered dim0 quantization launch."""
    cols = value.shape[1]
    rows = row_index.shape[0]
    _require_int32_indexing("gather packed output", rows * cols // 2)

    _require_kernels()
    from ..flydsl.kernels.mxfp4.quant_dim0 import (
        GROUPS_PER_WG,
        _compile_quant_dim0_gather,
    )

    launcher = _compile_quant_dim0_gather(cols, _flydsl_dtype_name(value.dtype))
    num_groups = rows * (cols // _SCALE_BLOCK)
    grid_blocks = (num_groups + GROUPS_PER_WG - 1) // GROUPS_PER_WG
    packed, scales = flydsl_call(
        value.reshape(-1),
        row_index,
        limit_operand,
        kernel=launcher,
        out_shape=(
            jax.ShapeDtypeStruct((rows * cols // 2,), jnp.uint8),
            jax.ShapeDtypeStruct((rows * cols // _SCALE_BLOCK,), jnp.uint8),
        ),
        scalars={"n_groups": num_groups, "grid_blocks": grid_blocks},
        output_positions=(2, 3),
    )
    return (
        packed.reshape(rows, cols // 2),
        scales.reshape(rows, cols // _SCALE_BLOCK),
    )


def quantize_mxfp4_dim0_gather(
    value, row_index, *, row_limit=None, _max_index_elements: int = _INT32_MAX
):
    """dim0 quantization of gathered rows without materializing the gather.

    Output row ``r`` is the dim0 quantization of ``value[row_index[r]]``; a
    negative index gives an all-zero row. ``row_limit`` (device int32 scalar)
    leaves output rows at or past it unwritten.
    """
    value, row_index = _validate_gather(value, row_index)
    cols = value.shape[1]
    rows = row_index.shape[0]
    if value.size * value.dtype.itemsize >= 1 << 32:
        raise ValueError(
            f"dim0 gather source {value.shape} {value.dtype} exceeds 4 GiB "
            "buffer addressing"
        )
    limit = rows if row_limit is None else row_limit
    chunks = _row_chunks(
        rows, cols // 2, alignment=1, max_index_elements=_max_index_elements
    )
    outputs = [
        _quantize_mxfp4_dim0_gather_single(
            value, row_index[start:end], _row_limit_operands(limit, start, end)[0]
        )
        for start, end in chunks
    ]
    if len(outputs) == 1:
        return outputs[0]
    return (
        jnp.concatenate([output[0] for output in outputs], axis=0),
        jnp.concatenate([output[1] for output in outputs], axis=0),
    )


def _quantize_mxfp4_dim1_single(value, row_limit_operands=()):
    """One int32-addressable dim1 quantization launch."""
    rows, cols = value.shape
    # The kernel bases its input per tile; only the packed output is int32-indexed.
    _require_int32_indexing("dim1 packed output", rows * cols // 2)

    _require_kernels()
    from ..flydsl.kernels.mxfp4.quant_dim1 import (
        _K_TILE,
        _compile_quant_dim1,
        _pick_layout,
    )

    dtype_name = _flydsl_dtype_name(value.dtype)
    stack = _pick_layout(rows, cols, value.dtype.itemsize)
    launcher, tile_rows = _compile_quant_dim1(
        cols, dtype_name, stack, bool(row_limit_operands)
    )
    outputs = flydsl_call(
        value.reshape(-1),
        *row_limit_operands,
        kernel=launcher,
        out_shape=(
            jax.ShapeDtypeStruct((cols * rows // 2,), jnp.uint8),
            jax.ShapeDtypeStruct(
                (cols * rows // _SCALE_BLOCK,), jnp.uint8
            ),
        ),
        scalars={
            "m_rows": rows,
            "grid_m": rows // tile_rows,
            "grid_k": (cols + _K_TILE - 1) // _K_TILE,
        },
        output_positions=(1, 2),
    )
    packed, scales = outputs
    return (
        packed.reshape(cols, rows // 2),
        scales.reshape(cols, rows // _SCALE_BLOCK),
    )


def _quantize_mxfp4_dim1_maybe_empty(value):
    """Skip a transposing tail launch when every source row is zero."""
    rows, cols = value.shape
    return jax.lax.cond(
        jnp.any(value != jnp.zeros((), value.dtype)),
        _quantize_mxfp4_dim1_single,
        lambda _: (
            jnp.zeros((cols, rows // 2), jnp.uint8),
            jnp.zeros((cols, rows // _SCALE_BLOCK), jnp.uint8),
        ),
        value,
    )


def quantize_mxfp4_dim1(
    value, *, row_limit=None, _max_index_elements: int = _INT32_MAX
):
    """Compact RCEIL quantization while transposing the token dimension.

    ``[M, K] -> ([K, M/2] packed FP4, [K, M/32] E8M0)``.

    ``row_limit`` (device int32 scalar) skips source rows from the 32-128 row
    tile holding it onward; their output columns are left unwritten.
    """
    value = _validate_matrix(
        "value", value, dtypes=(jnp.bfloat16, jnp.float32)
    )
    rows, cols = value.shape
    if rows % _SCALE_BLOCK or cols % _SCALE_BLOCK:
        raise ValueError(
            "dim1 quantization requires M and K divisible by 32, got "
            f"{value.shape}"
        )
    chunks = _row_chunks(
        rows,
        cols // 2,
        alignment=_SCALE_BLOCK,
        max_index_elements=_max_index_elements,
    )
    outputs = []
    for chunk_index, (start, end) in enumerate(chunks):
        chunk = value[start:end]
        if row_limit is not None:
            outputs.append(
                _quantize_mxfp4_dim1_single(
                    chunk, _row_limit_operands(row_limit, start, end)
                )
            )
        else:
            outputs.append(
                _quantize_mxfp4_dim1_single(chunk)
                if chunk_index == 0
                else _quantize_mxfp4_dim1_maybe_empty(chunk)
            )
    if len(outputs) == 1:
        return outputs[0]
    return (
        jnp.concatenate([output[0] for output in outputs], axis=1),
        jnp.concatenate([output[1] for output in outputs], axis=1),
    )


def _validate_gather(value, row_index):
    value = _validate_matrix(
        "value", value, dtypes=(jnp.bfloat16, jnp.float32)
    )
    row_index = jnp.asarray(row_index)
    if row_index.ndim != 1 or row_index.dtype != jnp.int32:
        raise TypeError(
            f"row_index must be rank-1 int32, got {row_index.shape} {row_index.dtype}"
        )
    if value.shape[1] % _SCALE_BLOCK:
        raise ValueError(f"MXFP4 quantization requires K % 32 == 0, got {value.shape[1]}")
    _require_int32_indexing("gather source", value.size)
    return value, row_index


def _quantize_mxfp4_dim1_gather_single(value, row_index, limit_operand):
    """One int32-addressable gathered dim1 quantization launch."""
    cols = value.shape[1]
    rows = row_index.shape[0]
    _require_int32_indexing("gather packed output", rows * cols // 2)

    _require_kernels()
    from ..flydsl.kernels.mxfp4.quant_dim1 import (
        _K_TILE,
        _compile_quant_dim1,
        _pick_layout,
    )

    stack = _pick_layout(rows, cols, value.dtype.itemsize)
    launcher, tile_rows = _compile_quant_dim1(
        cols, _flydsl_dtype_name(value.dtype), stack, gather=True
    )
    packed, scales = flydsl_call(
        value.reshape(-1),
        row_index,
        limit_operand,
        kernel=launcher,
        out_shape=(
            jax.ShapeDtypeStruct((cols * rows // 2,), jnp.uint8),
            jax.ShapeDtypeStruct((cols * rows // _SCALE_BLOCK,), jnp.uint8),
        ),
        scalars={
            "m_rows": rows,
            "grid_m": rows // tile_rows,
            "grid_k": (cols + _K_TILE - 1) // _K_TILE,
        },
        output_positions=(2, 3),
    )
    return (
        packed.reshape(cols, rows // 2),
        scales.reshape(cols, rows // _SCALE_BLOCK),
    )


def quantize_mxfp4_dim1_gather(
    value, row_index, *, row_limit=None, _max_index_elements: int = _INT32_MAX
):
    """dim1 quantization of gathered rows without materializing the gather.

    Equals ``quantize_mxfp4_dim1`` of the ``[M, K]`` matrix whose row ``r`` is
    ``value[row_index[r]]``, or zero for a negative index. ``row_limit`` is as
    in :func:`quantize_mxfp4_dim1`.
    """
    value, row_index = _validate_gather(value, row_index)
    cols = value.shape[1]
    rows = row_index.shape[0]
    if rows % _SCALE_BLOCK:
        raise ValueError(f"dim1 quantization requires M % 32 == 0, got {rows}")
    limit = rows if row_limit is None else row_limit
    chunks = _row_chunks(
        rows, cols // 2, alignment=_SCALE_BLOCK, max_index_elements=_max_index_elements
    )
    outputs = [
        _quantize_mxfp4_dim1_gather_single(
            value, row_index[start:end], _row_limit_operands(limit, start, end)[0]
        )
        for start, end in chunks
    ]
    if len(outputs) == 1:
        return outputs[0]
    return (
        jnp.concatenate([output[0] for output in outputs], axis=1),
        jnp.concatenate([output[1] for output in outputs], axis=1),
    )


def _grouped_mxfp4_fwd_da_single(
    lhs_packed,
    rhs_packed,
    lhs_scales,
    rhs_scales,
    group_end_offsets,
):
    """Ragged grouped MXFP4 GEMM used by both forward and dA.

    Args:
      lhs_packed: ``[M, K/2]`` uint8.
      rhs_packed: ``[E, N, K/2]`` uint8.
      lhs_scales: ``[M, K/32]`` uint8.
      rhs_scales: ``[E, N, K/32]`` uint8.
      group_end_offsets: device ``int32[E]`` cumulative padded row ends.
    """
    lhs_packed = _validate_matrix(
        "lhs_packed", lhs_packed, dtypes=(jnp.uint8,)
    )
    rhs_packed = jnp.asarray(rhs_packed)
    lhs_scales = _validate_matrix(
        "lhs_scales", lhs_scales, dtypes=(jnp.uint8,)
    )
    rhs_scales = jnp.asarray(rhs_scales)
    offsets = jnp.asarray(group_end_offsets)

    if rhs_packed.ndim != 3 or rhs_packed.dtype != jnp.uint8:
        raise TypeError("rhs_packed must be uint8[E, N, K/2]")
    if rhs_scales.ndim != 3 or rhs_scales.dtype != jnp.uint8:
        raise TypeError("rhs_scales must be uint8[E, N, K/32]")
    if offsets.ndim != 1 or offsets.dtype != jnp.int32:
        raise TypeError("group_end_offsets must be int32[E]")

    rows, packed_k = lhs_packed.shape
    experts, out_cols, rhs_packed_k = rhs_packed.shape
    logical_k = packed_k * 2
    if rhs_packed_k != packed_k:
        raise ValueError("lhs and rhs packed contraction dimensions differ")
    if offsets.shape != (experts,):
        raise ValueError(
            f"Expected {experts} group offsets, got {offsets.shape}"
        )
    if logical_k % 128 or (logical_k + 255) // 256 < 4:
        raise ValueError(
            "grouped forward/dA requires K divisible by 128 and at least "
            f"four pipeline steps, got K={logical_k}"
        )
    _require_int32_indexing("grouped lhs", rows * packed_k)
    _require_int32_indexing(
        "grouped rhs", experts * out_cols * rhs_packed_k
    )
    _require_int32_indexing("grouped output", rows * out_cols)
    if lhs_scales.shape != (rows, logical_k // _SCALE_BLOCK):
        raise ValueError("lhs_scales shape does not match lhs_packed")
    if rhs_scales.shape != (
        experts,
        out_cols,
        logical_k // _SCALE_BLOCK,
    ):
        raise ValueError("rhs_scales shape does not match rhs_packed")

    _require_kernels()
    from ..flydsl.kernels.mxfp4._fwd_kernel_fp4 import (
        BLOCK_R,
        cached_launch,
        ceildiv,
        pick_block_c,
    )

    block_cols = pick_block_c(out_cols)
    col_tiles = ceildiv(out_cols, block_cols)
    block_count = (ceildiv(rows, BLOCK_R) + experts) * col_tiles
    launcher = cached_launch(logical_k, out_cols, experts, block_cols)

    # The kernel writes every row, including zeros after offsets[-1], so the
    # output needs no zero-filled seed.
    output = flydsl_call(
        lhs_packed.reshape(-1),
        rhs_packed.reshape(-1),
        lhs_scales.reshape(-1),
        rhs_scales.reshape(-1),
        offsets,
        kernel=launcher,
        out_shape=jax.ShapeDtypeStruct(
            (rows * out_cols,), jnp.bfloat16
        ),
        scalars={
            "n_blocks": block_count,
            "n_c_tiles": col_tiles,
            "out_m": rows,
            "out_n": out_cols,
        },
        output_positions=(2,),
    )
    return output.reshape(rows, out_cols)


def grouped_mxfp4_fwd_da(
    lhs_packed,
    rhs_packed,
    lhs_scales,
    rhs_scales,
    group_end_offsets,
    *,
    _max_index_elements: int = _INT32_MAX,
):
    """Ragged grouped forward/dA, split into int32-addressable row slabs."""
    lhs_packed = _validate_matrix(
        "lhs_packed", lhs_packed, dtypes=(jnp.uint8,)
    )
    rhs_packed = jnp.asarray(rhs_packed)
    lhs_scales = _validate_matrix(
        "lhs_scales", lhs_scales, dtypes=(jnp.uint8,)
    )
    offsets = jnp.asarray(group_end_offsets)
    if rhs_packed.ndim != 3 or rhs_packed.dtype != jnp.uint8:
        raise TypeError("rhs_packed must be uint8[E, N, K/2]")
    if offsets.ndim != 1 or offsets.dtype != jnp.int32:
        raise TypeError("group_end_offsets must be int32[E]")

    rows, packed_k = lhs_packed.shape
    experts, out_cols, _ = rhs_packed.shape
    if offsets.shape != (experts,):
        raise ValueError(
            f"Expected {experts} group offsets, got {offsets.shape}"
        )
    chunks = _row_chunks(
        rows,
        max(packed_k, lhs_scales.shape[1], out_cols),
        alignment=_GROUP_ALIGNMENT,
        max_index_elements=_max_index_elements,
    )
    if len(chunks) == 1:
        return _grouped_mxfp4_fwd_da_single(
            lhs_packed,
            rhs_packed,
            lhs_scales,
            rhs_scales,
            offsets,
        )

    bounds = jnp.concatenate((jnp.zeros((1,), jnp.int32), offsets))
    outputs = []
    for chunk_index, (start, end) in enumerate(chunks):
        local_offsets = _local_group_bounds(bounds, start, end)[1:]
        operands = (
            lhs_packed[start:end],
            rhs_packed,
            lhs_scales[start:end],
            rhs_scales,
            local_offsets,
        )
        if chunk_index == 0:
            outputs.append(_grouped_mxfp4_fwd_da_single(*operands))
        else:
            outputs.append(
                jax.lax.cond(
                    local_offsets[-1] > 0,
                    lambda args: _grouped_mxfp4_fwd_da_single(*args),
                    lambda args: jnp.zeros(
                        (args[0].shape[0], args[1].shape[1]),
                        jnp.bfloat16,
                    ),
                    operands,
                )
            )
    return jnp.concatenate(outputs, axis=0)


def grouped_mxfp4_dw(
    grad_output_transposed,
    grad_output_scales,
    input_transposed,
    input_scales,
    group_end_offsets,
    *,
    out_dtype=jnp.bfloat16,
):
    """Ragged grouped weight gradient with FP32 accumulation."""
    grad_output_transposed = _validate_matrix(
        "grad_output_transposed",
        grad_output_transposed,
        dtypes=(jnp.uint8,),
    )
    input_transposed = _validate_matrix(
        "input_transposed", input_transposed, dtypes=(jnp.uint8,)
    )
    grad_output_scales = _validate_matrix(
        "grad_output_scales", grad_output_scales, dtypes=(jnp.uint8,)
    )
    input_scales = _validate_matrix(
        "input_scales", input_scales, dtypes=(jnp.uint8,)
    )
    offsets = jnp.asarray(group_end_offsets)
    if offsets.ndim != 1 or offsets.dtype != jnp.int32:
        raise TypeError("group_end_offsets must be int32[E]")
    if out_dtype != jnp.bfloat16:
        raise TypeError("grouped_mxfp4_dw currently returns bfloat16 only")

    out_rows, packed_tokens = grad_output_transposed.shape
    out_cols, packed_tokens_rhs = input_transposed.shape
    if packed_tokens_rhs != packed_tokens:
        raise ValueError("dW operands must share the packed token dimension")
    token_rows = packed_tokens * 2
    if token_rows % _GROUP_ALIGNMENT:
        raise ValueError(
            f"dW token capacity must be divisible by 128, got {token_rows}"
        )
    expected_scale_cols = token_rows // _SCALE_BLOCK
    if grad_output_scales.shape != (out_rows, expected_scale_cols):
        raise ValueError("grad_output_scales shape mismatch")
    if input_scales.shape != (out_cols, expected_scale_cols):
        raise ValueError("input_scales shape mismatch")
    _require_int32_indexing(
        "grouped dW operand",
        max(out_rows, out_cols) * packed_tokens,
    )
    _require_int32_indexing(
        "grouped dW output", offsets.shape[0] * out_rows * out_cols
    )

    experts = offsets.shape[0]
    _require_kernels()
    from ..flydsl.kernels.mxfp4._wgrad_kernel_fp4 import (
        BLOCK_C,
        BLOCK_R,
        cached_launch,
        ceildiv_py,
    )

    row_tiles = ceildiv_py(out_rows, BLOCK_R)
    col_tiles = ceildiv_py(out_cols, BLOCK_C)
    block_count = experts * row_tiles * col_tiles
    scale_pair = token_rows % (2 * _GROUP_ALIGNMENT) == 0
    order = "lpt" if experts <= 32 else "id"
    launcher = cached_launch(
        out_rows, out_cols, experts, scale_pair, order
    )

    output = flydsl_call(
        grad_output_transposed.reshape(-1),
        input_transposed.reshape(-1),
        grad_output_scales.reshape(-1),
        input_scales.reshape(-1),
        offsets,
        kernel=launcher,
        out_shape=jax.ShapeDtypeStruct(
            (experts * out_rows * out_cols,), jnp.float32
        ),
        scalars={
            "n_blocks": block_count,
            "n_r_tiles": row_tiles,
            "n_c_tiles": col_tiles,
            "out_r": out_rows,
            "out_c": out_cols,
            "m_row": token_rows,
        },
        output_positions=(2,),
    )
    return output.reshape(experts, out_rows, out_cols).astype(out_dtype)


def _ceil_div(value: int, divisor: int) -> int:
    return -(-value // divisor)


def _validate_group_bounds(group_bounds) -> jax.Array:
    bounds = jnp.asarray(group_bounds)
    if bounds.ndim != 1 or bounds.dtype != jnp.int32 or bounds.shape[0] < 2:
        raise TypeError("group_bounds must be int32[E+1] with E >= 1")
    return bounds


def _e8m0_words(scales: jax.Array) -> jax.Array:
    """View ``[..., K/32]`` E8M0 bytes as little-endian ``[..., K/128]`` int32."""
    return jax.lax.bitcast_convert_type(
        scales.reshape(*scales.shape[:-1], scales.shape[-1] // 4, 4), jnp.int32
    )


def _int64_bound_table(bounds: jax.Array) -> jax.Array:
    """Lay out non-negative int32 bounds as the int64 table the kernels read."""
    return jnp.stack((bounds, jnp.zeros_like(bounds)), axis=-1).reshape(-1)


@functools.lru_cache(maxsize=None)
def _wholeloop_fwd_da_launcher(contraction, experts, out_cols, schedule):
    _require_kernels()
    from ..flydsl.kernels.mxfp4.wholeloop import grouped_gemm_mxfp4_kernel

    group_m, num_xcds, group_n, xcd_span, streaming_store = schedule
    launcher, _ = grouped_gemm_mxfp4_kernel._compile_grouped_mxfp4_nt_fused(
        contraction,
        experts,
        out_cols,
        group_m,
        num_xcds,
        group_n,
        wlv=10,
        elgk=9,
        out_fp16=False,
        k_real=contraction,
        span=xcd_span,
        cst_nt=streaming_store,
    )
    return launcher


@functools.lru_cache(maxsize=None)
def _wholeloop_dw_launcher(out_rows, out_cols, experts, schedule):
    _require_kernels()
    from ..flydsl.kernels.mxfp4.wholeloop import grouped_gemm_mxfp4_kernel

    group_m, num_xcds, group_n, streaming_store, wg_tiles = schedule
    # The bf16 epilogue needs a full LDS drain at the phase barrier (elgk=0).
    # dW also drains vmcnt there (wlv=0): its last global->LDS loads of a phase
    # are read by other waves in the next phase with no further wait, so any
    # left in flight race under memory-latency spikes (wlv=10 corrupted ~1 in
    # 2000 calls).
    launcher, _ = grouped_gemm_mxfp4_kernel._compile_grouped_mxfp4_wgrad_fused(
        out_rows,
        out_cols,
        experts,
        group_m,
        num_xcds,
        group_n,
        streaming_store,
        wg_tiles,
        wlv=0,
        elgk=0,
        out_fp16=False,
        beta_is_one=False,
    )
    return launcher


def _grouped_mxfp4_fwd_da_wholeloop_single(
    lhs_packed,
    rhs_packed,
    lhs_scales,
    rhs_scales,
    group_bounds,
    *,
    schedule_rows: int | None = None,
    zero_tail: bool = True,
):
    """Whole-loop ragged grouped MXFP4 GEMM used by both forward and dA.

    Operands match :func:`grouped_mxfp4_fwd_da`, except that ``group_bounds``
    is device ``int32[E+1]`` holding ``[0, end_0, ..., end_{E-1}]``. Groups
    need no row alignment. The kernel does not write rows at or after
    ``group_bounds[-1]``; ``zero_tail`` masks them to zero.

    ``schedule_rows`` is the static row count that picks the tile schedule
    and defaults to the row capacity of ``lhs_packed``.
    """
    lhs_packed = _validate_matrix(
        "lhs_packed", lhs_packed, dtypes=(jnp.uint8,)
    )
    rhs_packed = jnp.asarray(rhs_packed)
    lhs_scales = _validate_matrix(
        "lhs_scales", lhs_scales, dtypes=(jnp.uint8,)
    )
    rhs_scales = jnp.asarray(rhs_scales)
    bounds = _validate_group_bounds(group_bounds)
    if rhs_packed.ndim != 3 or rhs_packed.dtype != jnp.uint8:
        raise TypeError("rhs_packed must be uint8[E, N, K/2]")
    if rhs_scales.ndim != 3 or rhs_scales.dtype != jnp.uint8:
        raise TypeError("rhs_scales must be uint8[E, N, K/32]")

    rows, packed_k = lhs_packed.shape
    experts, out_cols, rhs_packed_k = rhs_packed.shape
    contraction = packed_k * 2
    if rhs_packed_k != packed_k:
        raise ValueError("lhs and rhs packed contraction dimensions differ")
    if bounds.shape != (experts + 1,):
        raise ValueError(
            f"Expected {experts + 1} group bounds, got {bounds.shape}"
        )
    if contraction % _WHOLELOOP_ALIGNMENT:
        raise ValueError(
            f"whole-loop forward/dA requires K % 256 == 0, got K={contraction}"
        )
    if lhs_scales.shape != (rows, contraction // _SCALE_BLOCK):
        raise ValueError("lhs_scales shape does not match lhs_packed")
    if rhs_scales.shape != (
        experts,
        out_cols,
        contraction // _SCALE_BLOCK,
    ):
        raise ValueError("rhs_scales shape does not match rhs_packed")
    # The output needs no limit: the kernel stores through 64-bit row-band bases.
    _require_int32_indexing("grouped lhs", rows * packed_k)
    _require_int32_indexing("grouped rhs", experts * out_cols * packed_k)

    _require_kernels()
    from ..flydsl.kernels.mxfp4.wholeloop import grouped_gemm_mxfp4_kernel

    schedule = grouped_gemm_mxfp4_kernel._select_gmxfp4_nt_cfg(
        rows if schedule_rows is None else int(schedule_rows), experts
    )
    launcher = _wholeloop_fwd_da_launcher(
        contraction, experts, out_cols, schedule
    )
    scale_words = contraction // 128
    slab_rows = (
        _ceil_div(rows, _WHOLELOOP_ALIGNMENT) + experts
    ) * _WHOLELOOP_ALIGNMENT
    rhs_scale_rows = (
        _ceil_div(out_cols, _WHOLELOOP_ALIGNMENT) * _WHOLELOOP_ALIGNMENT
    )
    output, _, _ = flydsl_call(
        lhs_packed,
        rhs_packed,
        _e8m0_words(lhs_scales),
        _e8m0_words(rhs_scales),
        _int64_bound_table(bounds),
        kernel=launcher,
        out_shape=(
            jax.ShapeDtypeStruct((rows, out_cols), jnp.bfloat16),
            jax.ShapeDtypeStruct((slab_rows * scale_words,), jnp.int32),
            jax.ShapeDtypeStruct(
                (experts * rhs_scale_rows * scale_words,), jnp.int32
            ),
        ),
        scalars={"c_m": rows, "slab_rows": slab_rows},
        output_positions=(2, 5, 6),
        compile_hints=grouped_gemm_mxfp4_kernel._GMXFP4_SCHED_HINTS,
    )
    if zero_tail:
        live_rows = jnp.arange(rows, dtype=jnp.int32)[:, None] < bounds[-1]
        output = jnp.where(live_rows, output, jnp.zeros((), output.dtype))
    return output


def grouped_mxfp4_fwd_da_wholeloop(
    lhs_packed,
    rhs_packed,
    lhs_scales,
    rhs_scales,
    group_bounds,
    *,
    schedule_rows: int | None = None,
    zero_tail: bool = True,
    _max_index_elements: int = _INT32_MAX,
):
    """Whole-loop grouped forward/dA split into addressable row slabs."""
    lhs_packed = _validate_matrix(
        "lhs_packed", lhs_packed, dtypes=(jnp.uint8,)
    )
    rhs_packed = jnp.asarray(rhs_packed)
    lhs_scales = _validate_matrix(
        "lhs_scales", lhs_scales, dtypes=(jnp.uint8,)
    )
    bounds = _validate_group_bounds(group_bounds)
    if rhs_packed.ndim != 3 or rhs_packed.dtype != jnp.uint8:
        raise TypeError("rhs_packed must be uint8[E, N, K/2]")

    rows, packed_k = lhs_packed.shape
    experts, out_cols, _ = rhs_packed.shape
    if bounds.shape != (experts + 1,):
        raise ValueError(
            f"Expected {experts + 1} group bounds, got {bounds.shape}"
        )
    chunks = _row_chunks(
        rows,
        max(packed_k, lhs_scales.shape[1]),
        alignment=_WHOLELOOP_ALIGNMENT,
        max_index_elements=_max_index_elements,
    )
    effective_schedule_rows = rows if schedule_rows is None else schedule_rows
    if len(chunks) == 1:
        return _grouped_mxfp4_fwd_da_wholeloop_single(
            lhs_packed,
            rhs_packed,
            lhs_scales,
            rhs_scales,
            bounds,
            schedule_rows=effective_schedule_rows,
            zero_tail=zero_tail,
        )

    outputs = []
    for chunk_index, (start, end) in enumerate(chunks):
        local_bounds = _local_group_bounds(bounds, start, end)
        operands = (
            lhs_packed[start:end],
            rhs_packed,
            lhs_scales[start:end],
            rhs_scales,
            local_bounds,
        )
        if chunk_index == 0:
            outputs.append(
                _grouped_mxfp4_fwd_da_wholeloop_single(
                    *operands,
                    schedule_rows=effective_schedule_rows,
                    zero_tail=zero_tail,
                )
            )
        else:
            outputs.append(
                jax.lax.cond(
                    local_bounds[-1] > 0,
                    lambda args: _grouped_mxfp4_fwd_da_wholeloop_single(
                        *args,
                        schedule_rows=effective_schedule_rows,
                        zero_tail=zero_tail,
                    ),
                    lambda args: jnp.zeros(
                        (args[0].shape[0], args[1].shape[1]),
                        jnp.bfloat16,
                    ),
                    operands,
                )
            )
    return jnp.concatenate(outputs, axis=0)


def grouped_mxfp4_dw_wholeloop(
    grad_output_transposed,
    grad_output_scales,
    input_transposed,
    input_scales,
    group_bounds,
    *,
    schedule_rows: int | None = None,
):
    """Whole-loop ragged grouped weight gradient, returning bf16 ``[E, R, C]``.

    Operands match :func:`grouped_mxfp4_dw`, except that ``group_bounds`` is
    device ``int32[E+1]`` holding ``[0, end_0, ..., end_{E-1}]`` with every
    entry a multiple of 256. Token rows at or after ``group_bounds[-1]`` are
    ignored.

    ``schedule_rows`` is the static token count that picks the tile schedule
    and defaults to the token capacity of the operands.
    """
    grad_output_transposed = _validate_matrix(
        "grad_output_transposed",
        grad_output_transposed,
        dtypes=(jnp.uint8,),
    )
    input_transposed = _validate_matrix(
        "input_transposed", input_transposed, dtypes=(jnp.uint8,)
    )
    grad_output_scales = _validate_matrix(
        "grad_output_scales", grad_output_scales, dtypes=(jnp.uint8,)
    )
    input_scales = _validate_matrix(
        "input_scales", input_scales, dtypes=(jnp.uint8,)
    )
    bounds = _validate_group_bounds(group_bounds)

    out_rows, packed_tokens = grad_output_transposed.shape
    out_cols, packed_tokens_rhs = input_transposed.shape
    if packed_tokens_rhs != packed_tokens:
        raise ValueError("dW operands must share the packed token dimension")
    token_rows = packed_tokens * 2
    if token_rows % _WHOLELOOP_ALIGNMENT:
        raise ValueError(
            f"whole-loop dW token capacity must be divisible by 256, got {token_rows}"
        )
    expected_scale_cols = token_rows // _SCALE_BLOCK
    if grad_output_scales.shape != (out_rows, expected_scale_cols):
        raise ValueError("grad_output_scales shape mismatch")
    if input_scales.shape != (out_cols, expected_scale_cols):
        raise ValueError("input_scales shape mismatch")
    experts = bounds.shape[0] - 1
    _require_int32_indexing(
        "grouped dW operand",
        max(out_rows, out_cols) * packed_tokens,
    )
    _require_int32_indexing(
        "grouped dW output", experts * out_rows * out_cols
    )

    _require_kernels()
    from ..flydsl.kernels.mxfp4.wholeloop import grouped_gemm_mxfp4_kernel

    schedule = grouped_gemm_mxfp4_kernel._select_gmxfp4_wgrad_cfg(
        token_rows if schedule_rows is None else int(schedule_rows),
        experts,
        out_rows,
        out_cols,
    )
    launcher = _wholeloop_dw_launcher(out_rows, out_cols, experts, schedule)
    scale_words = token_rows // 128
    output, _, _ = flydsl_call(
        grad_output_transposed,
        input_transposed,
        _e8m0_words(grad_output_scales),
        _e8m0_words(input_scales),
        _int64_bound_table(bounds),
        kernel=launcher,
        out_shape=(
            jax.ShapeDtypeStruct((experts, out_rows, out_cols), jnp.bfloat16),
            jax.ShapeDtypeStruct(
                (
                    _ceil_div(out_rows, _WHOLELOOP_ALIGNMENT)
                    * _WHOLELOOP_ALIGNMENT
                    * scale_words,
                ),
                jnp.int32,
            ),
            jax.ShapeDtypeStruct(
                (
                    _ceil_div(out_cols, _WHOLELOOP_ALIGNMENT)
                    * _WHOLELOOP_ALIGNMENT
                    * scale_words,
                ),
                jnp.int32,
            ),
        ),
        scalars={"m_total": token_rows},
        output_positions=(2, 5, 6),
    )
    return output


__all__ = [
    "quantize_mxfp4_dim0",
    "quantize_mxfp4_dim1",
    "grouped_mxfp4_fwd_da",
    "grouped_mxfp4_fwd_da_wholeloop",
    "grouped_mxfp4_dw",
    "grouped_mxfp4_dw_wholeloop",
]
