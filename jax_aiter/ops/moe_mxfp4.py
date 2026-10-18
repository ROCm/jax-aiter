# SPDX-License-Identifier: MIT
"""Raw FlyDSL MXFP4 operations for ragged grouped expert training.

These functions intentionally define no differentiation rule. MaxText's
grouped-GEMM custom VJP selects the forward/dA and dW bodies explicitly.

The casts and grouped GEMMs launch the Primus-Turbo FlyDSL kernels vendored
under ``jax_aiter/flydsl/kernels/primus_turbo/`` and JAX-AITER's own casts
built on them (``jax_aiter/flydsl/kernels/primus_turbo_mxfp4_quant.py``).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from ..flydsl import flydsl_call

_SCALE_BLOCK = 32
_COL_TILE = 128  # rows and columns of the column-wise cast's tiles
_WHOLELOOP_ALIGNMENT = 256
_INT32_MAX = (1 << 31) - 1


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


def _validate_cast_input(value) -> jax.Array:
    value = jnp.asarray(value)
    if value.dtype != jnp.bfloat16:
        raise TypeError(
            f"MXFP4 casts take bfloat16 input, got {value.dtype}; convert with "
            ".astype(jnp.bfloat16) first. The scales then follow the rounded "
            "values, so they can differ from scales of the original values."
        )
    return _validate_matrix("value", value, dtypes=(jnp.bfloat16,))


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


def _chunk_row_limit(row_limit, start: int):
    """``row_limit`` relative to a row chunk starting at ``start``."""
    if row_limit is None or start == 0:
        return row_limit
    return jnp.asarray(row_limit, jnp.int32) - jnp.int32(start)


def _concat_chunks(outputs, axis: int):
    if len(outputs) == 1:
        return outputs[0]
    return (
        jnp.concatenate([output[0] for output in outputs], axis=axis),
        jnp.concatenate([output[1] for output in outputs], axis=axis),
    )


def _empty_mxfp4(shape: tuple[int, int]):
    """All-zero packed values and scales of a ``shape`` cast output."""
    rows, cols = shape
    return (
        jnp.zeros((rows, cols // 2), jnp.uint8),
        jnp.zeros((rows, cols // _SCALE_BLOCK), jnp.uint8),
    )


def _quantize_chunks(quantize, value, chunks, row_limit, *, rowwise: bool):
    """``quantize`` each row chunk of ``value`` and join the outputs.

    Without ``row_limit``, a tail chunk that masked expert padding made all
    zero skips its launch.
    """

    def zeros(chunk):
        rows, cols = chunk.shape
        return _empty_mxfp4((rows, cols) if rowwise else (cols, rows))

    outputs = []
    for start, end in chunks:
        chunk = value[start:end]
        if row_limit is not None or start == 0:
            outputs.append(
                quantize(chunk, row_limit=_chunk_row_limit(row_limit, start))
            )
        else:
            outputs.append(
                jax.lax.cond(
                    jnp.any(chunk != jnp.zeros((), chunk.dtype)),
                    quantize,
                    zeros,
                    chunk,
                )
            )
    return _concat_chunks(outputs, axis=0 if rowwise else 1)


def _take_rows(value, row_index):
    """``value[row_index]``, with a zero row for each negative index."""
    if value.shape[0] == 0:
        return jnp.zeros((row_index.shape[0], value.shape[1]), value.dtype)
    rows = jnp.take(value, jnp.maximum(row_index, 0), axis=0)
    return jnp.where((row_index >= 0)[:, None], rows, jnp.zeros((), value.dtype))


def _gather_source_fits(value) -> bool:
    """Whether the gather kernels can address ``value`` and a row of 32-bit offsets past its end."""
    return value.size * value.dtype.itemsize + 4 * value.shape[1] <= 1 << 32


def quantize_mxfp4_dim0(
    value, *, row_limit=None, _max_index_elements: int = _INT32_MAX
):
    """Compact RCEIL quantization along the contiguous matrix dimension.

    ``[M, K] -> ([M, K/2] packed FP4, [M, K/32] E8M0)``.

    ``row_limit`` (device int32 scalar) leaves output rows at or past it
    unwritten, for buffers whose tail rows no reader uses.
    """
    from .moe_mxfp4_ptquant import quantize_mxfp4_rows

    value = _validate_cast_input(value)
    rows, cols = value.shape
    if cols % _SCALE_BLOCK:
        raise ValueError(f"dim0 quantization requires K % 32 == 0, got {cols}")
    if value.size == 0:
        return _empty_mxfp4((rows, cols))
    # The kernel bases its input per wave; only the packed output is int32-indexed.
    # A smaller _max_index_elements is the test hook that forces chunked launches.
    chunks = _row_chunks(
        rows, cols // 2, alignment=1, max_index_elements=_max_index_elements
    )
    return _quantize_chunks(
        quantize_mxfp4_rows, value, chunks, row_limit, rowwise=True
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


def quantize_mxfp4_dim0_gather(
    value, row_index, *, row_limit=None, _max_index_elements: int = _INT32_MAX
):
    """dim0 quantization of gathered rows without materializing the gather.

    Output row ``r`` is the dim0 quantization of ``value[row_index[r]]``; a
    negative index gives an all-zero row. ``row_limit`` (device int32 scalar)
    leaves output rows at or past it unwritten. A source too large for the
    gather kernel's buffer addressing is gathered first, with the same bytes.
    """
    from .moe_mxfp4_ptquant import quantize_mxfp4_rows_gather

    value, row_index = _validate_gather(value, row_index)
    if not _gather_source_fits(value):
        return quantize_mxfp4_dim0(
            _take_rows(value, row_index),
            row_limit=row_limit,
            _max_index_elements=_max_index_elements,
        )
    rows, cols = row_index.shape[0], value.shape[1]
    if rows == 0 or cols == 0:
        return _empty_mxfp4((rows, cols))
    chunks = _row_chunks(
        rows, cols // 2, alignment=1, max_index_elements=_max_index_elements
    )
    return _concat_chunks(
        [
            quantize_mxfp4_rows_gather(
                value,
                row_index[start:end],
                row_limit=_chunk_row_limit(row_limit, start),
            )
            for start, end in chunks
        ],
        axis=0,
    )


def quantize_mxfp4_dim1(
    value, *, row_limit=None, _max_index_elements: int = _INT32_MAX
):
    """Compact RCEIL quantization while transposing the token dimension.

    ``[M, K] -> ([K, M/2] packed FP4, [K, M/32] E8M0)``.

    ``row_limit`` (device int32 scalar) skips the kernel's 128- or 256-row
    tiles that start at or past it; their output columns are left unwritten.
    """
    from .moe_mxfp4_ptquant import quantize_mxfp4_cols

    value = _validate_cast_input(value)
    rows, cols = value.shape
    if rows % _SCALE_BLOCK or cols % _SCALE_BLOCK:
        raise ValueError(
            "dim1 quantization requires M and K divisible by 32, got "
            f"{value.shape}"
        )
    if value.size == 0:
        return _empty_mxfp4((cols, rows))
    # The kernel takes 128 x 128 tiles. Zero rows padded to fill them form 32-row
    # blocks of their own, and zero columns only add output rows; both are cut off.
    pad_rows, pad_cols = -rows % _COL_TILE, -cols % _COL_TILE
    if pad_rows or pad_cols:
        value = jnp.pad(value, ((0, pad_rows), (0, pad_cols)))
    # The kernel bases its input per tile; only the packed output is int32-indexed.
    # A smaller _max_index_elements is the test hook that forces chunked launches.
    chunks = _row_chunks(
        value.shape[0],
        value.shape[1] // 2,
        alignment=_COL_TILE,
        max_index_elements=_max_index_elements,
    )
    packed, scales = _quantize_chunks(
        quantize_mxfp4_cols, value, chunks, row_limit, rowwise=False
    )
    if pad_rows or pad_cols:
        packed, scales = packed[:cols, : rows // 2], scales[:cols, : rows // _SCALE_BLOCK]
    return packed, scales


def _validate_gather(value, row_index):
    value = _validate_cast_input(value)
    row_index = jnp.asarray(row_index)
    if row_index.ndim != 1 or row_index.dtype != jnp.int32:
        raise TypeError(
            f"row_index must be rank-1 int32, got {row_index.shape} {row_index.dtype}"
        )
    if value.shape[1] % _SCALE_BLOCK:
        raise ValueError(f"MXFP4 quantization requires K % 32 == 0, got {value.shape[1]}")
    return value, row_index


def quantize_mxfp4_dim1_gather(
    value, row_index, *, row_limit=None, _max_index_elements: int = _INT32_MAX
):
    """dim1 quantization of gathered rows without materializing the gather.

    Equals ``quantize_mxfp4_dim1`` of the ``[M, K]`` matrix whose row ``r`` is
    ``value[row_index[r]]``, or zero for a negative index. ``row_limit`` is as
    in :func:`quantize_mxfp4_dim1`. ``K % 128 != 0``, or a source too large
    for the gather kernel's buffer addressing, is gathered first, with the
    same bytes.
    """
    from .moe_mxfp4_ptquant import quantize_mxfp4_cols_gather

    value, row_index = _validate_gather(value, row_index)
    rows, cols = row_index.shape[0], value.shape[1]
    if rows % _SCALE_BLOCK:
        raise ValueError(f"dim1 quantization requires M % 32 == 0, got {rows}")
    if cols % _COL_TILE or not _gather_source_fits(value):
        return quantize_mxfp4_dim1(
            _take_rows(value, row_index),
            row_limit=row_limit,
            _max_index_elements=_max_index_elements,
        )
    if rows == 0 or cols == 0:
        return _empty_mxfp4((cols, rows))
    # Index -1 reads a zero row, so it fills the kernel's 128-row tiles.
    pad_rows = -rows % _COL_TILE
    if pad_rows:
        row_index = jnp.pad(row_index, (0, pad_rows), constant_values=-1)
    chunks = _row_chunks(
        row_index.shape[0],
        cols // 2,
        alignment=_COL_TILE,
        max_index_elements=_max_index_elements,
    )
    packed, scales = _concat_chunks(
        [
            quantize_mxfp4_cols_gather(
                value,
                row_index[start:end],
                row_limit=_chunk_row_limit(row_limit, start),
            )
            for start, end in chunks
        ],
        axis=1,
    )
    if pad_rows:
        packed, scales = packed[:, : rows // 2], scales[:, : rows // _SCALE_BLOCK]
    return packed, scales


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
    from ..flydsl.kernels import primus_turbo_mxfp4

    group_m, num_xcds, group_n, xcd_span, streaming_store = schedule
    launcher, n_blocks = primus_turbo_mxfp4.compile_grouped_mxfp4_nt(
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
    return launcher, n_blocks


@functools.lru_cache(maxsize=None)
def _wholeloop_dw_launcher(out_rows, out_cols, experts, schedule):
    from ..flydsl.kernels.primus_turbo.grouped_gemm import grouped_gemm_mxfp4_kernel

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

    Args:
      lhs_packed: ``[M, K/2]`` uint8.
      rhs_packed: ``[E, N, K/2]`` uint8.
      lhs_scales: ``[M, K/32]`` uint8.
      rhs_scales: ``[E, N, K/32]`` uint8.
      group_bounds: device ``int32[E+1]`` holding ``[0, end_0, ..., end_{E-1}]``.

    Groups need no row alignment. The kernel does not write rows at or after
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

    from ..flydsl.kernels.primus_turbo.grouped_gemm import grouped_gemm_mxfp4_kernel

    schedule = grouped_gemm_mxfp4_kernel._select_gmxfp4_nt_cfg(
        rows if schedule_rows is None else int(schedule_rows), experts
    )
    launcher, n_blocks = _wholeloop_fwd_da_launcher(
        contraction, experts, out_cols, schedule
    )
    scale_words = contraction // 128
    slab_rows = (
        _ceil_div(rows, _WHOLELOOP_ALIGNMENT) + experts
    ) * _WHOLELOOP_ALIGNMENT
    preshuffle_words = (
        grouped_gemm_mxfp4_kernel._PRESHUF_FO
        * grouped_gemm_mxfp4_kernel._PRESHUF_BLK
    )
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
        scalars={
            "c_m": rows,
            "c_n": out_cols,
            "slab_rows": slab_rows,
            "a_pre_grid": _ceil_div(slab_rows * scale_words, preshuffle_words),
            "grid_upper": (
                _ceil_div(rows, _WHOLELOOP_ALIGNMENT) + experts
            ) * n_blocks,
        },
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

    ``grad_output_transposed`` ``[R, T/2]`` and ``input_transposed``
    ``[C, T/2]`` are uint8 FP4 packed along the ``T`` token rows, with E8M0
    scales ``grad_output_scales`` ``[R, T/32]`` and ``input_scales``
    ``[C, T/32]``. ``group_bounds`` is device ``int32[E+1]`` holding
    ``[0, end_0, ..., end_{E-1}]`` with every entry a multiple of 256. Token
    rows at or after ``group_bounds[-1]`` are ignored.

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

    from ..flydsl.kernels.primus_turbo.grouped_gemm import grouped_gemm_mxfp4_kernel

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
    "grouped_mxfp4_fwd_da_wholeloop",
    "grouped_mxfp4_dw_wholeloop",
]
