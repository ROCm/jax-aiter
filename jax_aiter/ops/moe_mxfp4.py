# SPDX-License-Identifier: MIT
"""Raw FlyDSL MXFP4 operations for ragged grouped expert training.

These functions intentionally define no differentiation rule. MaxText's
grouped-GEMM custom VJP selects the forward/dA and dW bodies explicitly.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from ..flydsl import flydsl_call

_SCALE_BLOCK = 32
_GROUP_ALIGNMENT = 128
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


def _quantize_mxfp4_dim0_single(value):
    """One int32-addressable dim0 quantization launch."""
    rows, cols = value.shape
    _require_int32_indexing("dim0 input", rows * cols)

    from ..flydsl.kernels.mxfp4.quant_dim0 import (
        BLOCK,
        _compile_quant_dim0,
    )

    launcher = _compile_quant_dim0(cols, _flydsl_dtype_name(value.dtype))
    num_groups = rows * (cols // _SCALE_BLOCK)
    grid_blocks = (num_groups + BLOCK - 1) // BLOCK
    outputs = flydsl_call(
        value.reshape(-1),
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
    value, *, _max_index_elements: int = _INT32_MAX
):
    """Compact RCEIL quantization along the contiguous matrix dimension.

    ``[M, K] -> ([M, K/2] packed FP4, [M, K/32] E8M0)``.
    """
    value = _validate_matrix(
        "value", value, dtypes=(jnp.bfloat16, jnp.float32)
    )
    rows, cols = value.shape
    if cols % _SCALE_BLOCK:
        raise ValueError(f"dim0 quantization requires K % 32 == 0, got {cols}")
    chunks = _row_chunks(
        rows,
        cols,
        alignment=1,
        max_index_elements=_max_index_elements,
    )
    outputs = []
    for chunk_index, (start, end) in enumerate(chunks):
        chunk = value[start:end]
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


def _quantize_mxfp4_dim1_single(value):
    """One int32-addressable dim1 quantization launch."""
    rows, cols = value.shape
    _require_int32_indexing("dim1 input", rows * cols)

    from ..flydsl.kernels.mxfp4.quant_dim1 import (
        _K_TILE,
        _compile_quant_dim1,
        _pick_layout,
    )

    dtype_name = _flydsl_dtype_name(value.dtype)
    stack = _pick_layout(rows, cols, value.dtype.itemsize)
    launcher, tile_rows = _compile_quant_dim1(cols, dtype_name, stack)
    outputs = flydsl_call(
        value.reshape(-1),
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
    value, *, _max_index_elements: int = _INT32_MAX
):
    """Compact RCEIL quantization while transposing the token dimension.

    ``[M, K] -> ([K, M/2] packed FP4, [K, M/32] E8M0)``.
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
        cols,
        alignment=_SCALE_BLOCK,
        max_index_elements=_max_index_elements,
    )
    outputs = []
    for chunk_index, (start, end) in enumerate(chunks):
        chunk = value[start:end]
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
    from ..flydsl.kernels.mxfp4.wholeloop import grouped_gemm_mxfp4_kernel

    group_m, num_xcds, group_n, streaming_store, wg_tiles = schedule
    # The bf16 epilogue needs a full LDS drain at the phase barrier (elgk=0).
    launcher, _ = grouped_gemm_mxfp4_kernel._compile_grouped_mxfp4_wgrad_fused(
        out_rows,
        out_cols,
        experts,
        group_m,
        num_xcds,
        group_n,
        streaming_store,
        wg_tiles,
        wlv=10,
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
    _require_int32_indexing("grouped lhs", rows * packed_k)
    _require_int32_indexing("grouped rhs", experts * out_cols * packed_k)
    _require_int32_indexing("grouped output", rows * out_cols)

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
        max(packed_k, lhs_scales.shape[1], out_cols),
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
