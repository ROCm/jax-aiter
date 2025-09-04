# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl
from typing import Optional


@triton.jit
def _per_token_quant(
    x,
    y_scale_ptr,
    row_max,
    row_idx,
    DTYPE_MAX: tl.constexpr,
    scale_ub_ptr=None,
    EPS_8BIT: tl.constexpr = 1e-12,
    CLAMP_MAX: tl.constexpr = False,
    CLAMP_OUT: tl.constexpr = False,
):
    """
    #TODO: Add Doc
    """

    if CLAMP_MAX:
        ub = tl.load(scale_ub_ptr)
        row_max = tl.clamp(row_max, EPS_8BIT, ub)

    scale_out = row_max / DTYPE_MAX
    scale_out = tl.where(scale_out == 0, 1.0, scale_out)

    scale_recip = 1 / scale_out

    qx = x * scale_recip

    if CLAMP_OUT:
        qx = tl.clamp(qx, -DTYPE_MAX, DTYPE_MAX)

    tl.store(y_scale_ptr + row_idx, scale_out.to(y_scale_ptr.dtype.element_ty))

    return qx


@triton.jit
def _rms_norm_kernel(
    # Pointers to matrices
    input_ptr,
    output_ptr,
    g_ptr,
    rsigma_ptr,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `input_row_stride` is
    # how much to increase `input_ptr` by to get the element one row down.
    input_row_stride,
    output_row_stride,
    # Matrix dimensions
    n_rows,
    n_cols,
    # Epsilon to avoid division by zero
    epsilon,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
    USE_BLOCKED: tl.constexpr,
    NUM_PRGMS: tl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call rms_norm function
    below.

    Applies Root Mean Square Layer Normalization over a mini-batch of inputs.

    Key parameters:
    - Input: The input tensor to be normalized with shape (n_rows, n_cols).
    - Output: The output tensor with shape (n_rows, n_cols).
    - G: The learnable weights tensor with shape (n_cols, ).
    """
    # Map the program id to the first row of input and output it should compute.
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    if USE_BLOCKED:
        # Persistent loop for rows
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=1):
            row_input_ptr = input_ptr + row_idx * input_row_stride
            row_output_ptr = output_ptr + row_idx * output_row_stride

            # Accumulate sum of squares
            n_cols_blks = tl.cdiv(n_cols, BLOCK_SIZE) - 1
            sum_squares = 0.0
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                input_ptrs = tl.multiple_of(input_ptrs, (16,))
                x = tl.load(input_ptrs).to(tl.float32)
                sum_squares += tl.sum(x * x, axis=0)

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            input_ptrs = tl.multiple_of(input_ptrs, (16,))
            x = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(
                tl.float32
            )
            sum_squares += tl.sum(x * x, axis=0)

            # Compute normalization factor
            mean_square = sum_squares / n_cols
            norm_factor = tl.rsqrt(mean_square + epsilon)

            # Store rsigma (norm_factor)
            tl.store(rsigma_ptr + row_idx, norm_factor)

            # Normalize and write output
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                input_ptrs = tl.multiple_of(input_ptrs, (16,))
                x = tl.load(input_ptrs).to(tl.float32)
                g_ptrs = g_ptr + cols
                g = tl.load(g_ptrs).to(tl.float32)
                rms_norm = x * norm_factor * g
                output_ptrs = row_output_ptr + cols
                tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty))

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            x = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(
                tl.float32
            )
            g_ptrs = g_ptr + cols
            g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
            rms_norm = x * norm_factor * g
            output_ptrs = row_output_ptr + cols
            tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty), mask=mask)

    else:
        mask = col_offsets < n_cols
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=2):
            input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
            input_ptrs = tl.multiple_of(input_ptrs, (16,))
            row = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(
                tl.float32
            )
            g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            row_norm = row * row
            row_norm = tl.sum(row_norm, axis=-1)
            norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)

            # Store rsigma (norm_factor)
            tl.store(rsigma_ptr + row_idx, norm_factor)

            rms_norm = row * norm_factor * g

            output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
            output_ptrs = tl.multiple_of(output_ptrs, (16,))
            tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty), mask=mask)


@triton.jit
def _quant_rms_norm_kernel(
    # Pointers to matrices
    input_ptr,
    output_ptr,
    x_scale_ptr,
    y_scale_ptr,
    g_ptr,
    # Auxiliary tensor to store intermediate data
    aux_ptr,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `input_row_stride` is
    # how much to increase `input_ptr` by to get the element one row down.
    input_row_stride,
    output_row_stride,
    aux_row_stride,
    # Matrix dimensions
    n_rows,
    n_cols,
    # Epsilon to avoid division by zero
    epsilon,
    # Optional pointers
    scale_ub_ptr,  # Pointer to the scale upper bound tensor
    out_intermediate_ptr,  # Pointer to the intermediate output tensor
    # Dtype max for quantization
    DTYPE_MAX: tl.constexpr,
    # Meta-parameters
    IS_SMOOTH: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    CLAMP_OUT: tl.constexpr,
    DUMP_INTERMEDIATE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    USE_BLOCKED: tl.constexpr,
    NUM_PRGMS: tl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call rmsnorm2d_fwd_with_smoothquant or
    rmsnorm2d_fwd_with_dynamicquant functions below.

    Applies Root Mean Square Layer Normalization over a mini-batch of inputs and quantizes the result.

    Key parameters:
    - Input: The input tensor to be normalized with shape (n_rows, n_cols).
    - Output: The output tensor with shape (n_rows, n_cols).
    - X_scale: The tensor to be multiplied by the RMSNorm output if IS_SMOOTH is true, with shape (n_cols, ).
    - Y_scale: The tensor where the scale for each row will be stored with shape (n_rows, ).
    - G: The learnable weights tensor with shape (n_cols, ).
    """
    # Map the program id to the first row of input and output it should compute.
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    if USE_BLOCKED:
        # Persistent loop for rows
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=1):
            row_input_ptr = input_ptr + row_idx * input_row_stride
            row_output_ptr = output_ptr + row_idx * output_row_stride
            row_aux_ptr = aux_ptr + row_idx * aux_row_stride

            # Accumulate sum of squares
            n_cols_blks = tl.cdiv(n_cols, BLOCK_SIZE) - 1
            sum_squares = 0.0
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                input_ptrs = tl.multiple_of(input_ptrs, (16,))
                x = tl.load(input_ptrs).to(tl.float32)
                sum_squares += tl.sum(x * x, axis=0)

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            input_ptrs = tl.multiple_of(input_ptrs, (16,))
            x = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(
                tl.float32
            )
            sum_squares += tl.sum(x * x, axis=0)

            # Compute normalization factor
            mean_square = sum_squares / n_cols
            norm_factor = tl.rsqrt(mean_square + epsilon)

            row_max = 0.0

            # Normalize and write output temporarily as fp32
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                input_ptrs = tl.multiple_of(input_ptrs, (16,))
                x = tl.load(input_ptrs).to(tl.float32)
                g_ptrs = g_ptr + cols
                g = tl.load(g_ptrs).to(tl.float32)
                rms_norm = x * norm_factor * g

                if DUMP_INTERMEDIATE:
                    tl.store(
                        out_intermediate_ptr + row_idx * n_cols + cols,
                        rms_norm.to(out_intermediate_ptr.type.element_ty),
                    )

                if IS_SMOOTH:
                    x_scale_ptrs = x_scale_ptr + cols
                    x_scale_ptrs = tl.multiple_of(x_scale_ptrs, (16,))
                    x_scale = tl.load(x_scale_ptrs)
                    rms_norm *= x_scale

                blk_max = tl.max(tl.abs(rms_norm), axis=-1)
                row_max = max(row_max, blk_max)

                aux_ptrs = row_aux_ptr + cols
                tl.store(aux_ptrs, rms_norm)

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            x = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(
                tl.float32
            )
            g_ptrs = g_ptr + cols
            g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
            rms_norm = x * norm_factor * g

            if DUMP_INTERMEDIATE:
                tl.store(
                    out_intermediate_ptr + row_idx * n_cols + cols,
                    rms_norm.to(out_intermediate_ptr.type.element_ty),
                    mask=mask,
                )

            if IS_SMOOTH:
                x_scale_ptrs = x_scale_ptr + cols
                x_scale = tl.load(
                    x_scale_ptrs, mask=mask, other=0.0, cache_modifier=".cg"
                )
                rms_norm *= x_scale

            blk_max = tl.max(tl.abs(rms_norm), axis=-1)
            row_max = max(row_max, blk_max)

            aux_ptrs = row_aux_ptr + cols
            tl.store(aux_ptrs, rms_norm, mask=mask)

            # Apply quantization and write output
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                aux_ptrs = row_aux_ptr + cols
                aux_ptrs = tl.multiple_of(aux_ptrs, (16,))
                aux = tl.load(aux_ptrs)

                output = _per_token_quant(
                    aux,
                    y_scale_ptr,
                    row_max,
                    row_idx,
                    DTYPE_MAX,
                    scale_ub_ptr=scale_ub_ptr,
                    CLAMP_MAX=CLAMP_MAX,
                    CLAMP_OUT=CLAMP_OUT,
                )

                output_ptrs = row_output_ptr + cols
                tl.store(output_ptrs, output.to(output_ptr.dtype.element_ty))

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            aux_ptrs = row_aux_ptr + cols
            aux = tl.load(aux_ptrs, mask=mask, other=0.0, cache_modifier=".cg")

            output = _per_token_quant(
                aux,
                y_scale_ptr,
                row_max,
                row_idx,
                DTYPE_MAX,
                scale_ub_ptr=scale_ub_ptr,
                CLAMP_MAX=CLAMP_MAX,
                CLAMP_OUT=CLAMP_OUT,
            )

            output_ptrs = row_output_ptr + cols
            tl.store(output_ptrs, output.to(output_ptr.dtype.element_ty), mask=mask)
    else:
        mask = col_offsets < n_cols
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=2):
            input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
            input_ptrs = tl.multiple_of(input_ptrs, (16,))
            row = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(
                tl.float32
            )
            g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            row_norm = row * row
            row_norm = tl.sum(row_norm, axis=-1)
            norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)

            rms_norm = row * norm_factor * g

            if DUMP_INTERMEDIATE:
                tl.store(
                    out_intermediate_ptr + row_idx * n_cols + col_offsets,
                    rms_norm.to(out_intermediate_ptr.type.element_ty),
                    mask=mask,
                )

            if IS_SMOOTH:
                x_scale_ptrs = x_scale_ptr + col_offsets
                x_scale_ptrs = tl.multiple_of(x_scale_ptrs, (16,))
                x_scale = tl.load(
                    x_scale_ptrs, mask=mask, other=0.0, cache_modifier=".cg"
                )
                rms_norm *= x_scale

            row_max = tl.max(tl.abs(rms_norm), axis=-1)
            rms_norm = _per_token_quant(
                rms_norm,
                y_scale_ptr,
                row_max,
                row_idx,
                DTYPE_MAX,
                scale_ub_ptr=scale_ub_ptr,
                CLAMP_MAX=CLAMP_MAX,
                CLAMP_OUT=CLAMP_OUT,
            )

            output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
            output_ptrs = tl.multiple_of(output_ptrs, (16,))
            tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty), mask=mask)


@triton.jit
def _fused_add_rmsnorm_kernel(
    # Pointers to matrices
    input_ptr,
    output_ptr,
    res_in_ptr,
    res_out_ptr,
    g_ptr,
    rsigma_ptr,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `input_row_stride` is
    # how much to increase `input_ptr` by to get the element one row down.
    input_row_stride,
    output_row_stride,
    # Matrix dimensions
    n_rows,
    n_cols,
    # Epsilon to avoid division by zero
    epsilon,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
    USE_BLOCKED: tl.constexpr,
    NUM_PRGMS: tl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call
    rmsnorm2d_fwd_with_add function below.

    Performs an addition between two inputs and then applies Root Mean Square Layer Normalization over
    the addition result.

    Key parameters:
    - Input: The input tensor to be normalized with shape (n_rows, n_cols).
    - Output: The output tensor with shape (n_rows, n_cols).
    - Res_in: The tensor to be added to the Input tensor with shape (n_rows, n_cols).
    - Res_out: The tensor in which the addition result will be stored with shape (n_rows, n_cols).
    - G: The learnable weights tensor with shape (n_cols, ).
    """
    # Map the program id to the first row of input and output it should compute.
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    if USE_BLOCKED:
        # Persistent loop for rows
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=1):
            row_input_ptr = input_ptr + row_idx * input_row_stride
            row_output_ptr = output_ptr + row_idx * output_row_stride
            row_res_in_ptr = res_in_ptr + row_idx * input_row_stride
            row_res_out_ptr = res_out_ptr + row_idx * input_row_stride

            # Accumulate sum of squares
            n_cols_blks = tl.cdiv(n_cols, BLOCK_SIZE) - 1
            sum_squares = 0.0
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                input_ptrs = tl.multiple_of(input_ptrs, (16,))
                x = tl.load(input_ptrs)
                res_in_ptrs = row_res_in_ptr + cols
                res_in_ptrs = tl.multiple_of(res_in_ptrs, (16,))
                res_in = tl.load(res_in_ptrs)
                x += res_in
                # Stores residual_out
                res_out_ptrs = row_res_out_ptr + cols
                tl.store(res_out_ptrs, x.to(res_out_ptr.type.element_ty))

                x = x.to(tl.float32)
                sum_squares += tl.sum(x * x, axis=0)

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            input_ptrs = tl.multiple_of(input_ptrs, (16,))
            x = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
            res_in_ptrs = row_res_in_ptr + cols
            res_in_ptrs = tl.multiple_of(res_in_ptrs, (16,))
            res_in = tl.load(res_in_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
            x += res_in
            # Stores residual_out
            res_out_ptrs = row_res_out_ptr + cols
            tl.store(res_out_ptrs, x.to(res_out_ptr.type.element_ty), mask=mask)

            x = x.to(tl.float32)
            sum_squares += tl.sum(x * x, axis=0)

            # Compute normalization factor
            mean_square = sum_squares / n_cols
            norm_factor = tl.rsqrt(mean_square + epsilon)

            # Store rsigma (norm_factor)
            tl.store(rsigma_ptr + row_idx, norm_factor)

            # Normalize and write output
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                res_out_ptrs = row_res_out_ptr + cols
                res_out_ptrs = tl.multiple_of(res_out_ptrs, (16,))
                x = tl.load(res_out_ptrs).to(tl.float32)
                g_ptrs = g_ptr + cols
                g = tl.load(g_ptrs).to(tl.float32)
                rms_norm = x * norm_factor * g
                output_ptrs = row_output_ptr + cols
                tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty))

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            res_out_ptrs = row_res_out_ptr + cols
            x = tl.load(res_out_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(
                tl.float32
            )
            g_ptrs = g_ptr + cols
            g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
            rms_norm = x * norm_factor * g
            output_ptrs = row_output_ptr + cols
            tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty), mask=mask)

    else:
        mask = col_offsets < n_cols
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=2):
            input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
            input_ptrs = tl.multiple_of(input_ptrs, (16,))
            row = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
            res_in_ptrs = res_in_ptr + row_idx * input_row_stride + col_offsets
            res_in_ptrs = tl.multiple_of(res_in_ptrs, (16,))
            res_in = tl.load(res_in_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
            row += res_in
            # Stores residual_out
            res_out_ptrs = res_out_ptr + row_idx * input_row_stride + col_offsets
            res_out_ptrs = tl.multiple_of(res_out_ptrs, (16,))
            tl.store(res_out_ptrs, row.to(res_out_ptr.type.element_ty), mask=mask)
            row = row.to(tl.float32)

            g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            row_norm = row * row
            row_norm = tl.sum(row_norm, axis=-1)
            norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)

            # Store rsigma (norm_factor)
            tl.store(rsigma_ptr + row_idx, norm_factor)

            rms_norm = row * norm_factor * g

            output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
            output_ptrs = tl.multiple_of(output_ptrs, (16,))
            tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty), mask=mask)


@triton.jit
def _quant_fused_add_rmsnorm_kernel(
    # Pointers to matrices
    input_ptr,
    output_ptr,
    res_in_ptr,
    res_out_ptr,
    x_scale_ptr,
    y_scale_ptr,
    g_ptr,
    # Auxiliary tensor to store intermediate data
    aux_ptr,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `input_row_stride` is
    # how much to increase `input_ptr` by to get the element one row down.
    input_row_stride,
    output_row_stride,
    aux_row_stride,
    # Matrix dimensions
    n_rows,
    n_cols,
    # Epsilon to avoid division by zero
    epsilon,
    # Dtype max for quantization
    DTYPE_MAX: tl.constexpr,
    # Meta-parameters
    IS_SMOOTH: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE: tl.constexpr,
    USE_BLOCKED: tl.constexpr,
    NUM_PRGMS: tl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call
    rmsnorm2d_fwd_with_add_smoothquant or rmsnorm2d_fwd_with_add_dynamicquant functions below.

    Performs an addition between two inputs and then applies Root Mean Square Layer Normalization over
    the addition result followed by a quantization.

    Key parameters:
    - Input: The input tensor to be normalized with shape (n_rows, n_cols).
    - Output: The output tensor with shape (n_rows, n_cols).
    - Res_in: The tensor to be added to the Input tensor with shape (n_rows, n_cols).
    - Res_out: The tensor in which the addition result will be stored with shape (n_rows, n_cols).
    - X_scale: The tensor to be multiplied by the RMSNorm output if IS_SMOOTH is true, with shape (n_cols, ).
    - Y_scale: The tensor where the scale for each row will be stored with shape (n_rows, ).
    - G: The learnable weights tensor with shape (n_cols, ).
    """
    # Map the program id to the first row of input and output it should compute.
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    if USE_BLOCKED:
        # Persistent loop for rows
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=1):
            row_input_ptr = input_ptr + row_idx * input_row_stride
            row_output_ptr = output_ptr + row_idx * output_row_stride
            row_res_in_ptr = res_in_ptr + row_idx * input_row_stride
            row_res_out_ptr = res_out_ptr + row_idx * input_row_stride
            row_aux_ptr = aux_ptr + row_idx * aux_row_stride

            # Accumulate sum of squares
            n_cols_blks = tl.cdiv(n_cols, BLOCK_SIZE) - 1
            sum_squares = 0.0
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                input_ptrs = tl.multiple_of(input_ptrs, (16,))
                x = tl.load(input_ptrs)
                res_in_ptrs = row_res_in_ptr + cols
                res_in_ptrs = tl.multiple_of(res_in_ptrs, (16,))
                res_in = tl.load(res_in_ptrs)
                x += res_in
                # Stores residual_out
                res_out_ptrs = row_res_out_ptr + cols
                tl.store(res_out_ptrs, x.to(res_out_ptr.type.element_ty))

                x = x.to(tl.float32)
                sum_squares += tl.sum(x * x, axis=0)

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            input_ptrs = tl.multiple_of(input_ptrs, (16,))
            x = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
            res_in_ptrs = row_res_in_ptr + cols
            res_in_ptrs = tl.multiple_of(res_in_ptrs, (16,))
            res_in = tl.load(res_in_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
            x += res_in
            # Stores residual_out
            res_out_ptrs = row_res_out_ptr + cols
            tl.store(res_out_ptrs, x.to(res_out_ptr.type.element_ty), mask=mask)

            x = x.to(tl.float32)
            sum_squares += tl.sum(x * x, axis=0)

            # Compute normalization factor
            mean_square = sum_squares / n_cols
            norm_factor = tl.rsqrt(mean_square + epsilon)

            row_max = 0.0

            # Normalize and write output temporarily as fp32
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                res_out_ptrs = row_res_out_ptr + cols
                res_out_ptrs = tl.multiple_of(res_out_ptrs, (16,))
                x = tl.load(res_out_ptrs).to(tl.float32)
                g_ptrs = g_ptr + cols
                g = tl.load(g_ptrs).to(tl.float32)
                rms_norm = x * norm_factor * g

                if IS_SMOOTH:
                    x_scale_ptrs = x_scale_ptr + cols
                    x_scale_ptrs = tl.multiple_of(x_scale_ptrs, (16,))
                    x_scale = tl.load(x_scale_ptrs)
                    rms_norm *= x_scale

                blk_max = tl.max(tl.abs(rms_norm), axis=-1)
                row_max = max(row_max, blk_max)

                aux_ptrs = row_aux_ptr + cols
                tl.store(aux_ptrs, rms_norm)

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            res_out_ptrs = row_res_out_ptr + cols
            x = tl.load(res_out_ptrs, mask=mask, other=0.0, cache_modifier=".cg").to(
                tl.float32
            )
            g_ptrs = g_ptr + cols
            g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
            rms_norm = x * norm_factor * g

            if IS_SMOOTH:
                x_scale_ptrs = x_scale_ptr + cols
                x_scale = tl.load(
                    x_scale_ptrs, mask=mask, other=0.0, cache_modifier=".cg"
                )
                rms_norm *= x_scale

            blk_max = tl.max(tl.abs(rms_norm), axis=-1)
            row_max = max(row_max, blk_max)

            aux_ptrs = row_aux_ptr + cols
            tl.store(aux_ptrs, rms_norm, mask=mask)

            # Apply quantization and write output
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                aux_ptrs = row_aux_ptr + cols
                aux_ptrs = tl.multiple_of(aux_ptrs, (16,))
                aux = tl.load(aux_ptrs)

                output = _per_token_quant(
                    aux,
                    y_scale_ptr,
                    row_max,
                    row_idx,
                    DTYPE_MAX,
                )

                output_ptrs = row_output_ptr + cols
                tl.store(output_ptrs, output.to(output_ptr.dtype.element_ty))

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            aux_ptrs = row_aux_ptr + cols
            aux = tl.load(aux_ptrs, mask=mask, other=0.0, cache_modifier=".cg")

            output = _per_token_quant(
                aux,
                y_scale_ptr,
                row_max,
                row_idx,
                DTYPE_MAX,
            )

            output_ptrs = row_output_ptr + cols
            tl.store(output_ptrs, output.to(output_ptr.dtype.element_ty), mask=mask)

    else:
        mask = col_offsets < n_cols
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=2):
            input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
            input_ptrs = tl.multiple_of(input_ptrs, (16,))
            row = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
            res_in_ptrs = res_in_ptr + row_idx * input_row_stride + col_offsets
            res_in_ptrs = tl.multiple_of(res_in_ptrs, (16,))
            res_in = tl.load(res_in_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
            row += res_in
            # Stores residual_out
            res_out_ptrs = res_out_ptr + row_idx * input_row_stride + col_offsets
            res_out_ptrs = tl.multiple_of(res_out_ptrs, (16,))
            tl.store(res_out_ptrs, row.to(res_out_ptr.type.element_ty), mask=mask)
            row = row.to(tl.float32)

            g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
            row_norm = row * row
            row_norm = tl.sum(row_norm, axis=-1)
            norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)

            rms_norm = row * norm_factor * g

            if IS_SMOOTH:
                x_scale_ptrs = x_scale_ptr + col_offsets
                x_scale_ptrs = tl.multiple_of(x_scale_ptrs, (16,))
                x_scale = tl.load(
                    x_scale_ptrs, mask=mask, other=0.0, cache_modifier=".cg"
                )
                rms_norm *= x_scale

            row_max = tl.max(tl.abs(rms_norm), axis=-1)
            rms_norm = _per_token_quant(
                rms_norm,
                y_scale_ptr,
                row_max,
                row_idx,
                DTYPE_MAX,
            )

            output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
            output_ptrs = tl.multiple_of(output_ptrs, (16,))
            tl.store(output_ptrs, rms_norm.to(output_ptr.type.element_ty), mask=mask)


@triton.jit
def _rmsnorm_bwd_triton(
    grad_output_ptr,
    input_ptr,
    g_ptr,
    rsigma_ptr,
    dx_ptr,
    dg_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    USE_BLOCKED: tl.constexpr,
    NUM_PRGMS: tl.constexpr,
):
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)

    if USE_BLOCKED:
        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=1):
            row_input_ptr = input_ptr + row_idx * input_row_stride
            row_grad_output_ptr = grad_output_ptr + row_idx * output_row_stride
            row_dx_ptr = dx_ptr + row_idx * input_row_stride
            row_dg_ptr = dg_ptr + row_idx * input_row_stride

            # Compute gradients sum of all colums for each row
            n_cols_blks = tl.cdiv(n_cols, BLOCK_SIZE) - 1
            # older version of triton doesn't accept below init
            # comment out for now to make it compatible with triton 3.1
            # grad_sum: tl.float32 = 0.0
            grad_sum = 0.0
            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                grad_output_ptrs = row_grad_output_ptr + cols

                input_ptrs = tl.multiple_of(input_ptrs, (16,))
                grad_output_ptrs = tl.multiple_of(grad_output_ptrs, (16,))

                x = tl.load(input_ptrs).to(tl.float32)
                grad_output = tl.load(grad_output_ptrs).to(tl.float32)
                g_ptrs = g_ptr + cols
                g = tl.load(g_ptrs).to(tl.float32)
                grad_sum += tl.sum(grad_output * x * g, axis=0)

            # remainder for grad_sum:
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols
            input_ptrs = row_input_ptr + cols
            x = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
            grad_output_ptrs = row_grad_output_ptr + cols
            grad_output = tl.load(grad_output_ptrs, mask=mask, other=0.0).to(tl.float32)
            g_ptrs = g_ptr + cols
            g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
            grad_sum += tl.sum(grad_output * x * g, axis=0)

            # Load r_sigma
            norm_factor = tl.load(rsigma_ptr + row_idx).to(tl.float32)

            for blk_idx in tl.range(0, n_cols_blks, num_stages=2):
                cols = blk_idx * BLOCK_SIZE + col_offsets
                input_ptrs = row_input_ptr + cols
                grad_output_ptrs = row_grad_output_ptr + cols

                input_ptrs = tl.multiple_of(input_ptrs, (16,))
                grad_output_ptrs = tl.multiple_of(grad_output_ptrs, (16,))

                x = tl.load(input_ptrs).to(tl.float32)
                grad_output = tl.load(grad_output_ptrs).to(tl.float32)

                g_ptrs = g_ptr + cols
                g = tl.load(g_ptrs).to(tl.float32)
                grad_input = grad_output * norm_factor * g - (
                    norm_factor * norm_factor * norm_factor
                ) * x * (grad_sum / n_cols)

                dx_ptrs = row_dx_ptr + cols
                tl.store(dx_ptrs, grad_input.to(dx_ptr.type.element_ty))

                dg = grad_output * x * norm_factor
                dg_ptrs = row_dg_ptr + cols
                tl.store(dg_ptrs, dg.to(tl.float32))

            # Handle remainder
            cols = n_cols_blks * BLOCK_SIZE + col_offsets
            mask = cols < n_cols

            input_ptrs = row_input_ptr + cols
            x = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
            grad_output_ptrs = row_grad_output_ptr + cols
            grad_output = tl.load(grad_output_ptrs, mask=mask, other=0.0).to(tl.float32)
            g_ptrs = g_ptr + cols
            g = tl.load(g_ptrs, mask=mask, other=0.0).to(tl.float32)
            grad_input = grad_output * norm_factor * g - (
                norm_factor * norm_factor * norm_factor
            ) * x * (grad_sum / n_cols)

            dx_ptrs = row_dx_ptr + cols
            tl.store(dx_ptrs, grad_input.to(dx_ptr.type.element_ty), mask=mask)

            dg = grad_output * x * norm_factor
            dg_ptrs = row_dg_ptr + cols
            tl.store(dg_ptrs, dg.to(tl.float32), mask=mask)

    else:
        mask = col_offsets < n_cols
        dg_col_redux = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

        for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=2):
            input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
            grad_output_ptrs = (
                grad_output_ptr + row_idx * output_row_stride + col_offsets
            )
            dx_ptrs = dx_ptr + row_idx * input_row_stride + col_offsets

            input_ptrs = tl.multiple_of(input_ptrs, (16,))
            grad_output_ptrs = tl.multiple_of(grad_output_ptrs, (16,))
            dx_ptrs = tl.multiple_of(dx_ptrs, (16,))

            x = tl.load(input_ptrs, mask=mask, other=0.0).to(tl.float32)
            grad_output = tl.load(grad_output_ptrs, mask=mask, other=0.0).to(tl.float32)
            g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

            norm_factor = tl.load(rsigma_ptr + row_idx).to(tl.float32)
            grad_sum = tl.sum(grad_output * x * g, axis=0)

            grad_input = grad_output * norm_factor * g - (
                norm_factor * norm_factor * norm_factor
            ) * x * (grad_sum / n_cols)
            tl.store(dx_ptrs, grad_input.to(dx_ptr.type.element_ty), mask=mask)

            dg = grad_output * x * norm_factor
            dg_col_redux += dg.to(tl.float32)

        tl.store(
            dg_ptr + tl.program_id(0) * input_row_stride + col_offsets,
            dg_col_redux,
            mask=mask,
        )


@triton.jit
def _rmsnorm_bwd_dg_reduce_triton(
    dg_in_ptr,
    dg_out_ptr,
    dg_in_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # we want parallelism in N direction
    # if N is small, we will just use one CU,
    # otherwise, it can be split by N/BLOCK_SIZE
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(0, n_rows, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < n_rows) & (cols[None, :] < n_cols)
        offs = rows[:, None] * n_cols + cols[None, :]
        acc += tl.load(dg_in_ptr + offs, mask=mask, other=0.0, cache_modifier=".cg").to(
            tl.float32
        )

    sum_dg = tl.sum(acc, axis=0)
    tl.store(
        dg_out_ptr + cols, sum_dg.to(dg_out_ptr.type.element_ty), mask=cols < n_cols
    )
