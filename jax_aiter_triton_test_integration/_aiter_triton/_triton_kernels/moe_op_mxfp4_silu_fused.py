# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl
from .activation import _silu_exp2
from ..utils._triton.pid_preprocessing import pid_grid, remap_xcd
from ..utils._triton.moe_common import _write_zeros_to_output


def get_scaled_dot_format_string(dtype: tl.dtype):
    mapping = {
        tl.float16: "fp16",
        tl.bfloat16: "bf16",
        tl.uint8: "e2m1",
        tl.float8e4nv: "e4m3",
        tl.float8e5: "e5m2",
    }
    return mapping[dtype]


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
    }
)
@triton.jit
def _fused_moe_kernel_mxfp4_silu(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    a_mx_scale_ptr,
    b_mx_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # Strides
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_amxm,
    stride_amxk,
    stride_bmxe,
    stride_bmxk,
    stride_bmxn,
    # Meta-parameters
    A_DTYPE_FORMAT: tl.constexpr,
    B_DTYPE_FORMAT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    SWIZZLE_MX_A: tl.constexpr,  # TODO add swizzle support
    SWIZZLE_MX_B: tl.constexpr,  # TODO add swizzle support
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    is_a_microscaled_format: tl.constexpr = a_mx_scale_ptr is not None
    is_b_microscaled_format: tl.constexpr = b_mx_scale_ptr is not None
    MX_PACK_DIVISOR: tl.constexpr = 32
    if is_a_microscaled_format:
        a_type: tl.constexpr = a_ptr.dtype.element_ty
        tl.static_assert(
            a_type == tl.uint8 or (a_type == tl.float8e4nv or a_type == tl.float8e5),
            "mx_weight_ptr must be 1 byte",
        )
        tl.static_assert(
            a_mx_scale_ptr.dtype.element_ty == tl.uint8, "a_mx_scale_ptr must be uint8"
        )
        tl.static_assert(
            BLOCK_SIZE_K % MX_PACK_DIVISOR == 0,
            "BLOCK_SIZE_K must be a multiple of MX_PACK_DIVISOR",
        )
    if is_b_microscaled_format:
        b_type: tl.constexpr = b_ptr.dtype.element_ty
        tl.static_assert(
            b_type == tl.uint8 or (b_type == tl.float8e4nv or b_type == tl.float8e5),
            "mx_weight_ptr must be 1 byte",
        )
        tl.static_assert(
            b_mx_scale_ptr.dtype.element_ty == tl.uint8, "b_mx_scale_ptr must be uint8"
        )
        tl.static_assert(
            BLOCK_SIZE_K % MX_PACK_DIVISOR == 0,
            "BLOCK_SIZE_K must be a multiple of MX_PACK_DIVISOR",
        )

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)

    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    NUM_XCDS: tl.constexpr = 8

    GRID_MN = num_pid_n * num_pid_m
    if pid < GRID_MN:
        pid = remap_xcd(pid, GRID_MN, NUM_XCDS)
    else:
        return  # rest of the tiles are dummy paddings
    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_expert == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        _write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    BLOCK_SIZE_HALF: tl.constexpr = BLOCK_SIZE_N // 2
    i = tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    # [0, 0, 1, 1, ..., BLOCK_SIZE_HALF - 1, BLOCK_SIZE_HALF - 1]
    i_floor = i // 2
    offs_half = ((pid_n * (BLOCK_SIZE_N // 2) + i_floor) % (N // 2)).to(tl.int64)
    # (i % 2): [0, 1, 0, 1,...] (alternating)
    # (i % 2) * (N // 2) : [0, (N // 2), 0, (N // 2),...]
    # So offs_bn now takes element from the first BLOCK_SIZE_HALF half and the second BLOCK_SIZE_HALF half in an alternating way (This allows us to do reshape without permute)
    offs_b_n = ((offs_half + (i % 2) * (N // 2)) % N).to(tl.int64)

    # Load a_scale, b_scale
    a_scale = tl.load(a_scale_ptr)
    b_scale = tl.load(b_scale_ptr + off_expert)
    # Set offsets of B on dim N
    offs_b_n = tl.max_contiguous(
        tl.multiple_of(offs_b_n % N, BLOCK_SIZE_N), BLOCK_SIZE_N
    )
    # Load a_mx_scale
    if is_a_microscaled_format:
        # We have pack 2 fp4 values in a byte
        A_PACK_DIVISOR: tl.constexpr = 2 if a_ptr.dtype.element_ty == tl.uint8 else 1
        PACKED_BLOCK_K_A: tl.constexpr = BLOCK_SIZE_K // A_PACK_DIVISOR  # 64
        MX_SCALE_BLOCK_K_A: tl.constexpr = BLOCK_SIZE_K // MX_PACK_DIVISOR  # 4

        if SWIZZLE_MX_A:
            tl.static_assert(BLOCK_SIZE_M % 128 == 0)
            tl.static_assert(MX_SCALE_BLOCK_K_A % 4 == 0)
            PACKED_MX_BLOCK_A: tl.constexpr = (MX_SCALE_BLOCK_K_A // 4) * 32 * 4 * 4
            offs_inner = tl.arange(0, PACKED_MX_BLOCK_A)
            offs_scale_m = (
                pid_m * (BLOCK_SIZE_M // 128) + tl.arange(0, BLOCK_SIZE_M // 128)
            ) % N
            offs_scale_m = tl.max_contiguous(
                tl.multiple_of(offs_scale_m, BLOCK_SIZE_M // 128), BLOCK_SIZE_M // 128
            )

            a_mx_scale_ptrs = (
                a_mx_scale_ptr
                + offs_scale_m.to(tl.int64)[:, None] * stride_amxm
                + offs_inner[None, :]
            )
        else:
            offs_scale_ak = tl.arange(0, MX_SCALE_BLOCK_K_A)
            offs_scale_m = offs_token
            # K dimension must be the last dimension for the scales
            a_mx_scale_ptrs = (
                a_mx_scale_ptr
                + offs_scale_ak.to(tl.int64)[None, :] * stride_amxk
                + offs_scale_m.to(tl.int64)[:, None] // top_k * stride_amxm
            )
    else:
        a_mx_scale_ptrs = None
        A_PACK_DIVISOR: tl.constexpr = 1
        MX_SCALE_BLOCK_K_A: tl.constexpr = 1
        PACKED_BLOCK_K_A: tl.constexpr = BLOCK_SIZE_K
    # Load b_mx_scale
    if is_b_microscaled_format:
        # We have pack 2 fp4 values in a byte
        B_PACK_DIVISOR: tl.constexpr = 2 if b_ptr.dtype.element_ty == tl.uint8 else 1
        PACKED_BLOCK_K_B: tl.constexpr = BLOCK_SIZE_K // B_PACK_DIVISOR  # 64
        MX_SCALE_BLOCK_K_B: tl.constexpr = BLOCK_SIZE_K // MX_PACK_DIVISOR  # 4

        b_mx_scale_ptr += off_expert * stride_bmxe

        if SWIZZLE_MX_B:
            tl.static_assert(BLOCK_SIZE_N % 128 == 0)
            tl.static_assert(MX_SCALE_BLOCK_K_B % 4 == 0)
            PACKED_MX_BLOCK_B: tl.constexpr = (MX_SCALE_BLOCK_K_B // 4) * 32 * 4 * 4
            offs_inner = tl.arange(0, PACKED_MX_BLOCK_B)
            offs_scale_n = (
                pid_n * (BLOCK_SIZE_N // 128) + tl.arange(0, BLOCK_SIZE_N // 128)
            ) % N
            offs_scale_n = tl.max_contiguous(
                tl.multiple_of(offs_scale_n, BLOCK_SIZE_N // 128), BLOCK_SIZE_N // 128
            )

            b_mx_scale_ptrs = (
                b_mx_scale_ptr
                # + offs_scale_n.to(tl.int64)[:, None] * stride_bmxn
                + offs_scale_n.to(tl.int64)[:, None]
                * PACKED_MX_BLOCK_B
                * (K // MX_SCALE_BLOCK_K_B // (MX_PACK_DIVISOR // B_PACK_DIVISOR))
                + offs_inner[None, :]
            )
        else:
            offs_scale_bk = tl.arange(0, MX_SCALE_BLOCK_K_B)
            offs_scale_n = offs_b_n
            # K dimension must be the last dimension for the scales
            b_mx_scale_ptrs = (
                b_mx_scale_ptr
                + offs_scale_bk.to(tl.int64)[None, :] * stride_bmxk
                + offs_scale_n.to(tl.int64)[:, None] * stride_bmxn
            )
    else:
        b_mx_scale_ptrs = None
        B_PACK_DIVISOR: tl.constexpr = 1
        MX_SCALE_BLOCK_K_B: tl.constexpr = 1
        PACKED_BLOCK_K_B: tl.constexpr = BLOCK_SIZE_K

    offs_a_k = tl.arange(0, PACKED_BLOCK_K_A)
    offs_b_k = tl.arange(0, PACKED_BLOCK_K_B)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_a_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + off_expert * stride_be
        + (offs_b_k[:, None] * stride_bk + offs_b_n[None, :] * stride_bn)
    )

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, PACKED_BLOCK_K_A)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        if EVEN_K:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None],
                other=0.0,
            )
            b = tl.load(b_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None]
                & (offs_a_k[None, :] < (K - k * PACKED_BLOCK_K_A)),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=offs_b_k[:, None] < (K - k * PACKED_BLOCK_K_B),
                other=0.0,
            )
        # We accumulate along the K dimension.
        if is_a_microscaled_format or is_b_microscaled_format:
            if is_a_microscaled_format:
                # if SWIZZLE_MX_A:
                #    a_mx_scales = _unswizzle_mx_block(tl.load(a_mx_scale_ptrs))
                # else:
                mask_ak_scale = offs_scale_ak < (K - k * PACKED_BLOCK_K_A) // (
                    MX_PACK_DIVISOR // A_PACK_DIVISOR
                )
                a_mx_scales = tl.load(
                    a_mx_scale_ptrs, mask=mask_ak_scale[None, :], other=0.0
                )
            else:
                a_mx_scales = None
            # if SWIZZLE_MX_B:
            #    b_mx_scales = _unswizzle_mx_block(tl.load(b_mx_scale_ptrs))
            # else:
            mask_bk_scale = offs_scale_bk < (K - k * PACKED_BLOCK_K_B) // (
                MX_PACK_DIVISOR // B_PACK_DIVISOR
            )
            b_mx_scales = tl.load(
                b_mx_scale_ptrs, mask=mask_bk_scale[None, :], other=0.0
            )

            accumulator = tl.dot_scaled(
                a,
                a_mx_scales,
                A_DTYPE_FORMAT,
                b,
                b_mx_scales,
                B_DTYPE_FORMAT,
                acc=accumulator,
                fast_math=True,
            )

            if is_a_microscaled_format:
                if SWIZZLE_MX_A:
                    a_mx_scale_ptrs += MX_SCALE_BLOCK_K_A // 4 * stride_amxk
                else:
                    a_mx_scale_ptrs += MX_SCALE_BLOCK_K_A * stride_amxk
            if SWIZZLE_MX_B:
                b_mx_scale_ptrs += MX_SCALE_BLOCK_K_B // 4 * 512
            else:
                b_mx_scale_ptrs += MX_SCALE_BLOCK_K_B * stride_bmxk
        # Advance the ptrs to the next K block.
        a_ptrs += PACKED_BLOCK_K_A * stride_ak
        b_ptrs += PACKED_BLOCK_K_B * stride_bk

    # Multiply with the scalar weight
    accumulator *= a_scale * b_scale
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]
    accumulator = accumulator.to(compute_type)

    silu_acc, mul_acc = (
        accumulator.to(tl.float32).reshape(BLOCK_SIZE_M, BLOCK_SIZE_HALF, 2).split()
    )

    silu_acc = _silu_exp2(silu_acc)
    accumulator = (silu_acc * mul_acc).to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_HALF + tl.arange(0, BLOCK_SIZE_HALF)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N // 2)
    tl.store(c_ptrs, accumulator, mask=c_mask)
