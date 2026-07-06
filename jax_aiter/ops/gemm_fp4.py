# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Raw FP4 GEMM and MXFP4 quantization FFI wrappers.

Single-kernel ops with no custom_vjp or custom_partitioning.
All parameters are explicit -- no environment variable reads.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..ffi.registry import register_ffi_target


def _ensure_gemm_registered():
    register_ffi_target("GemmFp4FwdJA", "ROCM")


def _ensure_cast_registered():
    register_ffi_target("CastMxfp4JA", "ROCM")


def _ensure_dual_registered():
    register_ffi_target("CastMxfp4DualJA", "ROCM")


def gemm_fp4(a_packed, b_packed, a_scale, b_scale):
    """FP4 ASM GEMM via AITER FFI. Inputs must already be quantized and shuffled.

    Args:
        a_packed: [M, K/2] uint8 — packed MXFP4 A operand.
        b_packed: [N, K/2] uint8 — packed + B-preshuffle shuffled MXFP4 B operand.
        a_scale:  [M_pad, scale_n_pad] uint8 — shuffled E8M0 scales for A.
        b_scale:  [N_pad, scale_n_pad] uint8 — shuffled E8M0 scales for B.

    Returns:
        out: [M, N] bfloat16.
    """
    _ensure_gemm_registered()
    M = a_packed.shape[0]
    N = b_packed.shape[0]
    call = jax.ffi.ffi_call(
        "GemmFp4FwdJA",
        jax.ShapeDtypeStruct((M, N), jnp.bfloat16),
        vmap_method="broadcast_all",
        has_side_effect=False,
    )
    return jax.jit(call)(a_packed, b_packed, a_scale, b_scale)


def cast_mxfp4(x, *, shuffle_fp4, shuffle_scales=True, use_hadamard=False,
               use_sr=False, scale_margin=0, scale_mode=0, use_2d_scale=False):
    """Fused BF16 -> MXFP4 quantization + shuffle via HIP kernel (single FFI call).

    Args:
        x: [M, K] bfloat16 input.
        shuffle_fp4: Whether to apply B-preshuffle to FP4 data.
            True for weight operands, False for activation operands.
        shuffle_scales: Whether to shuffle E8M0 scales (always True for AITER ASM).
        use_hadamard: Apply Hadamard transform before quantization.
        use_sr: Use stochastic rounding (unbiased) instead of RNE for the
            FP32->FP4 convert. Default False => byte-identical RNE path.
        scale_margin: Extra E8M0 headroom (int). The per-32-block scale is
            ``2^(exp - 2 - scale_margin)``; ``>0`` shrinks the scale so small
            entries survive the FP4 cast (fixes under-flush) at the cost of
            clipping the largest entries. Default ``0`` => byte-identical to the
            legacy ``exp-2`` cast.

    Returns:
        (fp4_packed, scales):
            fp4_packed: [M, K/2] uint8 — packed MXFP4 data.
            scales: [M_pad, scale_n_pad] uint8 — E8M0 block scales.
    """
    _ensure_cast_registered()
    M, K = x.shape
    scale_n = (K + 31) // 32
    m_pad = ((M + 255) // 256) * 256
    scale_n_pad = ((scale_n + 7) // 8) * 8

    out_shapes = (
        jax.ShapeDtypeStruct((M, K // 2), jnp.uint8),
        jax.ShapeDtypeStruct((m_pad, scale_n_pad), jnp.uint8),
    )
    call = jax.ffi.ffi_call(
        "CastMxfp4JA", out_shapes,
        vmap_method="broadcast_all",
        has_side_effect=False,
    )
    return call(x, shuffle_fp4=shuffle_fp4,
                shuffle_scales=shuffle_scales, use_hadamard=use_hadamard,
                use_sr=use_sr, scale_margin=np.int32(scale_margin),
                scale_mode=np.int32(scale_mode), use_2d_scale=use_2d_scale)


def cast_mxfp4_dual(x, *, shuffle_fp4, shuffle_colwise_fp4=True,
                     shuffle_scales=True, use_hadamard=False, use_sr=False,
                     scale_margin=0, use_hadamard_col=None, use_sr_col=None,
                     scale_mode=0, use_2d_scale=False):
    """Fused BF16 -> MXFP4 with BOTH rowwise and columnwise output in one kernel launch.

    Returns rowwise (for forward GEMM) + columnwise (for dA/dB backward GEMM).

    Args:
        x: [M, K] bfloat16 input.
        shuffle_fp4: Controls rowwise B-preshuffle (True for weights, False for activations).
        shuffle_colwise_fp4: Controls columnwise B-preshuffle.
            True  -> colwise output suitable as GEMM B operand (dA backward).
            False -> colwise output is linear layout, equivalent to rowwise of x^T
                     (suitable as GEMM A operand for dB backward).
        shuffle_scales: Whether to shuffle E8M0 scales.
        use_hadamard: Apply Hadamard to the ROWWISE output before quantization.
        use_sr: Stochastic rounding (unbiased) for the ROWWISE output. Default
            False => byte-identical RNE path.
        scale_margin: Extra E8M0 headroom (int). The per-32-block scale is
            ``2^(exp - 2 - scale_margin)``; ``>0`` shrinks the scale so small
            entries survive the FP4 cast (fixes under-flush). Default ``0`` =>
            byte-identical to the legacy ``exp-2`` cast.
        use_hadamard_col: Apply Hadamard to the COLWISE output, independent of
            the rowwise flag. ``None`` (default) => same as ``use_hadamard`` so
            existing callers are byte-identical. Setting it differently lets ONE
            dual launch emit asymmetric row/col Hadamard (no frontend split cast).
        use_sr_col: Stochastic rounding for the COLWISE output, independent of
            ``use_sr``. ``None`` (default) => same as ``use_sr``.

    Returns:
        (row_fp4, row_scale, col_fp4, col_scale):
            row_fp4:   [M, K/2] uint8.
            row_scale: [M_pad, rscale_n_pad] uint8.
            col_fp4:   [K, M/2] uint8.
            col_scale: [K_pad, cscale_n_pad] uint8.
    """
    _ensure_dual_registered()
    if use_hadamard_col is None:
        use_hadamard_col = use_hadamard
    if use_sr_col is None:
        use_sr_col = use_sr
    M, K = x.shape

    rscale_n = (K + 31) // 32
    r_m_pad = ((M + 255) // 256) * 256
    rscale_n_pad = ((rscale_n + 7) // 8) * 8

    cscale_n = (M + 31) // 32
    c_k_pad = ((K + 255) // 256) * 256
    cscale_n_pad = ((cscale_n + 7) // 8) * 8

    out_shapes = (
        jax.ShapeDtypeStruct((M, K // 2), jnp.uint8),
        jax.ShapeDtypeStruct((r_m_pad, rscale_n_pad), jnp.uint8),
        jax.ShapeDtypeStruct((K, M // 2), jnp.uint8),
        jax.ShapeDtypeStruct((c_k_pad, cscale_n_pad), jnp.uint8),
    )
    call = jax.ffi.ffi_call(
        "CastMxfp4DualJA", out_shapes,
        vmap_method="broadcast_all",
        has_side_effect=False,
    )
    return call(x, shuffle_fp4=shuffle_fp4,
                shuffle_colwise_fp4=shuffle_colwise_fp4,
                use_hadamard=use_hadamard, use_hadamard_col=use_hadamard_col,
                use_sr=use_sr, use_sr_col=use_sr_col,
                scale_margin=np.int32(scale_margin),
                scale_mode=np.int32(scale_mode), use_2d_scale=use_2d_scale)
