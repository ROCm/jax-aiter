# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""MI350 FP8 block-scale GEMM via AITER ASM kernels.

Computes Out = dequant(A, a_scale) @ dequant(B, b_scale)^T
where A:[M,K] fp8, B:[N,K] fp8 with block scales.

Constraints:
  - A: [M, K] float8_e4m3fn, B: [N, K] float8_e4m3fn
  - x_scale: [K/128, M] float32, w_scale: [K/128, N/128] float32
  - N divisible by 256, K divisible by 128, M >= 16, K >= 512
  - Output: [M, N] bfloat16
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..ffi.registry import register_ffi_target


def _ensure_registered():
    register_ffi_target("GemmFp8Mi350FwdJA", "ROCM")


def _gemm_fp8_fwd_call(out_shape):
    call = jax.ffi.ffi_call(
        "GemmFp8Mi350FwdJA",
        jax.ShapeDtypeStruct(out_shape, jnp.bfloat16),
        vmap_method="broadcast_all",
    )
    return jax.jit(call)


def gemm_fp8_mi350(
    xq: jnp.ndarray,
    wq: jnp.ndarray,
    x_scale: jnp.ndarray,
    w_scale: jnp.ndarray,
) -> jnp.ndarray:
    """FP8 block-scale GEMM for MI350.

    Args:
        xq: [M, K] float8_e4m3fn activations
        wq: [N, K] float8_e4m3fn weights
        x_scale: [K/128, M] float32 activation block scales
        w_scale: [K/128, N/128] float32 weight block scales

    Returns:
        out: [M, N] bfloat16
    """
    _ensure_registered()
    M, K = xq.shape
    N = wq.shape[0]
    fn = _gemm_fp8_fwd_call((M, N))
    return fn(xq, wq, x_scale, w_scale)
