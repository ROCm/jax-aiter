# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""INT8 GEMM via AITER ASM kernels (gfx942/MI300 only).

Out[M,N] bf16 = dequant(A[M,K] i8, a_scale) @ dequant(B[N,K] i8, b_scale)^T + bias
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..ffi.registry import register_ffi_target


def _ensure_registered():
    register_ffi_target("GemmI8FwdJA", "ROCM")


def gemm_i8(
    a: jnp.ndarray,
    b: jnp.ndarray,
    a_scale: jnp.ndarray,
    b_scale: jnp.ndarray,
    bias: jnp.ndarray,
) -> jnp.ndarray:
    """INT8 GEMM (gfx942 only).

    Args:
        a: [M, K] int8
        b: [N, K] int8 (shuffled layout)
        a_scale: [M, 1] float32
        b_scale: [1, N] float32
        bias: [1, N] float32

    Returns:
        out: [M, N] bfloat16
    """
    _ensure_registered()
    M = a.shape[0]
    N = b.shape[0]
    call = jax.ffi.ffi_call(
        "GemmI8FwdJA",
        jax.ShapeDtypeStruct((M, N), jnp.bfloat16),
        vmap_method="broadcast_all",
    )
    return jax.jit(call)(a, b, a_scale, b_scale, bias)
