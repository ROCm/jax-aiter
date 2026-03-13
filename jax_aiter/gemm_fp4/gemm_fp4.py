# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""FP4 GEMM via AITER ASM kernels (gfx942 + gfx950).

A:[M,K/2] packed fp4x2, B:[N,K/2] packed fp4x2, with e8m0 block scales.
Output: [M,N] bf16.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..ffi.registry import register_ffi_target


def _ensure_registered():
    register_ffi_target("GemmFp4FwdJA", "ROCM")


def gemm_fp4(
    a: jnp.ndarray,
    b: jnp.ndarray,
    a_scale: jnp.ndarray,
    b_scale: jnp.ndarray,
) -> jnp.ndarray:
    """FP4 GEMM with block scales.

    Args:
        a: [M, K/2] uint8 (packed fp4x2, two FP4 values per byte)
        b: [N, K/2] uint8 (packed fp4x2)
        a_scale: [M, K/32] uint8 (e8m0 block scales)
        b_scale: [N, K/32] uint8 (e8m0 block scales)

    Returns:
        out: [M, N] bfloat16
    """
    _ensure_registered()
    M = a.shape[0]
    N = b.shape[0]
    call = jax.ffi.ffi_call(
        "GemmFp4FwdJA",
        jax.ShapeDtypeStruct((M, N), jnp.bfloat16),
        vmap_method="broadcast_all",
    )
    return jax.jit(call)(a, b, a_scale, b_scale)
