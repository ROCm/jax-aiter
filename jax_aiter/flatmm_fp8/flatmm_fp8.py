# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""FP8 flat matmul via AITER ASM kernel (gfx942/MI300 only).

Single kernel: flatmm_uk_gfx9_f16f8_128x256x128.
Output is fp16. N % 256 == 0, K % 128 == 0.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..ffi.registry import register_ffi_target


def _ensure_registered():
    register_ffi_target("FlatmmFp8FwdJA", "ROCM")


def flatmm_fp8(
    xq: jnp.ndarray,
    wq: jnp.ndarray,
    x_scale: jnp.ndarray,
    w_scale: jnp.ndarray,
) -> jnp.ndarray:
    """FP8 flat matmul (gfx942 only).

    Args:
        xq: [M, K] float8_e4m3fn
        wq: [N, K] float8_e4m3fn
        x_scale: [K/128, M] float32
        w_scale: [K/128, N/128] float32

    Returns:
        out: [M, N] float16
    """
    _ensure_registered()
    M, K = xq.shape
    N = wq.shape[0]
    call = jax.ffi.ffi_call(
        "FlatmmFp8FwdJA",
        jax.ShapeDtypeStruct((M, N), jnp.float16),
        vmap_method="broadcast_all",
    )
    return jax.jit(call)(xq, wq, x_scale, w_scale)
