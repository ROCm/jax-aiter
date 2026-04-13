# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Raw BF16 GEMM via AITER ASM kernels (single FFI call).

No custom_vjp or custom_partitioning -- use jax_aiter.gemm.gemm for training.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..ffi.registry import register_ffi_target


def _ensure_registered():
    register_ffi_target("GemmFwdJA", "ROCM")


def _gemm_fwd_call(out_shape, sem_shape, dtype):
    call = jax.ffi.ffi_call(
        "GemmFwdJA",
        (
            jax.ShapeDtypeStruct(out_shape, dtype),
            jax.ShapeDtypeStruct(sem_shape, jnp.uint32),
        ),
        vmap_method="broadcast_all",
        input_layouts=[None, None],
        output_layouts=[None, None],
        has_side_effect=False,
    )

    def _invoke(a, b):
        out, _ = call(a, b)
        return out

    return jax.jit(_invoke)


def gemm_bf16(a, b):
    """BF16 ASM GEMM via AITER FFI. A[M,K] @ B[N,K]^T -> Out[M,N].

    Args:
        a: [M, K] bfloat16.
        b: [N, K] bfloat16 (not transposed; kernel computes A @ B^T).

    Returns:
        out: [M, N] bfloat16.
    """
    _ensure_registered()
    M, K = a.shape
    N = b.shape[0]
    fn = _gemm_fwd_call((M, N), (16, 64), a.dtype)
    return fn(a, b)
