# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""FFI wrapper for WvSplitKQ_Bridge (fp8 x fp8 -> f16/bf16 with quantization)."""

from __future__ import annotations
import logging
import numpy as np
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import jax
import jax.numpy as jnp

from ..ja_compat import dtypes
from ..ffi.registry import get_lib_handle

log = logging.getLogger("aiter.wv_splitkq")

_LIB = get_lib_handle()
jax.ffi.register_ffi_target(
    "WvSplitKQ",
    jax.ffi.pycapsule(_LIB.WvSplitKQ),
    platform="ROCM",
)


@lru_cache(maxsize=None)
def _cached_jitted_call(out_shape, out_dtype, cu_count):
    """Create ffi_call + JIT once per signature."""
    call = jax.ffi.ffi_call(
        "WvSplitKQ",
        jax.ShapeDtypeStruct(out_shape, out_dtype),
        vmap_method="broadcast_all",
    )

    def _invoke(A, B, scale_a, scale_b):
        return call(
            A,
            B,
            scale_a.astype(jnp.float32),
            scale_b.astype(jnp.float32),
            CuCount=np.int32(cu_count),
        )

    # donate buffers to reduce copies; attrs are baked into the jitted graph.
    # return jax.jit(_invoke, donate_argnums=(0, 1, 2, 3))
    return jax.jit(_invoke)


def wv_splitkq_fp8(
    A: jnp.ndarray,
    B: jnp.ndarray,
    scale_a: jnp.ndarray,
    scale_b: jnp.ndarray,
    cu_count: int = 120,
    dtype=dtypes.bf16,
    check: bool = False,
) -> jnp.ndarray:
    """
    Perform quantized matrix multiplication using WvSplitKQ kernel.

    Args:
        A: Input tensor A [M, K] in FP8 format
        B: Input tensor B [K, N] in FP8 format
        scale_a: Scale tensor for A [M, 1] in F32 format
        scale_b: Scale tensor for B [1, N] in F32 format
        cu_count: Number of compute units to use (default: 120)
        dtype: Output data type (F16 or BF16, default: BF16)
        check: Whether to perform input validation (default: False)

    Returns:
        Output tensor C [M, N] in specified dtype
    """
    # if check:
    #     assert dtype in [dtypes.fp16, dtypes.bf16], f"{dtype=} not supported"
    #     assert A.dtype == dtypes.fp8, f"A must be FP8, got {A.dtype}"
    #     assert B.dtype == dtypes.fp8, f"B must be FP8, got {B.dtype}"
    #     assert scale_a.dtype == dtypes.fp32, f"scale_a must be F32, got {scale_a.dtype}"
    #     assert scale_b.dtype == dtypes.fp32, f"scale_b must be F32, got {scale_b.dtype}"

    #     # Validate shapes
    #     m, k_a = A.shape
    #     k_b, n = B.shape
    #     assert k_a == k_b, f"K dimensions must match: A.shape[1]={k_a}, B.shape[0]={k_b}"

    #     # Validate scale shapes
    #     assert scale_a.shape == (m, 1) or scale_a.shape == (m,), f"scale_a shape {scale_a.shape} invalid for A shape {A.shape}"
    #     assert scale_b.shape == (1, n) or scale_b.shape == (n,), f"scale_b shape {scale_b.shape} invalid for B shape {B.shape}"

    # (Ruturaj4): This is done according to -
    # https://github.com/ROCm/aiter/blob/508c0b69bdd284a0c17eb8c77dd90a1a0b0563dd/op_tests/test_gemm_a8w8.py#L47
    out_shape = (B.shape[0], A.shape[0])
    out_dtype = dtype

    fn = _cached_jitted_call(out_shape, out_dtype, cu_count)
    y = fn(A, B, scale_a, scale_b)

    log.debug(f"WvSplitKQ: A{A.shape} @ B{B.shape} -> C{y.shape}, cu_count={cu_count}")
    return y
