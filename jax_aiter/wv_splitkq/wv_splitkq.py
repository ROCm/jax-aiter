# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
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
from ..ffi.registry import register_ffi_target

log = logging.getLogger("aiter.wv_splitkq")


def _ensure_lib_loaded():
    """Lazy load the library and register FFI target when first needed."""
    register_ffi_target("WvSplitKQJA", "ROCM")


@lru_cache(maxsize=None)
def _cached_jitted_call(out_shape, out_dtype, cu_count):
    """Create ffi_call + JIT once per signature."""
    call = jax.ffi.ffi_call(
        "WvSplitKQJA",
        jax.ShapeDtypeStruct(out_shape, out_dtype),
        vmap_method="broadcast_all",
    )

    def _invoke(A, B, scale_a, scale_b):
        return call(
            A,
            B,
            scale_a,
            scale_b,
            CuCount=cu_count,
        )

    # donate buffers to reduce copies; attrs are baked into the jitted graph.
    # return jax.jit(_invoke, donate_argnums=(0, 1, 2, 3))
    return jax.jit(_invoke)


def wv_splitkq_fp8(
    A: jnp.ndarray,
    B: jnp.ndarray,
    scale_a: jnp.ndarray,
    scale_b: jnp.ndarray,
    cu_count: np.int32 = 120,
    dtype=dtypes.bf16,
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
    # Ensure library is loaded before using
    _ensure_lib_loaded()

    # (Ruturaj4): This is done according to -
    # https://github.com/ROCm/aiter/blob/508c0b69bdd284a0c17eb8c77dd90a1a0b0563dd/op_tests/test_gemm_a8w8.py#L47
    out_shape = (B.shape[0], A.shape[0])
    out_dtype = dtype

    fn = _cached_jitted_call(out_shape, out_dtype, cu_count)
    y = fn(A, B, scale_a, scale_b)

    log.debug(f"WvSplitKQ: A{A.shape} @ B{B.shape} -> C{y.shape}, cu_count={cu_count}")
    return y
