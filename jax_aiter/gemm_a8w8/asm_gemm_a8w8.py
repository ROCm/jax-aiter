# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""FFI wrapper for GemmA8W8_Bridge (int8 x int8 -> bf16)."""

from __future__ import annotations
import logging
import numpy as np
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import jax
import jax.numpy as jnp

from ..ja_compat import dtypes
from ..ja_compat.tuning import get_ASMGEMM_config
from ..ffi.registry import register_ffi_target

log = logging.getLogger("aiter.asm.a8w8")


def _ensure_lib_loaded():
    """Lazy load the library and register FFI target when first needed."""
    register_ffi_target("GemmA8W8JA", "ROCM")


@lru_cache(maxsize=None)
def _cached_jitted_call(out_shape, out_dtype, sub_m, sub_n, splitK):
    """Create ffi_call + JIT once per signature."""
    call = jax.ffi.ffi_call(
        "GemmA8W8JA",
        jax.ShapeDtypeStruct(out_shape, out_dtype),
        vmap_method="broadcast_all",
    )

    def _invoke(XQ, WQ, xs, ws, b):
        return call(
            XQ,
            WQ,
            xs.astype(jnp.float32),
            ws.astype(jnp.float32),
            b.astype(jnp.float32),
            sub_m=np.int32(sub_m),
            sub_n=np.int32(sub_n),
            pad_a=np.int32(0),
            pad_b=np.int32(0),
            pad_c=np.int32(0),
            splitK=np.int32(splitK),
        )

    # donate buffers to reduce copies; attrs are baked into the jitted graph.
    return jax.jit(_invoke, donate_argnums=(0, 1, 2, 3, 4))


def gemm_a8w8_ASM(
    XQ: jnp.ndarray,
    WQ: jnp.ndarray,
    x_scale: jnp.ndarray,
    w_scale: jnp.ndarray,
    bias: jnp.ndarray,
    dtype=dtypes.bf16,
    check: bool = False,
) -> jnp.ndarray | None:
    # Ensure library is loaded before using
    _ensure_lib_loaded()

    if check:
        assert dtype in [dtypes.bf16], f"{dtype=} not supported"
        assert (
            x_scale.dtype == dtypes.fp32 and w_scale.dtype == dtypes.fp32
        ), "scales must be fp32"
        assert bias is not None, "ASM requires bias (fp32)"

    m = int(XQ.shape[0])
    n = int(WQ.shape[0])
    k = int(XQ.shape[-1])
    cfg = get_ASMGEMM_config(m, n, k, bias is not None, dtype) or {}

    # Robustly read tuned params
    splitK = int(cfg.get("splitK", cfg.get("ks", 0)))
    sub_m = int(cfg.get("sub_m", cfg.get("BM", 128)))
    sub_n = int(cfg.get("sub_n", cfg.get("BN", 128)))

    out_shape = (m, n)
    out_dtype = dtype

    fn = _cached_jitted_call(out_shape, out_dtype, sub_m, sub_n, splitK)
    y = fn(XQ, WQ, x_scale, w_scale, bias)
    # y keeps dtype=bf16 as requested.
    return y
