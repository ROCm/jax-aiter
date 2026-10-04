# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Layer 1: Single-kernel FFI wrappers.

Each function is one FFI call with no custom_vjp or custom_partitioning.
These are reusable building blocks for custom training recipes.

Usage::

    from jax_aiter.ops import gemm_fp4, cast_mxfp4, gemm_bf16, mha_fwd
"""

from .gemm_fp4 import gemm_fp4, cast_mxfp4, cast_mxfp4_dual
from .gemm_bf16 import gemm_bf16
from .mha import mha_fwd, mha_bwd, MhaFwdConfig, MhaBwdConfig
from .rmsnorm import rmsnorm_fwd
from .activation import silu_and_mul
from .buffers import uninitialized
from .moe_mxfp4 import (
    grouped_mxfp4_dw,
    grouped_mxfp4_dw_wholeloop,
    grouped_mxfp4_fwd_da,
    grouped_mxfp4_fwd_da_wholeloop,
    quantize_mxfp4_dim0,
    quantize_mxfp4_dim1,
)

__all__ = [
    "gemm_fp4",
    "cast_mxfp4",
    "cast_mxfp4_dual",
    "gemm_bf16",
    "mha_fwd",
    "mha_bwd",
    "MhaFwdConfig",
    "MhaBwdConfig",
    "rmsnorm_fwd",
    "silu_and_mul",
    "uninitialized",
    "quantize_mxfp4_dim0",
    "quantize_mxfp4_dim1",
    "grouped_mxfp4_fwd_da",
    "grouped_mxfp4_fwd_da_wholeloop",
    "grouped_mxfp4_dw",
    "grouped_mxfp4_dw_wholeloop",
]
