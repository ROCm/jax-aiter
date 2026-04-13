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
]
