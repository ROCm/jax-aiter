# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""
Public exports for the `jax_aiter.mha` sub-package.

This module provides Multi-Head Attention (MHA) implementations with automatic
kernel dispatch between standard MHA and optimized FMHA v3 kernels.

Public API:
    flash_attn_func: High-level Flash Attention function (recommended)

Low-level kernels:
    mha_fwd, mha_bwd: Standard MHA kernels
    fmha_v3_fwd, fmha_v3_bwd: Optimized FMHA v3 kernels

Internal functions:
    _flash_attn_forward, _flash_attn_backward: Kernel dispatch logic

Utilities:
    debug_kernel_params: Debug function for kernel parameters
    _can_impl_fmha_v3_fwd, _can_impl_fmha_v3_bwd: Kernel capability checking
"""

# Import comprehensive MHA implementation.
from .mha import (
    # High-level Flash Attention API.
    flash_attn_func,
    # Low-level kernel functions.
    mha_fwd,
    mha_bwd,
    fmha_v3_fwd,
    fmha_v3_bwd,
    # Internal functions.
    _flash_attn_forward,
    _flash_attn_backward,
    debug_kernel_params,
    _can_impl_fmha_v3_fwd,
    _can_impl_fmha_v3_bwd,
    _normalize_window_size,
    _create_generator_tensor,
)

__all__ = [
    # High-level Flash Attention API (recommended for most users).
    "flash_attn_func",
    # Low-level kernel functions.
    "mha_fwd",
    "mha_bwd",
    "fmha_v3_fwd",
    "fmha_v3_bwd",
    # Internal functions.
    "_flash_attn_forward",
    "_flash_attn_backward",
    "debug_kernel_params",
    "_can_impl_fmha_v3_fwd",
    "_can_impl_fmha_v3_bwd",
    "_normalize_window_size",
    "_create_generator_tensor",
]
