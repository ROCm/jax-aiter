# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""
Public exports for the `jax_aiter.mha` sub-package.
"""

# Import comprehensive MHA implementation
from .mha import (
    # Low-level kernel functions
    mha_fwd,
    mha_bwd,
    fmha_v3_fwd,
    fmha_v3_bwd,
    mha_varlen_fwd,
    mha_varlen_bwd,
    fmha_v3_varlen_bwd,
    mha_batch_prefill,
    # High-level Flash Attention API
    flash_attn_func,
    flash_attn_varlen_func,
    mha_batch_prefill_func,
    # Internal functions for advanced usage
    _flash_attn_forward,
    _flash_attn_backward,
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward,
    _mha_batch_prefill,
    # JAX Autograd Classes
    FlashAttnFunc,
    FlashAttnVarlenFunc,
)

__all__ = [
    # High-level Flash Attention API (recommended)
    "flash_attn_func",
    "flash_attn_varlen_func",
    "mha_batch_prefill_func",
    # Low-level kernel functions
    "mha_fwd",
    "mha_bwd",
    "fmha_v3_fwd",
    "fmha_v3_bwd",
    "mha_varlen_fwd",
    "mha_varlen_bwd",
    "fmha_v3_varlen_bwd",
    "mha_batch_prefill",
    # JAX Autograd Classes
    "FlashAttnFunc",
    "FlashAttnVarlenFunc",
    # Internal functions (for advanced usage)
    "_flash_attn_forward",
    "_flash_attn_backward",
    "_flash_attn_varlen_forward",
    "_flash_attn_varlen_backward",
    "_mha_batch_prefill",
]
