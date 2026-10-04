# SPDX-License-Identifier: Apache-2.0
"""Packed / variable-length MHA via AITER Triton kernels.

Default-off. JAX owns the VJP. Does not call Torch autograd wrappers.
"""

from .varlen import flash_attn_varlen_triton

__all__ = ["flash_attn_varlen_triton"]
