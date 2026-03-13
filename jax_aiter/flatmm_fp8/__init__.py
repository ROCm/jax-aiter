# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""FP8 flat matmul via AITER ASM kernel (gfx942/MI300 only)."""

from .flatmm_fp8 import flatmm_fp8

__all__ = ["flatmm_fp8"]
