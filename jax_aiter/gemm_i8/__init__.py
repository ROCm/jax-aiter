# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""INT8 GEMM via AITER ASM kernels (gfx942/MI300 only)."""

from .gemm_i8 import gemm_i8

__all__ = ["gemm_i8"]
