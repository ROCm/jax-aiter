# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""FP4 GEMM via AITER ASM kernels (gfx942 + gfx950)."""

from .gemm_fp4 import gemm_fp4

__all__ = ["gemm_fp4"]
