# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""FP8 block-scale GEMM for MI350 via AITER ASM kernels."""

from .gemm_fp8_mi350 import gemm_fp8_mi350

__all__ = ["gemm_fp8_mi350"]
