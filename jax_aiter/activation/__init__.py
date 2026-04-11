# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""AITER fused activation ops for JAX."""

from .activation import silu_and_mul

__all__ = ["silu_and_mul"]
