# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""NEOX RoPE via a self-contained AITER-style HIP kernel (custom_vjp)."""

from .rope import rope

__all__ = ["rope"]
