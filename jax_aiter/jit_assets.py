# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Immutable JIT-asset identity embedded in the alpha2 wheel.

Update this file whenever a hard JIT input changes. CPU CI checks these values
against ``ci/jit_libs_manifest.py`` so a stale wheel binding cannot publish.
"""

AITER_SHA = "df3e0b2f6f5e0007e90c6ad87bb5efef9499a053"
GPU_ARCHS = "gfx950"
ROCM_VERSION = "10.0.0"
ASSET_CONTRACT_VERSION = 2
JIT_RECIPE_HASH = (
    "sha256:67263ac848cd631b7d8e41e8210df1ec6d525e16efd7ba72cd25abe2f5eb2455"
)
CACHE_ID = "61adbc9b84bc0667"
RELEASE_TAG = f"jit-libs-{CACHE_ID}"
