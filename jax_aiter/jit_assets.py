# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Immutable JIT-asset identity embedded in the alpha2 wheel.

Update this file whenever a hard JIT input changes. CPU CI checks these values
against ``ci/jit_libs_manifest.py`` so a stale wheel binding cannot publish.
"""

AITER_SHA = "31350226161346314b3d8882c8085bd31dce6a34"
GPU_ARCHS = "gfx950"
ROCM_VERSION = "7.14.0"
ASSET_CONTRACT_VERSION = 2
JIT_RECIPE_HASH = (
    "sha256:00f289bdb09952211f91026572cb6f05f5488d25902479bee6f1547d637718ba"
)
CACHE_ID = "3cb8f9991834db0a"
RELEASE_TAG = f"jit-libs-{CACHE_ID}"
