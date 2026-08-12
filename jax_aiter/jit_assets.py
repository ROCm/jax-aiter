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
    "sha256:86d3c6cfd5d10bf11e525e24ddbdac900fbd7d4d6754b44c5e6d15e05f3b06e8"
)
CACHE_ID = "6c4ff74316215479"
RELEASE_TAG = f"jit-libs-{CACHE_ID}"
