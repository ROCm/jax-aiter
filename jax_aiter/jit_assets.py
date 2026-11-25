# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Immutable JIT-asset identity embedded in the alpha2 wheel.

Update this file whenever a hard JIT input changes. CPU CI checks these values
against ``ci/jit_libs_manifest.py`` so a stale wheel binding cannot publish.
"""

AITER_SHA = "ab0223ec4e96f06a54fd21a4340a44ad141c9cd8"
GPU_ARCHS = "gfx950"
ROCM_VERSION = "10.0.0"
ASSET_CONTRACT_VERSION = 2
JIT_RECIPE_HASH = (
    "sha256:5e3fb8ed973c02478de6c3db7bc79990578412970f196a12ba104b09b8110c8a"
)
CACHE_ID = "c814f3f654b90785"
RELEASE_TAG = f"jit-libs-{CACHE_ID}"
