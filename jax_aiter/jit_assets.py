# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Immutable JIT-asset identity embedded in the alpha2 wheel.

Update this file whenever a hard JIT input changes. CPU CI checks these values
against ``ci/jit_libs_manifest.py`` so a stale wheel binding cannot publish.
"""

AITER_SHA = "9da5928186becafe81dd2b63a9031dfdabcece52"
GPU_ARCHS = "gfx950"
ROCM_VERSION = "10.0.0"
ASSET_CONTRACT_VERSION = 2
JIT_RECIPE_HASH = (
    "sha256:fb05c8fa98b2a9462168fae743ca30a881302c72e674feb658f9060acded00b4"
)
CACHE_ID = "31693eb89913dbf9"
RELEASE_TAG = f"jit-libs-{CACHE_ID}"
