# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Immutable JIT-asset identity embedded in the alpha2 wheel.

Update this file whenever a hard JIT input changes. CPU CI checks these values
against ``ci/jit_libs_manifest.py`` so a stale wheel binding cannot publish.
"""

AITER_SHA = "a3158e4800b6080e3adb25eb58bc48686bc1859c"
GPU_ARCHS = "gfx950"
ROCM_VERSION = "10.0.0"
ASSET_CONTRACT_VERSION = 2
JIT_RECIPE_HASH = (
    "sha256:664ada9d12e18b61beaa923cf82888b31aff110683f4a9abd6dc4970a63a4c26"
)
CACHE_ID = "9c9ec53233a801b4"
RELEASE_TAG = f"jit-libs-{CACHE_ID}"
