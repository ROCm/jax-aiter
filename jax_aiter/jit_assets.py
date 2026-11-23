# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Immutable JIT-asset identity embedded in the alpha2 wheel.

Update this file whenever a hard JIT input changes. CPU CI checks these values
against ``ci/jit_libs_manifest.py`` so a stale wheel binding cannot publish.
"""

AITER_SHA = "2eb9ef3657777eebc0da3434074f4a67e1151c54"
GPU_ARCHS = "gfx950"
ROCM_VERSION = "10.0.0"
ASSET_CONTRACT_VERSION = 2
JIT_RECIPE_HASH = (
    "sha256:67263ac848cd631b7d8e41e8210df1ec6d525e16efd7ba72cd25abe2f5eb2455"
)
CACHE_ID = "68726a4a5dfb786e"
RELEASE_TAG = f"jit-libs-{CACHE_ID}"
