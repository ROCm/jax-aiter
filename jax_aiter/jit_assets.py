# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Immutable JIT-asset identity embedded in the alpha2 wheel.

Update this file whenever a hard JIT input changes. CPU CI checks these values
against ``ci/jit_libs_manifest.py`` so a stale wheel binding cannot publish.
"""

AITER_SHA = "31350226161346314b3d8882c8085bd31dce6a34"
GPU_ARCHS = "gfx950"
JIT_RECIPE_HASH = (
    "sha256:a282de667c4da15ca39646695505cc3539f004788e06d907a56c2a0dfd8c31ee"
)
CACHE_ID = "7f43f5a19ac40eb3"
RELEASE_TAG = f"jit-libs-{CACHE_ID}"
