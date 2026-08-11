# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Immutable JIT-asset identity embedded in the alpha2 wheel.

Update this file whenever a hard JIT input changes. CPU CI checks these values
against ``ci/jit_libs_manifest.py`` so a stale wheel binding cannot publish.
"""

AITER_SHA = "31350226161346314b3d8882c8085bd31dce6a34"
GPU_ARCHS = "gfx950"
ROCM_VERSION = "7.14.0"
JIT_RECIPE_HASH = (
    "sha256:10e1fb419bf9b51d1805e111c5b11bcaed3498e1070c412f064f78fe7d25eeae"
)
CACHE_ID = "7f0998ac2cbef534"
RELEASE_TAG = f"jit-libs-{CACHE_ID}"
