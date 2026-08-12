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
    "sha256:40ed549c7294c9d4b874cb218e5d3d5e392594c0fa56147db9d29bbf21271a1f"
)
CACHE_ID = "9ac1d6deaf5d12b3"
RELEASE_TAG = f"jit-libs-{CACHE_ID}"
