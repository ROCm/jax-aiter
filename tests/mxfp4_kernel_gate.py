# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Module-level gate for tests that need the license-gated MXFP4 MoE kernels.

Without FlyDSL or the separately distributed kernel sources, gated modules
skip with the reason below. ``JA_REQUIRE_MXFP4_KERNELS=1`` turns that skip into
a collection failure, for jobs that provision the kernels and must not pass by
skipping.
"""

import os

import pytest


def require_mxfp4_kernels() -> None:
    missing = None
    try:
        import flydsl.compiler  # noqa: F401
    except ImportError as error:
        missing = f"FlyDSL is not importable: {error}"
    else:
        from jax_aiter.ops.moe_mxfp4 import mxfp4_moe_kernels_available

        if not mxfp4_moe_kernels_available():
            missing = (
                "MXFP4 MoE kernels are not installed; they are license-gated "
                "and distributed separately (see jax_aiter.ops.moe_mxfp4)"
            )
    if missing is None:
        return
    if os.environ.get("JA_REQUIRE_MXFP4_KERNELS") == "1":
        pytest.fail(missing, pytrace=False)
    pytest.skip(missing, allow_module_level=True)
