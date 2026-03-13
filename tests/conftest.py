# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
import os
from pathlib import Path

import pytest


def get_gpu_arch():
    """Detect GPU architecture string (e.g., 'gfx942', 'gfx950')."""
    try:
        import subprocess
        result = subprocess.run(
            ["rocminfo"], capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.split("\n"):
            if "gfx9" in line and "Name:" in line:
                return line.split(":")[-1].strip()
    except Exception:
        pass
    return os.environ.get("GPU_ARCHS", "gfx950").split(";")[0]


_gpu_arch = None

def gpu_arch():
    global _gpu_arch
    if _gpu_arch is None:
        _gpu_arch = get_gpu_arch()
    return _gpu_arch


def _is_gfx942():
    return gpu_arch() == "gfx942"

def _is_gfx950():
    return gpu_arch() == "gfx950"

requires_gfx942 = pytest.mark.skipif(
    not _is_gfx942(),
    reason="Requires gfx942 (MI300) GPU"
)

requires_gfx950 = pytest.mark.skipif(
    not _is_gfx950(),
    reason="Requires gfx950 (MI350) GPU"
)


def pytest_configure(config):
    root = str(Path(__file__).resolve().parent.parent)
    os.environ.setdefault("JA_ROOT_DIR", root)
    os.environ.setdefault("AITER_SYMBOL_VISIBLE", "1")
    os.environ.setdefault("GPU_ARCHS", "gfx950")
    os.environ.setdefault("AITER_ASM_DIR", os.path.join(root, "third_party", "aiter", "hsa/"))
    os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "")
    if worker_id:
        gpu_idx = int(worker_id.replace("gw", ""))
        visible = os.environ.get("HIP_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
        os.environ["HIP_VISIBLE_DEVICES"] = visible[gpu_idx % len(visible)]


def pytest_collection_modifyitems(config, items):
    total = int(os.environ.get("PYTEST_SHARD_TOTAL", "1"))
    idx = int(os.environ.get("PYTEST_SHARD_INDEX", "0"))
    if total > 1:
        items[:] = [item for i, item in enumerate(items) if i % total == idx]

    max_tests = int(os.environ.get("PYTEST_MAX_TESTS", "0"))
    if max_tests > 0:
        items[:] = items[:max_tests]
