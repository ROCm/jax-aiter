# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
import os
from pathlib import Path


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
