# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
import os


def pytest_collection_modifyitems(config, items):
    total = int(os.environ.get("PYTEST_SHARD_TOTAL", "1"))
    idx = int(os.environ.get("PYTEST_SHARD_INDEX", "0"))
    if total <= 1:
        return
    selected = [item for i, item in enumerate(items) if i % total == idx]
    items[:] = selected
