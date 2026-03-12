# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
# jax_aiter: JAX bindings for AITER kernels via XLA FFI.

import importlib as _importlib

from .ja_compat.config import set_aiter_asm_dir

set_aiter_asm_dir()

__all__ = ["mha"]


def __getattr__(name):
    """Lazy loading of submodules."""
    if name in __all__:
        return _importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
