# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""
Public exports for the `jax_aiter.mha` sub-package.

Uses unified AITER entry point (aiter::mha_fwd / aiter::mha_bwd) for both
batch and variable-length attention. CK vs ASM v3 dispatch handled internally
by AITER.

Public API:
    flash_attn_func: Batch flash attention with custom_vjp
    flash_attn_varlen: Variable-length flash attention with custom_vjp

Lite-variant guard: the lite wheel ships without the MHA kernels (the
multi-GB ``libmha_fwd.so``/``libmha_bwd.so`` JIT libs and their
``mha_*_ja.so`` FFI shims are dropped). If those libs are absent we raise a
clear ``ModuleNotFoundError`` here at import time instead of surfacing a
cryptic FFI "Module not loaded" RuntimeError deep inside the first
``flash_attn_func`` call.
"""

from ..ja_compat import config as _ja_config


def _mha_libs_present() -> bool:
    """True iff both the MHA JIT lib and its FFI shim are on disk.

    Works for the dev layout (``$JA_ROOT_DIR/build/...``) and the installed
    wheel layout (``jax_aiter/_lib/...``); both are resolved by
    ``ja_compat.config.get_lib_root()``.
    """
    try:
        lib_root = _ja_config.get_lib_root()
        aiter_lib = lib_root / "aiter_build" / "libmha_fwd.so"
        ja_shim = lib_root / "jax_aiter_build" / "mha_fwd_ja.so"
        return bool(aiter_lib.exists() and ja_shim.exists())
    except Exception:
        return False


if not _mha_libs_present():
    raise ModuleNotFoundError(
        "jax_aiter was built as the 'lite' variant without MHA kernels; "
        "install the 'full' variant to use flash attention."
    )

from .mha import (
    flash_attn_func,
    flash_attn_varlen,
)
__all__ = [
    "flash_attn_func",
    "flash_attn_varlen",
]
