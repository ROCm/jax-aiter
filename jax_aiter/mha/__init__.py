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

Default-wheel guard: the wheel includes the thin ``mha_*_ja.so`` FFI shims but
omits the multi-GB ``libmha_fwd.so``/``libmha_bwd.so`` JIT libraries. Users add
those with ``jax-aiter-fetch-mha``. If they are absent we raise a clear
``ModuleNotFoundError`` here instead of surfacing a cryptic FFI
"Module not loaded" RuntimeError deep inside the first ``flash_attn_func`` call.
"""

from ..ja_compat import config as _ja_config


def _mha_libs_present() -> bool:
    """True iff both the MHA JIT lib and its FFI shim are on disk.

    Works for the dev layout (``$JA_ROOT_DIR/build/...``) and the installed
    wheel layout, where downloaded JIT libraries live in a versioned user cache
    and thin FFI shims stay under ``jax_aiter/_lib``.
    """
    try:
        aiter_dir = _ja_config.get_aiter_lib_dir()
        ja_dir = _ja_config.get_jax_aiter_lib_dir()
        return bool(
            (aiter_dir / "libmha_fwd.so").is_file()
            and (aiter_dir / "libmha_bwd.so").is_file()
            and (ja_dir / "mha_fwd_ja.so").is_file()
            and (ja_dir / "mha_bwd_ja.so").is_file()
        )
    except Exception:
        return False


if not _mha_libs_present():
    raise ModuleNotFoundError(
        "jax_aiter is installed without the flash-attention libraries.\n"
        "\n"
        "They are ~2.6 GB, so the default wheel omits them. Download the\n"
        "prebuilt ones (a minute or two, versus a 2-3 hour source build):\n"
        "\n"
        "    jax-aiter-fetch-mha\n"
        "\n"
        "Everything else -- MXFP4/FP4 GEMM, BF16 GEMM, MXFP4 cast, RMSNorm,\n"
        "SiLU-and-Mul -- works without them."
    )

from .mha import (
    flash_attn_func,
    flash_attn_varlen,
    flash_attn_varlen_raw,
)
__all__ = [
    "flash_attn_func",
    "flash_attn_varlen",
    "flash_attn_varlen_raw",
]
