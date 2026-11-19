# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Persistent compilation cache key salt for jax_aiter settings.

On a multi-device mesh, ``custom_partitioning`` lowers to an opaque
``CustomSPMDPartitioning`` call. JAX removes its callback id from the
persistent cache key, and the FFI attributes chosen by our partition callbacks
appear only when XLA partitions the module, after the key is computed. Neither
the trace-time ``JA_*`` / ``AITER_*`` switches nor the package sources reach
the key, so a cache shared between runs serves an executable built under other
settings. ``install`` hashes both into JAX's ``cache_key.custom_hook``.
"""

import hashlib
import logging
import os
from pathlib import Path

logger = logging.getLogger("JAX_AITER")

_ENV_PREFIXES = ("JA_", "AITER_")
# Locations and download knobs; their contents are covered by the source hash
# or do not affect a compiled program.
_ENV_IGNORED = frozenset({
    "JA_ROOT_DIR", "JA_PY_OVERLAY", "JA_CAPTURE_DIR",
    "JA_FETCH_TIMEOUT_S", "JA_FETCH_RETRIES", "AITER_ASM_DIR",
})

_source_digest = None


def _package_source_digest() -> str:
    global _source_digest
    if _source_digest is None:
        root = Path(__file__).resolve().parent.parent
        h = hashlib.sha256()
        for path in sorted(root.rglob("*.py")):
            h.update(str(path.relative_to(root)).encode())
            h.update(b"\0")
            h.update(path.read_bytes())
        _source_digest = h.hexdigest()
    return _source_digest


def cache_key_salt() -> str:
    env = sorted(
        (k, v) for k, v in os.environ.items()
        if k.startswith(_ENV_PREFIXES) and k not in _ENV_IGNORED
    )
    h = hashlib.sha256(_package_source_digest().encode())
    for k, v in env:
        h.update(f"\0{k}={v}".encode())
    return f"jax_aiter:{h.hexdigest()}"


def install():
    try:
        from jax._src import cache_key
    except ImportError:
        return
    previous = getattr(cache_key, "custom_hook", None)
    if previous is None:
        logger.warning("jax._src.cache_key.custom_hook is missing; "
                       "jax_aiter settings will not reach the compilation cache key.")
        return
    if getattr(previous, "_jax_aiter", False):
        return

    def custom_hook() -> str:
        return previous() + cache_key_salt()

    custom_hook._jax_aiter = True
    cache_key.custom_hook = custom_hook
