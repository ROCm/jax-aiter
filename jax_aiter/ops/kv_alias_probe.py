# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""M0 aliasing probe: in-place scatter-add into a donated pool through XLA FFI.

Exists only to prove the ``input_output_aliases`` contract on ROCm before the
paged-KV work depends on it. M1's ``append_kv`` replaces it in production.

Deliberately not wrapped in ``jax.jit`` here: donation is the caller's decision,
so the probe has to compose into a caller's jit with ``donate_argnums``.
"""

from __future__ import annotations

import jax

from ..ffi.registry import register_ffi_target

TARGET = "KvAliasProbeJA"


def _ensure_registered():
    register_ffi_target(TARGET, "ROCM")


def kv_alias_probe(pool, row_idx, vals):
    """Return ``pool`` with ``pool[row_idx[i]] += vals[i]`` applied in place.

    The pool is operand 0 and result 0 under ``input_output_aliases={0: 0}``, so
    XLA must reuse the buffer rather than allocate a replacement. The handler
    additionally refuses to run if the two resolve to different device pointers.

    Rows whose index is negative or beyond the pool are skipped, which is the
    behaviour the padding sentinel relies on at M4.

    Args:
        pool: [pool_rows, ...] f32 or bf16. Donate this at the jit boundary.
        row_idx: [n_rows] int32 destination row per value row.
        vals: [n_rows, ...] same trailing dims and dtype as ``pool``.

    Returns:
        The mutated pool, aliasing the input buffer.
    """
    _ensure_registered()
    call = jax.ffi.ffi_call(
        TARGET,
        jax.ShapeDtypeStruct(pool.shape, pool.dtype),
        input_output_aliases={0: 0},
    )
    return call(pool, row_idx, vals)
