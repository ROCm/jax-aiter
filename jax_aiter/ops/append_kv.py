# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""M1: append_kv -- write new K/V into a paged KV pool, in place.

Bridges ``aiter::reshape_and_cache_flash`` through XLA FFI. Layout is the v1
recommendation: NHD, K and V in separate pools, no x-packing.

The pools are operands 5 and 6 and results 0 and 1, aliased, so the caller must
pass ``input_output_aliases=APPEND_KV_ALIASES`` and donate both pools at the jit
boundary. :func:`append_kv` does that for you.

The op is pure: the mutation is expressed as a value, and paged attention
consumes the *returned* pools. That data dependence is what orders the read after
the write, so no ``has_side_effect`` is needed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..ffi.registry import register_ffi_target

TARGET = "AppendKvJA"

# operand index -> result index
APPEND_KV_ALIASES = {5: 0, 6: 1}


def _ensure_registered():
    register_ffi_target(TARGET, "ROCM")


def append_kv_raw(
    k_new,
    v_new,
    slot_mapping,
    k_scale,
    v_scale,
    k_pool,
    v_pool,
    *,
    kv_cache_dtype: str = "auto",
):
    """Raw aliased FFI call. Use inside a jit that donates both pools.

    Args:
        k_new: [num_tokens, num_kv_heads, head_dim]
        v_new: [num_tokens, num_kv_heads, head_dim]
        slot_mapping: [num_tokens] int32 absolute token slot; negative skips the
            token, which is how padded rows are handled.
        k_scale: [1] float32, dequant scale. Unused when kv_cache_dtype="auto".
        v_scale: [1] float32.
        k_pool: [num_pages, tokens_per_page, num_kv_heads, head_dim]
        v_pool: same shape as k_pool.
        kv_cache_dtype: "auto" keeps the pool dtype; "fp8"/"fp8_e4m3" quantise.

    Returns:
        (k_pool, v_pool) after the write, aliasing the inputs.
    """
    _ensure_registered()
    call = jax.ffi.ffi_call(
        TARGET,
        (
            jax.ShapeDtypeStruct(k_pool.shape, k_pool.dtype),
            jax.ShapeDtypeStruct(v_pool.shape, v_pool.dtype),
        ),
        input_output_aliases=APPEND_KV_ALIASES,
    )
    return call(
        k_new,
        v_new,
        slot_mapping,
        k_scale,
        v_scale,
        k_pool,
        v_pool,
        kv_cache_dtype=kv_cache_dtype,
    )


def append_kv(
    k_new,
    v_new,
    slot_mapping,
    k_pool,
    v_pool,
    *,
    k_scale=None,
    v_scale=None,
    kv_cache_dtype: str = "auto",
):
    """Convenience form: fills in unit scales and orders arguments readably.

    This is still a traceable call, so it composes into a caller's jit. The
    caller remains responsible for donating the pools -- that is deliberate,
    since donation is a property of the surrounding step, not of this op.
    """
    if k_scale is None:
        k_scale = jnp.ones((1,), dtype=jnp.float32)
    if v_scale is None:
        v_scale = jnp.ones((1,), dtype=jnp.float32)

    return append_kv_raw(
        k_new,
        v_new,
        slot_mapping,
        k_scale,
        v_scale,
        k_pool,
        v_pool,
        kv_cache_dtype=kv_cache_dtype,
    )
