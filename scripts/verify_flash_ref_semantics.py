#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Cross-check the torch reference scatter added to aiter's op_tests.

The aiter-side test cannot run here (no PyTorch in this container), so the part
that can go wrong silently -- the slot arithmetic and NHD indexing in its
reference -- is re-expressed in numpy and checked against the real kernel through
the jax-aiter shim, which is already validated at exact equality.

If these agree, the reference in op_tests/test_kvcache.py describes the same
mapping the kernel implements, and only the torch API surface remains unexercised.
"""

import numpy as np

import jax
import jax.numpy as jnp

from jax_aiter.ops.append_kv import append_kv

TOKENS_PER_PAGE = 16
NUM_PAGES = 32
NUM_KV_HEADS = 4
HEAD_DIM = 128


def torch_reference_in_numpy(key, value, k_cache, v_cache, slot_mapping):
    """Line-for-line numpy port of run_torch_flash from op_tests/test_kvcache.py."""
    block_size = k_cache.shape[1]
    keep = slot_mapping >= 0
    slots = slot_mapping[keep].astype(np.int64)
    blocks = slots // block_size
    offsets = slots % block_size

    k_cache[blocks, offsets] = key[keep]
    v_cache[blocks, offsets] = value[keep]
    return k_cache, v_cache


def main():
    rng = np.random.default_rng(0)
    num_tokens = 40

    shape = (num_tokens, NUM_KV_HEADS, HEAD_DIM)
    key = rng.standard_normal(shape).astype(np.float32)
    value = rng.standard_normal(shape).astype(np.float32)

    # Scattered slots with padding, mirroring the aiter test.
    slots = rng.permutation(NUM_PAGES * TOKENS_PER_PAGE)[:num_tokens].astype(np.int32)
    slots[1] = -1
    slots[-2] = -1

    cache_shape = (NUM_PAGES, TOKENS_PER_PAGE, NUM_KV_HEADS, HEAD_DIM)

    # 1. The numpy port of the torch reference.
    ref_k, ref_v = torch_reference_in_numpy(
        key, value,
        np.zeros(cache_shape, dtype=np.float32),
        np.zeros(cache_shape, dtype=np.float32),
        slots,
    )

    # 2. The real kernel, through the validated shim.
    k_pool = jnp.zeros(cache_shape, dtype=jnp.float32)
    v_pool = jnp.zeros(cache_shape, dtype=jnp.float32)
    step = jax.jit(
        lambda kp, vp, k, v, s: append_kv(k, v, s, kp, vp), donate_argnums=(0, 1)
    )
    got_k, got_v = step(
        k_pool, v_pool, jnp.asarray(key), jnp.asarray(value), jnp.asarray(slots)
    )

    k_ok = np.array_equal(np.asarray(got_k), ref_k)
    v_ok = np.array_equal(np.asarray(got_v), ref_v)
    print(f"  written slots: {int((slots >= 0).sum())} of {num_tokens}")
    print(f"  k_cache matches kernel exactly: {k_ok}")
    print(f"  v_cache matches kernel exactly: {v_ok}")

    if not (k_ok and v_ok):
        raise SystemExit("reference semantics DISAGREE with the kernel")
    print("\nOK: the op_tests reference describes the same mapping as the kernel.")


if __name__ == "__main__":
    main()
