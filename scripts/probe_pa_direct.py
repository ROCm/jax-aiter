#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Call a prebuilt pa_ragged kernel directly, bypassing the XLA FFI handler.

Isolation harness for splitting kernel problems from binding problems. Buffers
come from JAX arrays so the layout matches the real path exactly; only the call
site differs. If this crashes, the kernel is being driven wrong -- shapes,
strides, workspace or metadata. If it survives but the FFI path misbehaves, the
handler's wiring is at fault.

This is how the M2b launch fault was pinned down: it reproduced here with no
XLA and no torch in the picture, which ruled out the FFI shim in one step.
"""

from __future__ import annotations

import ctypes
import math
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jax_aiter.kv import pa_config  # noqa: E402


def ptr(arr) -> ctypes.c_void_p:
    return ctypes.c_void_p(arr.unsafe_buffer_pointer())


def main() -> int:
    num_seqs, num_heads, num_kv_heads = 1, 8, 8
    head_dim, tokens_per_page, num_pages = 128, 16, 32
    max_seq_len = 256
    seq_len = 16
    dtype = jnp.bfloat16

    cfg = pa_config.make_config(
        gqa_ratio=num_heads // num_kv_heads,
        head_size=head_dim,
        max_seq_len=max_seq_len,
        dtype=dtype,
        block_size=tokens_per_page,
    )
    name = pa_config.func_name(cfg)
    lib_path = os.path.expanduser(f"~/.aiter/build/{name}/lib.so")
    print(f"config   : {cfg}")
    print(f"func_name: {name}")
    print(f"lib      : {lib_path}  exists={os.path.exists(lib_path)}")
    if not os.path.exists(lib_path):
        print("not built; run scripts/prebuild_pa_ragged.py first", file=sys.stderr)
        return 1

    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.standard_normal((num_seqs, num_heads, head_dim)), dtype=dtype)
    pool = jnp.asarray(
        rng.standard_normal((num_pages, tokens_per_page, num_kv_heads, head_dim)),
        dtype=dtype,
    )
    k_pool, v_pool = pool, pool
    out = jnp.zeros((num_seqs, num_heads, head_dim), dtype=dtype)

    indptr = jnp.asarray([0, 1], dtype=jnp.int32)
    pages = jnp.asarray([1], dtype=jnp.int32)
    last_lens = jnp.asarray([seq_len], dtype=jnp.int32)
    ones = jnp.ones((1,), dtype=jnp.float32)

    parts = pa_config.max_num_partitions(max_seq_len)
    ws_bytes = pa_config.workspace_bytes(num_seqs, num_heads, head_dim,
                                         max_seq_len, dtype)
    workspace = jnp.zeros((ws_bytes,), dtype=jnp.uint8)
    print(f"partitions={parts} workspace={ws_bytes} bytes")

    for a in (q, k_pool, out, indptr, pages, last_lens, ones, workspace):
        a.block_until_ready()

    lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_LOCAL)
    fn = getattr(lib, name)
    fn.restype = None
    fn.argtypes = [
        ctypes.c_void_p,  # out
        ctypes.c_void_p,  # workspace
        ctypes.c_void_p,  # query
        ctypes.c_void_p,  # key_cache
        ctypes.c_void_p,  # value_cache
        ctypes.c_void_p,  # kv_indptr
        ctypes.c_void_p,  # kv_page_indices
        ctypes.c_void_p,  # kv_last_page_lens
        ctypes.c_void_p,  # alibi_slopes
        ctypes.c_void_p,  # q_scale
        ctypes.c_void_p,  # k_scale
        ctypes.c_void_p,  # v_scale
        ctypes.c_void_p,  # fp8_out_scale
        ctypes.c_float,   # scale
        ctypes.c_float,   # logits_soft_cap
        ctypes.c_int,     # num_seqs
        ctypes.c_int,     # num_kv_heads
        ctypes.c_int,     # num_heads
        ctypes.c_int,     # max_num_partitions
        ctypes.c_int,     # q_stride
        ctypes.c_int,     # kv_block_stride
        ctypes.c_int,     # kv_head_stride
        ctypes.c_int,     # kv_seq_stride
        ctypes.c_void_p,  # stream
    ]

    kv_seq_stride = num_kv_heads * head_dim
    kv_block_stride = tokens_per_page * kv_seq_stride
    print(f"strides: q={num_heads * head_dim} block={kv_block_stride} "
          f"head={head_dim} seq={kv_seq_stride}")

    print("calling kernel on the null (default) stream ...", flush=True)
    fn(
        ptr(out), ptr(workspace), ptr(q), ptr(k_pool), ptr(v_pool),
        ptr(indptr), ptr(pages), ptr(last_lens),
        None, None, ptr(ones), ptr(ones), None,
        ctypes.c_float(1.0 / math.sqrt(head_dim)), ctypes.c_float(0.0),
        num_seqs, num_kv_heads, num_heads, parts,
        num_heads * head_dim, kv_block_stride, head_dim, kv_seq_stride,
        None,
    )

    hip = ctypes.CDLL("libamdhip64.so")
    rc = hip.hipDeviceSynchronize()
    print(f"hipDeviceSynchronize -> {rc}")
    print("survived the call")
    return 0


if __name__ == "__main__":
    sys.exit(main())
