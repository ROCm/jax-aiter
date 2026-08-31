# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""M2b: paged decode over a KV pool, via aiter's paged_attention_ragged.

Reads the NHD pool that :mod:`jax_aiter.ops.append_kv` writes, with no
conversion between the two.

Metadata is FlashInfer-shaped and int32 throughout, which is what the control
plane already produces:

    kv_indptr          [num_seqs + 1]  exclusive prefix sum over page counts
    kv_page_indices    [total_pages]   page ids, concatenated in request order
    kv_last_page_lens  [num_seqs]      occupancy of each request's final page

The op is pure. It consumes the pools and returns attention output, so the data
dependence on ``append_kv``'s aliased result is what orders the read after the
write.

The kernel configurations are compiled into ``paged_attention_ja.so`` by
``Makefile.kv`` (sources rendered by ``scripts/gen_pa_ragged.py``), so the set
is fixed at link time. An unlisted configuration is a clear error naming the
generator, rather than aiter shelling out to Python from inside a kernel
launch.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from ..ffi.registry import register_ffi_target
from ..kv import pa_config

TARGET = "PagedAttentionJA"

PARTITION_SIZE = pa_config.PARTITION_SIZE


def _ensure_registered():
    register_ffi_target(TARGET, "ROCM")


def max_num_partitions(max_seq_len: int) -> int:
    return pa_config.max_num_partitions(max_seq_len)


def workspace_shape(num_seqs: int, num_heads: int, head_dim: int,
                    max_seq_len: int, dtype) -> tuple[int]:
    """Scratch the kernel needs, as a flat uint8 buffer."""
    return (pa_config.workspace_bytes(num_seqs, num_heads, head_dim,
                                      max_seq_len, dtype),)


def paged_attention(
    query,
    k_pool,
    v_pool,
    kv_indptr,
    kv_page_indices,
    kv_last_page_lens,
    *,
    max_seq_len: int,
    scale: float | None = None,
    logits_soft_cap: float = 0.0,
    kv_cache_dtype: str = "auto",
    k_scale=None,
    v_scale=None,
    workspace=None,
):
    """Paged decode attention: one query token per sequence.

    Args:
        query: [num_seqs, num_heads, head_dim]
        k_pool: [num_pages, tokens_per_page, num_kv_heads, head_dim]
        v_pool: same shape as ``k_pool``
        kv_indptr: [num_seqs + 1] int32
        kv_page_indices: [total_pages] int32
        kv_last_page_lens: [num_seqs] int32
        max_seq_len: longest context in the batch. Only affects partition count
            and workspace size, but it is part of the compiled kernel
            configuration, so bucket it rather than passing an exact per-step
            value.
        scale: softmax scale; defaults to ``1/sqrt(head_dim)``.
        kv_cache_dtype: ``"auto"`` keeps the pool dtype.
        workspace: optional preallocated scratch. A driver should hoist this out
            of the step rather than reallocating every call.

    Returns:
        [num_seqs, num_heads, head_dim] attention output, in the query dtype.
    """
    _ensure_registered()

    num_seqs, num_heads, head_dim = query.shape
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    if k_scale is None:
        k_scale = jnp.ones((1,), dtype=jnp.float32)
    if v_scale is None:
        v_scale = jnp.ones((1,), dtype=jnp.float32)
    if workspace is None:
        workspace = jnp.zeros(
            workspace_shape(num_seqs, num_heads, head_dim, max_seq_len, query.dtype),
            dtype=jnp.uint8,
        )

    # Name the prebuilt configuration here rather than in the handler: the same
    # helper the prebuild script uses, so the folder it wrote and the folder the
    # handler opens cannot drift.
    num_kv_heads = k_pool.shape[2]
    config = pa_config.make_config(
        gqa_ratio=num_heads // num_kv_heads,
        head_size=head_dim,
        max_seq_len=max_seq_len,
        dtype=query.dtype,
        block_size=k_pool.shape[1],
        kv_cache_dtype=kv_cache_dtype,
    )

    call = jax.ffi.ffi_call(
        TARGET,
        jax.ShapeDtypeStruct(query.shape, query.dtype),
    )
    return call(
        query,
        k_pool,
        v_pool,
        kv_indptr,
        kv_page_indices,
        kv_last_page_lens,
        k_scale,
        v_scale,
        workspace,
        scale=np.float32(scale),
        logits_soft_cap=np.float32(logits_soft_cap),
        max_num_partitions=np.int64(pa_config.max_num_partitions(max_seq_len)),
        func_name=pa_config.func_name(config),
    )
