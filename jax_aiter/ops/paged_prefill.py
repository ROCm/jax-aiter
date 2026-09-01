# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""M2c: paged prefill over a KV pool, via aiter's batch-prefill attention.

Attends a ragged batch of new query tokens over the same NHD pool that
:mod:`jax_aiter.ops.append_kv` writes and :mod:`jax_aiter.ops.paged_attention`
reads, using the same page table. Prefill and decode therefore share one pool and
one control plane, with no repacking when a request crosses between them.

Queries are ragged rather than padded. Sequence ``i`` owns
``query[cu_seqlens_q[i] : cu_seqlens_q[i + 1]]`` and attends over the pages
``kv_page_indices[kv_indptr[i] : kv_indptr[i + 1]]``:

    cu_seqlens_q       [batch + 1]     exclusive prefix sum over query lengths
    kv_indptr          [batch + 1]     exclusive prefix sum over page counts
    kv_page_indices    [total_pages]   page ids, concatenated in request order
    kv_last_page_lens  [batch]         occupancy of each request's final page

The op is pure: it consumes the pools and returns attention output, so the data
dependence on ``append_kv``'s aliased result orders the read after the write.

Unlike paged decode there is no ahead-of-time prebuild step to run. The kernels
are compiled into the shim by ``make -f Makefile.kv paged_prefill``; a
configuration outside the generated set is a build-time concern, and the handler
reports it as one.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from ..ffi.registry import register_ffi_target

TARGET = "PagedPrefillJA"


def _ensure_registered():
    register_ffi_target(TARGET, "ROCM")


def paged_prefill(
    query,
    k_pool,
    v_pool,
    cu_seqlens_q,
    kv_indptr,
    kv_page_indices,
    kv_last_page_lens,
    *,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scale: float | None = None,
    causal: bool = True,
    logits_soft_cap: float = 0.0,
    window_size: tuple[int, int] = (-1, -1),
):
    """Paged prefill attention over a ragged batch of query tokens.

    Args:
        query: [total_q, num_heads, head_dim], all sequences concatenated.
        k_pool: [num_pages, tokens_per_page, num_kv_heads, head_dim]
        v_pool: same shape as ``k_pool``
        cu_seqlens_q: [batch + 1] int32
        kv_indptr: [batch + 1] int32
        kv_page_indices: [total_pages] int32
        kv_last_page_lens: [batch] int32
        max_seqlen_q: longest query segment in the batch.
        max_seqlen_k: longest context in the batch.
        scale: softmax scale; defaults to ``1/sqrt(head_dim)``.
        causal: bottom-right aligned causal mask. With a non-empty prefix in the
            pool this is what makes query token ``j`` see key positions up to
            ``seqlen_k - seqlen_q + j``, so appended tokens attend over their
            history but not over each other's futures.
        window_size: ``(left, right)`` sliding window; ``(-1, -1)`` disables it.

    Returns:
        [total_q, num_heads, head_dim] attention output, in the query dtype.
    """
    _ensure_registered()

    if query.ndim != 3:
        raise ValueError(
            f"query must be [total_q, num_heads, head_dim], got {query.shape}"
        )
    head_dim = query.shape[2]
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    window_left, window_right = window_size

    call = jax.ffi.ffi_call(
        TARGET,
        jax.ShapeDtypeStruct(query.shape, query.dtype),
    )
    return call(
        query,
        k_pool,
        v_pool,
        cu_seqlens_q,
        kv_indptr,
        kv_page_indices,
        kv_last_page_lens,
        scale=np.float32(scale),
        logits_soft_cap=np.float32(logits_soft_cap),
        max_seqlen_q=np.int64(max_seqlen_q),
        max_seqlen_k=np.int64(max_seqlen_k),
        causal=bool(causal),
        window_size_left=np.int64(window_left),
        window_size_right=np.int64(window_right),
    )
