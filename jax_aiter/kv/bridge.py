# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The seam between the neutral KV vocabulary and AITER's physical ABI.

M2a defines two layers on purpose. The control plane owns semantics -- pool
geometry and per-step page bookkeeping -- and knows nothing about strides,
packing or kernel shapes. :mod:`jax_aiter.kv.abi` owns the physical contract.
This module is the conversion between them, and it is the only place that has to
know both.

It deliberately does **not** import the control plane. Everything here is
duck-typed against two small protocols, which is what keeps the dependency
pointing one way: a front end may depend on this library, but this library must
not depend on a front end. MaxText's ``KvStorageLayoutV1`` and ``KvPageTableV1``
satisfy these protocols today; anything else exposing the same handful of members
works unchanged, which is the property that lets a vendor-neutral paged path have
peer backends at all.

A layout must provide::

    num_pages  tokens_per_page  head_dim  dtype  padding_page_id
    heads_per_shard()

A page table must provide::

    num_requests  num_tokens  query_lens
    validate(tokens_per_page)  indptr()  flat_page_indices()
    last_page_lens(tokens_per_page)  slot_mapping(tokens_per_page, padding_page_id)

The single source of truth for every physical number is the ABI object: pool
shapes and strides come from it rather than being recomputed here, so there is
one derivation to be wrong rather than two that can silently disagree.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np

from .abi import AiterPagedAttentionAbiV1

__all__ = [
    "PagedStepPlan",
    "allocate_pools",
    "plan_step",
    "append_step",
    "attend_step",
]


@dataclasses.dataclass(frozen=True)
class PagedStepPlan:
    """One step's metadata, converted to the device arrays the kernels take.

    Every array is int32, which is what all three kernels require and what a JAX
    front end produces without ``jax_enable_x64``.
    """

    kv_indptr: jnp.ndarray          # [num_requests + 1]
    kv_page_indices: jnp.ndarray    # [total_pages]
    kv_last_page_lens: jnp.ndarray  # [num_requests]
    slot_mapping: jnp.ndarray       # [num_tokens]
    cu_seqlens_q: jnp.ndarray       # [num_requests + 1]
    max_seqlen_q: int
    max_seqlen_k: int
    is_decode: bool

    @property
    def num_requests(self) -> int:
        return int(self.cu_seqlens_q.shape[0]) - 1


def allocate_pools(layout, abi: AiterPagedAttentionAbiV1 | None = None,
                   *, num_q_heads: int | None = None):
    """Allocate the K and V pools for one layer of one shard.

    ``num_q_heads`` is optional but worth passing: it lets the ABI reject a
    non-integral GQA ratio here, at allocation, instead of inside a kernel.
    """
    abi = abi or AiterPagedAttentionAbiV1()
    if num_q_heads is not None:
        abi.validate(layout, num_q_heads)

    shape = abi.pool_shape(layout)
    dtype = jnp.dtype(layout.dtype)
    return jnp.zeros(shape, dtype=dtype), jnp.zeros(shape, dtype=dtype)


def plan_step(layout, page_table,
              abi: AiterPagedAttentionAbiV1 | None = None) -> PagedStepPlan:
    """Convert a neutral page table into the device arrays the kernels take.

    The page table is validated first. That check is cheap and the failure it
    prevents is not: an overstated last-page length is how an attention kernel
    reads bytes left behind by a recycled page's previous occupant.
    """
    abi = abi or AiterPagedAttentionAbiV1()
    tokens_per_page = layout.tokens_per_page
    page_table.validate(tokens_per_page)

    query_lens = np.asarray(page_table.query_lens, dtype=np.int32)
    cu_seqlens_q = np.zeros((page_table.num_requests + 1,), dtype=np.int32)
    if page_table.num_requests:
        np.cumsum(query_lens, out=cu_seqlens_q[1:])

    last_page_lens = page_table.last_page_lens(tokens_per_page)
    seq_lens = np.asarray(page_table.seq_lens, dtype=np.int32)

    # The padding sentinel comes from the layout rather than the ABI default:
    # the control plane picks which page id it reserves, and the ABI only states
    # that one is reserved.
    slot_mapping = page_table.slot_mapping(
        tokens_per_page, padding_page_id=layout.padding_page_id
    )

    return PagedStepPlan(
        kv_indptr=jnp.asarray(page_table.indptr(), dtype=jnp.int32),
        kv_page_indices=jnp.asarray(page_table.flat_page_indices(), dtype=jnp.int32),
        kv_last_page_lens=jnp.asarray(last_page_lens, dtype=jnp.int32),
        slot_mapping=jnp.asarray(slot_mapping, dtype=jnp.int32),
        cu_seqlens_q=jnp.asarray(cu_seqlens_q, dtype=jnp.int32),
        max_seqlen_q=int(query_lens.max()) if query_lens.size else 0,
        max_seqlen_k=int(seq_lens.max()) if seq_lens.size else 0,
        is_decode=bool(query_lens.size and np.all(query_lens == 1)),
    )


def append_step(k_new, v_new, k_pool, v_pool, plan: PagedStepPlan,
                abi: AiterPagedAttentionAbiV1 | None = None):
    """Write this step's new K/V into the pools, in place.

    Returns the aliased pools; rebind rather than keep the originals.
    """
    from ..ops.append_kv import append_kv

    abi = abi or AiterPagedAttentionAbiV1()
    return append_kv(
        k_new, v_new, plan.slot_mapping, k_pool, v_pool,
        kv_cache_dtype=abi.kv_cache_dtype,
    )


def attend_step(query, k_pool, v_pool, plan: PagedStepPlan,
                abi: AiterPagedAttentionAbiV1 | None = None,
                *, scale: float | None = None, causal: bool = True):
    """Attend over the pool, routing to decode or prefill by query shape.

    Both kernels read the same pool and the same page table, so the only thing
    that decides between them is whether every request contributes exactly one
    token. Callers that already know which phase they are in can call the ops
    directly; this exists so a driver does not have to branch.
    """
    from ..ops.paged_attention import paged_attention
    from ..ops.paged_prefill import paged_prefill

    abi = abi or AiterPagedAttentionAbiV1()

    if plan.is_decode:
        # Decode takes one token per sequence as [num_seqs, heads, dim], while
        # the ragged form is [total_q, heads, dim] with total_q == num_seqs.
        return paged_attention(
            query, k_pool, v_pool,
            plan.kv_indptr, plan.kv_page_indices, plan.kv_last_page_lens,
            max_seq_len=plan.max_seqlen_k, scale=scale,
            kv_cache_dtype=abi.kv_cache_dtype,
        )

    return paged_prefill(
        query, k_pool, v_pool, plan.cu_seqlens_q,
        plan.kv_indptr, plan.kv_page_indices, plan.kv_last_page_lens,
        max_seqlen_q=plan.max_seqlen_q, max_seqlen_k=plan.max_seqlen_k,
        scale=scale, causal=causal,
    )
