# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Context parallelism helpers for AITER flash attention.

Provides building blocks for context parallelism that a future Phase 3
implementation can use.  Context parallelism requires JAX primitive
registration (core.Primitive + mlir.register_lowering) to properly embed
lax.all_gather / psum_scatter inside the SPMD partitioner; the current
custom_partitioning wrapper approach does not support nested collectives.
"""

from __future__ import annotations


def kv_seqlens_for_rank(rank, kv_max_seqlen, cp_size, load_balanced):
    """Return KV slice lengths for each sub-chunk of a given CP rank.

    Each rank's local Q is split into 2 halves.  For causal attention,
    each half attends to a different KV slice (earlier half sees less
    context).  Returns ``[kv_len_half_0, kv_len_half_1]``.
    """
    kv_per_subrank = kv_max_seqlen // (cp_size * 2)
    if load_balanced:
        return [
            (rank + 1) * kv_per_subrank,
            kv_max_seqlen - rank * kv_per_subrank,
        ]
    return [
        (rank * 2 + 1) * kv_per_subrank,
        (rank * 2 + 2) * kv_per_subrank,
    ]
