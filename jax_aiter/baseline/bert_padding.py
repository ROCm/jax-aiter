# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""
JAX equivalents of AITer's bert_padding helpers used by tests:
- unpad_input: flattens [B,S,...] to [N,...] using an attention mask (and optional unused mask),
  also returns indices, cumulative sequence lengths, and max seqlen-in-batch.
- pad_input: inverse operation that scatters back to [B,S,...] given indices.

Accuracy-first implementation using jnp operations. Suitable for use outside or inside jit
for now; later we can optimize with lax if needed for strict static-shape compilation.
"""

from __future__ import annotations

import jax.numpy as jnp


def unpad_input(
    hidden_states: jnp.ndarray,
    attention_mask: jnp.ndarray,
    unused_mask: jnp.ndarray | None = None,
):
    """
    Arguments:
        hidden_states: [batch, seqlen, ...]
        attention_mask: [batch, seqlen], bool/int, 1 means valid, 0 means masked
        unused_mask:   [batch, seqlen], bool/int, 1 means allocated-but-unused (optional)

    Returns:
        unpadded:                [total_nnz, ...]
        indices:                 [total_nnz] (int32) indices into flattened [B*S]
        cu_seqlens:              [batch + 1] (int32) cumulative sequence lengths
        max_seqlen_in_batch:     scalar int32
        used_seqlens_in_batch:   [batch] (int32) counts from attention_mask only
    """
    assert hidden_states.ndim >= 2, "hidden_states must have [B,S,...]"

    attention_mask_bool = attention_mask.astype(bool)
    if unused_mask is not None:
        all_masks = attention_mask_bool | unused_mask.astype(bool)
    else:
        all_masks = attention_mask_bool

    seqlens_in_batch = all_masks.sum(axis=-1).astype(jnp.int32)
    used_seqlens_in_batch = attention_mask_bool.sum(axis=-1).astype(jnp.int32)

    flat_mask = all_masks.reshape(-1)
    indices = jnp.nonzero(flat_mask, size=None)[0].astype(jnp.int32)

    flat_states = hidden_states.reshape((-1,) + hidden_states.shape[2:])
    unpadded = flat_states[indices]

    cu = jnp.cumsum(seqlens_in_batch, dtype=jnp.int32)
    cu_seqlens = jnp.pad(cu, (1, 0))

    max_seqlen_in_batch = jnp.max(seqlens_in_batch).astype(jnp.int32)

    return (
        unpadded,
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
        used_seqlens_in_batch,
    )


def pad_input(
    hidden_states: jnp.ndarray,
    indices: jnp.ndarray,
    batch: int,
    seqlen: int,
):
    """
    Arguments:
        hidden_states: [total_nnz, ...]
        indices:       [total_nnz] (int32) positions in flattened [B*S]
        batch:         int
        seqlen:        int

    Returns:
        output: [batch, seqlen, ...]
    """
    flat_size = int(batch) * int(seqlen)
    out_shape = (flat_size,) + hidden_states.shape[1:]
    out = jnp.zeros(out_shape, dtype=hidden_states.dtype)
    out = out.at[indices.astype(jnp.int32)].set(hidden_states)
    return out.reshape((int(batch), int(seqlen)) + hidden_states.shape[1:])
