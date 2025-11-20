# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""
Simple JAX attention baseline for testing jax-aiter MHA kernels.

- torch-like order of ops
- bias OR ALiBi
- MHA / MQA / GQA
- fp32 compute, cast back
- includes the helpers the tests were re-defining
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from einops import rearrange, repeat
from .bert_padding import pad_input, unpad_input
from jax_aiter.ja_compat import dtypes


def ck_randval_to_dropout_mask(randval: jnp.ndarray, p: float) -> jnp.ndarray:
    """Return a mask in the same style as the PyTorch helper:
    values >= 0  -> keep
    values <  0  -> drop
    Assumes randval is in [0, 255].
    """
    return jnp.floor(255.0 * (1.0 - p)).astype(dtypes.fp32) - randval.astype(
        dtypes.fp32
    )


def construct_local_mask(
    seqlen_q: int,
    seqlen_k: int,
    window_size=(-1, -1),
    query_padding_mask: jnp.ndarray | None = None,
    key_padding_mask: jnp.ndarray | None = None,
    key_leftpad: jnp.ndarray | None = None,
):
    row_idx = rearrange(jnp.arange(seqlen_q, dtype=jnp.int32), "s -> s 1")
    col_idx = jnp.arange(seqlen_k, dtype=jnp.int32)

    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = jnp.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2 ** 32)

    if key_padding_mask is not None:
        sk = rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    else:
        sk = seqlen_k

    if query_padding_mask is not None:
        sq = rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    else:
        sq = seqlen_q

    wl, wr = window_size
    base = row_idx + sk - sq

    if wl < 0:
        mask = col_idx > base + wr
    else:
        right = jnp.minimum(base + wr, sk)
        left = base - wl
        mask = jnp.logical_or(col_idx > right, col_idx < left)

    # normalize to (B, sq, sk)
    if mask.ndim == 2:
        mask = mask[None, ...]
    elif mask.ndim == 4 and mask.shape[1] == 1:
        # If mask is (B, 1, sq, sk), squeeze the singleton dimension
        mask = mask.squeeze(1)
    return mask


def convert_flash_attn_S_to_softmax(
    S: jnp.ndarray,
    seqlen_q: int,
    seqlen_k: int,
    query_padding_mask: jnp.ndarray | None,
    key_padding_mask: jnp.ndarray | None,
    head_dim: int,
    is_dropout: bool,
    causal: bool = False,
    window_size=(-1, -1),
) -> jnp.ndarray:
    if causal:
        window_size = (window_size[0], 0)

    B, H, seqlen_q_rounded, seqlen_k_rounded = S.shape
    S_converted = S

    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            key_leftpad=None,
        )  # (B, sq, sk)

        pad_q = seqlen_q_rounded - seqlen_q
        pad_k = seqlen_k_rounded - seqlen_k
        local_mask = jnp.pad(
            local_mask,
            ((0, 0), (0, pad_q), (0, pad_k)),
            constant_values=True,
        )
        S_converted = jnp.where(local_mask[:, None, :, :], 0.0, S_converted)

    seqlen_q_og = (
        query_padding_mask.shape[-1]
        if query_padding_mask is not None
        else seqlen_q_rounded
    )
    if query_padding_mask is not None:
        qpm = jnp.pad(
            query_padding_mask,
            ((0, 0), (0, seqlen_q_rounded - seqlen_q_og)),
            constant_values=False,
        )
        qpm_inv = rearrange(~qpm, "b s -> b 1 s 1")
        S_converted = jnp.where(qpm_inv, 0.0, S_converted)

    seqlen_k_og = (
        key_padding_mask.shape[-1] if key_padding_mask is not None else seqlen_k
    )
    if key_padding_mask is not None:
        kpm = jnp.pad(
            key_padding_mask,
            ((0, 0), (0, seqlen_k_rounded - seqlen_k_og)),
            constant_values=False,
        )
        kpm_inv = rearrange(~kpm, "b s -> b 1 1 s")
        S_converted = jnp.where(kpm_inv, 0.0, S_converted)

    if seqlen_q_og > seqlen_q_rounded:
        S_converted = jnp.pad(
            S_converted,
            ((0, 0), (0, 0), (0, seqlen_q_og - seqlen_q_rounded), (0, 0)),
            constant_values=0.0,
        )
        seqlen_q_rounded = seqlen_q_og

    if seqlen_k_og > seqlen_k_rounded:
        S_converted = jnp.pad(
            S_converted,
            ((0, 0), (0, 0), (0, 0), (0, seqlen_k_og - seqlen_k_rounded)),
            constant_values=0.0,
        )
        seqlen_k_rounded = seqlen_k_og

    return S_converted[:, :, :seqlen_q, :seqlen_k]


def attn_bias_from_alibi_slopes(
    slopes,
    seqlen_q,
    seqlen_k,
    query_padding_mask=None,
    key_padding_mask=None,
    causal=False,
    key_leftpad=None,
):
    batch, nheads = slopes.shape
    slopes = rearrange(slopes, "b h -> b h 1 1")
    if causal:
        return jnp.arange(-seqlen_k + 1, 1, dtype=dtypes.fp32) * slopes
    else:
        row_idx = rearrange(jnp.arange(seqlen_q, dtype=jnp.int32), "s -> s 1")
        col_idx = jnp.arange(seqlen_k, dtype=jnp.int32)
        if key_leftpad is not None:
            key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
            col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
            col_idx = jnp.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2 ** 32)
        sk = (
            seqlen_k
            if key_padding_mask is None
            else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
        )
        sq = (
            seqlen_q
            if query_padding_mask is None
            else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
        )
        relative_pos = jnp.abs(row_idx + sk - sq - col_idx)
        return -slopes * relative_pos.astype(slopes.dtype)


def generate_qkv(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    kvpacked=False,
    qkvpacked=False,
    input_layout="BSHD",
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, d)
        k: (batch_size, seqlen_k, nheads_k, d)
        v: (batch_size, seqlen_k, nheads_k, d_v)
        query_padding_mask: (batch_size, seqlen), bool
        key_padding_mask: (batch_size, seqlen), bool
        input_layout: "BSHD", "BHSD", "SBHD"
    """
    assert not (kvpacked and qkvpacked)
    batch_size, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_k, _ = k.shape
    _, _, _, d_v = v.shape
    assert k.shape == (batch_size, seqlen_k, nheads_k, d)
    assert v.shape == (batch_size, seqlen_k, nheads_k, d_v)

    if input_layout == "BHSD":
        # PyTorch version does q.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)
        # which is effectively a no-op. We match that behavior.
        # No transpose needed
        pass
    elif input_layout == "SBHD":
        # Similar to BHSD, PyTorch does a double transpose that cancels out
        # No transpose needed
        pass

    if query_padding_mask is not None:
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, _ = unpad_input(
            q, query_padding_mask
        )
        output_pad_fn = lambda output_unpad: pad_input(
            output_unpad, indices_q, batch_size, seqlen_q
        )
    else:
        q_unpad = rearrange(q, "b s h d -> (b s) h d")
        cu_seqlens_q = jnp.arange(
            0,
            (batch_size + 1) * seqlen_q,
            step=seqlen_q,
            dtype=dtypes.i32,
        )
        max_seqlen_q = seqlen_q
        output_pad_fn = lambda output_unpad: rearrange(
            output_unpad, "(b s) h d -> b s h d", b=batch_size
        )

    if key_padding_mask is not None:
        k_unpad, indices_k, cu_seqlens_k, max_seqlen_k, _ = unpad_input(
            k, key_padding_mask
        )
        v_unpad, _, _, _, _ = unpad_input(v, key_padding_mask)
    else:
        k_unpad = rearrange(k, "b s h d -> (b s) h d")
        v_unpad = rearrange(v, "b s h d -> (b s) h d")
        cu_seqlens_k = jnp.arange(
            0,
            (batch_size + 1) * seqlen_k,
            step=seqlen_k,
            dtype=dtypes.i32,
        )
        max_seqlen_k = seqlen_k

    if qkvpacked:
        assert (query_padding_mask is None and key_padding_mask is None) or (
            jnp.all(query_padding_mask == key_padding_mask)
        )
        assert seqlen_q == seqlen_k
        assert nheads == nheads_k
        assert d == d_v
        qkv_unpad = jnp.concatenate([q_unpad, k_unpad, v_unpad], axis=1)
        qkv = jnp.concatenate([q, k, v], axis=2)
        if query_padding_mask is not None:
            dqkv_pad_fn = lambda dqkv_unpad: pad_input(
                dqkv_unpad, indices_q, batch_size, seqlen_q
            )
        else:
            dqkv_pad_fn = lambda dqkv_unpad: rearrange(
                dqkv_unpad, "(b s) h d -> b s h d", b=batch_size
            )
        q_unpad, k_unpad, v_unpad = jnp.split(
            qkv_unpad, [q_unpad.shape[1], q_unpad.shape[1] + k_unpad.shape[1]], axis=1
        )
        q, k, v = jnp.split(qkv, [q.shape[2], q.shape[2] + k.shape[2]], axis=2)
        return (
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q,
            k,
            v,
            output_pad_fn,
            dqkv_pad_fn,
            dqkv_pad_fn,
        )
    elif kvpacked:
        assert d == d_v
        kv_unpad = jnp.stack([k_unpad, v_unpad], axis=1)
        kv = jnp.stack([k, v], axis=2)
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dkv_pad_fn = lambda dkv_unpad: pad_input(
                dkv_unpad, indices_k, batch_size, seqlen_k
            )
        else:
            dkv_pad_fn = lambda dkv_unpad: rearrange(
                dkv_unpad, "(b s) h d -> b s h d", b=batch_size
            )

        # Unbind k and v like torch does - this is key!
        k_unpad, v_unpad = jnp.split(kv_unpad, 2, axis=1)
        k_unpad = jnp.squeeze(k_unpad, axis=1)
        v_unpad = jnp.squeeze(v_unpad, axis=1)
        k, v = jnp.split(kv, 2, axis=2)
        k = jnp.squeeze(k, axis=2)
        v = jnp.squeeze(v, axis=2)

        return (
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q,
            k,
            v,
            output_pad_fn,
            dq_pad_fn,
            dkv_pad_fn,
        )
    else:
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dk_pad_fn = lambda dk_unpad: pad_input(
                dk_unpad, indices_k, batch_size, seqlen_k
            )
        else:
            dk_pad_fn = lambda dk_unpad: rearrange(
                dk_unpad, "(b s) h d -> b s h d", b=batch_size
            )
        return (
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q,
            k,
            v,
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        )


def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    key_leftpad=None,
):
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q = q.astype(dtypes.fp32)
        k = k.astype(dtypes.fp32)
        v = v.astype(dtypes.fp32)

    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]

    if not reorder_ops:
        scores = jnp.einsum("bthd,bshd->bhts", q / jnp.sqrt(d), k)
    else:
        scores = jnp.einsum("bthd,bshd->bhts", q, k / jnp.sqrt(d))

    if softcap > 0:
        scores = scores / softcap
        scores = jnp.tanh(scores)
        scores = scores * softcap

    if key_padding_mask is not None:
        scores = jnp.where(
            rearrange(~key_padding_mask, "b s -> b 1 1 s"),
            -jnp.inf,
            scores,
        )

    local_mask = None
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            key_leftpad=key_leftpad,
        )
        # broadcast to heads
        scores = jnp.where(local_mask[:, None, :, :], -jnp.inf, scores)

    if attn_bias is not None:
        scores = scores + attn_bias

    # lse like torch
    lse = jax.scipy.special.logsumexp(scores, axis=-1).astype(v.dtype)
    attention = jax.nn.softmax(scores, axis=-1).astype(v.dtype)

    if local_mask is not None:
        # local_mask has shape (B, sq, sk), attention has shape (B, H, sq, sk)
        # We need to check if all keys are masked for each query position
        mask_all_masked = jnp.all(local_mask, axis=-1, keepdims=True)  # (B, sq, 1)
        # Reshape to broadcast correctly with attention (B, H, sq, sk)
        mask_all_masked = mask_all_masked[:, None, :, :]  # (B, 1, sq, 1)
        attention = jnp.where(
            mask_all_masked,
            0.0,
            attention,
        )

    if query_padding_mask is not None:
        attention = jnp.where(
            rearrange(~query_padding_mask, "b s -> b 1 s 1"),
            0.0,
            attention,
        )

    dropout_scaling = 1.0 / (1.0 - dropout_p)
    if dropout_mask is not None:
        attention_drop = jnp.where(dropout_mask, attention, 0.0)
    else:
        attention_drop = attention

    # Defensive check: ensure attention_drop is 4D before the einsum
    # This maintains PyTorch parity where attention is always (B, H, Tq, Tk)
    if attention_drop.ndim != 4:
        raise ValueError(
            f"attention_drop must be 4D (batch, heads, seq_q, seq_k), got shape {attention_drop.shape}. "
            f"This indicates a bug in the masking logic."
        )

    output = jnp.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)

    if query_padding_mask is not None:
        output = jnp.where(
            rearrange(~query_padding_mask, "b s -> b s 1 1"),
            0.0,
            output,
        )

    return (
        output.astype(dtype_og),
        attention.astype(dtype_og),
        lse.astype(dtype_og),
    )


def pad_rearrange_dropout_mask_hts_to_bhss(
    S_dmask_hts: jnp.ndarray,
    cu_seqlens_q: jnp.ndarray,
    seqlen_q_rounded: int,
    seqlen_k_rounded: int,
) -> jnp.ndarray:
    """
    Convert varlen S_dmask from HTS layout to BHSS:
      Input:  (H, total_q, max_seqlen_k)
      Output: (B, H, seqlen_q_rounded, seqlen_k_rounded)

    We use cu_seqlens_q to split per batch and pad each chunk to (seqlen_q_rounded, seqlen_k_rounded).
    """
    nheads, total_q, max_seqlen_k = S_dmask_hts.shape
    batch_size = int(cu_seqlens_q.shape[0] - 1)

    out = jnp.zeros(
        (batch_size, nheads, seqlen_q_rounded, seqlen_k_rounded),
        dtype=S_dmask_hts.dtype,
    )

    # Python loop is fine here (test utility), not performance critical
    for b in range(batch_size):
        start = int(cu_seqlens_q[b])
        end = int(cu_seqlens_q[b + 1])
        q_len = end - start

        # Slice (H, q_len, max_seqlen_k)
        sl = S_dmask_hts[:, start:end, :]

        pad_q = seqlen_q_rounded - q_len
        pad_k = seqlen_k_rounded - sl.shape[2]
        # Pad with zeros; convert_flash_attn_S_to_softmax will mask via padding masks
        sl_padded = jnp.pad(sl, ((0, 0), (0, pad_q), (0, pad_k)), mode="constant")

        out = out.at[b].set(sl_padded)

    return out


def generate_random_padding_mask(
    max_seqlen: int,
    batch_size: int,
    key: jax.Array,
    mode: str = "random",
) -> jnp.ndarray:
    assert mode in ["full", "random", "third"]

    if mode == "full":
        lengths = jnp.full((batch_size,), max_seqlen, dtype=dtypes.i32)
    elif mode == "random":
        lo = max(1, max_seqlen - 20)
        hi = max_seqlen + 1
        lengths = jax.random.randint(key, (batch_size,), lo, hi, dtype=dtypes.i32)
    else:  # "third"
        lo = max_seqlen // 3
        hi = max_seqlen + 1
        lengths = jax.random.randint(key, (batch_size,), lo, hi, dtype=dtypes.i32)

    # (batch_size,) -> (batch_size, 1) for broadcasting
    lengths = lengths[:, None]

    ar = jnp.arange(max_seqlen, dtype=dtypes.i32)
    positions = repeat(ar, "s -> b s", b=batch_size)  # (batch_size, max_seqlen)

    padding_mask = positions < lengths  # (batch_size, max_seqlen), bool
    return padding_mask
