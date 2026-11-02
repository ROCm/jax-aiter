# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""FMHA v4 forward on the gfx950 ASM code objects.

The code objects live in ``hsa/<arch>/fmha_v4_fwd``. jax-aiter's other MHA
entry points (``flash_attn_func``) still go through CK / ASM v3. This module
launches the BF16 dense and sorted-sparse rows, plus the MXFP4 and MXFP6 dense
rows at head dims 128 and 256. The MX rows accept per-batch key lengths and
can return the log-sum-exp of the quantized scores, with the K-smoothing shift
restored.

MXFP4 and MXFP6 use the FMHA producers (Hadamard Q, coalesced K, FP6-P V), not
the GEMM quantization layouts. FP8 and INT8 rows are still not launched.
"""

from __future__ import annotations

import functools
import os

import jax
import jax.numpy as jnp
import numpy as np

from ..ffi.registry import register_ffi_target
from ..ja_compat.chip_info import get_gfx
from ..ja_compat.config import set_aiter_asm_dir

_Q_FORMAT_BF16 = 2


def _ensure(target: str) -> None:
    register_ffi_target(target, "ROCM")


def _manifest_path() -> str:
    set_aiter_asm_dir()
    asm = os.environ.get("AITER_ASM_DIR")
    if not asm:
        raise RuntimeError("AITER_ASM_DIR is not set; cannot find fmha_v4_fwd")
    return os.path.join(asm, get_gfx(), "fmha_v4_fwd", "fmha_v4_fwd.csv")


@functools.lru_cache(maxsize=None)
def _bf16_tiles(mode: int) -> tuple[int, int]:
    """Return (query tile, KV tile) for the BF16 manifest row at ``mode``."""
    path = _manifest_path()
    header = None
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            cols = [part.strip() for part in line.split(",")]
            if header is None:
                header = cols
                continue
            row = dict(zip(header, cols))
            if (
                int(row["q_format"]) == _Q_FORMAT_BF16
                and int(row["k_format"]) == _Q_FORMAT_BF16
                and int(row["v_format"]) == _Q_FORMAT_BF16
                and int(row["mode"]) == mode
            ):
                return int(row["ts_qo"]), int(row["ts_kv"])
    raise RuntimeError(f"no BF16 fmha_v4_fwd row with mode={mode} in {path}")


def mha_v4_q_tile() -> int:
    """Query tile of the BF16 sorted-sparse row (256 on gfx950)."""
    return _bf16_tiles(1)[0]


def mha_v4_kv_tile() -> int:
    """KV tile of the BF16 sorted-sparse row (128 on gfx950, 64 on gfx942)."""
    return _bf16_tiles(1)[1]


def _as_bf16(x, name: str) -> jax.Array:
    arr = jnp.asarray(x)
    if arr.dtype != jnp.bfloat16:
        raise ValueError(f"{name} must be bfloat16, got {arr.dtype}")
    if arr.ndim != 4:
        raise ValueError(f"{name} must be BSHD, got rank {arr.ndim}")
    return arr


def _check_qkv(q, k, v, *, head_dims=(128,)):
    q = _as_bf16(q, "q")
    k = _as_bf16(k, "k")
    v = _as_bf16(v, "v")
    head_dim = q.shape[-1]
    if head_dim not in head_dims or k.shape[-1] != head_dim or v.shape[-1] != head_dim:
        allowed = ", ".join(str(dim) for dim in head_dims)
        raise ValueError(f"MHA v4 head dim must be one of ({allowed})")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("q, k, v batch sizes must match")
    if k.shape[1] != v.shape[1] or k.shape[2] != v.shape[2]:
        raise ValueError("k and v sequence and head counts must match")
    hq, hk = q.shape[2], k.shape[2]
    if hk <= 0 or hq % hk != 0:
        raise ValueError("query heads must be divisible by KV heads")
    ratio = hq // hk
    if ratio > 16 or (ratio & (ratio - 1)) != 0:
        raise ValueError(f"GQA ratio must be a power of two <= 16, got {ratio}")
    if min(q.shape[0], q.shape[1], k.shape[1], hq) <= 0:
        raise ValueError("MHA v4 requires non-empty inputs")
    return q, k, v


def _scale_of(q, softmax_scale):
    if softmax_scale is None:
        return float(q.shape[-1]) ** -0.5
    return float(softmax_scale)


@functools.lru_cache(maxsize=None)
def _dense_call(out_shape, lse_shape):
    _ensure("MhaV4FwdJA")
    return jax.ffi.ffi_call(
        "MhaV4FwdJA",
        (
            jax.ShapeDtypeStruct(out_shape, jnp.bfloat16),
            jax.ShapeDtypeStruct(lse_shape, jnp.float32),
        ),
        has_side_effect=False,
    )


@functools.lru_cache(maxsize=None)
def _sparse_call(out_shape, table_len):
    _ensure("MhaV4SparseFwdJA")
    return jax.ffi.ffi_call(
        "MhaV4SparseFwdJA",
        (
            jax.ShapeDtypeStruct(out_shape, jnp.bfloat16),
            jax.ShapeDtypeStruct((table_len,), jnp.int32),
        ),
        has_side_effect=False,
    )


@functools.lru_cache(maxsize=None)
def _work_call(total):
    _ensure("MhaV4SparseWorkTableJA")
    return jax.ffi.ffi_call(
        "MhaV4SparseWorkTableJA",
        jax.ShapeDtypeStruct((total,), jnp.int32),
        has_side_effect=False,
    )


def _empty(dtype):
    return jnp.zeros((0,), dtype=dtype)


def block_mask_to_lut(block_mask, num_heads: int | None = None):
    """Ragged KV-block LUT for a boolean tile mask.

    ``block_mask`` is ``[B, H, Qtiles, KVtiles]`` or ``[B, Qtiles, KVtiles]``
    (broadcast across heads). True means the query tile may attend that KV tile.
    Returns ``(kv_block_indices, lut_start, lut_count)``, all int32. An empty
    row has ``lut_count == 0`` and ``lut_start == 0``. The index buffer is
    padded by one so a speculative read stays in range.
    """
    mask = jnp.asarray(block_mask)
    if mask.dtype != jnp.bool_:
        raise ValueError(f"block_mask must be bool, got {mask.dtype}")
    if mask.ndim == 3:
        if num_heads is None:
            raise ValueError("num_heads is required for a 3D block_mask")
        batch, q_tiles, kv_tiles = mask.shape
        mask = jnp.broadcast_to(mask[:, None, :, :], (batch, num_heads, q_tiles, kv_tiles))
    elif mask.ndim != 4:
        raise ValueError("block_mask must be 3D or 4D")

    batch, heads, q_tiles, kv_tiles = mask.shape
    counts = mask.sum(axis=-1).astype(jnp.int32)
    flat_counts = counts.reshape(-1)
    exclusive = jnp.cumsum(flat_counts) - flat_counts
    lut_start = jnp.where(flat_counts == 0, jnp.int32(0), exclusive).astype(jnp.int32)

    within = jnp.cumsum(mask.astype(jnp.int32), axis=-1) - 1
    write_at = lut_start.reshape(batch, heads, q_tiles, 1) + within
    tiles = jnp.broadcast_to(jnp.arange(kv_tiles, dtype=jnp.int32), mask.shape)
    flat_mask = mask.reshape(-1)
    n = batch * heads * q_tiles * kv_tiles
    buf_len = max(n, 1) + 1
    drop = jnp.int32(buf_len)
    idx = jnp.where(flat_mask, write_at.reshape(-1).astype(jnp.int32), drop)
    buf = jnp.zeros((buf_len,), dtype=jnp.int32)
    buf = buf.at[idx].set(tiles.reshape(-1).astype(jnp.int32), mode="drop")
    return buf, lut_start, flat_counts


def _normalize_mask(block_mask, q, k):
    q_tile = mha_v4_q_tile()
    kv_tile = mha_v4_kv_tile()
    if k.shape[1] % kv_tile != 0:
        raise ValueError(
            f"sorted-sparse MHA v4 requires the key length to be a multiple of {kv_tile}"
        )
    q_tiles = (q.shape[1] + q_tile - 1) // q_tile
    kv_tiles = k.shape[1] // kv_tile
    mask = jnp.asarray(block_mask)
    if mask.dtype != jnp.bool_:
        raise ValueError(f"block_mask must be bool, got {mask.dtype}")
    if mask.ndim == 4:
        expected = (q.shape[0], q.shape[2], q_tiles, kv_tiles)
    elif mask.ndim == 3:
        expected = (q.shape[0], q_tiles, kv_tiles)
    else:
        raise ValueError("block_mask must be 3D [B, Qtiles, KVtiles] or 4D [B, H, Qtiles, KVtiles]")
    if tuple(mask.shape) != expected:
        raise ValueError(f"block_mask shape {tuple(mask.shape)} != {expected}")
    indices, start, count = block_mask_to_lut(
        mask, num_heads=q.shape[2] if mask.ndim == 3 else None
    )
    return indices, start, count


def mha_v4(q, k, v, *, softmax_scale=None, return_lse=False, block_mask=None, seqlens_k=None):
    """BF16 non-causal FMHA v4.

    Q, K, V are BSHD bfloat16 with head dim 128. K and V may have fewer heads
    than Q; the ratio must be a power of two up to 16.

    ``block_mask`` selects the sorted-sparse code object. It is a bool tensor
    of shape ``[B, H, Qtiles, KVtiles]`` or ``[B, Qtiles, KVtiles]``. Tile
    geometry comes from the manifest (256 x 128 on gfx950).

    ``seqlens_k`` is an int32 ``[B]`` of valid key lengths for the dense kernel.
    The sparse path does not accept it. ``return_lse`` is dense-only and adds
    a float32 ``[B, H, Sq]`` log-sum-exp.
    """
    q, k, v = _check_qkv(q, k, v)
    scale = _scale_of(q, softmax_scale)
    if block_mask is not None and seqlens_k is not None:
        raise NotImplementedError("sorted-sparse MHA v4 does not accept per-batch key lengths")
    if block_mask is not None and return_lse:
        raise NotImplementedError("sorted-sparse MHA v4 does not return LSE")

    if block_mask is None:
        if seqlens_k is None:
            seqlens = _empty(jnp.int32)
        else:
            seqlens = jnp.asarray(seqlens_k)
            if seqlens.dtype != jnp.int32 or seqlens.ndim != 1 or seqlens.shape[0] < q.shape[0]:
                raise ValueError("seqlens_k must be int32 [batch]")
        lse_shape = (q.shape[0], q.shape[2], q.shape[1]) if return_lse else (0,)
        out, lse = _dense_call(tuple(q.shape), lse_shape)(
            q, k, v, seqlens, softmax_scale=np.float32(scale)
        )
        if return_lse:
            return out, lse
        return out

    indices, start, count = _normalize_mask(block_mask, q, k)
    out, _table = _sparse_call(tuple(q.shape), int(count.shape[0]))(
        q, k, v, indices, start, count, softmax_scale=np.float32(scale)
    )
    return out


_LOG2E = 1.4426950408889634
_K_SMOOTH_SAMPLE_ROWS = 2048
_K_SMOOTH_MIN_COMMON = 0.85


def _q_scale_bytes(batch, sequence, heads):
    logical = batch * sequence * heads * 4
    padded = ((sequence + 255) // 256) * 256
    return logical + (padded - sequence) * heads * 4


def _k_raw_bytes(batch, sequence, heads):
    return batch * heads * ((sequence + 127) // 128) * 8192


def _k_scale_bytes(batch, sequence, heads):
    logical = batch * sequence * heads * 4
    padded = ((sequence + 127) // 128) * 128 + 256
    return logical + (padded - sequence) * heads * 4 + 4


def _v_raw_bytes(batch, sequence, heads):
    return batch * heads * ((sequence + 127) // 128) * 8192 + 64


def _v_scale_bytes(batch, sequence, heads):
    return batch * heads * ((sequence + 127) // 128) * 512 + 1024


@functools.lru_cache(maxsize=None)
def _mxfp4_q_call(shape):
    batch, sequence, heads, _ = shape
    _ensure("MhaV4Mxfp4QuantQJA")
    return jax.ffi.ffi_call(
        "MhaV4Mxfp4QuantQJA",
        (
            jax.ShapeDtypeStruct((batch, sequence, heads, 64), jnp.uint8),
            jax.ShapeDtypeStruct((_q_scale_bytes(batch, sequence, heads),), jnp.uint8),
        ),
        has_side_effect=False,
    )


@functools.lru_cache(maxsize=None)
def _mxfp4_k_call(shape):
    batch, sequence, heads = shape
    _ensure("MhaV4Mxfp4QuantKJA")
    return jax.ffi.ffi_call(
        "MhaV4Mxfp4QuantKJA",
        (
            jax.ShapeDtypeStruct((_k_raw_bytes(batch, sequence, heads),), jnp.uint8),
            jax.ShapeDtypeStruct((_k_scale_bytes(batch, sequence, heads),), jnp.uint8),
        ),
        has_side_effect=False,
    )


@functools.lru_cache(maxsize=None)
def _mxfp4_v_call(shape):
    batch, sequence, heads = shape
    _ensure("MhaV4Mxfp4QuantVJA")
    return jax.ffi.ffi_call(
        "MhaV4Mxfp4QuantVJA",
        (
            jax.ShapeDtypeStruct((_v_raw_bytes(batch, sequence, heads),), jnp.uint8),
            jax.ShapeDtypeStruct((_v_scale_bytes(batch, sequence, heads),), jnp.uint8),
        ),
        has_side_effect=False,
    )


@functools.lru_cache(maxsize=None)
def _mxfp4_fwd_call(batch, sequence, heads, head_dim, lse_shape):
    _ensure("MhaV4Mxfp4FwdJA")
    return jax.ffi.ffi_call(
        "MhaV4Mxfp4FwdJA",
        (
            jax.ShapeDtypeStruct((batch, sequence, heads, head_dim), jnp.bfloat16),
            jax.ShapeDtypeStruct(lse_shape, jnp.float32),
        ),
        has_side_effect=False,
    )


@functools.lru_cache(maxsize=None)
def _mxfp6_q_call(shape):
    batch, sequence, heads, _ = shape
    _ensure("MhaV4Mxfp6QuantQJA")
    return jax.ffi.ffi_call(
        "MhaV4Mxfp6QuantQJA",
        (
            jax.ShapeDtypeStruct((batch, sequence, heads, 192), jnp.uint8),
            jax.ShapeDtypeStruct((batch, sequence, heads, 8), jnp.uint8),
        ),
        has_side_effect=False,
    )


@functools.lru_cache(maxsize=None)
def _mxfp6_k_call(batch, sequence, heads):
    _ensure("MhaV4Mxfp6QuantKJA")
    tiles = (sequence + 127) // 128
    raw_bytes = batch * heads * 2 * tiles * 13312 + 256
    scale_bytes = batch * sequence * heads * 8 + 64
    return jax.ffi.ffi_call(
        "MhaV4Mxfp6QuantKJA",
        (
            jax.ShapeDtypeStruct((raw_bytes,), jnp.uint8),
            jax.ShapeDtypeStruct((scale_bytes,), jnp.uint8),
        ),
        has_side_effect=False,
    )


@functools.lru_cache(maxsize=None)
def _mxfp6_v_call(batch, sequence, heads):
    _ensure("MhaV4Mxfp6QuantVJA")
    tiles = (sequence + 127) // 128
    raw_bytes = batch * heads * tiles * 24576 + 256
    scale_bytes = batch * heads * tiles * 1024
    return jax.ffi.ffi_call(
        "MhaV4Mxfp6QuantVJA",
        (
            jax.ShapeDtypeStruct((raw_bytes,), jnp.uint8),
            jax.ShapeDtypeStruct((scale_bytes,), jnp.uint8),
        ),
        has_side_effect=False,
    )


@functools.lru_cache(maxsize=None)
def _mxfp6_fwd_call(batch, sequence, heads, lse_shape):
    _ensure("MhaV4Mxfp6FwdJA")
    return jax.ffi.ffi_call(
        "MhaV4Mxfp6FwdJA",
        (
            jax.ShapeDtypeStruct((batch, sequence, heads, 256), jnp.bfloat16),
            jax.ShapeDtypeStruct(lse_shape, jnp.float32),
        ),
        has_side_effect=False,
    )


def mha_v4_k_smooth(k):
    """Per-(batch, head, channel) K mean used by the MXFP4 K producer.

    Matches aiter's gate: a strided sample, kept only when that shared component
    is large relative to the sample RMS. Otherwise the returned mean is zero.
    """
    k = _as_bf16(k, "k")
    stride = max(1, k.shape[1] // _K_SMOOTH_SAMPLE_ROWS)
    sample = k[:, ::stride].astype(jnp.float32)
    mean = sample.mean(axis=1)
    row_sq = jnp.square(sample).sum(axis=-1).mean(axis=1)
    gate = jnp.square(mean).sum(axis=-1) > (_K_SMOOTH_MIN_COMMON**2) * row_sq
    return mean * gate[..., None]


def _half_heads(x):
    """View ``[B, S, H, 256]`` as the ``[B, S, 2H, 128]`` the hd128 producers accept."""
    batch, sequence, heads, _ = x.shape
    return x.reshape(batch, sequence, heads * 2, 128)


def _permute_k_hd256(raw, batch, heads, sequence):
    """Reorder an hd128 K buffer into the hd256 tile the ASM kernel gathers."""
    tiles = (sequence + 127) // 128
    view = raw.reshape(batch, heads, 2, tiles, 4, 2, 64, 16)
    return jnp.transpose(view, (0, 1, 3, 5, 2, 4, 6, 7)).reshape(-1)


def _permute_v_hd256(raw, scale, batch, heads, sequence):
    """Reorder hd128 V data and scales, then restore the zeroed gather slack."""
    tiles = (sequence + 127) // 128
    payload = batch * heads * 2 * tiles * 8192
    elements = batch * heads * 2 * tiles * 512
    data = raw[:payload].reshape(batch, heads, 2, tiles, 4, 2, 1024)
    data = jnp.transpose(data, (0, 1, 3, 5, 2, 4, 6)).reshape(-1)
    data = jnp.pad(data, (0, 128))
    scales = scale[:elements].reshape(batch, heads, 2, tiles, 2, 256)
    scales = jnp.transpose(scales, (0, 1, 3, 4, 2, 5)).reshape(-1)
    return data, jnp.pad(scales, (0, 1024))


def mha_v4_mxfp4_quantize_q(q, multiplier):
    """Rotate and pack BSHD Q to MXFP4 plus E8M0 scales.

    Head dim 128 packs to ``[B, S, H, 64]``. Head dim 256 runs the same kernel
    on each 128-wide half and packs to ``[B, S, H, 128]``.
    """
    q = _as_bf16(q, "q")
    head_dim = q.shape[-1]
    if head_dim == 128:
        return _mxfp4_q_call(tuple(q.shape))(q, multiplier=np.float32(multiplier))
    if head_dim != 256:
        raise ValueError("MXFP4 Q head dim must be 128 or 256")
    batch, sequence, heads, _ = q.shape
    packed, scale = _mxfp4_q_call((batch, sequence, heads * 2, 128))(
        _half_heads(q), multiplier=np.float32(multiplier)
    )
    return packed.reshape(batch, sequence, heads, 128), scale


def mha_v4_mxfp4_quantize_k(k, mean=None):
    """Rotate and pack BSHD K into the coalesced MXFP4 tile buffer.

    ``mean`` matches K's head layout: ``[B, H, 128]`` or ``[B, H, 256]``. Head
    dim 256 permutes the hd128 tile into the order ``fwd_hd256_mxfp4.co`` reads.
    """
    k = _as_bf16(k, "k")
    head_dim = k.shape[-1]
    if head_dim not in (128, 256):
        raise ValueError("MXFP4 K head dim must be 128 or 256")
    batch, sequence, heads, _ = k.shape
    if mean is None:
        mean = _empty(jnp.float32)
    else:
        mean = jnp.asarray(mean, dtype=jnp.float32)
        if head_dim == 256:
            mean = jnp.reshape(mean, (batch, heads * 2, 128))
    source = k if head_dim == 128 else _half_heads(k)
    quant_heads = heads if head_dim == 128 else heads * 2
    raw, scale = _mxfp4_k_call((batch, sequence, quant_heads))(source, mean)
    if head_dim == 256:
        raw = _permute_k_hd256(raw, batch, heads, sequence)
    return raw, scale


def mha_v4_mxfp4_quantize_v(v):
    """Pack BSHD V into the FP6-P MXFP4 tile order.

    Head dim 128 feeds ``fwd_hd128_mxfp4.co``. Head dim 256 permutes that tile
    into the order ``fwd_hd256_mxfp4.co`` reads.
    """
    v = _as_bf16(v, "v")
    head_dim = v.shape[-1]
    if head_dim not in (128, 256):
        raise ValueError("MXFP4 V head dim must be 128 or 256")
    batch, sequence, heads, _ = v.shape
    if head_dim == 128:
        return _mxfp4_v_call((batch, sequence, heads))(v)
    raw, scale = _mxfp4_v_call((batch, sequence, heads * 2))(_half_heads(v))
    return _permute_v_hd256(raw, scale, batch, heads, sequence)


def _restore_k_mean_in_lse(lse, q, k_mean, softmax_scale):
    """Add back the score shift removed by K smoothing.

    The kernel attends over ``k - k_mean``. A shift shared by every key cancels
    in the softmax, so the output is unchanged, but the log-sum-exp moves by
    ``softmax_scale * q · k_mean``. Query head ``j`` reads KV head ``j // ratio``.
    """
    heads_q, heads_k = q.shape[2], k_mean.shape[1]
    mean = k_mean if heads_q == heads_k else jnp.repeat(k_mean, heads_q // heads_k, axis=1)
    shift = jnp.einsum("bshd,bhd->bhs", q.astype(jnp.float32), mean.astype(jnp.float32))
    return lse + shift * jnp.float32(softmax_scale)


def _seqlens_arg(seqlens_k, batch):
    if seqlens_k is None:
        return _empty(jnp.int32)
    seqlens = jnp.asarray(seqlens_k)
    if seqlens.dtype != jnp.int32 or seqlens.ndim != 1 or seqlens.shape[0] < batch:
        raise ValueError("seqlens_k must be int32 [batch]")
    return seqlens


def mha_v4_mxfp4(q, k, v, *, softmax_scale=None, return_lse=False, seqlens_k=None):
    """Dense non-causal MXFP4 FMHA v4. Output is bfloat16 BSHD.

    Head dim 128 launches ``fwd_hd128_mxfp4.co``. Head dim 256 launches
    ``fwd_hd256_mxfp4.co`` after the half-head producers and their permute.
    Q is premultiplied by ``softmax_scale * log2(e)`` inside the quantizer.
    K uses the smoothed mean above. V is packed for the FP6-P operand.

    ``seqlens_k`` is an int32 ``[B]`` of valid key lengths. The kernel clamps
    each batch to that length inside the padded K/V buffer. ``return_lse`` adds
    a float32 ``[B, H, Sq]`` log-sum-exp of the quantized scores, with the
    K-smoothing shift restored. It is safe for merging chunks from this same
    recipe, and it is not an exact-math log-sum-exp.
    """
    q, k, v = _check_qkv(q, k, v, head_dims=(128, 256))
    scale = _scale_of(q, softmax_scale)
    seqlens = _seqlens_arg(seqlens_k, q.shape[0])
    k_mean = mha_v4_k_smooth(k)
    packed_q, q_scale = mha_v4_mxfp4_quantize_q(q, scale * _LOG2E)
    raw_k, k_scale = mha_v4_mxfp4_quantize_k(k, k_mean)
    raw_v, v_scale = mha_v4_mxfp4_quantize_v(v)
    lse_shape = (q.shape[0], q.shape[2], q.shape[1]) if return_lse else (0,)
    out, lse = _mxfp4_fwd_call(q.shape[0], q.shape[1], q.shape[2], q.shape[-1], lse_shape)(
        packed_q,
        raw_k,
        raw_v,
        q_scale,
        k_scale,
        v_scale,
        seqlens,
        seq_k=np.int32(k.shape[1]),
        heads_k=np.int32(k.shape[2]),
        softmax_scale=np.float32(scale),
        head_dim=np.int32(q.shape[-1]),
    )
    if return_lse:
        return out, _restore_k_mean_in_lse(lse, q, k_mean, scale)
    return out


def mha_v4_mxfp6(q, k, v, *, softmax_scale=None, return_lse=False, seqlens_k=None):
    """Dense non-causal HD256 MXFP6 FMHA v4. Output is bfloat16 BSHD.

    This launches AITER's gfx950 ``fwd_hd256_mxfp6.co`` row. Its Q/K/V
    producers operate on a two-half-head HD128 view with ``head_split=2``,
    which writes the HD256 kernel's native 64-token records directly.
    """
    q, k, v = _check_qkv(q, k, v, head_dims=(256,))
    scale = _scale_of(q, softmax_scale)
    seqlens = _seqlens_arg(seqlens_k, q.shape[0])
    k_mean = mha_v4_k_smooth(k)
    packed_q, q_scale = _mxfp6_q_call(tuple(q.shape))(q, multiplier=np.float32(scale * _LOG2E))
    raw_k, k_scale = _mxfp6_k_call(q.shape[0], k.shape[1], k.shape[2])(k, k_mean)
    raw_v, v_scale = _mxfp6_v_call(q.shape[0], v.shape[1], v.shape[2])(v)
    lse_shape = (q.shape[0], q.shape[2], q.shape[1]) if return_lse else (0,)
    out, lse = _mxfp6_fwd_call(q.shape[0], q.shape[1], q.shape[2], lse_shape)(
        packed_q,
        raw_k,
        raw_v,
        q_scale,
        k_scale,
        v_scale,
        seqlens,
        seq_k=np.int32(k.shape[1]),
        heads_k=np.int32(k.shape[2]),
        softmax_scale=np.float32(scale),
    )
    if return_lse:
        return out, _restore_k_mean_in_lse(lse, q, k_mean, scale)
    return out


def mha_v4_sparse_work_table(lut_count, batch: int, nhead: int, q_tiles: int):
    """Pack the longest-LUT-first work table the sparse kernel consumes.

    ``lut_count`` is int32 of length ``batch * nhead * q_tiles``. Ties keep
    raster order. Tables up to 8192 entries are ranked on device; larger
    tables use a host stable sort.
    """
    counts = jnp.asarray(lut_count)
    total = int(batch) * int(nhead) * int(q_tiles)
    if counts.dtype != jnp.int32 or counts.ndim != 1 or counts.shape[0] != total:
        raise ValueError("lut_count must be int32 [batch * nhead * q_tiles]")
    if batch <= 0 or nhead <= 0 or q_tiles <= 0 or batch >= 256 or nhead >= 256 or q_tiles >= 65536:
        raise ValueError("work table requires 0 < batch,heads < 256 and 0 < q_tiles < 65536")
    return _work_call(total)(
        counts,
        batch=np.int32(batch),
        nhead=np.int32(nhead),
        q_tiles=np.int32(q_tiles),
    )
