# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""BF16 sorted-sparse fmha_v4_fwd coverage, ported from aiter op_tests/test_mha_v4_sparse.py.

The Torch file sweeps every quantized recipe. This file keeps the BF16 sparse
contract: LUT ordering, all-true masks, tile selection, empty rows, and accuracy
as the KV tile count grows.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from jax_aiter.ja_compat.chip_info import get_gfx
from jax_aiter.ops.mha_v4 import mha_v4, mha_v4_kv_tile, mha_v4_q_tile, mha_v4_sparse_work_table


pytestmark = pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 BF16 fmha_v4_fwd sparse")


def _cosine(a, b):
    af = np.asarray(a, dtype=np.float32).ravel()
    bf = np.asarray(b, dtype=np.float32).ravel()
    denom = np.linalg.norm(af) * np.linalg.norm(bf)
    if denom == 0:
        return 1.0 if np.allclose(af, bf) else 0.0
    return float(np.dot(af, bf) / denom)


def _rand(key, shape):
    return jax.random.normal(key, shape, dtype=jnp.bfloat16)


def _masked_reference(q, k, v, block_mask):
    qf = np.asarray(q, dtype=np.float32)
    kf = np.asarray(k, dtype=np.float32)
    vf = np.asarray(v, dtype=np.float32)
    scale = qf.shape[-1] ** -0.5
    scores = np.einsum("bqhd,bkhd->bhqk", qf, kf) * scale
    mask = np.asarray(block_mask)
    if mask.ndim == 3:
        mask = np.broadcast_to(mask[:, None], (qf.shape[0], qf.shape[2], mask.shape[1], mask.shape[2]))
    q_tile, kv_tile = mha_v4_q_tile(), mha_v4_kv_tile()
    keep = np.repeat(mask, q_tile, axis=2)[:, :, : qf.shape[1], :]
    keep = np.repeat(keep, kv_tile, axis=3)[:, :, :, : kf.shape[1]]
    scores = np.where(keep, scores, -np.inf)
    finite = np.isfinite(scores)
    scores = np.where(finite, scores, 0)
    peak = np.max(np.where(finite, scores, -np.inf), axis=-1, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0)
    weights = np.exp(scores - peak) * finite
    denom = np.sum(weights, axis=-1, keepdims=True)
    weights = np.divide(weights, denom, out=np.zeros_like(weights), where=denom > 0)
    return np.einsum("bhqk,bkhd->bqhd", weights, vf)


def _tile_mask(heads, kv_tiles, tiles, q_tiles=1, batch=1):
    mask = np.zeros((batch, heads, q_tiles, kv_tiles), dtype=bool)
    for tile in tiles:
        mask[:, :, :, tile] = True
    return jnp.asarray(mask)


def _work_table_counts(total, pattern, seed=7):
    rng = np.random.default_rng(seed)
    if pattern == "uniform":
        counts = np.full((total,), 7, dtype=np.int32)
    elif pattern == "zeros":
        counts = np.zeros((total,), dtype=np.int32)
    elif pattern == "random":
        counts = rng.integers(0, 64, size=total, dtype=np.int32)
    elif pattern == "wide_random":
        counts = rng.integers(0, 8192, size=total, dtype=np.int32)
    elif pattern == "two_values":
        counts = np.where(np.arange(total) % 3 == 0, 9, 4).astype(np.int32)
    elif pattern == "descending":
        counts = np.arange(total, 0, -1, dtype=np.int32)
    else:
        counts = np.arange(1, total + 1, dtype=np.int32)
    return jnp.asarray(counts)


def _unpack(table, nhead, q_tiles):
    table = np.asarray(table, dtype=np.int32).astype(np.uint32)
    q_idx = table & np.uint32(0xFFFF)
    h_idx = (table >> np.uint32(16)) & np.uint32(0xFF)
    b_idx = table >> np.uint32(24)
    return ((b_idx.astype(np.int64) * nhead + h_idx.astype(np.int64)) * q_tiles + q_idx).astype(np.int64)


@pytest.mark.parametrize(
    ("batch", "nhead", "q_tiles"),
    [
        (1, 1, 1),
        (1, 8, 3),
        (2, 5, 7),
        (1, 16, 32),
        (1, 32, 32),
        (1, 5, 296),
        (4, 16, 64),
        (8, 16, 64),
        (8, 32, 64),
    ],
)
@pytest.mark.parametrize(
    "pattern",
    ["uniform", "zeros", "random", "wide_random", "two_values", "descending", "ascending"],
)
def test_mha_v4_sparse_work_table_is_longest_lut_first(batch, nhead, q_tiles, pattern):
    total = batch * nhead * q_tiles
    counts = _work_table_counts(total, pattern)
    table = mha_v4_sparse_work_table(counts, batch, nhead, q_tiles)
    visited = _unpack(table, nhead, q_tiles)
    assert np.array_equal(np.sort(visited), np.arange(total))
    ordered = np.asarray(counts)[visited]
    assert np.all(ordered[:-1] >= ordered[1:])
    expected = np.argsort(-np.asarray(counts), kind="stable")
    assert np.array_equal(visited, expected)


@pytest.mark.parametrize(
    "batch,nhead,q_tiles", [(1, 16, 32), (1, 5, 296), (8, 16, 64), (8, 32, 64)]
)
def test_mha_v4_sparse_work_table_leaves_uniform_counts_in_raster_order(batch, nhead, q_tiles):
    total = batch * nhead * q_tiles
    counts = jnp.full((total,), 5, dtype=jnp.int32)
    table = mha_v4_sparse_work_table(counts, batch, nhead, q_tiles)
    visited = _unpack(table, nhead, q_tiles)
    assert np.array_equal(visited, np.arange(total))


def test_mha_v4_sparse_all_true_mask_matches_dense():
    key = jax.random.PRNGKey(41)
    q = _rand(key, (1, 511, 5, 128))
    k = _rand(jax.random.fold_in(key, 1), (1, 512, 5, 128))
    v = _rand(jax.random.fold_in(key, 2), k.shape)
    mask = jnp.ones((1, 5, 2, 512 // mha_v4_kv_tile()), dtype=jnp.bool_)
    dense = mha_v4(q, k, v)
    sparse = mha_v4(q, k, v, block_mask=mask)
    assert _cosine(sparse, dense) > 0.99
    assert np.isfinite(np.asarray(sparse, dtype=np.float32)).all()


@pytest.mark.parametrize("tiles", [(0,), (1,), (3,), (0, 2), (1, 2, 3)])
def test_mha_v4_sparse_reads_only_the_kv_tiles_the_lut_names(tiles):
    heads = 2
    kv_tile = mha_v4_kv_tile()
    kv_tiles = 4
    key = jax.random.PRNGKey(3)
    q = _rand(key, (1, 128, heads, 128))
    k = _rand(jax.random.fold_in(key, 1), (1, kv_tiles * kv_tile, heads, 128))
    v = _rand(jax.random.fold_in(key, 2), k.shape)
    mask = _tile_mask(heads, kv_tiles, tiles)
    sparse = mha_v4(q, k, v, block_mask=mask)
    parts_k = [k[:, tile * kv_tile : (tile + 1) * kv_tile] for tile in tiles]
    parts_v = [v[:, tile * kv_tile : (tile + 1) * kv_tile] for tile in tiles]
    dense = mha_v4(q, jnp.concatenate(parts_k, axis=1), jnp.concatenate(parts_v, axis=1))
    assert _cosine(sparse, dense) > 0.99


def test_mha_v4_sparse_distinct_kv_tiles_give_distinct_results():
    heads = 2
    kv_tile = mha_v4_kv_tile()
    kv_tiles = 4
    key = jax.random.PRNGKey(5)
    q = _rand(key, (1, 64, heads, 128))
    k = _rand(jax.random.fold_in(key, 1), (1, kv_tiles * kv_tile, heads, 128))
    v = _rand(jax.random.fold_in(key, 2), k.shape)
    outputs = [
        np.asarray(mha_v4(q, k, v, block_mask=_tile_mask(heads, kv_tiles, (tile,))))
        for tile in range(kv_tiles)
    ]
    for tile in range(1, kv_tiles):
        assert not np.array_equal(outputs[0], outputs[tile])


def test_mha_v4_sparse_gives_each_head_its_own_kv_tiles():
    heads = 2
    kv_tile = mha_v4_kv_tile()
    key = jax.random.PRNGKey(9)
    q = _rand(key, (1, 64, heads, 128))
    k = _rand(jax.random.fold_in(key, 1), (1, 4 * kv_tile, heads, 128))
    v = _rand(jax.random.fold_in(key, 2), k.shape)
    mask = jnp.zeros((1, heads, 1, 4), dtype=jnp.bool_)
    mask = mask.at[:, 0, :, 0].set(True)
    mask = mask.at[:, 1, :, 1].set(True)
    out = mha_v4(q, k, v, block_mask=mask)
    head0 = mha_v4(q[:, :, :1], k[:, :, :1], v[:, :, :1], block_mask=_tile_mask(1, 4, (0,)))
    head1 = mha_v4(q[:, :, 1:], k[:, :, 1:], v[:, :, 1:], block_mask=_tile_mask(1, 4, (1,)))
    assert _cosine(out[:, :, :1], head0) > 0.99
    assert _cosine(out[:, :, 1:], head1) > 0.99


def test_mha_v4_sparse_follows_the_lut_across_query_tiles():
    kv_tile = mha_v4_kv_tile()
    q_tile = mha_v4_q_tile()
    key = jax.random.PRNGKey(11)
    q = _rand(key, (1, q_tile * 2, 2, 128))
    k = _rand(jax.random.fold_in(key, 1), (1, kv_tile * 2, 2, 128))
    v = _rand(jax.random.fold_in(key, 2), k.shape)
    mask = jnp.zeros((1, 2, 2, 2), dtype=jnp.bool_)
    mask = mask.at[:, :, 0, 0].set(True)
    mask = mask.at[:, :, 1, 1].set(True)
    out = np.asarray(mha_v4(q, k, v, block_mask=mask), dtype=np.float32)
    first = _masked_reference(q[:, :q_tile], k, v, mask[:, :, :1])
    second = _masked_reference(q[:, q_tile:], k, v, mask[:, :, 1:])
    assert _cosine(out[:, :q_tile], first) > 0.99
    assert _cosine(out[:, q_tile:], second) > 0.99


@pytest.mark.parametrize("tail_rows", [64, 128, 200])
def test_mha_v4_sparse_partial_query_tile_follows_the_lut(tail_rows):
    q_tile = mha_v4_q_tile()
    kv_tile = mha_v4_kv_tile()
    sq = q_tile + tail_rows
    key = jax.random.PRNGKey(13 + tail_rows)
    q = _rand(key, (1, sq, 2, 128))
    k = _rand(jax.random.fold_in(key, 1), (1, kv_tile * 3, 2, 128))
    v = _rand(jax.random.fold_in(key, 2), k.shape)
    mask = _tile_mask(2, 3, (1,), q_tiles=2)
    out = mha_v4(q, k, v, block_mask=mask)
    ref = _masked_reference(q, k, v, mask)
    assert _cosine(out, ref) > 0.99


def test_mha_v4_sparse_empty_row_writes_zeros():
    heads = 2
    kv_tile = mha_v4_kv_tile()
    kv_tiles = 4
    key = jax.random.PRNGKey(0)
    q = _rand(key, (1, 256, heads, 128))
    k = _rand(jax.random.fold_in(key, 1), (1, kv_tiles * kv_tile, heads, 128))
    v = _rand(jax.random.fold_in(key, 2), k.shape)
    mask = jnp.zeros((1, heads, 1, kv_tiles), dtype=jnp.bool_)
    mask = mask.at[:, 0, :, 0].set(True)
    mask = mask.at[:, 0, :, 1].set(True)
    out = np.asarray(mha_v4(q, k, v, block_mask=mask))
    assert np.array_equal(out[:, :, 1], np.zeros_like(out[:, :, 1]))
    assert np.isfinite(out.astype(np.float32)).all()
    assert np.max(np.abs(out[:, :, 0].astype(np.float32))) > 0

    populated = mask.at[:, 1, :, 0].set(True)
    again = np.asarray(mha_v4(q, k, v, block_mask=populated))
    assert np.array_equal(out[:, :, 0], again[:, :, 0])


@pytest.mark.parametrize("tiles", [2, 4, 8, 16])
def test_mha_v4_sparse_accuracy_holds_as_tile_count_grows(tiles):
    kv_tile = mha_v4_kv_tile()
    key = jax.random.PRNGKey(41 + tiles)
    q = _rand(key, (1, 256, 5, 128))
    k = _rand(jax.random.fold_in(key, 1), (1, tiles * kv_tile, 5, 128))
    v = _rand(jax.random.fold_in(key, 2), k.shape)
    q_tiles = (q.shape[1] + mha_v4_q_tile() - 1) // mha_v4_q_tile()
    mask = jnp.ones((1, 5, q_tiles, tiles), dtype=jnp.bool_)
    out = mha_v4(q, k, v, block_mask=mask)
    ref = _masked_reference(q, k, v, mask)
    assert np.isfinite(np.asarray(out, dtype=np.float32)).all()
    assert _cosine(out, ref) > 0.95


def test_mha_v4_sparse_skipping_lut_holds_as_tile_count_grows():
    kv_tile = mha_v4_kv_tile()
    tiles = 8
    key = jax.random.PRNGKey(17)
    q = _rand(key, (1, 256, 4, 128))
    k = _rand(jax.random.fold_in(key, 1), (1, tiles * kv_tile, 4, 128))
    v = _rand(jax.random.fold_in(key, 2), k.shape)
    mask = jnp.zeros((1, 4, 1, tiles), dtype=jnp.bool_)
    mask = mask.at[:, :, :, ::2].set(True)
    out = mha_v4(q, k, v, block_mask=mask)
    ref = _masked_reference(q, k, v, mask)
    assert _cosine(out, ref) > 0.95


def test_mha_v4_rejects_bad_block_mask():
    q = jnp.zeros((1, 256, 2, 128), dtype=jnp.bfloat16)
    k = jnp.zeros((1, mha_v4_kv_tile(), 2, 128), dtype=jnp.bfloat16)
    bad = jnp.ones((1, 2, 1, 1), dtype=jnp.int32)
    with pytest.raises(ValueError, match="bool"):
        mha_v4(q, k, k, block_mask=bad)
    wrong = jnp.ones((1, 2, 2, 1), dtype=jnp.bool_)
    with pytest.raises(ValueError, match="block_mask shape"):
        mha_v4(q, k, k, block_mask=wrong)
    short = jnp.zeros((1, 64, 2, 128), dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match="multiple"):
        mha_v4(q, short, short, block_mask=jnp.ones((1, 2, 1, 1), dtype=jnp.bool_))
