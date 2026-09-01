# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""M2a: the neutral vocabulary driving real kernels, end to end.

The other M2 suites build their page tables by hand in numpy, which tests the
kernels but says nothing about the two-layer split: the neutral types and the
vendor ABI could each be internally consistent and still disagree with each
other. This is the test that makes them meet. A real ``KvStorageLayoutV1`` and
``KvPageTableV1`` are converted by :mod:`jax_aiter.kv.bridge` and used to drive
append, prefill and decode over one pool, against a dense reference.

The neutral modules are loaded straight from their files rather than imported as
``maxtext.inference.kv_common``, for the reason the MaxText suite documents:
importing them by package path executes ``maxtext/__init__.py`` and pulls in the
whole config stack. Loading the files keeps this suite honest about the layer
being framework-free, and it is also the only way it runs here, since jax-aiter
must not depend on a front end.

Set ``MAXTEXT_ROOT`` if the checkout is not the sibling directory.
"""

import importlib.util
import math
import os
import pathlib
import sys

import numpy as np
import pytest

import jax.numpy as jnp

from jax_aiter.ffi.registry import standalone_symbol_available
from jax_aiter.kv.abi import AiterPagedAttentionAbiV1
from jax_aiter.kv import bridge

TOKENS_PER_PAGE = 16
HEAD_DIM = 128
NUM_PAGES = 256
NUM_KV_HEADS = 8
NUM_Q_HEADS = 8
DTYPE = jnp.bfloat16


def _kv_common_dir() -> pathlib.Path:
    root = os.environ.get("MAXTEXT_ROOT")
    if root:
        base = pathlib.Path(root)
    else:
        base = pathlib.Path(__file__).resolve().parents[2] / "maxtext"
    return base / "src" / "maxtext" / "inference" / "kv_common"


def _load_neutral_types():
    directory = _kv_common_dir()
    loaded = {}
    for name in ("storage_layout", "page_table"):
        path = directory / f"{name}.py"
        if not path.exists():
            return None
        spec = importlib.util.spec_from_file_location(f"_kv_bridge_{name}", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    return (
        loaded["storage_layout"].KvStorageLayoutV1,
        loaded["page_table"].KvPageTableV1,
    )


_NEUTRAL = _load_neutral_types()

pytestmark = [
    pytest.mark.skipif(
        _NEUTRAL is None,
        reason=f"neutral KV types not found under {_kv_common_dir()}; set MAXTEXT_ROOT",
    ),
    pytest.mark.skipif(
        not standalone_symbol_available("AppendKvJA")
        or not standalone_symbol_available("PagedAttentionJA")
        or not standalone_symbol_available("PagedPrefillJA"),
        reason="paged-KV FFI modules not built (run 'make -f Makefile.kv ja_kv')",
    ),
]


def _layout(**overrides):
    KvStorageLayoutV1, _ = _NEUTRAL
    kwargs = dict(
        tokens_per_page=TOKENS_PER_PAGE,
        num_pages=NUM_PAGES,
        num_layers=1,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        dtype="bfloat16",
    )
    kwargs.update(overrides)
    return KvStorageLayoutV1(**kwargs)


class _Allocator:
    """Hands out shuffled page ids, never the padding sentinel."""

    def __init__(self, seed=0, padding_page_id=0):
        rng = np.random.default_rng(seed)
        pages = [p for p in rng.permutation(NUM_PAGES) if p != padding_page_id]
        self._free = [int(p) for p in pages]

    def take(self, n):
        return [self._free.pop() for _ in range(n)]


def _page_table(page_ids, seq_lens, query_lens):
    _, KvPageTableV1 = _NEUTRAL
    write_positions = np.concatenate(
        [
            np.arange(seq - q, seq, dtype=np.int32)
            for seq, q in zip(seq_lens, query_lens)
        ]
    ).astype(np.int32)
    return KvPageTableV1(
        page_ids=page_ids,
        seq_lens=np.asarray(seq_lens, dtype=np.int32),
        query_lens=np.asarray(query_lens, dtype=np.int32),
        write_positions=write_positions,
        request_order=np.arange(len(seq_lens), dtype=np.int32),
    )


def _dense_attention(q, k_hist, v_hist, scale, causal):
    """Reference attention for one request. q is [q_len, heads, dim]."""
    q_len, num_heads, _ = q.shape
    kv_len = k_hist.shape[0]
    gqa = num_heads // k_hist.shape[1]
    out = np.zeros_like(q)
    for h in range(num_heads):
        kv_h = h // gqa
        scores = (q[:, h, :] @ k_hist[:, kv_h, :].T) * scale
        if causal:
            j = np.arange(q_len)[:, None]
            keys = np.arange(kv_len)[None, :]
            scores = np.where(keys <= kv_len - q_len + j, scores, -np.inf)
        scores -= scores.max(axis=-1, keepdims=True)
        p = np.exp(scores)
        p /= p.sum(axis=-1, keepdims=True)
        out[:, h, :] = p @ v_hist[:, kv_h, :]
    return out


def _assert_close(got, want, label, dtype=DTYPE):
    rtol = float(jnp.finfo(dtype).eps)
    atol = 4.0 * rtol * max(float(np.abs(want).max()), 1e-6)
    err = np.abs(got - want)
    assert np.all(err <= atol + rtol * np.abs(want)), (
        f"{label}: max abs error {float(err.max()):.5f} exceeds "
        f"atol={atol:.5f} + rtol={rtol:.5f}*|want|"
    )


def test_abi_strides_match_the_allocated_pool():
    """The ABI's stride triple must describe the array the bridge allocates.

    Two independent derivations of the same layout is exactly the drift the
    split is supposed to prevent, so pin them against each other rather than
    against a hand-written constant.
    """
    layout = _layout()
    abi = AiterPagedAttentionAbiV1()
    k_pool, v_pool = bridge.allocate_pools(layout, abi, num_q_heads=NUM_Q_HEADS)

    assert k_pool.shape == abi.pool_shape(layout)
    assert v_pool.shape == k_pool.shape

    block, head, seq = abi.strides(layout)
    mirror = np.zeros(k_pool.shape, dtype=np.uint16)
    elem = [s // mirror.itemsize for s in mirror.strides]
    assert (elem[0], elem[2], elem[1]) == (block, head, seq)


def test_plan_matches_the_neutral_page_table():
    """The bridge must not reinterpret the page table, only re-type it."""
    layout = _layout()
    alloc = _Allocator(seed=1, padding_page_id=layout.padding_page_id)
    seq_lens = [37, 16, 5]
    page_ids = [alloc.take(-(-n // TOKENS_PER_PAGE)) for n in seq_lens]
    table = _page_table(page_ids, seq_lens, seq_lens)

    plan = bridge.plan_step(layout, table)

    np.testing.assert_array_equal(np.asarray(plan.kv_indptr), table.indptr())
    np.testing.assert_array_equal(
        np.asarray(plan.kv_page_indices), table.flat_page_indices()
    )
    np.testing.assert_array_equal(
        np.asarray(plan.kv_last_page_lens), table.last_page_lens(TOKENS_PER_PAGE)
    )
    np.testing.assert_array_equal(
        np.asarray(plan.cu_seqlens_q), np.array([0, 37, 53, 58], dtype=np.int32)
    )
    assert plan.max_seqlen_q == 37
    assert plan.max_seqlen_k == 37
    assert plan.is_decode is False


def test_decode_is_detected_from_query_lens():
    layout = _layout()
    alloc = _Allocator(seed=2, padding_page_id=layout.padding_page_id)
    seq_lens = [40, 12]
    page_ids = [alloc.take(-(-n // TOKENS_PER_PAGE)) for n in seq_lens]
    table = _page_table(page_ids, seq_lens, [1, 1])
    assert bridge.plan_step(layout, table).is_decode is True


def test_prefill_then_decode_round_trip():
    """Neutral types -> bridge -> append -> prefill -> append -> decode.

    One pool, one page table vocabulary, both attention phases. The reference
    keeps its own dense copy of each sequence, so nothing about the pool layout
    is shared between the two sides.
    """
    layout = _layout()
    abi = AiterPagedAttentionAbiV1()
    alloc = _Allocator(seed=3, padding_page_id=layout.padding_page_id)
    rng = np.random.default_rng(17)
    scale = 1.0 / math.sqrt(HEAD_DIM)

    k_pool, v_pool = bridge.allocate_pools(layout, abi, num_q_heads=NUM_Q_HEADS)

    # --- prefill -------------------------------------------------------------
    prompt_lens = [37, 16, 5]
    page_ids = [alloc.take(-(-n // TOKENS_PER_PAGE)) for n in prompt_lens]
    table = _page_table(page_ids, prompt_lens, prompt_lens)
    plan = bridge.plan_step(layout, table)

    total_q = sum(prompt_lens)
    k_new = rng.standard_normal((total_q, NUM_KV_HEADS, HEAD_DIM)).astype(np.float32)
    v_new = rng.standard_normal((total_q, NUM_KV_HEADS, HEAD_DIM)).astype(np.float32)
    q_pre = rng.standard_normal((total_q, NUM_Q_HEADS, HEAD_DIM)).astype(np.float32)

    k_pool, v_pool = bridge.append_step(
        jnp.asarray(k_new, dtype=DTYPE), jnp.asarray(v_new, dtype=DTYPE),
        k_pool, v_pool, plan, abi,
    )
    got = np.asarray(
        bridge.attend_step(
            jnp.asarray(q_pre, dtype=DTYPE), k_pool, v_pool, plan, abi, scale=scale
        ).astype(jnp.float32)
    )

    # Dense mirror, per request.
    hist_k, hist_v, want = [], [], np.zeros_like(q_pre)
    off = 0
    for n in prompt_lens:
        hist_k.append(k_new[off:off + n])
        hist_v.append(v_new[off:off + n])
        want[off:off + n] = _dense_attention(
            q_pre[off:off + n], hist_k[-1], hist_v[-1], scale, causal=True
        )
        off += n
    _assert_close(got, want, "prefill")

    # --- decode: one more token per request ----------------------------------
    seq_lens = [n + 1 for n in prompt_lens]
    for i, n in enumerate(seq_lens):
        if -(-n // TOKENS_PER_PAGE) > len(page_ids[i]):
            page_ids[i] = page_ids[i] + alloc.take(1)
    table = _page_table(page_ids, seq_lens, [1] * len(seq_lens))
    plan = bridge.plan_step(layout, table)
    assert plan.is_decode is True

    num_seqs = len(seq_lens)
    k_step = rng.standard_normal((num_seqs, NUM_KV_HEADS, HEAD_DIM)).astype(np.float32)
    v_step = rng.standard_normal((num_seqs, NUM_KV_HEADS, HEAD_DIM)).astype(np.float32)
    q_dec = rng.standard_normal((num_seqs, NUM_Q_HEADS, HEAD_DIM)).astype(np.float32)

    k_pool, v_pool = bridge.append_step(
        jnp.asarray(k_step, dtype=DTYPE), jnp.asarray(v_step, dtype=DTYPE),
        k_pool, v_pool, plan, abi,
    )
    got = np.asarray(
        bridge.attend_step(
            jnp.asarray(q_dec, dtype=DTYPE), k_pool, v_pool, plan, abi, scale=scale
        ).astype(jnp.float32)
    )

    want = np.zeros_like(q_dec)
    for i in range(num_seqs):
        hist_k[i] = np.concatenate([hist_k[i], k_step[i:i + 1]])
        hist_v[i] = np.concatenate([hist_v[i], v_step[i:i + 1]])
        want[i] = _dense_attention(
            q_dec[i:i + 1], hist_k[i], hist_v[i], scale, causal=True
        )[0]
    _assert_close(got, want, "decode")


def test_extend_over_context_written_by_a_previous_step():
    """A chunk appended to an existing sequence attends over its own history."""
    layout = _layout()
    abi = AiterPagedAttentionAbiV1()
    alloc = _Allocator(seed=5, padding_page_id=layout.padding_page_id)
    rng = np.random.default_rng(23)
    scale = 1.0 / math.sqrt(HEAD_DIM)

    k_pool, v_pool = bridge.allocate_pools(layout, abi, num_q_heads=NUM_Q_HEADS)

    prompt_lens = [48, 20]
    page_ids = [alloc.take(-(-n // TOKENS_PER_PAGE)) for n in prompt_lens]
    table = _page_table(page_ids, prompt_lens, prompt_lens)
    plan = bridge.plan_step(layout, table)

    total = sum(prompt_lens)
    k0 = rng.standard_normal((total, NUM_KV_HEADS, HEAD_DIM)).astype(np.float32)
    v0 = rng.standard_normal((total, NUM_KV_HEADS, HEAD_DIM)).astype(np.float32)
    k_pool, v_pool = bridge.append_step(
        jnp.asarray(k0, dtype=DTYPE), jnp.asarray(v0, dtype=DTYPE),
        k_pool, v_pool, plan, abi,
    )
    hist_k = [k0[:48], k0[48:]]
    hist_v = [v0[:48], v0[48:]]

    # Second chunk: several new tokens per request over the existing context.
    chunk = [12, 5]
    seq_lens = [p + c for p, c in zip(prompt_lens, chunk)]
    for i, n in enumerate(seq_lens):
        need = -(-n // TOKENS_PER_PAGE) - len(page_ids[i])
        if need > 0:
            page_ids[i] = page_ids[i] + alloc.take(need)
    table = _page_table(page_ids, seq_lens, chunk)
    plan = bridge.plan_step(layout, table)
    assert plan.is_decode is False

    total_c = sum(chunk)
    k1 = rng.standard_normal((total_c, NUM_KV_HEADS, HEAD_DIM)).astype(np.float32)
    v1 = rng.standard_normal((total_c, NUM_KV_HEADS, HEAD_DIM)).astype(np.float32)
    q1 = rng.standard_normal((total_c, NUM_Q_HEADS, HEAD_DIM)).astype(np.float32)

    k_pool, v_pool = bridge.append_step(
        jnp.asarray(k1, dtype=DTYPE), jnp.asarray(v1, dtype=DTYPE),
        k_pool, v_pool, plan, abi,
    )
    got = np.asarray(
        bridge.attend_step(
            jnp.asarray(q1, dtype=DTYPE), k_pool, v_pool, plan, abi, scale=scale
        ).astype(jnp.float32)
    )

    want = np.zeros_like(q1)
    off = 0
    for i, c in enumerate(chunk):
        hist_k[i] = np.concatenate([hist_k[i], k1[off:off + c]])
        hist_v[i] = np.concatenate([hist_v[i], v1[off:off + c]])
        want[off:off + c] = _dense_attention(
            q1[off:off + c], hist_k[i], hist_v[i], scale, causal=True
        )
        off += c
    _assert_close(got, want, "extend")


def test_abi_rejects_a_layout_the_kernels_cannot_serve():
    """Validation belongs at allocation, not inside a kernel launch."""
    abi = AiterPagedAttentionAbiV1()

    with pytest.raises(ValueError, match="GQA ratio"):
        abi.validate(_layout(num_kv_heads=3), num_q_heads=8)

    with pytest.raises(ValueError, match="head_dim"):
        bridge.allocate_pools(_layout(head_dim=96), abi, num_q_heads=NUM_Q_HEADS)

    with pytest.raises(ValueError, match="tokens_per_page"):
        bridge.allocate_pools(_layout(tokens_per_page=8), abi, num_q_heads=NUM_Q_HEADS)
