# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""M2c: paged prefill correctness against a dense reference.

The reference gathers each sequence's K/V out of the pool by page id and runs
plain masked softmax attention in float32. As in M2b that is deliberately the
long way round: it shares no code with the kernel, so a shared misunderstanding
of the page layout cannot cancel out.

Two regimes matter and both are covered. A fresh prefill has ``seqlen_q ==
seqlen_k``, where top-left and bottom-right causal masks coincide. An extend --
appending a chunk to a sequence that already has context in the pool -- has
``seqlen_k > seqlen_q``, and there the two conventions disagree, so those cases
are what actually pin down the alignment.

The kernels are compiled into the shim, so unlike M2b there is nothing to
prebuild:

    make -f Makefile.kv paged_prefill GFX=gfx942 -j16
"""

import math

import numpy as np
import pytest

import jax.numpy as jnp

from jax_aiter.ffi.registry import standalone_symbol_available
from jax_aiter.ops.paged_prefill import TARGET, paged_prefill

TOKENS_PER_PAGE = 16
HEAD_DIM = 128
NUM_PAGES = 256

pytestmark = pytest.mark.skipif(
    not standalone_symbol_available(TARGET),
    reason=f"{TARGET} FFI module not built (run 'make -f Makefile.kv paged_prefill')",
)


def _tolerance(dtype):
    """One ulp of the output dtype; see the note in test_paged_attention_ja."""
    return float(jnp.finfo(dtype).eps)


def _build_batch(kv_lens, q_lens, num_kv_heads, dtype, seed=0):
    """Lay a batch out in the pool and return pools, metadata and a dense copy.

    Each sequence holds ``kv_len`` tokens of context, the last ``q_len`` of which
    are the tokens being prefilled. Pages are handed out shuffled so the test
    cannot pass on a contiguous page map.
    """
    rng = np.random.default_rng(seed)
    assert len(kv_lens) == len(q_lens)

    pool_shape = (NUM_PAGES, TOKENS_PER_PAGE, num_kv_heads, HEAD_DIM)
    k_pool = np.zeros(pool_shape, dtype=np.float32)
    v_pool = np.zeros(pool_shape, dtype=np.float32)

    # Page 0 is the padding sentinel and is never handed out.
    free_pages = list(rng.permutation(np.arange(1, NUM_PAGES)))

    page_ids, dense_k, dense_v = [], [], []
    for kv_len in kv_lens:
        n_pages = -(-kv_len // TOKENS_PER_PAGE)
        pages = [int(free_pages.pop()) for _ in range(n_pages)]
        page_ids.append(pages)

        k_seq = rng.standard_normal((kv_len, num_kv_heads, HEAD_DIM)).astype(np.float32)
        v_seq = rng.standard_normal((kv_len, num_kv_heads, HEAD_DIM)).astype(np.float32)
        dense_k.append(k_seq)
        dense_v.append(v_seq)

        for t in range(kv_len):
            page = pages[t // TOKENS_PER_PAGE]
            k_pool[page, t % TOKENS_PER_PAGE] = k_seq[t]
            v_pool[page, t % TOKENS_PER_PAGE] = v_seq[t]

    kv_indptr = np.zeros((len(kv_lens) + 1,), dtype=np.int32)
    np.cumsum([len(p) for p in page_ids], out=kv_indptr[1:])
    flat_pages = np.concatenate([np.asarray(p, dtype=np.int32) for p in page_ids])
    last_lens = np.asarray(
        [((n - 1) % TOKENS_PER_PAGE) + 1 for n in kv_lens], dtype=np.int32
    )

    cu_seqlens_q = np.zeros((len(q_lens) + 1,), dtype=np.int32)
    np.cumsum(q_lens, out=cu_seqlens_q[1:])

    return (
        jnp.asarray(k_pool, dtype=dtype),
        jnp.asarray(v_pool, dtype=dtype),
        jnp.asarray(cu_seqlens_q),
        jnp.asarray(kv_indptr),
        jnp.asarray(flat_pages),
        jnp.asarray(last_lens),
        dense_k,
        dense_v,
    )


def _dense_reference(query, cu_seqlens_q, kv_lens, dense_k, dense_v, scale, causal):
    """Masked softmax attention per sequence, in float32.

    The causal mask is bottom-right aligned: within a sequence, query token ``j``
    of ``q_len`` sees key positions up to ``kv_len - q_len + j`` inclusive. For a
    fresh prefill (``kv_len == q_len``) that reduces to the familiar triangle.
    """
    q = np.asarray(query, dtype=np.float32)
    _, num_heads, _ = q.shape
    num_kv_heads = dense_k[0].shape[1]
    gqa_ratio = num_heads // num_kv_heads
    cu = np.asarray(cu_seqlens_q)

    out = np.zeros_like(q)
    for i, kv_len in enumerate(kv_lens):
        lo, hi = int(cu[i]), int(cu[i + 1])
        q_len = hi - lo
        if q_len == 0:
            continue
        k_seq, v_seq = dense_k[i], dense_v[i]
        for h in range(num_heads):
            kv_h = h // gqa_ratio
            # [q_len, kv_len]
            scores = (q[lo:hi, h, :] @ k_seq[:, kv_h, :].T) * scale
            if causal:
                j = np.arange(q_len)[:, None]
                keys = np.arange(kv_len)[None, :]
                scores = np.where(keys <= kv_len - q_len + j, scores, -np.inf)
            scores -= scores.max(axis=-1, keepdims=True)
            p = np.exp(scores)
            p /= p.sum(axis=-1, keepdims=True)
            out[lo:hi, h, :] = p @ v_seq[:, kv_h, :]
    return out


def _run(kv_lens, q_lens, num_heads, num_kv_heads, dtype, seed=0, causal=True):
    (k_pool, v_pool, cu_seqlens_q, kv_indptr, pages, last_lens,
     dense_k, dense_v) = _build_batch(kv_lens, q_lens, num_kv_heads, dtype, seed)

    rng = np.random.default_rng(seed + 999)
    total_q = int(sum(q_lens))
    query_np = rng.standard_normal((total_q, num_heads, HEAD_DIM)).astype(np.float32)
    query = jnp.asarray(query_np, dtype=dtype)
    scale = 1.0 / math.sqrt(HEAD_DIM)

    got = paged_prefill(
        query, k_pool, v_pool, cu_seqlens_q, kv_indptr, pages, last_lens,
        max_seqlen_q=int(max(q_lens)), max_seqlen_k=int(max(kv_lens)),
        scale=scale, causal=causal,
    )
    got = np.asarray(got.astype(jnp.float32))
    want = _dense_reference(np.asarray(query.astype(jnp.float32)), cu_seqlens_q,
                            kv_lens, dense_k, dense_v, scale, causal)
    return got, want


def _assert_close(got, want, dtype, label):
    """Bound the error by ``atol + rtol * |want|``, as in the M2b suite."""
    rtol = _tolerance(dtype)
    scale = float(np.abs(want).max())
    atol = 4.0 * rtol * max(scale, 1e-6)

    err = np.abs(got - want)
    bound = atol + rtol * np.abs(want)
    worst = int(np.argmax(err - bound))
    assert np.all(err <= bound), (
        f"{label}: exceeds atol={atol:.5f} + rtol={rtol:.5f}*|want|\n"
        f"  worst element: got {got.flat[worst]:.5f}, want {want.flat[worst]:.5f}, "
        f"error {err.flat[worst]:.5f} vs bound {bound.flat[worst]:.5f}\n"
        f"  max abs error {float(err.max()):.5f}, "
        f"reference range [{want.min():.3f}, {want.max():.3f}]"
    )


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
def test_fresh_prefill_matches_dense(dtype):
    """Whole sequences prefilled at once: seqlen_q == seqlen_k."""
    lens = [37, 16, 5, 64]
    got, want = _run(lens, lens, num_heads=8, num_kv_heads=8, dtype=dtype)
    _assert_close(got, want, dtype, "fresh-prefill")


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
def test_fresh_prefill_gqa(dtype):
    """GQA: several query heads share each KV head."""
    lens = [33, 48, 7]
    got, want = _run(lens, lens, num_heads=8, num_kv_heads=2, dtype=dtype, seed=3)
    _assert_close(got, want, dtype, "gqa")


def test_extend_over_existing_context():
    """Chunked prefill: the pool already holds a prefix the new tokens attend to.

    This is the case that distinguishes bottom-right from top-left causal
    alignment, so it is the one that actually pins the convention down.
    """
    kv_lens = [64, 40, 17]
    q_lens = [16, 8, 1]
    got, want = _run(kv_lens, q_lens, num_heads=8, num_kv_heads=8,
                     dtype=jnp.bfloat16, seed=5)
    _assert_close(got, want, jnp.bfloat16, "extend")


def test_extend_gqa_ragged():
    """Extend with GQA and query segments of markedly different lengths."""
    kv_lens = [128, 33, 96, 20]
    q_lens = [32, 1, 48, 4]
    got, want = _run(kv_lens, q_lens, num_heads=8, num_kv_heads=2,
                     dtype=jnp.bfloat16, seed=7)
    _assert_close(got, want, jnp.bfloat16, "extend-gqa")


def test_non_causal_full_attention():
    """Without a mask every query token sees the whole context."""
    lens = [24, 48]
    got, want = _run(lens, lens, num_heads=8, num_kv_heads=8,
                     dtype=jnp.bfloat16, seed=11, causal=False)
    _assert_close(got, want, jnp.bfloat16, "non-causal")


def test_sequences_ending_on_page_boundary():
    """Every sequence fills its last page exactly."""
    lens = [16, 32, 48]
    got, want = _run(lens, lens, num_heads=8, num_kv_heads=8,
                     dtype=jnp.bfloat16, seed=13)
    _assert_close(got, want, jnp.bfloat16, "full-pages")


def test_single_token_queries():
    """One query token per sequence: the decode shape, through the prefill path."""
    kv_lens = [31, 64, 7]
    got, want = _run(kv_lens, [1, 1, 1], num_heads=8, num_kv_heads=8,
                     dtype=jnp.bfloat16, seed=17)
    _assert_close(got, want, jnp.bfloat16, "single-token")


def test_long_sequence_spans_many_pages():
    got, want = _run([511], [511], num_heads=8, num_kv_heads=8,
                     dtype=jnp.bfloat16, seed=19)
    _assert_close(got, want, jnp.bfloat16, "long")


def test_batch_of_one():
    got, want = _run([96], [96], num_heads=8, num_kv_heads=8,
                     dtype=jnp.bfloat16, seed=23)
    _assert_close(got, want, jnp.bfloat16, "batch-of-one")


def test_rejects_non_int32_metadata():
    """The page table must be int32; the handler must say so rather than misread it.

    float32 is the useful probe here. int64 would be the natural one, but JAX
    silently truncates it to int32 unless jax_enable_x64 is set, so casting to it
    tests nothing -- which is itself the reason the int32 contract is comfortable
    to rely on from a JAX front end.
    """
    lens = [16]
    (k_pool, v_pool, cu_seqlens_q, kv_indptr, pages, last_lens,
     _, _) = _build_batch(lens, lens, 8, jnp.bfloat16, seed=29)
    query = jnp.zeros((16, 8, HEAD_DIM), dtype=jnp.bfloat16)

    with pytest.raises(Exception) as excinfo:
        paged_prefill(
            query, k_pool, v_pool, cu_seqlens_q.astype(jnp.float32), kv_indptr,
            pages, last_lens, max_seqlen_q=16, max_seqlen_k=16,
        ).block_until_ready()
    assert "int32" in str(excinfo.value).lower()
