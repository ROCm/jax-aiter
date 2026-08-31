# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""M2b: paged decode correctness against a dense reference.

The reference gathers each sequence's K/V out of the pool by page id and runs
plain softmax attention in float32. That is deliberately the long way round: it
shares no code with the kernel, so a shared misunderstanding of the page layout
cannot cancel out.

Unlike M1, this is an arithmetic kernel, so the comparison is a tolerance rather
than exact equality. Coverage spans MHA and GQA head ratios, sequences that end
mid-page, and page maps that are deliberately non-contiguous.

The kernel configurations are compiled into paged_attention_ja.so, so these
need nothing beyond the module itself:

    make -f Makefile.kv ja_kv
"""

import math

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from jax_aiter.ffi.registry import standalone_symbol_available
from jax_aiter.ops.paged_attention import TARGET, paged_attention

TOKENS_PER_PAGE = 16
HEAD_DIM = 128
NUM_PAGES = 128
MAX_SEQ_LEN = 4096  # only sets the partition count and workspace size

pytestmark = pytest.mark.skipif(
    not standalone_symbol_available(TARGET),
    reason=f"{TARGET} FFI module not built (run 'make -f Makefile.kv ja_kv')",
)


def _tolerance(dtype):
    """One ulp of the output dtype.

    Measured across every shape in this suite the kernel lands within one ulp
    of the dense reference at the scale of the tensor (worst observed: 0.75 ulp
    for bf16, 0.6 for fp16), so the usual eps**(2/3) fudge would be about seven
    times looser than the kernel needs and would wave real regressions through.
    """
    return float(jnp.finfo(dtype).eps)


def _build_batch(seq_lens, num_kv_heads, dtype, seed=0):
    """Lay out a batch in the pool and return pools, metadata and a dense copy.

    Pages are handed out in a shuffled order so the test cannot accidentally
    pass on contiguous page maps.
    """
    rng = np.random.default_rng(seed)
    num_seqs = len(seq_lens)

    pool_shape = (NUM_PAGES, TOKENS_PER_PAGE, num_kv_heads, HEAD_DIM)
    k_pool = np.zeros(pool_shape, dtype=np.float32)
    v_pool = np.zeros(pool_shape, dtype=np.float32)

    # Page 0 is the padding sentinel and is never handed out.
    free_pages = list(rng.permutation(np.arange(1, NUM_PAGES)))

    page_ids, dense_k, dense_v = [], [], []
    for seq_len in seq_lens:
        n_pages = -(-seq_len // TOKENS_PER_PAGE)
        pages = [int(free_pages.pop()) for _ in range(n_pages)]
        page_ids.append(pages)

        k_seq = rng.standard_normal((seq_len, num_kv_heads, HEAD_DIM)).astype(np.float32)
        v_seq = rng.standard_normal((seq_len, num_kv_heads, HEAD_DIM)).astype(np.float32)
        dense_k.append(k_seq)
        dense_v.append(v_seq)

        for t in range(seq_len):
            page = pages[t // TOKENS_PER_PAGE]
            off = t % TOKENS_PER_PAGE
            k_pool[page, off] = k_seq[t]
            v_pool[page, off] = v_seq[t]

    indptr = np.zeros((num_seqs + 1,), dtype=np.int32)
    np.cumsum([len(p) for p in page_ids], out=indptr[1:])
    flat_pages = np.concatenate([np.asarray(p, dtype=np.int32) for p in page_ids])
    last_lens = np.asarray(
        [((s - 1) % TOKENS_PER_PAGE) + 1 for s in seq_lens], dtype=np.int32
    )

    return (
        jnp.asarray(k_pool, dtype=dtype),
        jnp.asarray(v_pool, dtype=dtype),
        jnp.asarray(indptr),
        jnp.asarray(flat_pages),
        jnp.asarray(last_lens),
        dense_k,
        dense_v,
    )


def _dense_reference(query, dense_k, dense_v, scale):
    """Plain softmax attention per sequence, in float32.

    query: [num_seqs, num_heads, head_dim]; dense_k/v: per-seq [len, kv_heads, dim].
    """
    q = np.asarray(query, dtype=np.float32)
    num_seqs, num_heads, head_dim = q.shape
    num_kv_heads = dense_k[0].shape[1]
    gqa_ratio = num_heads // num_kv_heads

    out = np.zeros_like(q)
    for i in range(num_seqs):
        k_seq, v_seq = dense_k[i], dense_v[i]
        for h in range(num_heads):
            kv_h = h // gqa_ratio
            scores = (k_seq[:, kv_h, :] @ q[i, h]) * scale       # [seq_len]
            scores -= scores.max()
            p = np.exp(scores)
            p /= p.sum()
            out[i, h] = p @ v_seq[:, kv_h, :]
    return out


def _run(seq_lens, num_heads, num_kv_heads, dtype, seed=0):
    k_pool, v_pool, indptr, pages, last_lens, dense_k, dense_v = _build_batch(
        seq_lens, num_kv_heads, dtype, seed=seed
    )
    rng = np.random.default_rng(seed + 999)
    query_np = rng.standard_normal((len(seq_lens), num_heads, HEAD_DIM)).astype(np.float32)
    query = jnp.asarray(query_np, dtype=dtype)
    scale = 1.0 / math.sqrt(HEAD_DIM)

    got = paged_attention(
        query, k_pool, v_pool, indptr, pages, last_lens,
        max_seq_len=MAX_SEQ_LEN, scale=scale,
    )
    got = np.asarray(got.astype(jnp.float32))
    want = _dense_reference(query.astype(jnp.float32), dense_k, dense_v, scale)
    return got, want


def _assert_close(got, want, dtype, label):
    """Bound the error by ``atol + rtol * |want|``, numpy's convention.

    Attention outputs pass through zero, and a purely element-wise relative
    metric is meaningless there: one ulp of error against a near-zero element
    reads as a 100% miss while saying nothing about correctness. The absolute
    floor is tied to the dtype's resolution at the magnitude of the tensor,
    which is what the softmax accumulation error actually scales with.
    """
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
def test_decode_mha_matches_dense(dtype):
    """num_heads == num_kv_heads, sequences ending mid-page."""
    got, want = _run([37, 16, 5, 64], num_heads=8, num_kv_heads=8, dtype=dtype)
    _assert_close(got, want, dtype, "mha")


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16])
def test_decode_gqa_matches_dense(dtype):
    """GQA: several query heads share each KV head."""
    got, want = _run([33, 48, 7], num_heads=8, num_kv_heads=2, dtype=dtype, seed=3)
    _assert_close(got, want, dtype, "gqa")


def test_decode_full_pages():
    """Every sequence ends exactly on a page boundary."""
    got, want = _run([16, 32, 48], num_heads=8, num_kv_heads=8,
                     dtype=jnp.bfloat16, seed=7)
    _assert_close(got, want, jnp.bfloat16, "full-pages")


def test_decode_single_token_sequences():
    """One token of context: the shortest possible last page."""
    got, want = _run([1, 1, 1, 1], num_heads=8, num_kv_heads=8,
                     dtype=jnp.bfloat16, seed=11)
    _assert_close(got, want, jnp.bfloat16, "single-token")


def test_decode_long_sequence_spans_many_pages():
    got, want = _run([511], num_heads=8, num_kv_heads=8, dtype=jnp.bfloat16, seed=13)
    _assert_close(got, want, jnp.bfloat16, "long")


def test_uncompiled_config_is_an_actionable_error():
    """A config that was never compiled in must fail loudly, not shell out.

    gqa_ratio 3 is not in gen_pa_ragged.py's default set, so this exercises the
    path where aiter would otherwise try to spawn a Python interpreter from
    inside the kernel launch.

    The kernels are compiled into paged_attention_ja.so, so unlike the previous
    dlopen design there is no cache that could make this succeed by accident:
    the set is fixed at link time. Hence no else-branch -- not raising is a
    failure.
    """
    with pytest.raises(Exception) as excinfo:  # noqa: PT011 - XLA's own error type
        _run([16], num_heads=6, num_kv_heads=2, dtype=jnp.bfloat16, seed=17)

    msg = str(excinfo.value)
    # Name the script that fixes it, and show what IS available -- the whole
    # point of the message is that the reader should not have to go digging.
    assert "gen_pa_ragged.py" in msg, f"message must name the generator: {msg}"
    assert "not compiled into this module" in msg, msg
    assert "pa_ragged_" in msg, f"message must list the compiled configs: {msg}"
