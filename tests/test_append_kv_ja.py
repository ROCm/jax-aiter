# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""M1: append_kv correctness and in-place behaviour.

Validates the ``aiter::reshape_and_cache_flash`` bridge against a pure-JAX
scatter reference, and re-applies the M0 aliasing assertions to the real op
rather than the throwaway probe.

Metadata here is built by hand from the jax-aiter ABI only. That is deliberate:
the neutral page-table type lives in the control plane, and a vendor repo's tests
should not need it. If this file ever wanted to import it, the layering would be
wrong.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from jax_aiter.ffi.registry import standalone_symbol_available
from jax_aiter.kv.abi import AiterPagedAttentionAbiV1
from jax_aiter.kv.aliasing import assert_flat_live_memory, assert_in_place_mutation
from jax_aiter.ops.append_kv import APPEND_KV_ALIASES, TARGET, append_kv

# The paged-KV shims are built by their own make target, so a tree built with
# plain `make ja_mods` legitimately does not have them. Skip rather than error,
# and name the target in the reason so a missing build is obvious.
pytestmark = pytest.mark.skipif(
    not standalone_symbol_available(TARGET),
    reason=f"{TARGET} FFI module not built (run 'make -f Makefile.kv ja_kv')",
)

TOKENS_PER_PAGE = 16
NUM_PAGES = 64
HEAD_DIM = 128

# jit argument index -> FFI operand index, for the pools in `_step` below.
# These differ because the FFI call takes the pools last while the jit takes them
# first, and the alias assertions need both.
APPEND_KV_ALIASES_BY_ARG = {0: 5, 1: 6}

# (num_kv_heads, label)
HEAD_CONFIGS = [(8, "mha-like"), (2, "gqa-like"), (1, "mqa-like")]
DTYPES = [jnp.bfloat16, jnp.float16]


def _pools(num_kv_heads, dtype, num_pages=NUM_PAGES):
    shape = (num_pages, TOKENS_PER_PAGE, num_kv_heads, HEAD_DIM)
    return jnp.zeros(shape, dtype=dtype), jnp.zeros(shape, dtype=dtype)


def _new_kv(num_tokens, num_kv_heads, dtype, seed=0):
    rng = np.random.default_rng(seed)
    shape = (num_tokens, num_kv_heads, HEAD_DIM)
    k = jnp.asarray(rng.standard_normal(shape), dtype=dtype)
    v = jnp.asarray(rng.standard_normal(shape), dtype=dtype)
    return k, v


def _scatter_ref(pool, new, slots, tokens_per_page):
    """Pure-JAX reference: pool[slot // tpp, slot % tpp] = new[token].

    Tokens with a negative slot write their existing value back, which is the
    functional equivalent of the kernel skipping them.
    """
    valid = slots >= 0
    safe = jnp.where(valid, slots, 0)
    pages = safe // tokens_per_page
    offs = safe % tokens_per_page

    current = pool[pages, offs]
    written = jnp.where(valid[:, None, None], new, current)
    return pool.at[pages, offs].set(written)


def _step(k_pool, v_pool, k_new, v_new, slots):
    return append_kv(k_new, v_new, slots, k_pool, v_pool)


def _jitted_step():
    # Both pools are donated: aliasing is {5: 0, 6: 1}, so a half-honoured
    # donation would leave one pool correct and one silently replaced.
    return jax.jit(_step, donate_argnums=(0, 1))


# ---------------------------------------------------------------------------
# Correctness against the reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_kv_heads,label", HEAD_CONFIGS)
def test_decode_write_matches_reference(dtype, num_kv_heads, label):
    """One token per sequence, scattered across non-contiguous pages."""
    num_tokens = 8
    k_pool, v_pool = _pools(num_kv_heads, dtype)
    k_new, v_new = _new_kv(num_tokens, num_kv_heads, dtype, seed=1)

    # Deliberately unordered and non-contiguous, as a real page map would be.
    slots = jnp.asarray(
        [3 * 16 + 5, 1 * 16 + 0, 17 * 16 + 15, 2 * 16 + 7,
         40 * 16 + 1, 9 * 16 + 9, 63 * 16 + 3, 5 * 16 + 12],
        dtype=jnp.int32,
    )

    k_ref = _scatter_ref(k_pool, k_new, slots, TOKENS_PER_PAGE)
    v_ref = _scatter_ref(v_pool, v_new, slots, TOKENS_PER_PAGE)

    k_got, v_got = _jitted_step()(k_pool, v_pool, k_new, v_new, slots)

    np.testing.assert_array_equal(np.asarray(k_got), np.asarray(k_ref))
    np.testing.assert_array_equal(np.asarray(v_got), np.asarray(v_ref))


@pytest.mark.parametrize("dtype", DTYPES)
def test_prefill_write_crosses_page_boundaries(dtype):
    """A prefill writes many contiguous tokens spanning several pages."""
    num_kv_heads = 8
    num_tokens = 40  # 2.5 pages at 16 tokens per page
    k_pool, v_pool = _pools(num_kv_heads, dtype)
    k_new, v_new = _new_kv(num_tokens, num_kv_heads, dtype, seed=2)

    # Pages 7, 8, 9 held by one request.
    base_pages = [7, 8, 9]
    slots = np.empty((num_tokens,), dtype=np.int32)
    for t in range(num_tokens):
        page = base_pages[t // TOKENS_PER_PAGE]
        slots[t] = page * TOKENS_PER_PAGE + (t % TOKENS_PER_PAGE)
    slots = jnp.asarray(slots)

    k_ref = _scatter_ref(k_pool, k_new, slots, TOKENS_PER_PAGE)
    v_ref = _scatter_ref(v_pool, v_new, slots, TOKENS_PER_PAGE)

    k_got, v_got = _jitted_step()(k_pool, v_pool, k_new, v_new, slots)

    np.testing.assert_array_equal(np.asarray(k_got), np.asarray(k_ref))
    np.testing.assert_array_equal(np.asarray(v_got), np.asarray(v_ref))


@pytest.mark.parametrize("dtype", DTYPES)
def test_negative_slots_are_skipped(dtype):
    """Padded rows carry slot -1 and must leave the pool untouched."""
    num_kv_heads = 4
    k_pool, v_pool = _pools(num_kv_heads, dtype)
    k_new, v_new = _new_kv(6, num_kv_heads, dtype, seed=3)

    slots = jnp.asarray([2 * 16 + 0, -1, 4 * 16 + 3, -1, -1, 6 * 16 + 8],
                        dtype=jnp.int32)

    # Reference first: the call below donates k_pool, after which it is deleted.
    k_ref = _scatter_ref(k_pool, k_new, slots, TOKENS_PER_PAGE)
    k_got, v_got = _jitted_step()(k_pool, v_pool, k_new, v_new, slots)

    np.testing.assert_array_equal(np.asarray(k_got), np.asarray(k_ref))

    # Everything outside the three written slots is still zero.
    got = np.asarray(k_got.astype(jnp.float32))
    touched = {(2, 0), (4, 3), (6, 8)}
    for page in range(NUM_PAGES):
        for off in range(TOKENS_PER_PAGE):
            if (page, off) not in touched:
                assert np.all(got[page, off] == 0.0), f"page {page} off {off} moved"


def test_repeated_appends_accumulate_in_the_same_pool():
    """Successive decode steps build up context in one pool."""
    dtype = jnp.bfloat16
    num_kv_heads = 8
    k_pool, v_pool = _pools(num_kv_heads, dtype)
    fn = _jitted_step()

    ref_k = k_pool
    for step in range(8):
        k_new, v_new = _new_kv(2, num_kv_heads, dtype, seed=100 + step)
        # Two requests advancing one token each, in pages 1 and 2.
        slots = jnp.asarray([1 * 16 + step, 2 * 16 + step], dtype=jnp.int32)
        ref_k = _scatter_ref(ref_k, k_new, slots, TOKENS_PER_PAGE)
        k_pool, v_pool = fn(k_pool, v_pool, k_new, v_new, slots)

    np.testing.assert_array_equal(np.asarray(k_pool), np.asarray(ref_k))


# ---------------------------------------------------------------------------
# The M0 contract, re-applied to the real op
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES)
def test_both_pools_are_aliased_in_place(dtype):
    """M0's assertions, now on append_kv, for both K and V.

    Note the two index spaces: K is jit argument 0 but FFI operand 5, V is jit
    argument 1 but FFI operand 6. Checking only one of the pair would miss a
    half-honoured alias.
    """
    num_kv_heads = 8
    k_pool, v_pool = _pools(num_kv_heads, dtype)
    k_new, v_new = _new_kv(8, num_kv_heads, dtype, seed=4)
    slots = jnp.arange(8, dtype=jnp.int32) * TOKENS_PER_PAGE

    fn = _jitted_step()
    for argnum, operand in APPEND_KV_ALIASES_BY_ARG.items():
        ev = assert_in_place_mutation(
            fn, k_pool, v_pool, k_new, v_new, slots,
            pool_argnum=argnum, ffi_operand_index=operand, target=TARGET,
        )
        assert ev.pool_sized_copies == ()
        print(f"\n[jit arg {argnum} / ffi operand {operand}] " + ev.report())


def test_live_memory_flat_over_long_decode_run():
    """No per-step replacement allocation for either pool."""
    dtype = jnp.bfloat16
    num_kv_heads = 8
    k_pool, v_pool = _pools(num_kv_heads, dtype)
    k_new, v_new = _new_kv(8, num_kv_heads, dtype, seed=5)
    slots = jnp.arange(8, dtype=jnp.int32) * TOKENS_PER_PAGE
    fn = _jitted_step()

    # assert_flat_live_memory threads a single pool, so pair the two here.
    def paired(pools, *rest):
        return fn(pools[0], pools[1], *rest)

    baseline_pool = k_pool
    pool_pair = (k_pool, v_pool)
    for _ in range(8):
        pool_pair = paired(pool_pair, k_new, v_new, slots)
    jax.block_until_ready(pool_pair)
    baseline = jax.local_devices()[0].memory_stats()["bytes_in_use"]

    for _ in range(512):
        pool_pair = paired(pool_pair, k_new, v_new, slots)
    jax.block_until_ready(pool_pair)
    final = jax.local_devices()[0].memory_stats()["bytes_in_use"]

    pool_bytes = baseline_pool.size * np.dtype(baseline_pool.dtype).itemsize
    print(f"\nlive bytes baseline={baseline} final={final} delta={final - baseline}")
    assert final - baseline <= pool_bytes // 2, (
        f"live memory grew {final - baseline} bytes over 512 steps; a per-step "
        f"pool replacement is the likely cause"
    )


def test_dropping_the_alias_fails_loudly():
    """Without the aliases the handler must refuse rather than write nowhere."""
    num_kv_heads = 8
    dtype = jnp.bfloat16
    k_pool, v_pool = _pools(num_kv_heads, dtype)
    k_new, v_new = _new_kv(4, num_kv_heads, dtype, seed=6)
    slots = jnp.arange(4, dtype=jnp.int32) * TOKENS_PER_PAGE
    ones = jnp.ones((1,), dtype=jnp.float32)

    def unaliased(k_pool, v_pool, k_new, v_new, slots):
        call = jax.ffi.ffi_call(
            TARGET,
            (
                jax.ShapeDtypeStruct(k_pool.shape, k_pool.dtype),
                jax.ShapeDtypeStruct(v_pool.shape, v_pool.dtype),
            ),
        )
        return call(k_new, v_new, slots, ones, ones, k_pool, v_pool,
                    kv_cache_dtype="auto")

    with pytest.raises(Exception) as excinfo:
        out = jax.jit(unaliased)(k_pool, v_pool, k_new, v_new, slots)
        jax.block_until_ready(out)

    msg = str(excinfo.value).lower()
    assert "alias" in msg or "pointer" in msg, f"unexpected error: {excinfo.value!r}"


# ---------------------------------------------------------------------------
# Shape and dtype validation
# ---------------------------------------------------------------------------


def test_int64_slot_mapping_matches_int32():
    """The int64 path must keep working -- it is what vLLM-shaped callers pass.

    vLLM stores slot_mapping as torch.int64 while SGLang and sglang-jax use
    int32, so aiter has to serve both. Both widths go through the same load
    helper, so this is a regression test for the int64 side rather than a new
    feature.
    """
    num_kv_heads = 4
    dtype = jnp.bfloat16
    k_new, v_new = _new_kv(6, num_kv_heads, dtype, seed=11)
    slots_np = np.asarray([2 * 16 + 1, -1, 5 * 16 + 7, 9 * 16 + 0, -1, 3 * 16 + 15])

    k_pool, v_pool = _pools(num_kv_heads, dtype)
    got32, _ = _jitted_step()(
        k_pool, v_pool, k_new, v_new, jnp.asarray(slots_np, dtype=jnp.int32)
    )
    got32 = np.asarray(got32)

    jax.config.update("jax_enable_x64", True)
    try:
        slots64 = jnp.asarray(slots_np, dtype=jnp.int64)
        assert slots64.dtype == jnp.int64, "x64 did not take effect"
        k_pool64, v_pool64 = _pools(num_kv_heads, dtype)
        got64, _ = _jitted_step()(k_pool64, v_pool64, k_new, v_new, slots64)
        got64 = np.asarray(got64)
    finally:
        jax.config.update("jax_enable_x64", False)

    np.testing.assert_array_equal(got64, got32)


def test_head_count_mismatch_is_rejected():
    dtype = jnp.bfloat16
    k_pool, v_pool = _pools(8, dtype)
    k_new, v_new = _new_kv(4, 4, dtype, seed=7)  # 4 heads vs pool's 8
    slots = jnp.arange(4, dtype=jnp.int32) * TOKENS_PER_PAGE

    with pytest.raises(Exception) as excinfo:
        out = _jitted_step()(k_pool, v_pool, k_new, v_new, slots)
        jax.block_until_ready(out)
    assert "head" in str(excinfo.value).lower()


def test_abi_strides_match_the_pool_layout():
    """The ABI's stride triple must describe the pool the kernels actually read."""

    class _Layout:
        num_pages = NUM_PAGES
        tokens_per_page = TOKENS_PER_PAGE
        head_dim = HEAD_DIM
        dtype = "bfloat16"

        def heads_per_shard(self):
            return 8

    abi = AiterPagedAttentionAbiV1()
    layout = _Layout()

    assert abi.pool_shape(layout) == (NUM_PAGES, TOKENS_PER_PAGE, 8, HEAD_DIM)

    block, head, seq = abi.strides(layout)
    # Row-major over [pages, tokens, heads, dim].
    assert seq == 8 * HEAD_DIM
    assert block == TOKENS_PER_PAGE * 8 * HEAD_DIM
    assert head == HEAD_DIM

    # Cross-check against numpy's own strides for the same shape.
    arr = np.zeros(abi.pool_shape(layout), dtype=np.uint16)
    elem_strides = [s // arr.itemsize for s in arr.strides]
    assert elem_strides[0] == block
    assert elem_strides[1] == seq
    assert elem_strides[2] == head


def test_aliases_are_declared_where_the_handler_expects():
    """Guard against the alias map drifting from the handler's argument order."""
    assert APPEND_KV_ALIASES == {5: 0, 6: 1}
