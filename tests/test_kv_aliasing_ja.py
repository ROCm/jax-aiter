# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""M0: prove in-place KV mutation through XLA FFI on ROCm.

The contract under test is that a pool bound as both operand and result of an
FFI call is mutated in place, with no pool-sized replacement allocation and no
pool-sized copy, and that a *failure* of that contract is loud rather than
silent.

Evidence collected, in the order the milestone requires it:
  1. the alias declaration is present in lowered StableHLO,
  2. it survives into optimised HLO on the custom call,
  3. the pool is donated at the jit boundary and the module records it,
  4. no pool-sized copy and no pool-sized temp allocation appear,
  5. live memory is flat across a long decode-shaped run,
  6. the mutation is numerically correct,
  7. a declined donation raises instead of warning,
  8. dropping the alias fails loudly rather than silently writing elsewhere.

Pointer identity is recorded as a diagnostic only, per the milestone.
"""

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from jax_aiter.ffi.registry import standalone_symbol_available
from jax_aiter.kv.aliasing import (
    AliasContractError,
    assert_flat_live_memory,
    assert_in_place_mutation,
    collect_alias_evidence,
    donation_warnings_as_errors,
    live_bytes,
    pool_shard_pointers,
)
from jax_aiter.ops.kv_alias_probe import TARGET, kv_alias_probe

# The paged-KV shims are built by their own make target, so a tree built with
# plain `make ja_mods` legitimately does not have them. Skip rather than error,
# and name the target in the reason so a missing build is obvious.
pytestmark = pytest.mark.skipif(
    not standalone_symbol_available(TARGET),
    reason=f"{TARGET} FFI module not built (run 'make -f Makefile.kv ja_kv')",
)

# Decode-shaped: one token row per request into a much larger pool.
POOL_ROWS = 256
ROW_ELEMS = 512
N_ROWS = 8

DTYPES = [jnp.float32, jnp.bfloat16]


def _step(pool, row_idx, vals):
    return kv_alias_probe(pool, row_idx, vals)


def _jitted():
    return jax.jit(_step, donate_argnums=(0,))


def _inputs(dtype, n_rows=N_ROWS, pool_rows=POOL_ROWS):
    pool = jnp.zeros((pool_rows, ROW_ELEMS), dtype=dtype)
    row_idx = jnp.arange(n_rows, dtype=jnp.int32)
    vals = jnp.ones((n_rows, ROW_ELEMS), dtype=dtype)
    return pool, row_idx, vals


@pytest.mark.parametrize("dtype", DTYPES)
def test_alias_declared_and_survives_compilation(dtype):
    """Checks 1-4: the alias reaches the compiler and no copy is introduced."""
    pool, row_idx, vals = _inputs(dtype)
    ev = assert_in_place_mutation(
        _jitted(), pool, row_idx, vals, pool_argnum=0, target=TARGET
    )
    print("\n" + ev.report())

    assert ev.lowered_alias_operands == (0,)
    assert ev.compiled_alias_operands == (0,)
    assert ev.compiled_entry_args == (0,)
    assert ev.pool_sized_copies == ()
    assert ev.alias_size_in_bytes >= ev.pool_bytes


@pytest.mark.parametrize("dtype", DTYPES)
def test_mutation_is_correct(dtype):
    """Check 6: the write lands where it should and nowhere else."""
    pool, row_idx, vals = _inputs(dtype)
    out = _jitted()(pool, row_idx, vals)
    got = np.asarray(out.astype(jnp.float32))

    np.testing.assert_allclose(got[:N_ROWS], 1.0, rtol=0, atol=0)
    np.testing.assert_allclose(got[N_ROWS:], 0.0, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_accumulates_across_steps(dtype):
    """The pool carries state forward, so repeated steps accumulate."""
    pool, row_idx, vals = _inputs(dtype)
    fn = _jitted()

    steps = 16
    for _ in range(steps):
        pool = fn(pool, row_idx, vals)

    got = np.asarray(pool.astype(jnp.float32))
    np.testing.assert_allclose(got[:N_ROWS], float(steps), rtol=0, atol=0)
    np.testing.assert_allclose(got[N_ROWS:], 0.0, rtol=0, atol=0)


def test_sentinel_and_out_of_range_rows_are_skipped():
    """Negative and out-of-range indices are dropped, as page 0 padding needs."""
    pool, _, vals = _inputs(jnp.float32, n_rows=4)
    row_idx = jnp.asarray([2, -1, POOL_ROWS + 5, 3], dtype=jnp.int32)

    out = _jitted()(pool, row_idx, vals)
    got = np.asarray(out)

    assert np.all(got[2] == 1.0)
    assert np.all(got[3] == 1.0)
    touched = {2, 3}
    for r in range(POOL_ROWS):
        if r not in touched:
            assert np.all(got[r] == 0.0), f"row {r} should be untouched"


@pytest.mark.parametrize("dtype", DTYPES)
def test_live_memory_is_flat_over_long_run(dtype):
    """Check 5: no per-step replacement allocation across many steps."""
    pool, row_idx, vals = _inputs(dtype)
    fn = _jitted()

    baseline, final, _ = assert_flat_live_memory(
        fn, pool, (row_idx, vals), steps=1024, warmup=8
    )
    print(f"\nlive bytes: baseline={baseline} final={final} delta={final - baseline}")


def test_reusing_a_donated_buffer_is_an_error():
    """Check 7a: consuming an already-donated pool fails hard.

    On this build XLA rejects it outright rather than warning, which is the
    behaviour we want; the test pins it so a future relaxation is visible.
    """
    pool, row_idx, vals = _inputs(jnp.float32)
    fn = _jitted()

    with pytest.raises(Exception) as excinfo:
        with donation_warnings_as_errors():
            # `pool` is still referenced here, so the second call is invalid.
            _ = fn(pool, row_idx, vals)
            _ = fn(pool, row_idx, vals)
    assert "donat" in str(excinfo.value).lower(), (
        f"expected a donation-related failure, got: {excinfo.value!r}"
    )


def test_unusable_donation_warning_is_escalated():
    """Check 7b: a donation JAX merely *warns* about becomes an error.

    This is the dangerous variant the milestone singles out. Donating an
    argument that no output can alias -- here the values buffer, which has a
    different shape from the result -- makes JAX emit "Some donated buffers were
    not usable" and carry on. Silently carrying on is what must not happen,
    because the pool would then be reallocated every step while still computing
    the right answer.
    """
    pool, row_idx, vals = _inputs(jnp.float32)
    # Donate argument 2 (vals): the only result aliases argument 0, so this
    # donation cannot be taken.
    fn = jax.jit(_step, donate_argnums=(2,))

    with pytest.raises(UserWarning) as excinfo:
        with donation_warnings_as_errors():
            out = fn(pool, row_idx, vals)
            out.block_until_ready()

    assert "donat" in str(excinfo.value).lower(), (
        f"expected the unusable-donation warning, got: {excinfo.value!r}"
    )


def test_dropping_the_alias_fails_loudly():
    """Check 8: without input_output_aliases the handler refuses to run.

    This is what makes the contract self-policing: the .Arg and .Ret resolve to
    different pointers, the handler detects it, and the step fails instead of
    writing into a buffer that is about to be discarded.
    """

    def unaliased_step(pool, row_idx, vals):
        # Same target, alias deliberately omitted.
        call = jax.ffi.ffi_call(
            TARGET, jax.ShapeDtypeStruct(pool.shape, pool.dtype)
        )
        return call(pool, row_idx, vals)

    pool, row_idx, vals = _inputs(jnp.float32)
    fn = jax.jit(unaliased_step)

    with pytest.raises(Exception) as excinfo:
        out = fn(pool, row_idx, vals)
        out.block_until_ready()

    msg = str(excinfo.value).lower()
    assert "alias" in msg or "pointer" in msg, (
        f"expected the handler's alias check to fire, got: {excinfo.value!r}"
    )


def test_alias_evidence_reports_missing_alias():
    """The harness itself must fail when the alias is absent."""

    def unaliased_step(pool, row_idx, vals):
        call = jax.ffi.ffi_call(
            TARGET, jax.ShapeDtypeStruct(pool.shape, pool.dtype)
        )
        return call(pool, row_idx, vals)

    pool, row_idx, vals = _inputs(jnp.float32)
    ev = collect_alias_evidence(
        jax.jit(unaliased_step), pool, row_idx, vals, target=TARGET
    )
    assert not ev.ok
    assert any("no alias on FFI operand 0" in f for f in ev.failures), ev.report()

    with pytest.raises(AliasContractError):
        assert_in_place_mutation(
            jax.jit(unaliased_step), pool, row_idx, vals, target=TARGET
        )


def test_pointer_identity_diagnostic():
    """Diagnostic: single-device pointer stability under donation."""
    pool, row_idx, vals = _inputs(jnp.float32)
    fn = _jitted()

    pool = fn(pool, row_idx, vals)
    pool.block_until_ready()
    first = pool_shard_pointers(pool)

    for _ in range(64):
        pool = fn(pool, row_idx, vals)
    pool.block_until_ready()
    last = pool_shard_pointers(pool)

    print(f"\nshard pointers: first={first} last={last}")
    assert first == last, (
        f"pool moved during an aliased run: {first} -> {last}. "
        f"Under external registration this would be fatal."
    )
