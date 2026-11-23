# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""flash_attn_varlen_raw must stay differentiable inside ``shard_map``.

MaxText's ``attention=aiter_flash`` calls it per device inside ``shard_map``
with no logical-length arrays (one full segment per row). Every array in the
custom_vjp residuals then crosses the shard_map boundary. A zero-length array
there segfaults XLA's Shardy import in JAX 0.11.2, so an absent logical array
has to travel as ``None``.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest

import jax
import jax.numpy as jnp

pytest.importorskip("jax_aiter.mha", reason="AITER MHA libraries missing")

gpu = pytest.mark.skipif(not jax.devices("gpu"), reason="needs a GPU")

ROWS, SEQ, HEADS, DIM = 2, 128, 2, 128
STATIC = (SEQ, SEQ, 0.0, DIM ** -0.5, True, (-1, -1))


def _tight_inputs(rows=ROWS, seed=0):
    rng = np.random.default_rng(seed)
    q, k, v = (jnp.asarray(rng.normal(size=(rows * SEQ, HEADS, DIM)), jnp.bfloat16)
               for _ in range(3))
    seqstart = jnp.arange(rows + 1, dtype=jnp.int32) * SEQ
    return q, k, v, seqstart


def _leaves(tree):
    return jax.tree_util.tree_leaves(tree)


@gpu
def test_residuals_carry_no_zero_length_array_without_padding():
    from jax_aiter.mha.mha import _favr_fwd

    q, k, v, seqstart = _tight_inputs()
    fwd = jax.jit(_favr_fwd, static_argnums=(7, 8, 9, 10, 11, 12))
    empty = jnp.zeros((0,), jnp.int32)
    for logical in (None, empty):
        _, res = fwd(q, k, v, seqstart, seqstart, logical, logical, *STATIC)
        assert res[-2] is None and res[-1] is None
        assert all(leaf.size > 0 for leaf in _leaves(res)), [x.shape for x in _leaves(res)]


@gpu
def test_residuals_keep_a_real_logical_array():
    from jax_aiter.mha.mha import _favr_fwd

    q, k, v, seqstart = _tight_inputs()
    cu = seqstart - jnp.arange(ROWS + 1, dtype=jnp.int32)  # one padding token per row
    _, res = jax.jit(_favr_fwd, static_argnums=(7, 8, 9, 10, 11, 12))(
        q, k, v, seqstart, seqstart, cu, cu, *STATIC)
    np.testing.assert_array_equal(np.asarray(res[-2]), np.asarray(cu))
    np.testing.assert_array_equal(np.asarray(res[-1]), np.asarray(cu))


# Run in a child process: the failure mode is a segfault inside the XLA compile,
# which would otherwise take the whole pytest session down with it.
_SHARD_MAP_GRAD = textwrap.dedent("""
    import numpy as np, jax, jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from jax_aiter.mha import flash_attn_varlen_raw

    jax.config.update("jax_use_shardy_partitioner", True)
    SEQ, HEADS, DIM, ROWS_PER_DEV = 128, 2, 128, 2
    devices = jax.devices()
    mesh = jax.make_mesh((len(devices),), ("fsdp",))
    batch = ROWS_PER_DEV * len(devices)
    spec = P("fsdp", None, None, None)

    def attend(q, k, v):
        rows = q.shape[0]
        seqstart = jnp.arange(rows + 1, dtype=jnp.int32) * SEQ
        flat = lambda x: x.reshape(rows * SEQ, HEADS, DIM)
        out = flash_attn_varlen_raw(flat(q), flat(k), flat(v), seqstart, seqstart,
                                    None, None, SEQ, SEQ, 0.0, DIM ** -0.5, True, (-1, -1))
        return out.reshape(rows, SEQ, HEADS, DIM)

    sharded = jax.shard_map(attend, mesh=mesh, in_specs=(spec,) * 3, out_specs=spec,
                            check_vma=False)

    rng = np.random.default_rng(0)
    q, k, v, dout = (rng.normal(size=(batch, SEQ, HEADS, DIM)).astype(np.float32)
                     for _ in range(4))
    put = lambda x: jax.device_put(jnp.asarray(x, jnp.bfloat16), NamedSharding(mesh, spec))
    weight = put(dout).astype(jnp.float32)

    def loss(fn, q_, k_, v_, w_):
        return jnp.sum(fn(q_, k_, v_).astype(jnp.float32) * w_)

    got = jax.jit(jax.grad(lambda *a: loss(sharded, *a), argnums=(0, 1, 2)))(
        put(q), put(k), put(v), weight)
    local = lambda x: jnp.asarray(x, jnp.bfloat16)
    ref = jax.jit(jax.grad(lambda *a: loss(attend, *a), argnums=(0, 1, 2)))(
        local(q), local(k), local(v), jnp.asarray(dout, jnp.bfloat16).astype(jnp.float32))
    worst = 0.0
    for g, r in zip(got, ref):
        g, r = np.asarray(g, np.float32), np.asarray(r, np.float32)
        assert np.isfinite(g).all()
        worst = max(worst, float(np.linalg.norm(g - r) / np.linalg.norm(r)))
    print(f"SHARD_MAP_GRAD devices={len(devices)} worst_rel={worst:.3e}")
    assert worst < 1e-2, worst
""")


@gpu
@pytest.mark.multigpu
@pytest.mark.skipif(len(jax.devices()) < 2, reason="shard_map over a mesh needs >= 2 devices")
def test_grad_inside_shard_map_compiles_under_shardy():
    proc = subprocess.run([sys.executable, "-c", _SHARD_MAP_GRAD],
                          capture_output=True, text=True, timeout=900)
    log = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"exit {proc.returncode}\n{log[-4000:]}"
    assert "SHARD_MAP_GRAD devices=" in log, log[-4000:]
