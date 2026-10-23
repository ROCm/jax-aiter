# SPDX-License-Identifier: Apache-2.0
"""FLA Gated DeltaNet through TritonDispatchJA against a float32 recurrence."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_aiter.ja_compat import config as ja_config


def _bridge_present() -> bool:
    try:
        return (ja_config.get_jax_aiter_lib_dir() / "triton_bridge_ja.so").is_file()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _bridge_present(),
    reason="triton_bridge_ja.so missing; run make -f Makefile.triton bridge",
)


def _reference(q, k, v, g, beta, scale):
    """Token-by-token gated delta rule (FLA naive_recurrent_gated_delta_rule)."""
    groups = v.shape[2] // q.shape[2]
    q = jnp.repeat(q, groups, axis=2).astype(jnp.float32) * scale
    k = jnp.repeat(k, groups, axis=2).astype(jnp.float32)
    v, g, beta = (x.astype(jnp.float32) for x in (v, g, beta))
    b, _, hv, dk = q.shape
    state = jnp.zeros((b, hv, dk, v.shape[-1]), jnp.float32)

    def step(h, xs):
        q_t, k_t, v_t, g_t, b_t = xs
        h = h * jnp.exp(g_t)[..., None, None]
        v_t = (v_t - jnp.einsum("bhk,bhkv->bhv", k_t, h)) * b_t[..., None]
        h = h + k_t[..., :, None] * v_t[..., None, :]
        return h, jnp.einsum("bhk,bhkv->bhv", q_t, h)

    xs = tuple(jnp.moveaxis(x, 1, 0) for x in (q, k, v, g, beta))
    _, o = jax.lax.scan(step, state, xs)
    return jnp.moveaxis(o, 0, 1)


def _inputs(b, t, h, hv, d, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 6)
    l2 = lambda x: x / jnp.linalg.norm(x, axis=-1, keepdims=True)
    q = l2(jax.random.normal(keys[0], (b, t, h, d))).astype(jnp.bfloat16)
    k = l2(jax.random.normal(keys[1], (b, t, h, d))).astype(jnp.bfloat16)
    v = jax.random.normal(keys[2], (b, t, hv, d)).astype(jnp.bfloat16)
    g = -0.5 * jax.nn.softplus(jax.random.normal(keys[3], (b, t, hv)))
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (b, t, hv))).astype(jnp.bfloat16)
    do = jax.random.normal(keys[5], (b, t, hv, d)).astype(jnp.bfloat16)
    return q, k, v, g.astype(jnp.float32), beta, do


def _rel(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


@pytest.mark.parametrize("h,hv", [(4, 4), (2, 8)])
def test_gdn_fla_matches_recurrence(h, hv):
    from jax_aiter.triton.gdn import chunk_gated_delta_rule_fla

    b, t, d = 2, 256, 128
    q, k, v, g, beta, do = _inputs(b, t, h, hv, d)
    scale = d ** -0.5

    out, vjp = jax.vjp(
        lambda *xs: chunk_gated_delta_rule_fla(*xs, scale=scale), q, k, v, g, beta
    )
    ref, ref_vjp = jax.vjp(lambda *xs: _reference(*xs, scale), q, k, v, g, beta)

    assert out.shape == (b, t, hv, d) and out.dtype == v.dtype
    assert _rel(out, ref) < 1e-2

    grads = vjp(do)
    ref_grads = ref_vjp(do.astype(jnp.float32))
    for name, got, want in zip("q k v g beta".split(), grads, ref_grads):
        assert got.shape == want.shape, name
        assert np.isfinite(np.asarray(got, np.float32)).all(), name
        assert _rel(got, want) < 2e-2, (name, _rel(got, want))
