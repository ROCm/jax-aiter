# SPDX-License-Identifier: MIT
"""Coverage for jax_aiter.ops.uninitialized."""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("flydsl.compiler")

from jax_aiter.ops import uninitialized


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32, jnp.int32])
def test_uninitialized_shape_and_dtype(dtype):
    out = jax.jit(
        lambda d: uninitialized((1000, 7168), dtype, depends_on=d)
    )(jnp.arange(8, dtype=jnp.int32))
    out.block_until_ready()
    assert out.shape == (1000, 7168)
    assert out.dtype == dtype


def test_uninitialized_is_fresh_per_scan_iteration():
    rows, cols, steps = 64, 128, 4

    def body(carry, step):
        buffer = uninitialized((rows, cols), jnp.float32, depends_on=step)
        value = jnp.full((8, cols), step.astype(jnp.float32) + 1.0)
        buffer = jax.lax.dynamic_update_slice(buffer, value, (step * 8, 0))
        row = jnp.arange(rows)[:, None]
        written = (row >= step * 8) & (row < step * 8 + 8)
        return carry + jnp.sum(jnp.where(written, buffer, 0.0)), None

    fn = jax.jit(lambda: jax.lax.scan(body, jnp.float32(0), jnp.arange(steps))[0])
    hlo = fn.lower().compile().as_text()
    assert "jax_aiter_uninitialized" in hlo or "custom-call" in hlo
    expected = sum(8 * cols * (step + 1.0) for step in range(steps))
    np.testing.assert_allclose(float(fn()), expected)


def test_uninitialized_distinct_keys_are_not_merged():
    rows, cols = 64, 128

    def fn(sizes):
        outputs = []
        for key in (0, 1):
            buffer = uninitialized((rows, cols), jnp.float32, depends_on=sizes, key=key)
            value = jnp.full((8, cols), key + 1.0, jnp.float32)
            outputs.append(jax.lax.dynamic_update_slice(buffer, value, (sizes[0], 0)))
        return tuple(outputs)

    sizes = jnp.array([8, 0], dtype=jnp.int32)
    compiled = jax.jit(fn).lower(sizes).compile()
    hlo = compiled.as_text()
    assert hlo.count('custom_call_target="FlydslDispatchJA"') == 2
    assert not re.search(r"= f32\[64,128\]\{1,0\} copy\(", hlo)
    first, second = compiled(sizes)
    np.testing.assert_array_equal(np.asarray(first)[8:16], 1.0)
    np.testing.assert_array_equal(np.asarray(second)[8:16], 2.0)
