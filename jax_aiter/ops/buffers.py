# SPDX-License-Identifier: MIT
"""Fresh device buffers whose contents are never initialized.

XLA materializes ``jnp.zeros``/``lax.empty`` as a fill, and inside a scanned
layer it may hoist that fill out of the loop. An op that writes its output
operand in place, such as ``ragged_all_to_all``, then forces a full copy of
the hoisted buffer on every iteration. :func:`uninitialized` instead returns
the output of a launch that writes nothing, so the buffer costs only its
allocation.
"""

import functools

import jax
import jax.numpy as jnp

from ..flydsl import flydsl_call


@functools.cache
def _uninitialized_launcher():
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    @flyc.kernel(name="jax_aiter_uninitialized")
    def noop_kernel(output: fx.Tensor):
        del output

    @flyc.jit
    def launch(
        depends_on: fx.Tensor,
        output: fx.Tensor,
        key: fx.Int32,
        stream: fx.Stream,
    ):
        del depends_on, key
        noop_kernel(output).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    return launch


def uninitialized(shape, dtype, *, depends_on: jax.Array, key: int = 0) -> jax.Array:
    """Return a ``shape``/``dtype`` buffer with unspecified contents.

    ``depends_on`` must be a value computed inside the same loop iteration as
    the buffer's consumer, so the buffer cannot be hoisted out of that loop.
    Its contents are not read. Every element the caller reads must first be
    written or masked.

    XLA merges calls with equal ``shape``, ``dtype`` and ``key`` whose
    ``depends_on`` values it can prove equal, and then copies the shared buffer
    for every in-place writer after the first. Give buffers that can be live at
    the same time distinct static ``key`` values.
    """
    depends_on = jnp.asarray(depends_on)
    return flydsl_call(
        depends_on.reshape(-1),
        kernel=_uninitialized_launcher(),
        out_shape=jax.ShapeDtypeStruct(tuple(int(d) for d in shape), dtype),
        scalars={"key": int(key)},
    )


__all__ = ["uninitialized"]
