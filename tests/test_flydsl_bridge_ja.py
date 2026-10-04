# SPDX-License-Identifier: Apache-2.0
"""Kernel-independent smoke coverage for the typed FlyDSL bridge."""

from __future__ import annotations

import functools
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

flyc = pytest.importorskip("flydsl.compiler")
fx = pytest.importorskip("flydsl.expr")
from flydsl.expr.typing import T

from jax_aiter.flydsl import flydsl_call
from jax_aiter.flydsl.kernels.mxfp4 import _buffer_ops as buffer_ops

_BLOCK = 256


@functools.cache
def _copy_launcher():
    @flyc.kernel
    def copy_kernel(source: fx.Tensor, destination: fx.Tensor):
        index = fx.block_idx.x * fx.Int32(_BLOCK) + fx.thread_idx.x
        source_resource = buffer_ops.create_buffer_resource(source, max_size=True)
        destination_resource = buffer_ops.create_buffer_resource(
            destination, max_size=True
        )
        value = buffer_ops.buffer_load(
            source_resource, index, vec_width=1, dtype=T.f32
        )
        buffer_ops.buffer_store(value, destination_resource, index)

    @flyc.jit
    def launch(
        source: fx.Tensor,
        destination: fx.Tensor,
        size: fx.Int32,
        stream: fx.Stream,
    ):
        copy_kernel(source, destination).launch(
            grid=((size + _BLOCK - 1) // _BLOCK, 1, 1),
            block=(_BLOCK, 1, 1),
            stream=stream,
        )

    return launch


def _copy(value):
    return flydsl_call(
        value,
        kernel=_copy_launcher(),
        out_shape=jax.ShapeDtypeStruct(value.shape, value.dtype),
        scalars={"size": value.size},
    )


def test_flydsl_copy_jit():
    value = jnp.arange(_BLOCK, dtype=jnp.float32)
    actual = jax.jit(_copy)(value)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(value))


def test_flydsl_compile_hints_reach_trace_and_compile_key():
    from flydsl.compiler.kernel_function import CompilationContext

    traced_hints = []

    @flyc.kernel
    def copy_kernel(source: fx.Tensor, destination: fx.Tensor):
        index = fx.block_idx.x * fx.Int32(_BLOCK) + fx.thread_idx.x
        source_resource = buffer_ops.create_buffer_resource(source, max_size=True)
        destination_resource = buffer_ops.create_buffer_resource(
            destination, max_size=True
        )
        value = buffer_ops.buffer_load(
            source_resource, index, vec_width=1, dtype=T.f32
        )
        buffer_ops.buffer_store(value, destination_resource, index)

    @flyc.jit
    def launch(
        source: fx.Tensor,
        destination: fx.Tensor,
        size: fx.Int32,
        stream: fx.Stream,
    ):
        traced_hints.append(dict(CompilationContext.get_compile_hints()))
        copy_kernel(source, destination).launch(
            grid=((size + _BLOCK - 1) // _BLOCK, 1, 1),
            block=(_BLOCK, 1, 1),
            stream=stream,
        )

    def copy(value, compile_hints):
        return flydsl_call(
            value,
            kernel=launch,
            out_shape=jax.ShapeDtypeStruct(value.shape, value.dtype),
            scalars={"size": value.size},
            compile_hints=compile_hints,
        )

    def compile_key(lowered):
        return re.search(r"compile_key = (-?\d+)", lowered.as_text()).group(1)

    value = jnp.arange(_BLOCK, dtype=jnp.float32)
    hints = {"llvm_options": {"enable-post-misched": True}}
    plain = jax.jit(functools.partial(copy, compile_hints=None)).lower(value)
    assert traced_hints == [{}]
    hinted = jax.jit(functools.partial(copy, compile_hints=hints)).lower(value)
    assert traced_hints == [{}, hints]

    assert compile_key(plain) != compile_key(hinted)
    np.testing.assert_array_equal(
        np.asarray(hinted.compile()(value)), np.asarray(value)
    )
