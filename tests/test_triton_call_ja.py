# SPDX-License-Identifier: Apache-2.0
"""Host-side checks of the Triton launch packing; no kernel runs."""

from __future__ import annotations

import numpy as np
import pytest

from jax_aiter.triton import call


def _kernel(x_ptr, out_ptr, n, BLOCK):
    pass


_kernel.arg_names = ("x_ptr", "out_ptr", "n", "BLOCK")


def _launch(runtime_types, constexprs):
    return call.triton_call(
        fn=_kernel,
        runtime_types=runtime_types,
        constexprs=constexprs,
        kinds=("buffer",) * (len(runtime_types) - 1) + ("scalar",),
        scalars={"n": 8},
        out_shape=(),
        output_names=(),
        grid=(1, 1, 1),
    )


@pytest.mark.parametrize("value", [-1.0, -0.0, 1.5, -2.5e300])
def test_fp64_scalar_round_trips_through_int64(value):
    packed = np.asarray([call._scalar_to_int64("fp64", value)], dtype=np.int64)
    unpacked = packed.view(np.float64)[0]
    assert unpacked == value
    assert np.signbit(unpacked) == np.signbit(value)


@pytest.mark.parametrize(
    "runtime_types, constexprs",
    [
        ({"out_ptr": "*fp32", "x_ptr": "*fp32", "n": "i32"}, {"BLOCK": 64}),
        ({"x_ptr": "*fp32", "out_ptr": "*fp32", "BLOCK": "i32", "n": "i32"}, {"BLOCK": 64}),
    ],
    ids=["out_of_order", "constexpr_as_runtime"],
)
def test_runtime_types_must_follow_the_signature(runtime_types, constexprs):
    with pytest.raises(ValueError, match="signature order"):
        _launch(runtime_types, constexprs)


def test_runtime_types_in_signature_order_pass_the_check(monkeypatch):
    class Reached(Exception):
        pass

    def stop():
        raise Reached

    monkeypatch.setattr(call, "prepare_aiter_triton", stop)
    with pytest.raises(Reached):
        _launch({"x_ptr": "*fp32", "out_ptr": "*fp32", "n": "i32"}, {"BLOCK": 64})
