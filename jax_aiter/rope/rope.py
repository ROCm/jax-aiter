# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""NEOX RoPE with custom_vjp via the self-contained RopeFwdJA HIP kernel.

Forward:  out = x*cos + rotate_half(x)*sin (matches MaxText apply_rotary).
Backward: RoPE is a per-(j, j+half) orthogonal 2x2 rotation [[c,-s],[s,c]];
          its adjoint is [[c, s],[-s, c]] = the SAME forward kernel with sin
          negated. So grad_x = rope_fwd(grad_out, cos, -sin). cos/sin are
          constants (built from positions) -> nondiff.
"""

from __future__ import annotations
from functools import partial

import jax

from ..ops.rope import rope_fwd as _rope_fwd_op


@partial(jax.custom_vjp, nondiff_argnums=())
def rope(x, cos, sin):
    """NEOX RoPE. x:[B,S,N,D] bf16, cos/sin:[B,S,D] bf16 full-width."""
    return _rope_fwd_op(x, cos, sin)


def _rope_fwd(x, cos, sin):
    return _rope_fwd_op(x, cos, sin), (cos, sin)


def _rope_bwd(residuals, grad_out):
    cos, sin = residuals
    # Adjoint of the rotation = forward with sin negated.
    grad_x = _rope_fwd_op(grad_out, cos, (-sin).astype(sin.dtype))
    # cos/sin are positional constants; pass zero cotangents.
    return grad_x, jax.numpy.zeros_like(cos), jax.numpy.zeros_like(sin)


rope.defvjp(_rope_fwd, _rope_bwd)
