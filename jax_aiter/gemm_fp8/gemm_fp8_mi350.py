# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""MI350 FP8 block-scale GEMM via AITER ASM kernels with custom_vjp.

Forward: Out = dequant(A, a_scale) @ dequant(B, b_scale)^T using FP8 ASM kernels.
Backward: Gradients computed in BF16 via the BF16 GEMM path (standard mixed-precision
training pattern -- forward in FP8 for speed, backward in BF16 for accuracy).

Constraints:
  - A: [M, K] float8_e4m3fn, B: [N, K] float8_e4m3fn
  - x_scale: [K/128, M] float32, w_scale: [K/128, N/128] float32
  - N divisible by 256, K divisible by 128, M >= 16, K >= 512
  - Output: [M, N] bfloat16
"""

from __future__ import annotations
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import NamedSharding, PartitionSpec as P

from ..ffi.registry import register_ffi_target


def _ensure_registered():
    register_ffi_target("GemmFp8Mi350FwdJA", "ROCM")


def _gemm_fp8_fwd_call(out_shape):
    call = jax.ffi.ffi_call(
        "GemmFp8Mi350FwdJA",
        jax.ShapeDtypeStruct(out_shape, jnp.bfloat16),
        vmap_method="broadcast_all",
    )
    return jax.jit(call)


def _gemm_fp8_raw(xq, wq, x_scale, w_scale):
    """Raw FP8 GEMM FFI call."""
    _ensure_registered()
    M, K = xq.shape
    N = wq.shape[0]
    fn = _gemm_fp8_fwd_call((M, N))
    return fn(xq, wq, x_scale, w_scale)


# ---------------------------------------------------------------------------
# custom_partitioning for FP8 GEMM.
# Same sharding semantics as BF16: M freely shardable, K replicated.
# ---------------------------------------------------------------------------
@custom_partitioning
def _gemm_fp8_partitioned(xq, wq, x_scale, w_scale):
    return _gemm_fp8_raw(xq, wq, x_scale, w_scale)


def _fp8_resolve_specs(xq_spec, wq_spec):
    m_axis = xq_spec[0]
    n_axis = wq_spec[0]
    if m_axis is not None and m_axis == n_axis:
        n_axis = None
    return (P(m_axis, None), P(n_axis, None),
            P(None, m_axis), P(None, None),
            P(m_axis, n_axis))


def _fp8_infer_sharding(mesh, arg_shapes, result_shape):
    xq_info, wq_info = arg_shapes[0], arg_shapes[1]
    *_, out_spec = _fp8_resolve_specs(xq_info.sharding.spec, wq_info.sharding.spec)
    return NamedSharding(mesh, out_spec)


def _fp8_partition(mesh, arg_shapes, result_shape):
    xq_info, wq_info = arg_shapes[0], arg_shapes[1]
    xq_pspec, wq_pspec, xs_pspec, ws_pspec, out_pspec = _fp8_resolve_specs(
        xq_info.sharding.spec, wq_info.sharding.spec)

    def _lowered(xq, wq, x_scale, w_scale):
        return _gemm_fp8_raw(xq, wq, x_scale, w_scale)

    return (mesh, _lowered,
            NamedSharding(mesh, out_pspec),
            (NamedSharding(mesh, xq_pspec), NamedSharding(mesh, wq_pspec),
             NamedSharding(mesh, xs_pspec), NamedSharding(mesh, ws_pspec)))


_gemm_fp8_partitioned.def_partition(
    _fp8_partition,
    infer_sharding_from_operands=_fp8_infer_sharding,
    sharding_rule="m k, n k, kblock m, kblock nblock -> m n",
    need_replication_factors=("k", "kblock", "nblock"),
)


# ---------------------------------------------------------------------------
# Public API with custom_vjp for training.
# Forward: FP8 GEMM (fast) using shuffled FP8 weight.
# Backward: BF16 GEMM using original (unshuffled) BF16 activations/weights.
#
# The caller passes both the shuffled FP8 weight (for forward) and the
# original BF16 tensors (for backward). This avoids casting shuffled FP8
# data to BF16 in the backward, which would produce garbled values.
# ---------------------------------------------------------------------------
@partial(jax.custom_vjp, nondiff_argnums=())
def gemm_fp8_mi350(
    xq: jnp.ndarray,
    wq: jnp.ndarray,
    x_scale: jnp.ndarray,
    w_scale: jnp.ndarray,
    a_bf16: jnp.ndarray,
    b_bf16: jnp.ndarray,
) -> jnp.ndarray:
    """FP8 block-scale GEMM for MI350 with training support.

    Args:
        xq: [M, K] float8_e4m3fn quantized activations
        wq: [N, K] float8_e4m3fn quantized + shuffled weights
        x_scale: [K/128, M] float32 activation block scales
        w_scale: [N/128, K/128] float32 weight block scales
        a_bf16: [M, K] bfloat16 original activations (for backward)
        b_bf16: [N, K] bfloat16 original weights (for backward)

    Returns:
        out: [M, N] bfloat16
    """
    return _gemm_fp8_partitioned(xq, wq, x_scale, w_scale)


def _gemm_fp8_fwd_rule(xq, wq, x_scale, w_scale, a_bf16, b_bf16):
    out = gemm_fp8_mi350(xq, wq, x_scale, w_scale, a_bf16, b_bf16)
    return out, (a_bf16, b_bf16, x_scale, w_scale)


def _gemm_fp8_bwd_rule(residuals, grad_out):
    a_bf16, b_bf16, x_scale, w_scale = residuals
    g = grad_out.astype(jnp.bfloat16)

    # Use lax.dot_general for backward -- XLA handles transposed layouts natively
    # (no explicit transpose kernels needed, unlike AITER which only supports A @ B^T).
    # dA = grad_out @ B  (contract on N dimension)
    da = jax.lax.dot_general(g, b_bf16, (((1,), (0,)), ((), ())))
    # dB = grad_out^T @ A  (contract on M dimension)
    db = jax.lax.dot_general(g, a_bf16, (((0,), (0,)), ((), ()))).astype(jnp.bfloat16)

    return (jnp.zeros_like(a_bf16, dtype=jnp.float8_e4m3fn),
            jnp.zeros_like(b_bf16, dtype=jnp.float8_e4m3fn),
            jnp.zeros_like(x_scale),
            jnp.zeros_like(w_scale),
            da, db)


gemm_fp8_mi350.defvjp(_gemm_fp8_fwd_rule, _gemm_fp8_bwd_rule)
