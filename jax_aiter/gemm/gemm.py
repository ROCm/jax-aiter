# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""BF16 GEMM via AITER ASM kernels with custom_vjp + custom_partitioning.

Forward: Out = A @ B^T using hand-tuned ASM kernels via FFI.
Backward:
  dA = dOut @ B   -- AITER ASM GEMM
  dB = dOut^T @ A -- lax.dot_general → hipBLASLt, so XLA pipelines it across
                     scan layers

GSPMD sharding: custom_partitioning tells XLA how to partition the FFI call.
  Out[M,N] = A[M,K] @ B[N,K]^T
  - M dimension: sharded freely (batch * seq)
  - N dimension: sharded freely (output features)
  - K dimension: contraction -- must be replicated on both A and B

Constraints:
  - A: [M, K] bf16, B: [N, K] bf16 -> Out: [M, N] bf16
  - K must be divisible by 64
  - Computes A @ B^T (B is in [N, K] layout, not transposed)
"""

from __future__ import annotations
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import NamedSharding, PartitionSpec as P

from ..ops.gemm_bf16 import gemm_bf16 as _gemm_raw


# ---------------------------------------------------------------------------
# custom_partitioning wrapper for GSPMD sharding support.
#
# Out[M,N] = A[M,K] @ B[N,K]^T
#   sharding_rule "m k, n k -> m n":
#     m = batch*seq dimension, freely shardable
#     n = output features dimension, freely shardable
#     k = contraction dimension, must be replicated (need_replication)
#
# partition callback: tells XLA how to lower to per-shard computation.
#   K must be replicated; M and N follow input shardings.
# ---------------------------------------------------------------------------
@custom_partitioning
def _gemm_partitioned(a, b):
    return _gemm_raw(a, b)


def _resolve_specs(a_spec, b_spec):
    """Resolve input/output PartitionSpecs for GEMM.

    A[M,K] @ B[N,K]^T -> Out[M,N]
    K must always be replicated. If M and N share any mesh axis,
    drop the overlapping axes from N to satisfy the no-duplicate constraint.
    """
    m_axis = a_spec[0]
    n_axis = b_spec[0]

    m_set = (set(m_axis) if isinstance(m_axis, tuple)
             else {m_axis} if m_axis is not None else set())
    n_set = (set(n_axis) if isinstance(n_axis, tuple)
             else {n_axis} if n_axis is not None else set())

    if m_set & n_set:
        remaining = n_set - m_set
        if not remaining:
            n_axis = None
        elif len(remaining) == 1:
            n_axis = next(iter(remaining))
        else:
            n_axis = tuple(sorted(remaining))

    return P(m_axis, None), P(n_axis, None), P(m_axis, n_axis)


def _gemm_infer_sharding(mesh, arg_shapes, result_shape):
    a_info, b_info = arg_shapes
    _, _, out_spec = _resolve_specs(a_info.sharding.spec, b_info.sharding.spec)
    return NamedSharding(mesh, out_spec)


def _gemm_partition(mesh, arg_shapes, result_shape):
    a_info, b_info = arg_shapes
    a_pspec, b_pspec, out_pspec = _resolve_specs(
        a_info.sharding.spec, b_info.sharding.spec)

    def _lowered(a, b):
        return _gemm_raw(a, b)

    return (mesh, _lowered,
            NamedSharding(mesh, out_pspec),
            (NamedSharding(mesh, a_pspec), NamedSharding(mesh, b_pspec)))


_gemm_partitioned.def_partition(
    _gemm_partition,
    infer_sharding_from_operands=_gemm_infer_sharding,
    sharding_rule="m k, n k -> m n",
    need_replication_factors=("k",),
)


# ---------------------------------------------------------------------------
# Public API with custom_vjp for gradient support.
# ---------------------------------------------------------------------------
@partial(jax.custom_vjp, nondiff_argnums=())
def gemm(
    a: jnp.ndarray,
    b: jnp.ndarray,
) -> jnp.ndarray:
    """Compute A @ B^T using AITER ASM GEMM.

    Supports GSPMD sharding: M and N dimensions can be sharded across devices.
    K (contraction) dimension is automatically replicated.

    Args:
        a: [M, K] bf16
        b: [N, K] bf16

    Returns:
        out: [M, N] bf16
    """
    return _gemm_partitioned(a, b)


def _gemm_fwd(a, b):
    out = gemm(a, b)
    # Save B untransposed — avoid storing B^T through the scan carry.
    # B is a closed-over weight constant in scan, so saving it directly is cheaper.
    # The transpose needed for backward dA is done in _gemm_bwd instead.
    return out, (a, b)


def _gemm_bwd(residuals, grad_out):
    a, b = residuals

    # Forward was: Out[M,N] = A[M,K] @ B[N,K]^T

    # dA[M,K] = grad_out[M,N] @ B[N,K]
    b_t = jnp.transpose(b, (1, 0))
    da = gemm(grad_out, b_t)

    # dB[N,K] = grad_out[M,N]^T @ A[M,K]. XLA-native so it pipelines across
    # scan layers.
    db = jax.lax.dot_general(grad_out, a, (((0,), (0,)), ((), ())))

    return da, db


gemm.defvjp(_gemm_fwd, _gemm_bwd)
