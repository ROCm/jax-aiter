# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""BF16 GEMM via AITER ASM kernels with custom_vjp + custom_partitioning.

Forward: Out = A @ B^T using hand-tuned ASM kernels via FFI.
Backward:
  dA = dOut @ B -- AITER on 1-GPU (via pre-transposed B), hipBLASLt on multi-GPU
  dB = dOut^T @ A -- hipBLASLt (needs NN layout that AITER doesn't support)

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
import numpy as np
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import NamedSharding, PartitionSpec as P

from ..ffi.registry import register_ffi_target


def _ensure_registered():
    register_ffi_target("GemmFwdJA", "ROCM")


def _gemm_fwd_call(out_shape, sem_shape, dtype):
    call = jax.ffi.ffi_call(
        "GemmFwdJA",
        (
            jax.ShapeDtypeStruct(out_shape, dtype),
            jax.ShapeDtypeStruct(sem_shape, jnp.uint32),
        ),
        vmap_method="broadcast_all",
        input_layouts=[None, None],
        output_layouts=[None, None],
        has_side_effect=False,
    )

    def _invoke(a, b):
        out, _ = call(a, b)
        return out

    return jax.jit(_invoke)


# ---------------------------------------------------------------------------
# Raw (unpartitioned) GEMM -- used inside custom_partitioning lowering.
# ---------------------------------------------------------------------------
def _gemm_raw(a, b):
    """Raw FFI GEMM call. A[M,K] @ B[N,K]^T -> Out[M,N]."""
    _ensure_registered()
    M, K = a.shape
    N = b.shape[0]
    fn = _gemm_fwd_call((M, N), (16, 64), a.dtype)
    return fn(a, b)


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
    K must always be replicated. If M and N map to the same mesh axis,
    replicate B's N to avoid the duplicate-axis constraint.
    """
    m_axis = a_spec[0]
    n_axis = b_spec[0]
    if m_axis is not None and m_axis == n_axis:
        n_axis = None
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
    if jax.device_count() == 1:
        b_t = jnp.transpose(b, (1, 0))
        return out, (a, b_t)
    return out, (a, b)


def _gemm_bwd(residuals, grad_out):
    a, b_or_bt = residuals
    # Forward was: Out[M,N] = A[M,K] @ B[N,K]^T
    if jax.device_count() == 1:
        # 1-GPU: dA via AITER (pre-transposed B avoids layout issues)
        da = gemm(grad_out, b_or_bt)
        b = jnp.transpose(b_or_bt, (1, 0))
    else:
        # Multi-GPU: dA via hipBLASLt to preserve collective-matmul overlap
        b = b_or_bt
        da = jax.lax.dot_general(grad_out, b, (((1,), (0,)), ((), ())))
    # dB[N,K] = grad_out^T[N,M] @ A[M,K] -- always hipBLASLt (needs NN layout)
    db = jax.lax.dot_general(grad_out, a, (((0,), (0,)), ((), ())))

    return da, db


gemm.defvjp(_gemm_fwd, _gemm_bwd)
