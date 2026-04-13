# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""BF16 GEMM via AITER ASM kernels with custom_vjp + custom_partitioning.

Forward: Out = A @ B^T using hand-tuned ASM kernels via FFI.
Backward:
  dA = dOut @ B  -- always AITER ASM GEMM
  dB = dOut^T @ A -- multi-backend, selected by AITER_DB_BACKEND env var:
    "hipblaslt" (default) -- lax.dot_general → hipBLASLt, XLA-pipelined
    "ck"                  -- CK Col/Row/Row zero-copy GEMM via FFI
    "triton"              -- Triton NN-layout GEMM via jax_triton
    "fused"               -- Fused dA+dB (closed: OOM at production scale)

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


_DB_BACKEND_CACHE = None


def _get_db_backend():
    """Select the backward dB backend.

    Checked once at first call, then cached. Priority:

    1. AITER_DB_BACKEND env var (explicit selection):
       "hipblaslt" -- native lax.dot_general → hipBLASLt (default, XLA-pipelined)
       "ck"        -- CK Col/Row/Row zero-copy GEMM via FFI (no workspace)
       "triton"    -- Triton NN-layout GEMM via jax_triton (requires triton)
       "fused"     -- Fused dA+dB in one FFI call (closed: OOM at production scale)

    2. Legacy env var fallback (backward compat):
       AITER_CK_DB=1      → "ck"
       AITER_TRITON_DB=1   → "triton"
       AITER_FUSED_BWD=1   → "fused"
    """
    global _DB_BACKEND_CACHE
    if _DB_BACKEND_CACHE is not None:
        return _DB_BACKEND_CACHE

    import os
    backend = os.environ.get("AITER_DB_BACKEND", "").lower()

    if backend in ("hipblaslt", "ck", "fused"):
        _DB_BACKEND_CACHE = backend
        return backend

    if backend == "triton":
        from ..triton import is_triton_available
        if is_triton_available():
            _DB_BACKEND_CACHE = "triton"
            return "triton"
        _DB_BACKEND_CACHE = "hipblaslt"
        return "hipblaslt"

    if os.environ.get("AITER_CK_DB", "0") == "1":
        _DB_BACKEND_CACHE = "ck"
    elif os.environ.get("AITER_FUSED_BWD", "0") == "1":
        _DB_BACKEND_CACHE = "fused"
    elif os.environ.get("AITER_TRITON_DB", "0") == "1":
        from ..triton import is_triton_available
        if is_triton_available():
            _DB_BACKEND_CACHE = "triton"
        else:
            _DB_BACKEND_CACHE = "hipblaslt"
    else:
        _DB_BACKEND_CACHE = "hipblaslt"

    return _DB_BACKEND_CACHE


def gemm_ck_db(g, a):
    """CK dB: dB[N,K] = g[M,N]^T @ a[M,K] via zero-copy Col/Row/Row GEMM.

    Row-major g[M,N] is reinterpreted as col-major A[N,M] by CK without data
    movement. No transpose kernel, no workspace buffer.
    """
    from ..ffi.registry import register_ffi_target
    register_ffi_target("GemmCkDbJA", "ROCM")
    N = g.shape[1]
    K = a.shape[1]
    call = jax.ffi.ffi_call(
        "GemmCkDbJA",
        jax.ShapeDtypeStruct((N, K), jnp.bfloat16),
        vmap_method="broadcast_all",
    )
    return call(g, a)


def _gemm_fwd(a, b):
    out = gemm(a, b)
    # Save B untransposed — avoid storing B^T through the scan carry.
    # B is a closed-over weight constant in scan, so saving it directly is cheaper.
    # The transpose needed for backward dA is done in _gemm_bwd instead.
    return out, (a, b)


def _gemm_bwd(residuals, grad_out):
    a, b = residuals
    backend = _get_db_backend()

    # Forward was: Out[M,N] = A[M,K] @ B[N,K]^T

    # Fused backward computes both dA and dB in one FFI call.
    # Closed at production scale (OOM), kept behind AITER_DB_BACKEND=fused.
    if backend == "fused":
        from ..gemm_bwd_fused import gemm_bwd_fused
        g = grad_out.astype(jnp.bfloat16)
        da, db = gemm_bwd_fused(g, b, a)
        return da, db

    # --- dA: AITER ASM by default, hipBLASLt if AITER_HIPBLASLT_DA=1 ---
    # dA[M,K] = grad_out[M,N] @ B[N,K]
    import os
    if os.environ.get("AITER_HIPBLASLT_DA", "0") == "1":
        da = jax.lax.dot_general(grad_out, b, (((1,), (0,)), ((), ())))
    else:
        b_t = jnp.transpose(b, (1, 0))
        da = gemm(grad_out, b_t)

    # --- dB: backend-selected ---
    if backend == "ck":
        db = gemm_ck_db(grad_out.astype(jnp.bfloat16), a)
    elif backend == "triton":
        from ..triton import gemm_db_triton
        db = gemm_db_triton(grad_out, a)
    elif os.environ.get("AITER_FP8_DB", "0") == "1":
        from ..gemm_fp4.gemm_fp4 import _fp8_dot_general_db
        db = _fp8_dot_general_db(grad_out, a)
    else:
        # hipblaslt (default): XLA-native, gets pipelined across scan layers
        db = jax.lax.dot_general(grad_out, a, (((0,), (0,)), ((), ())))

    return da, db


gemm.defvjp(_gemm_fwd, _gemm_bwd)
