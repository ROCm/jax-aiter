# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""BF16 GEMM via AITER ASM kernels with custom_vjp + custom_partitioning.

Forward: Out = A @ B^T using hand-tuned ASM kernels via FFI.
Backward:
  dA = dOut @ B  -- always AITER ASM GEMM
  dB = dOut^T @ A -- multi-backend, selected by AITER_DB_BACKEND env var:
    "hipblaslt" (default) -- lax.dot_general → hipBLASLt, XLA-pipelined
    "aiter"               -- AITER bf16 ASM GEMM (transpose + FSDP-aware psum)
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
       "aiter"     -- AITER bf16 ASM GEMM (transpose + FSDP-aware psum)
       "triton"    -- Triton NN-layout GEMM via jax_triton (requires triton)
       "fused"     -- Fused dA+dB in one FFI call (closed: OOM at production scale)

    2. Legacy env var fallback (backward compat):
       AITER_TRITON_DB=1   → "triton"
       AITER_FUSED_BWD=1   → "fused"
    """
    global _DB_BACKEND_CACHE
    if _DB_BACKEND_CACHE is not None:
        return _DB_BACKEND_CACHE

    import os
    backend = os.environ.get("AITER_DB_BACKEND", "").lower()

    if backend in ("hipblaslt", "fused", "aiter"):
        _DB_BACKEND_CACHE = backend
        return backend

    if backend == "triton":
        from ..triton import is_triton_available
        if is_triton_available():
            _DB_BACKEND_CACHE = "triton"
            return "triton"
        _DB_BACKEND_CACHE = "hipblaslt"
        return "hipblaslt"

    if os.environ.get("AITER_FUSED_BWD", "0") == "1":
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


# ---------------------------------------------------------------------------
# BF16 GEMM wgrad (NT layout, M_batch is contraction -- FSDP-sharded).
#
#   grad_out [M, N]  (raw bf16 cotangent)
#   a        [M, K]  (raw bf16 activation, stashed by the fwd)
#   OUT = dB [N, K] = sum_M grad_out[M,N] * a[M,K]   (contraction over M)
#
# AITER bf16 twin of _fp4_ffi_partitioned_wgrad (gemm_fp4.py). The kernel
# computes Out[m',n'] = A[m',k'] @ B[n',k']^T with k' = trailing contraction
# axis, so we feed it the transposed operands gradT=[N,M] and aT=[K,M]:
#   dB[N,K] = gemm_bf16(gradT[N,M], aT[K,M])  (m'=N, n'=K, k'=M trailing).
#
# The contraction axis M is the FSDP-sharded batch/token axis, so the
# per-shard partial products MUST be reduced across the FSDP mesh: the
# partition callback emits jax.lax.psum over the M mesh axis (the role the FP4
# wgrad's explicit psum plays; a MISSING reduction gives ~0.5 norm-ratio in
# the 2-device check). dB is declared REPLICATED across the contraction
# (out_pspec drops the M axis). On a single device (no sharded axis) psum_axes
# is empty => plain local GEMM.
#
# Operand layouts: gradT=[N,M] (axis0=N output-feature, axis1=M contraction),
# aT=[K,M] (axis0=K, axis1=M contraction), out=[N,K].
# ---------------------------------------------------------------------------

# GemmFwdJA addresses operands/output with 32-bit unsigned offsets, so any
# buffer of >= 2**31 elements wraps and silently corrupts. gemm_bf16 already
# tiles when the OUTPUT overflows; in the wgrad mapping the transposed grad
# operand grad_t[N,M] is the large buffer (logits: 128256*32768 = 4.2e9 elems)
# while the output dB[N,K] stays small, so the output-only guard never fires.
# Tile grad_t over its leading axis (N, the dB output-feature axis -- NOT the
# contraction) so every kernel call's grad_t tile stays below 2**31 elements,
# then concatenate along N. Under FSDP the per-shard M is small so this is a
# no-op; it only engages for the full-M single-device path.
_WGRAD_OVERFLOW_LIMIT = 2 ** 31
_WGRAD_DEFAULT_TILE = 16384


def _wgrad_raw(grad_t, a_t):
    n, m = grad_t.shape
    if n * m < _WGRAD_OVERFLOW_LIMIT:
        return _gemm_raw(grad_t, a_t)
    tile_rows = min(_WGRAD_DEFAULT_TILE, max(1, (_WGRAD_OVERFLOW_LIMIT - 1) // max(int(m), 1)))
    outs = []
    for start in range(0, n, tile_rows):
        rows = min(tile_rows, n - start)
        outs.append(_gemm_raw(grad_t[start:start + rows], a_t))
    return jnp.concatenate(outs, axis=0)


@custom_partitioning
def _bf16_ffi_partitioned_wgrad(grad_t, a_t):
    return _wgrad_raw(grad_t, a_t)


def _spec_of(info):
    """Safely extract a PartitionSpec from an arg_shape (None sharding => all-None)."""
    if info.sharding is None:
        return P(*([None] * len(info.shape)))
    return info.sharding.spec


def _axis_at(spec, dim_index):
    if spec is None or dim_index >= len(spec):
        return None
    return spec[dim_index]


def _bf16_wgrad_infer_sharding(mesh, arg_shapes, result_shape):
    grad_spec = _spec_of(arg_shapes[0])
    a_spec = _spec_of(arg_shapes[1])
    n_axis = _axis_at(grad_spec, 0)
    k_axis = _axis_at(a_spec, 0)
    return NamedSharding(mesh, P(n_axis, k_axis))


def _bf16_wgrad_partition(mesh, arg_shapes, result_shape):
    grad_spec = _spec_of(arg_shapes[0])
    a_spec = _spec_of(arg_shapes[1])

    n_axis = _axis_at(grad_spec, 0)      # output-feature axis of gradT[N,M]
    k_axis = _axis_at(a_spec, 0)         # K axis of aT[K,M]
    m_axis_g = _axis_at(grad_spec, 1)    # contraction axis (M) on gradT
    m_axis_a = _axis_at(a_spec, 1)       # contraction axis (M) on aT

    m_axis = m_axis_g if m_axis_g is not None else m_axis_a
    if m_axis_g is not None and m_axis_a is not None and m_axis_g != m_axis_a:
        m_axis = m_axis_g

    # If N and K share a mesh axis, drop the overlap from K (no-duplicate rule).
    n_set = (set(n_axis) if isinstance(n_axis, tuple)
             else {n_axis} if n_axis is not None else set())
    k_set = (set(k_axis) if isinstance(k_axis, tuple)
             else {k_axis} if k_axis is not None else set())
    if n_set & k_set:
        remaining = k_set - n_set
        k_axis = (next(iter(remaining)) if len(remaining) == 1
                  else tuple(sorted(remaining)) if remaining else None)

    grad_pspec = P(n_axis, m_axis)
    a_pspec = P(k_axis, m_axis)
    out_pspec = P(n_axis, k_axis)

    if m_axis is None:
        psum_axes = ()
    elif isinstance(m_axis, tuple):
        psum_axes = m_axis
    else:
        psum_axes = (m_axis,)

    def _lowered(grad_t, a_t):
        partial = _wgrad_raw(grad_t, a_t)
        if psum_axes:
            partial = jax.lax.psum(partial, axis_name=psum_axes)
        return partial

    return (mesh, _lowered,
            NamedSharding(mesh, out_pspec),
            (NamedSharding(mesh, grad_pspec), NamedSharding(mesh, a_pspec)))


_bf16_ffi_partitioned_wgrad.def_partition(
    _bf16_wgrad_partition,
    infer_sharding_from_operands=_bf16_wgrad_infer_sharding,
    sharding_rule="n m, k m -> n k",
    need_replication_factors=(),
)


def _bf16_wgrad_aiter(grad_out, a):
    """dB[N,K] = sum_M grad_out[M,N] * a[M,K] via the AITER bf16 ASM GEMM.

    Transposes the operands into the kernel's trailing-contraction layout
    (gradT=[N,M], aT=[K,M]) and routes through the FSDP-aware partitioned FFI,
    which psum-reduces the sharded M contraction. bf16 in/out to match the
    custom_vjp cotangent dtype.
    """
    grad_t = jnp.transpose(grad_out.astype(jnp.bfloat16), (1, 0))
    a_t = jnp.transpose(a.astype(jnp.bfloat16), (1, 0))
    db = _bf16_ffi_partitioned_wgrad(grad_t, a_t)
    return db.astype(grad_out.dtype)


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
    if backend == "aiter":
        # AITER bf16 ASM WGRAD: transpose operands into trailing-contraction
        # layout and route through the FSDP-aware partitioned FFI (psum over M).
        db = _bf16_wgrad_aiter(grad_out, a)
    elif backend == "triton":
        from ..triton import gemm_db_triton
        db = gemm_db_triton(grad_out, a)
    else:
        # hipblaslt (default): XLA-native, gets pipelined across scan layers
        db = jax.lax.dot_general(grad_out, a, (((0,), (0,)), ((), ())))

    return da, db


gemm.defvjp(_gemm_fwd, _gemm_bwd)
