# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""FP4 GEMM via AITER ASM kernels with custom_vjp + custom_partitioning.

Forward: Out[M,N] = A_bf16[M,K] @ B_bf16[N,K]^T
  - Quantize A and B to MXFP4 (fused HIP kernel or JAX ops)
  - Shuffle B weights + scales for ASM B-preshuffle layout
  - Launch AITER FP4 ASM kernel via FFI (inside custom_partitioning)

Backward:
  dA = grad_out @ B     via FP4 ASM with pre-computed columnwise weight
  dB = grad_out^T @ A   via FP4 ASM (default), hipBLASLt FP8, or BF16

The FP4 dB path uses NT layout wgrad:
  1. Transpose grad_out [M,N] -> [N,M], quantize rowwise (A operand)
  2. Transpose input [M,K] -> [K,M], quantize + B-preshuffle (B operand)
  3. FP4 GEMM: dB[N,K] = quant(grad_out^T) @ quant(input^T)^T
The transpose makes kernel M = N_proj (small), avoiding the M > 65536 issue.
"""

from __future__ import annotations
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import NamedSharding, PartitionSpec as P

from ..ops.gemm_fp4 import (
    gemm_fp4 as _gemm_fp4_ffi,
    cast_mxfp4 as _cast_mxfp4_op,
    cast_mxfp4_dual as _cast_mxfp4_dual_op,
)
from ..ffi.registry import register_ffi_target
from .fp4_utils import bf16_to_mxfp4, e8m0_shuffle, shuffle_weight


_FUSED_QUANT_CACHE = None


def _use_fused_quant():
    """Use fused HIP kernel for MXFP4 quantization. On by default; disable with AITER_FUSED_QUANT=0.

    Falls back to JAX quant if CastMxfp4JA module is not built.
    """
    global _FUSED_QUANT_CACHE
    if _FUSED_QUANT_CACHE is None:
        import os
        if os.environ.get("AITER_FUSED_QUANT", "1") == "0":
            _FUSED_QUANT_CACHE = False
        else:
            try:
                register_ffi_target("CastMxfp4JA", "ROCM")
                register_ffi_target("CastMxfp4DualJA", "ROCM")
                _FUSED_QUANT_CACHE = True
            except Exception:
                import warnings
                warnings.warn(
                    "CastMxfp4JA module not found; falling back to JAX quant. "
                    "Build with 'make ja_mods' to enable fused quant.",
                    stacklevel=2,
                )
                _FUSED_QUANT_CACHE = False
    return _FUSED_QUANT_CACHE


_HADAMARD_CACHE = None


def _use_hadamard():
    """Use Hadamard transform in fused quant. Off by default; enable with AITER_FUSED_QUANT_HADAMARD=1."""
    global _HADAMARD_CACHE
    if _HADAMARD_CACHE is None:
        import os
        _HADAMARD_CACHE = os.environ.get("AITER_FUSED_QUANT_HADAMARD", "0") == "1"
    return _HADAMARD_CACHE


# ---------------------------------------------------------------------------
# FFI call wrappers — delegate to ops/ layer with env-var config
# ---------------------------------------------------------------------------

def _cast_mxfp4_fused_impl(x, shuffle_fp4):
    """Fused BF16 -> MXFP4 quantization + shuffle via HIP kernel (single FFI call)."""
    return _cast_mxfp4_op(x, shuffle_fp4=shuffle_fp4, use_hadamard=_use_hadamard())


def _cast_mxfp4_dual_impl(x, shuffle_fp4, shuffle_colwise_fp4=True):
    """Fused BF16 -> MXFP4 with BOTH rowwise and columnwise output in one kernel launch."""
    return _cast_mxfp4_dual_op(x, shuffle_fp4=shuffle_fp4,
                               shuffle_colwise_fp4=shuffle_colwise_fp4,
                               use_hadamard=_use_hadamard())


def _cast_mxfp4_raw_act(x):
    """Raw fused quant for activations (no FP4 data shuffle)."""
    return _cast_mxfp4_fused_impl(x, shuffle_fp4=False)


def _cast_mxfp4_raw_wt(x):
    """Raw fused quant for weights (with FP4 B-preshuffle for AITER GEMM)."""
    return _cast_mxfp4_fused_impl(x, shuffle_fp4=True)


def _cast_mxfp4_raw_dual(x):
    """Raw dual-mode quant for weights: rowwise (shuffled) + columnwise (shuffled)."""
    return _cast_mxfp4_dual_impl(x, shuffle_fp4=True, shuffle_colwise_fp4=True)


def _cast_mxfp4_raw_act_dual(x):
    """Raw dual-mode quant for activations: rowwise (NOT shuffled) + columnwise (shuffled).

    rowwise output has shuffle_fp4=False (suitable as GEMM A operand).
    columnwise output is B-preshuffle shuffled (suitable as GEMM B operand for dB).
    """
    return _cast_mxfp4_dual_impl(x, shuffle_fp4=False, shuffle_colwise_fp4=True)


def _cast_mxfp4_raw_grad_dual(x):
    """Raw dual-mode quant for grad_out in backward: rowwise + columnwise both unshuffled.

    rowwise output has shuffle_fp4=False (suitable as GEMM A operand for dA).
    columnwise output has shuffle_colwise_fp4=False: linear layout equivalent to
    rowwise FP4 of x^T (suitable as GEMM A operand for dB without JAX quant).
    """
    return _cast_mxfp4_dual_impl(x, shuffle_fp4=False, shuffle_colwise_fp4=False)


# ---------------------------------------------------------------------------
# custom_partitioning for fused quant FFI calls.
# Rowwise: Input x [M, K] bf16 -> fp4 [M, K/2], scales [M_pad, Sp].
# M is the shardable dimension; K and scale dims are replicated.
# ---------------------------------------------------------------------------
@custom_partitioning
def _cast_mxfp4_fused_act(x):
    return _cast_mxfp4_raw_act(x)


@custom_partitioning
def _cast_mxfp4_fused_wt(x):
    return _cast_mxfp4_raw_wt(x)


def _cast_infer_sharding(mesh, arg_shapes, result_shape):
    x_spec = _get_spec(arg_shapes[0])
    m_axis = x_spec[0]
    return (NamedSharding(mesh, P(m_axis, None)),
            NamedSharding(mesh, P(m_axis, None)))


def _cast_partition_act(mesh, arg_shapes, result_shape):
    x_spec = _get_spec(arg_shapes[0])
    m_axis = x_spec[0]
    in_spec = P(m_axis, None)
    out_spec = P(m_axis, None)

    def _lowered(x):
        return _cast_mxfp4_raw_act(x)

    return (mesh, _lowered,
            (NamedSharding(mesh, out_spec), NamedSharding(mesh, out_spec)),
            (NamedSharding(mesh, in_spec),))


def _cast_partition_wt(mesh, arg_shapes, result_shape):
    x_spec = _get_spec(arg_shapes[0])
    m_axis = x_spec[0]
    in_spec = P(m_axis, None)
    out_spec = P(m_axis, None)

    def _lowered(x):
        return _cast_mxfp4_raw_wt(x)

    return (mesh, _lowered,
            (NamedSharding(mesh, out_spec), NamedSharding(mesh, out_spec)),
            (NamedSharding(mesh, in_spec),))


_cast_mxfp4_fused_act.def_partition(
    _cast_partition_act,
    infer_sharding_from_operands=_cast_infer_sharding,
    sharding_rule="m k -> m kp, m sp",
    need_replication_factors=("k", "kp", "sp"),
)

_cast_mxfp4_fused_wt.def_partition(
    _cast_partition_wt,
    infer_sharding_from_operands=_cast_infer_sharding,
    sharding_rule="m k -> m kp, m sp",
    need_replication_factors=("k", "kp", "sp"),
)


# ---------------------------------------------------------------------------
# custom_partitioning for dual-mode fused quant (rowwise + columnwise).
# Input: x [M, K] bf16 (weight matrix, M=N_out, K=hidden).
# Outputs: rowwise_fp4 [M, K/2], rowwise_scale [M_pad, rSp],
#          colwise_fp4 [K, M/2], colwise_scale [K_pad, cSp].
# Rowwise outputs shard on M (first dim); colwise outputs shard M on second dim.
# ---------------------------------------------------------------------------
@custom_partitioning
def _cast_mxfp4_fused_dual(x):
    return _cast_mxfp4_raw_dual(x)


def _cast_dual_infer_sharding(mesh, arg_shapes, result_shape):
    x_spec = _get_spec(arg_shapes[0])
    m_axis = x_spec[0]
    return (NamedSharding(mesh, P(m_axis, None)),
            NamedSharding(mesh, P(m_axis, None)),
            NamedSharding(mesh, P(None, m_axis)),
            NamedSharding(mesh, P(None, m_axis)))


def _cast_dual_partition(mesh, arg_shapes, result_shape):
    x_spec = _get_spec(arg_shapes[0])
    m_axis = x_spec[0]
    in_spec = P(m_axis, None)
    row_spec = P(m_axis, None)
    col_spec = P(None, m_axis)

    def _lowered(x):
        return _cast_mxfp4_raw_dual(x)

    return (mesh, _lowered,
            (NamedSharding(mesh, row_spec), NamedSharding(mesh, row_spec),
             NamedSharding(mesh, col_spec), NamedSharding(mesh, col_spec)),
            (NamedSharding(mesh, in_spec),))


_cast_mxfp4_fused_dual.def_partition(
    _cast_dual_partition,
    infer_sharding_from_operands=_cast_dual_infer_sharding,
    sharding_rule="m k -> m rkp, m rsp, k cmp, k csp",
    need_replication_factors=("k", "rkp", "rsp", "cmp", "csp"),
)


@custom_partitioning
def _cast_mxfp4_fused_act_dual(x):
    """Dual-mode quant for activations: rowwise (NOT shuffled) + columnwise (shuffled)."""
    return _cast_mxfp4_raw_act_dual(x)


def _cast_act_dual_partition(mesh, arg_shapes, result_shape):
    x_spec = _get_spec(arg_shapes[0])
    m_axis = x_spec[0]
    in_spec = P(m_axis, None)
    row_spec = P(m_axis, None)
    col_spec = P(None, m_axis)

    def _lowered(x):
        return _cast_mxfp4_raw_act_dual(x)

    return (mesh, _lowered,
            (NamedSharding(mesh, row_spec), NamedSharding(mesh, row_spec),
             NamedSharding(mesh, col_spec), NamedSharding(mesh, col_spec)),
            (NamedSharding(mesh, in_spec),))


_cast_mxfp4_fused_act_dual.def_partition(
    _cast_act_dual_partition,
    infer_sharding_from_operands=_cast_dual_infer_sharding,
    sharding_rule="m k -> m rkp, m rsp, k cmp, k csp",
    need_replication_factors=("k", "rkp", "rsp", "cmp", "csp"),
)


@custom_partitioning
def _cast_mxfp4_fused_grad_dual(x):
    """Dual-mode quant for grad_out in backward: both rowwise and colwise unshuffled.

    Rowwise for dA (A operand), unshuffled colwise = rowwise of x^T for dB (A operand).
    """
    return _cast_mxfp4_raw_grad_dual(x)


def _cast_grad_dual_partition(mesh, arg_shapes, result_shape):
    x_spec = _get_spec(arg_shapes[0])
    m_axis = x_spec[0]
    in_spec = P(m_axis, None)
    row_spec = P(m_axis, None)
    col_spec = P(None, m_axis)

    def _lowered(x):
        return _cast_mxfp4_raw_grad_dual(x)

    return (mesh, _lowered,
            (NamedSharding(mesh, row_spec), NamedSharding(mesh, row_spec),
             NamedSharding(mesh, col_spec), NamedSharding(mesh, col_spec)),
            (NamedSharding(mesh, in_spec),))


_cast_mxfp4_fused_grad_dual.def_partition(
    _cast_grad_dual_partition,
    infer_sharding_from_operands=_cast_dual_infer_sharding,
    sharding_rule="m k -> m rkp, m rsp, k cmp, k csp",
    need_replication_factors=("k", "rkp", "rsp", "cmp", "csp"),
)


# ---------------------------------------------------------------------------
# FP4 GEMM FFI call + custom_partitioning
# ---------------------------------------------------------------------------

_KERNEL_SEL_CACHE = None


def _use_kernel_selection():
    """Use per-shape kernel selection via AITER's tuned CSV. Enable with AITER_KERNEL_SEL=1."""
    global _KERNEL_SEL_CACHE
    if _KERNEL_SEL_CACHE is None:
        import os
        _KERNEL_SEL_CACHE = os.environ.get("AITER_KERNEL_SEL", "0") == "1"
    return _KERNEL_SEL_CACHE


def _get_kernel_name(M, N, K):
    """Look up the best kernel for a given shape from AITER's tuned CSV."""
    if not _use_kernel_selection():
        return ""
    try:
        from aiter.ops.gemm_op_a4w4 import get_GEMM_config
        cfg = get_GEMM_config(M, N, K)
        if cfg is not None:
            return cfg["kernelName"]
    except Exception:
        pass
    return ""


def gemm_fp4(a, b, a_scale, b_scale):
    """Low-level FP4 GEMM with pre-quantized + pre-shuffled inputs."""
    return _gemm_fp4_ffi(a, b, a_scale, b_scale)


@custom_partitioning
def _fp4_ffi_partitioned(a_packed, b_packed, a_scale, b_scale):
    return _gemm_fp4_ffi(a_packed, b_packed, a_scale, b_scale)


def _get_spec(info):
    """Safely extract PartitionSpec from an arg_shape, handling None sharding."""
    if info.sharding is None:
        ndim = len(info.shape)
        return P(*([None] * ndim))
    return info.sharding.spec


def _resolve_fp4_specs(arg_shapes):
    a_spec = _get_spec(arg_shapes[0])
    b_spec = _get_spec(arg_shapes[1])
    m_axis = a_spec[0]
    n_axis = b_spec[0]

    m_set = (set(m_axis) if isinstance(m_axis, tuple)
             else {m_axis} if m_axis is not None else set())
    n_set = (set(n_axis) if isinstance(n_axis, tuple)
             else {n_axis} if n_axis is not None else set())
    if m_set & n_set:
        remaining = n_set - m_set
        n_axis = (next(iter(remaining)) if len(remaining) == 1
                  else tuple(sorted(remaining)) if remaining else None)

    return m_axis, n_axis


def _fp4_infer_sharding(mesh, arg_shapes, result_shape):
    m_axis, n_axis = _resolve_fp4_specs(arg_shapes)
    return NamedSharding(mesh, P(m_axis, n_axis))


def _fp4_partition(mesh, arg_shapes, result_shape):
    m_axis, n_axis = _resolve_fp4_specs(arg_shapes)

    a_pspec = P(m_axis, None)
    b_pspec = P(n_axis, None)
    out_pspec = P(m_axis, n_axis)

    def _lowered(a_packed, b_packed, a_scale, b_scale):
        return _gemm_fp4_ffi(a_packed, b_packed, a_scale, b_scale)

    return (mesh, _lowered,
            NamedSharding(mesh, out_pspec),
            (NamedSharding(mesh, a_pspec), NamedSharding(mesh, b_pspec),
             NamedSharding(mesh, a_pspec), NamedSharding(mesh, b_pspec)))


_fp4_ffi_partitioned.def_partition(
    _fp4_partition,
    infer_sharding_from_operands=_fp4_infer_sharding,
    sharding_rule="m kp, n kp, m ks, n ks -> m n",
    need_replication_factors=("kp", "ks"),
)


# ---------------------------------------------------------------------------
# Wgrad-layout FP4 GEMM with correct NT-layout sharding.
#
# The FP4 FFI computes OUT[M_k, N_k] = A[M_k, K_k/2] @ B[N_k, K_k/2]^T where
# the K_k axis is the contraction. For **fprop/dgrad** K_k is a hidden
# dimension (typically replicated across FSDP), so the existing
# `_fp4_ffi_partitioned` declares K as replication-required.
#
# For **wgrad** the contraction axis is M_batch (batch * seq), which IS
# FSDP-sharded. We therefore need a separate partition callback that:
#   - Accepts sharding on the SECOND (packed-K) axis of both operands.
#   - Computes the local partial GEMM on each shard.
#   - Emits ``jax.lax.psum`` across the FSDP mesh axis to sum partials.
# This mirrors how XLA lowers ``lax.dot_general`` over a sharded contraction:
# local matmul + reduce-scatter/all-reduce.
#
# Operand layout for wgrad (after dual-cast of grad_out and activation):
#   A = grad_col_fp4   [N, M/2]   (grad_out's columnwise, unshuffled)
#   B = input_col_fp4  [K, M/2]   (activation's columnwise, shuffled for B)
#   OUT = dB           [N, K]     (weight-gradient)
# ---------------------------------------------------------------------------
@custom_partitioning
def _fp4_ffi_partitioned_wgrad(a_packed, b_packed, a_scale, b_scale):
    """FP4 GEMM for wgrad: contraction axis is allowed to be sharded."""
    return _gemm_fp4_ffi(a_packed, b_packed, a_scale, b_scale)


def _extract_fsdp_axis(spec, dim_index):
    """Return the mesh-axis name(s) sharded on ``spec[dim_index]`` or ``None``."""
    if spec is None:
        return None
    if dim_index >= len(spec):
        return None
    return spec[dim_index]


def _fp4_wgrad_infer_sharding(mesh, arg_shapes, result_shape):
    """Output sharding for wgrad.

    Output ``dB[N, K]``: the first dim (N) inherits from A's row sharding and
    the second dim (K) inherits from B's row sharding. Both are usually
    replicated after the reduction, matching how XLA outputs a reduce-scatter
    result.
    """
    a_spec = _get_spec(arg_shapes[0])
    b_spec = _get_spec(arg_shapes[1])
    n_axis = a_spec[0] if a_spec is not None and len(a_spec) >= 1 else None
    k_axis = b_spec[0] if b_spec is not None and len(b_spec) >= 1 else None
    return NamedSharding(mesh, P(n_axis, k_axis))


def _fp4_wgrad_partition(mesh, arg_shapes, result_shape):
    """Partition callback that locally computes dB then ``psum``s over M.

    Expects inputs with sharding:
      a_packed: P(N_axis_or_None, M_axis)
      b_packed: P(K_axis_or_None, M_axis)
    The M_axis shared between the two second dims is the FSDP contraction
    axis. After the local FP4 GEMM we ``psum`` across that mesh axis.
    """
    a_spec = _get_spec(arg_shapes[0])
    b_spec = _get_spec(arg_shapes[1])

    n_axis = _extract_fsdp_axis(a_spec, 0)
    k_axis = _extract_fsdp_axis(b_spec, 0)
    m_axis_a = _extract_fsdp_axis(a_spec, 1)
    m_axis_b = _extract_fsdp_axis(b_spec, 1)

    m_axis = m_axis_a if m_axis_a is not None else m_axis_b
    if m_axis_a is not None and m_axis_b is not None and m_axis_a != m_axis_b:
        m_axis = m_axis_a

    n_set = (set(n_axis) if isinstance(n_axis, tuple)
             else {n_axis} if n_axis is not None else set())
    k_set = (set(k_axis) if isinstance(k_axis, tuple)
             else {k_axis} if k_axis is not None else set())
    if n_set & k_set:
        remaining = k_set - n_set
        k_axis = (next(iter(remaining)) if len(remaining) == 1
                  else tuple(sorted(remaining)) if remaining else None)

    a_pspec = P(n_axis, m_axis)
    b_pspec = P(k_axis, m_axis)
    s_a_pspec = P(n_axis, m_axis)
    s_b_pspec = P(k_axis, m_axis)
    out_pspec = P(n_axis, k_axis)

    # Normalize the reduction axis to an iterable for ``jax.lax.psum``.
    if m_axis is None:
        psum_axes = ()
    elif isinstance(m_axis, tuple):
        psum_axes = m_axis
    else:
        psum_axes = (m_axis,)

    def _lowered(a_packed, b_packed, a_scale, b_scale):
        partial = _gemm_fp4_ffi(a_packed, b_packed, a_scale, b_scale)
        if psum_axes:
            partial = jax.lax.psum(partial, axis_name=psum_axes)
        return partial

    return (mesh, _lowered,
            NamedSharding(mesh, out_pspec),
            (NamedSharding(mesh, a_pspec), NamedSharding(mesh, b_pspec),
             NamedSharding(mesh, s_a_pspec), NamedSharding(mesh, s_b_pspec)))


# Shardy rule: the reduction axis is labelled ``mp`` (packed M) and it appears
# on both inputs' second dims but **not** on the output. In Shardy, an axis
# that is present on inputs but absent from the output is a reduction axis;
# XLA will automatically insert the reduce-scatter / all-reduce around the
# local call. This matches how ``lax.dot_general`` lowers a sharded
# contraction.
_fp4_ffi_partitioned_wgrad.def_partition(
    _fp4_wgrad_partition,
    infer_sharding_from_operands=_fp4_wgrad_infer_sharding,
    sharding_rule="n mp, k mp, n ms, k ms -> n k",
    need_replication_factors=(),
)


# ---------------------------------------------------------------------------
# All-FP4 training recipe (TE-parity): FP4 fwd + FP4 dA (NN) + FP4 dB (NT).
# ---------------------------------------------------------------------------
_ALL_FP4_CACHE = None


def _use_all_fp4():
    """Use the full all-FP4 training recipe (FP4 fwd + FP4 dA + FP4 dB).

    When enabled, backward uses :func:`_fp4_ffi_partitioned_wgrad` for dB with
    correct NT-layout sharding (M_batch as reduction axis), removing the
    dependency on FP8 ``dot_general`` for weight gradients.

    Requires fused HIP quant (``AITER_FUSED_QUANT=1``). Off by default while
    this path is being perf-validated; enable with ``AITER_ALL_FP4=1``.
    """
    global _ALL_FP4_CACHE
    if _ALL_FP4_CACHE is None:
        import os
        _ALL_FP4_CACHE = os.environ.get("AITER_ALL_FP4", "0") == "1"
    return _ALL_FP4_CACHE


# ---------------------------------------------------------------------------
# Weight pre-packing utility
# ---------------------------------------------------------------------------
def prepack_fp4_weight(b_bf16):
    """Pre-quantize and shuffle a weight matrix for weight-only FP4.

    Call once at init; reuse the packed tensors every step.

    Args:
        b_bf16: [N, K] bfloat16 weight matrix.

    Returns:
        b_packed: [N, K//2] uint8 — packed + shuffled FP4 weights.
        b_scales: [N_pad, K_pad//32] uint8 — shuffled E8M0 block scales.
    """
    if _use_fused_quant():
        return _cast_mxfp4_fused_wt(b_bf16)
    b_packed, b_scales = bf16_to_mxfp4(b_bf16)
    return shuffle_weight(b_packed), e8m0_shuffle(b_scales)


# ---------------------------------------------------------------------------
# Composite FP4 forward: Cast(A) + Cast(B) + GEMM in one custom_partitioning
# boundary, reducing 3 opaque XLA barriers to 1.
# ---------------------------------------------------------------------------
_COMPOSITE_FP4_CACHE = None


def _use_composite_fp4():
    """Use composite Cast+GEMM FP4 forward. OFF by default; enable with AITER_COMPOSITE_FP4=1.

    WARNING: E2E testing showed this causes ~2.4x regression at 8B due to XLA
    scan serialization. The single larger opaque custom_call blocks pipelining
    more than three smaller ones. Keep disabled until XLA scheduling improves.
    """
    global _COMPOSITE_FP4_CACHE
    if _COMPOSITE_FP4_CACHE is None:
        import os
        _COMPOSITE_FP4_CACHE = os.environ.get("AITER_COMPOSITE_FP4", "0") == "1"
    return _COMPOSITE_FP4_CACHE


def _composite_fp4_fwd_raw(a, b):
    """Raw composite: BF16 A,B -> Cast both -> FP4 GEMM -> BF16 out.

    All three FFI calls happen inside one function body so that
    custom_partitioning wraps them as a single opaque boundary.
    """
    a_packed, a_scales = _cast_mxfp4_fused_impl(a, shuffle_fp4=False)
    b_packed, b_scales = _cast_mxfp4_fused_impl(b, shuffle_fp4=True)
    return _gemm_fp4_ffi(a_packed, b_packed, a_scales, b_scales)


def _composite_fp4_fwd_dual_raw(a, b):
    """Raw composite forward with dual-mode weight cast for dA backward.

    Returns (out, col_b_fp4, col_b_scale) so backward can reuse columnwise
    weight data without re-quantizing.
    """
    a_packed, a_scales = _cast_mxfp4_fused_impl(a, shuffle_fp4=False)
    (b_packed, b_scales,
     col_b_fp4, col_b_scale) = _cast_mxfp4_dual_impl(b, shuffle_fp4=True)
    out = _gemm_fp4_ffi(a_packed, b_packed, a_scales, b_scales)
    return out, col_b_fp4, col_b_scale


def _composite_fp4_da_raw(grad_out, col_b_fp4, col_b_scale):
    """Raw composite dA: Cast(grad_out) + FP4 GEMM with pre-computed colwise weight."""
    go_packed, go_scales = _cast_mxfp4_fused_impl(grad_out, shuffle_fp4=False)
    return _gemm_fp4_ffi(go_packed, col_b_fp4, go_scales, col_b_scale)


@custom_partitioning
def _composite_fp4_fwd_partitioned(a, b):
    return _composite_fp4_fwd_raw(a, b)


def _composite_fwd_infer_sharding(mesh, arg_shapes, result_shape):
    a_spec = _get_spec(arg_shapes[0])
    b_spec = _get_spec(arg_shapes[1])
    m_axis = a_spec[0]
    n_axis = b_spec[0]
    m_set = (set(m_axis) if isinstance(m_axis, tuple)
             else {m_axis} if m_axis is not None else set())
    n_set = (set(n_axis) if isinstance(n_axis, tuple)
             else {n_axis} if n_axis is not None else set())
    if m_set & n_set:
        remaining = n_set - m_set
        n_axis = (next(iter(remaining)) if len(remaining) == 1
                  else tuple(sorted(remaining)) if remaining else None)
    return NamedSharding(mesh, P(m_axis, n_axis))


def _composite_fwd_partition(mesh, arg_shapes, result_shape):
    a_spec = _get_spec(arg_shapes[0])
    b_spec = _get_spec(arg_shapes[1])
    m_axis = a_spec[0]
    n_axis = b_spec[0]
    m_set = (set(m_axis) if isinstance(m_axis, tuple)
             else {m_axis} if m_axis is not None else set())
    n_set = (set(n_axis) if isinstance(n_axis, tuple)
             else {n_axis} if n_axis is not None else set())
    if m_set & n_set:
        remaining = n_set - m_set
        n_axis = (next(iter(remaining)) if len(remaining) == 1
                  else tuple(sorted(remaining)) if remaining else None)

    a_pspec = P(m_axis, None)
    b_pspec = P(n_axis, None)
    out_pspec = P(m_axis, n_axis)

    def _lowered(a, b):
        return _composite_fp4_fwd_raw(a, b)

    return (mesh, _lowered,
            NamedSharding(mesh, out_pspec),
            (NamedSharding(mesh, a_pspec), NamedSharding(mesh, b_pspec)))


_composite_fp4_fwd_partitioned.def_partition(
    _composite_fwd_partition,
    infer_sharding_from_operands=_composite_fwd_infer_sharding,
    sharding_rule="m k, n k -> m n",
    need_replication_factors=("k",),
)


@custom_partitioning
def _composite_fp4_da_partitioned(grad_out, col_b_fp4, col_b_scale):
    return _composite_fp4_da_raw(grad_out, col_b_fp4, col_b_scale)


def _composite_da_infer_sharding(mesh, arg_shapes, result_shape):
    g_spec = _get_spec(arg_shapes[0])
    m_axis = g_spec[0]
    col_spec = _get_spec(arg_shapes[1])
    k_axis = col_spec[0]
    m_set = (set(m_axis) if isinstance(m_axis, tuple)
             else {m_axis} if m_axis is not None else set())
    k_set = (set(k_axis) if isinstance(k_axis, tuple)
             else {k_axis} if k_axis is not None else set())
    if m_set & k_set:
        remaining = k_set - m_set
        k_axis = (next(iter(remaining)) if len(remaining) == 1
                  else tuple(sorted(remaining)) if remaining else None)
    return NamedSharding(mesh, P(m_axis, k_axis))


def _composite_da_partition(mesh, arg_shapes, result_shape):
    g_spec = _get_spec(arg_shapes[0])
    m_axis = g_spec[0]
    col_spec = _get_spec(arg_shapes[1])
    k_axis = col_spec[0]
    m_set = (set(m_axis) if isinstance(m_axis, tuple)
             else {m_axis} if m_axis is not None else set())
    k_set = (set(k_axis) if isinstance(k_axis, tuple)
             else {k_axis} if k_axis is not None else set())
    if m_set & k_set:
        remaining = k_set - m_set
        k_axis = (next(iter(remaining)) if len(remaining) == 1
                  else tuple(sorted(remaining)) if remaining else None)

    g_pspec = P(m_axis, None)
    col_fp4_pspec = P(None, k_axis)
    col_scale_pspec = P(None, k_axis)
    out_pspec = P(m_axis, k_axis)

    def _lowered(grad_out, col_b_fp4, col_b_scale):
        return _composite_fp4_da_raw(grad_out, col_b_fp4, col_b_scale)

    return (mesh, _lowered,
            NamedSharding(mesh, out_pspec),
            (NamedSharding(mesh, g_pspec),
             NamedSharding(mesh, col_fp4_pspec),
             NamedSharding(mesh, col_scale_pspec)))


_composite_fp4_da_partitioned.def_partition(
    _composite_da_partition,
    infer_sharding_from_operands=_composite_da_infer_sharding,
    sharding_rule="m n, k mp, k sp -> m k",
    need_replication_factors=("n", "mp", "sp"),
)


# ---------------------------------------------------------------------------
# High-level API: BF16 in, FP4 forward, BF16 backward
# ---------------------------------------------------------------------------
@partial(jax.custom_vjp, nondiff_argnums=())
def gemm_fp4_bf16(a, b):
    """Compute A @ B^T with FP4 forward, BF16 backward.

    Forward: quantize A,B to MXFP4 -> AITER FP4 ASM kernel (partitioned).
    Backward: dA via FP4 ASM using pre-computed columnwise weight (no re-quant).
              dB via hipBLASLt FP8 or BF16.

    When AITER_COMPOSITE_FP4=1 (off by default), the forward uses a single
    custom_partitioning boundary that internally chains Cast+Cast+GEMM,
    reducing opaque XLA barriers from 3 to 1 per projection.

    Args:
        a: [M, K] bfloat16
        b: [N, K] bfloat16

    Returns:
        out: [M, N] bfloat16
    """
    if _use_fused_quant() and _use_composite_fp4():
        return _composite_fp4_fwd_partitioned(a, b)

    if _use_fused_quant():
        a_packed, a_scales_sh = _cast_mxfp4_fused_act(a)
        b_packed, b_scales_sh = _cast_mxfp4_fused_wt(b)
    else:
        a_packed, a_scales = bf16_to_mxfp4(a)
        b_packed, b_scales = bf16_to_mxfp4(b)
        b_packed = shuffle_weight(b_packed)
        a_scales_sh = e8m0_shuffle(a_scales)
        b_scales_sh = e8m0_shuffle(b_scales)

    return _fp4_ffi_partitioned(a_packed, b_packed, a_scales_sh, b_scales_sh)


_LAZY_WEIGHT_COL_CACHE = None


def _use_lazy_weight_col():
    """Defer weight columnwise quantization to backward (saves ~37 GB persistent).

    Forward produces weight rowwise only via CastMxfp4JA.  Backward creates
    columnwise on demand via CastMxfp4DualJA for dgrad, then discards it.
    Saves memory because the columnwise data doesn't persist across scan
    iterations.

    Enable with AITER_LAZY_WEIGHT_COL=1.  Requires fused quant + FP4 dA.
    """
    global _LAZY_WEIGHT_COL_CACHE
    if _LAZY_WEIGHT_COL_CACHE is None:
        import os
        _LAZY_WEIGHT_COL_CACHE = os.environ.get("AITER_LAZY_WEIGHT_COL", "0") == "1"
    return _LAZY_WEIGHT_COL_CACHE


_FP4_RESIDUALS_CACHE = None


def _use_fp4_residuals():
    """Save activation residuals as FP4 instead of BF16 (4x smaller scan carry).

    The forward saves (a_fp4_packed, a_fp4_scales, col_b_fp4, col_b_scale)
    instead of (a_bf16, col_b_fp4, col_b_scale).  The backward dequantizes
    a_fp4 back to BF16 for the native FP8 dB path, keeping FSDP
    reduce-scatter fusion intact while cutting residual memory ~4x.

    Enable with AITER_FP4_RESIDUALS=1.  Requires fused quant + FP4 dA.
    """
    global _FP4_RESIDUALS_CACHE
    if _FP4_RESIDUALS_CACHE is None:
        import os
        _FP4_RESIDUALS_CACHE = os.environ.get("AITER_FP4_RESIDUALS", "0") == "1"
    return _FP4_RESIDUALS_CACHE


def _dequant_fp4_for_db(a_packed, a_scales_sh):
    """Dequantize saved FP4 activation back to BF16 for native FP8 dB.

    a_packed is linear (shuffle_fp4=False) FP4 data from CastMxfp4JA.
    a_scales_sh is SHUFFLED E8M0 scales.  We unshuffle before decoding.
    """
    from .fp4_utils import MXFP4_BLOCK_SIZE
    M = a_packed.shape[0]
    K_half = a_packed.shape[1]
    K = K_half * 2

    lut = jnp.array([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ], dtype=jnp.float32)

    low = (a_packed & jnp.uint8(0xF)).astype(jnp.int32)
    high = (a_packed >> jnp.uint8(4)).astype(jnp.int32)
    vals = jnp.empty((M, K), dtype=jnp.float32)
    vals = vals.at[:, 0::2].set(lut[low])
    vals = vals.at[:, 1::2].set(lut[high])

    scales_flat = _unshuffle_e8m0(a_scales_sh)

    num_blocks = K // MXFP4_BLOCK_SIZE
    scale_f32 = jnp.exp2(scales_flat[:M, :num_blocks].astype(jnp.float32) - 127.0)
    scale_expanded = jnp.repeat(scale_f32, MXFP4_BLOCK_SIZE, axis=-1)

    return (vals * scale_expanded).astype(jnp.bfloat16)


def _unshuffle_e8m0(scales_sh):
    """Inverse of e8m0_shuffle: recover linear [M, N] layout from ASM-shuffled scales."""
    sm, sn = scales_sh.shape
    reshaped = scales_sh.reshape(sm // 32, sn // 32, 4, 16, 2, 2)
    permuted = reshaped.transpose(0, 5, 3, 1, 4, 2)
    return permuted.reshape(sm, sn)


def _gemm_fp4_bf16_fwd(a, b):
    if _use_fused_quant() and _use_fp4_da():
        if _use_composite_fp4():
            _ensure_registered()
            register_ffi_target("CastMxfp4JA", "ROCM")
            register_ffi_target("CastMxfp4DualJA", "ROCM")
            out, col_b_fp4, col_b_scale = _composite_fp4_fwd_dual_raw(a, b)
            return out, (a, col_b_fp4, col_b_scale)

        # TE-parity all-FP4 path: dual-cast activations so col_a is available
        # for the FP4 wgrad in backward. Same FFI kernel, just both outputs.
        if _use_all_fp4() or _use_fp4_db():
            (a_packed, a_scales_sh,
             col_a_fp4, col_a_scale) = _cast_mxfp4_fused_act_dual(a)
            (b_packed, b_scales_sh,
             col_b_fp4, col_b_scale) = _cast_mxfp4_fused_dual(b)
            out = _fp4_ffi_partitioned(a_packed, b_packed, a_scales_sh, b_scales_sh)
            return out, (col_a_fp4, col_a_scale, col_b_fp4, col_b_scale)

        if _use_fp4_residuals():
            a_packed, a_scales_sh = _cast_mxfp4_fused_act(a)
            (b_packed, b_scales_sh,
             col_b_fp4, col_b_scale) = _cast_mxfp4_fused_dual(b)
            out = _fp4_ffi_partitioned(a_packed, b_packed, a_scales_sh, b_scales_sh)
            return out, (a_packed, a_scales_sh, col_b_fp4, col_b_scale)

        if _use_lazy_weight_col():
            a_packed, a_scales_sh = _cast_mxfp4_fused_act(a)
            b_packed, b_scales_sh = _cast_mxfp4_fused_wt(b)
            out = _fp4_ffi_partitioned(a_packed, b_packed, a_scales_sh, b_scales_sh)
            return out, (a, b)

        a_packed, a_scales_sh = _cast_mxfp4_fused_act(a)
        (b_packed, b_scales_sh,
         col_b_fp4, col_b_scale) = _cast_mxfp4_fused_dual(b)
        out = _fp4_ffi_partitioned(a_packed, b_packed, a_scales_sh, b_scales_sh)
        return out, (a, col_b_fp4, col_b_scale)
    out = gemm_fp4_bf16(a, b)
    return out, (a, b)


_FP4_DA_CACHE = None


def _use_fp4_da():
    """Use FP4 ASM for dA backward. On by default; disable with AITER_FP4_DA=0."""
    global _FP4_DA_CACHE
    if _FP4_DA_CACHE is None:
        import os
        _FP4_DA_CACHE = os.environ.get("AITER_FP4_DA", "1") != "0"
    return _FP4_DA_CACHE


_FP4_DB_CACHE = None


def _use_fp4_db():
    """Use FP4 ASM GEMM for dB backward (NT layout wgrad).

    On by default when fused quant is available. Disable with AITER_FP4_DB=0.
    When disabled, falls back to FP8 dB (AITER_FP8_DB) or BF16 dot_general.

    Benchmarked at 1.18-1.35x faster than hipBLASLt FP8 at 70B dB shapes,
    and 1.70-2.17x at 8B shapes (2026-04-10 kernel benchmark).
    """
    global _FP4_DB_CACHE
    if _FP4_DB_CACHE is None:
        import os
        explicit = os.environ.get("AITER_FP4_DB")
        if explicit is not None:
            _FP4_DB_CACHE = explicit != "0"
        else:
            _FP4_DB_CACHE = _use_fused_quant()
    return _FP4_DB_CACHE


_FP8_DB_CACHE = None

_FP8_E4M3_MAX = 448.0


def _use_fp8_db():
    """Use hipBLASLt FP8 for dB backward. Fallback when FP4 dB is disabled."""
    global _FP8_DB_CACHE
    if _FP8_DB_CACHE is None:
        import os
        _FP8_DB_CACHE = os.environ.get("AITER_FP8_DB", "1") != "0"
    return _FP8_DB_CACHE


def _fp8_dot_general_db(grad_out, a):
    """dB[N,K] = grad_out^T[N,M] @ A[M,K] via native hipBLASLt FP8.

    Per-tensor dynamic scaling avoids FP8 saturation clipping:
      scale = 448 / amax(|x|), then cast (x * scale) to e4m3fn.
    The GEMM result is descaled by 1/(scale_g * scale_a).

    XLA GemmRewriter lowers dot_general(f8e4m3fn) to __cublas$lt$matmul$f8,
    which is ~2.3x faster than hipBLASLt BF16 at production dB shapes.
    """
    eps = jnp.finfo(jnp.float32).tiny
    amax_g = jnp.max(jnp.abs(grad_out))
    amax_a = jnp.max(jnp.abs(a))
    scale_g = jnp.float32(_FP8_E4M3_MAX) / (amax_g + eps)
    scale_a = jnp.float32(_FP8_E4M3_MAX) / (amax_a + eps)
    g_fp8 = (grad_out * scale_g).astype(jnp.float8_e4m3fn)
    a_fp8 = (a * scale_a).astype(jnp.float8_e4m3fn)
    db = jax.lax.dot_general(
        g_fp8, a_fp8, (((0,), (0,)), ((), ())),
        preferred_element_type=jnp.bfloat16,
    )
    return (db * (jnp.float32(1.0) / (scale_g * scale_a))).astype(jnp.bfloat16)


def _fp4_jax_quant_db_two_transpose(grad_out, a):
    """dB via JAX quant with two BF16 transposes (fallback when col_a not saved).

    Transposes both grad_out and input, quantizes both with JAX ops.
    This path OOMs at production batch sizes; use _fp4_db_with_col_a instead.
    """
    go_t = jnp.transpose(grad_out, (1, 0))
    a_t = jnp.transpose(a, (1, 0))

    go_t_packed, go_t_scales = bf16_to_mxfp4(go_t)
    go_t_scales_sh = e8m0_shuffle(go_t_scales)

    a_t_packed, a_t_scales = bf16_to_mxfp4(a_t)
    a_t_packed_sh = shuffle_weight(a_t_packed)
    a_t_scales_sh = e8m0_shuffle(a_t_scales)

    return _fp4_ffi_partitioned(go_t_packed, a_t_packed_sh, go_t_scales_sh, a_t_scales_sh)


def _fp4_db_with_col_a(grad_out, col_a_fp4, col_a_scale):
    """dB[N,K] = grad_out[M,N]^T @ input[M,K] via FP4 ASM GEMM.

    Uses pre-saved columnwise FP4 of input (from forward CastMxfp4DualJA)
    as the B operand. Only grad_out needs transposing + quantization.

    col_a_fp4 [K, M/2] is B-preshuffle shuffled (from CastMxfp4DualJA).
    grad_out^T [N, M] is quantized to rowwise FP4 (NOT shuffled) via JAX ops
    to avoid the FFI gradient corruption in backward scan body.

    This path eliminates one of the two BF16 transposes that caused OOM,
    and residuals are 4x smaller (FP4 vs BF16).
    """
    go_t = jnp.transpose(grad_out, (1, 0))
    go_t_packed, go_t_scales = bf16_to_mxfp4(go_t)
    go_t_scales_sh = e8m0_shuffle(go_t_scales)
    return _fp4_ffi_partitioned(go_t_packed, col_a_fp4, go_t_scales_sh, col_a_scale)


def _fp4_da_with_columnwise(grad_out, col_b_fp4, col_b_scale):
    """dA via FP4 GEMM using pre-computed columnwise weight from forward.

    The forward's dual-mode CastMxfp4DualJA produces columnwise FP4 data
    (the quantized transpose of B) in the same kernel launch as rowwise.
    The columnwise B data is pre-computed and needs no FFI in the backward.

    When composite mode is active, uses a single custom_partitioning
    boundary (Cast+GEMM) instead of separate Cast + GEMM barriers.
    """
    if _use_composite_fp4():
        return _composite_fp4_da_partitioned(grad_out, col_b_fp4, col_b_scale)
    go_packed, go_scales = _cast_mxfp4_fused_act(grad_out)
    return _fp4_ffi_partitioned(go_packed, col_b_fp4, go_scales, col_b_scale)


def _gemm_fp4_bf16_bwd(residuals, grad_out):
    # TE-parity all-FP4 path: dual-cast grad_out once, compute FP4 dA (NN)
    # and FP4 dB (NT with correct wgrad sharding) in one backward rule.
    if _use_fused_quant() and _use_fp4_da() and (_use_all_fp4() or _use_fp4_db()):
        col_a_fp4, col_a_scale, col_b_fp4, col_b_scale = residuals
        (go_packed, go_scales,
         go_col_fp4, go_col_scale) = _cast_mxfp4_fused_grad_dual(grad_out)
        da = _fp4_ffi_partitioned(go_packed, col_b_fp4, go_scales, col_b_scale)
        if _use_all_fp4():
            # NT-layout wgrad with M (batch) as the reduction axis. The
            # partition callback emits psum across the FSDP mesh axis.
            db = _fp4_ffi_partitioned_wgrad(
                go_col_fp4, col_a_fp4, go_col_scale, col_a_scale)
        else:
            # Legacy FP4 dB (no wgrad-aware sharding); kept for regression
            # comparison. Deprecated once AITER_ALL_FP4=1 becomes default.
            db = _fp4_ffi_partitioned(
                go_col_fp4, col_a_fp4, go_col_scale, col_a_scale)
    elif _use_fused_quant() and _use_fp4_da() and _use_fp4_residuals():
        a_packed, a_scales_sh, col_b_fp4, col_b_scale = residuals
        da = _fp4_da_with_columnwise(grad_out, col_b_fp4, col_b_scale)
        a_bf16 = _dequant_fp4_for_db(a_packed, a_scales_sh)
        if _use_fp8_db():
            db = _fp8_dot_general_db(grad_out, a_bf16)
        else:
            db = jax.lax.dot_general(grad_out, a_bf16, (((0,), (0,)), ((), ())))
    elif _use_fused_quant() and _use_fp4_da() and _use_lazy_weight_col():
        a, b = residuals
        (_, _, col_b_fp4, col_b_scale) = _cast_mxfp4_fused_dual(b)
        da = _fp4_da_with_columnwise(grad_out, col_b_fp4, col_b_scale)
        if _use_fp8_db():
            db = _fp8_dot_general_db(grad_out, a)
        else:
            db = jax.lax.dot_general(grad_out, a, (((0,), (0,)), ((), ())))
    elif _use_fused_quant() and _use_fp4_da():
        a, col_b_fp4, col_b_scale = residuals
        da = _fp4_da_with_columnwise(grad_out, col_b_fp4, col_b_scale)
        if _use_fp8_db():
            db = _fp8_dot_general_db(grad_out, a)
        else:
            db = jax.lax.dot_general(grad_out, a, (((0,), (0,)), ((), ())))
    else:
        a, b = residuals
        if _use_fp4_da():
            b_t = jnp.transpose(b, (1, 0))
            da = gemm_fp4_bf16(grad_out, b_t)
        else:
            da = jax.lax.dot_general(grad_out, b, (((1,), (0,)), ((), ())))
        if _use_fp4_db():
            db = _fp4_jax_quant_db_two_transpose(grad_out, a)
        elif _use_fp8_db():
            db = _fp8_dot_general_db(grad_out, a)
        else:
            db = jax.lax.dot_general(grad_out, a, (((0,), (0,)), ((), ())))
    return da, db


gemm_fp4_bf16.defvjp(_gemm_fp4_bf16_fwd, _gemm_fp4_bf16_bwd)


# ---------------------------------------------------------------------------
# Weight-only FP4: pre-packed weights, only activations quantized per-step.
# ---------------------------------------------------------------------------
@partial(jax.custom_vjp, nondiff_argnums=())
def gemm_fp4_weight_only(a, b_bf16, b_packed, b_scales):
    """Compute A @ B^T with pre-packed FP4 weights.

    Only activations are quantized per-step. Weights must be pre-packed via
    prepack_fp4_weight(). b_bf16 is carried for the BF16 backward pass.

    Args:
        a: [M, K] bfloat16 activations.
        b_bf16: [N, K] bfloat16 original weights (for backward).
        b_packed: [N, K//2] uint8 pre-packed FP4 weights.
        b_scales: [N_pad, K_pad//32] uint8 pre-shuffled E8M0 scales.

    Returns:
        out: [M, N] bfloat16
    """
    if _use_fused_quant():
        a_packed, a_scales_sh = _cast_mxfp4_fused_act(a)
    else:
        a_packed, a_scales = bf16_to_mxfp4(a)
        a_scales_sh = e8m0_shuffle(a_scales)
    return _fp4_ffi_partitioned(a_packed, b_packed, a_scales_sh, b_scales)


def _gemm_fp4_wo_fwd(a, b_bf16, b_packed, b_scales):
    out = gemm_fp4_weight_only(a, b_bf16, b_packed, b_scales)
    return out, (a, b_bf16, b_packed, b_scales)


def _gemm_fp4_wo_bwd(residuals, grad_out):
    a, b_bf16, b_packed, b_scales = residuals
    if _use_fp4_da():
        b_t = jnp.transpose(b_bf16, (1, 0))
        da = gemm_fp4_bf16(grad_out, b_t)
    else:
        da = jax.lax.dot_general(grad_out, b_bf16, (((1,), (0,)), ((), ())))
    if _use_fp4_db():
        db = _fp4_jax_quant_db_two_transpose(grad_out, a)
    elif _use_fp8_db():
        db = _fp8_dot_general_db(grad_out, a)
    else:
        db = jax.lax.dot_general(grad_out, a, (((0,), (0,)), ((), ())))
    return da, db, jnp.zeros_like(b_packed), jnp.zeros_like(b_scales)


gemm_fp4_weight_only.defvjp(_gemm_fp4_wo_fwd, _gemm_fp4_wo_bwd)
