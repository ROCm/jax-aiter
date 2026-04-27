# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""FP4 (MXFP4) GEMM via AITER ASM kernels with custom_vjp + custom_partitioning.

Single-recipe FP4 training. No env-flag-gated alternative paths.

Forward
-------
``Out[M, N] = A_bf16[M, K] @ B_bf16[N, K]^T``

  - Activation ``A`` is dual-cast (rowwise unshuffled + columnwise shuffled).
  - Weight ``B`` is dual-cast (rowwise shuffled + columnwise shuffled).
  - Forward GEMM uses rowwise FP4 of both via ``GemmFp4FwdJA``.
  - Columnwise FP4 outputs are saved as residuals for the backward pass --
    avoiding any re-quantization in backward.

Backward
--------
  - ``dA = grad_out @ B``     -- FP4 ASM with the saved columnwise weight.
  - ``dB = grad_out^T @ A``  -- FP4 wgrad GEMM via
    ``_fp4_ffi_partitioned_wgrad`` with FSDP-aware ``jax.lax.psum`` sharding
    (NT-layout: contraction over the FSDP-sharded ``M_batch`` axis).

``grad_out`` is dual-cast WITH Hadamard transform applied inside the fused
HIP kernel (TE parity). Hadamard decorrelates outliers in the gradient
distribution, tightening 70B loss convergence. Activations and weights are
quantized WITHOUT Hadamard (their distributions are already well-behaved
after RMSNorm + standard init).

Env vars (advanced)
-------------------
``AITER_FUSED_QUANT_HADAMARD=1``
    Apply Hadamard to ALL casts (act / weight / grad). Debug / ablation
    knob. Default applies Hadamard ONLY to the grad cast.
``AITER_KERNEL_SEL=1``
    Use AITER's per-shape tuned-kernel CSV instead of the default static
    heuristic. Opt-in.
"""

from __future__ import annotations

import os
from functools import partial

import jax
import jax.numpy as jnp  # noqa: F401  # imported for downstream consumers
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import NamedSharding, PartitionSpec as P

from ..ops.gemm_fp4 import (
    gemm_fp4 as _gemm_fp4_ffi,
    cast_mxfp4 as _cast_mxfp4_op,
    cast_mxfp4_dual as _cast_mxfp4_dual_op,
)
from ..ffi.registry import register_ffi_target
from .fp4_utils import bf16_to_mxfp4, e8m0_shuffle, shuffle_weight  # noqa: F401  # re-exported


# ---------------------------------------------------------------------------
# Module-load FFI registration. The FP4 recipe REQUIRES the fused HIP cast
# kernel and the FP4 ASM GEMM. If the build artifacts are missing we raise a
# clear ImportError instead of silently falling back to a slow path.
# ---------------------------------------------------------------------------
try:
    register_ffi_target("CastMxfp4JA", "ROCM")
    register_ffi_target("CastMxfp4DualJA", "ROCM")
    register_ffi_target("GemmFp4FwdJA", "ROCM")
except Exception as exc:  # pragma: no cover -- build-time error path
    raise ImportError(
        "FP4 FFI targets failed to register: {}\n"
        "Build the AITER FFI modules with 'make ja_mods' before importing "
        "jax_aiter.gemm_fp4.".format(exc)
    ) from exc


# Hadamard "force-on for ALL casts" override. Default is grad-only.
_HADAMARD_ALL = os.environ.get("AITER_FUSED_QUANT_HADAMARD", "0") == "1"


# ---------------------------------------------------------------------------
# Per-shape kernel selection (advanced, opt-in via AITER_KERNEL_SEL=1).
# ---------------------------------------------------------------------------
_KERNEL_SEL_CACHE = None


def _use_kernel_selection():
    global _KERNEL_SEL_CACHE
    if _KERNEL_SEL_CACHE is None:
        _KERNEL_SEL_CACHE = os.environ.get("AITER_KERNEL_SEL", "0") == "1"
    return _KERNEL_SEL_CACHE


def _get_kernel_name(M, N, K):
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


# ---------------------------------------------------------------------------
# Raw cast helpers -- role-specific flag combinations baked in.
# ---------------------------------------------------------------------------

def _cast_act_raw(x):
    """Activation cast: rowwise only, unshuffled, no Hadamard.

    Used by the forward-only primal path (no autograd).
    """
    return _cast_mxfp4_op(x, shuffle_fp4=False, use_hadamard=_HADAMARD_ALL)


def _cast_wt_raw(x):
    """Weight cast: rowwise only, B-preshuffle shuffled, no Hadamard.

    Used by the forward-only primal path (no autograd).
    """
    return _cast_mxfp4_op(x, shuffle_fp4=True, use_hadamard=_HADAMARD_ALL)


def _cast_act_dual_raw(x):
    """Activation dual-cast: rowwise unshuffled + columnwise shuffled, no Hadamard.

    rowwise -> A operand of fprop GEMM
    columnwise -> A operand of wgrad GEMM (via the FFI's NT-layout dB call).
    """
    return _cast_mxfp4_dual_op(
        x,
        shuffle_fp4=False,
        shuffle_colwise_fp4=True,
        use_hadamard=_HADAMARD_ALL,
    )


def _cast_wt_dual_raw(x):
    """Weight dual-cast: rowwise shuffled + columnwise shuffled, no Hadamard.

    rowwise -> B operand of fprop GEMM
    columnwise -> B operand of dgrad GEMM (avoids re-quantization in backward).
    """
    return _cast_mxfp4_dual_op(
        x,
        shuffle_fp4=True,
        shuffle_colwise_fp4=True,
        use_hadamard=_HADAMARD_ALL,
    )


def _cast_grad_dual_raw(x):
    """Grad dual-cast: rowwise unshuffled + columnwise unshuffled, Hadamard ON.

    rowwise -> A operand of dgrad GEMM (dA = grad_out @ B_col).
    columnwise -> A operand of wgrad GEMM (dB = grad_out_col @ A_col^T) --
    linear (not B-preshuffle) layout because it plays the A role here.

    Hadamard is applied inside the fused HIP cast kernel: a butterfly
    multiply over each 32-element block before extracting the E8M0 scale and
    packed FP4 values. This decorrelates outliers in the gradient
    distribution. Matches TE PyTorch MXFP4 grad-quantizer default.
    """
    return _cast_mxfp4_dual_op(
        x,
        shuffle_fp4=False,
        shuffle_colwise_fp4=False,
        use_hadamard=True,
    )


# ---------------------------------------------------------------------------
# custom_partitioning wrappers around the cast helpers.
# ---------------------------------------------------------------------------

def _get_spec(info):
    """Safely extract PartitionSpec from an arg_shape, handling None sharding."""
    if info.sharding is None:
        return P(*([None] * len(info.shape)))
    return info.sharding.spec


# ----- rowwise-only casts (act / wt) for the forward-only primal -----

@custom_partitioning
def _cast_act(x):
    return _cast_act_raw(x)


@custom_partitioning
def _cast_wt(x):
    return _cast_wt_raw(x)


def _cast_rowwise_infer_sharding(mesh, arg_shapes, result_shape):
    x_spec = _get_spec(arg_shapes[0])
    m_axis = x_spec[0]
    return (NamedSharding(mesh, P(m_axis, None)),
            NamedSharding(mesh, P(m_axis, None)))


def _make_rowwise_cast_partition(raw_fn):
    def _partition(mesh, arg_shapes, result_shape):
        x_spec = _get_spec(arg_shapes[0])
        m_axis = x_spec[0]
        in_spec = P(m_axis, None)
        out_spec = P(m_axis, None)

        def _lowered(x):
            return raw_fn(x)

        return (mesh, _lowered,
                (NamedSharding(mesh, out_spec), NamedSharding(mesh, out_spec)),
                (NamedSharding(mesh, in_spec),))
    return _partition


_cast_act.def_partition(
    _make_rowwise_cast_partition(_cast_act_raw),
    infer_sharding_from_operands=_cast_rowwise_infer_sharding,
    sharding_rule="m k -> m kp, m sp",
    need_replication_factors=("k", "kp", "sp"),
)
_cast_wt.def_partition(
    _make_rowwise_cast_partition(_cast_wt_raw),
    infer_sharding_from_operands=_cast_rowwise_infer_sharding,
    sharding_rule="m k -> m kp, m sp",
    need_replication_factors=("k", "kp", "sp"),
)


# ----- dual casts (rowwise + columnwise) for fwd-with-residuals + bwd -----
# Outputs (rowwise_fp4, rowwise_scale, colwise_fp4, colwise_scale).
# Rowwise outputs shard on M (first dim); colwise outputs shard M on second dim.

@custom_partitioning
def _cast_act_dual(x):
    return _cast_act_dual_raw(x)


@custom_partitioning
def _cast_wt_dual(x):
    return _cast_wt_dual_raw(x)


@custom_partitioning
def _cast_grad_dual(x):
    return _cast_grad_dual_raw(x)


def _cast_dual_infer_sharding(mesh, arg_shapes, result_shape):
    x_spec = _get_spec(arg_shapes[0])
    m_axis = x_spec[0]
    return (NamedSharding(mesh, P(m_axis, None)),
            NamedSharding(mesh, P(m_axis, None)),
            NamedSharding(mesh, P(None, m_axis)),
            NamedSharding(mesh, P(None, m_axis)))


def _make_dual_cast_partition(raw_fn):
    def _partition(mesh, arg_shapes, result_shape):
        x_spec = _get_spec(arg_shapes[0])
        m_axis = x_spec[0]
        in_spec = P(m_axis, None)
        row_spec = P(m_axis, None)
        col_spec = P(None, m_axis)

        def _lowered(x):
            return raw_fn(x)

        return (mesh, _lowered,
                (NamedSharding(mesh, row_spec), NamedSharding(mesh, row_spec),
                 NamedSharding(mesh, col_spec), NamedSharding(mesh, col_spec)),
                (NamedSharding(mesh, in_spec),))
    return _partition


for _wrapped, _raw in (
    (_cast_act_dual,  _cast_act_dual_raw),
    (_cast_wt_dual,   _cast_wt_dual_raw),
    (_cast_grad_dual, _cast_grad_dual_raw),
):
    _wrapped.def_partition(
        _make_dual_cast_partition(_raw),
        infer_sharding_from_operands=_cast_dual_infer_sharding,
        sharding_rule="m k -> m rkp, m rsp, k cmp, k csp",
        need_replication_factors=("k", "rkp", "rsp", "cmp", "csp"),
    )


# ---------------------------------------------------------------------------
# FP4 GEMM forward / dgrad (NN layout, K is contraction).
# ---------------------------------------------------------------------------

def gemm_fp4(a, b, a_scale, b_scale):
    """Low-level FP4 GEMM with pre-quantized + pre-shuffled inputs.

    Args:
        a: [M, K/2] uint8 -- packed MXFP4 A operand.
        b: [N, K/2] uint8 -- packed + B-preshuffle shuffled MXFP4 B operand.
        a_scale: [M_pad, scale_n_pad] uint8 -- shuffled E8M0 A scales.
        b_scale: [N_pad, scale_n_pad] uint8 -- shuffled E8M0 B scales.

    Returns:
        out: [M, N] bfloat16.
    """
    return _gemm_fp4_ffi(a, b, a_scale, b_scale)


@custom_partitioning
def _fp4_ffi_partitioned(a_packed, b_packed, a_scale, b_scale):
    return _gemm_fp4_ffi(a_packed, b_packed, a_scale, b_scale)


def _resolve_fp4_specs(arg_shapes):
    """For NN layout: a [M, Kp], b [N, Kp]. Returns (m_axis, n_axis)."""
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
# FP4 GEMM wgrad (NT layout, M_batch is contraction -- FSDP-sharded).
#
# A = grad_col_fp4   [N, M/2]   (grad_out columnwise, unshuffled)
# B = input_col_fp4  [K, M/2]   (input columnwise, B-preshuffled)
# OUT = dB           [N, K]
#
# Differs from the fprop/dgrad partition in that the contraction axis (M)
# is FSDP-sharded; the partition callback emits ``jax.lax.psum`` across the
# FSDP mesh axis to reduce partial results. XLA lowers psum to all-reduce
# or reduce-scatter depending on output sharding.
# ---------------------------------------------------------------------------

@custom_partitioning
def _fp4_ffi_partitioned_wgrad(a_packed, b_packed, a_scale, b_scale):
    return _gemm_fp4_ffi(a_packed, b_packed, a_scale, b_scale)


def _extract_fsdp_axis(spec, dim_index):
    if spec is None or dim_index >= len(spec):
        return None
    return spec[dim_index]


def _fp4_wgrad_infer_sharding(mesh, arg_shapes, result_shape):
    a_spec = _get_spec(arg_shapes[0])
    b_spec = _get_spec(arg_shapes[1])
    n_axis = a_spec[0] if a_spec is not None and len(a_spec) >= 1 else None
    k_axis = b_spec[0] if b_spec is not None and len(b_spec) >= 1 else None
    return NamedSharding(mesh, P(n_axis, k_axis))


def _fp4_wgrad_partition(mesh, arg_shapes, result_shape):
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


_fp4_ffi_partitioned_wgrad.def_partition(
    _fp4_wgrad_partition,
    infer_sharding_from_operands=_fp4_wgrad_infer_sharding,
    sharding_rule="n mp, k mp, n ms, k ms -> n k",
    need_replication_factors=(),
)


# ---------------------------------------------------------------------------
# Public high-level API: BF16 in, FP4 internally, BF16 out.
# ---------------------------------------------------------------------------

@partial(jax.custom_vjp, nondiff_argnums=())
def gemm_fp4_bf16(a, b):
    """Compute ``A @ B^T`` in MXFP4 with BF16 inputs and outputs.

    Forward: cast ``a`` and ``b`` to MXFP4 (rowwise only for the no-autograd
    primal) -> AITER FP4 ASM kernel.

    Backward: dual-cast ``a`` and ``b`` (rowwise + columnwise) so the
    backward pass can reuse the columnwise FP4 data. ``grad_out`` is
    dual-cast with Hadamard. ``dA`` runs as ``GemmFp4FwdJA(grad_out_row,
    weight_col)`` and ``dB`` runs as ``GemmFp4FwdJA(grad_out_col,
    input_col)`` via the wgrad partition with FSDP-aware ``psum``.

    Args:
        a: [M, K] bfloat16 -- typically the activation.
        b: [N, K] bfloat16 -- typically the weight.

    Returns:
        out: [M, N] bfloat16.
    """
    a_packed, a_scales = _cast_act(a)
    b_packed, b_scales = _cast_wt(b)
    return _fp4_ffi_partitioned(a_packed, b_packed, a_scales, b_scales)


def _gemm_fp4_bf16_fwd(a, b):
    """Forward-with-residuals: dual-cast keeps columnwise FP4 for backward."""
    a_row, a_row_s, col_a_fp4, col_a_scale = _cast_act_dual(a)
    b_row, b_row_s, col_b_fp4, col_b_scale = _cast_wt_dual(b)
    out = _fp4_ffi_partitioned(a_row, b_row, a_row_s, b_row_s)
    return out, (col_a_fp4, col_a_scale, col_b_fp4, col_b_scale)


def _gemm_fp4_bf16_bwd(residuals, grad_out):
    """Backward: FP4 dA and FP4 dB (wgrad-sharded) using saved columnwise data."""
    col_a_fp4, col_a_scale, col_b_fp4, col_b_scale = residuals

    # Hadamard ON: applied inside the fused cast kernel for grad only.
    go_row, go_row_s, go_col_fp4, go_col_scale = _cast_grad_dual(grad_out)

    # dA = grad_out @ B  -- NN layout, K is the contraction (replicated).
    da = _fp4_ffi_partitioned(go_row, col_b_fp4, go_row_s, col_b_scale)
    # dB = grad_out^T @ A -- NT layout, M_batch is the contraction (FSDP-
    # sharded). The partition callback emits psum across the FSDP mesh.
    db = _fp4_ffi_partitioned_wgrad(go_col_fp4, col_a_fp4, go_col_scale, col_a_scale)
    return da, db


gemm_fp4_bf16.defvjp(_gemm_fp4_bf16_fwd, _gemm_fp4_bf16_bwd)
