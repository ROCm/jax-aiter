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
``JA_FP4_HIGHP_PASSES=<comma-list of {fprop,dgrad,wgrad}>``
    Per-pass high-precision (bf16) fallback (debug / ablation). Each listed
    pass runs its GEMM in bf16 instead of FP4. ``wgrad`` and ``dgrad`` are
    wired (``fprop`` is not):
      * ``wgrad`` -- backward computes ``dB = grad_out^T @ a`` from the RAW
        bf16 grad + RAW bf16 activation (stashed by the fwd), reduced across
        the FSDP mesh with the SAME ``psum`` as the FP4 wgrad (M/batch axis is
        FSDP-sharded).
      * ``dgrad`` -- backward computes ``dA = grad_out @ b`` from the RAW bf16
        grad + RAW bf16 weight (stashed by the fwd). The contraction axis N
        (output features) is REPLICATED under FSDP, so this is a plain bf16
        ``dot_general`` -- NO collective.
    ``dgrad,wgrad`` => the entire backward runs bf16 (dA + dB) while the
    forward stays FP4, isolating forward-FP4 vs backward-FP4 error. **Unset
    (default) => byte-identical all-FP4 backward** (4-tuple residual, FP4
    dA + dB). Reversible debug knob.
``JA_FP4_DGRAD_PREC=fp8``
    FP8-precision dgrad (advanced, opt-in; independent of
    ``JA_FP4_HIGHP_PASSES``). When ``fp8``, the dgrad GEMM ``dA = grad_out @ b``
    runs in fp8 (``float8_e4m3fn``, the OCP fp8 native to gfx950) instead of
    FP4 -- a candidate mixed FP4-fwd / FP8-dgrad recipe and a probe of how much
    precision the dgrad needs. The fwd stashes the RAW bf16 weight (same
    residual as the bf16 dgrad); the bwd casts ``grad_out`` + weight to fp8 and
    ``dot_general``s them. XLA-ROCm lowers this to a REAL fp8 cublasLt matmul
    (``__cublas$...$f8``; ~1.45-1.8x faster than the bf16 dgrad dot on gfx950).
    The forward and wgrad stay FP4. N (contraction) is replicated under FSDP =>
    NO collective. **Unset (default) => byte-identical all-FP4 dgrad.**
    Reversible debug knob.
``JA_FP4_DGRAD_PARTITION=gather``
    FSDP/Shardy-correct FP4 dgrad (advanced, opt-in). The legacy dgrad reuses the
    fprop partition on the packed + colwise-SHUFFLED weight ``col_b``; under FSDP
    the weight is N(out-feature)-sharded so the partitioner ALL-GATHERS the
    shuffled shards, which is NOT concat-compatible (``pack_shuffle`` does not
    distribute over concatenation along N) -> logically scrambled, corrupt dA
    (the "MXFP4 won't converge" bug). ``gather`` (alias ``fix``) routes the dgrad
    through a dedicated ``custom_partitioning`` that forces the LOGICAL bf16
    weight to be gathered over N (need_replication on the N contraction; the
    OPPOSITE of the wgrad reduction), THEN colwise casts / packs / shuffles the
    FULL weight LOCALLY, THEN runs the FP4 GEMM -- concat-safe by construction.
    Byte-identical on a single device; validated under shardy=True. **Unset /
    ``legacy`` (default) => byte-identical legacy dgrad.** Reversible debug knob.
``JA_FP4_SCALE_MARGIN=<int>``
    E8M0 under-flush headroom for the DGRAD (gradient) cast (B2). The fused cast
    kernel computes the per-32-block scale as ``2^(exp - 2 - scale_margin)``;
    a positive margin SHRINKS the block scale so small gradient entries that
    would flush to FP4 code +/-0 instead survive as ``+/-0.5*scale`` -- directly
    targeting the §9 dgrad grad-operand under-flush (recovering the dgrad-bf16
    accuracy win at full FP4 speed, no bf16/fp8 backward GEMM) at the cost of
    clipping the few largest entries. Applied to the grad cast ONLY (fprop /
    weight casts stay at margin 0). The resolved value is printed once at import.
    **Unset / 0 (default) => byte-identical to the legacy ``exp-2`` cast.**
    Reversible debug knob.
``JA_FP4_SR_PASSES=<comma-list of {dgrad_grad,act,wt}>``
    Per-ROLE stochastic rounding (B1). SR makes the FP32->FP4 cast UNBIASED
    (``E[SR(x)] == x``) so small entries probabilistically round up instead of
    flushing to FP4 code 0 -- NVIDIA's actual DGRAD under-flush mitigation
    ("quantize WITH SR" on the dgrad gradient input). ``dgrad_grad`` scopes SR to
    the grad cast (both backward grad operands); ``act`` / ``wt`` scope the
    activation / weight casts. Reuses the per-call ``use_sr`` kernel attr
    (FRONTEND-only, no kernel rebuild). Unlike a positive ``scale_margin`` it does
    NOT shrink the block scale, so it fixes under-flush WITHOUT clipping the large
    grad entries, and can STACK with ``JA_FP4_SCALE_MARGIN``. Differs from the
    global ``AITER_FP4_SR=1`` (SR on every cast).
    **Unset (default) => RNE everywhere (byte-identical).** Reversible.
``JA_FP4_PACK_GATEUP_AG=1``
    Quantize the MLP gate/up weight ALL-GATHER (Lever B, advanced, opt-in).
    Under pure FSDP the gate/up DenseGeneral weight shards its contraction-K
    (``embed``) axis, so the rowwise MXFP4 weight cast (which declares its output
    K-replicated) forces XLA to all-gather the FULL weight in **bf16** before the
    cast. This flag makes the weight cast run LOCALLY on each K-shard (emitting
    packed FP4 + scales sharded on K, rowwise UNSHUFFLED) and gathers the **packed
    u8** operand along the FSDP axis INSIDE the fwd GEMM partition, where it is
    re-assembled with ``shuffle_weight`` / ``e8m0_shuffle`` before the ASM kernel.
    Cuts the gate/up weight all-gather from bf16 -> packed-FP4 (~3.76x less
    volume, ~-5.5 GB/step) with NO numeric change: MXFP4 block=32 along K and the
    per-shard K = embed/fsdp = 16*32 so blocks never cross a shard boundary
    (Hadamard is within-block) -- the per-shard cast is byte-identical to the
    full-K cast, and the gather/shuffle reconstruction is exact (verified). The
    colwise (dgrad) weight cast becomes K-sharded on its OUTPUT axis, which the
    existing GEMM partition gathers as packed u8 too (shuffle-compatible along the
    output axis). **Unset / 0 (default) => byte-identical to the legacy bf16
    weight all-gather (the cast forces K-replication as before).** Reversible:
    flag-gated registration, no kernel rebuild.
``JA_CAPTURE_DIR=<dir>``
    Offline-probe capture hook (debug / analysis only). When set, the
    forward dumps the bf16 ``(a, b)`` operands and the backward dumps the
    bf16 ``grad_out`` for each distinct ``(M, N, K)`` GEMM site (FIRST
    occurrence by default), so an offline replay can rebuild the
    ``(a, b, grad_out)`` triple per projection shape. **Unset (default) =>
    zero effect:** the guard is a trace-time Python check, so no callback /
    slice node is added to the graph and production numerics are
    byte-identical. Reversible: delete the ``_capture_*`` block + the two
    call sites in the fwd / bwd.
``JA_CAPTURE_AFTER_STEP=<N>``
    Steady-state capture knob (only meaningful with ``JA_CAPTURE_DIR`` set).
    The GEMM call site does not see the training step, so this is compared
    against a HOST-side per-shape GEMM-call (fire) counter: a site is
    captured only once its counter reaches ``N``. With ``scan_layers``
    (and remat) a shape fires ~``num_layers`` (or ~2x) times per step, so
    ``N`` ~= ``target_step * fires_per_step``. Default ``0`` => capture the
    FIRST fire (step-0 behavior, backward-compatible).
"""

from __future__ import annotations

import os
from functools import partial

import jax
import jax.numpy as jnp  # noqa: F401  # imported for downstream consumers
from jax.ad_checkpoint import checkpoint_name
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
# AITER_FP4_HADAMARD_OFF=1 fully disables Hadamard on EVERY cast incl. grad
# (debug/ablation; default off => unchanged grad-only / all-on behavior).
_HADAMARD_NONE = os.environ.get("AITER_FP4_HADAMARD_OFF", "0") == "1"
_HADAMARD_ALL = (os.environ.get("AITER_FUSED_QUANT_HADAMARD", "0") == "1") and not _HADAMARD_NONE


def _parse_hadamard_passes():
    raw = os.environ.get("JA_FP4_HADAMARD_PASSES", "").strip().lower()
    if not raw:
        if _HADAMARD_NONE:
            return frozenset()
        if _HADAMARD_ALL:
            return frozenset({"fprop", "dgrad", "wgrad"})
        return None  # Preserve legacy grad-dual behavior when unset.
    passes = {p.strip() for p in raw.split(",") if p.strip()}
    if "none" in passes or "off" in passes:
        return frozenset()
    if "all" in passes:
        return frozenset({"fprop", "dgrad", "wgrad"})
    if "backward" in passes or "bwd" in passes:
        passes.update({"dgrad", "wgrad"})
    allowed = {"grad", "fprop", "dgrad", "wgrad"}
    return frozenset(p for p in passes if p in allowed)


_HADAMARD_PASSES = _parse_hadamard_passes()
_HADAMARD_LEGACY_GRAD = _HADAMARD_PASSES is None
_H_FPROP = (not _HADAMARD_LEGACY_GRAD) and "fprop" in _HADAMARD_PASSES
_H_DGRAD = (not _HADAMARD_LEGACY_GRAD) and "dgrad" in _HADAMARD_PASSES
_H_WGRAD = (not _HADAMARD_LEGACY_GRAD) and "wgrad" in _HADAMARD_PASSES
_H_GRAD_COMPAT = (not _HADAMARD_LEGACY_GRAD) and "grad" in _HADAMARD_PASSES


# ---------------------------------------------------------------------------
# Stochastic rounding (SR) for the FP32->FP4 cast (advanced, opt-in via
# AITER_FP4_SR=1). Same E2M1 grid as the default RNE (NO extra precision) but
# unbiased: E[dequant(SR(x))] == x. RNE's per-step rounding bias does not
# cancel across steps; SR's does, so its payoff is cumulative over training --
# a lever a single-step calibration cannot observe. When enabled, SR is applied
# to ALL FP4 casts (act / weight / grad), which covers both dgrad operands
# (grad_out + weight). Read at import (parity with _HADAMARD_ALL); the cast
# helpers look the module global up at trace time so tests can monkeypatch it.
# Default off => byte-identical RNE path (the kernel compiles both; the runtime
# use_sr flag selects). Reversible debug knob.
# ---------------------------------------------------------------------------
_SR_ALL = os.environ.get("AITER_FP4_SR", "0") == "1"


# ---------------------------------------------------------------------------
# Per-ROLE stochastic rounding (B1, advanced, opt-in via JA_FP4_SR_PASSES).
# Comma-list of {dgrad_grad, act, wt} selecting WHICH cast roles use SR, vs the
# global AITER_FP4_SR (_SR_ALL) which forces SR on every cast. This is NVIDIA's
# actual DGRAD under-flush mitigation: SR on the gradient cast makes the cast
# UNBIASED (E[SR(x)] == x), so small grad entries probabilistically round UP to
# +/-0.5 instead of deterministically flushing to FP4 code 0 -- the under-flush
# is averaged out across steps WITHOUT shrinking the block scale (so, unlike a
# positive scale-margin, it does NOT clip the large grad entries). NVIDIA's
# nvfp4 recipe applies "quantize WITH SR" to the dgrad gradient input.
#   * dgrad_grad (aliases: grad, dgrad) -> SR on the grad cast (_cast_grad_dual):
#     its rowwise output is the dgrad A-operand (grad_out) and colwise is the
#     wgrad grad-operand -- i.e. SR on BOTH backward grad operands, one launch.
#   * act (aliases: fprop, fprop_act) -> SR on the activation casts.
#   * wt  (alias: weight)            -> SR on the weight casts.
# Reuses the per-call use_sr kernel attr (cast kernel compiles both RNE + SR
# paths; runtime-selected) => FRONTEND-ONLY, NO kernel rebuild. Read at import
# (parity with _SR_ALL); helpers look the globals up at trace time. Default
# empty + AITER_FP4_SR unset => every role RNE (byte-identical). Reversible.
# ---------------------------------------------------------------------------
def _parse_sr_passes():
    raw = os.environ.get("JA_FP4_SR_PASSES", "")
    return frozenset(p.strip().lower() for p in raw.split(",") if p.strip())


_SR_PASSES = _parse_sr_passes()
_SR_GRAD = _SR_ALL or bool(_SR_PASSES & {"dgrad_grad", "grad", "dgrad"})
_SR_DGRAD_ROW = _SR_GRAD or bool(_SR_PASSES & {"dgrad_row", "dgrad_grad_row"})
_SR_WGRAD_COL = _SR_GRAD or bool(_SR_PASSES & {"wgrad_col", "wgrad_grad_col"})
_SR_ACT = _SR_ALL or bool(_SR_PASSES & {"act", "fprop", "fprop_act"})
_SR_WT = _SR_ALL or bool(_SR_PASSES & {"wt", "weight"})


# ---------------------------------------------------------------------------
# Per-pass high-precision (bf16) fallback (advanced, opt-in via
# JA_FP4_HIGHP_PASSES). Comma-list of {fprop,dgrad,wgrad}; each listed pass
# runs its GEMM in bf16 instead of FP4. Default empty => every pass stays FP4
# (production recipe unchanged). Read at import (parity with _HADAMARD_ALL);
# the fwd/bwd look the module global up at trace time so tests can monkeypatch
# it. Reversible debug knob: unset => byte-identical all-FP4 behavior.
#
# "wgrad" and "dgrad" are wired (set BOTH => the whole backward runs bf16
# while the forward stays FP4, isolating forward-FP4 vs backward-FP4 error):
#   * wgrad -- dB = grad_out^T @ a (contraction over the FSDP-sharded M/batch
#     axis; needs an all-reduce). The FP4 pass most exposed to gradient
#     under-flush: token/M-axis blocks dominated by outlier tokens set the
#     shared E8M0 scale, flushing small entries to FP4 code 0; feeds the
#     optimizer weight update directly.
#   * dgrad -- dA = grad_out @ b (contraction over the output-feature axis N,
#     which is REPLICATED under FSDP, so a plain local dot -- NO collective).
# ---------------------------------------------------------------------------
def _parse_highp_passes():
    raw = os.environ.get("JA_FP4_HIGHP_PASSES", "")
    return frozenset(p.strip().lower() for p in raw.split(",") if p.strip())


_HIGHP_PASSES = _parse_highp_passes()
_HIGHP_WGRAD = "wgrad" in _HIGHP_PASSES
_HIGHP_DGRAD = "dgrad" in _HIGHP_PASSES


# ---------------------------------------------------------------------------
# FP8-precision dgrad (advanced, opt-in via JA_FP4_DGRAD_PREC=fp8). Independent
# of JA_FP4_HIGHP_PASSES (which selects the bf16 fallback). When "fp8" the dgrad
# GEMM dA = grad_out @ b runs in fp8 (float8_e4m3fn) instead of FP4 -- a
# candidate mixed FP4-fwd / FP8-dgrad recipe and a probe of how much precision
# the dgrad needs. fwd + wgrad stay FP4. Read at import (parity with
# _HIGHP_*); the fwd/bwd look the module global up at trace time so tests can
# monkeypatch it. Default empty => dgrad stays FP4 (byte-identical).
# ---------------------------------------------------------------------------
_DGRAD_PREC = os.environ.get("JA_FP4_DGRAD_PREC", "").strip().lower()
_FP8_DGRAD = _DGRAD_PREC == "fp8"


# ---------------------------------------------------------------------------
# Per-OPERAND dgrad precision split (diagnostic, opt-in via JA_FP4_DGRAD_SPLIT).
# Isolates whether the FP4-dgrad damage is the gradient (dY) quantization, the
# weight (W) quantization, or the FP4xFP4 interaction. The a4w4 kernel needs
# BOTH operands FP4, so this uses a pure-JAX MXFP4 FAKE-QUANT (quant->dequant,
# Hadamard OFF) on the selected operand(s) and a bf16 dot_general for dA:
#   dy   -> dY FP4-fakequant, W bf16
#   w    -> dY bf16,          W FP4-fakequant
#   both -> both fake-quant  (cross-check vs the real FP4 dgrad, nohad)
# Default "" => unchanged (byte-identical). Diagnostic only; not a recipe.
# ---------------------------------------------------------------------------
_DGRAD_SPLIT = os.environ.get("JA_FP4_DGRAD_SPLIT", "").strip().lower()
if _DGRAD_SPLIT not in ("", "dy", "w", "both"):
    _DGRAD_SPLIT = ""


# ---------------------------------------------------------------------------
# FSDP/Shardy dgrad partition mode (opt-in via JA_FP4_DGRAD_PARTITION).
#
# The default (legacy) FP4 dgrad reuses the fprop custom_partitioning on the
# COLUMNWISE weight cast ``col_b`` (packed + B-preshuffle). Under FSDP the weight
# is N(out-features)-sharded, so the dgrad contraction axis N is sharded and the
# partitioner ALL-GATHERS ``col_b`` over N. But the colwise shuffle is NOT
# concat-compatible along N -- ``pack_shuffle(concat(W_i)) != concat(pack_shuffle(W_i))``
# -- so the gathered buffer is physically valid but logically SCRAMBLED, giving a
# corrupt input-gradient dA (the multi-week "MXFP4 won't converge" bug). Proven:
#   * 1-GPU (no gather) dgrad is correct (probe_dgrad_kernel.py).
#   * FSDP fakeq / bf16 dgrad (gathers the LOGICAL weight) converge.
#   * offline probe_colwise_gather.py: concat-of-shuffled-shards dgrad cos~0.01
#     vs full-cast dgrad cos~0.99; the UNSHUFFLED cast is concat-safe (0% byte
#     diff) so the SHUFFLE is the culprit; concat of LOGICAL bf16 shards == W.
#
#   "gather" / "fix" -> dedicated dgrad custom_partitioning that forces the
#     LOGICAL bf16 weight to be gathered over N (need_replication on the N
#     contraction factor; the OPPOSITE of the wgrad's reduction_factors, because
#     dA is per-token not summed over the sharded axis), THEN colwise cast /
#     pack / shuffle the FULL weight LOCALLY, THEN the FP4 dgrad GEMM -- so the
#     shuffle never runs on gathered shuffled shards. Concat-safe by construction.
#     On a single device (no gather) it is byte-identical to the legacy path
#     (same cast params, same GEMM). Works under BOTH Shardy (sharding_rule
#     need_replication) and GSPMD (the partition callback declares the weight
#     replicated); validated under shardy=True (the user requirement).
#   unset / "legacy" -> legacy path (DEFAULT; byte-identical, unchanged).
# Read at import; the fwd/bwd look the module globals up at trace time so tests
# can monkeypatch. Reversible debug knob.
# ---------------------------------------------------------------------------
_DGRAD_PARTITION = os.environ.get("JA_FP4_DGRAD_PARTITION", "").strip().lower()
if _DGRAD_PARTITION not in ("", "legacy", "gather", "fix"):
    _DGRAD_PARTITION = ""
_DGRAD_GATHER = _DGRAD_PARTITION in ("gather", "fix")

# E2M1 grid (magnitudes) + round-to-nearest midpoints, for the fake-quant.
_E2M1_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MIDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def _fake_quant_mxfp4(x, axis):
    """Pure-JAX MXFP4 fake-quant (quant->dequant) over 32-blocks along ``axis``.

    Reproduces the kernel's 1x32 scheme (Hadamard OFF): per-block amax -> E8M0
    round-nearest-pow2 scale with the legacy ``exp-2`` headroom -> E2M1 (RNE)
    -> dequant. Diagnostic only (JA_FP4_DGRAD_SPLIT); not on any default path.
    """
    import jax.numpy as _jnp
    xf = x.astype(_jnp.float32)
    xm = _jnp.moveaxis(xf, axis, -1)
    shp = xm.shape
    nblk = shp[-1] // 32
    xb = xm.reshape(shp[:-1] + (nblk, 32))
    amax = _jnp.max(_jnp.abs(xb), axis=-1, keepdims=True)          # [..., nblk, 1]
    # E8M0 round-to-nearest power-of-two (mantissa-MSB carry), then exp-2 headroom.
    bits = jax.lax.bitcast_convert_type(amax, _jnp.uint32)
    bits = (bits + _jnp.uint32(0x200000)) & _jnp.uint32(0xFF800000)
    exp = ((bits >> 23) & _jnp.uint32(0xFF)).astype(_jnp.int32) - 127
    scale_exp = _jnp.where(amax > 0.0, exp - 2, 0)
    native = _jnp.exp2(scale_exp.astype(_jnp.float32))            # [..., nblk, 1]
    q = xb / native
    s = _jnp.sign(q)
    aq = _jnp.abs(q)
    mids = _jnp.asarray(_E2M1_MIDS, dtype=_jnp.float32)
    grid = _jnp.asarray(_E2M1_GRID, dtype=_jnp.float32)
    idx = _jnp.searchsorted(mids, aq)                            # 0..7 (RN on the grid)
    deq = (s * grid[idx]) * native
    deq = deq.reshape(shp)
    return _jnp.moveaxis(deq, -1, axis).astype(x.dtype)


# ---------------------------------------------------------------------------
# E8M0 scale-margin for the DGRAD (gradient) cast (B2, advanced, opt-in via
# JA_FP4_SCALE_MARGIN=<int>). The fused cast kernel computes the per-32-block
# E8M0 scale as 2^(exp - 2 - scale_margin). The hardcoded "-2" is the legacy
# headroom below the FP4 max; scale_margin adds EXTRA headroom:
#   * scale_margin > 0  -> SMALLER block scale -> small gradient entries that
#     would flush to FP4 code +/-0 instead survive as +/-0.5*scale. Directly
#     targets the §9 dgrad under-flush (grad-operand block-scale median ~2^-20,
#     ~14.5% of grad entries flushed to 0) at the cost of clipping the few
#     largest entries (a little more saturation). Recovers the dgrad-bf16
#     accuracy win at full FP4 speed (no bf16/fp8 backward GEMM).
#   * scale_margin < 0  -> larger block scale (more under-flush, less clip).
#   * scale_margin == 0 -> byte-identical to the legacy exp-2 cast (DEFAULT).
# Applied to the GRAD cast ONLY (_cast_grad_dual): its rowwise output is the
# dgrad A-operand (dA = grad_out @ b) -- the §9 culprit -- and its colwise
# output is the wgrad grad-operand (same gradient tensor). fprop activation +
# weight casts stay at margin 0. Read at import (parity with _HADAMARD_ALL /
# _SR_ALL); the grad cast helper looks the module global up at trace time so
# tests can monkeypatch it. Reversible debug knob.
# ---------------------------------------------------------------------------
def _parse_scale_margin():
    raw = os.environ.get("JA_FP4_SCALE_MARGIN", "").strip()
    if not raw:
        return 0
    try:
        return int(raw)
    except ValueError:
        return 0


_SCALE_MARGIN = _parse_scale_margin()


# ---------------------------------------------------------------------------
# OAS -- overflow-aware scale SELECTION (paper 2603.08713, cast-only, opt-in via
# JA_FP4_OAS=1). The cast kernel's compute_e8m0_scale gains a scale_mode: mode 1
# FLOORs amax to a power of two (exp = floor(log2 amax)) instead of rounding to
# the nearest, so the per-32-block max normalizes to [4, 8) and targets the high
# end of the E2M1 range (~(3.5,7]) -- letting the top of a binade OVERFLOW (clip
# to +/-6) in exchange for ~2x more resolution on the small entries that would
# otherwise flush to zero. DISTINCT from JA_FP4_SCALE_MARGIN: margin is a
# CONSTANT uniform integer shift on every block (the prior over-clip failure at
# +1); OAS is a DATA-DEPENDENT rounding-rule change (~half a bit average, bounded
# overflow). Applied to ALL casts (act / wt / grad) so the whole pipeline gets
# the overflow-aware scale. Read at import (helpers look it up at trace time so
# tests can monkeypatch). Default unset => scale_mode 0 => byte-identical.
# ---------------------------------------------------------------------------
_OAS_MODE = 1 if os.environ.get("JA_FP4_OAS", "0") == "1" else 0


# ---------------------------------------------------------------------------
# 2D weight scaling (cast-only, opt-in via JA_FP4_WT2D=1). The WEIGHT dual cast
# shares ONE UE8M0 scale per 32x32 tile across its rowwise + columnwise outputs
# (use_2d_scale in the cast kernel reduces the tile amax once and feeds both
# phases), so the weight FP4 codes are IDENTICAL in fprop (rowwise B) and dgrad
# (colwise B): W_fprop == W_dgrad, removing the row/col scale-inconsistency the
# 1x32 dual cast introduces (a coarse stand-in for NVFP4's 2D-16x16 weight
# scale, HW-infeasible on gfx950). Activations + grads stay 1x32 (their backward
# operands are not the same tensor cast two ways). Applied to the weight dual
# cast ONLY. Read at import. Default unset => byte-identical 1x32 weight cast.
# ---------------------------------------------------------------------------
_WT2D = os.environ.get("JA_FP4_WT2D", "0") == "1"


# ---------------------------------------------------------------------------
# Lever B -- quantized gate/up weight all-gather (opt-in via
# JA_FP4_PACK_GATEUP_AG=1). See the module docstring. Default OFF =>
# byte-identical bf16 weight all-gather (rowwise/dual weight casts force
# K-replication, exactly as before).
# ---------------------------------------------------------------------------
_PACK_GATEUP_AG = os.environ.get("JA_FP4_PACK_GATEUP_AG", "0") == "1"

# Policy log: emit the resolved DGRAD-cast scale margin once at import so each
# training leg's log records exactly which margin its grad cast used.
print("[ja-fp4] DGRAD cast scale_margin = %d "
      "(E8M0 scale 2^(exp-2-margin); 0 = legacy/bit-identical) | "
      "OAS scale_mode = %d (1 = floor/overflow-aware) | WT2D = %s "
      "(2D 32x32 weight scale) | pack_gateup_ag = %s "
      "(Lever B: quantized gate/up weight all-gather) | "
      "SR roles: grad=%s dgrad_row=%s wgrad_col=%s act=%s wt=%s | hadamard_passes=%s "
      "| dgrad_partition=%s"
      % (_SCALE_MARGIN, _OAS_MODE, _WT2D, _PACK_GATEUP_AG,
         _SR_GRAD, _SR_DGRAD_ROW, _SR_WGRAD_COL,
         _SR_ACT, _SR_WT,
         "legacy_grad" if _HADAMARD_LEGACY_GRAD else ",".join(sorted(_HADAMARD_PASSES)),
         _DGRAD_PARTITION or "legacy"),
      flush=True)


# ---------------------------------------------------------------------------
# Remat residual-save tagging (advanced, opt-in via JA_FP4_REMAT_SAVE_COL).
# Under MaxText layer-level remat (jax.checkpoint) the columnwise FP4 dual-cast
# residuals that the custom_vjp fwd declares are RECOMPUTED in the backward
# unless the remat policy is told to SAVE them. This flag tags those residual
# outputs with jax.ad_checkpoint.checkpoint_name so a matching MaxText
# save_only_these_names policy (e.g. remat_policy=minimal_flash_save_fp4col)
# keeps them as residuals instead of re-firing CastMxfp4DualJA in backward.
#
# This is a graph-scheduling lever ONLY: checkpoint_name is an identity at
# lowering (name_p -> x), so the saved FP4 residual is bit-identical to the
# recomputed one and numerics are unchanged. The names below MUST match the
# MaxText policy. Default unset => no name_p added => byte-identical graph to
# the recompute path (and a no-op for any non-matching remat policy).
#   unset / "0" / "none"  : tag nothing (default)
#   "wt"                  : tag the weight-columnwise residual only
#                           (~6 ms/step floor at 8B, lowest OOM risk)
#   "act"                 : tag the activation-columnwise residual only
#                           (~33 ms/step at 8B, the larger chunk)
#   "both" / "1" / "all"  : tag both columnwise residuals
# Read at import (parity with _HADAMARD_ALL / _HIGHP_*); the fwd looks the
# module globals up at trace time so tests can monkeypatch them.
# ---------------------------------------------------------------------------
_REMAT_SAVE_COL = os.environ.get("JA_FP4_REMAT_SAVE_COL", "").strip().lower()
_REMAT_SAVE_WT_COL = _REMAT_SAVE_COL in ("wt", "both", "all", "1")
_REMAT_SAVE_ACT_COL = _REMAT_SAVE_COL in ("act", "both", "all", "1")

# checkpoint_name tags for the columnwise FP4 residuals. MUST match the
# MaxText remat save policy (decoders.py get_remat_policy).
_MXFP4_ACT_COL_NAME = "mxfp4_act_col"
_MXFP4_WT_COL_NAME = "mxfp4_wt_col"


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

def _h_act_row():
    return _HADAMARD_ALL if _HADAMARD_LEGACY_GRAD else _H_FPROP


def _h_wt_row():
    return _HADAMARD_ALL if _HADAMARD_LEGACY_GRAD else _H_FPROP


def _h_act_col():
    return _HADAMARD_ALL if _HADAMARD_LEGACY_GRAD else _H_WGRAD


def _h_wt_col():
    return _HADAMARD_ALL if _HADAMARD_LEGACY_GRAD else _H_DGRAD


def _h_grad_row():
    return (not _HADAMARD_NONE) if _HADAMARD_LEGACY_GRAD else (_H_DGRAD or _H_GRAD_COMPAT)


def _h_grad_col():
    return (not _HADAMARD_NONE) if _HADAMARD_LEGACY_GRAD else (_H_WGRAD or _H_GRAD_COMPAT)


def _cast_dual_selective(x, *, shuffle_fp4, shuffle_colwise_fp4,
                         row_hadamard, col_hadamard, row_sr, col_sr,
                         scale_margin=0, scale_mode=0, use_2d_scale=False):
    """Dual-cast with independent row/col Hadamard/SR in ONE fused launch.

    The fused dual-cast kernel takes PER-DIRECTION Hadamard + SR flags
    (``CastMxfp4DualJA`` ``use_hadamard``/``use_hadamard_col`` +
    ``use_sr``/``use_sr_col``), so a single launch emits asymmetric row/col
    settings -- no split cast, no redundant rowwise recompute. When the row and
    col settings match this is byte-identical to the legacy matched path; when
    they differ it replaces the former 2-launch split, removing the split-cast
    throughput tax on selective Hadamard/SR placements.
    """
    return _cast_mxfp4_dual_op(
        x,
        shuffle_fp4=shuffle_fp4,
        shuffle_colwise_fp4=shuffle_colwise_fp4,
        use_hadamard=row_hadamard,
        use_hadamard_col=col_hadamard,
        use_sr=row_sr,
        use_sr_col=col_sr,
        scale_margin=scale_margin,
        scale_mode=scale_mode,
        use_2d_scale=use_2d_scale,
    )


def _cast_act_raw(x):
    """Activation cast: rowwise only, unshuffled, no Hadamard.

    Used by the forward-only primal path (no autograd).
    """
    return _cast_mxfp4_op(x, shuffle_fp4=False, use_hadamard=_h_act_row(),
                          use_sr=_SR_ACT, scale_mode=_OAS_MODE)


def _cast_wt_raw(x):
    """Weight cast: rowwise only, B-preshuffle shuffled, no Hadamard.

    Used by the forward-only primal path (no autograd).
    """
    return _cast_mxfp4_op(x, shuffle_fp4=True, use_hadamard=_h_wt_row(),
                          use_sr=_SR_WT, scale_mode=_OAS_MODE)


def _cast_act_dual_raw(x):
    """Activation dual-cast: rowwise unshuffled + columnwise shuffled, no Hadamard.

    rowwise -> A operand of fprop GEMM
    columnwise -> A operand of wgrad GEMM (via the FFI's NT-layout dB call).
    """
    return _cast_dual_selective(
        x,
        shuffle_fp4=False,
        shuffle_colwise_fp4=True,
        row_hadamard=_h_act_row(),
        col_hadamard=_h_act_col(),
        row_sr=_SR_ACT,
        col_sr=_SR_ACT,
        scale_mode=_OAS_MODE,
    )


def _cast_wt_dual_raw(x):
    """Weight dual-cast: rowwise shuffled + columnwise shuffled, no Hadamard.

    rowwise -> B operand of fprop GEMM
    columnwise -> B operand of dgrad GEMM (avoids re-quantization in backward).
    """
    return _cast_dual_selective(
        x,
        shuffle_fp4=True,
        shuffle_colwise_fp4=True,
        row_hadamard=_h_wt_row(),
        col_hadamard=_h_wt_col(),
        row_sr=_SR_WT,
        col_sr=_SR_WT,
        scale_mode=_OAS_MODE,
        use_2d_scale=_WT2D,
    )


# ----- Lever B: K-sharded weight cast helpers (JA_FP4_PACK_GATEUP_AG) -----
# When the weight's contraction-K axis is FSDP-sharded (gate/up), the rowwise
# weight cast runs LOCALLY on each K-shard and emits the FP4 data + scales
# UNSHUFFLED (shuffle_fp4=False, shuffle_scales=False). Unshuffled packed bytes
# and linear scales concatenate cleanly along K, so the fwd GEMM partition can
# all-gather the packed u8 along the FSDP axis and re-assemble the kernel layout
# with shuffle_weight / e8m0_shuffle. (Shuffled rowwise output would entangle
# K-blocks into the row axis and could NOT be gathered along K.)

def _cast_wt_row_unshuf_raw(x):
    """Rowwise weight cast, UNSHUFFLED (Lever B K-sharded path).

    Emits packed FP4 + LINEAR E8M0 scales (no B-preshuffle, no scale shuffle) so
    the per-K-shard outputs concatenate cleanly along K. The fwd GEMM partition
    shuffles the gathered full-K operand before the ASM kernel.
    """
    return _cast_mxfp4_op(x, shuffle_fp4=False, shuffle_scales=False,
                          use_hadamard=_h_wt_row(), use_sr=_SR_WT,
                          scale_mode=_OAS_MODE)


def _cast_wt_dual_ksharded_raw(x):
    """Weight dual-cast for a K-sharded weight (Lever B).

    Rowwise output is UNSHUFFLED (gathered + shuffled later in the fwd GEMM);
    colwise output is SHUFFLED as usual (its gather, along the OUTPUT axis, is
    shuffle-compatible). Two launches because the rowwise needs unshuffled scales
    while the colwise needs shuffled scales (the dual kernel shares one
    shuffle_scales flag). Byte-identical VALUES to ``_cast_wt_dual_raw`` -- only
    the rowwise shuffle/scale-shuffle layout differs (re-applied after gather).
    """
    r_packed, r_scale = _cast_wt_row_unshuf_raw(x)
    _, _, c_packed, c_scale = _cast_mxfp4_dual_op(
        x,
        shuffle_fp4=True,
        shuffle_colwise_fp4=True,
        use_hadamard=_h_wt_row(),
        use_hadamard_col=_h_wt_col(),
        use_sr=_SR_WT,
        use_sr_col=_SR_WT,
        scale_mode=_OAS_MODE,
        use_2d_scale=_WT2D,
    )
    return r_packed, r_scale, c_packed, c_scale


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
    return _cast_dual_selective(
        x,
        shuffle_fp4=False,
        shuffle_colwise_fp4=False,
        row_hadamard=_h_grad_row(),
        col_hadamard=_h_grad_col(),
        row_sr=_SR_DGRAD_ROW,
        col_sr=_SR_WGRAD_COL,
        scale_margin=_SCALE_MARGIN,
        scale_mode=_OAS_MODE,
    )


# ---------------------------------------------------------------------------
# custom_partitioning wrappers around the cast helpers.
# ---------------------------------------------------------------------------

def _get_spec(info):
    """Safely extract PartitionSpec from an arg_shape, handling None sharding."""
    if info.sharding is None:
        return P(*([None] * len(info.shape)))
    return info.sharding.spec


# ----- Lever B: K-shard padding-alignment guard (JA_FP4_PACK_GATEUP_AG) -------
# The fused MXFP4 cast pads its E8M0 scale column count to a multiple of 8 and
# its row/col extents to a multiple of 256. A per-K-shard cast therefore tiles
# cleanly under the declared K-sharding ONLY when each shard's K is a multiple of
# 256 (=> scale columns K_shard/32 is a multiple of 8 and the 256-pad is a no-op).
# Canonical 8B: embed=4096 / fsdp=8 = 512 (aligned). When NOT aligned the
# packed-AG path falls back to the legacy bf16 weight all-gather (still correct).

def _mesh_axis_size(mesh, axis):
    if axis is None:
        return 1
    if isinstance(axis, tuple):
        s = 1
        for ax in axis:
            s *= mesh.shape[ax]
        return s
    return mesh.shape[axis]


def _kshard_packed_ag_ok(mesh, k_axis, k_global):
    """True iff a K-sharded weight can use the packed-FP4 all-gather path."""
    if k_axis is None:
        return False
    n = _mesh_axis_size(mesh, k_axis)
    if n <= 1 or (k_global % n) != 0:
        return False
    return (k_global // n) % 256 == 0


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


# ----- Lever B: K-sharded-aware rowwise weight cast (JA_FP4_PACK_GATEUP_AG) ---
# Allows the weight's contraction-K axis to stay FSDP-sharded: the cast runs
# locally on the K-shard and emits UNSHUFFLED packed FP4 + linear scales sharded
# on K (gathered + shuffled later in the fwd GEMM). When K is NOT sharded (down
# proj, or unsharded) it behaves exactly like the legacy rowwise weight cast.

def _cast_wt_rowwise_infer_ag(mesh, arg_shapes, result_shape):
    x_spec = _get_spec(arg_shapes[0])
    m_axis = x_spec[0]
    k_axis = x_spec[1] if len(x_spec) > 1 else None
    if _kshard_packed_ag_ok(mesh, k_axis, arg_shapes[0].shape[1]):
        return (NamedSharding(mesh, P(m_axis, k_axis)),
                NamedSharding(mesh, P(m_axis, k_axis)))
    return _cast_rowwise_infer_sharding(mesh, arg_shapes, result_shape)


def _cast_wt_rowwise_partition_ag(mesh, arg_shapes, result_shape):
    x_spec = _get_spec(arg_shapes[0])
    m_axis = x_spec[0]
    k_axis = x_spec[1] if len(x_spec) > 1 else None
    if _kshard_packed_ag_ok(mesh, k_axis, arg_shapes[0].shape[1]):
        in_spec = P(m_axis, k_axis)
        out_spec = P(m_axis, k_axis)
        raw_fn = _cast_wt_row_unshuf_raw
    else:
        in_spec = P(m_axis, None)
        out_spec = P(m_axis, None)
        raw_fn = _cast_wt_raw

    def _lowered(x):
        return raw_fn(x)

    return (mesh, _lowered,
            (NamedSharding(mesh, out_spec), NamedSharding(mesh, out_spec)),
            (NamedSharding(mesh, in_spec),))


_cast_act.def_partition(
    _make_rowwise_cast_partition(_cast_act_raw),
    infer_sharding_from_operands=_cast_rowwise_infer_sharding,
    sharding_rule="m k -> m kp, m sp",
    need_replication_factors=("k", "kp", "sp"),
)
if _PACK_GATEUP_AG:
    _cast_wt.def_partition(
        _cast_wt_rowwise_partition_ag,
        infer_sharding_from_operands=_cast_wt_rowwise_infer_ag,
        sharding_rule="m k -> m kp, m sp",
        need_replication_factors=(),
    )
else:
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


# ----- Lever B: K-sharded-aware weight dual cast (JA_FP4_PACK_GATEUP_AG) ------
# For a K-sharded weight (gate/up) the dual cast runs locally on the K-shard:
#   * rowwise output  -> UNSHUFFLED, sharded on K (axis1) -> fprop B (gathered +
#     shuffled in the fwd GEMM).
#   * colwise output  -> SHUFFLED, sharded on K (axis0 = the dgrad OUTPUT axis)
#     -> dgrad B (gathered as packed u8 by the existing GEMM partition; the
#     colwise B-preshuffle is shuffle-compatible with a gather along its output
#     axis, verified).
# When K is NOT sharded (down proj) it behaves exactly like the legacy weight
# dual cast (single fused launch, shuffled rowwise).

def _cast_wt_dual_infer_ag(mesh, arg_shapes, result_shape):
    x_spec = _get_spec(arg_shapes[0])
    m_axis = x_spec[0]
    k_axis = x_spec[1] if len(x_spec) > 1 else None
    if _kshard_packed_ag_ok(mesh, k_axis, arg_shapes[0].shape[1]):
        return (NamedSharding(mesh, P(m_axis, k_axis)),
                NamedSharding(mesh, P(m_axis, k_axis)),
                NamedSharding(mesh, P(k_axis, None)),
                NamedSharding(mesh, P(k_axis, None)))
    return _cast_dual_infer_sharding(mesh, arg_shapes, result_shape)


def _cast_wt_dual_partition_ag(mesh, arg_shapes, result_shape):
    x_spec = _get_spec(arg_shapes[0])
    m_axis = x_spec[0]
    k_axis = x_spec[1] if len(x_spec) > 1 else None
    if _kshard_packed_ag_ok(mesh, k_axis, arg_shapes[0].shape[1]):
        in_spec = P(m_axis, k_axis)
        row_spec = P(m_axis, k_axis)
        col_spec = P(k_axis, None)
        raw_fn = _cast_wt_dual_ksharded_raw
    else:
        in_spec = P(m_axis, None)
        row_spec = P(m_axis, None)
        col_spec = P(None, m_axis)
        raw_fn = _cast_wt_dual_raw

    def _lowered(x):
        return raw_fn(x)

    return (mesh, _lowered,
            (NamedSharding(mesh, row_spec), NamedSharding(mesh, row_spec),
             NamedSharding(mesh, col_spec), NamedSharding(mesh, col_spec)),
            (NamedSharding(mesh, in_spec),))


# --- Dual-cast Shardy sharding rules -----------------------------------------
# The colwise outputs are [K, M/2] (fp4) and [K_pad, M/32] (scale). For the
# ACTIVATION and GRAD dual casts the colwise M-dim is the FSDP-sharded TOKEN
# axis, so Shardy must propagate the input-M sharding to the colwise outputs --
# otherwise (the legacy "replicated colwise M" rule) Shardy treats each device's
# local colwise shard as the full replicated tensor and the FP4 wgrad
# (`db = grad_col @ act_col` over the token axis) reads mismatched data/scales,
# giving a ~2^20 gradient blow-up under shardy=True (the 1B base.yml default).
# To tie the colwise packed (M/2) and scale (M/32) M-dims to the input M,
# express them as compound factors of a shared base factor c = M/32 (so the M
# sharding follows c on all four outputs): sf=32 (M = c*32), hf=16 (M/2 = c*16).
#   NOTE: GSPMD (shardy=False) ignores sharding_rule and uses the partition
#   callback (byte-identical to before); this is a Shardy-propagation-only fix.
_DUAL_CAST_RULE_SHARDED_M = "(c sf) k -> (c sf) rkp, (c sf) rsp, k (c hf), k c"
# The WEIGHT dual cast keeps the legacy replicated-colwise rule: its colwise
# M-dim (= weight N_out) is the dgrad CONTRACTION axis, REPLICATED under FSDP
# (the dgrad GEMM forces it replicated), so dgrad stays unchanged / working.
_DUAL_CAST_RULE_REPL_M = "m k -> m rkp, m rsp, k cmp, k csp"

for _wrapped, _raw in ((_cast_act_dual, _cast_act_dual_raw),
                       (_cast_grad_dual, _cast_grad_dual_raw)):
    _wrapped.def_partition(
        _make_dual_cast_partition(_raw),
        infer_sharding_from_operands=_cast_dual_infer_sharding,
        sharding_rule=_DUAL_CAST_RULE_SHARDED_M,
        # Shardy requires special-factor indices in ascending first-appearance
        # order: c=0, sf=1, k=2, rkp=3, rsp=4, hf=5 (c is the sharded M base).
        need_replication_factors=("sf", "k", "rkp", "rsp", "hf"),
        sf=32, hf=16,
    )

if _PACK_GATEUP_AG:
    _cast_wt_dual.def_partition(
        _cast_wt_dual_partition_ag,
        infer_sharding_from_operands=_cast_wt_dual_infer_ag,
        sharding_rule=_DUAL_CAST_RULE_REPL_M,
        need_replication_factors=(),
    )
else:
    _cast_wt_dual.def_partition(
        _make_dual_cast_partition(_cast_wt_dual_raw),
        infer_sharding_from_operands=_cast_dual_infer_sharding,
        sharding_rule=_DUAL_CAST_RULE_REPL_M,
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


# ----- Lever B: K-sharded-b fwd GEMM with packed all-gather (PACK_GATEUP_AG) --
# For gate/up the rowwise weight B arrives K-sharded (axis1 = contraction) and
# UNSHUFFLED (from the Lever-B weight cast), while A (activation) has full K
# (sharded on M). The partition all-gathers the packed u8 B + its linear scales
# along the FSDP axis, re-assembles the ASM layout (shuffle_weight / e8m0_shuffle)
# and runs the GEMM -- replacing the legacy bf16 weight all-gather with a
# ~3.76x-smaller packed-FP4 all-gather. When B's K axis is NOT sharded (down,
# dgrad, attention, replicated) it defers to the legacy ``_fp4_partition``.

def _fp4_b_k_axis(mesh, arg_shapes):
    """Return b's K (axis1, contraction) mesh axis iff it is sharded AND the
    packed all-gather path is padding-aligned; else None (legacy)."""
    b_spec = _get_spec(arg_shapes[1])
    b_k = b_spec[1] if b_spec is not None and len(b_spec) > 1 else None
    # arg_shapes[1] is the PACKED weight [N, K/2]; K_global = (K/2)*2.
    k_global = arg_shapes[1].shape[1] * 2
    return b_k if _kshard_packed_ag_ok(mesh, b_k, k_global) else None


def _fp4_infer_ag(mesh, arg_shapes, result_shape):
    b_k = _fp4_b_k_axis(mesh, arg_shapes)
    if b_k is not None:
        a_spec = _get_spec(arg_shapes[0])
        b_spec = _get_spec(arg_shapes[1])
        a_m = a_spec[0]
        b_n = b_spec[0]
        if b_n == a_m:
            b_n = None
        return NamedSharding(mesh, P(a_m, b_n))
    return _fp4_infer_sharding(mesh, arg_shapes, result_shape)


def _fp4_partition_ag(mesh, arg_shapes, result_shape):
    a_spec = _get_spec(arg_shapes[0])
    b_spec = _get_spec(arg_shapes[1])
    b_k = _fp4_b_k_axis(mesh, arg_shapes)

    if b_k is None:
        # No K-sharded B operand -> legacy behavior (down / dgrad / replicated).
        return _fp4_partition(mesh, arg_shapes, result_shape)

    a_m = a_spec[0]
    b_n = b_spec[0]
    out_n = None if b_n == a_m else b_n
    ag_axis = b_k

    a_pspec = P(a_m, None)        # activation: M-sharded, full K
    b_pspec = P(b_n, b_k)         # weight: K-sharded (axis1), unshuffled
    out_pspec = P(a_m, out_n)

    def _lowered(a_packed, b_packed, a_scale, b_scale):
        # Gather the packed u8 weight + linear scales to full K, then re-assemble
        # the ASM (B-preshuffle) layout. Concatenation along K is exact for the
        # unshuffled cast (MXFP4 block=32 never crosses the K-shard boundary).
        b_full = jax.lax.all_gather(b_packed, ag_axis, axis=1, tiled=True)
        bs_full = jax.lax.all_gather(b_scale, ag_axis, axis=1, tiled=True)
        b_sh = shuffle_weight(b_full)
        bs_sh = e8m0_shuffle(bs_full)
        return _gemm_fp4_ffi(a_packed, b_sh, a_scale, bs_sh)

    return (mesh, _lowered,
            NamedSharding(mesh, out_pspec),
            (NamedSharding(mesh, a_pspec), NamedSharding(mesh, b_pspec),
             NamedSharding(mesh, a_pspec), NamedSharding(mesh, b_pspec)))


if _PACK_GATEUP_AG:
    _fp4_ffi_partitioned.def_partition(
        _fp4_partition_ag,
        infer_sharding_from_operands=_fp4_infer_ag,
        sharding_rule="m kp, n kp, m ks, n ks -> m n",
        need_replication_factors=(),
    )
else:
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


# Shardy rule: the M (token/batch) axis is the CONTRACTION and is FSDP-sharded,
# so it must be declared as a REDUCTION factor -- a sharded reduction factor
# tells Shardy the result is all-reduced over that axis (the psum the partition
# callback emits). The packed (M/2) and scale (M/32) M-dims are tied via a
# shared base factor mc = M/32 (hf=16 => M/2 = mc*16) so Shardy sees them as the
# same sharded contraction axis. Leaving M undeclared (the legacy rule) made
# Shardy mishandle the sharded contraction -> the FP4 wgrad blow-up. GSPMD
# (shardy=False) is unaffected (it uses the partition callback).
_fp4_ffi_partitioned_wgrad.def_partition(
    _fp4_wgrad_partition,
    infer_sharding_from_operands=_fp4_wgrad_infer_sharding,
    sharding_rule="n (mc hf), k (mc hf), n mc, k mc -> n k",
    reduction_factors=("mc", "hf"),
    hf=16,
)


# ---------------------------------------------------------------------------
# BF16 GEMM wgrad high-precision fallback (opt-in via JA_FP4_HIGHP_PASSES=wgrad).
#
#   grad_out [M, N]  (raw bf16 cotangent)
#   a        [M, K]  (raw bf16 activation, stashed by the fwd)
#   OUT = dB [N, K] = grad_out^T @ a   (contraction over M -- FSDP-sharded)
#
# bf16 twin of _fp4_ffi_partitioned_wgrad. The contraction axis M is the
# FSDP-sharded batch/token axis, so the per-shard partial products MUST be
# reduced across the FSDP mesh -- the role the FP4 wgrad's explicit
# jax.lax.psum plays. Here dB is declared REPLICATED (P(None, None), matching
# the FP4 wgrad's replicated output) via dot_general's ``out_sharding``, so
# XLA GSPMD inserts the all-reduce over the sharded contraction automatically:
# the equivalent, verified reduction (2-device check: cos 1.0, norm-ratio 1.0;
# a MISSING reduction would give ~0.5 norm-ratio). A plain unconstrained
# dot_general is rejected by JAX 0.9 sharding-in-types ("Contracting
# dimensions are sharded ... specify the output sharding via out_sharding").
# On a single device (no sharded axis) out_sharding is None => plain local
# dot. bf16 in, fp32 accumulate, bf16 out (matching the FP4 wgrad dB dtype so
# the custom_vjp cotangent types agree).
# ---------------------------------------------------------------------------

def _replicated_out_sharding(x):
    """If ``x`` carries a non-trivial mesh sharding (some dim mapped to a mesh
    axis), return a replicated NamedSharding on that mesh -- this forces GSPMD
    to all-reduce a sharded contraction. Else None (single-device / unsharded
    => plain local dot)."""
    sh = getattr(getattr(x, "aval", None), "sharding", None)
    mesh = getattr(sh, "mesh", None)
    spec = getattr(sh, "spec", None)
    if mesh is not None and spec is not None and any(s is not None for s in spec):
        return NamedSharding(mesh, P(None, None))
    return None


def _bf16_wgrad(grad_out, a):
    """dB = grad_out^T @ a, contracting the M/batch axis (axis 0 of both),
    reduced across the FSDP mesh via a replicated-output GSPMD all-reduce."""
    db = jax.lax.dot_general(
        grad_out, a,
        dimension_numbers=(((0,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
        out_sharding=_replicated_out_sharding(grad_out),
    )
    return db.astype(grad_out.dtype)


# ---------------------------------------------------------------------------
# BF16 GEMM dgrad high-precision fallback (opt-in via JA_FP4_HIGHP_PASSES=dgrad).
#
#   grad_out [M, N]  (raw bf16 cotangent)
#   b        [N, K]  (raw bf16 weight, stashed by the fwd)
#   OUT = dA [M, K] = grad_out @ b   (contraction over N -- the output-feature
#                                     axis, REPLICATED under FSDP)
#
# bf16 twin of the FP4 dgrad (_fp4_ffi_partitioned over go_row + col_b). Unlike
# the wgrad, whose M contraction is FSDP-sharded and needs an all-reduce, the
# dgrad contraction axis N is REPLICATED under FSDP (the FP4 dgrad declares the
# contraction in need_replication_factors, i.e. each device holds the full N),
# so NO collective is required: dA[M, K] inherits grad_out's M sharding and the
# weight's K, the SAME output sharding as the FP4 dgrad. GSPMD therefore
# propagates the output sharding from the operands and a plain dot_general is
# accepted (the "Contracting dimensions are sharded" sharding-in-types error
# only fires for a SHARDED contraction -- not this one), so no out_sharding
# hint is needed. bf16 in, fp32 accumulate, bf16 out (matching the FP4 dgrad dA
# dtype so the custom_vjp cotangent types agree).
# ---------------------------------------------------------------------------

def _bf16_dgrad(grad_out, b):
    """dA = grad_out @ b, contracting the N/output-feature axis (axis 1 of
    grad_out, axis 0 of the weight b). N is replicated under FSDP => plain
    local dot, no collective."""
    da = jax.lax.dot_general(
        grad_out, b,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    return da.astype(grad_out.dtype)


# ---------------------------------------------------------------------------
# FP8-precision GEMM dgrad (opt-in via JA_FP4_DGRAD_PREC=fp8).
#
#   grad_out [M, N]  (raw bf16 cotangent)
#   b        [N, K]  (raw bf16 weight, stashed by the fwd)
#   OUT = dA [M, K] = grad_out @ b   (contraction over N -- REPLICATED under FSDP)
#
# fp8 twin of _bf16_dgrad: cast BOTH operands to float8_e4m3fn (the OCP fp8
# format native to gfx950 / MI350) then dot_general. XLA-ROCm rewrites the
# convert(fp8) -> dot pattern into a REAL fp8 cublasLt/hipBLASLt matmul
# (__cublas$...$f8 in the HLO; ~1.45-1.8x faster than the bf16 dgrad dot on
# gfx950 -- verified by the Step-0 probe). Like the bf16 dgrad, the N
# contraction is replicated under FSDP => plain local matmul, NO collective and
# NO out_sharding hint (the casts are elementwise, so GSPMD propagates the same
# operand sharding as the bf16/FP4 dgrad). fp8 in, fp32 accumulate, bf16 out
# (matching the FP4 dgrad dA dtype so the custom_vjp cotangent types agree). dB
# (wgrad) stays FP4.
# ---------------------------------------------------------------------------

def _fp8_dgrad(grad_out, b):
    """dA = grad_out @ b in fp8 (e4m3), contracting the N/output-feature axis
    (axis 1 of grad_out, axis 0 of the weight b). N is replicated under FSDP =>
    plain local fp8 matmul, no collective."""
    da = jax.lax.dot_general(
        grad_out.astype(jnp.float8_e4m3fn),
        b.astype(jnp.float8_e4m3fn),
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    return da.astype(grad_out.dtype)


# ---------------------------------------------------------------------------
# FSDP/Shardy-correct FP4 dgrad via LOGICAL-weight gather (JA_FP4_DGRAD_PARTITION
# = gather). See the _DGRAD_GATHER flag block for the full rationale.
#
#   grad_out [M, N]  (raw bf16 cotangent, M/token-sharded under FSDP)
#   b        [N, K]  (raw bf16 weight,   N/out-feature-sharded under FSDP)
#   OUT = dA [M, K] = grad_out @ b   (contraction over N)
#
# The dgrad N contraction is FSDP-sharded, so each rank needs the FULL weight
# over N. Unlike the wgrad (whose sharded M contraction is SUMMED across ranks =>
# reduction_factors + psum), the dgrad output dA is PER-TOKEN (M-sharded, not
# summed), so the correct collective is an ALL-GATHER of the operand over N, NOT
# a reduce. This op declares the N factor need_replication (Shardy) / the weight
# operand replicated (GSPMD partition callback) so the partitioner all-gathers
# the LOGICAL bf16 weight (concat-safe along N). The colwise cast + B-preshuffle
# then runs LOCALLY on the full weight -- never on gathered shuffled shards.
# ---------------------------------------------------------------------------

def _dgrad_gather_local(grad_out, b):
    """Single-device FP4 dgrad from RAW bf16 grad_out + FULL bf16 weight.

    Reuses the SAME raw dual casts as the legacy backward -- the grad ROWWISE
    half of ``_cast_grad_dual_raw`` (dA A operand) and the weight COLUMNWISE half
    of ``_cast_wt_dual_raw`` (dA B operand) -- so on one device (or after the
    logical-weight gather) this is BYTE-IDENTICAL to the legacy
    ``_fp4_ffi_partitioned(go_row, col_b, go_row_s, col_b_scale)``.
    """
    go_row, go_row_s, _, _ = _cast_grad_dual_raw(grad_out)
    _, _, col_b, col_b_s = _cast_wt_dual_raw(b)
    return _gemm_fp4_ffi(go_row, col_b, go_row_s, col_b_s)


@custom_partitioning
def _fp4_dgrad_gather(grad_out, b):
    return _dgrad_gather_local(grad_out, b)


def _dgrad_gather_infer_sharding(mesh, arg_shapes, result_shape):
    g_spec = _get_spec(arg_shapes[0])          # grad_out [M, N]
    m_axis = g_spec[0]
    return NamedSharding(mesh, P(m_axis, None))  # dA [M, K]: M-sharded, K replicated


def _dgrad_gather_partition(mesh, arg_shapes, result_shape):
    g_spec = _get_spec(arg_shapes[0])
    m_axis = g_spec[0]
    g_pspec = P(m_axis, None)                   # grad_out: M-sharded, N replicated
    b_pspec = P(None, None)                     # weight: REPLICATED -> all-gather
                                                # the LOGICAL bf16 weight over N.
    out_pspec = P(m_axis, None)

    def _lowered(grad_out, b):
        return _dgrad_gather_local(grad_out, b)

    return (mesh, _lowered,
            NamedSharding(mesh, out_pspec),
            (NamedSharding(mesh, g_pspec), NamedSharding(mesh, b_pspec)))


# Shardy rule: the dgrad contracts N (out-features). Declaring N need_replication
# tells Shardy to GATHER (replicate) the operand along N rather than reduce over
# it -- the opposite of the wgrad's reduction_factors. dA[M,K] keeps grad_out's M
# sharding; K follows the (replicated-under-FSDP) weight K.
_fp4_dgrad_gather.def_partition(
    _dgrad_gather_partition,
    infer_sharding_from_operands=_dgrad_gather_infer_sharding,
    sharding_rule="m n, n k -> m k",
    need_replication_factors=("n",),
)


# ---------------------------------------------------------------------------
# Offline-capture hook (env-gated via JA_CAPTURE_DIR; OFF by default).
#
# Purpose: dump the real bf16 (a, b, grad_out) per GEMM site so an offline
# replay can localize which FP4 backward pass (fprop / dgrad / wgrad) damages
# the training signal. Lives here because it must observe the *exact* tensors
# the custom_vjp fwd / bwd receive.
#
# Production no-op guarantee: _capture_dir() is read at trace time. When unset
# the helpers `return` immediately, so NO slice / callback node is emitted and
# numerics are byte-identical. Fully reversible -- delete this block plus the
# two `_capture_*` calls in _gemm_fp4_bf16_fwd / _gemm_fp4_bf16_bwd.
#
# Bounding: only a fixed leading slice (_CAPTURE_ROW_CAP rows along the token /
# M axis, and along N for the weight) is transferred + saved, keyed by the
# ORIGINAL (M, N, K) shape. A first-wins O_EXCL write means whichever scan
# iteration / device fires first (past the JA_CAPTURE_AFTER_STEP threshold)
# for a shape captures it; the rest skip. So the capture is a single
# representative per-shard slice regardless of FSDP degree.
#
# Steady-state capture (JA_CAPTURE_AFTER_STEP): the GEMM call site cannot see
# the training step, so we keep a HOST-side per-(role, shape) GEMM-call (fire)
# counter -- bumped each time the runtime callback fires -- and only write once
# a site's counter reaches the threshold. Default threshold 0 => write on the
# first fire (step-0 behavior, backward-compatible). This lets an offline probe
# compare step-0 vs steady-state under-flush from the same hook.
# ---------------------------------------------------------------------------

_CAPTURE_ROW_CAP = 4096

# Host-side per-(role, shape) runtime fire counter for JA_CAPTURE_AFTER_STEP.
_CAPTURE_CALL_COUNTS = {}


def _capture_dir():
    return os.environ.get("JA_CAPTURE_DIR")


def _capture_after_step():
    """Per-shape GEMM-call (fire) threshold; capture once a site's counter
    reaches it. 0 (default / unparsable) => capture the first fire."""
    v = os.environ.get("JA_CAPTURE_AFTER_STEP")
    try:
        return int(v) if v else 0
    except ValueError:
        return 0


def _capture_count_ready(counter_key):
    """Host-side: bump the per-(role, shape) fire counter and report whether
    it has reached the JA_CAPTURE_AFTER_STEP threshold. Returns (ready, count)."""
    n = _CAPTURE_CALL_COUNTS.get(counter_key, 0) + 1
    _CAPTURE_CALL_COUNTS[counter_key] = n
    return n >= _capture_after_step(), n


def _capture_write(path, arr):
    """First-wins host writer: O_EXCL means exactly one fire per shape wins.
    Returns True if this call wrote the file, False if it already existed."""
    import errno
    import numpy as np
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            return False  # already captured this site -- first-wins, skip.
        raise
    try:
        with os.fdopen(fd, "wb") as fh:
            np.save(fh, np.asarray(arr, dtype=np.float32))
        return True
    except Exception:  # never let capture I/O perturb the run.
        try:
            os.unlink(path)
        except OSError:
            pass
        return False


def _capture_fwd(a, b):
    """Dump bf16 (a, b) for this (M, N, K) site once its fire counter crosses
    the JA_CAPTURE_AFTER_STEP threshold (default: first fire)."""
    cap_dir = _capture_dir()
    if not cap_dir:
        return
    M, K = a.shape
    N = b.shape[0]
    key = "M%d_N%d_K%d" % (M, N, K)
    cap = _CAPTURE_ROW_CAP
    a_s, b_s = a[:cap], b[:cap]

    def _w(a_h, b_h):
        ready, n = _capture_count_ready("fwd_" + key)
        if not ready:
            return
        os.makedirs(cap_dir, exist_ok=True)
        wrote = _capture_write(os.path.join(cap_dir, "site_%s__a.npy" % key), a_h)
        _capture_write(os.path.join(cap_dir, "site_%s__b.npy" % key), b_h)
        if wrote:
            print("[ja-capture] fwd site %s (a,b) captured at fire #%d" % (key, n),
                  flush=True)

    jax.debug.callback(_w, a_s, b_s)


def _capture_bwd(grad_out, k_dim):
    """Dump bf16 grad_out for this (M, N, K) site once its fire counter crosses
    the JA_CAPTURE_AFTER_STEP threshold (default: first fire)."""
    cap_dir = _capture_dir()
    if not cap_dir:
        return
    M, N = grad_out.shape
    key = "M%d_N%d_K%d" % (M, N, k_dim)
    cap = _CAPTURE_ROW_CAP
    g_s = grad_out[:cap, :cap]

    def _w(g_h):
        ready, n = _capture_count_ready("bwd_" + key)
        if not ready:
            return
        os.makedirs(cap_dir, exist_ok=True)
        wrote = _capture_write(os.path.join(cap_dir, "site_%s__grad_out.npy" % key), g_h)
        if wrote:
            print("[ja-capture] bwd site %s grad_out captured at fire #%d" % (key, n),
                  flush=True)

    jax.debug.callback(_w, g_s)


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
    """Forward-with-residuals: dual-cast keeps columnwise FP4 for backward.

    When ``JA_FP4_HIGHP_PASSES`` selects ``wgrad`` / ``dgrad`` the RAW bf16
    activation ``a`` / weight ``b`` is also stashed so the backward can run
    that GEMM in bf16 instead of FP4. The extra residuals are appended in a
    FIXED order -- ``a`` for wgrad, then ``b`` for dgrad -- so the backward
    recovers them from the SAME module flags. Default (both unset) keeps the
    original 4-tuple residual => byte-identical all-FP4 backward.
    """
    _capture_fwd(a, b)  # env-gated offline-probe hook (no-op unless JA_CAPTURE_DIR set).
    a_row, a_row_s, col_a_fp4, col_a_scale = _cast_act_dual(a)
    b_row, b_row_s, col_b_fp4, col_b_scale = _cast_wt_dual(b)
    out = _fp4_ffi_partitioned(a_row, b_row, a_row_s, b_row_s)
    # Optionally tag the columnwise FP4 residuals so a MaxText remat policy can
    # SAVE them instead of recomputing the dual-cast in backward (graph-
    # scheduling only; checkpoint_name is identity => byte-identical numerics).
    # BOTH the packed FP4 and its E8M0 scale must be tagged, else the cast
    # kernel still re-fires to recompute the untagged half. Default off
    # (_REMAT_SAVE_* both False) => no name_p added => unchanged graph.
    if _REMAT_SAVE_ACT_COL:
        col_a_fp4 = checkpoint_name(col_a_fp4, _MXFP4_ACT_COL_NAME)
        col_a_scale = checkpoint_name(col_a_scale, _MXFP4_ACT_COL_NAME)
    if _REMAT_SAVE_WT_COL:
        col_b_fp4 = checkpoint_name(col_b_fp4, _MXFP4_WT_COL_NAME)
        col_b_scale = checkpoint_name(col_b_scale, _MXFP4_WT_COL_NAME)
    residual = (col_a_fp4, col_a_scale, col_b_fp4, col_b_scale)
    if _HIGHP_WGRAD:
        residual = residual + (a,)  # RAW bf16 activation for the bf16 wgrad dB.
    if _HIGHP_DGRAD or _FP8_DGRAD or _DGRAD_SPLIT or _DGRAD_GATHER:
        residual = residual + (b,)  # RAW bf16 weight for the bf16/fp8/split/gather dgrad dA.
    return out, residual


def _gemm_fp4_bf16_bwd(residuals, grad_out):
    """Backward: FP4 (default) or bf16 dA / dB per ``JA_FP4_HIGHP_PASSES``.

    dA is the input-gradient (dgrad) GEMM ``grad_out @ B``, contracting the
    output-feature axis N (REPLICATED under FSDP). dB is the weight-gradient
    (wgrad) GEMM ``grad_out^T @ A``, contracting the FSDP-sharded M/batch axis.
    When ``dgrad`` / ``wgrad`` is selected for high precision, that GEMM becomes
    a bf16 dot of the RAW ``grad_out`` and the RAW bf16 weight / activation
    (stashed in fwd) -- see ``_bf16_dgrad`` / ``_bf16_wgrad``. Selecting BOTH
    => the whole backward is bf16 while the forward stays FP4. Alternatively
    ``JA_FP4_DGRAD_PREC=fp8`` runs ONLY the dgrad in fp8 (e4m3) via
    ``_fp8_dgrad`` (candidate mixed FP4-fwd / FP8-dgrad recipe), leaving wgrad
    FP4. Whatever stays FP4 still consumes the Hadamard-transformed grad
    dual-cast.
    """
    col_a_fp4, col_a_scale, col_b_fp4, col_b_scale = residuals[:4]
    idx = 4
    a_bf16 = b_bf16 = None
    _dgrad_highp = _HIGHP_DGRAD or _FP8_DGRAD  # dgrad runs bf16 OR fp8 (not FP4)
    # split + gather also need the RAW bf16 weight (gather casts it locally post-AG).
    _dgrad_needs_raw_b = _dgrad_highp or bool(_DGRAD_SPLIT) or _DGRAD_GATHER
    if _HIGHP_WGRAD:
        a_bf16 = residuals[idx]
        idx += 1
    if _dgrad_needs_raw_b:
        b_bf16 = residuals[idx]
        idx += 1

    # env-gated offline-probe hook (no-op unless JA_CAPTURE_DIR set). K is the
    # contraction dim, recovered from a saved columnwise residual ([K, .../2]).
    _capture_bwd(grad_out, col_b_fp4.shape[0])

    # Hadamard ON: applied inside the fused cast kernel for grad only. Skip the
    # cast only when NEITHER backward GEMM is FP4 (no FP4 grad operand is
    # consumed) so the high-precision throughput cost stays honest. dgrad is
    # high-precision when bf16 OR fp8.
    go_row = go_row_s = go_col_fp4 = go_col_scale = None
    if (not _dgrad_highp) or (not _HIGHP_WGRAD):
        go_row, go_row_s, go_col_fp4, go_col_scale = _cast_grad_dual(grad_out)

    # dA = grad_out @ B -- NN layout, contraction over N (replicated under FSDP).
    if _DGRAD_SPLIT:
        # Diagnostic per-operand split: fake-quant the selected operand(s) to
        # MXFP4 (Hadamard OFF) and bf16 dot. Contraction axis N (replicated =>
        # no collective, same as _bf16_dgrad). grad_out [M,N] axis1=N; b [N,K] axis0=N.
        gq = _fake_quant_mxfp4(grad_out, 1) if _DGRAD_SPLIT in ("dy", "both") else grad_out
        wq = _fake_quant_mxfp4(b_bf16, 0) if _DGRAD_SPLIT in ("w", "both") else b_bf16
        da = jax.lax.dot_general(
            gq, wq, dimension_numbers=(((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32).astype(grad_out.dtype)
    elif _FP8_DGRAD:
        # FP8-precision dA from RAW grad_out + RAW weight cast to e4m3 (real fp8
        # cublasLt matmul; no FP4 quant, no Hadamard). N replicated => no
        # collective. dB (wgrad) stays FP4 -- candidate mixed FP4-fwd/FP8-dgrad.
        da = _fp8_dgrad(grad_out, b_bf16)
    elif _HIGHP_DGRAD:
        # High-precision bf16 dA from RAW grad_out + RAW weight (no FP4 quant,
        # no Hadamard). N is replicated => plain local dot, no collective.
        da = _bf16_dgrad(grad_out, b_bf16)
    elif _DGRAD_GATHER:
        # FSDP/Shardy-correct FP4 dgrad: gather the LOGICAL bf16 weight over N
        # (concat-safe), THEN colwise cast/pack/shuffle LOCALLY, THEN FP4 GEMM.
        # Fixes the corrupt-dA-under-FSDP bug (legacy gathers the SHUFFLED weight).
        da = _fp4_dgrad_gather(grad_out, b_bf16)
    else:
        da = _fp4_ffi_partitioned(go_row, col_b_fp4, go_row_s, col_b_scale)
    # dB = grad_out^T @ A -- NT layout, M_batch is the contraction (FSDP-
    # sharded). The partition callback emits psum across the FSDP mesh.
    if _HIGHP_WGRAD:
        # High-precision bf16 dB from RAW grad_out + RAW activation (no FP4
        # quant, no Hadamard); same net FSDP reduction as the FP4 wgrad.
        db = _bf16_wgrad(grad_out, a_bf16)
    else:
        db = _fp4_ffi_partitioned_wgrad(go_col_fp4, col_a_fp4, go_col_scale, col_a_scale)
    return da, db


gemm_fp4_bf16.defvjp(_gemm_fp4_bf16_fwd, _gemm_fp4_bf16_bwd)
