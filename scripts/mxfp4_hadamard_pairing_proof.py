#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only proof: does the MXFP4 grad-only Hadamard bias the backward GEMMs?

Read-only w.r.t. production code. numpy on host, NO GPU, quantization DISABLED
(pure orthonormal rotation -- no FP4 rounding) so any deviation from the
no-Hadamard reference is the *rotation* alone, not quantization noise.

What this models (faithful to the live recipe)
----------------------------------------------
jax_aiter/gemm_fp4/gemm_fp4.py  +  csrc/ffi/cast_mxfp4/
cast_transpose_mxfp4_kernel_shuffled.cu  (function `hadamard16_inplace`).

GEMMs (NT/NN layout from the custom_vjp):
  fprop : out = A @ B^T              contraction over K
  dgrad : dA  = grad_out @ B         contraction over N  (= grad_row @ wt_col^T)
  wgrad : dB  = grad_out^T @ A       contraction over M  (= grad_col @ act_col^T)

Cast (fused HIP kernel) applies a 16-point block-diagonal Hadamard
(orthonormal, two independent H16 per 32-element quant block) in BOTH the
rowwise and columnwise phases when USE_HADAMARD is set:
  * rowwise output rotates along the input's LAST axis (d1).
  * columnwise output is the transpose; it rotates along the input's FIRST
    axis (d0).

Live default: grad cast has Hadamard ON; act/weight casts have Hadamard OFF
(AITER_FUSED_QUANT_HADAMARD=0). AITER_FUSED_QUANT_HADAMARD=1 turns Hadamard
ON for act/weight/grad ("all-on").

Recipes compared (vs the no-Hadamard reference):
  a no-H          : reference.
  b paired        : H on BOTH contraction partners of every GEMM.
  c one-operand   : H on exactly ONE contraction partner per GEMM.
  d grad-only     : the LIVE default -- only the grad cast carries H.
  e all-on        : AITER_FUSED_QUANT_HADAMARD=1 -- act/weight/grad all H.

Metrics: cosine(out_variant, out_ref) and relative-L2 ||v-r||/||r||.
"""

from __future__ import annotations

import argparse
import numpy as np


# ---------------------------------------------------------------------------
# 16-point Hadamard, reconstructed EXACTLY from hadamard16_inplace (the .cu).
# 16 elements are laid out as 4 threads x 4 values; element e = thread*4 + value.
# ---------------------------------------------------------------------------

def _hadamard16_apply_vec(x16: np.ndarray) -> np.ndarray:
    """Mirror of `hadamard16_inplace` acting on one length-16 vector."""
    v = x16.reshape(4, 4).astype(np.float64).copy()  # v[thread, value]

    # Stage 1: local 4-point Hadamard on each thread's 4 values.
    #   H4 = [[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]]
    a0 = v[:, 0] + v[:, 1]
    a1 = v[:, 0] - v[:, 1]
    a2 = v[:, 2] + v[:, 3]
    a3 = v[:, 2] - v[:, 3]
    out = np.empty_like(v)
    out[:, 0] = a0 + a2
    out[:, 2] = a0 - a2
    out[:, 1] = a1 + a3
    out[:, 3] = a1 - a3
    v = out

    # Stage 2: XOR-1 cross-thread butterfly (partner = thread ^ 1).
    #   sign2 = thread & 1 -> odd threads subtract.
    p = v[[1, 0, 3, 2], :]
    sign2 = np.array([0, 1, 0, 1]).reshape(4, 1)
    v = np.where(sign2 == 1, p - v, p + v)

    # Stage 3: XOR-2 cross-thread butterfly (partner = thread ^ 2).
    #   sign3 = (thread >> 1) & 1 -> threads 2,3 subtract.
    p = v[[2, 3, 0, 1], :]
    sign3 = np.array([0, 0, 1, 1]).reshape(4, 1)
    v = np.where(sign3 == 1, p - v, p + v)

    # Normalize by 1/sqrt(16) = 0.25.
    v = v * 0.25
    return v.reshape(16)


def build_h16() -> np.ndarray:
    """Materialize the 16x16 normalized Hadamard matrix (T(x) = H @ x)."""
    H = np.zeros((16, 16), dtype=np.float64)
    for e in range(16):
        basis = np.zeros(16, dtype=np.float64)
        basis[e] = 1.0
        H[:, e] = _hadamard16_apply_vec(basis)
    return H


def apply_block_hadamard(x: np.ndarray, axis: int, H16: np.ndarray) -> np.ndarray:
    """Apply H16 block-diagonally to consecutive 16-element blocks along `axis`.

    A 32-element quant block carries two independent H16 (block-diagonal-16),
    which is identical to applying H16 to every consecutive 16-block.
    """
    x = np.moveaxis(x, axis, -1)
    shp = x.shape
    L = shp[-1]
    assert L % 16 == 0, f"axis length {L} must be a multiple of 16 (32-block /2)"
    xr = x.reshape(shp[:-1] + (L // 16, 16))
    yr = xr @ H16.T            # y[..., i] = sum_j H16[i, j] x[..., j]
    y = yr.reshape(shp)
    return np.moveaxis(y, -1, axis)


# ---------------------------------------------------------------------------
# Faithful dual-cast: (rowwise, columnwise) outputs with optional Hadamard.
# ---------------------------------------------------------------------------

def cast_dual(x: np.ndarray, use_hadamard: bool, H16: np.ndarray):
    """Return (row, col) FP4-equivalent operands (quantization disabled).

    row : x rotated along its LAST axis (d1).           shape [d0, d1]
    col : (x rotated along its FIRST axis (d0)) ^T.      shape [d1, d0]
    """
    if use_hadamard:
        row = apply_block_hadamard(x, axis=1, H16=H16)
        col = apply_block_hadamard(x, axis=0, H16=H16).T
    else:
        row = x.copy()
        col = x.T.copy()
    return row, col


def gemms(A, B, G, H16, *, fprop_aH, fprop_bH,
          dgrad_gH, dgrad_bH, wgrad_gH, wgrad_aH):
    """Compute (fprop, dgrad, wgrad) for an explicit per-operand H choice.

    fprop : A_row @ B_row^T            (rotate A,B along K)
    dgrad : grad_row @ wt_col^T        (rotate grad along N, weight-col along N)
    wgrad : grad_col @ act_col^T       (rotate grad-col along M, act-col along M)
    """
    a_row = apply_block_hadamard(A, 1, H16) if fprop_aH else A
    b_row = apply_block_hadamard(B, 1, H16) if fprop_bH else B
    fprop = a_row @ b_row.T

    g_row = apply_block_hadamard(G, 1, H16) if dgrad_gH else G
    b_col = (apply_block_hadamard(B, 0, H16) if dgrad_bH else B).T
    dgrad = g_row @ b_col.T

    g_col = (apply_block_hadamard(G, 0, H16) if wgrad_gH else G).T
    a_col = (apply_block_hadamard(A, 0, H16) if wgrad_aH else A).T
    wgrad = g_col @ a_col.T
    return fprop, dgrad, wgrad


def gemms_via_cast_flags(A, B, G, H16, *, act_H, wt_H, grad_H):
    """Route through the REAL per-tensor cast flags (production wiring).

    Mirrors gemm_fp4.py exactly:
      a_row,_=cast(act); _,b_col=cast(wt); g_row,g_col=cast(grad); ...
    Used to prove AITER_FUSED_QUANT_HADAMARD=1 realizes correct pairing.
    """
    a_row, a_col = cast_dual(A, act_H, H16)
    b_row, b_col = cast_dual(B, wt_H, H16)
    g_row, g_col = cast_dual(G, grad_H, H16)
    fprop = a_row @ b_row.T
    dgrad = g_row @ b_col.T
    wgrad = g_col @ a_col.T
    return fprop, dgrad, wgrad


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------

def cosine(v: np.ndarray, r: np.ndarray) -> float:
    v = v.ravel().astype(np.float64)
    r = r.ravel().astype(np.float64)
    nv = np.linalg.norm(v)
    nr = np.linalg.norm(r)
    if nv == 0 or nr == 0:
        return float("nan")
    return float(np.dot(v, r) / (nv * nr))


def rel_l2(v: np.ndarray, r: np.ndarray) -> float:
    r = r.ravel().astype(np.float64)
    nr = np.linalg.norm(r)
    if nr == 0:
        return float("nan")
    return float(np.linalg.norm(v.ravel().astype(np.float64) - r) / nr)


RECIPES = {
    "a no-H        ": dict(fprop_aH=False, fprop_bH=False,
                           dgrad_gH=False, dgrad_bH=False,
                           wgrad_gH=False, wgrad_aH=False),
    "b paired      ": dict(fprop_aH=True,  fprop_bH=True,
                           dgrad_gH=True,  dgrad_bH=True,
                           wgrad_gH=True,  wgrad_aH=True),
    "c one-operand ": dict(fprop_aH=True,  fprop_bH=False,
                           dgrad_gH=True,  dgrad_bH=False,
                           wgrad_gH=True,  wgrad_aH=False),
    "d grad-only   ": dict(fprop_aH=False, fprop_bH=False,
                           dgrad_gH=True,  dgrad_bH=False,
                           wgrad_gH=True,  wgrad_aH=False),
    "e all-on      ": dict(fprop_aH=True,  fprop_bH=True,
                           dgrad_gH=True,  dgrad_bH=True,
                           wgrad_gH=True,  wgrad_aH=True),
}


def run_shape(name, M, K, N, seed, heavy_tail_grad=False):
    print(f"\n{'='*78}\nSHAPE {name}: M={M} K={K} N={N}  (contraction: "
          f"fprop=K, dgrad=N, wgrad=M)\n{'='*78}")
    for d, tag in ((K, "K"), (N, "N"), (M, "M")):
        assert d % 32 == 0, f"{tag}={d} must be a multiple of 32"

    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M, K)).astype(np.float64)        # activation
    B = rng.standard_normal((N, K)).astype(np.float64)        # weight
    if heavy_tail_grad:
        # gradient with outliers (Student-t-ish) -- the rotation's motivation.
        G = (rng.standard_normal((M, N)) *
             rng.standard_gamma(0.5, (M, N))).astype(np.float64)
    else:
        G = rng.standard_normal((M, N)).astype(np.float64)

    H16 = build_h16()

    ref = gemms(A, B, G, H16, **RECIPES["a no-H        "])
    ref_f, ref_d, ref_w = ref

    rows = {}
    for label, flags in RECIPES.items():
        f, dg, wg = gemms(A, B, G, H16, **flags)
        rows[label] = (
            (cosine(f, ref_f), rel_l2(f, ref_f)),
            (cosine(dg, ref_d), rel_l2(dg, ref_d)),
            (cosine(wg, ref_w), rel_l2(wg, ref_w)),
        )

    # Cross-check: the REAL per-tensor flag wiring for all-on must equal the
    # idealized "paired" recipe (proves the flag realizes correct pairing).
    e_flag = gemms_via_cast_flags(A, B, G, H16, act_H=True, wt_H=True, grad_H=True)
    d_flag = gemms_via_cast_flags(A, B, G, H16, act_H=False, wt_H=False, grad_H=True)
    paired = gemms(A, B, G, H16, **RECIPES["b paired      "])
    gradonly = gemms(A, B, G, H16, **RECIPES["d grad-only   "])
    max_e = max(float(np.max(np.abs(x - y))) for x, y in zip(e_flag, paired))
    max_d = max(float(np.max(np.abs(x - y))) for x, y in zip(d_flag, gradonly))

    hdr = f"{'recipe':14s} | {'fprop cos':>11s} {'relL2':>10s} | " \
          f"{'dgrad cos':>11s} {'relL2':>10s} | {'wgrad cos':>11s} {'relL2':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for label in RECIPES:
        (cf, rf), (cd, rd), (cw, rw) = rows[label]
        print(f"{label:14s} | {cf:11.6f} {rf:10.3e} | "
              f"{cd:11.6f} {rd:10.3e} | {cw:11.6f} {rw:10.3e}")

    print(f"\n[cross-check] all-on via real cast flags  == idealized paired : "
          f"max|Δ|={max_e:.3e}")
    print(f"[cross-check] grad-only via real cast flags == idealized grad-only: "
          f"max|Δ|={max_d:.3e}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20260602)
    ap.add_argument("--prod-M", type=int, default=128,
                    help="batch*seq for the prod-ish 8B mlp_gate shape (subsampled)")
    ap.add_argument("--skip-prod", action="store_true")
    args = ap.parse_args()

    np.set_printoptions(precision=4, suppress=True, linewidth=140)

    # --- Orthonormality of the reconstructed 16-point Hadamard ---
    H16 = build_h16()
    ortho_err = float(np.max(np.abs(H16 @ H16.T - np.eye(16))))
    det = float(np.linalg.det(H16))
    print(f"{'='*78}\n16-POINT HADAMARD (reconstructed from hadamard16_inplace)\n{'='*78}")
    print("H16 (normalized by 0.25):")
    print(H16)
    print(f"\nmax|H @ H^T - I| = {ortho_err:.3e}   (orthonormal if ~0)")
    print(f"det(H16)         = {det:+.6f}   (|det|=1 for orthonormal)")
    assert ortho_err < 1e-12, "H16 is NOT orthonormal -- reconstruction bug"

    # --- Small MLP-like shape (all dims multiple of 32) ---
    run_shape("small", M=256, K=128, N=512, seed=args.seed)

    # --- One real production-ish shape: 8B mlp_gate K=4096, N=14336 ---
    if not args.skip_prod:
        run_shape("8B_mlp_gate", M=args.prod_M, K=4096, N=14336, seed=args.seed + 1)

    # --- Heavy-tailed gradient (rotation's stated motivation) -- algebraic
    #     identity is distribution-independent, shown here for completeness ---
    run_shape("small_heavytail_grad", M=256, K=128, N=512,
              seed=args.seed + 2, heavy_tail_grad=True)

    print(f"\n{'='*78}\nINTERPRETATION\n{'='*78}")
    print("cos=1.000000 & relL2~0  -> rotation cancels (output == no-H reference).")
    print("cos<1 / relL2 nonzero    -> rotation is UNPAIRED -> biased output.")
    print("fprop column for 'd grad-only' is the reference (grad not in fprop).")


if __name__ == "__main__":
    main()
