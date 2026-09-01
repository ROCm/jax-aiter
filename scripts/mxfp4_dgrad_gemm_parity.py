#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Is the AITER MXFP4 f4gemm faithful for the dgrad (dA) GEMM, or does the
force-selected kernel / splitK degrade dA *beyond* the FP4 representation?

This closes the last unverified assumption behind the "E2M1 precision wall"
conclusion. The cast is proven bit-exact (docs/perf/mxfp4_dpp_cast_parity);
bf16/fp8 dgrad bypass the f4gemm (XLA dot_general), so the f4gemm's
faithfulness for the dA shape was never directly checked.

The dgrad call (jax_aiter/gemm_fp4/gemm_fp4.py `_gemm_fp4_bf16_bwd`):
    da = _fp4_dgrad_kgather(go_row, col_b_unshuf, go_row_s, col_b_scale_lin)
    -> shuffle_weight / e8m0_shuffle after the packed all-gather
    -> GemmFp4FwdJA -> aiter::f4gemm
    Out[M, P] = go_row[M, C] @ col_b[P, C]^T   (FFI M=tokens, N=P=K_orig,
    K=C=N_orig is the contraction = the projection's output-feature dim).

Discriminating test (per dA shape):
  (a) AITER f4gemm  = gemm_fp4(a_packed, col_b_fp4, a_scale, col_b_scale).
  (b) IDEAL-FP4 ref = fp32 matmul of the *dequantised exact FP4 operands*
      (a_deq @ b_deq^T). This is the best-possible GEMM for these operands.
  Plus the bf16 "no-FP4" product of the *pre-quant* operands (the wall).

  AITER ~= IDEAL-FP4  -> the GEMM is FAITHFUL; the dgrad loss is the FP4
    representation -> precision wall confirmed at the GEMM level. Stop.
  AITER WORSE than IDEAL-FP4 -> kernel (accumulation / tile tails / splitK /
    variant) degrades dA beyond FP4 -> fixable; sweep for a better variant.

Step 3 sweep re-runs the dA GEMM under force-OFF (heuristic) + other cataloged
variants / splitK (via AITER_FORCE_KERNEL_NAME / AITER_FORCE_LOG2_K_SPLIT, read
per-call in the FFI) and compares each to IDEAL-FP4.

Read-only w.r.t. the kernel source. Uses the built gemm_fp4_ja.so + cast
kernel via JA_ROOT_DIR and the source jax_aiter via PYTHONPATH (pre-check
asserts both). No rebuild. RNE path (use_sr unset). Variant selection is env
only -- no code edits. Reuses the proven cast helpers from
scripts/mxfp4_dpp_cast_parity.py.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime

import numpy as np

# Proven E2M1 / dequant / unshuffle / host-cast helpers (read-only reuse).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mxfp4_dpp_cast_parity as cp  # noqa: E402

FORCE_NAME_ENV = "AITER_FORCE_KERNEL_NAME"
FORCE_SPLIT_ENV = "AITER_FORCE_LOG2_K_SPLIT"

# The production-forced kernel (scripts/run_fresh_maxtext_e2e.sh set_aiter_env).
FORCED_KNL = "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E"


# ---------------------------------------------------------------------------
# Heuristic replication (mirror of gemm_fp4_ja.cu::select_fp4_kernel).
# K is NOT used by the heuristic; selection depends only on (M, N, tile, splitK).
# ---------------------------------------------------------------------------
def load_cfgs(csv_path):
    cfgs = []
    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            cfgs.append((int(row["tile_M"]), int(row["tile_N"]),
                         int(row["splitK"]), int(row["bpreshuffle"]),
                         row["knl_name"]))
    return cfgs


def select_fp4_kernel(M, N, num_cu, cfgs):
    empty_cu = num_cu
    best_round = 0xffffffff
    c2m = 1.0
    sel_knl, sel_split = "", 0
    for tM, tN, splitK_cap, bps, knl in cfgs:
        if bps != 1:
            continue
        if not (tM != 128 or tN != 512 or (N % tN) == 0):
            continue
        split_list = [1, 2, 4, 8, 16] if splitK_cap else [1]
        for sK in split_list:
            tg_M = (M + tM - 1) // tM
            tg_N = (N + tN - 1) // tN
            tg = tg_M * tg_N * sK
            local_round = (tg + num_cu - 1) // num_cu
            local_c2m = (tM * tN) / (tM + tN)
            is_earlier = local_round < best_round
            is_same = local_round == best_round
            fewer_empty = empty_cu > (local_round * num_cu - tg)
            better = local_c2m > c2m
            if is_earlier or (is_same and (fewer_empty or better)):
                best_round = local_round
                empty_cu = local_round * num_cu - tg
                c2m = local_c2m
                sel_knl = knl
                log2, tmp = 0, sK
                while tmp > 1:
                    tmp >>= 1
                    log2 += 1
                sel_split = log2
    return sel_knl, sel_split


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics(x, ref):
    """cos, rel-L2, max-abs of x vs ref (both [..]-> flat float64)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    r = np.asarray(ref, dtype=np.float64).ravel()
    nx, nr = np.linalg.norm(x), np.linalg.norm(r)
    cos = float(np.dot(x, r) / (nx * nr + 1e-30))
    rel_l2 = float(np.linalg.norm(x - r) / (nr + 1e-30))
    max_abs = float(np.max(np.abs(x - r)))
    return {"cos": cos, "rel_l2": rel_l2, "max_abs": max_abs}


# ---------------------------------------------------------------------------
# Numerical-neutrality mode (kernel-selection study, todo 1).
# Role-COMPLETE by MNK-equivalence: every fprop/dgrad/wgrad role of the 8B
# full-scope model maps onto one of the manifest's distinct (M,N,K) cells, so
# sweeping these cells covers fwd + dgrad + wgrad. For each cell we build ONE
# set of pre-quantised FP4 operands (same path the microbench uses:
# bf16_to_mxfp4 + shuffle_weight + e8m0_shuffle) and run every candidate
# (kernel, splitK) variant, comparing each output to the production reference
# (256x256/sK1). Kernel choice is numerically NEUTRAL iff every variant agrees
# with the reference to within fp32-accumulate reorder + bf16-output rounding
# (cos ~ 1, rel_l2 at bf16-ULP scale). A variant computing different math would
# show cos << 1 -> STOP.
# ---------------------------------------------------------------------------
# 8B full-scope role -> (M,N,K) cells (FFI convention Out[M,N]=A[M,K]@B[N,K]).
NEUTRALITY_CELLS_8B = [
    ("attn_qo_fwd/dgrad", 32768, 4096, 4096),
    ("attn_kv_fwd",       32768, 1024, 4096),
    ("attn_kv_dgrad",     32768, 4096, 1024),
    ("attn_qo_wgrad",      4096, 4096, 32768),
    ("attn_kv_wgrad",      1024, 4096, 32768),
    ("mlp_gateup_fwd/down_dgrad", 32768, 14336, 4096),
    ("mlp_gateup_dgrad/down_fwd", 32768, 4096, 14336),
    ("mlp_gateup_wgrad",  14336, 4096, 32768),
    ("mlp_down_wgrad",     4096, 14336, 32768),
]

REF_KNL = "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E"
REF_SPLIT = 0  # log2_split=0 => splitK=1


def run_neutrality(jnp, gemm_fp4, sweep, cells, seed, out_path):
    """Run every (kernel,splitK) variant per cell; compare to 256x256/sK1.

    Returns the results list and a boolean ``all_neutral``.
    """
    import jax
    from jax_aiter.gemm_fp4.fp4_utils import (
        bf16_to_mxfp4, e8m0_shuffle, shuffle_weight)

    # Neutral threshold: bf16 has ~8 mantissa bits => 1 ULP ~ 3.9e-3 relative.
    # fp32-accumulate reorder + bf16 output rounding keeps rel_l2 well under
    # this whole-matrix; we flag anything with cos<0.999 or rel_l2>0.02.
    COS_MIN = 0.999
    REL_L2_MAX = 0.02

    print(f"\n{'='*78}\n=== NUMERICAL-NEUTRALITY SWEEP (kernel choice = loss-free?) ===")
    print(f"reference variant = 256x256/sK1 (production FORCE pin)")
    records = []
    all_neutral = True
    for (label, M, N, K) in cells:
        key = jax.random.PRNGKey(seed)
        k1, k2 = jax.random.split(key)
        a = (jax.random.normal(k1, (M, K), dtype=jnp.bfloat16) * 0.1)
        b = (jax.random.normal(k2, (N, K), dtype=jnp.bfloat16) * 0.1)
        ap, a_s = bf16_to_mxfp4(a)
        bp, b_s = bf16_to_mxfp4(b)
        bp_sh = shuffle_weight(bp)
        as_sh = e8m0_shuffle(a_s)
        bs_sh = e8m0_shuffle(b_s)
        op = {"a_packed": ap, "col_b_fp4": bp_sh,
              "a_scale": as_sh, "col_b_scale": bs_sh}
        ref = run_gemm(gemm_fp4, op, REF_KNL, REF_SPLIT)
        ref_norm = float(np.linalg.norm(ref))
        print(f"\n# {label}  (M={M} N={N} K={K})  |ref|={ref_norm:.3e}")
        print(f"#   {'variant':>16} | {'cos(var,ref)':>14} | {'rel_l2':>10} "
              f"| {'max_abs':>10} | neutral")
        cell_rec = {"label": label, "M": M, "N": N, "K": K, "variants": {}}
        cell_worst = 0.0
        for (vlabel, knl, s) in sweep:
            try:
                out = run_gemm(gemm_fp4, op, knl, s)
            except Exception as exc:  # noqa: BLE001
                print(f"#   {vlabel:>16} | ERROR: {str(exc)[:70]}")
                cell_rec["variants"][vlabel] = {"error": str(exc)[:200]}
                continue
            m = metrics(out, ref)
            neutral = (m["cos"] >= COS_MIN and m["rel_l2"] <= REL_L2_MAX)
            cell_worst = max(cell_worst, m["rel_l2"])
            if not neutral:
                all_neutral = False
            cell_rec["variants"][vlabel] = {**m, "neutral": neutral}
            print(f"#   {vlabel:>16} | {m['cos']:>14.8f} | {m['rel_l2']:>10.3e} "
                  f"| {m['max_abs']:>10.3e} | {neutral}")
        cell_rec["worst_rel_l2_vs_ref"] = cell_worst
        records.append(cell_rec)

    print(f"\n{'='*78}\n=== NEUTRALITY VERDICT: "
          f"{'ALL NEUTRAL' if all_neutral else 'NON-NEUTRAL VARIANT FOUND'} ===")
    worst = max((c['worst_rel_l2_vs_ref'] for c in records), default=0.0)
    print(f"worst rel_l2 (any variant vs 256x256/sK1, any cell) = {worst:.3e}")
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"all_neutral": all_neutral, "worst_rel_l2": worst,
                       "cos_min": COS_MIN, "rel_l2_max": REL_L2_MAX,
                       "cells": records}, f, indent=2)
        print(f"[neutrality] wrote {out_path}")
    return records, all_neutral


# ---------------------------------------------------------------------------
# Operand construction for the dA GEMM of one site.
#   grad : [M, C] bf16 raw cotangent (C = N_orig = contraction = proj out-feat)
#   wt   : [C, P] bf16 raw weight    (P = K_orig = proj in-feat = dA result dim)
#   dA   : [M, P] = (Hadamard(grad) @ wt) with both FP4-quantised.
# Returns the AITER operands + the dequantised exact operands + the pre-quant
# fp32 operands (for the "wall" reference).
# ---------------------------------------------------------------------------
def build_dA_operands(grad_in, wt_in, cast_mxfp4, cast_mxfp4_dual,
                      jnp, had_weight):
    # The fused cast kernel reads bf16 bytes -> feed bf16 (NOT f32), and derive
    # the fp32 references from the bf16-widened values so kernel + host agree.
    grad_bf16 = jnp.asarray(grad_in, dtype=jnp.bfloat16)
    wt_bf16 = jnp.asarray(wt_in, dtype=jnp.bfloat16)
    M, C = grad_bf16.shape
    Cw, P = wt_bf16.shape
    assert Cw == C, f"weight rows {Cw} != contraction {C}"

    # ---- A operand = grad rowwise cast over C, Hadamard ON (go_row). ----
    a_packed, a_scale = cast_mxfp4(grad_bf16, shuffle_fp4=False,
                                   shuffle_scales=True, use_hadamard=True,
                                   use_sr=False)
    a_packed_np = np.asarray(a_packed)
    a_scale_np = np.asarray(a_scale)
    a_codes = cp.unpack_fp4_linear(a_packed_np)                 # [M, C]
    a_e8m0 = cp.unshuffle_scales(a_scale_np, M, C // 32)         # [M, C/32]
    a_native = (a_e8m0.astype(np.uint32) << np.uint32(23)).view(np.float32)
    a_deq = cp.dequant_codes(a_codes, a_native)                 # [M, C] exact

    # ---- B operand = weight columnwise cast (col_b_fp4). ----
    # logical b_deq is [P, C] with 32-blocks along C (the contraction).
    # Feed AITER the B-preshuffled colwise output; dequant the shuffle=False
    # twin (identical logical values -- shuffle is a store-time permutation).
    _, _, col_b_fp4, col_b_scale = cast_mxfp4_dual(
        wt_bf16, shuffle_fp4=True, shuffle_colwise_fp4=True,
        shuffle_scales=True, use_hadamard=had_weight, use_sr=False)
    _, _, cb_lin, cb_scale = cast_mxfp4_dual(
        wt_bf16, shuffle_fp4=True, shuffle_colwise_fp4=False,
        shuffle_scales=True, use_hadamard=had_weight, use_sr=False)
    cb_lin_np = np.asarray(cb_lin)
    cb_scale_np = np.asarray(cb_scale)
    b_codes = cp.unpack_fp4_linear(cb_lin_np)                   # [P, C]
    b_e8m0 = cp.unshuffle_scales(cb_scale_np, P, C // 32)       # [P, C/32]
    b_native = (b_e8m0.astype(np.uint32) << np.uint32(23)).view(np.float32)
    b_deq = cp.dequant_codes(b_codes, b_native)                 # [P, C] exact

    # ---- pre-quant fp32 operands (the "no-FP4" wall reference) ----
    grad_f32 = np.asarray(grad_bf16.astype(jnp.float32), dtype=np.float32)
    wt_f32 = np.asarray(wt_bf16.astype(jnp.float32), dtype=np.float32)
    a_unq = cp.hadamard16_blockdiag(grad_f32, axis=1)           # [M, C]
    b_unq = (cp.hadamard16_blockdiag(wt_f32, axis=0) if had_weight
             else wt_f32).T.copy()                              # [P, C]

    return dict(a_packed=a_packed, a_scale=a_scale,
                col_b_fp4=col_b_fp4, col_b_scale=col_b_scale,
                a_deq=a_deq, b_deq=b_deq, a_unq=a_unq, b_unq=b_unq)


def run_gemm(gemm_fp4, op, knl_name, log2_split):
    """Run the FFI gemm under a forced kernel/split (env read per-call)."""
    if knl_name is None:        # force OFF -> heuristic
        os.environ.pop(FORCE_NAME_ENV, None)
        os.environ.pop(FORCE_SPLIT_ENV, None)
    else:
        os.environ[FORCE_NAME_ENV] = knl_name
        os.environ[FORCE_SPLIT_ENV] = str(log2_split)
    out = gemm_fp4(op["a_packed"], op["col_b_fp4"], op["a_scale"],
                   op["col_b_scale"])
    out.block_until_ready()
    return np.asarray(out.astype(out.dtype)).astype(np.float32)


def confirm_production_choice(gemm_fp4, cast_mxfp4, cast_mxfp4_dual, jnp,
                              cfgs, num_cu, seed):
    """At the PRODUCTION token dim M=32768, confirm the heuristic (force OFF)
    selects exactly the forced 256x256/splitK=1 for each unique dA (N=P)
    shape -- i.e. the force is not overriding a different dgrad choice.

    Compares force_off vs forced(256x256/sK1) bitwise on synthetic operands
    (only the SELECTED kernel matters for byte-equality)."""
    rng = np.random.default_rng(seed)
    recs = []
    print(f"\n{'='*78}\n=== STEP 1: production-M (32768) heuristic vs forced ===")
    # unique dA (P=N_ffi, C=K_ffi) across the 4 dgrad sites.
    for P, C in [(4096, 4096), (14336, 4096)]:
        pred = select_fp4_kernel(32768, P, num_cu, cfgs)
        grad = jnp.asarray(rng.standard_normal((32768, C)), jnp.bfloat16)
        wt = jnp.asarray(rng.standard_normal((C, P)) * 0.02, jnp.bfloat16)
        op = build_dA_operands(grad, wt, cast_mxfp4, cast_mxfp4_dual, jnp, False)
        o_off = run_gemm(gemm_fp4, op, None, 0)
        o_forced = run_gemm(gemm_fp4, op, FORCED_KNL, 0)
        identical = bool(np.array_equal(o_off, o_forced))
        tag = pred[0].split("Fp4_")[-1]
        print(f"  dA N={P:>5} K={C}: heuristic={tag} splitK={1 << pred[1]} "
              f"| force_off==forced(256x256,sK1) bitwise: {identical}")
        recs.append({"N": P, "K": C,
                     "heuristic": [pred[0], int(pred[1])],
                     "forceoff_equals_forced": identical})
        del op, o_off, o_forced
    return recs


# ---------------------------------------------------------------------------
# Inputs: real captured 8B tensors + synthetic real-ish (incl. full K).
# ---------------------------------------------------------------------------
def site_list(args):
    """Return [(label, grad[M,C], wt[C,P], note)] for each dA site.

    Real captures are keyed by the FORWARD (M, N_orig, K_orig):
      __grad_out.npy = grad_out[M, N_orig]; __b.npy = weight[N_orig, K_orig].
    For dA: C = N_orig (contraction), P = K_orig.  grad = grad_out,
    wt = weight^T  (so wt is [C, P] = [N_orig, K_orig]).
    """
    rng = np.random.default_rng(args.seed)
    cap = args.captures_dir
    out = []

    # forward (M, N_orig, K_orig) -> dA (M, C=N_orig, P=K_orig)
    real_sites = [
        ("attn_qo",     4096,  4096),   # dA (M, 4096, 4096)
        ("attn_kv",     1024,  4096),   # dA (M, 1024, 4096)
        ("mlp_gate_up", 14336, 4096),   # dA (M, 14336, 4096) -- capped to 4096
        ("mlp_down",    4096,  14336),  # dA (M, 4096, 14336)
    ]
    if not args.no_real:
        for name, N_orig, K_orig in real_sites:
            g = f"{cap}/site_M32768_N{N_orig}_K{K_orig}__grad_out.npy"
            b = f"{cap}/site_M32768_N{N_orig}_K{K_orig}__b.npy"
            if not (os.path.exists(g) and os.path.exists(b)):
                print(f"[warn] missing capture for {name}: {g}", flush=True)
                continue
            grad = np.load(g)                       # grad_out[<=4096, <=N_orig]
            wt_nk = np.load(b)                       # weight[<=N_orig, K_orig]=[C,P]
            M = grad.shape[0] - grad.shape[0] % 256
            C = min(grad.shape[1], wt_nk.shape[0])   # contraction = N_orig (capped)
            C -= C % 64
            grad = np.ascontiguousarray(grad[:M, :C], dtype=np.float32)   # [M, C]
            wt = np.ascontiguousarray(wt_nk[:C, :], dtype=np.float32)     # [C, P]
            note = (f"REAL cap; C(contract)={C}"
                    + ("" if C == N_orig else f" (capped from {N_orig})"))
            out.append((f"real_{name}_dA(M{M},N{wt.shape[1]},K{C})",
                        grad, wt, note))

    if not args.no_synth:
        # Synthetic real-ish at the TRUE dA FFI shapes (incl. K=14336 contraction
        # that the real captures cannot reach). grad ~ heavy-tailed; wt ~ normal.
        M = args.synth_m
        synth = [
            ("attn_qo",     4096,  4096),
            ("attn_kv",     1024,  4096),
            ("mlp_gate_up", 14336, 4096),   # large contraction C=14336
            ("mlp_down",    4096,  14336),
        ]
        for name, C, P in synth:           # C=N_orig contraction, P=K_orig
            grad = (rng.standard_normal((M, C)) *
                    rng.standard_gamma(0.5, (M, C))).astype(np.float32)
            wt = (rng.standard_normal((C, P)) * 0.02).astype(np.float32)
            out.append((f"synth_{name}_dA(M{M},N{P},K{C})", grad, wt,
                        "SYNTH heavy-tail grad + normal wt"))
    if args.quick:
        out = [o for o in out if "attn_kv" in o[0]][:1] or out[:1]
    return out


# ---------------------------------------------------------------------------
def precheck(source_prefix, ja_root):
    import jax
    import jax_aiter
    jaf = os.path.abspath(jax_aiter.__file__)
    gemm_so = os.path.join(ja_root, "build/jax_aiter_build/gemm_fp4_ja.so")
    cast_so = os.path.join(ja_root, "build/jax_aiter_build/cast_mxfp4_ja.so")

    def _md5(p):
        return hashlib.md5(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None

    info = {
        "jax_aiter__file__": jaf,
        "source_override_ok": jaf.startswith(os.path.abspath(source_prefix)),
        "JA_ROOT_DIR": os.environ.get("JA_ROOT_DIR"),
        "gemm_fp4_ja.so": gemm_so, "gemm_so_md5": _md5(gemm_so),
        "cast_mxfp4_ja.so_md5": _md5(cast_so),
        "jax_devices": [str(d) for d in jax.devices()],
        "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
        "AITER_FUSED_QUANT_HADAMARD": os.environ.get("AITER_FUSED_QUANT_HADAMARD"),
    }
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captures-dir", default="/ruvaidya/aiter_proj/docs/logs/captures/20260602_8b_allon_steady")
    ap.add_argument("--num-cu", type=int, default=128)
    ap.add_argument("--synth-m", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=20260603)
    ap.add_argument("--no-real", action="store_true")
    ap.add_argument("--no-synth", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-confirm", action="store_true",
                    help="skip the production-M=32768 force-off vs forced confirm")
    ap.add_argument("--had-weight", action="store_true",
                    help="apply Hadamard to the weight cast too (--hadamard-all parity)")
    ap.add_argument("--csv", default=None,
                    help="f4gemm cfg CSV (default: $JA_ROOT_DIR/third_party/.../f4gemm_bf16_per1x32Fp4.csv)")
    ap.add_argument("--source-prefix", default="/ruvaidya/aiter_proj/jax-aiter/jax_aiter")
    ap.add_argument("--out", default="")
    ap.add_argument("--neutrality", action="store_true",
                    help="Kernel-selection study todo 1: instead of the dgrad "
                         "faithfulness sweep, run a numerical-neutrality sweep "
                         "over all 8B full-scope role cells (MNK-complete) "
                         "comparing every (kernel,splitK) variant to the "
                         "256x256/sK1 production reference. Exits after.")
    ap.add_argument("--neutrality-out", default="",
                    help="JSON output path for the --neutrality results.")
    args = ap.parse_args()
    np.set_printoptions(precision=6, suppress=True, linewidth=140)

    ja_root = os.environ.get("JA_ROOT_DIR", "/ruvaidya/aiter_proj/jax-aiter")
    csv_path = args.csv or os.path.join(
        ja_root, "third_party/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4.csv")
    cfgs = load_cfgs(csv_path)

    import jax  # noqa: F401
    import jax.numpy as jnp
    from jax_aiter.ops.gemm_fp4 import cast_mxfp4, cast_mxfp4_dual, gemm_fp4

    info = precheck(args.source_prefix, ja_root)
    print("=== PRE-CHECK / PROVENANCE ===")
    for k, v in info.items():
        print(f"  {k}: {v}")
    if not info["source_override_ok"]:
        print(f"[FATAL] jax_aiter not from source ({args.source_prefix}).")
        sys.exit(2)

    # ---- control: confirm env override is read per-call (bogus name errors) ----
    print("\n=== CONTROL: env override is live (bogus kernel must error) ===")
    env_live = False
    try:
        rng = np.random.default_rng(0)
        gctl = jnp.asarray(rng.standard_normal((256, 1024)), jnp.bfloat16)
        wctl = jnp.asarray(rng.standard_normal((1024, 256)) * 0.02, jnp.bfloat16)
        opctl = build_dA_operands(gctl, wctl, cast_mxfp4, cast_mxfp4_dual,
                                  jnp, args.had_weight)
        run_gemm(gemm_fp4, opctl, "_ZN5aiter_BOGUS_KERNEL_DOES_NOT_EXISTE", 0)
        print("  [WARN] bogus kernel did NOT error -- env override may be cached!")
    except Exception as exc:
        env_live = True
        print(f"  OK: bogus kernel rejected by FFI -> env override is per-call live.")
        print(f"      ({type(exc).__name__}: {str(exc)[:120]})")
    finally:
        os.environ.pop(FORCE_NAME_ENV, None)
        os.environ.pop(FORCE_SPLIT_ENV, None)

    # ---- variant sweep list (knl short -> mangled) ----
    KNL = {tM_tN: knl for (tM, tN, sk, bps, knl) in cfgs
           for tM_tN in [f"{tM}x{tN}"]}
    SPLITK_CAP = {f"{tM}x{tN}": bool(sk) for (tM, tN, sk, bps, knl) in cfgs}
    sweep = [("force_off", None, 0)]
    for tile, splits in [("256x256", [0, 1, 2, 3]), ("128x512", [0, 1, 2, 3]),
                         ("128x256", [0]), ("256x128", [0]), ("224x256", [0]),
                         ("192x256", [0]), ("128x384", [0]), ("160x256", [0]),
                         ("160x384", [0])]:
        if tile not in KNL:
            continue
        for s in splits:
            if s > 0 and not SPLITK_CAP.get(tile):
                continue
            sweep.append((f"{tile}/sK{1 << s}", KNL[tile], s))

    # ---- kernel-selection study todo 1: numerical-neutrality sweep ----
    if args.neutrality:
        run_neutrality(jnp, gemm_fp4, sweep, NEUTRALITY_CELLS_8B,
                       args.seed, args.neutrality_out)
        return

    results = {"provenance": info, "env_override_live": env_live,
               "num_cu": args.num_cu, "had_weight": args.had_weight,
               "forced_recipe": {"knl": FORCED_KNL, "log2_split": 0},
               "sites": []}

    if not args.quick and not args.no_confirm:
        results["production_choice"] = confirm_production_choice(
            gemm_fp4, cast_mxfp4, cast_mxfp4_dual, jnp, cfgs, args.num_cu,
            args.seed)

    for label, grad, wt, note in site_list(args):
        M, C = grad.shape
        P = wt.shape[1]
        print(f"\n{'='*78}\n# {label}\n#   {note}")
        print(f"#   dA FFI (M={M}, N={P}, K={C}); contraction C={C}")

        # Step 1: heuristic prediction at production M=32768 AND test M.
        pred_prod = select_fp4_kernel(32768, P, args.num_cu, cfgs)
        pred_test = select_fp4_kernel(M, P, args.num_cu, cfgs)
        print(f"#   heuristic @M=32768: {pred_prod[0].split('Fp4_')[-1]} "
              f"splitK={1 << pred_prod[1]}   | @M={M}: "
              f"{pred_test[0].split('Fp4_')[-1]} splitK={1 << pred_test[1]}")
        print(f"#   forced (production): BpreShuffle_256x256 splitK=1")

        op = build_dA_operands(grad, wt, cast_mxfp4, cast_mxfp4_dual,
                               jnp, args.had_weight)

        # validate the dequant/unshuffle path vs an independent host cast.
        rc, _, rn, _ = cp.quantize_lastaxis(op["a_unq"])
        a_deq_host = cp.dequant_codes(rc, rn)
        a_val = metrics(op["a_deq"], a_deq_host)
        rcb, _, rnb, _ = cp.quantize_lastaxis(op["b_unq"])
        b_deq_host = cp.dequant_codes(rcb, rnb)
        b_val = metrics(op["b_deq"], b_deq_host)
        print(f"#   operand validation (fed-deq vs host-cast): "
              f"A cos={a_val['cos']:.8f} maxabs={a_val['max_abs']:.2e} | "
              f"B cos={b_val['cos']:.8f} maxabs={b_val['max_abs']:.2e}")

        # IDEAL-FP4 (fp32) + bf16 "wall" reference.
        ideal = (op["a_deq"].astype(np.float32) @ op["b_deq"].astype(np.float32).T)
        wall = (op["a_unq"].astype(np.float32) @ op["b_unq"].astype(np.float32).T)
        bf16_floor = metrics(ideal.astype(jnp.bfloat16).astype(np.float32), ideal)
        rep_wall = metrics(ideal, wall)
        print(f"#   bf16(ideal) vs ideal  [output bf16 floor]: "
              f"rel_l2={bf16_floor['rel_l2']:.3e} cos={bf16_floor['cos']:.8f}")
        print(f"#   ideal-FP4 vs no-FP4   [E2M1 representation wall]: "
              f"rel_l2={rep_wall['rel_l2']:.3e} cos={rep_wall['cos']:.6f}")

        # Step 2 + 3: AITER (forced) + variant sweep vs IDEAL-FP4.
        site_rec = {"label": label, "note": note, "M": M, "N": P, "K": C,
                    "heuristic_pred_M32768": [pred_prod[0], int(pred_prod[1])],
                    "heuristic_pred_Mtest": [pred_test[0], int(pred_test[1])],
                    "operand_valid": {"A": a_val, "B": b_val},
                    "bf16_output_floor": bf16_floor,
                    "representation_wall": rep_wall, "variants": {}}
        forced_out = None
        forceoff_out = None
        print(f"#   {'variant':>16} | {'cos(AITER,ideal)':>18} | "
              f"{'rel_l2':>10} | {'max_abs':>10} | {'cos(AITER,noFP4)':>16}")
        for vlabel, knl, s in sweep:
            try:
                out = run_gemm(gemm_fp4, op, knl, s)
            except Exception as exc:
                print(f"#   {vlabel:>16} | ERROR: {str(exc)[:80]}")
                site_rec["variants"][vlabel] = {"error": str(exc)[:200]}
                continue
            m_ideal = metrics(out, ideal)
            m_wall = metrics(out, wall)
            site_rec["variants"][vlabel] = {"vs_ideal": m_ideal, "vs_noFP4": m_wall}
            print(f"#   {vlabel:>16} | {m_ideal['cos']:>18.8f} | "
                  f"{m_ideal['rel_l2']:>10.3e} | {m_ideal['max_abs']:>10.3e} | "
                  f"{m_wall['cos']:>16.6f}")
            if vlabel == "force_off":
                forceoff_out = out
            if vlabel == "256x256/sK1":
                forced_out = out

        # empirical: does force_off == forced 256x256 splitK=1 (bitwise)?
        if forced_out is not None and forceoff_out is not None:
            identical = bool(np.array_equal(forced_out, forceoff_out))
            site_rec["forceoff_equals_forced"] = identical
            print(f"#   force_off == forced(256x256,sK1) bitwise: {identical}")

        # best variant vs forced
        vs = {k: v["vs_ideal"]["rel_l2"] for k, v in site_rec["variants"].items()
              if "vs_ideal" in v}
        if vs:
            best = min(vs, key=vs.get)
            forced_rel = vs.get("256x256/sK1", float("nan"))
            site_rec["best_variant"] = best
            site_rec["best_rel_l2"] = vs[best]
            site_rec["forced_rel_l2"] = forced_rel
            improve = (forced_rel - vs[best]) / forced_rel if forced_rel else 0.0
            site_rec["best_improvement_frac"] = improve
            print(f"#   BEST vs ideal: {best} (rel_l2={vs[best]:.3e}); "
                  f"forced=256x256/sK1 (rel_l2={forced_rel:.3e}); "
                  f"improvement={improve*100:.2f}%")
        results["sites"].append(site_rec)

    # ---------------- verdict ----------------
    print(f"\n{'='*78}\n=== VERDICT ===")
    faithful = True
    degraded_sites, better_variant_sites = [], []
    for s in results["sites"]:
        forced = s.get("forced_rel_l2", float("nan"))
        floor = s["bf16_output_floor"]["rel_l2"]
        # faithful if forced kernel is within ~3x the bf16 output floor of ideal.
        if not (forced <= max(3 * floor, floor + 5e-3)):
            faithful = False
            degraded_sites.append(s["label"])
        if s.get("best_improvement_frac", 0.0) > 0.10 and s.get("best_variant") != "256x256/sK1":
            better_variant_sites.append((s["label"], s["best_variant"],
                                         s["best_improvement_frac"]))
    results["verdict"] = {
        "f4gemm_faithful_for_dA": faithful and not better_variant_sites,
        "degraded_sites": degraded_sites,
        "better_variant_sites": better_variant_sites,
    }
    print(json.dumps(results["verdict"], indent=2))
    if faithful and not better_variant_sites:
        print("\nf4gemm is FAITHFUL for dA: the forced 256x256/splitK=1 matches "
              "IDEAL-FP4 to the bf16 output floor, and no variant is meaningfully "
              "closer. The dgrad loss is the genuine E2M1 precision wall -- "
              "confirmed at the GEMM level. No E2E warranted.")
    else:
        print("\nf4gemm DEGRADES dA beyond FP4 (or a better variant exists). "
              "See better_variant_sites; an E2E routing dA through that variant "
              "may recover R.")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[results] wrote {args.out}")


if __name__ == "__main__":
    main()
