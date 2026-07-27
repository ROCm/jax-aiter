#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Direct cast-vs-cast parity: our DPP/ds_swizzle MXFP4 cast vs a non-DPP reference.

Question this answers
---------------------
Does the fused HIP cast kernel
(``csrc/ffi/cast_mxfp4/cast_transpose_mxfp4_kernel_shuffled.cu``), whose
cross-lane Hadamard XOR-1/XOR-2 and amax-reduce XOR-4/2/1 steps use AMD DPP
``quad_perm`` + ``ds_swizzle_b32 offset:0x101F``, faithfully realize the
*intended* Hadamard + amax + E8M0 + E2M1 quantization?

We do NOT compare against fp32 truth (both our cast and a faithful reference
sit on the same ~cos 0.987 E2M1 noise floor, which cannot distinguish them).
Instead we compare our DPP cast output DIRECTLY against an independent,
non-DPP, host reference that replicates the EXACT math of the .cu:

  1. bf16 -> fp32 widening (low-16-zero, lossless), identical to the kernel.
  2. 16-point block-diagonal Hadamard (`hadamard16_inplace`): per-thread H4 +
     XOR-1 + XOR-2 cross-thread butterfly, x0.25 norm, two H16 per 32-block.
     We replicate the EXACT fp32 operand order (no matrix multiply), so a
     correct DPP cast must agree BIT-EXACTLY on the post-Hadamard values.
  3. per-32-block amax (max|.|).
  4. `compute_e8m0_scale`: round-amax-to-pow2 via (bits+0x200000)&0xFF800000,
     exp-2 headroom, clamp [-127,127] -> E8M0 byte + native_scale=2^su.
  5. E2M1 quant of (v / native_scale) with round-to-nearest-EVEN ties
     (confirmed against AITER CK `convert_to_type`, which mirrors the
     `v_cvt_scalef32_pk_fp4_f32` hardware).

Because the reference reproduces the kernel's exact fp32 arithmetic order, a
faithful DPP cast should match the reference essentially bit-exactly:
  * E8M0 scales: depend only on amax (a tie-free max reduction) -> ANY scale
    mismatch is a real cross-lane bug (Hadamard produced wrong values, or the
    amax reduce dropped a lane).
  * FP4 codes: should match except (a) signed-zero (+0 vs -0, value-identical)
    and (b) rare exact float midpoints (RNE ties) -> small, UNBIASED, boundary.

Both the rowwise AND columnwise/transpose cast paths are tested (the columnwise
path is the dgrad weight operand + wgrad site -- another DPP+swizzle location).

Read-only w.r.t. the cast kernel source. Uses the freshly-built
``cast_mxfp4_ja.so`` via ``JA_ROOT_DIR`` and the source ``jax_aiter`` package
via ``PYTHONPATH`` (a pre-check asserts both). No rebuild. RNE path (use_sr
unset). GPU is used only for our cast.

Verdict logic
-------------
  YES (DPP faithful): scale mismatches == 0 AND code mismatches are confined to
    signed-zero / exact-midpoint ties, unbiased, tiny fraction. -> The FP4 8B
    loss ceiling is a genuine E2M1 precision wall.
  NO  (DPP bug):      any systematic / biased / large code divergence, scale
    mismatches, or column-path-specific divergence. -> A cast bug is (partly)
    inflating FP4 damage; localize (Hadamard / amax-reduce / scale / transpose).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

# ===========================================================================
# E2M1 grid (OCP MXFP4) and rounding helpers
# ===========================================================================
# code (3-bit magnitude) -> value:  S000=0 S001=.5 S010=1 S011=1.5
#                                    S100=2 S101=3  S110=4 S111=6
E2M1_GRID = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
# Decision midpoints between consecutive grid magnitudes.
E2M1_MIDS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)
# Even-mantissa neighbour code at each midpoint (round-to-nearest-EVEN tie):
#   0.25->0(0.0) 0.75->2(1.0) 1.25->2(1.0) 1.75->4(2.0)
#   2.5->4(2.0)  3.5->6(4.0)  5.0->6(4.0)
E2M1_MID_EVEN = np.array([0, 2, 2, 4, 4, 6, 6], dtype=np.int32)
# Full signed LUT (index = 4-bit code) for dequantisation.
E2M1_LUT = np.concatenate([E2M1_GRID, -E2M1_GRID]).astype(np.float32)


def e2m1_magnitude_code(qabs: np.ndarray, rne_even: bool = True) -> np.ndarray:
    """|q| (>=0) -> 3-bit E2M1 magnitude code (0..7), saturating at 6.0.

    rne_even=True  : round-to-nearest, ties-to-even  (hardware semantics).
    rne_even=False : round-to-nearest, ties-up        (== fp4_utils._f32_to_e2m1).
    """
    qabs = np.asarray(qabs, dtype=np.float32)
    # side='right': at an exact midpoint, fall into the upper bin (ties-up).
    code = np.searchsorted(E2M1_MIDS, qabs.ravel(), side="right").reshape(qabs.shape)
    code = np.minimum(code.astype(np.int32), 7)
    if rne_even:
        for i in range(E2M1_MIDS.shape[0]):
            at_mid = qabs == E2M1_MIDS[i]
            if at_mid.any():
                code = np.where(at_mid, E2M1_MID_EVEN[i], code)
    return code


# ===========================================================================
# Bit-exact replication of the .cu kernel math (host, numpy float32, NO DPP)
# ===========================================================================

def hadamard16_blockdiag(x: np.ndarray, axis: int) -> np.ndarray:
    """Apply the EXACT `hadamard16_inplace` butterfly block-diagonally (16/block).

    Mirrors the .cu operand-by-operand in float32: per-thread H4 (stage 1),
    XOR-1 cross-thread butterfly (stage 2, partner thread^1, odd threads
    subtract), XOR-2 (stage 3, partner thread^2, threads 2-3 subtract), x0.25.

    Within each contiguous 16-element block, element e maps to
    (thread=e//4, value=e%4) -- exactly the kernel's lane/value layout.
    """
    x = np.moveaxis(x, axis, -1)
    shp = x.shape
    L = shp[-1]
    assert L % 16 == 0, f"axis length {L} must be a multiple of 16"
    v = np.ascontiguousarray(x, dtype=np.float32).reshape(*shp[:-1], L // 16, 4, 4)
    # v[..., block, thread, value]
    v0 = v[..., 0]; v1 = v[..., 1]; v2 = v[..., 2]; v3 = v[..., 3]  # [...,blk,4]

    # Stage 1: local 4-point Hadamard (per thread).
    a0 = v0 + v1
    a1 = v0 - v1
    a2 = v2 + v3
    a3 = v2 - v3
    n0 = a0 + a2
    n2 = a0 - a2
    n1 = a1 + a3
    n3 = a1 - a3
    v = np.stack([n0, n1, n2, n3], axis=-1).astype(np.float32)  # [...,blk,4,4]

    # Stage 2: XOR-1 butterfly across the thread axis (partner = thread^1).
    p = v[..., [1, 0, 3, 2], :]
    sign2 = np.array([0, 1, 0, 1], dtype=np.int32).reshape(4, 1)  # odd -> subtract
    v = np.where(sign2 == 1, p - v, p + v).astype(np.float32)

    # Stage 3: XOR-2 butterfly across the thread axis (partner = thread^2).
    p = v[..., [2, 3, 0, 1], :]
    sign3 = np.array([0, 0, 1, 1], dtype=np.int32).reshape(4, 1)  # threads 2,3 subtract
    v = np.where(sign3 == 1, p - v, p + v).astype(np.float32)

    v = (v * np.float32(0.25)).astype(np.float32)
    y = v.reshape(*shp)
    return np.moveaxis(y, -1, axis)


def compute_e8m0_scale(amax: np.ndarray):
    """Bit-exact replication of the .cu `compute_e8m0_scale`.

    Returns (e8m0_byte[uint8], native_scale[float32]) with the kernel's
    amax==0 special case (native_scale=1, e8m0=127).
    """
    amax = np.asarray(amax, dtype=np.float32)
    bits = amax.view(np.uint32)
    bits = (bits + np.uint32(0x200000)) & np.uint32(0xFF800000)  # round amax to pow2
    exp = ((bits >> np.uint32(23)) & np.uint32(0xFF)).astype(np.int32) - 127
    su = np.clip(exp - 2, -127, 127).astype(np.int32)            # -2 headroom + clamp
    # native_scale = reinterpret((127+su) << 23) == reinterpret(e8m0 << 23) (bit-exact)
    native = ((127 + su).astype(np.uint32) << np.uint32(23)).view(np.float32)
    e8m0 = (su + 127).astype(np.uint8)
    zero = amax == np.float32(0.0)
    native = np.where(zero, np.float32(1.0), native).astype(np.float32)
    e8m0 = np.where(zero, np.uint8(127), e8m0).astype(np.uint8)
    return e8m0, native


def quantize_lastaxis(v: np.ndarray, rne_even: bool = True):
    """Quantise a [..., L] fp32 array (Hadamard already applied) along the last
    axis in 32-element blocks -> (codes[...,L] uint8 incl. sign bit,
    e8m0[..., L//32] uint8, native_scale[..., L//32] f32, qabs[...,L] f32).
    """
    v = np.ascontiguousarray(v, dtype=np.float32)
    shp = v.shape
    L = shp[-1]
    assert L % 32 == 0, f"last axis {L} must be a multiple of 32"
    nb = L // 32
    blocks = v.reshape(*shp[:-1], nb, 32)
    amax = np.max(np.abs(blocks), axis=-1).astype(np.float32)         # [...,nb]
    e8m0, native = compute_e8m0_scale(amax)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = (blocks / native[..., None]).astype(np.float32)          # exact pow2 div
    qabs = np.abs(q).astype(np.float32)
    sign = (q < 0) | ((q == 0) & (np.signbit(q)))                    # negative incl -0
    mag = e2m1_magnitude_code(qabs, rne_even=rne_even).astype(np.uint8)
    code = np.where((sign) & (mag != 0), mag | np.uint8(8), mag).astype(np.uint8)
    code = code.reshape(*shp)
    qabs = qabs.reshape(*shp)
    return code, e8m0, native, qabs


def dequant_codes(codes: np.ndarray, native_scale_blocks: np.ndarray) -> np.ndarray:
    """codes[...,L] (4-bit incl sign) + native_scale[...,L//32] -> fp32 values."""
    codes = np.asarray(codes)
    shp = codes.shape
    L = shp[-1]
    nb = L // 32
    vals = E2M1_LUT[codes.astype(np.int32)].reshape(*shp[:-1], nb, 32)
    deq = vals * native_scale_blocks[..., None]
    return deq.reshape(*shp).astype(np.float32)


# ===========================================================================
# Kernel-output unpacking + scale-shuffle inversion (exact .cu formulas)
# ===========================================================================

def unpack_fp4_linear(packed: np.ndarray) -> np.ndarray:
    """[R, C/2] uint8 packed (low nibble=even col, high=odd) -> [R, C] codes."""
    packed = np.asarray(packed, dtype=np.uint8)
    R, Ch = packed.shape
    out = np.empty((R, Ch * 2), dtype=np.uint8)
    out[:, 0::2] = packed & np.uint8(0x0F)
    out[:, 1::2] = packed >> np.uint8(4)
    return out


def _shuffle_index(row, col, scale_n_pad):
    """Exact mirror of the .cu `compute_shuffle_index` (flat C-order index)."""
    i0 = row >> 5
    i1 = (row >> 4) & 1
    i2 = row & 15
    i3 = col >> 3
    i4 = (col >> 2) & 1
    i5 = col & 3
    return ((i0 * (scale_n_pad >> 3)) << 8) + (i3 << 8) + (i5 << 6) + (i2 << 2) + (i4 << 1) + i1


def unshuffle_scales(shuf: np.ndarray, n_rows: int, n_blocks: int) -> np.ndarray:
    """Invert the kernel E8M0 scale shuffle -> linear [n_rows, n_blocks] uint8."""
    shuf = np.asarray(shuf, dtype=np.uint8)
    scale_n_pad = shuf.shape[1]
    flat = shuf.reshape(-1)
    rows = np.arange(n_rows)[:, None]
    cols = np.arange(n_blocks)[None, :]
    idx = _shuffle_index(rows, cols, scale_n_pad)
    return flat[idx]


def canon_code(c: np.ndarray) -> np.ndarray:
    """Canonicalise signed-zero: any zero-magnitude code (0 or 8) -> 0."""
    c = np.asarray(c, dtype=np.uint8)
    return np.where((c & np.uint8(0x07)) == 0, np.uint8(0), c)


# ===========================================================================
# Comparison metrics
# ===========================================================================

def compare_path(name, kern_codes, kern_native_blocks, kern_e8m0_lin,
                 ref_codes, ref_native_blocks, ref_e8m0_lin, ref_qabs):
    """Compute parity metrics for one cast path (rowwise or columnwise)."""
    kern_codes = np.asarray(kern_codes, dtype=np.uint8)
    ref_codes = np.asarray(ref_codes, dtype=np.uint8)
    n = kern_codes.size

    # ---- E8M0 scales (tie-free; any mismatch == cross-lane bug) ----
    scale_mismatch = int(np.sum(kern_e8m0_lin != ref_e8m0_lin))
    scale_total = int(ref_e8m0_lin.size)
    # signed scale bias (kernel - ref) in exponent units, on mismatches
    scale_diff = kern_e8m0_lin.astype(np.int32) - ref_e8m0_lin.astype(np.int32)
    scale_bias = float(np.mean(scale_diff)) if scale_total else 0.0
    scale_absmax = int(np.max(np.abs(scale_diff))) if scale_total else 0

    # ---- raw FP4 codes ----
    raw_mismatch = int(np.sum(kern_codes != ref_codes))
    # signed-zero-only (codes differ only by +0 vs -0)
    kc = canon_code(kern_codes); rc = canon_code(ref_codes)
    canon_mismatch_mask = kc != rc
    canon_mismatch = int(np.sum(canon_mismatch_mask))
    signed_zero_only = raw_mismatch - canon_mismatch

    # ---- classify canonical code mismatches: boundary tie vs systematic ----
    # boundary = ref |q| within tol of an E2M1 midpoint (RNE tie can flip code).
    boundary = systematic = 0
    systematic_examples = []
    if canon_mismatch > 0:
        idx = np.argwhere(canon_mismatch_mask)
        qv = ref_qabs[canon_mismatch_mask]
        dist = np.min(np.abs(qv[:, None] - E2M1_MIDS[None, :]), axis=1)
        tol = 1e-4 * np.maximum(qv, 1.0)        # relative float tol near midpoint
        is_boundary = dist <= tol
        boundary = int(np.sum(is_boundary))
        systematic = int(np.sum(~is_boundary))
        sys_idx = idx[~is_boundary]
        for r in sys_idx[:8]:
            r = tuple(int(t) for t in r)
            systematic_examples.append({
                "pos": r,
                "qabs": float(ref_qabs[r]),
                "kern_code": int(kern_codes[r]),
                "ref_code": int(ref_codes[r]),
            })

    # ---- dequantised value comparison ----
    deq_k = dequant_codes(kern_codes, kern_native_blocks)
    deq_r = dequant_codes(ref_codes, ref_native_blocks)
    d = (deq_k - deq_r).astype(np.float64)
    max_abs = float(np.max(np.abs(d)))
    mean_abs = float(np.mean(np.abs(d)))
    bias = float(np.mean(d))                    # signed -> detects directional bias
    rstd = float(np.std(deq_r)) + 1e-30
    # cosine of kernel-cast vs ref-cast dequant (should be ~1.0 if faithful)
    fk = deq_k.ravel().astype(np.float64); fr = deq_r.ravel().astype(np.float64)
    denom = (np.linalg.norm(fk) * np.linalg.norm(fr)) + 1e-30
    cos = float(np.dot(fk, fr) / denom)

    return {
        "path": name,
        "n_codes": int(n),
        "scale_total": scale_total,
        "scale_mismatch": scale_mismatch,
        "scale_mismatch_frac": scale_mismatch / max(scale_total, 1),
        "scale_bias_expunits": scale_bias,
        "scale_absmax_expunits": scale_absmax,
        "raw_code_mismatch": raw_mismatch,
        "raw_code_mismatch_frac": raw_mismatch / max(n, 1),
        "signed_zero_only": int(signed_zero_only),
        "canon_code_mismatch": canon_mismatch,
        "canon_code_mismatch_frac": canon_mismatch / max(n, 1),
        "boundary_tie": boundary,
        "systematic": systematic,
        "systematic_examples": systematic_examples,
        "dequant_max_abs_diff": max_abs,
        "dequant_mean_abs_diff": mean_abs,
        "dequant_signed_bias": bias,
        "dequant_bias_over_std": bias / rstd,
        "dequant_cosine": cos,
    }


# ===========================================================================
# Input generation
# ===========================================================================

def make_inputs(args):
    """Yield (label, X_f32[M,K]) representative inputs (M%128==0, K%64==0)."""
    rng = np.random.default_rng(args.seed)
    out = []

    # --- synthetic activation-like: normal + sparse heavy outliers ---
    M, K = 512, 4096
    act = rng.standard_normal((M, K)).astype(np.float32)
    mask = rng.random((M, K)) < 0.001
    act[mask] *= rng.uniform(8.0, 40.0, size=int(mask.sum())).astype(np.float32)
    out.append(("synthetic_activation_normal+outliers_512x4096", act))

    # --- synthetic gradient-like: heavy-tailed (normal * gamma(0.5)) ---
    M, K = 512, 4096
    grad = (rng.standard_normal((M, K)) *
            rng.standard_gamma(0.5, (M, K))).astype(np.float32)
    out.append(("synthetic_gradient_heavytail_512x4096", grad))

    # --- synthetic structured: many equal values (stresses RNE ties) ---
    M, K = 256, 2048
    tie = (rng.integers(-4, 5, size=(M, K)).astype(np.float32) * 0.25)  # multiples of 0.25
    tie *= (2.0 ** rng.integers(-2, 3, size=(M, 1))).astype(np.float32)  # per-row scale
    out.append(("synthetic_tiestress_quartersteps_256x2048", tie))

    # --- captured real 8B tensors (subsampled rows) ---
    if not args.no_captures:
        cap = args.captures_dir
        picks = [
            ("captured_grad_out_N4096_K4096", f"{cap}/site_M32768_N4096_K4096__grad_out.npy"),
            ("captured_weight_b_N4096_K4096", f"{cap}/site_M32768_N4096_K4096__b.npy"),
            ("captured_activation_a_N4096_K14336", f"{cap}/site_M32768_N4096_K14336__a.npy"),
        ]
        for label, path in picks:
            if not os.path.exists(path):
                print(f"[warn] capture not found, skipping: {path}")
                continue
            arr = np.load(path, mmap_mode="r")
            r = min(args.cap_rows, arr.shape[0])
            r -= r % 128                      # keep M a multiple of 128
            c = arr.shape[1] - (arr.shape[1] % 64)
            sub = np.array(arr[:r, :c], dtype=np.float32)
            out.append((f"{label}_{r}x{c}", sub))

    if args.quick:
        out = out[:2]
    return out


# ===========================================================================
# Main
# ===========================================================================

def precheck(expect_source_prefix):
    """Assert source-override + fresh .so, return provenance dict."""
    import jax
    import jax_aiter
    from jax_aiter.ja_compat import config as ja_config

    jaf = os.path.abspath(jax_aiter.__file__)
    umb = ja_config.get_umbrella_lib()
    cast_so = umb.parent / "cast_mxfp4_ja.so"
    md5 = hashlib.md5(open(cast_so, "rb").read()).hexdigest() if cast_so.exists() else None
    info = {
        "jax_aiter__file__": jaf,
        "JA_ROOT_DIR": os.environ.get("JA_ROOT_DIR"),
        "PYTHONPATH": os.environ.get("PYTHONPATH"),
        "umbrella_lib": str(umb),
        "cast_mxfp4_ja.so": str(cast_so),
        "cast_so_md5": md5,
        "cast_so_mtime": (datetime.fromtimestamp(cast_so.stat().st_mtime).isoformat()
                          if cast_so.exists() else None),
        "jax_devices": [str(d) for d in jax.devices()],
        "AITER_FP4_SR": os.environ.get("AITER_FP4_SR"),
    }
    ok_src = jaf.startswith(os.path.abspath(expect_source_prefix))
    info["source_override_ok"] = bool(ok_src)
    return info


def run(args):
    import jax
    import jax.numpy as jnp
    from jax_aiter.ops.gemm_fp4 import cast_mxfp4, cast_mxfp4_dual

    info = precheck(args.source_prefix)
    print("=== PRE-CHECK / PROVENANCE ===")
    for k, v in info.items():
        print(f"  {k}: {v}")
    if not info["source_override_ok"]:
        print(f"[FATAL] jax_aiter not imported from source ({args.source_prefix}).")
        sys.exit(2)
    print()

    rne_even = not args.ties_up
    results = {"provenance": info, "config": {
        "rne_even": rne_even, "seed": args.seed, "cap_rows": args.cap_rows,
        "use_sr": False}, "runs": []}

    for label, X in make_inputs(args):
        M, K = X.shape
        # Identical bf16 input for kernel and reference (lossless widen back).
        x_bf16 = jnp.asarray(X, dtype=jnp.bfloat16)
        Xw = np.asarray(x_bf16.astype(jnp.float32), dtype=np.float32)  # what both see

        for hadamard in (False, True):
            tag = f"{label} | hadamard={hadamard}"
            print(f"---- {tag}  (M={M}, K={K}) ----")

            # ===== ROWWISE: cast_mxfp4 (rowwise only), all shuffles OFF =====
            r_packed, r_scale = cast_mxfp4(
                x_bf16, shuffle_fp4=False, shuffle_scales=False,
                use_hadamard=hadamard, use_sr=False)
            r_packed = np.asarray(r_packed); r_scale = np.asarray(r_scale)
            k_row_codes = unpack_fp4_linear(r_packed)                  # [M,K]
            scale_N = K // 32
            k_row_e8m0 = r_scale[:M, :scale_N]                         # linear (no shuffle)

            row_in = hadamard16_blockdiag(Xw, axis=1) if hadamard else Xw
            ref_row_codes, ref_row_e8m0, ref_row_native, ref_row_qabs = quantize_lastaxis(
                row_in, rne_even=rne_even)
            # kernel native_scale = reinterpret(e8m0 << 23) -- bit-exact to the .cu
            k_row_native = (k_row_e8m0.astype(np.uint32) << np.uint32(23)).view(np.float32)
            row_res = compare_path(
                "rowwise", k_row_codes, k_row_native, k_row_e8m0,
                ref_row_codes, ref_row_native, ref_row_e8m0, ref_row_qabs)

            # ===== COLUMNWISE: cast_mxfp4_dual, fp4 linear; scales shuffled =====
            _, _, c_packed, c_scale = cast_mxfp4_dual(
                x_bf16, shuffle_fp4=False, shuffle_colwise_fp4=False,
                use_hadamard=hadamard, use_sr=False)
            c_packed = np.asarray(c_packed); c_scale = np.asarray(c_scale)
            k_col_codes = unpack_fp4_linear(c_packed)                  # [K, M]
            col_blocks = M // 32
            k_col_e8m0 = unshuffle_scales(c_scale, K, col_blocks)      # [K, M//32]

            col_in = hadamard16_blockdiag(Xw, axis=0) if hadamard else Xw
            col_in = np.ascontiguousarray(col_in.T)                    # [K, M]
            ref_col_codes, ref_col_e8m0, ref_col_native, ref_col_qabs = quantize_lastaxis(
                col_in, rne_even=rne_even)
            k_col_native = (k_col_e8m0.astype(np.uint32) << np.uint32(23)).view(np.float32)
            col_res = compare_path(
                "columnwise", k_col_codes, k_col_native, k_col_e8m0,
                ref_col_codes, ref_col_native, ref_col_e8m0, ref_col_qabs)

            for res in (row_res, col_res):
                print(f"    [{res['path']:>10}] scale_mismatch={res['scale_mismatch']}/{res['scale_total']}"
                      f" (bias={res['scale_bias_expunits']:+.2e})"
                      f"  code_mismatch={res['canon_code_mismatch']}/{res['n_codes']}"
                      f" ({res['canon_code_mismatch_frac']*100:.4f}%)"
                      f"  [bnd={res['boundary_tie']} sys={res['systematic']} sgn0={res['signed_zero_only']}]"
                      f"  deq_maxabs={res['dequant_max_abs_diff']:.3e}"
                      f" bias={res['dequant_signed_bias']:+.3e}"
                      f" cos={res['dequant_cosine']:.8f}")
                if res["systematic_examples"]:
                    print(f"        systematic examples: {res['systematic_examples']}")

            results["runs"].append({"label": label, "hadamard": hadamard,
                                    "M": M, "K": K, "rowwise": row_res,
                                    "columnwise": col_res})
            print()

    # ---- reference self cross-check vs the in-tree JAX bf16_to_mxfp4 ----
    try:
        from jax_aiter.gemm_fp4.fp4_utils import bf16_to_mxfp4
        rng = np.random.default_rng(args.seed + 99)
        Xc = rng.standard_normal((256, 1024)).astype(np.float32)
        xb = jnp.asarray(Xc, jnp.bfloat16)
        Xcw = np.asarray(xb.astype(jnp.float32), np.float32)
        jp, js = bf16_to_mxfp4(xb)           # JAX ref (no Hadamard, ties-up)
        jp = np.asarray(jp); js = np.asarray(js)
        jcodes = unpack_fp4_linear(jp)
        # match its ties-up convention for an apples-to-apples cross-check
        rc, re8, rn, rq = quantize_lastaxis(Xcw, rne_even=False)
        xcheck = {
            "scale_exact_frac": float(np.mean(js == re8)),
            "code_exact_frac": float(np.mean(canon_code(jcodes) == canon_code(rc))),
        }
        print(f"[cross-check] host-ref(ties-up) vs in-tree bf16_to_mxfp4: "
              f"scale_exact={xcheck['scale_exact_frac']*100:.3f}%  "
              f"code_exact={xcheck['code_exact_frac']*100:.3f}%")
        results["ref_crosscheck_vs_bf16_to_mxfp4"] = xcheck
    except Exception as e:  # pragma: no cover
        print(f"[cross-check] skipped: {e}")

    # ---- verdict ----
    tot_scale_mm = sum(r[p]["scale_mismatch"] for r in results["runs"] for p in ("rowwise", "columnwise"))
    tot_sys = sum(r[p]["systematic"] for r in results["runs"] for p in ("rowwise", "columnwise"))
    max_bias_over_std = max(abs(r[p]["dequant_bias_over_std"]) for r in results["runs"] for p in ("rowwise", "columnwise"))
    max_code_frac = max(r[p]["canon_code_mismatch_frac"] for r in results["runs"] for p in ("rowwise", "columnwise"))
    verdict = "YES_DPP_FAITHFUL" if (tot_scale_mm == 0 and tot_sys == 0) else "NO_DPP_BUG"
    results["verdict"] = {
        "verdict": verdict,
        "total_scale_mismatch": int(tot_scale_mm),
        "total_systematic_code_mismatch": int(tot_sys),
        "max_canon_code_mismatch_frac": float(max_code_frac),
        "max_abs_dequant_bias_over_std": float(max_bias_over_std),
    }
    print("\n=== VERDICT ===")
    print(json.dumps(results["verdict"], indent=2))

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[results] wrote {args.out}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=20260602)
    ap.add_argument("--cap-rows", type=int, default=2048,
                    help="row subsample for captured 8B tensors (multiple of 128)")
    ap.add_argument("--captures-dir", type=str,
                    default="/ruvaidya/aiter_proj/docs/logs/captures/20260603_8b_allon")
    ap.add_argument("--no-captures", action="store_true")
    ap.add_argument("--ties-up", action="store_true",
                    help="reference uses round-half-up instead of RNE-even (debug)")
    ap.add_argument("--quick", action="store_true", help="only 2 synthetic inputs")
    ap.add_argument("--source-prefix", type=str,
                    default="/ruvaidya/aiter_proj/jax-aiter/jax_aiter")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    np.set_printoptions(precision=6, suppress=True, linewidth=140)
    run(args)


if __name__ == "__main__":
    main()
