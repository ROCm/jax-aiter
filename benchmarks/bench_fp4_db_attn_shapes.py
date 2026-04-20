#!/usr/bin/env python3
"""FP4 viability benchmark at dB (wgrad) and attention shapes.

Tests whether FP4 ASM GEMM can replace:
  1) hipBLASLt FP8 for dB backward (currently _fp8_dot_general_db)
  2) AITER BF16 ASM for attention projections (currently the 7.1pp penalty)

The dB approach uses NT layout wgrad:
  - Forward saves columnwise FP4 of input activation (CastMxfp4DualJA)
  - Backward dB: quantize grad_out^T to rowwise FP4 + saved col_input -> FP4 GEMM
  NT layout keeps kernel M = N_proj (small), avoiding large-M constraints.

Shapes are production shapes (8-way FSDP):
  8B:  M=73728 (batch=9*seq=8192), hidden=4096, intermediate=14336
  70B: M=81920 (batch=10*seq=8192), hidden=8192, intermediate=28672

Usage (inside container, single GPU):
  cd /ruvaidya/aiter_proj/jax-aiter && \
    JA_ROOT_DIR=$PWD AITER_ASM_DIR=$PWD/third_party/aiter/hsa/ \
    AITER_SYMBOL_VISIBLE=1 GPU_ARCHS=gfx950 \
    HIP_VISIBLE_DEVICES=0 \
    python3 benchmarks/bench_fp4_db_attn_shapes.py [--model 8b|70b|both]
"""

import argparse
import os
import time

os.environ.setdefault("XLA_FLAGS",
    os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_cublaslt=True")
os.environ["AITER_FUSED_QUANT"] = "0"

import jax
import jax.numpy as jnp
import numpy as np


DB_SHAPES_8B = {
    "mlp_gate_up_dB": (14336, 4096, 73728),
    "mlp_down_dB":    (4096,  14336, 73728),
}

DB_SHAPES_70B = {
    "mlp_gate_up_dB": (28672, 8192, 81920),
    "mlp_down_dB":    (8192,  28672, 81920),
}

ATTN_SHAPES_8B = {
    "attn_qo": (73728, 4096, 4096),
    "attn_kv": (73728, 1024, 4096),
}

ATTN_SHAPES_70B = {
    "attn_qo": (81920, 8192, 8192),
    "attn_kv": (81920, 1024, 8192),
}

WARMUP = 5
ITERS = 20

FP8_MAX = 448.0


def tflops(M, N, K, time_s):
    return 2 * M * N * K / time_s / 1e12


def bench(fn, *args):
    for _ in range(WARMUP):
        out = fn(*args)
        if isinstance(out, tuple):
            out[0].block_until_ready()
        else:
            out.block_until_ready()
    jax.effects_barrier()
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        out = fn(*args)
        if isinstance(out, tuple):
            out[0].block_until_ready()
        else:
            out.block_until_ready()
        times.append(time.perf_counter() - t0)
    return np.median(times), np.std(times)


def run_db_shape(name, M, N, K):
    """Benchmark dB shape: dB[M,N] = grad_out[K,M]^T @ input[K,N]

    In production:
      grad_out is [batch*seq, M_proj] and input is [batch*seq, N_hidden].
      dB = grad_out^T @ input  => shape [M_proj, N_hidden].
      So K=batch*seq, M=N_proj, N=K_hidden.
    """
    flop = 2 * M * N * K
    print(f"\n{'='*90}")
    print(f"  {name}  M={M}  N={N}  K={K}  [{flop/1e12:.2f} TFLOP]")
    print(f"  dB[{M},{N}] = grad_out[{K},{M}]^T @ input[{K},{N}]")
    print(f"{'='*90}")
    print(f"  {'Backend':<55s}  {'ms':>8s}  {'TFLOP/s':>8s}  {'vs FP8':>7s}")
    print(f"  {'-'*55}  {'-'*8}  {'-'*8}  {'-'*7}")

    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    grad_out = jax.random.normal(k1, (K, M), dtype=jnp.bfloat16) * 0.01
    inp = jax.random.normal(k2, (K, N), dtype=jnp.bfloat16) * 0.01

    results = {}
    fp8_tf = None

    def report(label, tag, med):
        nonlocal fp8_tf
        tf = tflops(M, N, K, med)
        results[tag] = {"time_ms": med * 1000, "tflops": tf}
        vs = ""
        if fp8_tf is not None:
            vs = f"{tf / fp8_tf:.2f}x"
        print(f"  {label:<55s}  {med*1000:>7.2f}  {tf:>7.0f}T  {vs:>7s}")

    # -- hipBLASLt BF16: dB = grad_out^T @ input --
    @jax.jit
    def _bf16_db(g, x):
        return jax.lax.dot_general(g, x, (((0,), (0,)), ((), ())),
                                   preferred_element_type=jnp.bfloat16)
    med, _ = bench(_bf16_db, grad_out, inp)
    report("hipBLASLt BF16 (dot_general grad^T @ input)", "hb_bf16", med)

    # -- hipBLASLt FP8 pipeline (current production dB path) --
    @jax.jit
    def _fp8_db(g, x):
        eps = jnp.finfo(jnp.float32).tiny
        sg = jnp.float32(FP8_MAX) / (jnp.max(jnp.abs(g)) + eps)
        sx = jnp.float32(FP8_MAX) / (jnp.max(jnp.abs(x)) + eps)
        gq = (g * sg).astype(jnp.float8_e4m3fn)
        xq = (x * sx).astype(jnp.float8_e4m3fn)
        out = jax.lax.dot_general(
            gq, xq, (((0,), (0,)), ((), ())),
            preferred_element_type=jnp.bfloat16)
        return (out * (jnp.float32(1.0) / (sg * sx))).astype(jnp.bfloat16)
    try:
        med, _ = bench(_fp8_db, grad_out, inp)
        fp8_tf = tflops(M, N, K, med)
        report("hipBLASLt FP8 pipeline (current dB path)", "hb_fp8", med)
    except Exception as e:
        print(f"  hipBLASLt FP8 pipeline: SKIP ({e})")

    # -- FP4 kernel only (pre-quantized, no quant overhead) --
    # Simulates: A = grad_out^T quantized rowwise, B = input columnwise pre-shuffled
    try:
        from jax_aiter.gemm_fp4 import gemm_fp4
        from jax_aiter.gemm_fp4.fp4_utils import bf16_to_mxfp4, e8m0_shuffle, shuffle_weight

        grad_out_t = jnp.transpose(grad_out, (1, 0)).copy()
        a_p, a_s = bf16_to_mxfp4(grad_out_t)
        b_p, b_s = bf16_to_mxfp4(inp.T.copy())
        b_p_sh = shuffle_weight(b_p)
        a_s_sh = e8m0_shuffle(a_s)
        b_s_sh = e8m0_shuffle(b_s)

        med, _ = bench(jax.jit(gemm_fp4), a_p, b_p_sh, a_s_sh, b_s_sh)
        report("FP4 kernel only (pre-quantized, TE wgrad style)", "fp4_kern", med)
    except Exception as e:
        print(f"  FP4 kernel only: SKIP ({e})")

    # -- FP4 full pipeline: quant grad_out^T + saved col_input + FP4 GEMM --
    # This is what the production path would look like:
    #   Forward saves: col_input (from CastMxfp4DualJA)
    #   Backward: quant(grad_out^T) + FP4 GEMM(quant_grad_out_t, col_input)
    try:
        from jax_aiter.gemm_fp4.gemm_fp4 import _cast_mxfp4_fused_impl, _gemm_fp4_ffi
        from jax_aiter.ffi.registry import register_ffi_target
        register_ffi_target("CastMxfp4JA", "ROCM")
        register_ffi_target("CastMxfp4DualJA", "ROCM")
        register_ffi_target("GemmFp4FwdJA", "ROCM")

        # Pre-compute col_input (simulates forward's CastMxfp4DualJA on activation)
        from jax_aiter.gemm_fp4.gemm_fp4 import _cast_mxfp4_dual_impl
        _, _, col_inp_fp4, col_inp_scale = jax.jit(
            lambda x: _cast_mxfp4_dual_impl(x, shuffle_fp4=False)
        )(inp)

        @jax.jit
        def _fp4_db_pipe(g_t, col_b_fp4, col_b_scale):
            a_p, a_sc = _cast_mxfp4_fused_impl(g_t, shuffle_fp4=False)
            return _gemm_fp4_ffi(a_p, col_b_fp4, a_sc, col_b_scale)

        grad_out_t_c = jnp.transpose(grad_out, (1, 0)).copy()
        med, _ = bench(_fp4_db_pipe, grad_out_t_c, col_inp_fp4, col_inp_scale)
        report("FP4 fused pipeline (quant grad^T + saved col_inp)", "fp4_fused", med)
    except Exception as e:
        import traceback
        print(f"  FP4 fused pipeline: SKIP ({e})")
        traceback.print_exc()

    # -- FP4 full pipeline including transpose --
    try:
        @jax.jit
        def _fp4_db_pipe_full(g, col_b_fp4, col_b_scale):
            g_t = jnp.transpose(g, (1, 0))
            a_p, a_sc = _cast_mxfp4_fused_impl(g_t, shuffle_fp4=False)
            return _gemm_fp4_ffi(a_p, col_b_fp4, a_sc, col_b_scale)

        med, _ = bench(_fp4_db_pipe_full, grad_out, col_inp_fp4, col_inp_scale)
        report("FP4 fused pipeline (with transpose)", "fp4_fused_t", med)
    except Exception as e:
        print(f"  FP4 fused pipeline (with transpose): SKIP ({e})")

    return results


def run_attn_shape(name, M, N, K):
    """Benchmark attention projection shapes with FP4."""
    flop = 2 * M * N * K
    print(f"\n{'='*90}")
    print(f"  {name}  M={M}  N={N}  K={K}  [{flop/1e12:.2f} TFLOP]")
    print(f"{'='*90}")
    print(f"  {'Backend':<55s}  {'ms':>8s}  {'TFLOP/s':>8s}  {'vs FP8':>7s}")
    print(f"  {'-'*55}  {'-'*8}  {'-'*8}  {'-'*7}")

    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    a = jax.random.normal(k1, (M, K), dtype=jnp.bfloat16) * 0.01
    b = jax.random.normal(k2, (N, K), dtype=jnp.bfloat16) * 0.01

    results = {}
    fp8_tf = None

    def report(label, tag, med):
        nonlocal fp8_tf
        tf = tflops(M, N, K, med)
        results[tag] = {"time_ms": med * 1000, "tflops": tf}
        vs = ""
        if fp8_tf is not None:
            vs = f"{tf / fp8_tf:.2f}x"
        print(f"  {label:<55s}  {med*1000:>7.2f}  {tf:>7.0f}T  {vs:>7s}")

    # -- hipBLASLt BF16 --
    @jax.jit
    def _bf16(a, b):
        return jax.lax.dot_general(a, b, (((1,), (1,)), ((), ())))
    med, _ = bench(_bf16, a, b)
    report("hipBLASLt BF16", "hb_bf16", med)

    # -- hipBLASLt FP8 pipeline --
    @jax.jit
    def _fp8(a, b):
        eps = jnp.finfo(jnp.float32).tiny
        sa = jnp.float32(FP8_MAX) / (jnp.max(jnp.abs(a)) + eps)
        sb = jnp.float32(FP8_MAX) / (jnp.max(jnp.abs(b)) + eps)
        aq = (a * sa).astype(jnp.float8_e4m3fn)
        bq = (b * sb).astype(jnp.float8_e4m3fn)
        out = jax.lax.dot_general(
            aq, bq, (((1,), (1,)), ((), ())),
            preferred_element_type=jnp.bfloat16)
        return out * (jnp.float32(1.0) / (sa * sb))
    try:
        med, _ = bench(_fp8, a, b)
        fp8_tf = tflops(M, N, K, med)
        report("hipBLASLt FP8 pipeline", "hb_fp8", med)
    except Exception as e:
        print(f"  hipBLASLt FP8 pipeline: SKIP ({e})")

    # -- AITER BF16 ASM (current attention path) --
    try:
        from jax_aiter.gemm import gemm as aiter_gemm
        med, _ = bench(jax.jit(aiter_gemm), a, b)
        report("AITER BF16 ASM (current attn path)", "aiter_bf16", med)
    except Exception as e:
        print(f"  AITER BF16 ASM: SKIP ({e})")

    # -- FP4 kernel only (pre-quantized) --
    try:
        from jax_aiter.gemm_fp4 import gemm_fp4
        from jax_aiter.gemm_fp4.fp4_utils import bf16_to_mxfp4, e8m0_shuffle, shuffle_weight
        ap, a_s = bf16_to_mxfp4(a)
        bp, b_s = bf16_to_mxfp4(b)
        bp_sh = shuffle_weight(bp)
        as_sh = e8m0_shuffle(a_s)
        bs_sh = e8m0_shuffle(b_s)
        med, _ = bench(jax.jit(gemm_fp4), ap, bp_sh, as_sh, bs_sh)
        report("FP4 kernel only (pre-quantized)", "fp4_kern", med)
    except Exception as e:
        print(f"  FP4 kernel only: SKIP ({e})")

    # -- FP4 full pipeline (fused quant + GEMM) --
    try:
        from jax_aiter.gemm_fp4.gemm_fp4 import _cast_mxfp4_fused_impl, _gemm_fp4_ffi
        from jax_aiter.ffi.registry import register_ffi_target
        register_ffi_target("CastMxfp4JA", "ROCM")
        register_ffi_target("GemmFp4FwdJA", "ROCM")

        @jax.jit
        def _fp4_pipe(a, b):
            a_p, a_sc = _cast_mxfp4_fused_impl(a, shuffle_fp4=False)
            b_p, b_sc = _cast_mxfp4_fused_impl(b, shuffle_fp4=True)
            return _gemm_fp4_ffi(a_p, b_p, a_sc, b_sc)
        med, _ = bench(_fp4_pipe, a, b)
        report("FP4 fused pipeline (CastMxfp4 + FP4 ASM)", "fp4_fused", med)
    except Exception as e:
        print(f"  FP4 fused pipeline: SKIP ({e})")

    return results


def print_db_summary(all_results, model_name):
    print(f"\n{'='*90}")
    print(f"  dB SUMMARY — {model_name}")
    print(f"{'='*90}")

    tags = ["hb_bf16", "hb_fp8", "fp4_kern", "fp4_fused", "fp4_fused_t"]
    labels = ["BF16", "FP8 pipe", "FP4 kern", "FP4 fused", "FP4+transpose"]
    print(f"\n  {'Shape':<20s}", end="")
    for lbl in labels:
        print(f"  {lbl:>12s}", end="")
    print()
    print(f"  {'-'*20}" + f"  {'-'*12}" * len(labels))

    for sname, res in all_results.items():
        row = f"  {sname:<20s}"
        for tag in tags:
            if tag in res:
                row += f"  {res[tag]['tflops']:>10.0f}T"
            else:
                row += f"  {'--':>12s}"
        print(row)

    # Decision: is FP4 dB viable?
    print(f"\n  --- Viability Assessment ---")
    for sname, res in all_results.items():
        fp8_t = res.get("hb_fp8", {}).get("tflops", 0)
        fp4k_t = res.get("fp4_kern", {}).get("tflops", 0)
        fp4f_t = res.get("fp4_fused", {}).get("tflops", 0)
        if fp8_t > 0 and fp4k_t > 0:
            print(f"  {sname}: FP4_kern/FP8 = {fp4k_t/fp8_t:.2f}x", end="")
            if fp4f_t > 0:
                print(f"  FP4_fused/FP8 = {fp4f_t/fp8_t:.2f}x", end="")
            verdict = "VIABLE" if fp4k_t / fp8_t > 0.85 else "MARGINAL" if fp4k_t / fp8_t > 0.7 else "NOT VIABLE"
            print(f"  [{verdict}]")


def print_attn_summary(all_results, model_name):
    print(f"\n{'='*90}")
    print(f"  ATTENTION SUMMARY — {model_name}")
    print(f"{'='*90}")

    tags = ["hb_bf16", "hb_fp8", "aiter_bf16", "fp4_kern", "fp4_fused"]
    labels = ["BF16", "FP8 pipe", "AITER BF16", "FP4 kern", "FP4 fused"]
    print(f"\n  {'Shape':<20s}", end="")
    for lbl in labels:
        print(f"  {lbl:>12s}", end="")
    print()
    print(f"  {'-'*20}" + f"  {'-'*12}" * len(labels))

    for sname, res in all_results.items():
        row = f"  {sname:<20s}"
        for tag in tags:
            if tag in res:
                row += f"  {res[tag]['tflops']:>10.0f}T"
            else:
                row += f"  {'--':>12s}"
        print(row)

    print(f"\n  --- Viability Assessment ---")
    for sname, res in all_results.items():
        fp8_t = res.get("hb_fp8", {}).get("tflops", 0)
        aiter_t = res.get("aiter_bf16", {}).get("tflops", 0)
        fp4k_t = res.get("fp4_kern", {}).get("tflops", 0)
        fp4f_t = res.get("fp4_fused", {}).get("tflops", 0)
        if fp8_t > 0:
            parts = [f"  {sname}:"]
            if aiter_t > 0:
                parts.append(f"AITER_BF16/FP8 = {aiter_t/fp8_t:.2f}x")
            if fp4k_t > 0:
                parts.append(f"FP4_kern/FP8 = {fp4k_t/fp8_t:.2f}x")
            if fp4f_t > 0:
                parts.append(f"FP4_fused/FP8 = {fp4f_t/fp8_t:.2f}x")
            # FP4 attention viable if it beats AITER BF16 (the current path)
            if fp4f_t > 0 and aiter_t > 0:
                if fp4f_t > aiter_t:
                    parts.append("[FP4 > AITER BF16, consider switching]")
                else:
                    parts.append("[FP4 < AITER BF16, keep BF16]")
            print("  ".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["8b", "70b", "both"], default="both")
    args = parser.parse_args()

    print(f"JAX {jax.__version__}  |  Device: {jax.devices()[0].device_kind}")
    print(f"Benchmark: {WARMUP} warmup + {ITERS} iters (median)")
    print(f"\nPurpose: Determine viability of FP4 for dB (wgrad) and attention shapes")
    print(f"  dB:   Can FP4 GEMM replace hipBLASLt FP8 for weight gradients?")
    print(f"  Attn: Can FP4 GEMM replace AITER BF16 ASM for attention projections?")

    models = []
    if args.model in ("8b", "both"):
        models.append(("Llama3.1-8B", DB_SHAPES_8B, ATTN_SHAPES_8B))
    if args.model in ("70b", "both"):
        models.append(("Llama3.3-70B", DB_SHAPES_70B, ATTN_SHAPES_70B))

    for model_name, db_shapes, attn_shapes in models:
        print(f"\n{'#'*90}")
        print(f"  {model_name}")
        print(f"{'#'*90}")

        print(f"\n  --- dB (wgrad) shapes ---")
        db_results = {}
        for name, (M, N, K) in db_shapes.items():
            db_results[name] = run_db_shape(name, M, N, K)
        print_db_summary(db_results, model_name)

        print(f"\n  --- Attention shapes ---")
        attn_results = {}
        for name, (M, N, K) in attn_shapes.items():
            attn_results[name] = run_attn_shape(name, M, N, K)
        print_attn_summary(attn_results, model_name)

    print()


if __name__ == "__main__":
    main()
