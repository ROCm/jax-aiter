#!/usr/bin/env python3
"""FP4 vs FP8 kernel comparison at 8B and 70B production shapes.

Investigates why MXFP4 beats FP8 at 8B (+1.5%) but loses at 70B (-7.6%).
Benchmarks the exact kernel pipelines used in each E2E training path.

  MXFP4 MLP fwd/dA:   CastMxfp4JA (fused quant) + FP4 ASM GEMM
  MXFP4 attn fwd/dA:  AITER BF16 ASM GEMM
  FP8 all:             hipBLASLt FP8 pipeline (cast + dot_general + scale)

Shapes are production shapes (8-way FSDP):
  8B:  M=73728 (batch=9*seq=8192), hidden=4096, intermediate=14336
  70B: M=81920 (batch=10*seq=8192), hidden=8192, intermediate=28672

Usage (inside container, single GPU):
  cd /ruvaidya/aiter_proj/jax-aiter && \\
    JA_ROOT_DIR=$PWD AITER_ASM_DIR=$PWD/third_party/aiter/hsa/ \\
    AITER_SYMBOL_VISIBLE=1 GPU_ARCHS=gfx950 \\
    HIP_VISIBLE_DEVICES=0 \\
    python3 benchmarks/bench_fp4_vs_fp8_70b.py [--model 8b|70b|both]
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


SHAPES_8B = {
    "mlp_gate_up_fwd": (73728, 14336, 4096),
    "mlp_fused_fwd":   (73728, 28672, 4096),   # fused_mlp: gate+up combined (N=2*14336)
    "mlp_down_fwd":    (73728, 4096,  14336),
    "attn_qo_fwd":     (73728, 4096,  4096),
    "attn_kv_fwd":     (73728, 1024,  4096),
    "mlp_gate_up_dB":  (14336, 4096,  73728),
    "mlp_down_dB":     (4096,  14336, 73728),
}

SHAPES_70B = {
    "mlp_gate_up_fwd": (81920, 28672, 8192),
    "mlp_fused_fwd":   (81920, 57344, 8192),   # fused_mlp: gate+up combined (N=2*28672)
    "mlp_down_fwd":    (81920, 8192,  28672),
    "attn_qo_fwd":     (81920, 8192,  8192),
    "attn_kv_fwd":     (81920, 1024,  8192),
    "mlp_gate_up_dB":  (28672, 8192,  81920),
    "mlp_down_dB":     (8192,  28672, 81920),
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


def run_shape(name, M, N, K):
    flop = 2 * M * N * K
    is_mlp_fwd = "mlp" in name and "dB" not in name

    print(f"\n{'='*85}")
    print(f"  {name}  M={M}  N={N}  K={K}  [{flop/1e12:.2f} TFLOP]")
    print(f"{'='*85}")
    print(f"  {'Backend':<48s}  {'ms':>8s}  {'TFLOP/s':>8s}  {'vs FP8':>7s}")
    print(f"  {'-'*48}  {'-'*8}  {'-'*8}  {'-'*7}")

    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    a = jax.random.normal(k1, (M, K), dtype=jnp.bfloat16) * 0.01
    b = jax.random.normal(k2, (N, K), dtype=jnp.bfloat16) * 0.01

    results = {}
    fp8_pipe_tf = None

    def report(label, tag, med):
        nonlocal fp8_pipe_tf
        tf = tflops(M, N, K, med)
        results[tag] = {"time_ms": med * 1000, "tflops": tf}
        vs = ""
        if fp8_pipe_tf is not None:
            vs = f"{tf / fp8_pipe_tf:.2f}x"
        print(f"  {label:<48s}  {med*1000:>7.2f}  {tf:>7.0f}T  {vs:>7s}")

    # -- hipBLASLt BF16 --
    @jax.jit
    def _bf16(a, b):
        return jax.lax.dot_general(a, b, (((1,), (1,)), ((), ())))
    med, _ = bench(_bf16, a, b)
    report("hipBLASLt BF16", "hb_bf16", med)

    # -- hipBLASLt FP8 pipeline (production FP8 path) --
    @jax.jit
    def _fp8_pipe(a, b):
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
        med, _ = bench(_fp8_pipe, a, b)
        fp8_pipe_tf = tflops(M, N, K, med)
        report("hipBLASLt FP8 pipeline (production path)", "hb_fp8_pipe", med)
    except Exception as e:
        print(f"  hipBLASLt FP8 pipeline: SKIP ({e})")

    # -- AITER BF16 ASM --
    try:
        from jax_aiter.gemm import gemm as aiter_gemm
        med, _ = bench(jax.jit(aiter_gemm), a, b)
        report("AITER BF16 ASM (MXFP4 attention path)", "aiter_bf16", med)
    except Exception as e:
        print(f"  AITER BF16 ASM: SKIP ({e})")

    if is_mlp_fwd:
        # -- FP4 full pipeline via gemm_fp4_bf16 (JAX quant, AITER_FUSED_QUANT=0) --
        try:
            from jax_aiter.gemm_fp4 import gemm_fp4_bf16
            med, _ = bench(jax.jit(gemm_fp4_bf16), a, b)
            report("FP4 pipeline (JAX quant + FP4 ASM)", "fp4_jax_pipe", med)
        except Exception as e:
            print(f"  FP4 pipeline: SKIP ({e})")

        # -- FP4 kernel only (pre-quantized, no quant overhead) --
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

        # -- Fused quant kernel only (CastMxfp4JA, bypass custom_partitioning) --
        try:
            from jax_aiter.gemm_fp4.gemm_fp4 import _cast_mxfp4_fused_impl
            from jax_aiter.ffi.registry import register_ffi_target
            register_ffi_target("CastMxfp4JA", "ROCM")

            @jax.jit
            def _fused_quant(x):
                return _cast_mxfp4_fused_impl(x, shuffle_fp4=False)
            med, _ = bench(_fused_quant, a)
            report("Fused quant only (CastMxfp4JA)", "fq_only", med)
        except Exception as e:
            print(f"  Fused quant only: SKIP ({e})")

        # -- Full fused pipeline: fused quant + FP4 GEMM (bypass custom_partitioning) --
        try:
            from jax_aiter.gemm_fp4.gemm_fp4 import _cast_mxfp4_fused_impl, _gemm_fp4_ffi
            from jax_aiter.ffi.registry import register_ffi_target
            register_ffi_target("CastMxfp4JA", "ROCM")
            register_ffi_target("GemmFp4FwdJA", "ROCM")

            @jax.jit
            def _fused_pipe(a, b):
                a_p, a_sc = _cast_mxfp4_fused_impl(a, shuffle_fp4=False)
                b_p, b_sc = _cast_mxfp4_fused_impl(b, shuffle_fp4=True)
                return _gemm_fp4_ffi(a_p, b_p, a_sc, b_sc)
            med, _ = bench(_fused_pipe, a, b)
            report("FP4 fused pipeline (CastMxfp4+GEMM)", "fp4_fused_pipe", med)
        except Exception as e:
            print(f"  FP4 fused pipeline: SKIP ({e})")

    return results


def print_summary(all_results, shapes, model_name):
    print(f"\n{'='*85}")
    print(f"  SUMMARY — {model_name}")
    print(f"{'='*85}")

    tags = ["hb_bf16", "hb_fp8_pipe", "aiter_bf16", "fp4_fused_pipe", "fp4_kern"]
    labels = ["hBLAS BF16", "FP8 pipe", "AITER BF16", "FP4 fused", "FP4 kern"]
    print(f"\n  {'Shape':<20s}", end="")
    for lbl in labels:
        print(f"  {lbl:>10s}", end="")
    print()
    print(f"  {'-'*20}" + f"  {'-'*10}" * len(labels))

    for name, res in all_results.items():
        row = f"  {name:<20s}"
        for tag in tags:
            if tag in res:
                row += f"  {res[tag]['tflops']:>8.0f}T"
            else:
                row += f"  {'--':>10s}"
        print(row)

    # Per-projection time estimate
    print(f"\n  --- Per-projection forward time (single layer) ---")
    print(f"  {'Projection':<22s} {'Cnt':>3s}  {'MXFP4 ms':>9s}  {'FP8 ms':>8s}  "
          f"{'Delta':>8s}  {'MXFP4 backend':>18s}")
    print(f"  {'-'*22} {'-'*3}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*18}")

    mxfp4_tot = 0.0
    fp8_tot = 0.0

    # Separate MLP path (gate_up x2 + down x1 + attn)
    fwd_shapes_separate = ["mlp_gate_up_fwd", "mlp_down_fwd", "attn_qo_fwd", "attn_kv_fwd"]
    for name in fwd_shapes_separate:
        if name not in all_results:
            continue
        res = all_results[name]
        cnt = 2 if ("gate" in name or "kv" in name or "qo" in name) else 1

        if "mlp" in name:
            mxfp4_tag = "fp4_fused_pipe" if "fp4_fused_pipe" in res else "fp4_jax_pipe"
            backend = "FP4 fused" if "fp4_fused_pipe" in res else "FP4 JAX quant"
        else:
            mxfp4_tag = "aiter_bf16"
            backend = "AITER BF16 ASM"

        mxfp4_ms = res.get(mxfp4_tag, {}).get("time_ms", 0)
        fp8_ms = res.get("hb_fp8_pipe", {}).get("time_ms", 0)
        if mxfp4_ms > 0 and fp8_ms > 0:
            mt = mxfp4_ms * cnt
            ft = fp8_ms * cnt
            mxfp4_tot += mt
            fp8_tot += ft
            print(f"  {name:<22s} x{cnt:>1d}  {mt:>8.2f}  {ft:>8.2f}  "
                  f"{mt - ft:>+7.2f}  {backend:>18s}")

    if mxfp4_tot > 0 and fp8_tot > 0:
        r = mxfp4_tot / fp8_tot
        print(f"\n  Forward total (separate): MXFP4={mxfp4_tot:.1f}ms  FP8={fp8_tot:.1f}ms  "
              f"ratio={r:.3f}  ({(r-1)*100:+.1f}%)")

    # Fused MLP path comparison (mlp_fused x1 replaces gate_up x2)
    if "mlp_fused_fwd" in all_results:
        fused_res = all_results["mlp_fused_fwd"]
        fused_tag = "fp4_fused_pipe" if "fp4_fused_pipe" in fused_res else "fp4_jax_pipe"
        fused_fp4_ms = fused_res.get(fused_tag, {}).get("time_ms", 0)
        fused_fp8_ms = fused_res.get("hb_fp8_pipe", {}).get("time_ms", 0)
        sep_res = all_results.get("mlp_gate_up_fwd", {})
        sep_tag = "fp4_fused_pipe" if "fp4_fused_pipe" in sep_res else "fp4_jax_pipe"
        sep_fp4_ms = sep_res.get(sep_tag, {}).get("time_ms", 0)
        sep_fp8_ms = sep_res.get("hb_fp8_pipe", {}).get("time_ms", 0)

        if fused_fp4_ms > 0 and sep_fp4_ms > 0:
            print(f"\n  --- fused_mlp comparison (gate+up combined) ---")
            print(f"  Separate: 2 x gate_up  FP4={2*sep_fp4_ms:.2f}ms  FP8={2*sep_fp8_ms:.2f}ms")
            print(f"  Fused:    1 x fused    FP4={fused_fp4_ms:.2f}ms  FP8={fused_fp8_ms:.2f}ms")
            fp4_saving = 2*sep_fp4_ms - fused_fp4_ms
            fp8_saving = 2*sep_fp8_ms - fused_fp8_ms
            print(f"  FP4 saving: {fp4_saving:+.2f}ms ({fp4_saving/(2*sep_fp4_ms)*100:+.1f}%)")
            print(f"  FP8 saving: {fp8_saving:+.2f}ms ({fp8_saving/(2*sep_fp8_ms)*100:+.1f}%)")
            if fused_fp8_ms > 0:
                print(f"  Fused FP4/FP8 ratio: {fused_fp4_ms/fused_fp8_ms:.3f}x")

            # Fused MLP forward total (fused gate_up + down + attn)
            fused_mxfp4_tot = fused_fp4_ms
            fused_fp8_tot = fused_fp8_ms
            for name in ["mlp_down_fwd", "attn_qo_fwd", "attn_kv_fwd"]:
                if name not in all_results:
                    continue
                res = all_results[name]
                cnt = 2 if ("kv" in name or "qo" in name) else 1
                if "mlp" in name:
                    t = "fp4_fused_pipe" if "fp4_fused_pipe" in res else "fp4_jax_pipe"
                else:
                    t = "aiter_bf16"
                m = res.get(t, {}).get("time_ms", 0)
                f = res.get("hb_fp8_pipe", {}).get("time_ms", 0)
                fused_mxfp4_tot += m * cnt
                fused_fp8_tot += f * cnt
            if fused_mxfp4_tot > 0 and fused_fp8_tot > 0:
                r = fused_mxfp4_tot / fused_fp8_tot
                print(f"  Fused forward total: MXFP4={fused_mxfp4_tot:.1f}ms  FP8={fused_fp8_tot:.1f}ms  "
                      f"ratio={r:.3f}  ({(r-1)*100:+.1f}%)")

    # dB comparison
    db_shapes = ["mlp_gate_up_dB", "mlp_down_dB"]
    db_mxfp4 = 0.0
    db_fp8 = 0.0
    for name in db_shapes:
        if name not in all_results:
            continue
        res = all_results[name]
        cnt = 2 if "gate" in name else 1
        mxfp4_ms = res.get("hb_fp8_pipe", {}).get("time_ms", 0)
        fp8_ms = res.get("hb_fp8_pipe", {}).get("time_ms", 0)
        db_mxfp4 += mxfp4_ms * cnt
        db_fp8 += fp8_ms * cnt
    if db_mxfp4 > 0:
        print(f"  dB total (both use FP8 hipBLASLt): {db_mxfp4:.1f}ms (same for both paths)")

    if mxfp4_tot > 0 and fp8_tot > 0:
        print(f"\n  With full remat (70B): fwd counted 2x; dA has same shapes as fwd.")
        full_mxfp4 = mxfp4_tot * 3 + db_mxfp4
        full_fp8 = fp8_tot * 3 + db_fp8
        r2 = full_mxfp4 / full_fp8
        print(f"  Full layer GEMM: MXFP4={full_mxfp4:.1f}ms  FP8={full_fp8:.1f}ms  "
              f"ratio={r2:.3f}  ({(r2-1)*100:+.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["8b", "70b", "both"], default="both")
    args = parser.parse_args()

    print(f"JAX {jax.__version__}  |  Device: {jax.devices()[0].device_kind}")
    print(f"AITER_FUSED_QUANT={os.environ.get('AITER_FUSED_QUANT', '?')}")
    print(f"Benchmark: {WARMUP} warmup + {ITERS} iters (median)")

    models = []
    if args.model in ("8b", "both"):
        models.append(("Llama3.1-8B", SHAPES_8B))
    if args.model in ("70b", "both"):
        models.append(("Llama3.3-70B", SHAPES_70B))

    for model_name, shapes in models:
        print(f"\n{'#'*85}")
        print(f"  {model_name}")
        print(f"{'#'*85}")

        all_results = {}
        for name, (M, N, K) in shapes.items():
            all_results[name] = run_shape(name, M, N, K)
        print_summary(all_results, shapes, model_name)

    print()


if __name__ == "__main__":
    main()
