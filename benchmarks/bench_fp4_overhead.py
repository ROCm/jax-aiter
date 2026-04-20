"""FP4 overhead breakdown: quant vs kernel vs total at inference batch sizes.

Isolates where FP4 time goes by benchmarking each piece separately:
  1. Quant only:  bf16_to_mxfp4 + shuffles (no GEMM)
  2. Kernel only: gemm_fp4 with pre-quantized inputs (raw FFI, no quant)
  3. Full pipeline: gemm_fp4_bf16 (quant + shuffle + kernel)
  4. AITER BF16:   gemm (BF16 ASM reference)

Shape: 70B gate/up — A[M, K=8192] @ B[N=28672, K=8192]^T

Usage (inside container, single GPU):
  cd /ruvaidya/aiter_proj/jax-aiter && \
    JA_ROOT_DIR=$PWD AITER_ASM_DIR=$PWD/third_party/aiter/hsa/ \
    AITER_SYMBOL_VISIBLE=1 GPU_ARCHS=gfx950 \
    HIP_VISIBLE_DEVICES=0 \
    python3 benchmarks/bench_fp4_overhead.py
"""

import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
import jax.numpy as jnp
import numpy as np

N, K = 28672, 8192
M_VALUES = [1, 8, 32, 128, 512, 1024]

WARMUP = 5
ITERS = 30


def bench(fn, *args):
    for _ in range(WARMUP):
        out = fn(*args)
        if isinstance(out, tuple):
            out[0].block_until_ready()
        else:
            out.block_until_ready()
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        out = fn(*args)
        if isinstance(out, tuple):
            out[0].block_until_ready()
        else:
            out.block_until_ready()
        times.append(time.perf_counter() - t0)
    return np.median(times)


def main():
    from jax_aiter.gemm_fp4 import gemm_fp4, gemm_fp4_bf16, bf16_to_mxfp4, e8m0_shuffle, shuffle_weight
    from jax_aiter.gemm import gemm as aiter_gemm

    print(f"JAX {jax.__version__}  |  Devices: {jax.devices()}")
    print("=" * 95)
    print("  FP4 Overhead Breakdown — 70B gate/up: A[M, K=8192] @ B[N=28672, K=8192]^T")
    print("=" * 95)

    hdr = (f"  {'M':>6s}  {'Quant ms':>10s}  {'Kernel ms':>10s}  "
           f"{'Pipeline ms':>12s}  {'BF16 ms':>10s}  "
           f"{'Q%':>6s}  {'K%':>6s}  {'Pipe/BF16':>10s}")
    print(hdr)
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*6}  {'-'*6}  {'-'*10}")

    summary = []

    for M in M_VALUES:
        total_flops = 2 * M * N * K

        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        a = jax.random.normal(k1, (M, K), dtype=jnp.bfloat16) * 0.1
        b = jax.random.normal(k2, (N, K), dtype=jnp.bfloat16) * 0.1

        # Pre-quantize for kernel-only benchmark
        a_packed, a_scales = bf16_to_mxfp4(a)
        b_packed, b_scales = bf16_to_mxfp4(b)
        b_packed_sh = shuffle_weight(b_packed)
        a_scales_sh = e8m0_shuffle(a_scales)
        b_scales_sh = e8m0_shuffle(b_scales)

        # 1. Quant only (no GEMM)
        @jax.jit
        def quant_only(a, b):
            ap, asc = bf16_to_mxfp4(a)
            bp, bsc = bf16_to_mxfp4(b)
            bps = shuffle_weight(bp)
            ascs = e8m0_shuffle(asc)
            bscs = e8m0_shuffle(bsc)
            return ap, bps, ascs, bscs

        t_quant = bench(quant_only, a, b)

        # 2. Kernel only (pre-quantized)
        @jax.jit
        def kernel_only(ap, bp, asc, bsc):
            return gemm_fp4(ap, bp, asc, bsc)

        t_kernel = bench(kernel_only, a_packed, b_packed_sh, a_scales_sh, b_scales_sh)

        # 3. Full pipeline
        t_pipeline = bench(jax.jit(gemm_fp4_bf16), a, b)

        # 4. AITER BF16
        t_bf16 = bench(jax.jit(aiter_gemm), a, b)

        quant_pct = t_quant / t_pipeline * 100 if t_pipeline > 0 else 0
        kernel_pct = t_kernel / t_pipeline * 100 if t_pipeline > 0 else 0
        pipe_vs_bf16 = t_pipeline / t_bf16 if t_bf16 > 0 else 0

        print(f"  {M:>6d}  {t_quant*1000:>10.3f}  {t_kernel*1000:>10.3f}  "
              f"{t_pipeline*1000:>12.3f}  {t_bf16*1000:>10.3f}  "
              f"{quant_pct:>5.1f}%  {kernel_pct:>5.1f}%  {pipe_vs_bf16:>9.2f}x")

        tf_kernel = total_flops / t_kernel / 1e12 if t_kernel > 0 else 0
        tf_bf16 = total_flops / t_bf16 / 1e12 if t_bf16 > 0 else 0
        tf_pipe = total_flops / t_pipeline / 1e12 if t_pipeline > 0 else 0

        summary.append({
            "M": M, "t_quant": t_quant, "t_kernel": t_kernel,
            "t_pipeline": t_pipeline, "t_bf16": t_bf16,
            "quant_pct": quant_pct, "kernel_pct": kernel_pct,
            "tf_kernel": tf_kernel, "tf_bf16": tf_bf16, "tf_pipe": tf_pipe,
        })

    # TFLOP/s summary
    print(f"\n{'='*95}")
    print("  TFLOP/s COMPARISON (kernel speed without quant overhead)")
    print(f"{'='*95}")
    print(f"  {'M':>6s}  {'FP4 kernel':>12s}  {'FP4 pipeline':>13s}  "
          f"{'AITER BF16':>12s}  {'Kern/BF16':>10s}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*13}  {'-'*12}  {'-'*10}")
    for s in summary:
        ratio = s["tf_kernel"] / s["tf_bf16"] if s["tf_bf16"] > 0 else 0
        print(f"  {s['M']:>6d}  {s['tf_kernel']:>10.1f}Ts  {s['tf_pipe']:>11.1f}Ts  "
              f"{s['tf_bf16']:>10.1f}Ts  {ratio:>9.2f}x")

    print(f"\n{'='*95}")
    print("  DIAGNOSIS")
    print(f"{'='*95}")
    avg_quant_pct = np.mean([s["quant_pct"] for s in summary])
    print(f"  Average quant overhead: {avg_quant_pct:.1f}% of pipeline time")
    if avg_quant_pct > 50:
        print("  --> QUANT DOMINATED: Fused quant FFI will recover most of the gap")
    else:
        print("  --> KERNEL DOMINATED: FP4 ASM kernel itself is the bottleneck")
    print(f"{'='*95}")


if __name__ == "__main__":
    main()
