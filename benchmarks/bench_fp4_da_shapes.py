"""Microbench: FP4 vs hipBLASLt BF16 at production dA shapes.

Tests kernel-level speed for the dA backward path:
  dA[M,K] = grad_out[M,N] @ B[N,K]

FP4 path: b_t = transpose(b); da = gemm_fp4_bf16(grad_out, b_t)
hipBLASLt path: da = lax.dot_general(grad_out, b, contract=(1,0))

Llama3-8B shapes (8-way FSDP, M=batch*seq/gpus = 9*8192/8 = 9216):
  gate/up dA: (9216, 4096, 14336)  — grad[M,N] @ B[N,K], N=small K=large
  down dA:    (9216, 14336, 4096)  — grad[M,N] @ B[N,K], N=large K=small

Usage (inside container):
  HIP_VISIBLE_DEVICES=0 python3 benchmarks/bench_fp4_da_shapes.py
"""

import time
import jax
import jax.numpy as jnp

DA_SHAPES = {
    "gate_up_dA": (9216, 4096, 14336),
    "down_dA":    (9216, 14336, 4096),
}

WARMUP = 5
ITERS = 20


def tflops(M, N, K, time_s):
    return 2 * M * N * K / time_s / 1e12


def bench(fn, *args, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        out = fn(*args)
        out.block_until_ready()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(*args)
        out.block_until_ready()
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    return avg


def main():
    print("=" * 70)
    print("FP4 dA Feasibility Microbench — Production Shapes")
    print("=" * 70)

    for name, (M, N, K) in DA_SHAPES.items():
        print(f"\n--- {name}: M={M}, N={N}, K={K} ---")
        print(f"    dA[{M},{K}] = grad[{M},{N}] @ B[{N},{K}]")

        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        grad_out = jax.random.normal(k1, (M, N), dtype=jnp.bfloat16)
        b = jax.random.normal(k2, (N, K), dtype=jnp.bfloat16)

        # --- hipBLASLt BF16 (current dA path) ---
        @jax.jit
        def hipblaslt_da(grad_out, b):
            return jax.lax.dot_general(grad_out, b, (((1,), (0,)), ((), ())))

        t_hip = bench(hipblaslt_da, grad_out, b)
        tf_hip = tflops(M, N, K, t_hip)
        print(f"    hipBLASLt BF16:  {t_hip*1000:8.2f} ms  {tf_hip:8.1f} TFLOP/s")

        # --- FP4 (proposed dA path): transpose + gemm_fp4_bf16 ---
        try:
            from jax_aiter.gemm_fp4 import gemm_fp4_bf16

            b_t = jnp.transpose(b, (1, 0))

            @jax.jit
            def fp4_da(grad_out, b_t):
                return gemm_fp4_bf16(grad_out, b_t)

            t_fp4 = bench(fp4_da, grad_out, b_t)
            tf_fp4 = tflops(M, N, K, t_fp4)
            speedup = tf_fp4 / tf_hip
            print(f"    FP4 (quant+FFI): {t_fp4*1000:8.2f} ms  {tf_fp4:8.1f} TFLOP/s  ({speedup:.2f}x vs hipBLASLt)")
        except Exception as e:
            print(f"    FP4: SKIP ({e})")
            continue

        # --- FP4 transpose included in timing ---
        @jax.jit
        def fp4_da_with_transpose(grad_out, b):
            b_t = jnp.transpose(b, (1, 0))
            return gemm_fp4_bf16(grad_out, b_t)

        t_fp4t = bench(fp4_da_with_transpose, grad_out, b)
        tf_fp4t = tflops(M, N, K, t_fp4t)
        speedup_t = tf_fp4t / tf_hip
        print(f"    FP4 (+transpose):{t_fp4t*1000:8.2f} ms  {tf_fp4t:8.1f} TFLOP/s  ({speedup_t:.2f}x vs hipBLASLt)")

        # --- AITER BF16 ASM for reference ---
        try:
            from jax_aiter.gemm import gemm as aiter_gemm

            @jax.jit
            def aiter_bf16_da(grad_out, b_t):
                return aiter_gemm(grad_out, b_t)

            t_aiter = bench(aiter_bf16_da, grad_out, b_t)
            tf_aiter = tflops(M, N, K, t_aiter)
            speedup_a = tf_aiter / tf_hip
            print(f"    AITER BF16 ASM:  {t_aiter*1000:8.2f} ms  {tf_aiter:8.1f} TFLOP/s  ({speedup_a:.2f}x vs hipBLASLt)")
        except Exception as e:
            print(f"    AITER BF16: SKIP ({e})")

    print("\n" + "=" * 70)
    print("Go/no-go: FP4 must beat hipBLASLt BF16 at both shapes")
    print("=" * 70)


if __name__ == "__main__":
    main()
