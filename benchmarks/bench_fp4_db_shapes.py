"""Microbench: FP4 vs hipBLASLt BF16 at production dB shapes.

dB[N,K] = grad_out^T[N,M] @ A[M,K]
FP4 path: grad_t = transpose(grad); a_t = transpose(a); db = gemm_fp4_bf16(grad_t, a_t)
hipBLASLt: db = lax.dot_general(grad, a, contract=(0,0))

Llama3-8B dB shapes (8-way FSDP, M=9216):
  gate/up dB: (14336, 4096, 9216) — N=14336, K=4096, contraction=M=9216
  down dB:    (4096, 14336, 9216) — N=4096, K=14336, contraction=M=9216

Usage: HIP_VISIBLE_DEVICES=0 python3 benchmarks/bench_fp4_db_shapes.py
"""

import time
import jax
import jax.numpy as jnp

DB_SHAPES = {
    "gate_up_dB": (14336, 4096, 9216),
    "down_dB":    (4096, 14336, 9216),
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
    return sum(times) / len(times)


def main():
    print("=" * 70)
    print("FP4 dB Feasibility Microbench — Production Shapes")
    print("=" * 70)

    for name, (N, K, M) in DB_SHAPES.items():
        print(f"\n--- {name}: dB[{N},{K}] = grad_t[{N},{M}] @ A[{M},{K}] ---")
        flop_MNK = M  # contraction dim for TFLOP calc
        total_flops = 2 * N * K * M

        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        grad_out = jax.random.normal(k1, (M, N), dtype=jnp.bfloat16)
        a = jax.random.normal(k2, (M, K), dtype=jnp.bfloat16)

        # --- hipBLASLt BF16 (current dB path) ---
        @jax.jit
        def hipblaslt_db(grad_out, a):
            return jax.lax.dot_general(grad_out, a, (((0,), (0,)), ((), ())))

        t_hip = bench(hipblaslt_db, grad_out, a)
        tf_hip = total_flops / t_hip / 1e12
        print(f"    hipBLASLt BF16:     {t_hip*1000:8.2f} ms  {tf_hip:8.1f} TFLOP/s")

        # --- FP4 (proposed dB): double transpose + gemm_fp4_bf16 ---
        try:
            from jax_aiter.gemm_fp4 import gemm_fp4_bf16

            grad_t = jnp.transpose(grad_out, (1, 0))
            a_t = jnp.transpose(a, (1, 0))

            @jax.jit
            def fp4_db(grad_t, a_t):
                return gemm_fp4_bf16(grad_t, a_t)

            t_fp4 = bench(fp4_db, grad_t, a_t)
            tf_fp4 = total_flops / t_fp4 / 1e12
            speedup = tf_fp4 / tf_hip
            print(f"    FP4 (pre-transposed): {t_fp4*1000:7.2f} ms  {tf_fp4:8.1f} TFLOP/s  ({speedup:.2f}x vs hipBLASLt)")
        except Exception as e:
            print(f"    FP4: SKIP ({e})")
            continue

        # --- FP4 with transposes included ---
        @jax.jit
        def fp4_db_full(grad_out, a):
            grad_t = jnp.transpose(grad_out, (1, 0))
            a_t = jnp.transpose(a, (1, 0))
            return gemm_fp4_bf16(grad_t, a_t)

        t_fp4f = bench(fp4_db_full, grad_out, a)
        tf_fp4f = total_flops / t_fp4f / 1e12
        speedup_f = tf_fp4f / tf_hip
        print(f"    FP4 (+2 transposes): {t_fp4f*1000:7.2f} ms  {tf_fp4f:8.1f} TFLOP/s  ({speedup_f:.2f}x vs hipBLASLt)")

    print("\n" + "=" * 70)
    print("Go/no-go: FP4 must beat hipBLASLt BF16 at both dB shapes")
    print("=" * 70)


if __name__ == "__main__":
    main()
