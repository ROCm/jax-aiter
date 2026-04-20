#!/usr/bin/env python3
"""Micro-benchmark: GEMM performance across all backends.

Tests kernel-level performance and scan serialization for:
  1. hipBLASLt (lax.dot_general) — XLA native
  2. AITER ASM GEMM (FFI) — NT layout
  3. Triton GEMM (jax-triton) — NN layout (if available)

Also tests scan behavior with different backend combinations.

Run with:
  HIP_VISIBLE_DEVICES=0 python3 benchmarks/bench_gemm_perf.py

Runs in ~1 minute on a single GPU.
"""

import os
import time
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
import jax.numpy as jnp
import numpy as np

# Llama3-8B shapes (per GPU, 8-way FSDP)
SHAPES = {
    "attn_qkvo": (9216, 4096, 4096),   # M, N, K
    "mlp_gate_up": (9216, 14336, 4096),
    "mlp_down": (9216, 4096, 14336),
    "small": (1024, 1024, 1024),
}


def tflops(M, N, K, time_s):
    """Compute TFLOP/s for a GEMM."""
    flops = 2 * M * N * K
    return flops / time_s / 1e12


def bench_fn(fn, *args, warmup=3, iters=10, label=""):
    """Benchmark a function, return avg time in ms."""
    # Warmup
    for _ in range(warmup):
        result = fn(*args)
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()

    # Timed
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        result = fn(*args)
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        end = time.perf_counter()
        times.append(end - start)

    avg_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    return avg_ms, std_ms


def print_header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_result(label, M, N, K, avg_ms, std_ms):
    tf = tflops(M, N, K, avg_ms / 1000)
    print(f"  {label:40s}  {avg_ms:8.2f} ± {std_ms:5.2f} ms  {tf:8.1f} TFLOP/s")


# ============================================================
# 1. Isolated GEMM benchmarks
# ============================================================

def bench_isolated_gemm():
    """Benchmark isolated GEMM operations (no scan)."""
    print_header("1. ISOLATED GEMM (single call, no scan)")

    for shape_name, (M, N, K) in SHAPES.items():
        print(f"\n  Shape: {shape_name} (M={M}, N={N}, K={K})")
        print(f"  {'-'*60}")

        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)

        # --- Forward: Out[M,N] = A[M,K] @ B[N,K]^T ---
        a = jax.random.normal(k1, (M, K), dtype=jnp.bfloat16)
        b = jax.random.normal(k2, (N, K), dtype=jnp.bfloat16)

        # hipBLASLt (NT layout via lax.dot_general)
        def hipblaslt_nt(a, b):
            return jax.lax.dot_general(a, b, (((1,), (1,)), ((), ())))

        avg, std = bench_fn(jax.jit(hipblaslt_nt), a, b)
        print_result("FWD hipBLASLt (dot_general NT)", M, N, K, avg, std)

        # AITER ASM (NT layout via FFI)
        try:
            from jax_aiter.gemm import gemm as aiter_gemm
            avg, std = bench_fn(jax.jit(aiter_gemm), a, b)
            print_result("FWD AITER ASM (FFI NT)", M, N, K, avg, std)
        except Exception as e:
            print(f"  FWD AITER ASM: SKIP ({e})")

        # --- dB: dB[N,K] = grad^T[N,M] @ A[M,K] (NN layout) ---
        grad = jax.random.normal(k1, (M, N), dtype=jnp.bfloat16)

        # hipBLASLt (NN layout via lax.dot_general)
        def hipblaslt_db(grad, a):
            return jax.lax.dot_general(grad, a, (((0,), (0,)), ((), ())))

        avg, std = bench_fn(jax.jit(hipblaslt_db), grad, a)
        print_result("dB hipBLASLt (dot_general NN)", N, K, M, avg, std)

        # Triton (NN layout via jax-triton)
        try:
            from jax_aiter.triton.gemm_triton import gemm_db_triton
            avg, std = bench_fn(jax.jit(gemm_db_triton), grad, a)
            print_result("dB Triton (jax-triton NN)", N, K, M, avg, std)
        except Exception as e:
            print(f"  dB Triton: SKIP ({e})")


# ============================================================
# 2. Full backward (fwd + bwd) benchmarks
# ============================================================

def bench_full_backward():
    """Benchmark full forward+backward with different dB backends."""
    print_header("2. FULL BACKWARD (fwd + grad, no scan)")

    for shape_name in ["small", "attn_qkvo"]:
        M, N, K = SHAPES[shape_name]
        print(f"\n  Shape: {shape_name} (M={M}, N={N}, K={K})")
        print(f"  {'-'*60}")

        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        a = jax.random.normal(k1, (M, K), dtype=jnp.bfloat16)
        b = jax.random.normal(k2, (N, K), dtype=jnp.bfloat16)

        # Pure hipBLASLt backward
        def fwd_bwd_hipblaslt(a, b):
            def f(a, b):
                return jnp.sum(jax.lax.dot_general(a, b, (((1,), (1,)), ((), ()))))
            return jax.grad(f, argnums=(0, 1))(a, b)

        avg, std = bench_fn(jax.jit(fwd_bwd_hipblaslt), a, b)
        print_result("FWD+BWD hipBLASLt only", M, N, K, avg, std)

        # AITER fwd + AITER dA + hipBLASLt dB (current default)
        try:
            from jax_aiter.gemm import gemm as aiter_gemm
            def fwd_bwd_aiter(a, b):
                def f(a, b):
                    return jnp.sum(aiter_gemm(a, b))
                return jax.grad(f, argnums=(0, 1))(a, b)

            avg, std = bench_fn(jax.jit(fwd_bwd_aiter), a, b)
            print_result("FWD+BWD AITER (ASM fwd+dA, hipBLASLt dB)", M, N, K, avg, std)
        except Exception as e:
            print(f"  FWD+BWD AITER: SKIP ({e})")


# ============================================================
# 3. Scan serialization tests
# ============================================================

def bench_scan():
    """Test if custom calls serialize inside jax.lax.scan."""
    print_header("3. SCAN SERIALIZATION (32 layers, like Llama3-8B)")

    M, N, K = 1024, 1024, 1024  # Smaller for faster scan test
    num_layers = 32

    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    a = jax.random.normal(k1, (M, K), dtype=jnp.bfloat16)
    b = jax.random.normal(k2, (N, K), dtype=jnp.bfloat16)

    # --- Scan with hipBLASLt only (baseline) ---
    def scan_hipblaslt(a, b):
        def body(carry, _):
            x = carry
            out = jax.lax.dot_general(x, b, (((1,), (1,)), ((), ())))
            # Simulate backward dB
            db = jax.lax.dot_general(out, x, (((0,), (0,)), ((), ())))
            return x + db * 0.001, None
        final, _ = jax.lax.scan(body, a, None, length=num_layers)
        return final

    avg, std = bench_fn(jax.jit(scan_hipblaslt), a, b, warmup=2, iters=5)
    print_result(f"SCAN {num_layers}x hipBLASLt only", M, N, K, avg, std)

    # --- Scan with 1 FFI per body (AITER fwd only) ---
    try:
        from jax_aiter.gemm import gemm as aiter_gemm
        def scan_1ffi(a, b):
            def body(carry, _):
                x = carry
                out = aiter_gemm(x, b)  # 1 FFI call
                db = jax.lax.dot_general(out, x, (((0,), (0,)), ((), ())))
                return x + db * 0.001, None
            final, _ = jax.lax.scan(body, a, None, length=num_layers)
            return final

        avg, std = bench_fn(jax.jit(scan_1ffi), a, b, warmup=2, iters=5)
        print_result(f"SCAN {num_layers}x 1-FFI (AITER fwd)", M, N, K, avg, std)
    except Exception as e:
        print(f"  SCAN 1-FFI: SKIP ({e})")

    # --- Scan with 2 FFI per body (AITER fwd + AITER dB) ---
    try:
        from jax_aiter.gemm import gemm as aiter_gemm
        def scan_2ffi(a, b):
            def body(carry, _):
                x = carry
                out = aiter_gemm(x, b)  # FFI #1
                # dB via pre-transpose + AITER ASM (FFI #2)
                out_t = jnp.transpose(out, (1, 0))
                x_t = jnp.transpose(x, (1, 0))
                db = aiter_gemm(out_t, x_t)
                return x + db * 0.001, None
            final, _ = jax.lax.scan(body, a, None, length=num_layers)
            return final

        avg, std = bench_fn(jax.jit(scan_2ffi), a, b, warmup=2, iters=5)
        print_result(f"SCAN {num_layers}x 2-FFI (AITER fwd+dB)", M, N, K, avg, std)
    except Exception as e:
        print(f"  SCAN 2-FFI: SKIP ({e})")

    # --- Scan with Triton dB ---
    try:
        from jax_aiter.triton.gemm_triton import gemm_triton
        def scan_triton(a, b):
            def body(carry, _):
                x = carry
                out = jax.lax.dot_general(x, b, (((1,), (1,)), ((), ())))
                # dB via Triton NN GEMM
                out_t = jnp.transpose(out, (1, 0))
                db = gemm_triton(out_t, x)
                return x + db * 0.001, None
            final, _ = jax.lax.scan(body, a, None, length=num_layers)
            return final

        avg, std = bench_fn(jax.jit(scan_triton), a, b, warmup=2, iters=5)
        print_result(f"SCAN {num_layers}x Triton dB", M, N, K, avg, std)
    except Exception as e:
        print(f"  SCAN Triton: SKIP ({e})")


# ============================================================
# 4. FP8 GEMM benchmarks
# ============================================================

def bench_fp8_gemm():
    """Benchmark AITER FP8 GEMM vs BF16 baselines."""
    print_header("4. FP8 GEMM (AITER CK block-scale FP8 vs BF16)")

    try:
        from jax_aiter.gemm_fp8 import gemm_fp8_mi350, fp8_supported_for_shape
    except Exception as e:
        print(f"  FP8 GEMM not available: {e}")
        return

    for shape_name, (M, N, K) in SHAPES.items():
        print(f"\n  Shape: {shape_name} (M={M}, N={N}, K={K})")
        print(f"  {'-'*60}")

        # Check FP8 shape support
        fp8_ok = fp8_supported_for_shape(M, N, K)
        print(f"  FP8 supported: {fp8_ok} (M≥16, K≥512, K%128==0)")

        if not fp8_ok:
            print(f"  SKIP: Shape not supported by FP8 kernel")
            continue

        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        a = jax.random.normal(k1, (M, K), dtype=jnp.bfloat16)
        b = jax.random.normal(k2, (N, K), dtype=jnp.bfloat16)

        # BF16 baselines for comparison
        def hipblaslt_nt(a, b):
            return jax.lax.dot_general(a, b, (((1,), (1,)), ((), ())))

        avg, std = bench_fn(jax.jit(hipblaslt_nt), a, b)
        print_result("BF16 hipBLASLt (dot_general NT)", M, N, K, avg, std)

        try:
            from jax_aiter.gemm import gemm as aiter_gemm
            avg, std = bench_fn(jax.jit(aiter_gemm), a, b)
            print_result("BF16 AITER ASM (FFI NT)", M, N, K, avg, std)
        except Exception as e:
            print(f"  BF16 AITER ASM: SKIP ({e})")

        # FP8 forward (per-call scaling)
        try:
            avg, std = bench_fn(jax.jit(gemm_fp8_mi350), a, b)
            print_result("FP8 AITER CK (block-scale, fwd only)", M, N, K, avg, std)
        except Exception as e:
            print(f"  FP8 AITER CK: SKIP ({e})")

        # FP8 forward + backward
        try:
            def fp8_fwd_bwd(a, b):
                def f(a, b):
                    return jnp.sum(gemm_fp8_mi350(a, b))
                return jax.grad(f, argnums=(0, 1))(a, b)

            avg, std = bench_fn(jax.jit(fp8_fwd_bwd), a, b)
            print_result("FP8 FWD+BWD (FP8 fwd, BF16 bwd)", M, N, K, avg, std)
        except Exception as e:
            print(f"  FP8 FWD+BWD: SKIP ({e})")

        # Raw FP8 GEMM kernel only (pre-quantized inputs, no quantization overhead)
        try:
            from jax_aiter.gemm_fp8.gemm_fp8_mi350 import _gemm_fp8_raw
            # Pre-quantize inputs manually to isolate kernel performance
            M_, K_ = a.shape
            N_ = b.shape[0]
            # Simple per-tensor quantization for benchmarking
            a_f32 = a.astype(jnp.float32)
            b_f32 = b.astype(jnp.float32)
            a_amax = jnp.max(jnp.abs(a_f32))
            b_amax = jnp.max(jnp.abs(b_f32))
            a_scale = jnp.maximum(a_amax, 1e-12) / 448.0
            b_scale = jnp.maximum(b_amax, 1e-12) / 448.0
            xq = jnp.clip(a_f32 / a_scale, -448, 448).astype(jnp.float8_e4m3fn)
            wq = jnp.clip(b_f32 / b_scale, -448, 448).astype(jnp.float8_e4m3fn)
            # Create scale arrays in expected layout
            x_scale_arr = jnp.ones((K_ // 128, M_), dtype=jnp.float32) * a_scale
            w_scale_arr = jnp.ones((N_ // 128, K_ // 128), dtype=jnp.float32) * b_scale

            avg, std = bench_fn(jax.jit(_gemm_fp8_raw), xq, wq, x_scale_arr, w_scale_arr)
            print_result("FP8 RAW KERNEL ONLY (pre-quantized)", M, N, K, avg, std)
        except Exception as e:
            print(f"  FP8 RAW KERNEL: SKIP ({e})")

    print(f"""
  Note: FP8 GEMM uses per-call scaling (quantize A,B to FP8 e4m3 with
  block-scale, compute in FP8, output in BF16). Backward uses BF16
  (STE pattern). Shape constraints: M≥16, K≥512, K%128==0, N padded to 256.
  
  "FP8 AITER CK" includes full quantization pipeline:
    padding → per-row amax → scale computation → BF16→FP8 cast → weight
    shuffle → scale array construction → FFI GEMM call
  
  "FP8 RAW KERNEL ONLY" calls just the FFI GEMM with pre-quantized
  inputs — this isolates the actual FP8 matrix multiply performance.
""")


# ============================================================
# 5. Triton deep-dive: why is it slow?
# ============================================================

def bench_triton_deepdive():
    """Deep analysis of Triton kernel performance issues."""
    print_header("5. TRITON DEEP-DIVE: Why is AITER's Triton kernel slow via jax-triton?")

    try:
        from jax_aiter.triton.gemm_triton import gemm_triton, _ensure_kernel_loaded
    except Exception as e:
        print(f"  Triton not available: {e}")
        return

    # Load kernel + config to inspect what config was selected
    kernel, get_config = _ensure_kernel_loaded()

    for shape_name in ["small", "attn_qkvo", "mlp_gate_up"]:
        M, N, K = SHAPES[shape_name]
        print(f"\n  Shape: {shape_name} (M={M}, N={N}, K={K})")
        print(f"  {'-'*60}")

        # Show what config AITER selects
        config, config_name = get_config(M, N, K)
        if config:
            print(f"  AITER config: {config_name}")
            print(f"    BLOCK_SIZE_M={config.get('BLOCK_SIZE_M')}, "
                  f"BLOCK_SIZE_N={config.get('BLOCK_SIZE_N')}, "
                  f"BLOCK_SIZE_K={config.get('BLOCK_SIZE_K')}")
            print(f"    NUM_KSPLIT={config.get('NUM_KSPLIT', 1)}, "
                  f"num_warps={config.get('num_warps')}, "
                  f"num_stages={config.get('num_stages')}")
        else:
            print(f"  AITER config: None (using fallback defaults)")

        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        a = jax.random.normal(k1, (M, K), dtype=jnp.bfloat16)
        b = jax.random.normal(k2, (K, N), dtype=jnp.bfloat16)  # NN layout

        # Triton NN GEMM (through jax-triton with input_output_aliases)
        avg, std = bench_fn(jax.jit(gemm_triton), a, b, warmup=3, iters=10)
        print_result("Triton NN (jax-triton + aliases)", M, N, K, avg, std)

        # hipBLASLt NN for comparison
        def hipblaslt_nn(a, b):
            return jax.lax.dot_general(a, b, (((1,), (0,)), ((), ())))
        avg, std = bench_fn(jax.jit(hipblaslt_nn), a, b, warmup=3, iters=10)
        print_result("hipBLASLt NN (dot_general)", M, N, K, avg, std)

        # Just the zero buffer allocation (to measure memcpy overhead)
        def just_zeros(a, b):
            return jnp.zeros((M, N), dtype=jnp.bfloat16)
        avg_z, std_z = bench_fn(jax.jit(just_zeros), a, b, warmup=3, iters=10)
        print_result("jnp.zeros buffer alloc only", M, N, K, avg_z, std_z)

    print(f"""
  Analysis: Why Triton kernel via jax-triton is slow
  ─────────────────────────────────────────────────────
  1. input_output_aliases penalty: jax-triton memcpy's the zero buffer
     (host→device) for every call. For large matrices (e.g. 9216×14336×2B
     ≈ 252MB), this is significant.

  2. Triton compiler quality for gfx950: The Triton→AMDGCN compilation
     path is less optimized than hand-tuned ASM or hipBLASLt. AMD's Triton
     backend is newer and produces less optimal GPU code.

  3. jax-triton uses FFI under the hood: triton_call() lowers to
     jax.ffi.ffi_lowering() — it's an opaque custom-call to XLA, same
     as our ASM FFI path. No XLA pipeline advantage over FFI.

  4. Kernel tile sizes may not be optimal: The config loaded from AITER
     was tuned for PyTorch's calling convention, not jax-triton's
     (different stride handling, buffer management).

  5. Per-call overhead: Each jax-triton call has compilation cache lookup,
     argument marshaling, and kernel dispatch overhead that native
     hipBLASLt avoids.
""")


if __name__ == "__main__":
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"Platform: {jax.default_backend()}")

    bench_isolated_gemm()
    bench_full_backward()
    bench_scan()
    bench_fp8_gemm()
    bench_triton_deepdive()

    print(f"\n{'='*70}")
    print("  DONE")
    print(f"{'='*70}")
