# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Smoke test for all GEMM variants. Skips variants not available on current GPU."""

import subprocess
import sys

import jax
import jax.numpy as jnp


def get_arch():
    try:
        r = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=10)
        for line in r.stdout.split("\n"):
            if "gfx9" in line and "Name:" in line:
                return line.split(":")[-1].strip()
    except Exception:
        pass
    return "unknown"


def smoke_bf16_gemm():
    from jax_aiter.gemm import gemm

    M, N, K = 256, 256, 256
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    a = jax.random.normal(k1, (M, K), dtype=jnp.bfloat16)
    b = jax.random.normal(k2, (N, K), dtype=jnp.bfloat16)

    out = gemm(a, b)
    ref = (a.astype(jnp.float32) @ b.astype(jnp.float32).T).astype(jnp.bfloat16)

    max_diff = float(jnp.max(jnp.abs(out.astype(jnp.float32) - ref.astype(jnp.float32))))
    max_ref = float(jnp.max(jnp.abs(ref.astype(jnp.float32))))
    rel_err = max_diff / max(max_ref, 1e-6)
    assert rel_err < 0.02, f"BF16 GEMM smoke FAILED: rel_err={rel_err}"
    print(f"  BF16 GEMM: PASSED (rel_err={rel_err:.6f})")


def smoke_fp8_mi350_gemm():
    from jax_aiter.gemm_fp8 import gemm_fp8_mi350

    M, N, K = 256, 256, 512
    key = jax.random.PRNGKey(1)
    k1, k2 = jax.random.split(key)
    xq = (jax.random.normal(k1, (M, K), dtype=jnp.float32) * 0.1).astype(jnp.float8_e4m3fn)
    wq = (jax.random.normal(k2, (N, K), dtype=jnp.float32) * 0.1).astype(jnp.float8_e4m3fn)
    x_scale = jnp.ones((K // 128, M), dtype=jnp.float32)
    w_scale = jnp.ones((K // 128, N // 128), dtype=jnp.float32)

    out = gemm_fp8_mi350(xq, wq, x_scale, w_scale)
    assert out.shape == (M, N)
    assert out.dtype == jnp.bfloat16
    assert jnp.all(jnp.isfinite(out)), "FP8 MI350 smoke: NaN/Inf in output"
    print(f"  FP8 MI350 GEMM: PASSED (shape={out.shape})")


def smoke_flatmm_fp8():
    from jax_aiter.flatmm_fp8 import flatmm_fp8

    M, N, K = 128, 256, 128
    key = jax.random.PRNGKey(2)
    k1, k2 = jax.random.split(key)
    xq = (jax.random.normal(k1, (M, K), dtype=jnp.float32) * 0.1).astype(jnp.float8_e4m3fn)
    wq = (jax.random.normal(k2, (N, K), dtype=jnp.float32) * 0.1).astype(jnp.float8_e4m3fn)
    x_scale = jnp.ones((K // 128, M), dtype=jnp.float32)
    w_scale = jnp.ones((K // 128, N // 128), dtype=jnp.float32)

    out = flatmm_fp8(xq, wq, x_scale, w_scale)
    assert out.shape == (M, N)
    assert out.dtype == jnp.float16
    assert jnp.all(jnp.isfinite(out)), "FlatMM FP8 smoke: NaN/Inf"
    print(f"  FlatMM FP8: PASSED (shape={out.shape})")


def smoke_i8_gemm():
    from jax_aiter.gemm_i8 import gemm_i8

    M, N, K = 128, 128, 128
    key = jax.random.PRNGKey(3)
    k1, k2 = jax.random.split(key)
    a = jax.random.randint(k1, (M, K), -128, 127, dtype=jnp.int8)
    b = jax.random.randint(k2, (N, K), -128, 127, dtype=jnp.int8)
    a_scale = jnp.ones((M, 1), dtype=jnp.float32) * 0.01
    b_scale = jnp.ones((1, N), dtype=jnp.float32) * 0.01
    bias = jnp.zeros((1, N), dtype=jnp.float32)

    out = gemm_i8(a, b, a_scale, b_scale, bias)
    assert out.shape == (M, N)
    assert out.dtype == jnp.bfloat16
    assert jnp.all(jnp.isfinite(out)), "INT8 GEMM smoke: NaN/Inf"
    print(f"  INT8 GEMM: PASSED (shape={out.shape})")


def main():
    arch = get_arch()
    print(f"GPU arch: {arch}")
    print(f"JAX devices: {jax.devices()}")
    print()

    passed = 0
    skipped = 0
    failed = 0

    # BF16 GEMM: gfx942 + gfx950
    try:
        smoke_bf16_gemm()
        passed += 1
    except Exception as e:
        print(f"  BF16 GEMM: FAILED ({e})")
        failed += 1

    # FP8 MI350: gfx950 only
    if arch == "gfx950":
        try:
            smoke_fp8_mi350_gemm()
            passed += 1
        except Exception as e:
            print(f"  FP8 MI350 GEMM: FAILED ({e})")
            failed += 1
    else:
        print(f"  FP8 MI350 GEMM: SKIPPED (requires gfx950, have {arch})")
        skipped += 1

    # FlatMM FP8: gfx942 only
    if arch == "gfx942":
        try:
            smoke_flatmm_fp8()
            passed += 1
        except Exception as e:
            print(f"  FlatMM FP8: FAILED ({e})")
            failed += 1
    else:
        print(f"  FlatMM FP8: SKIPPED (requires gfx942, have {arch})")
        skipped += 1

    # INT8 GEMM: gfx942 only
    if arch == "gfx942":
        try:
            smoke_i8_gemm()
            passed += 1
        except Exception as e:
            print(f"  INT8 GEMM: FAILED ({e})")
            failed += 1
    else:
        print(f"  INT8 GEMM: SKIPPED (requires gfx942, have {arch})")
        skipped += 1

    # FP4 GEMM: skipped (needs proper fp4x2 quantization)
    print(f"  FP4 GEMM: SKIPPED (needs fp4x2 quantization utilities)")
    skipped += 1

    print()
    print(f"GEMM smoke summary: {passed} passed, {skipped} skipped, {failed} failed")
    if failed > 0:
        sys.exit(1)
    print("All GEMM smoke tests PASSED")


if __name__ == "__main__":
    main()
