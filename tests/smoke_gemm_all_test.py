# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Smoke test for all GEMM variants. Skips variants not available on current GPU."""

import sys

import jax
import jax.numpy as jnp


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


def smoke_fp4_gemm():
    from jax_aiter.gemm_fp4 import gemm_fp4_bf16

    M, N, K = 256, 256, 256
    key = jax.random.PRNGKey(1)
    k1, k2 = jax.random.split(key)
    a = jax.random.normal(k1, (M, K), dtype=jnp.bfloat16) * 0.1
    b = jax.random.normal(k2, (N, K), dtype=jnp.bfloat16) * 0.1

    out = gemm_fp4_bf16(a, b)
    ref = (a.astype(jnp.float32) @ b.astype(jnp.float32).T).astype(jnp.bfloat16)

    assert out.shape == (M, N)
    assert out.dtype == jnp.bfloat16
    assert jnp.all(jnp.isfinite(out)), "FP4 GEMM smoke: NaN/Inf in output"
    max_diff = float(jnp.max(jnp.abs(out.astype(jnp.float32) - ref.astype(jnp.float32))))
    max_ref = float(jnp.max(jnp.abs(ref.astype(jnp.float32))))
    rel_err = max_diff / max(max_ref, 1e-6)
    print(f"  FP4 GEMM: PASSED (rel_err={rel_err:.6f})")


def main():
    print(f"JAX devices: {jax.devices()}")

    passed = 0
    failed = 0

    for name, fn in [("BF16 GEMM", smoke_bf16_gemm),
                     ("FP4 GEMM", smoke_fp4_gemm)]:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  {name}: FAILED ({e})")
            failed += 1

    print()
    print(f"GEMM smoke summary: {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
    print("All GEMM smoke tests PASSED")


if __name__ == "__main__":
    main()
