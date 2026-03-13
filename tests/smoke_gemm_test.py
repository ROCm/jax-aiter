# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Smoke test for BF16 GEMM via AITER ASM kernels."""

import jax
import jax.numpy as jnp

from jax_aiter.gemm import gemm


def main():
    print(f"JAX devices: {jax.devices()}")

    M, N, K = 256, 256, 256
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    a = jax.random.normal(k1, (M, K), dtype=jnp.bfloat16)
    b = jax.random.normal(k2, (N, K), dtype=jnp.bfloat16)

    out = gemm(a, b)
    ref = (a.astype(jnp.float32) @ b.astype(jnp.float32).T).astype(jnp.bfloat16)

    out_f32 = out.astype(jnp.float32)
    ref_f32 = ref.astype(jnp.float32)
    max_diff = float(jnp.max(jnp.abs(out_f32 - ref_f32)))
    print(f"GEMM smoke test: A[{M},{K}] @ B[{N},{K}]^T -> [{M},{N}]")
    print(f"  max_diff = {max_diff:.6f}")
    print(f"  dtype = {out.dtype}, shape = {out.shape}")
    print(f"  out[0,:5]  = {out_f32[0,:5]}")
    print(f"  ref[0,:5]  = {ref_f32[0,:5]}")
    print(f"  out mean   = {float(jnp.mean(jnp.abs(out_f32))):.6f}")
    print(f"  ref mean   = {float(jnp.mean(jnp.abs(ref_f32))):.6f}")
    print(f"  out all-zero? = {bool(jnp.all(out_f32 == 0))}")

    # For GEMM, tolerance scales with sqrt(K) due to bf16 accumulation.
    max_ref = float(jnp.max(jnp.abs(ref_f32)))
    rel_err = max_diff / max(max_ref, 1e-6)
    print(f"  rel_err    = {rel_err:.6f}")
    assert rel_err < 0.02, f"GEMM smoke FAILED: rel_err={rel_err} > 0.02"
    print("GEMM smoke test PASSED")


if __name__ == "__main__":
    main()
