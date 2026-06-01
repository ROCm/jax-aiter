# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Clean-room smoke test for the jax-aiter lite wheel: FP4 (MXFP4) GEMM.

Exercises the public high-level FP4 GEMM, ``jax_aiter.gemm_fp4.gemm_fp4_bf16``
(see jax_aiter/gemm_fp4/gemm_fp4.py and jax_aiter/ops/gemm_fp4.py). It takes
BF16 inputs, quantizes to MXFP4 internally via the fused AITER cast + ASM
GEMM kernels, and returns BF16. The op computes ``A @ B^T``:

    a: [M, K] bf16,  b: [N, K] bf16  ->  out: [M, N] bf16

Asserts the output shape and finiteness, prints a PASS line with shape +
timing, and exits non-zero on any failure.

Only dependencies: jax + jax_aiter (no torch, no AITER source checkout).
"""

import sys
import time

import jax
import jax.numpy as jnp


def main() -> int:
    import jax_aiter

    print(f"jax_aiter __version__ = {jax_aiter.__version__}")

    from jax_aiter.gemm_fp4 import gemm_fp4_bf16

    M, K, N = 1024, 4096, 4096
    ka, kb = jax.random.split(jax.random.key(0))
    a = jax.random.normal(ka, (M, K), dtype=jnp.bfloat16)
    b = jax.random.normal(kb, (N, K), dtype=jnp.bfloat16)

    # Warmup: compile + ASM kernel preload.
    out = gemm_fp4_bf16(a, b)
    out.block_until_ready()

    # Timed run.
    t0 = time.perf_counter()
    out = gemm_fp4_bf16(a, b)
    out.block_until_ready()
    dt_ms = (time.perf_counter() - t0) * 1e3

    expected = (M, N)
    if tuple(out.shape) != expected:
        print(f"FP4 GEMM smoke FAIL: shape {tuple(out.shape)} != {expected}")
        return 1
    if not bool(jnp.all(jnp.isfinite(out.astype(jnp.float32)))):
        print("FP4 GEMM smoke FAIL: output contains non-finite values")
        return 1

    print(
        f"FP4 GEMM smoke PASS shape={tuple(out.shape)} dtype={out.dtype} "
        f"time={dt_ms:.2f}ms devices={jax.devices()}"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 -- smoke test: report + fail.
        import traceback

        traceback.print_exc()
        print(f"FP4 GEMM smoke FAIL: {type(exc).__name__}: {exc}")
        sys.exit(1)
