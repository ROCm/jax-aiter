# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Clean-room smoke test for the jax-aiter FULL wheel: MHA flash attention.

Exercises the public flash attention entry ``jax_aiter.mha.flash_attn_func``
(forward) plus a backward via ``jax.grad`` -- this proves the multi-GB MHA
JIT libs bundled in the FULL wheel (``libmha_fwd.so`` + ``libmha_bwd.so``)
load and run from the wheel alone, with NO AITER source checkout.

    q,k,v: (batch, seqlen, nheads, headdim) bf16  ->  out: same shape

Asserts fwd + dq/dk/dv shapes and finiteness, prints a PASS line, and exits
non-zero on any failure. Only deps: jax + jax_aiter (no torch, no source).
"""

import sys
import time

import jax
import jax.numpy as jnp


def main() -> int:
    import jax_aiter

    print(f"jax_aiter __version__ = {jax_aiter.__version__}")

    from jax_aiter.mha import flash_attn_func

    b, s, h, d = 2, 256, 4, 64
    k0, k1, k2 = jax.random.split(jax.random.key(0), 3)
    q = jax.random.normal(k0, (b, s, h, d), dtype=jnp.bfloat16)
    k = jax.random.normal(k1, (b, s, h, d), dtype=jnp.bfloat16)
    v = jax.random.normal(k2, (b, s, h, d), dtype=jnp.bfloat16)
    expected = (b, s, h, d)

    # Forward: warmup (compile + ASM/CK kernel preload), then timed.
    out = flash_attn_func(q, k, v, causal=True)[0]
    out.block_until_ready()
    t0 = time.perf_counter()
    out = flash_attn_func(q, k, v, causal=True)[0]
    out.block_until_ready()
    dt_ms = (time.perf_counter() - t0) * 1e3

    if tuple(out.shape) != expected:
        print(f"MHA smoke FAIL: fwd shape {tuple(out.shape)} != {expected}")
        return 1
    if not bool(jnp.all(jnp.isfinite(out.astype(jnp.float32)))):
        print("MHA smoke FAIL: fwd output contains non-finite values")
        return 1

    # Backward: exercises libmha_bwd via custom_vjp.
    def loss(q_, k_, v_):
        return flash_attn_func(q_, k_, v_, causal=True)[0].astype(jnp.float32).sum()

    dq, dk, dv = jax.grad(loss, argnums=(0, 1, 2))(q, k, v)
    for name, g in (("dq", dq), ("dk", dk), ("dv", dv)):
        if tuple(g.shape) != expected:
            print(f"MHA smoke FAIL: {name} shape {tuple(g.shape)} != {expected}")
            return 1
        if not bool(jnp.all(jnp.isfinite(g.astype(jnp.float32)))):
            print(f"MHA smoke FAIL: {name} contains non-finite values")
            return 1

    print(
        f"MHA smoke PASS fwd+bwd shape={tuple(out.shape)} dtype={out.dtype} "
        f"fwd_time={dt_ms:.2f}ms devices={jax.devices()}"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 -- smoke test: report + fail.
        import traceback

        traceback.print_exc()
        print(f"MHA smoke FAIL: {type(exc).__name__}: {exc}")
        sys.exit(1)
