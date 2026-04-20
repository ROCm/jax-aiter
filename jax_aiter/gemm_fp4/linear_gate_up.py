# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Gate+Up fusion for gated-MLP blocks (SwiGLU / GeGLU).

A gated MLP projects the input with two separate weights ``W_gate`` and
``W_up`` (``[N, K]`` each), applies an activation to the gate output and
an element-wise product with the up output. Calling the FP4 GEMM twice
performs two independent quantization + GEMM pipelines.

This module folds both projections into a single FP4 GEMM by concatenating
the weights along the N axis. The forward pass runs **one** cast on the
activation, **one** cast on the combined weight, and **one** FP4 GEMM.
The backward is produced automatically by ``gemm_fp4_bf16``'s
``custom_vjp`` — so the backward also runs once for the combined weight.

Savings (per MLP layer, gate + up split out of the total MLP stack):

| Config   | fwd FFI | bwd FFI | total |
|----------|--------:|--------:|------:|
| separate | 6       | 4       | 10    |
| fused    | 3       | 3       | 6     |

At 32 layers (Llama3-8B) = **128 dispatches/step saved**.
At 80 layers (Llama3.3-70B) = **320 dispatches/step saved**.

The concat / split operations are zero-cost in XLA — they fuse with the
surrounding pointwise ops.

No new FFI handler is required; only the existing ``GemmFp4FwdJA`` +
``CastMxfp4JA`` + ``CastMxfp4DualJA`` are used.

**Single-GPU kernel-level benchmark caveat**: The FP4 ASM GEMM is typically
~10-25% slower per-FLOP when called once with ``N=2*N_proj`` than twice
with ``N=N_proj`` (tile-size mismatch at doubled N). See
``benchmarks/bench_gate_up_fused.py``. The fusion is therefore primarily
useful when dispatch overhead dominates (very small kernels, heavy FSDP
pipelining, gate+up sharing the same activation cast). Always benchmark
E2E on the target training config before turning this on in production.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .gemm_fp4 import gemm_fp4_bf16


def gemm_fp4_gate_up_raw(x, w_gate, w_up):
    """Fused gate+up via concat / GEMM / split (no custom_vjp wrapping).

    Args:
        x: ``[M, K]`` bfloat16 activation.
        w_gate: ``[N, K]`` bfloat16 gate projection weight.
        w_up:   ``[N, K]`` bfloat16 up   projection weight.

    Returns:
        gate: ``[M, N]`` bfloat16.
        up:   ``[M, N]`` bfloat16.
    """
    if w_gate.shape != w_up.shape:
        raise ValueError(
            f"gate and up weights must share shape; got {w_gate.shape} vs {w_up.shape}")
    N = w_gate.shape[0]
    # Concat along N dim: W_combined[2*N, K].
    w_combined = jnp.concatenate([w_gate, w_up], axis=0)
    # Single FP4 GEMM via the standard custom_vjp-wrapped helper. That call
    # provides its own forward/backward; JAX will trace the concat+split
    # around it and deliver dW_combined, which we then unpack.
    out = gemm_fp4_bf16(x, w_combined)
    gate = out[:, :N]
    up = out[:, N:]
    return gate, up


# Public alias: a differentiable function. We do NOT attach a custom_vjp
# here; ``gemm_fp4_bf16`` already has its own, and JAX will handle the
# concat/split layers around it automatically (both are pure JAX ops).
def gemm_fp4_gate_up_bf16(x, w_gate, w_up):
    """Fused gate+up FP4 GEMM with automatic gradient support.

    This is simply a thin wrapper around ``gemm_fp4_gate_up_raw``. Because
    ``gemm_fp4_bf16`` (called inside) already defines its ``custom_vjp``,
    ``jax.grad`` / ``jax.value_and_grad`` work out-of-the-box: the
    ``concat`` and ``split`` are differentiable primitives in JAX.
    """
    return gemm_fp4_gate_up_raw(x, w_gate, w_up)
