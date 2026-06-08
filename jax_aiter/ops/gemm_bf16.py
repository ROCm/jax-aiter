# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Raw BF16 GEMM via AITER ASM kernels.

No custom_vjp or custom_partitioning -- use jax_aiter.gemm.gemm for training.

32-bit output-offset overflow guard (M-tiling)
----------------------------------------------
The AITER bf16 ASM kernel (``GemmFwdJA``) addresses the output through
``GemmKernelArgs`` which carries *unsigned-int* (32-bit) strides/M/N. Once the
output exceeds ``2**31`` elements the element offset wraps and the kernel
silently corrupts a band of rows (no NaN/Inf, no error). This bites the
per-device logits/unembedding shape (8B 4.2e9, 70B 7.4e9 output elems).

Fix (frontend, no third_party / ASM change): when ``M*N >= 2**31`` we split the
M dimension into tiles of <= 16384 rows (worker-validated correct at full
speed), call the kernel per tile, and concatenate. Each tile's output stays
below 2**31 elements so no offset overflows. Outputs below 2**31 take the
original single-call path unchanged (byte-identical), so all existing
``jax_aiter.gemm`` callers (MLP/attn per-device shapes) are unaffected.

Env knob ``JA_GEMM_BF16_MTILE`` (default auto):
  unset / "auto"  -> tile only when M*N >= 2**31, tile rows = min(16384, 2**31//N)
  "0" / "off"     -> disable tiling (legacy single call; reproduces the bug,
                     used for A/B measurement of the broken path)
  <positive int>  -> force tiling for ANY shape with that many rows per tile
                     (bounded by 2**31//N for safety; used to prove the tiled
                     path is bitwise-identical to the single call on a cheap
                     shape).
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp

from ..ffi.registry import register_ffi_target

# GemmKernelArgs output offsets are 32-bit unsigned -> overflow at 2**31 elems.
_OUTPUT_OVERFLOW_LIMIT = 2 ** 31
# Worker-validated tile height: (16384, 128256, *) is correct at full speed and
# 16384 * 128256 = 2.10e9 < 2**31.
_DEFAULT_M_TILE = 16384


def _mtile_config(N: int):
    """Resolve (tile_rows, force) for the M-tiling overflow guard.

    tile_rows: max output rows per kernel call (0 => tiling disabled).
    force:     tile regardless of output size (forced/testing mode).
    """
    raw = os.environ.get("JA_GEMM_BF16_MTILE", "").strip().lower()
    # Largest tile height that keeps tile_rows * N strictly below 2**31.
    safe_cap = max(1, (_OUTPUT_OVERFLOW_LIMIT - 1) // max(int(N), 1))

    if raw in ("0", "off", "none", "disable", "disabled"):
        return 0, False
    if raw and raw not in ("auto", "on", "default"):
        try:
            forced = int(raw)
        except ValueError:
            forced = _DEFAULT_M_TILE
        if forced > 0:
            return min(forced, safe_cap), True
    return min(_DEFAULT_M_TILE, safe_cap), False


def _ensure_registered():
    register_ffi_target("GemmFwdJA", "ROCM")


def _gemm_fwd_call(out_shape, sem_shape, dtype):
    call = jax.ffi.ffi_call(
        "GemmFwdJA",
        (
            jax.ShapeDtypeStruct(out_shape, dtype),
            jax.ShapeDtypeStruct(sem_shape, jnp.uint32),
        ),
        vmap_method="broadcast_all",
        input_layouts=[None, None],
        output_layouts=[None, None],
        has_side_effect=False,
    )

    def _invoke(a, b):
        out, _ = call(a, b)
        return out

    return jax.jit(_invoke)


def gemm_bf16(a, b):
    """BF16 ASM GEMM via AITER FFI. A[M,K] @ B[N,K]^T -> Out[M,N].

    For outputs of 2**31 elements or more the M dimension is split into tiles
    of <= 16384 rows to dodge the kernel's 32-bit output-offset overflow (see
    the module docstring); each tile is computed by a separate kernel call and
    the results are concatenated. Smaller outputs take the single-call path
    unchanged.

    Args:
        a: [M, K] bfloat16.
        b: [N, K] bfloat16 (not transposed; kernel computes A @ B^T).

    Returns:
        out: [M, N] bfloat16.
    """
    _ensure_registered()
    M, K = a.shape
    N = b.shape[0]

    tile_rows, force = _mtile_config(N)
    need_tile = (
        tile_rows > 0
        and (force or M * N >= _OUTPUT_OVERFLOW_LIMIT)
        and M > tile_rows
    )

    if not need_tile:
        fn = _gemm_fwd_call((M, N), (16, 64), a.dtype)
        return fn(a, b)

    # Overflow regime: tile M so every kernel call's output < 2**31 elements.
    fn_full = _gemm_fwd_call((tile_rows, N), (16, 64), a.dtype)
    outs = []
    for start in range(0, M, tile_rows):
        rows = min(tile_rows, M - start)
        a_tile = a[start:start + rows]
        fn = fn_full if rows == tile_rows else _gemm_fwd_call((rows, N), (16, 64), a.dtype)
        outs.append(fn(a_tile, b))
    return jnp.concatenate(outs, axis=0)
