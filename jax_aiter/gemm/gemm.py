# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""BF16 GEMM via AITER ASM kernels with custom_vjp.

Forward: Out = A @ B^T using hand-tuned ASM kernels via FFI.
Backward: dA = dOut @ B, dB = dOut^T @ A (both via the same GEMM FFI).

Constraints:
  - A: [M, K] bf16, B: [N, K] bf16 -> Out: [M, N] bf16
  - K must be divisible by 64
  - Computes A @ B^T (B is in [N, K] layout, not transposed)
"""

from __future__ import annotations
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ..ffi.registry import register_ffi_target


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
    )

    def _invoke(a, b):
        out, _ = call(a, b)
        return out

    return jax.jit(_invoke)


@partial(jax.custom_vjp, nondiff_argnums=())
def gemm(
    a: jnp.ndarray,
    b: jnp.ndarray,
) -> jnp.ndarray:
    """Compute A @ B^T using AITER ASM GEMM.

    Args:
        a: [M, K] bf16
        b: [N, K] bf16

    Returns:
        out: [M, N] bf16
    """
    _ensure_registered()
    M, K = a.shape
    N = b.shape[0]
    fn = _gemm_fwd_call((M, N), (16, 64), a.dtype)
    return fn(a, b)


def _gemm_fwd(a, b):
    out = gemm(a, b)
    return out, (a, b)


def _gemm_bwd(residuals, grad_out):
    a, b = residuals
    # out = A @ B^T, so:
    # dA = dOut @ B        (dOut:[M,N] @ B:[N,K] -> [M,K])
    #   = gemm(dOut, B^T)  but gemm computes X @ Y^T
    #   so we need dOut @ B = dOut @ (B^T)^T -- this is just matmul
    #   Actually: gemm(dOut, B) = dOut @ B^T, but we want dOut @ B.
    #   dOut @ B = (dOut:[M,N]) @ (B:[N,K]) -- standard matmul
    #   Since gemm computes X @ Y^T, we need Y s.t. Y^T = B, i.e. Y = B^T.
    #   So dA = gemm(dOut, B^T) where B^T is [K, N].
    #   But B^T has shape [K, N] and gemm expects Y:[P, Q] to compute X @ Y^T.
    #   gemm(dOut:[M,N], B^T:[K,N]) = dOut @ (B^T)^T = dOut @ B = [M,K]. Correct!
    #
    # dB = dOut^T @ A      (dOut^T:[N,M] @ A:[M,K] -> [N,K])
    #   = gemm(dOut^T, A^T) where A^T:[K,M]
    #   gemm(dOut^T:[N,M], A^T:[K,M]) = dOut^T @ (A^T)^T = dOut^T @ A = [N,K]. Correct!

    _ensure_registered()
    M, K = a.shape
    N = b.shape[0]

    # dA = dOut @ B = gemm(dOut, B^T)
    # gemm(X, Y) = X @ Y^T. We want dOut @ B.
    # Let Y = B^T:[K,N]. Then X @ Y^T = dOut:[M,N] @ B:[N,K] = [M,K]. Correct.
    b_t = jnp.array(b.T, copy=True)  # [K, N], contiguous
    fn_da = _gemm_fwd_call((M, K), (16, 64), a.dtype)
    da = fn_da(grad_out, b_t)

    # dB = dOut^T @ A = gemm(dOut^T, A^T)
    # gemm(X, Y) = X @ Y^T. We want dOut^T @ A.
    # Let X = dOut^T:[N,M], Y = A^T:[K,M]. Then X @ Y^T = dOut^T @ A = [N,K]. Correct.
    grad_out_t = jnp.array(grad_out.T, copy=True)  # [N, M], contiguous
    a_t = jnp.array(a.T, copy=True)  # [K, M], contiguous
    fn_db = _gemm_fwd_call((N, K), (16, 64), a.dtype)
    db = fn_db(grad_out_t, a_t)

    return da, db


gemm.defvjp(_gemm_fwd, _gemm_bwd)
