# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Pure-JAX MXFP4 quantization utilities.

Ports the quantization logic from AITER's fp4_utils.py (PyTorch/Triton)
to pure JAX for use in jax-aiter FP4 GEMM testing and prototyping.

Format: OCP MX FP4 E2M1 with E8M0 block scales (block size 32 along K).
Packing: two FP4 values per uint8 byte (even index = low nibble, odd = high nibble).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

MXFP4_BLOCK_SIZE = 32

MXFP4_LUT = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=np.float32)


def bf16_to_mxfp4(
    x: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Quantize BF16 [M, K] to packed MXFP4 + E8M0 block scales.

    Args:
        x: [M, K] bfloat16 or float32 input. K must be divisible by 64
           (32 for block scale, x2 for packing).

    Returns:
        packed: [M, K//2] uint8 — two E2M1 FP4 values per byte.
        scales: [M, K//32] uint8 — E8M0 block exponent scales.
    """
    M, K = x.shape
    assert K % (MXFP4_BLOCK_SIZE * 2) == 0, (
        f"K={K} must be divisible by {MXFP4_BLOCK_SIZE * 2}"
    )

    x_f32 = x.astype(jnp.float32)

    blocks = x_f32.reshape(M, K // MXFP4_BLOCK_SIZE, MXFP4_BLOCK_SIZE)

    amax = jnp.max(jnp.abs(blocks), axis=-1)
    amax = jnp.maximum(amax, jnp.finfo(jnp.float32).tiny)

    amax_u32 = amax.view(jnp.uint32)
    amax_u32 = (amax_u32 + jnp.uint32(0x200000)) & jnp.uint32(0xFF800000)
    amax_rounded = amax_u32.view(jnp.float32)

    scale_exp_unbiased = jnp.floor(jnp.log2(amax_rounded)) - 2.0
    scale_exp_unbiased = jnp.clip(scale_exp_unbiased, -127.0, 127.0)
    quant_scale = jnp.exp2(-scale_exp_unbiased)

    scales_e8m0 = (scale_exp_unbiased.astype(jnp.int32) + 127).astype(jnp.uint8)

    quant_scale_expanded = jnp.repeat(quant_scale, MXFP4_BLOCK_SIZE, axis=-1)
    qx = x_f32 * quant_scale_expanded

    e2m1_vals = _f32_to_e2m1(qx)

    packed = _pack_fp4(e2m1_vals)

    return packed, scales_e8m0


def mxfp4_to_bf16(
    packed: jnp.ndarray,
    scales: jnp.ndarray,
) -> jnp.ndarray:
    """Dequantize packed MXFP4 + E8M0 scales back to BF16.

    Args:
        packed: [M, K//2] uint8 — packed E2M1 FP4 values.
        scales: [M, K//32] uint8 — E8M0 block scales.

    Returns:
        x: [M, K] bfloat16.
    """
    M = packed.shape[0]
    K_half = packed.shape[1]
    K = K_half * 2

    lut = jnp.array(MXFP4_LUT, dtype=jnp.float32)

    low = (packed & 0xF).astype(jnp.int32)
    high = (packed >> 4).astype(jnp.int32)
    interleaved = jnp.empty((M, K), dtype=jnp.int32)
    interleaved = interleaved.at[:, 0::2].set(low)
    interleaved = interleaved.at[:, 1::2].set(high)

    x_f32 = lut[interleaved]

    scale_f32 = e8m0_to_f32(scales)
    scale_expanded = jnp.repeat(scale_f32, MXFP4_BLOCK_SIZE, axis=-1)

    return (x_f32 * scale_expanded).astype(jnp.bfloat16)


def e8m0_to_f32(scales_uint8: jnp.ndarray) -> jnp.ndarray:
    """Convert E8M0 biased exponent bytes to FP32 scale values.

    E8M0 is just an 8-bit biased exponent: scale = 2^(byte - 127).
    """
    return jnp.exp2(scales_uint8.astype(jnp.float32) - 127.0)


def e8m0_shuffle(scales: jnp.ndarray) -> jnp.ndarray:
    """Reorder E8M0 scale bytes into the layout expected by B-preshuffle ASM kernels.

    The ASM kernels with bpreshuffle=1 expect scales in a specific interleaved
    layout. This mirrors AITER's e8m0_shuffle.

    Args:
        scales: [M, K//32] uint8 (unshuffled E8M0 scales).

    Returns:
        scales_shuffled: [M_padded, K_padded//32] uint8 in ASM layout.
    """
    m, n = scales.shape
    m_pad = ((m + 255) // 256) * 256
    n_pad = ((n + 7) // 8) * 8

    padded = jnp.zeros((m_pad, n_pad), dtype=jnp.uint8)
    padded = padded.at[:m, :n].set(scales)

    sm, sn = padded.shape
    reshaped = padded.reshape(sm // 32, 2, 16, sn // 8, 2, 4)
    permuted = reshaped.transpose(0, 3, 5, 2, 4, 1)
    return permuted.reshape(sm, sn)


def shuffle_weight(x: jnp.ndarray, layout: tuple[int, int] = (16, 16)) -> jnp.ndarray:
    """Shuffle packed FP4 weight tensor for ASM B-preshuffle kernels.

    Mirrors AITER's shuffle.shuffle_weight with layout=(16,16).

    Args:
        x: [N, K//2] uint8 (packed fp4x2 weights).

    Returns:
        x_shuffled: [N, K//2] uint8 in ASM-expected layout.
    """
    IN, IK = layout
    BK = IK * 2
    K_elem = 16  # uint8 element size = 1 byte, 16/1=16
    BN = IN

    N, K_packed = x.shape
    assert N % BN == 0, f"N={N} not divisible by BN={BN}"
    assert K_packed % BK == 0, f"K_packed={K_packed} not divisible by BK={BK}"

    x_ = x.reshape(N // BN, BN, K_packed // BK, BK // K_elem, K_elem)
    x_ = x_.transpose(0, 2, 3, 1, 4)
    return x_.reshape(N, K_packed)


def _f32_to_e2m1(x: jnp.ndarray) -> jnp.ndarray:
    """Convert FP32 values (assumed pre-scaled to E2M1 range) to 4-bit E2M1 codes.

    E2M1 format (4 bits): S(1) E(2) M(1)
        S000 = +/-0, S001 = +/-0.5, S010 = +/-1.0, S011 = +/-1.5,
        S100 = +/-2.0, S101 = +/-3.0, S110 = +/-4.0, S111 = +/-6.0
    """
    _u32 = jnp.uint32
    x_u32 = x.view(jnp.uint32)
    s = x_u32 & _u32(0x80000000)
    e = (x_u32 >> _u32(23)) & _u32(0xFF)
    m = x_u32 & _u32(0x7FFFFF)

    E8_BIAS = _u32(127)
    E2_BIAS = _u32(1)

    adjusted_exp = E8_BIAS - (e + _u32(1))
    adjusted_exp = jnp.minimum(adjusted_exp, _u32(23))
    m_denorm = (_u32(0x400000) | (m >> _u32(1))) >> adjusted_exp
    m = jnp.where(e < E8_BIAS, m_denorm, m)

    e = jnp.maximum(e, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)

    e2m1_tmp = jnp.minimum(
        (((e << _u32(2)) | (m >> _u32(21))) + _u32(1)) >> _u32(1), _u32(0x7)
    )
    e2m1_value = ((s >> _u32(28)) | e2m1_tmp).astype(jnp.uint8)
    return e2m1_value


def _pack_fp4(codes: jnp.ndarray) -> jnp.ndarray:
    """Pack E2M1 codes (uint8, 4-bit values) into fp4x2 packed bytes.

    Even-indexed elements go to low nibble, odd-indexed to high nibble.
    Input shape: [M, K], output shape: [M, K//2].
    """
    assert codes.shape[-1] % 2 == 0
    evens = codes[..., 0::2]
    odds = codes[..., 1::2]
    return evens | (odds << 4)
