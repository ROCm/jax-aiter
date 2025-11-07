# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
import jax.numpy as jnp
from .chip_info import get_gfx

defaultDtypes = {
    "gfx942": {"fp8": jnp.float8_e4m3fnuz},
    "gfx950": {"fp8": jnp.float8_e4m3fn},
}

_8bit_fallback = jnp.uint8


def get_dtype_fp8():
    return defaultDtypes.get(get_gfx(), {"fp8": _8bit_fallback})["fp8"]


i4x2 = getattr(jnp, "int4", _8bit_fallback)
# (Ruturaj4): float4_e2m1fn_x2 is used by pytorch.
fp4x2 = getattr(jnp, "float4_e2m1fn", _8bit_fallback)
fp8 = get_dtype_fp8()
fp8_e8m0 = getattr(jnp, "float8_e8m0fnu", _8bit_fallback)
fp16 = jnp.float16
bf16 = jnp.bfloat16
fp32 = jnp.float32
u32 = jnp.uint32
i32 = jnp.int32
i16 = jnp.int16
i8 = jnp.int8

d_dtypes = {
    "fp8": fp8,
    "fp8_e8m0": fp8_e8m0,
    "fp16": fp16,
    "bf16": bf16,
    "fp32": fp32,
    "i4x2": i4x2,
    "fp4x2": fp4x2,
    "u32": u32,
    "i32": i32,
    "i16": i16,
    "i8": i8,
}


# String helpers (parity with aiter).
def str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("yes", "true", "t", "y", "1"):
        return True
    if s in ("no", "false", "f", "n", "0"):
        return False
    raise ValueError("Boolean value expected.")


def str2tuple(v):
    try:
        parts = str(v).strip("()").split(",")
        return tuple(int(p.strip()) for p in parts if p.strip() != "")
    except Exception as e:
        raise ValueError(f"invalid format of input: {v}") from e


def to_jax_dtype(dtype_spec):
    """Convert a dtype specification to a JAX dtype.

    Args:
        dtype_spec: Can be:
            - A string key from d_dtypes (e.g., "fp16", "bf16", "fp32")
            - Common synonym strings (e.g., "float16", "bfloat16", "float32")
            - A dtype-like object (JAX/numpy dtype or Python type)

    Returns:
        A JAX dtype object

    Raises:
        ValueError: If the dtype_spec is not recognized
    """
    if isinstance(dtype_spec, str):
        dtype_str = dtype_spec.lower().strip()

        # Check if it's already in d_dtypes
        if dtype_str in d_dtypes:
            return d_dtypes[dtype_str]

        # Map common synonyms
        synonyms = {
            "half": "fp16",
            "float16": "fp16",
            "bfloat16": "bf16",
            "float32": "fp32",
            "float": "fp32",
            "int32": "i32",
            "uint32": "u32",
            "int16": "i16",
            "int8": "i8",
        }

        if dtype_str in synonyms:
            return d_dtypes[synonyms[dtype_str]]

        raise ValueError(f"Unknown dtype string: {dtype_spec}")

    try:
        return jnp.dtype(dtype_spec)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Cannot convert {dtype_spec} to JAX dtype") from e
