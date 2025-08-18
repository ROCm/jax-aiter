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
