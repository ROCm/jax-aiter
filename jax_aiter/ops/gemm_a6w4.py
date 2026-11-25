"""gfx950 A6W4 GEMM: MXFP6 activations by MXFP4 weights.

``gemm_a6w4(a, b)`` quantizes the BF16 activation ``a`` (``[M, K]``) into AITER's
``mxfp6_c0c1_256_padk2`` layout and the BF16 weight ``b`` (``[N, K]``) into the
compact ``mxfp6_mxfp4_c0_256_padk2`` MXFP4 layout, then returns ``a @ b.T``.
Both quantizers apply the same per-1x32 Hadamard rotation, which cancels in the
product. The assembly kernels work on 256x256 output tiles and a 128-wide K
step; this wrapper pads internally and slices the result back to ``[M, N]``.

The activation packer is :func:`jax_aiter.ops.gemm_mxfp6.quantize_mxfp6_gemm`.
Kernel selection reads AITER's pre-tuned table
``third_party/aiter/aiter/configs/a6w4_asm_tuned_gemm.csv`` when it is present,
then applies every CSV in ``JA_A6W4_GEMM_CONFIG`` (separate several files with
the platform path separator). A later file overrides an earlier one for the same
``M,N,K``. Either file may use the JAX tuner format (``M,N,K,kernel_name``) or
AITER's (``gfx,cu_num,M,N,K,kernelName,...``); AITER rows apply only when their
``gfx`` and ``cu_num`` match this GPU. A shape with no row uses AITER's
untuned default for A6W4.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ..ffi.registry import register_ffi_target
from .gemm_mxfp6 import (
    _K_GUARD_TILES,
    _K_TILE,
    _SCALE_TILE_BYTES,
    _TILE,
    _ceil,
    load_mxfp6_gemm_config,
    load_mxfp6_gemm_configs,
    quantize_mxfp6_gemm,
)

# RoundUp is AITER's MX_DEFAULT_ROUND_MODE; the others are 0 down, 2 even, 3 ceil.
DEFAULT_ROUND_MODE = 1

_PACKED_W_TILE_BYTES = 16384
_MAX_BUFFER_BYTES = 1 << 31
_MAX_KERNEL_K = (1 << 31) - 1
_GROUPED_SWIZZLE_MAX_M = 131072
_GROUPED_SWIZZLE_MAX_N = 16384
_GROUPED_SWIZZLE_MAX_K = 6144

_MFMA32_SMALL_KERNEL = "_ZN5aiter40f6f4gemm_bf16_per1x32Fp6Fp4_m32_s0_a4_ntE"
_MFMA32_SWZ0_KERNEL = "_ZN5aiter39f6f4gemm_bf16_per1x32Fp6Fp4_m32_s0_a4_tE"
_MFMA32_GROUPED_KERNEL = "_ZN5aiter39f6f4gemm_bf16_per1x32Fp6Fp4_m32_s3_a4_tE"
_MFMA32_LONG_K_KERNEL = "_ZN5aiter39f6f4gemm_bf16_per1x32Fp6Fp4_m32_s3_a5_tE"


def _ensure_registered() -> None:
    register_ffi_target("Mxfp4GemmQuantJA", "ROCM")
    register_ffi_target("A6w4GemmFwdJA", "ROCM")


def mxfp4_gemm_pack_size(rows: int, k: int) -> tuple[int, int]:
    """Return uint8 packed-weight and scale sizes for a logical ``[rows, K]`` matrix."""
    if rows <= 0 or k <= 0:
        raise ValueError(
            f"mxfp4_gemm_pack_size requires positive dimensions, got {(rows, k)}"
        )
    tiles = _ceil(rows, _TILE) // _TILE * (_ceil(k, _K_TILE) // _K_TILE + _K_GUARD_TILES)
    sizes = (tiles * _PACKED_W_TILE_BYTES, tiles * _SCALE_TILE_BYTES)
    if max(sizes) > _MAX_BUFFER_BYTES:
        raise ValueError("mxfp4_gemm_pack_size exceeds the kernel's 2 GiB range")
    return sizes


def _normalize_round_mode(round_mode) -> int:
    try:
        value = int(round_mode)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"quantize_mxfp4_gemm expected round_mode in {{0,1,2,3}}, got {round_mode!r}"
        ) from error
    if value != round_mode or value not in range(4):
        raise ValueError(
            f"quantize_mxfp4_gemm expected round_mode in {{0,1,2,3}}, got {round_mode!r}"
        )
    return value


@functools.lru_cache(maxsize=None)
def _quant_call(rows: int, k: int, round_mode: int):
    packed_bytes, scale_bytes = mxfp4_gemm_pack_size(rows, k)
    _ensure_registered()
    call = jax.ffi.ffi_call(
        "Mxfp4GemmQuantJA",
        (
            jax.ShapeDtypeStruct((packed_bytes,), jnp.uint8),
            jax.ShapeDtypeStruct((scale_bytes,), jnp.uint8),
        ),
        has_side_effect=False,
    )
    return functools.partial(call, round_mode=np.int64(round_mode))


def quantize_mxfp4_gemm(x, *, round_mode: int = DEFAULT_ROUND_MODE):
    """Quantize BF16 ``[rows, K]`` into AITER's compact A6W4 MXFP4 weight buffers."""
    x = jnp.asarray(x)
    if x.dtype != jnp.bfloat16 or x.ndim != 2:
        raise ValueError(
            f"A6W4 GEMM weight quantization requires bfloat16 [rows, K], got {x.dtype} {x.shape}"
        )
    return _quant_call(*x.shape, _normalize_round_mode(round_mode))(x)


def load_a6w4_gemm_config(path, *, device: tuple[str, int] | None = None):
    """Load a JAX or AITER A6W4 tuning CSV (same formats as the A6W6 loader)."""
    return load_mxfp6_gemm_config(path, device=device)


def load_a6w4_gemm_configs(paths, *, device: tuple[str, int] | None = None):
    """Merge A6W4 tuning CSVs. A later file overrides an earlier file for one shape."""
    return load_mxfp6_gemm_configs(paths, device=device)


def _pretrained_config() -> Path | None:
    """Return AITER's shipped A6W4 tuning table when this checkout has it."""
    candidates = []
    configured = os.environ.get("AITER_CONFIG_GEMM_A6W4_ASM", "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    root = os.environ.get("JA_ROOT_DIR", "").strip()
    relative = Path("third_party/aiter/aiter/configs/a6w4_asm_tuned_gemm.csv")
    if root:
        candidates.append(Path(root) / relative)
    candidates.append(Path(__file__).resolve().parents[2] / relative)
    for path in candidates:
        if path.is_file():
            return path
    return None


def _config_paths() -> list[Path]:
    paths: list[Path] = []
    pretrained = _pretrained_config()
    if pretrained is not None:
        paths.append(pretrained)
    for part in os.environ.get("JA_A6W4_GEMM_CONFIG", "").strip().split(os.pathsep):
        if part.strip():
            paths.append(Path(part.strip()).expanduser())
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        key = path.resolve() if path.exists() else path
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


@functools.lru_cache(maxsize=1)
def _tuned_kernels() -> dict[tuple[int, int, int], str]:
    return load_a6w4_gemm_configs(_config_paths())


def clear_a6w4_gemm_config_cache() -> None:
    """Reload the pre-tuned and ``JA_A6W4_GEMM_CONFIG`` tables on the next trace."""
    _tuned_kernels.cache_clear()


def default_a6w4_gemm_kernel(m: int, n: int, k: int) -> str:
    """Return AITER's untuned A6W4 kernel for a logical shape."""
    pad_m, pad_n, pad_k = _ceil(m, _TILE), _ceil(n, _TILE), _ceil(k, _K_TILE)
    grouped_in_bounds = pad_m <= _GROUPED_SWIZZLE_MAX_M and pad_n <= _GROUPED_SWIZZLE_MAX_N
    if m < 2048:
        return _MFMA32_SMALL_KERNEL
    if k > n and (pad_k > _GROUPED_SWIZZLE_MAX_K or grouped_in_bounds):
        return _MFMA32_LONG_K_KERNEL
    if pad_k <= _GROUPED_SWIZZLE_MAX_K and grouped_in_bounds:
        return _MFMA32_GROUPED_KERNEL
    return _MFMA32_SWZ0_KERNEL


def _select_kernel(m: int, n: int, k: int, kernel_name: str | None) -> str:
    if kernel_name:
        return kernel_name
    configured = _tuned_kernels()
    tuned = configured.get((m, n, k))
    if tuned is None:
        tuned = configured.get((_ceil(m, _TILE), _ceil(n, _TILE), _ceil(k, _K_TILE)))
    if tuned is not None:
        return tuned
    return default_a6w4_gemm_kernel(m, n, k)


def _validate_launch(m: int, n: int, k: int) -> tuple[int, int]:
    """Reject launches the kernel ABI cannot express, before any allocation."""
    if m <= 0 or n <= 0 or k <= 0:
        raise ValueError(f"gemm_a6w4 requires positive dimensions, got {(m, n, k)}")
    if k > _MAX_KERNEL_K:
        raise ValueError(f"gemm_a6w4 K is outside the int32 kernel ABI: {k}")
    if _ceil(k, _K_TILE) > _MAX_KERNEL_K:
        raise ValueError(f"gemm_a6w4 padded K is outside the int32 kernel ABI: {_ceil(k, _K_TILE)}")
    pad_m, pad_n = _ceil(m, _TILE), _ceil(n, _TILE)
    if pad_m * pad_n * 2 > _MAX_BUFFER_BYTES:
        raise ValueError("gemm_a6w4 output exceeds the kernel's 2 GiB address range")
    return pad_m, pad_n


@functools.lru_cache(maxsize=None)
def _gemm_call(m: int, n: int, k: int, kernel_name: str):
    padded_m, padded_n = _validate_launch(m, n, k)
    _ensure_registered()
    return jax.ffi.ffi_call(
        "A6w4GemmFwdJA",
        jax.ShapeDtypeStruct((padded_m, padded_n), jnp.bfloat16),
        has_side_effect=False,
    )


def gemm_a6w4_packed(a_packed, b_packed, a_scale, b_scale, *, m: int, n: int, k: int,
                     kernel_name: str | None = None):
    """Launch A6W4 from packed operands.

    ``a_packed`` and ``a_scale`` come from :func:`quantize_mxfp6_gemm`;
    ``b_packed`` and ``b_scale`` come from :func:`quantize_mxfp4_gemm`.
    """
    _validate_launch(m, n, k)
    selected = _select_kernel(m, n, k, kernel_name)
    padded = _gemm_call(m, n, k, selected)(
        a_packed,
        b_packed,
        a_scale,
        b_scale,
        m=np.int64(m),
        n=np.int64(n),
        k=np.int64(k),
        kernel_name=selected,
    )
    return padded[:m, :n]


def gemm_a6w4(a, b, *, round_mode: int = DEFAULT_ROUND_MODE, kernel_name: str | None = None):
    """Return BF16 ``a @ b.T`` with gfx950 MXFP6 activations and MXFP4 weights.

    Both operands must be 2D BF16 tensors: ``a`` is ``[M, K]`` and ``b`` is
    ``[N, K]``. ``round_mode`` selects the weight scale rounding (0 down,
    1 up, 2 even, 3 ceil). Kernel selection is static at JIT trace time.
    """
    a, b = jnp.asarray(a), jnp.asarray(b)
    if a.dtype != jnp.bfloat16 or b.dtype != jnp.bfloat16 or a.ndim != 2 or b.ndim != 2:
        raise ValueError("gemm_a6w4 requires two bfloat16 rank-2 operands")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"gemm_a6w4 K mismatch: {a.shape} vs {b.shape}")
    _validate_launch(a.shape[0], b.shape[0], a.shape[1])
    a_packed, a_scale = quantize_mxfp6_gemm(a)
    b_packed, b_scale = quantize_mxfp4_gemm(b, round_mode=round_mode)
    return gemm_a6w4_packed(
        a_packed, b_packed, a_scale, b_scale, m=a.shape[0], n=b.shape[0], k=a.shape[1],
        kernel_name=kernel_name,
    )
