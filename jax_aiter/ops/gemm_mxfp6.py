"""gfx950 A6W6 MXFP6 GEMM and its packed-operand helpers.

``gemm_mxfp6(a, b)`` quantizes BF16 matrices into AITER's
``mxfp6_c0c1_256_padk2`` layout and returns ``a @ b.T``.  The assembly kernels
operate on 256x256 output tiles and a 128-wide K dimension; this wrapper pads
internally and slices the logical result back to ``[M, N]``.

Kernel selection reads AITER's pre-tuned table
``third_party/aiter/aiter/configs/a6w6_blockscale_tuned_gemm.csv`` when that
file is present, then applies every CSV in ``JA_MXFP6_GEMM_CONFIG``. Separate
multiple customer files with the platform path separator. A customer row
replaces the pre-tuned kernel for the same ``M,N,K``. Either file may be the
JAX tuner format (``M,N,K,kernel_name``) or AITER's format
(``gfx,cu_num,M,N,K,kernelName,...``). AITER rows are used only when their
``gfx`` and ``cu_num`` match this GPU. Unmatched shapes use AITER's safe
dmabig kernel.
"""

from __future__ import annotations

import csv
import functools
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ..ffi.registry import register_ffi_target

_TILE = 256
_K_TILE = 128
_K_GUARD_TILES = 2
_PACKED_TILE_BYTES = 24576
_SCALE_TILE_BYTES = 1024
_DEFAULT_KERNEL = "_ZN5aiter29f6gemm_bf16_per1x32Fp6_dmabigE"
_SAFE_FALLBACK_KERNEL = "_ZN5aiter34f6gemm_bf16_per1x32Fp6_dmabig_swz0E"


def _ensure_registered() -> None:
    register_ffi_target("Mxfp6GemmQuantJA", "ROCM")
    register_ffi_target("Mxfp6GemmFwdJA", "ROCM")


def _ceil(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _layout(rows: int, k: int) -> tuple[int, int, int, int]:
    if rows <= 0 or k <= 0:
        raise ValueError(f"MXFP6 GEMM requires positive dimensions, got rows={rows}, K={k}")
    padded_rows = _ceil(rows, _TILE)
    padded_k = _ceil(k, _K_TILE)
    tiles = padded_rows // _TILE * (padded_k // _K_TILE + _K_GUARD_TILES)
    return padded_rows, padded_k, tiles * _PACKED_TILE_BYTES, tiles * _SCALE_TILE_BYTES


def mxfp6_gemm_pack_size(rows: int, k: int) -> tuple[int, int]:
    """Return uint8 packed-operand and scale sizes for a logical ``[rows, K]`` matrix."""
    _, _, packed_bytes, scale_bytes = _layout(rows, k)
    return packed_bytes, scale_bytes


@functools.lru_cache(maxsize=None)
def _quant_call(rows: int, k: int):
    _, _, packed_bytes, scale_bytes = _layout(rows, k)
    _ensure_registered()
    return jax.ffi.ffi_call(
        "Mxfp6GemmQuantJA",
        (
            jax.ShapeDtypeStruct((packed_bytes,), jnp.uint8),
            jax.ShapeDtypeStruct((scale_bytes,), jnp.uint8),
        ),
        has_side_effect=False,
    )


def quantize_mxfp6_gemm(x):
    """Quantize BF16 ``[rows, K]`` into AITER's A6W6 packed MXFP6 buffers."""
    x = jnp.asarray(x)
    if x.dtype != jnp.bfloat16 or x.ndim != 2:
        raise ValueError(f"MXFP6 GEMM quantization requires bfloat16 [rows, K], got {x.dtype} {x.shape}")
    return _quant_call(*x.shape)(x)


def load_mxfp6_gemm_config(path, *, device: tuple[str, int] | None = None) -> dict[tuple[int, int, int], str]:
    """Load a JAX or AITER MXFP6 GEMM tuning CSV.

    AITER's table is keyed by ``gfx,cu_num,M,N,K`` and names the kernel
    ``kernelName``. Rows for another GPU are ignored. The JAX tuner table is
    keyed only by ``M,N,K`` and names the kernel ``kernel_name``.
    """
    config = Path(path).expanduser()
    if not config.is_file():
        raise FileNotFoundError(f"MXFP6 GEMM config does not exist: {config}")
    with config.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fields = set(reader.fieldnames or [])
        if not {"M", "N", "K"}.issubset(fields):
            raise ValueError(f"{config} must contain M, N, and K columns")
        aiter_config = {"gfx", "cu_num", "kernelName", "splitK"}.issubset(fields)
        if not aiter_config and "kernel_name" not in fields:
            raise ValueError(
                f"{config} must contain kernel_name or AITER's gfx,cu_num,kernelName,splitK columns"
            )
        if aiter_config and device is None:
            from ..ja_compat.chip_info import get_cu_num, get_gfx

            device = (get_gfx(), get_cu_num())
        selected: dict[tuple[int, int, int], str] = {}
        for line_number, row in enumerate(reader, start=2):
            if aiter_config:
                if int(row["splitK"]) != 0:
                    raise ValueError(f"{config}:{line_number} has unsupported splitK")
                if (row["gfx"].strip(), int(row["cu_num"])) != device:
                    continue
                kernel = row["kernelName"].strip()
            else:
                kernel = row["kernel_name"].strip()
            if not kernel:
                continue
            shape = (int(row["M"]), int(row["N"]), int(row["K"]))
            previous = selected.get(shape)
            if previous is not None and previous != kernel:
                raise ValueError(f"{config} has conflicting kernels for shape {shape}")
            selected[shape] = kernel
    return selected


def load_mxfp6_gemm_configs(paths, *, device: tuple[str, int] | None = None) -> dict[tuple[int, int, int], str]:
    """Merge tuning CSVs. A later file overrides an earlier file for one shape."""
    merged: dict[tuple[int, int, int], str] = {}
    for path in paths:
        merged.update(load_mxfp6_gemm_config(path, device=device))
    return merged


def _pretrained_config() -> Path | None:
    """Return AITER's shipped A6W6 tuning table when this checkout has it."""
    candidates = []
    configured = os.environ.get("AITER_CONFIG_GEMM_A6W6", "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    root = os.environ.get("JA_ROOT_DIR", "").strip()
    relative = Path("third_party/aiter/aiter/configs/a6w6_blockscale_tuned_gemm.csv")
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
    configured = os.environ.get("JA_MXFP6_GEMM_CONFIG", "").strip()
    for part in configured.split(os.pathsep):
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
    return load_mxfp6_gemm_configs(_config_paths())


def clear_mxfp6_gemm_config_cache() -> None:
    """Reload the pre-tuned and ``JA_MXFP6_GEMM_CONFIG`` tables on the next trace."""
    _tuned_kernels.cache_clear()


def default_mxfp6_gemm_kernel(m: int, n: int, k: int) -> str:
    """Return AITER's manifest-safe untuned A6W6 kernel for a logical shape."""
    if _ceil(k, _K_TILE) <= 6144 and (
        _ceil(m, _TILE) > 131072 or _ceil(n, _TILE) > 16384
    ):
        return _SAFE_FALLBACK_KERNEL
    return _DEFAULT_KERNEL


def _select_kernel(m: int, n: int, k: int, kernel_name: str | None) -> str:
    if kernel_name is not None:
        return kernel_name
    configured = _tuned_kernels()
    tuned = configured.get((m, n, k))
    if tuned is None:
        tuned = configured.get((_ceil(m, _TILE), _ceil(n, _TILE), _ceil(k, _K_TILE)))
    if tuned is not None:
        return tuned
    return default_mxfp6_gemm_kernel(m, n, k)


@functools.lru_cache(maxsize=None)
def _gemm_call(m: int, n: int, k: int, kernel_name: str):
    padded_m = _layout(m, k)[0]
    padded_n = _ceil(n, _TILE)
    _ensure_registered()
    return jax.ffi.ffi_call(
        "Mxfp6GemmFwdJA",
        jax.ShapeDtypeStruct((padded_m, padded_n), jnp.bfloat16),
        has_side_effect=False,
    ), padded_m, padded_n


def gemm_mxfp6_packed(a_packed, b_packed, a_scale, b_scale, *, m: int, n: int, k: int,
                      kernel_name: str | None = None):
    """Launch A6W6 from packed operands returned by :func:`quantize_mxfp6_gemm`."""
    selected = _select_kernel(m, n, k, kernel_name)
    call, _, _ = _gemm_call(m, n, k, selected)
    padded = call(
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


def gemm_mxfp6(a, b, *, kernel_name: str | None = None):
    """Return BF16 ``a @ b.T`` with gfx950 A6W6 MXFP6 quantization and ASM GEMM.

    Both operands must be 2D BF16 tensors: ``a`` is ``[M, K]`` and ``b`` is
    ``[N, K]``.  Kernel selection is static at JIT trace time.
    """
    a, b = jnp.asarray(a), jnp.asarray(b)
    if a.dtype != jnp.bfloat16 or b.dtype != jnp.bfloat16 or a.ndim != 2 or b.ndim != 2:
        raise ValueError("gemm_mxfp6 requires two bfloat16 rank-2 operands")
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"gemm_mxfp6 K mismatch: {a.shape} vs {b.shape}")
    a_packed, a_scale = quantize_mxfp6_gemm(a)
    b_packed, b_scale = quantize_mxfp6_gemm(b)
    return gemm_mxfp6_packed(
        a_packed, b_packed, a_scale, b_scale, m=a.shape[0], n=b.shape[0], k=a.shape[1],
        kernel_name=kernel_name,
    )
