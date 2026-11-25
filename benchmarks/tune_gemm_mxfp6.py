#!/usr/bin/env python3
"""Tune gfx950 A6W6 MXFP6 GEMM kernels for one or more ``M,N,K`` shapes.

Examples::

    JA_ROOT_DIR=$PWD AITER_ASM_DIR=$PWD/third_party/aiter/hsa \
      python benchmarks/tune_gemm_mxfp6.py --shape 8192,3072,9216 \
      --output /tmp/mxfp6_gemm_tuned.csv

    python benchmarks/tune_gemm_mxfp6.py --shapes shapes.yaml \
      --output /tmp/mxfp6_gemm_tuned.csv

    python benchmarks/tune_gemm_mxfp6.py \
      -i third_party/aiter/aiter/configs/a6w6_blockscale_untuned_gemm.csv \
      -o /tmp/mxfp6_gemm_tuned.csv

``--shapes`` accepts AITER's shape CSV (columns ``M,N,K``; extra columns are
ignored) or a YAML list. YAML is either a list or a mapping whose ``shapes``
value is a list. Each YAML entry is ``[M, N, K]``, ``"M,N,K"``, or
``{M: ..., N: ..., K: ...}``::

    shapes:
      - [8192, 3072, 9216]
      - {M: 256, N: 256, K: 128}

Set ``JA_MXFP6_GEMM_CONFIG`` to the resulting CSV when launching JAX.  The
tuner only admits manifest candidates whose exact-shape/swizzle contract
covers the padded launch, and retains a candidate only if it is bit-identical
to AITER's safe dmabig reference.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

# Permit ``python benchmarks/tune_gemm_mxfp6.py`` from a built checkout.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if (_REPO_ROOT / "build" / "jax_aiter_build").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax
import jax.numpy as jnp

from jax_aiter.ops.gemm_mxfp6 import (
    default_mxfp6_gemm_kernel,
    gemm_mxfp6_packed,
    quantize_mxfp6_gemm,
)

_MANIFEST = "f6gemm_bf16_per1x32Fp6.csv"


def _ceil(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _parse_shape(text: str) -> tuple[int, int, int]:
    try:
        shape = tuple(int(value) for value in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid shape {text!r}; expected M,N,K") from error
    if len(shape) != 3 or min(shape) <= 0:
        raise argparse.ArgumentTypeError(f"invalid shape {text!r}; expected positive M,N,K")
    return shape


def _shape_from_item(item, where: str) -> tuple[int, int, int]:
    if isinstance(item, str):
        try:
            return _parse_shape(item)
        except argparse.ArgumentTypeError as error:
            raise ValueError(f"{where}: {error}") from error
    if isinstance(item, (list, tuple)):
        values = item
    elif isinstance(item, dict):
        lowered = {str(key).lower(): value for key, value in item.items()}
        try:
            values = [lowered[key] for key in ("m", "n", "k")]
        except KeyError as error:
            raise ValueError(f"{where} must contain M, N, and K") from error
    else:
        raise ValueError(f"{where} must be [M, N, K], 'M,N,K', or a M/N/K mapping")
    try:
        shape = tuple(int(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{where} has a non-integer dimension") from error
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError(f"{where} must be three positive dimensions, got {shape}")
    return shape


def _load_csv_shapes(path: Path) -> list[tuple[int, int, int]]:
    """Load ``M,N,K`` rows from an AITER-style untuned or tuned GEMM CSV."""
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fields = set(reader.fieldnames or [])
        if not {"M", "N", "K"}.issubset(fields):
            raise ValueError(f"{path} must contain M, N, and K columns")
        shapes = [
            _shape_from_item(
                {"M": row["M"], "N": row["N"], "K": row["K"]},
                f"{path} row {index}",
            )
            for index, row in enumerate(reader, start=2)
        ]
    if not shapes:
        raise ValueError(f"{path} contains no shapes")
    return shapes


def _load_shapes(path: Path) -> list[tuple[int, int, int]]:
    """Load GEMM shapes from an AITER CSV or a YAML list."""
    if not path.is_file():
        raise FileNotFoundError(f"shape file not found: {path}")
    if path.suffix.lower() == ".csv":
        return _load_csv_shapes(path)
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"{path} must be a .csv, .yaml, or .yml shape file")
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("reading a YAML shape file requires PyYAML") from error
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "shapes" not in payload:
            raise ValueError(f"{path} must be a list of shapes or a mapping with 'shapes'")
        payload = payload["shapes"]
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{path} contains no shapes")
    return [_shape_from_item(item, f"{path} shape {index}") for index, item in enumerate(payload, 1)]


def _manifest_path() -> Path:
    asm_root = os.environ.get("AITER_ASM_DIR", "").strip()
    if not asm_root:
        raise RuntimeError("set AITER_ASM_DIR to the gfx950 hsa directory")
    root = Path(asm_root)
    for directory in (root, root / "gfx950"):
        path = directory / "f6gemm" / _MANIFEST
        if path.is_file():
            return path
    raise FileNotFoundError(f"MXFP6 manifest not found below: {root}")


def _candidates(m: int, n: int, k: int) -> list[str]:
    padded = (_ceil(m, 256), _ceil(n, 256), _ceil(k, 128))
    valid: list[str] = []
    with _manifest_path().open(newline="") as file:
        for row in csv.DictReader(file):
            if int(row["splitK"]) != 0 or row["pack_layout"] != "mxfp6_c0c1_256_padk2":
                continue
            exact = tuple(int(row[key]) for key in ("exact_M", "exact_N", "exact_K"))
            if exact != (0, 0, 0):
                if exact != padded:
                    continue
            else:
                limits = tuple(
                    int(row[key]) for key in ("swizzle_max_M", "swizzle_max_N", "swizzle_max_K")
                )
                # Zero max-K means that this kernel has no shape restriction.
                if limits[2] > 0 and any(value > limit for value, limit in zip(padded, limits)):
                    continue
            valid.append(row["knl_name"])
    return list(dict.fromkeys(valid))


def _median_us(fn, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn().block_until_ready()
        samples.append((time.perf_counter() - start) * 1e6)
    return statistics.median(samples)


def tune_shape(m: int, n: int, k: int, *, warmup: int, repeats: int, seed: int) -> tuple[str, float]:
    key_a, key_b = jax.random.split(jax.random.key(seed))
    a = jax.random.normal(key_a, (m, k), dtype=jnp.bfloat16)
    b = jax.random.normal(key_b, (n, k), dtype=jnp.bfloat16)
    ap, ascale = quantize_mxfp6_gemm(a)
    bp, bscale = quantize_mxfp6_gemm(b)
    ap.block_until_ready()
    bp.block_until_ready()

    calls = {}

    def call(kernel: str):
        if kernel not in calls:
            calls[kernel] = jax.jit(
                lambda kernel=kernel: gemm_mxfp6_packed(
                    ap, bp, ascale, bscale, m=m, n=n, k=k, kernel_name=kernel
                )
            )
        return calls[kernel]()

    default_kernel = default_mxfp6_gemm_kernel(m, n, k)
    reference = call(default_kernel)
    reference.block_until_ready()
    best_kernel, best_us = default_kernel, _median_us(lambda: call(default_kernel), repeats)
    for kernel in _candidates(m, n, k):
        for _ in range(warmup):
            call(kernel).block_until_ready()
        output = call(kernel)
        if not bool(jnp.array_equal(reference, output)):
            print(f"skip {kernel}: output differs from {default_kernel}")
            continue
        elapsed = _median_us(lambda: call(kernel), repeats)
        print(f"{m:6d},{n:6d},{k:6d}  {kernel:<75} {elapsed:9.2f} us")
        if elapsed < best_us:
            best_kernel, best_us = kernel, elapsed
    return best_kernel, best_us


def _write_config(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=("M", "N", "K", "kernel_name"))
        writer.writeheader()
        writer.writerows(rows)


def _load_completed_config(path: Path) -> list[dict[str, object]]:
    """Read a prior JAX tuner output so ``--resume`` preserves completed rows."""
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required = {"M", "N", "K", "kernel_name"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{path} is not a JAX MXFP6 tuner CSV")
        rows = [
            {
                "M": int(row["M"]),
                "N": int(row["N"]),
                "K": int(row["K"]),
                "kernel_name": row["kernel_name"].strip(),
            }
            for row in reader
            if row["kernel_name"].strip()
        ]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shape", type=_parse_shape, action="append", default=[],
                        help="one M,N,K shape; repeat to tune several")
    parser.add_argument("-i", "--shapes", type=Path,
                        help="CSV or YAML file containing multiple M,N,K shapes")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true",
                        help="keep completed rows in --output and skip their shapes")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats <= 0:
        parser.error("--warmup must be non-negative and --repeats positive")
    shapes = list(args.shape)
    if args.shapes is not None:
        try:
            shapes.extend(_load_shapes(args.shapes))
        except (OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
    if not shapes:
        parser.error("provide at least one --shape M,N,K or a --shapes CSV/YAML file")

    try:
        rows = _load_completed_config(args.output) if args.resume else []
    except (OSError, ValueError) as error:
        parser.error(str(error))
    completed = {(row["M"], row["N"], row["K"]) for row in rows}
    pending = [shape for shape in shapes if shape not in completed]
    if args.resume and len(pending) != len(shapes):
        print(f"resuming: {len(shapes) - len(pending)} completed, {len(pending)} remaining")
    for index, (m, n, k) in enumerate(pending):
        print(f"tuning shape {index + 1}/{len(pending)}: {m},{n},{k}")
        kernel, elapsed = tune_shape(
            m, n, k, warmup=args.warmup, repeats=args.repeats, seed=args.seed + index
        )
        print(f"selected {m},{n},{k}: {kernel} ({elapsed:.2f} us)")
        rows.append({"M": m, "N": n, "K": k, "kernel_name": kernel})
        _write_config(args.output, rows)


if __name__ == "__main__":
    main()
