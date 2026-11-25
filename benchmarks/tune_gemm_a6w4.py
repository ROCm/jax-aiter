#!/usr/bin/env python3
"""Tune gfx950 A6W4 (MXFP6 x MXFP4) GEMM kernels for one or more ``M,N,K`` shapes.

Examples::

    JA_ROOT_DIR=$PWD AITER_ASM_DIR=$PWD/third_party/aiter/hsa \
      python benchmarks/tune_gemm_a6w4.py --shape 9450,5120,5120 \
      --output /tmp/a6w4_gemm_tuned.csv

    python benchmarks/tune_gemm_a6w4.py --shapes shapes.yaml \
      --output /tmp/a6w4_gemm_tuned.csv

    python benchmarks/tune_gemm_a6w4.py \
      -i third_party/aiter/aiter/configs/a6w4_asm_untuned_gemm.csv \
      -o /tmp/a6w4_gemm_tuned.csv

``--shapes`` accepts the same CSV or YAML shape files as
``tune_gemm_mxfp6.py``. Set ``JA_A6W4_GEMM_CONFIG`` to the resulting CSV when
launching JAX; its rows override AITER's pre-tuned table for the same shapes.
The tuner only admits manifest candidates whose exact-shape and swizzle limits
cover the padded launch, and retains a candidate only if it is bit-identical to
AITER's untuned default kernel for that shape.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# Permit ``python benchmarks/tune_gemm_a6w4.py`` from a built checkout.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if (_REPO_ROOT / "build" / "jax_aiter_build").is_dir() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp

from jax_aiter.ops.gemm_a6w4 import (
    default_a6w4_gemm_kernel,
    gemm_a6w4_packed,
    quantize_mxfp4_gemm,
)
from jax_aiter.ops.gemm_mxfp6 import quantize_mxfp6_gemm
from tune_gemm_mxfp6 import (
    _ceil,
    _load_completed_config,
    _load_shapes,
    _median_us,
    _parse_shape,
    _write_config,
)

_MANIFEST = "f6f4gemm_bf16_per1x32Fp6Fp4.csv"
_PACK_LAYOUT = "mxfp6_mxfp4_c0_256_padk2"


def _manifest_path() -> Path:
    asm_root = os.environ.get("AITER_ASM_DIR", "").strip()
    if not asm_root:
        raise RuntimeError("set AITER_ASM_DIR to the gfx950 hsa directory")
    root = Path(asm_root)
    for directory in (root, root / "gfx950"):
        path = directory / "f6f4gemm" / _MANIFEST
        if path.is_file():
            return path
    raise FileNotFoundError(f"A6W4 manifest not found below: {root}")


def _candidates(m: int, n: int, k: int) -> list[str]:
    padded = (_ceil(m, 256), _ceil(n, 256), _ceil(k, 128))
    valid: list[str] = []
    with _manifest_path().open(newline="") as file:
        for row in csv.DictReader(file):
            if int(row["splitK"]) != 0 or row["pack_layout"] != _PACK_LAYOUT:
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
                if limits[2] > 0 and padded[2] <= limits[2] and any(
                    value > limit for value, limit in zip(padded[:2], limits[:2])
                ):
                    continue
            valid.append(row["knl_name"])
    return list(dict.fromkeys(valid))


def tune_shape(m: int, n: int, k: int, *, warmup: int, repeats: int, seed: int) -> tuple[str, float]:
    key_a, key_b = jax.random.split(jax.random.key(seed))
    a = jax.random.normal(key_a, (m, k), dtype=jnp.bfloat16)
    b = jax.random.normal(key_b, (n, k), dtype=jnp.bfloat16)
    ap, ascale = quantize_mxfp6_gemm(a)
    bp, bscale = quantize_mxfp4_gemm(b)
    ap.block_until_ready()
    bp.block_until_ready()

    calls = {}

    def call(kernel: str):
        if kernel not in calls:
            calls[kernel] = jax.jit(
                lambda kernel=kernel: gemm_a6w4_packed(
                    ap, bp, ascale, bscale, m=m, n=n, k=k, kernel_name=kernel
                )
            )
        return calls[kernel]()

    default_kernel = default_a6w4_gemm_kernel(m, n, k)
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
