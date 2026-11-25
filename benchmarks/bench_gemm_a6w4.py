# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmark gfx950 A6W4 (MXFP6 x MXFP4) GEMM in JAX, the JAX twin of AITER's
``op_tests/test_gemm_a6w4.py::test_gemm_a6w4_benchmark``.

Two candidates are timed per ``M,N,K`` shape and round mode, as in AITER:

* ``gemm``: both operands are packed once, outside the timed call, so only the
  assembly kernel is timed.
* ``quant_gemm``: the activation is re-quantized to MXFP6 inside the timed call
  and the weights stay packed, as in a serving step with static weights.
* ``gemm_slice``: ``gemm`` plus the copy that crops the padded tile result to a
  dense ``[M, N]`` array, which is what ``jax_aiter.ops.gemm_a6w4`` returns.
  AITER's torch ``gemm_a6w4`` returns a strided view and has no such copy; it
  costs nothing when ``M`` and ``N`` are multiples of 256.

Each candidate reports microseconds, TFLOPS (``2*M*N*K``), TB/s, and the
mismatch ratio against the dequantized reference. The reference is AITER's torch
recipe when PyTorch AITER is importable, otherwise a plain FP32 matmul of the
BF16 inputs (a looser, cosine-based check). ``--torch`` also times PyTorch AITER
on the same shapes, so the two frameworks sit side by side.

Each JAX candidate runs ``--iters`` times inside one jitted loop, with an
optimization barrier chaining iterations so XLA cannot merge them; the median
of ``--repeats`` loops is reported. That keeps host dispatch out of the number.

    cd /dockerx/jax-aiter
    PYTHONPATH=$PWD:$PWD/third_party/aiter \
      JA_ROOT_DIR=$PWD AITER_ASM_DIR=$PWD/third_party/aiter/hsa/ \
      GPU_ARCHS=gfx950 AITER_SYMBOL_VISIBLE=1 \
      python3 benchmarks/bench_gemm_a6w4.py --torch
"""

from __future__ import annotations

import argparse
import itertools
import os
import statistics
import sys
import time

# Running ``python benchmarks/<script>.py`` puts only ``benchmarks`` on
# sys.path. Use this checkout when it has been built, and load the vendored
# aiter tree for the PyTorch comparison when that tree is present. A checkout
# with no build must not hide an installed wheel.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_AITER_ROOT = os.path.join(_REPO_ROOT, "third_party", "aiter")
_checkout_built = any(
    os.path.isfile(os.path.join(_REPO_ROOT, *parts))
    for parts in (
        ("build", "jax_aiter_build", "libjax_aiter.so"),
        ("jax_aiter", "_lib", "jax_aiter_build", "libjax_aiter.so"),
    )
)
if _checkout_built and _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if os.path.isdir(os.path.join(_AITER_ROOT, "aiter")) and _AITER_ROOT not in sys.path:
    sys.path.insert(0, _AITER_ROOT)

import jax
import jax.numpy as jnp
import numpy as np

from jax_aiter.ja_compat.chip_info import get_gfx
from jax_aiter.ops import gemm_a6w4 as _unused  # noqa: F401  (registers the package)
from jax_aiter.ops.gemm_a6w4 import (
    DEFAULT_ROUND_MODE,
    _gemm_call,
    _select_kernel,
    gemm_a6w4_packed,
    quantize_mxfp4_gemm,
)
from jax_aiter.ops.gemm_mxfp6 import quantize_mxfp6_gemm

DEFAULT_SHAPES = ["512,5120,5120", "9450,5120,5120", "9450,13824,5120", "9450,5120,13824"]


def _ceil(value, multiple):
    return (value + multiple - 1) // multiple * multiple


def _parse_shape(text):
    try:
        shape = tuple(int(value) for value in text.split(","))
    except ValueError:
        shape = ()
    if len(shape) != 3 or min(shape, default=0) <= 0:
        raise argparse.ArgumentTypeError(f"invalid shape {text!r}; expected positive M,N,K")
    return shape


def _time_us(step, operands, reps, repeats):
    """Median device time per call of ``step(*operands)``, in microseconds.

    The calls run inside one jitted ``fori_loop`` so host dispatch is paid once
    per ``reps`` calls. An ``optimization_barrier`` ties each result to the next
    iteration's operands, so XLA can neither hoist nor merge the repeated call.
    """

    @jax.jit
    def many(operands):
        def body(_, state):
            current, acc = state
            out = step(*current)
            current, out = jax.lax.optimization_barrier((current, out))
            return current, acc + out[0, 0].astype(jnp.float32)

        return jax.lax.fori_loop(0, reps, body, (operands, jnp.float32(0)))[1]

    many(operands).block_until_ready()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        many(operands).block_until_ready()
        samples.append((time.perf_counter() - start) * 1e6 / reps)
    return statistics.median(samples)


def _aiter_modules():
    """Return PyTorch AITER's A6W4 pieces, or None when they cannot be imported."""
    try:
        import torch  # noqa: F401
        import aiter.ops.gemm_op_a6w4 as a6w4
        from aiter.ops import gemm_op_a6w6
        from aiter.ops.quant import per_1x32_f4_quant
        from aiter.utility import dtypes, fp4_utils
    except Exception as error:  # noqa: BLE001 - any import failure means no torch path
        print(f"PyTorch AITER unavailable ({type(error).__name__}: {error}); "
              "using the FP32 matmul reference and skipping --torch")
        return None
    return a6w4, gemm_op_a6w6, per_1x32_f4_quant, dtypes, fp4_utils


def _to_torch(value):
    import torch

    return torch.tensor(np.asarray(value, dtype=np.float32), device="cuda", dtype=torch.bfloat16)


def _torch_reference(aiter_mods, x, w, round_mode):
    """AITER's quantized reference: dequantized MXFP6 activation by rotated MXFP4 weight."""
    import torch
    import torch.nn.functional as F

    _, a6w6, per_1x32_f4_quant, dtypes, fp4_utils = aiter_mods
    padded_k = _ceil(x.shape[1], 128)
    x = F.pad(x, (0, padded_k - x.shape[1]))
    w = F.pad(w, (0, padded_k - w.shape[1]))
    x_codes, x_scales = a6w6.quant_mxfp6_torch(x)
    x_dequant = a6w6.dequant_mxfp6_torch(x_codes, x_scales)
    w_rotated = a6w6._rotate_k32_torch(w).to(torch.bfloat16)
    w_codes, w_scales = per_1x32_f4_quant(w_rotated, quant_dtype=dtypes.fp4x2, round_mode=round_mode)
    w_dequant = fp4_utils.mxfp4_to_f32(w_codes)
    w_dequant *= fp4_utils.e8m0_to_f32(w_scales.view(torch.uint8)).repeat_interleave(32, dim=1)
    return (x_dequant @ w_dequant.T).float()


def _mismatch_ratio(actual, expected, rtol=1e-2, atol=1e-2):
    """Fraction of elements outside AITER's checkAllclose tolerance."""
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    return float((~np.isclose(actual, expected, rtol=rtol, atol=atol)).mean())


def _torch_time_us(fn, iters, repeats):
    import torch

    fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1e3 / iters)
    return statistics.median(samples)


def bench_shape(m, n, k, round_mode, *, iters, repeats, aiter_mods, run_torch):
    key_a, key_b = jax.random.split(jax.random.key(m + n + k))
    a = jax.random.normal(key_a, (m, k), dtype=jnp.bfloat16)
    b = jax.random.normal(key_b, (n, k), dtype=jnp.bfloat16)

    a_packed, a_scale = quantize_mxfp6_gemm(a)
    b_packed, b_scale = quantize_mxfp4_gemm(b, round_mode=round_mode)
    jax.block_until_ready((a_packed, a_scale, b_packed, b_scale))

    kernel = _select_kernel(m, n, k, None)
    kernel_call = _gemm_call(m, n, k, kernel)
    attrs = dict(m=np.int64(m), n=np.int64(n), k=np.int64(k), kernel_name=kernel)

    # The padded kernel result is what AITER's ``gemm_a6w4`` returns (a strided
    # view), so ``gemm`` and ``quant_gemm`` time it. ``gemm_slice`` adds the
    # copy that returns a dense ``[M, N]`` array, the public ``gemm_a6w4`` result.
    def gemm_step(ap, bp, asc, bsc):
        return kernel_call(ap, bp, asc, bsc, **attrs)

    def quant_gemm_step(x, bp, bsc):
        dynamic_packed, dynamic_scale = quantize_mxfp6_gemm(x)
        return kernel_call(dynamic_packed, bp, dynamic_scale, bsc, **attrs)

    def gemm_slice_step(ap, bp, asc, bsc):
        return gemm_a6w4_packed(ap, bp, asc, bsc, m=m, n=n, k=k, kernel_name=kernel)

    if aiter_mods is not None:
        reference = _torch_reference(aiter_mods, _to_torch(a), _to_torch(b), round_mode).cpu().numpy()
    else:
        reference = np.asarray(a.astype(jnp.float32) @ b.astype(jnp.float32).T)

    flops = 2 * m * n * k
    output_nbytes = _ceil(m, 256) * _ceil(n, 256) * 2
    gemm_nbytes = (a_packed.nbytes + a_scale.nbytes + b_packed.nbytes + b_scale.nbytes
                   + output_nbytes)
    packed_operands = (a_packed, b_packed, a_scale, b_scale)
    candidates = {
        "gemm": (gemm_step, packed_operands, gemm_nbytes),
        "quant_gemm": (quant_gemm_step, (a, b_packed, b_scale),
                       gemm_nbytes + a.nbytes + a_packed.nbytes + a_scale.nbytes),
        "gemm_slice": (gemm_slice_step, packed_operands, gemm_nbytes + 2 * m * n * 2),
    }
    row = {"gfx": get_gfx(), "M": m, "N": n, "K": k, "round_mode": round_mode,
           "kernel": kernel.removeprefix("_ZN5aiter")}
    for name, (step, operands, nbytes) in candidates.items():
        us = _time_us(step, operands, iters, repeats)
        out = np.asarray(step(*operands))[:m, :n]
        if aiter_mods is not None:
            row[f"{name} err"] = _mismatch_ratio(out, reference)
        else:
            cosine = float(np.dot(out.ravel(), reference.ravel())
                           / (np.linalg.norm(out) * np.linalg.norm(reference)))
            row[f"{name} cos(bf16)"] = cosine
        row[f"{name} us"] = us
        row[f"{name} TFLOPS"] = flops / us / 1e6
        row[f"{name} TB/s"] = nbytes / us / 1e6

    if run_torch:
        a6w4, a6w6, _, _, _ = aiter_mods
        x, w = _to_torch(a), _to_torch(b)
        x_packed, x_scale = a6w6.quant_mxfp6_gemm(x)
        w_packed, w_scale = a6w4.quant_mxfp4_gemm(w, round_mode=round_mode)

        def torch_gemm():
            return a6w4.gemm_a6w4(x_packed, w_packed, x_scale, w_scale, m, n, k)

        def torch_quant_gemm():
            dyn_packed, dyn_scale = a6w6.quant_mxfp6_gemm(x)
            return a6w4.gemm_a6w4(dyn_packed, w_packed, dyn_scale, w_scale, m, n, k)

        for name, fn in {"gemm": torch_gemm, "quant_gemm": torch_quant_gemm}.items():
            us = _torch_time_us(fn, iters, repeats)
            row[f"torch {name} us"] = us
            row[f"torch {name} TFLOPS"] = flops / us / 1e6
            row[f"jax/torch {name}"] = row[f"{name} us"] / us
    return row


def _print_table(rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    try:
        import pandas as pd

        print(pd.DataFrame(rows, columns=columns).to_markdown(index=False))
        return
    except Exception:  # noqa: BLE001 - pandas or tabulate may be missing
        pass
    print(" | ".join(columns))
    for row in rows:
        print(" | ".join(f"{row[c]:.4g}" if isinstance(row.get(c), float) else str(row.get(c, ""))
                         for c in columns))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-s", "--mnk", type=_parse_shape, nargs="*",
                        default=[_parse_shape(shape) for shape in DEFAULT_SHAPES],
                        help="M,N,K shapes, e.g. -s 512,5120,5120 9450,5120,5120")
    parser.add_argument("--round-mode", type=int, nargs="*", choices=range(4),
                        default=[DEFAULT_ROUND_MODE],
                        help="MXFP4 weight scale round mode (0 down, 1 up, 2 even, 3 ceil)")
    parser.add_argument("--iters", type=int, default=20, help="calls per timed loop")
    parser.add_argument("--repeats", type=int, default=7, help="timed loops; the median is reported")
    parser.add_argument("--torch", action="store_true",
                        help="also time PyTorch AITER's gemm_a6w4 on the same shapes")
    args = parser.parse_args()
    if args.iters <= 0 or args.repeats <= 0:
        parser.error("--iters and --repeats must be positive")

    if get_gfx() != "gfx950":
        print(f"gemm_a6w4 benchmark requires gfx950; skipping sweep on {get_gfx()}")
        return 0

    aiter_mods = _aiter_modules()
    if args.torch and aiter_mods is None:
        parser.error("--torch needs PyTorch AITER, which could not be imported")
    rows = [
        bench_shape(m, n, k, round_mode, iters=args.iters, repeats=args.repeats,
                    aiter_mods=aiter_mods, run_torch=args.torch)
        for (m, n, k), round_mode in itertools.product(args.mnk, args.round_mode)
    ]
    print("\ngemm_a6w4 summary (markdown):")
    _print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
