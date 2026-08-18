#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Ahead-of-time build of aiter's paged-attention kernels for the KV runtime.

``aiter::paged_attention_ragged`` derives a per-configuration function name and,
when that configuration is not already built, **shells out to
``python3 -m csrc.cpp_itfs.pa.pa_ragged``** and ``dlopen``s the result
(``csrc/cpp_itfs/pa/pa_ragged.cpp``). Doing that from inside an XLA FFI handler is
not acceptable: it needs a Python interpreter and a working directory at the aiter
root, and it would stall the first execution of every new shape bucket. This
script compiles the configurations we need in advance so ``not_built()`` is false
by the time a handler runs.

Three things here are load-bearing and easy to get wrong.

**The function name must be computed the way C++ computes it.** The C++ side hashes
exactly nine values (``pa_ragged.cpp``), while ``compile_template_op`` would hash
all thirteen template kwargs if left to pick a default. That is precisely why the
C++ passes ``--func_name`` explicitly, and why this script does too. Getting it
wrong produces a perfectly good library in a folder nothing will ever look in.

**The cache root is spelled differently on each side.** C++ appends ``.aiter`` to
``$AITER_ROOT_DIR`` (``utils.h``), Python does not (``utils.py``). They agree only
when the variable is unset, both landing on ``$HOME/.aiter``. To place the cache
anywhere else the two must be given different values, which ``--aiter-root``
handles: it takes the C++ spelling and passes Python the ``.aiter`` suffixed form.

**The codegen reaches torch for a file lock.** ``compile_lib`` calls ``mp_lock``,
which imports ``FileBaton`` from the ``aiter`` package, whose ``__init__``
imports torch and bootstraps a JIT build needing ninja. ``FileBaton`` itself is
stdlib-only, so this script loads it directly from its file and registers it in
``sys.modules`` before the codegen runs. Nothing in aiter is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import sys
import types
from pathlib import Path

# The nine values pa_ragged.cpp joins, in order, to name a configuration.
_NAME_ARGS = (
    "gqa_ratio",
    "head_size",
    "npar_loops",
    "dtype",
    "kv_dtype",
    "kv_cache_dtype",
    "out_dtype",
    "block_size",
    "alibi_enabled",
)

# aiter's C++ type spellings, as chosen in pa_ragged.py's torch wrapper.
DTYPE_BF16 = "__hip_bfloat16"
DTYPE_FP16 = "_Float16"

WARP_SIZE = 64  # gfx9
DEFAULT_PARTITION_SIZE = 256


def cpp_func_name(cfg: dict) -> str:
    """Reproduce ``get_default_func_name("pa_ragged", args)`` from ``utils.h``."""
    signature = "_".join(str(cfg[k]).lower() for k in _NAME_ARGS)
    digest = hashlib.md5(signature.encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"pa_ragged_{digest}"


def npar_loops_for(max_seq_len: int, partition_size: int = DEFAULT_PARTITION_SIZE) -> int:
    """``ceil(max_num_partitions / warpSize)``, as the caller computes it."""
    max_num_partitions = -(-max_seq_len // partition_size)
    return -(-max_num_partitions // WARP_SIZE)


def _install_file_baton(aiter_dir: Path) -> None:
    """Make ``aiter.jit.utils.file_baton`` importable without importing torch.

    Registers empty parents so the real ``aiter/__init__.py`` never executes, then
    loads the stdlib-only FileBaton module straight from its file.
    """
    if "aiter.jit.utils.file_baton" in sys.modules:
        return
    for name in ("aiter", "aiter.jit", "aiter.jit.utils"):
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.__path__ = []  # mark as a package so submodule imports resolve
            sys.modules[name] = stub

    path = aiter_dir / "aiter" / "jit" / "utils" / "file_baton.py"
    spec = importlib.util.spec_from_file_location("aiter.jit.utils.file_baton", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["aiter.jit.utils.file_baton"] = module
    spec.loader.exec_module(module)


def build_one(cfg: dict, *, verbose: bool = True) -> str:
    """Compile one configuration. Returns its function name."""
    from csrc.cpp_itfs.pa import pa_ragged

    func_name = cpp_func_name(cfg)
    if verbose:
        print(f"  {func_name}")
        print(f"    {cfg}")

    pa_ragged.compile(
        gqa_ratio=cfg["gqa_ratio"],
        head_size=cfg["head_size"],
        npar_loops=cfg["npar_loops"],
        dtype=cfg["dtype"],
        kv_dtype=cfg["kv_dtype"],
        fp8_kv_dtype=cfg["kv_cache_dtype"],
        out_dtype=cfg["out_dtype"],
        block_size=cfg["block_size"],
        # The config carries "true"/"false" strings because the name hash is built
        # from their spelling. compile() renders this straight into a jinja
        # conditional, where any non-empty string is truthy -- so a bare "false"
        # would silently compile the alibi path in and read a null slopes pointer.
        alibi_enabled=(str(cfg["alibi_enabled"]).lower() == "true"),
        func_name=func_name,
    )
    _verify_rendered_alibi(func_name, cfg)
    return func_name


def _verify_rendered_alibi(func_name: str, cfg: dict) -> None:
    """Fail the build if the emitted kernel disagrees with the requested config.

    A mis-rendered alibi flag is invisible until the GPU faults: the kernel
    indexes a slopes array that the caller never supplies. The template argument
    is cheap to read back, so confirm it rather than trusting the round trip.
    """
    want = str(cfg["alibi_enabled"]).lower()
    # AITER_ROOT_DIR already carries the `.aiter` suffix when main() sets it.
    root = os.environ.get("AITER_ROOT_DIR")
    base = Path(root) if root else Path(os.path.expanduser("~")) / ".aiter"
    source = base / "build" / func_name / f"{func_name}.cpp"
    if not source.exists():
        return

    text = source.read_text()
    marker = "paged_attention_ll4mi_QKV_mfma16_kernel<"
    start = text.find(marker)
    if start == -1:
        return
    args = [a.strip() for a in text[start + len(marker):text.index(">", start)].split(",")]
    literals = [a for a in args if a in ("true", "false")]
    if literals and literals[0] != want:
        raise RuntimeError(
            f"{func_name}: kernel compiled with ALIBI_ENABLED={literals[0]} but the "
            f"configuration asked for {want}. The alibi path dereferences a slopes "
            f"pointer the caller passes as null, so this would fault on the GPU.\n"
            f"The cache is keyed on the configuration alone, so a stale entry is "
            f"reused as-is. Delete {source.parent} and re-run."
        )


def default_configs(head_size: int, block_size: int, npar_loops: int) -> list[dict]:
    """The set the M2 tests exercise: bf16 and fp16, MHA/GQA/MQA ratios."""
    configs = []
    for dtype in (DTYPE_BF16, DTYPE_FP16):
        for gqa_ratio in (1, 4, 8):
            configs.append(
                {
                    "gqa_ratio": gqa_ratio,
                    "head_size": head_size,
                    "npar_loops": npar_loops,
                    "dtype": dtype,
                    "kv_dtype": dtype,
                    "kv_cache_dtype": "auto",
                    "out_dtype": dtype,
                    "block_size": block_size,
                    "alibi_enabled": "false",
                }
            )
    return configs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--aiter-dir",
        default=str(Path(__file__).resolve().parent.parent / "third_party" / "aiter"),
        help="aiter checkout to compile from",
    )
    parser.add_argument(
        "--aiter-root",
        default=None,
        help="cache root in the C++ spelling; '.aiter/build' is created beneath it. "
        "Leave unset to use $HOME, which is the only value both sides agree on "
        "without help.",
    )
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="only affects npar_loops, which is part of the configuration key",
    )
    parser.add_argument("--gqa-ratio", type=int, action="append", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the configurations and their paths, compile nothing")
    args = parser.parse_args()

    aiter_dir = Path(args.aiter_dir).resolve()
    if not (aiter_dir / "csrc" / "cpp_itfs").is_dir():
        print(f"error: no aiter checkout at {aiter_dir}", file=sys.stderr)
        return 1

    # C++ appends `.aiter`; Python does not. Give each the spelling it expects.
    if args.aiter_root:
        cpp_root = Path(args.aiter_root).resolve()
        os.environ["AITER_ROOT_DIR"] = str(cpp_root / ".aiter")
        build_dir = cpp_root / ".aiter" / "build"
        print(f"cache root: {build_dir}")
        print(f"  set AITER_ROOT_DIR={cpp_root} for the process that runs the kernels")
    else:
        os.environ.pop("AITER_ROOT_DIR", None)
        build_dir = Path(os.path.expanduser("~")) / ".aiter" / "build"
        print(f"cache root: {build_dir}  (AITER_ROOT_DIR unset on both sides)")

    npar = npar_loops_for(args.max_seq_len)
    configs = default_configs(args.head_size, args.block_size, npar)
    if args.gqa_ratio:
        configs = [c for c in configs if c["gqa_ratio"] in set(args.gqa_ratio)]

    print(f"npar_loops={npar} for max_seq_len={args.max_seq_len}")
    print(f"{len(configs)} configuration(s):")

    if args.dry_run:
        for cfg in configs:
            name = cpp_func_name(cfg)
            built = (build_dir / name / "lib.so").exists()
            print(f"  {name}  {'BUILT' if built else 'missing'}")
            print(f"    {cfg}")
        return 0

    sys.path.insert(0, str(aiter_dir))
    os.chdir(aiter_dir)  # the codegen resolves templates relative to the root
    _install_file_baton(aiter_dir)

    for cfg in configs:
        build_one(cfg)

    missing = [
        cpp_func_name(c) for c in configs if not (build_dir / cpp_func_name(c) / "lib.so").exists()
    ]
    if missing:
        print(f"\nERROR: {len(missing)} configuration(s) did not produce lib.so:", file=sys.stderr)
        for name in missing:
            print(f"  {name}", file=sys.stderr)
        return 1

    print(f"\nAll {len(configs)} configuration(s) built under {build_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
