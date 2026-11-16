#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Unit tests for the hipified-source fixes in jax_aiter/jit/build_jit.py.

Run directly (``python3 ci/test_build_jit_source_fixes.py``) or via pytest.
"""
from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "build_jit", _ROOT / "jax_aiter" / "jit" / "build_jit.py"
)
build_jit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_jit)

# csrc/cpp_itfs/mha_bwd.cu in AITER 4726326b, before and after AITER's hipify,
# whose MATH_TRANSPILATIONS turn `std::max(` into `::max(`.
AITER_LINE = "            nsplit = std::clamp(want, 1u, std::max(kv_blocks, 1u));\n"
HIPIFIED_LINE = AITER_LINE.replace("std::max(", "::max(")
FIXED_LINE = (
    "            nsplit = std::clamp<uint32_t>(want, 1u, "
    "std::max<uint32_t>(kv_blocks, 1u));\n"
)


def _write(directory: Path, name: str, text: str) -> str:
    path = directory / name
    path.write_text(text)
    return str(path)


def test_hipified_mha_bwd_clamp_is_fixed_once():
    with tempfile.TemporaryDirectory() as td:
        src = _write(Path(td), "mha_bwd.hip", "// head\n" + HIPIFIED_LINE)
        assert build_jit.fix_hipified_sources([src]) == [src]
        assert Path(src).read_text() == "// head\n" + FIXED_LINE
        assert build_jit.fix_hipified_sources([src]) == []
        assert Path(src).read_text() == "// head\n" + FIXED_LINE
    print("PASS test_hipified_mha_bwd_clamp_is_fixed_once")


def test_sources_without_the_broken_text_are_untouched():
    cases = [
        ("mha_bwd.hip", "// an AITER without the deterministic split\n"),
        ("mha_bwd.hip", AITER_LINE),
        ("mha_fwd.hip", HIPIFIED_LINE),
    ]
    with tempfile.TemporaryDirectory() as td:
        for i, (name, text) in enumerate(cases):
            case_dir = Path(td) / str(i)
            case_dir.mkdir()
            src = _write(case_dir, name, text)
            assert build_jit.fix_hipified_sources([src]) == []
            assert Path(src).read_text() == text
    print("PASS test_sources_without_the_broken_text_are_untouched")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"\nAll {len(fns)} build_jit source-fix tests passed.")


if __name__ == "__main__":
    _run_all()
