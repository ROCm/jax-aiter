#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fail when a wheel needs a newer glibc than its own platform tag promises.

The alpha2 wheels bundle JIT libraries built by jit-libs.yml in an Ubuntu 24.04
image, so they cannot reach a low manylinux policy no matter which image builds
the wheel itself. Chasing broad manylinux compatibility is therefore not a goal.
What still matters is that the tag we publish is honest: a wheel labelled
manylinux_2_N must not require symbols newer than N, or pip installs it on a
system where it cannot load. By default the ceiling is read from the wheel's own
filename, so bumping a build image fails here until the tag is updated too.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

_TAG = re.compile(r"manylinux_2_(\d+)_x86_64")
_CONSTRAINT = re.compile(r"constrains the platform tag to \"?manylinux_2_(\d+)_x86_64")
_EXTERNAL = re.compile(r"([A-Za-z0-9_+.-]+\.so(?:\.\d+)*)\s+with versions")
_ALLOWED_EXTERNAL = {
    # manylinux/system ABI
    "libc.so.6",
    "libm.so.6",
    "libdl.so.2",
    "libpthread.so.0",
    "librt.so.1",
    "libutil.so.1",
    "libstdc++.so.6",
    "libgcc_s.so.1",
    "libelf.so.1",
    "libnuma.so.1",
    # Reached transitively: libhsa-runtime64 -> libelf -> libz.
    "libz.so.1",
    # Provided by the required TheRock ROCm runtime.
    "libamdhip64.so.7",
    "libhiprtc.so.7",
    "libhsa-runtime64.so.1",
}


def policy_minors(auditwheel_output: str) -> list[int]:
    return [int(match) for match in _TAG.findall(auditwheel_output)]


def required_minor(auditwheel_output: str) -> int | None:
    """The floor auditwheel derives from the symbols the wheel actually uses.

    Read from the "constrains the platform tag to" sentence rather than from
    every tag in the output, because the output also echoes the wheel's own
    filename -- comparing that against itself would always succeed.
    """
    flat = " ".join(auditwheel_output.split())
    match = _CONSTRAINT.search(flat)
    return int(match.group(1)) if match else None


def declared_minor(wheel_name: str) -> int | None:
    match = _TAG.search(wheel_name)
    return int(match.group(1)) if match else None


def external_libraries(auditwheel_output: str) -> set[str]:
    """Every external library auditwheel names, regardless of line wrapping.

    auditwheel re-flows this list to the terminal width, so "libfoo.so.1 with
    versions" is frequently split across a newline. Matching the raw text drops
    those entries and quietly weakens the allowlist, so flatten first.
    """
    return set(_EXTERNAL.findall(" ".join(auditwheel_output.split())))


def check(wheel: Path, max_minor: int | None) -> int:
    proc = subprocess.run(
        [sys.executable, "-m", "auditwheel", "show", str(wheel)],
        text=True,
        capture_output=True,
    )
    output = proc.stdout + proc.stderr
    print(output)
    if proc.returncode:
        print(f"ERROR: auditwheel rejected {wheel}", file=sys.stderr)
        return proc.returncode

    required = required_minor(output)
    ceiling = max_minor if max_minor is not None else declared_minor(wheel.name)
    if required is None:
        floor = "none reported"
    else:
        floor = f"manylinux_2_{required}"
    print(f"[policy] {wheel.name}: symbol floor {floor}")

    if required is not None and ceiling is not None and required > ceiling:
        print(
            f"ERROR: {wheel.name} requires manylinux_2_{required} but its tag "
            f"promises manylinux_2_{ceiling}; installs would fail to load",
            file=sys.stderr,
        )
        return 1
    if ceiling is None:
        print("[policy] wheel makes no manylinux claim; floor recorded only")

    externals = external_libraries(output)
    unexpected = externals - _ALLOWED_EXTERNAL
    if unexpected:
        print(
            "ERROR: unexpected external shared libraries: "
            + ", ".join(sorted(unexpected)),
            file=sys.stderr,
        )
        return 1
    print(f"PASS: {wheel.name} (external libraries: {sorted(externals)})")
    return 0


def _selftest() -> int:
    assert policy_minors('tag: "manylinux_2_28_x86_64"') == [28]
    assert policy_minors("manylinux_2_28_x86_64 ... manylinux_2_39_x86_64") == [
        28,
        39,
    ]
    assert policy_minors("linux_x86_64") == []

    # auditwheel wraps its prose, so the sentence must survive re-flowing.
    wrapped = 'This constrains the platform\ntag to "manylinux_2_39_x86_64". In order'
    assert required_minor(wrapped) == 39
    assert required_minor('consistent with tag: "manylinux_2_28_x86_64"') is None

    assert declared_minor("jax_aiter-0.1.0a2-cp312-cp312-manylinux_2_39_x86_64.whl") == 39
    assert declared_minor("jax_aiter-0.1.0a2-cp312-cp312-linux_x86_64.whl") is None

    text = "libc.so.6 with versions {'GLIBC_2.28'}, libamdhip64.so.7 with versions {'hip_6.2'}"
    assert external_libraries(text) == {"libc.so.6", "libamdhip64.so.7"}

    # A wrapped "with\nversions" must not hide a library from the allowlist.
    wrapped_ext = "libgcc_s.so.1 with versions\n{'GCC_3.0'}, libamdhip64.so.7 with\nversions {'hip_6.2'}"
    assert external_libraries(wrapped_ext) == {"libgcc_s.so.1", "libamdhip64.so.7"}

    assert "libz.so.1" in _ALLOWED_EXTERNAL
    print("check_manylinux_policy selftest OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", nargs="?")
    parser.add_argument(
        "--max-minor",
        type=int,
        default=None,
        help="ceiling to enforce; defaults to the wheel's own platform tag",
    )
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        return _selftest()
    if not args.wheel:
        parser.error("wheel is required unless --selftest is set")
    return check(Path(args.wheel), args.max_minor)


if __name__ == "__main__":
    sys.exit(main())
