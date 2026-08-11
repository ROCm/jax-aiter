#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fail when a wheel needs glibc newer than the declared manylinux policy."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

_TAG = re.compile(r"manylinux_2_(\d+)_x86_64")
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
    # Provided by the required TheRock ROCm runtime.
    "libamdhip64.so.7",
    "libhiprtc.so.7",
    "libhsa-runtime64.so.1",
}


def policy_minors(auditwheel_output: str) -> list[int]:
    return [int(match) for match in _TAG.findall(auditwheel_output)]


def external_libraries(auditwheel_output: str) -> set[str]:
    return set(_EXTERNAL.findall(auditwheel_output))


def check(wheel: Path, max_minor: int) -> int:
    proc = subprocess.run(
        ["auditwheel", "show", str(wheel)],
        text=True,
        capture_output=True,
    )
    output = proc.stdout + proc.stderr
    print(output)
    if proc.returncode:
        print(f"ERROR: auditwheel rejected {wheel}", file=sys.stderr)
        return proc.returncode

    minors = policy_minors(output)
    if not minors:
        print("ERROR: auditwheel reported no manylinux policy", file=sys.stderr)
        return 1
    required = max(minors)
    if required > max_minor:
        print(
            f"ERROR: {wheel.name} requires manylinux_2_{required}; "
            f"release maximum is manylinux_2_{max_minor}",
            file=sys.stderr,
        )
        return 1

    externals = external_libraries(output)
    unexpected = externals - _ALLOWED_EXTERNAL
    if unexpected:
        print(
            "ERROR: unexpected external shared libraries: "
            + ", ".join(sorted(unexpected)),
            file=sys.stderr,
        )
        return 1
    print(
        f"PASS: {wheel.name} symbol floor <= manylinux_2_{max_minor} "
        f"(reported policies: {sorted(set(minors))}; "
        f"external libraries: {sorted(externals)})"
    )
    return 0


def _selftest() -> int:
    assert policy_minors('tag: "manylinux_2_28_x86_64"') == [28]
    assert policy_minors("manylinux_2_28_x86_64 ... manylinux_2_39_x86_64") == [
        28,
        39,
    ]
    assert policy_minors("linux_x86_64") == []
    text = "libc.so.6 with versions {'GLIBC_2.28'}, libamdhip64.so.7 with versions {'hip_6.2'}"
    assert external_libraries(text) == {"libc.so.6", "libamdhip64.so.7"}
    print("check_manylinux_policy selftest OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", nargs="?")
    parser.add_argument("--max-minor", type=int, default=28)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        return _selftest()
    if not args.wheel:
        parser.error("wheel is required unless --selftest is set")
    return check(Path(args.wheel), args.max_minor)


if __name__ == "__main__":
    sys.exit(main())
