#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""REAL rocprof-trace dispatch assertion for the FP4 GEMM selector.

Wraps ``scripts/rocprof_kernel_bench.py`` (single FP4 GEMM, pre-quantised
inputs) with ``rocprofv3 --kernel-trace`` and inspects the resulting kernel
trace to prove which ASM kernel + splitK the handler actually launched for a
given (M,N,K) under a given selection env mode:

  * ``dispatch``  -> AITER_FP4_DISPATCH=1 (per-shape oracle table)
  * ``forced``    -> AITER_FORCE_KERNEL_NAME=256x256 / sK1 (production pin)
  * ``heuristic`` -> all selectors unset (occupancy heuristic)

The trace ``Kernel_Name`` carries the tile (e.g. ``...BpreShuffle_128x512``);
``Grid_Size_Z`` equals the splitK grid (= splitK, since workgroup_z=1). This is
the GPU-side ground truth the CPU-only env-plumbing test cannot give.

Used by tests/test_gemm_fp4_a1_kernel_override.py (skips gracefully when
rocprofv3 / GPU is unavailable) and as a standalone cross-check tool.

Exit code 0 = all checks passed; 2 = a dispatch mismatch; 3 = setup error.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import shutil
import statistics
import subprocess
import sys

REPO = "/ruvaidya/aiter_proj"
BENCH = f"{REPO}/scripts/rocprof_kernel_bench.py"
F4_RE = re.compile(r"f4gemm")
TILE_RE = re.compile(r"BpreShuffle_(\d+x\d+)")
FORCED_256 = "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E"


def _bench_env(env_mode: str) -> dict:
    env = os.environ.copy()
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".97"
    env.setdefault("HIP_VISIBLE_DEVICES", "0")
    for k in ("AITER_FP4_DISPATCH", "AITER_FORCE_KERNEL_NAME",
              "AITER_FORCE_LOG2_K_SPLIT", "AITER_FP4_KVWGRAD_128x512"):
        env.pop(k, None)
    if env_mode == "dispatch":
        env["AITER_FP4_DISPATCH"] = "1"
    elif env_mode == "forced":
        env["AITER_FORCE_KERNEL_NAME"] = FORCED_256
        env["AITER_FORCE_LOG2_K_SPLIT"] = "0"
    elif env_mode == "heuristic":
        pass
    else:
        raise ValueError(f"bad env_mode {env_mode}")
    return env


def capture(shape_mnk: str, env_mode: str, out_dir: str,
            warmup: int = 3, iters: int = 20) -> dict:
    """Run one rocprofv3 kernel-trace capture; return the dominant f4gemm info."""
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, "*")):
        os.remove(f)
    cmd = [
        "rocprofv3", "--kernel-trace", "-f", "csv", "-o", "ktrace",
        "-d", out_dir, "--",
        "python3", BENCH, "--shape", "verify", "--shape-mnk", shape_mnk,
        "--backend", "fp4_kern", "--warmup", str(warmup), "--iters", str(iters),
    ]
    proc = subprocess.run(cmd, env=_bench_env(env_mode),
                          cwd=f"{REPO}/jax-aiter",
                          capture_output=True, text=True, timeout=420)
    csvs = glob.glob(os.path.join(out_dir, "*kernel_trace*.csv"))
    if not csvs:
        return {"ok": False, "error": "no kernel-trace csv",
                "stderr": proc.stderr[-500:], "stdout": proc.stdout[-500:]}
    durs: dict[tuple, list] = {}
    with open(csvs[0]) as fh:
        for row in csv.DictReader(fh):
            name = row.get("Kernel_Name", "")
            if not F4_RE.search(name):
                continue
            m = TILE_RE.search(name)
            tile = m.group(1) if m else "?"
            gz = int(row.get("Grid_Size_Z", "1") or "1")
            wz = int(row.get("Workgroup_Size_Z", "1") or "1")
            splitk = max(1, gz // max(1, wz))
            dur = int(row["End_Timestamp"]) - int(row["Start_Timestamp"])
            durs.setdefault((tile, splitk), []).append(dur)
    if not durs:
        return {"ok": False, "error": "no f4gemm kernels in trace",
                "stderr": proc.stderr[-500:]}
    # dominant variant = most dispatches (steady state)
    (tile, splitk), dl = max(durs.items(), key=lambda kv: len(kv[1]))
    return {
        "ok": True, "shape_mnk": shape_mnk, "env_mode": env_mode,
        "tile": tile, "splitk": splitk, "n_dispatch": len(dl),
        "median_us": statistics.median(dl) / 1000.0,
        "min_us": min(dl) / 1000.0,
        "all_variants": {f"{t}/sK{s}": len(v) for (t, s), v in durs.items()},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="/tmp/fp4_dispatch_rocprof")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    # Each check: M,N,K:env_mode:expect_tile:expect_splitk
    ap.add_argument("--check", action="append", default=[],
                    help="M,N,K:env_mode:expect_tile:expect_splitk (repeatable). "
                         "Default = the 20260615 oracle dispatch assertions.")
    args = ap.parse_args()

    if shutil.which("rocprofv3") is None:
        print("SKIP: rocprofv3 not on PATH (GPU/rocprof host required).")
        sys.exit(0)

    checks = args.check or [
        "32768,4096,4096:dispatch:128x512:1",
        "4096,14336,32768:dispatch:256x256:1",
        "1024,4096,32768:dispatch:128x512:4",
        "32768,4096,4096:forced:256x256:1",
    ]
    results = []
    failed = 0
    for spec in checks:
        mnk, mode, exp_tile, exp_sk = spec.split(":")
        exp_sk = int(exp_sk)
        sub = os.path.join(args.out_dir, f"{mnk.replace(',', '_')}_{mode}")
        info = capture(mnk, mode, sub, args.warmup, args.iters)
        if not info.get("ok"):
            print(f"FAIL setup {spec}: {info.get('error')}\n  {info.get('stderr','')}")
            failed += 1
            results.append({"spec": spec, **info})
            continue
        ok_tile = info["tile"] == exp_tile
        ok_sk = info["splitk"] == exp_sk
        verdict = "PASS" if (ok_tile and ok_sk) else "FAIL"
        if verdict == "FAIL":
            failed += 1
        print(f"{verdict} {mnk} mode={mode}: got {info['tile']}/sK{info['splitk']} "
              f"expect {exp_tile}/sK{exp_sk} | {info['median_us']:.1f} us GPU "
              f"(n={info['n_dispatch']}) variants={info['all_variants']}")
        results.append({"spec": spec, "pass": verdict == "PASS", **info})

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
        json.dump(results, open(args.json_out, "w"), indent=2)
        print(f"[wrote] {args.json_out}")
    print(f"\n=== {len(results) - failed}/{len(results)} checks PASSED ===")
    sys.exit(0 if failed == 0 else 2)


if __name__ == "__main__":
    main()
