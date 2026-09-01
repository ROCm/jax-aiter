#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Parse a MaxText ``train.log`` perf leg and summarize throughput + loss.

Extends the ``completed step:`` regex used by ``scripts/build_leg_analysis.py``
to also capture ``loss``. Emits:

  * a JSON summary (``--out-json``) -- tail-N mean TFLOP/s/device, mean loss,
    final loss, and (when a baseline is present) the delta vs baseline.
  * a GitHub step-summary markdown snippet (printed to stdout and, with
    ``--step-summary FILE``, appended to that file -- pass
    ``$GITHUB_STEP_SUMMARY`` in CI).

Throughput gate (opt-in via ``--gate``): when a baseline JSON exists, FAIL
(exit nonzero) if tail-N mean TFLOP/s is more than ``--gate-threshold``
(default 5%) below the baseline. Loss is recorded + soft-warned only, never
gated. With no baseline, or without ``--gate``, the tool is record-only
(exit 0).

A matching ``completed step:`` line looks like::

    completed step: 2, seconds: 1.103, TFLOP/s/device: 1529.591, \
        Tokens/s/device: 29717.876, total_weights: 262144, loss: 12.181
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from statistics import mean

# step, seconds, TFLOP/s/device, then (later on the same line) loss.
STEP_LINE_RE = re.compile(
    r"completed step:\s+(\d+),\s+seconds:\s+([\d.]+),"
    r"\s+TFLOP/s/device:\s+([\d.]+).*?loss:\s+([-\d.eE+]+)"
)


def parse_steps(text: str) -> list[dict]:
    """Return one dict per ``completed step:`` line (in file order)."""
    steps = []
    for m in STEP_LINE_RE.finditer(text):
        steps.append(
            {
                "step": int(m.group(1)),
                "seconds": float(m.group(2)),
                "tflops_per_device": float(m.group(3)),
                "loss": float(m.group(4)),
            }
        )
    return steps


def summarize(steps: list[dict], tail_n: int, label: str) -> dict:
    """Compute tail-N mean TFLOP/s + mean/final loss from parsed steps."""
    tail = steps[-tail_n:] if len(steps) >= tail_n else steps[:]
    return {
        "label": label,
        "n_steps": len(steps),
        "tail_n": len(tail),
        "tail_first_step": tail[0]["step"],
        "tail_last_step": tail[-1]["step"],
        "mean_tflops_per_device": mean(s["tflops_per_device"] for s in tail),
        "mean_seconds": mean(s["seconds"] for s in tail),
        "mean_loss": mean(s["loss"] for s in tail),
        "final_step": steps[-1]["step"],
        "final_loss": steps[-1]["loss"],
    }


def compare_baseline(
    summ: dict, baseline: dict, gate: bool, gate_threshold: float, loss_tol: float
) -> dict:
    """Annotate ``summ`` with baseline deltas + a gate verdict (in place)."""
    bl_tf = baseline.get("mean_tflops_per_device")
    if bl_tf:
        summ["baseline_mean_tflops_per_device"] = bl_tf
        summ["tflops_delta_pct_vs_baseline"] = (
            (summ["mean_tflops_per_device"] - bl_tf) / bl_tf * 100.0
        )
        if gate:
            floor = bl_tf * (1.0 - gate_threshold)
            if summ["mean_tflops_per_device"] < floor:
                summ["gate_status"] = "FAIL"
                summ["gate_detail"] = (
                    f"tail-{summ['tail_n']} mean {summ['mean_tflops_per_device']:.1f} "
                    f"TFLOP/s/device is >{gate_threshold * 100:.0f}% below baseline "
                    f"{bl_tf:.1f} (floor {floor:.1f})."
                )
            else:
                summ["gate_status"] = "PASS"
        else:
            summ["gate_status"] = "record-only"
    bl_loss = baseline.get("mean_loss")
    if bl_loss:
        summ["baseline_mean_loss"] = bl_loss
        ld = (summ["mean_loss"] - bl_loss) / bl_loss * 100.0
        summ["loss_delta_pct_vs_baseline"] = ld
        if abs(summ["mean_loss"] - bl_loss) > abs(bl_loss) * loss_tol:
            summ["loss_warning"] = (
                f"mean loss drift {ld:+.1f}% vs baseline (soft check, not gated)."
            )
    return summ


def render_markdown(summ: dict) -> str:
    """Render a compact GitHub step-summary markdown block."""
    rows = [
        f"### Perf — `{summ['label']}` (MXFP4 8B)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Steps parsed | {summ['n_steps']} |",
        f"| Tail window | last {summ['tail_n']} "
        f"(step {summ['tail_first_step']}–{summ['tail_last_step']}) |",
        f"| **Tail-{summ['tail_n']} mean TFLOP/s/device** | "
        f"**{summ['mean_tflops_per_device']:.2f}** |",
        f"| Tail mean step time | {summ['mean_seconds']:.3f} s |",
        f"| Tail mean loss | {summ['mean_loss']:.4f} |",
        f"| Final loss (step {summ['final_step']}) | {summ['final_loss']:.4f} |",
    ]
    if "baseline_mean_tflops_per_device" in summ:
        rows.append(
            f"| Baseline TFLOP/s/device | "
            f"{summ['baseline_mean_tflops_per_device']:.2f} |"
        )
        rows.append(
            f"| Δ vs baseline | {summ['tflops_delta_pct_vs_baseline']:+.2f}% |"
        )
    rows.append(f"| Gate | {summ.get('gate_status', 'record-only')} |")
    if summ.get("gate_detail"):
        rows.append(f"| Gate detail | {summ['gate_detail']} |")
    if summ.get("loss_warning"):
        rows.append(f"| Loss warning | {summ['loss_warning']} |")
    return "\n".join(rows) + "\n"


def _selftest() -> int:
    """Tiny inline self-test on a synthetic train.log snippet."""
    sample = (
        "I0601 metric_logger.py:181] completed step: 0, seconds: 56.766, "
        "TFLOP/s/device: 29.711, Tokens/s/device: 577.248, total_weights: 262144, loss: 12.260\n"
        "noise line that should be ignored\n"
        "I0601 metric_logger.py:181] completed step: 1, seconds: 0.660, "
        "TFLOP/s/device: 100.000, Tokens/s/device: 49639.234, total_weights: 262144, loss: 12.000\n"
        "I0601 metric_logger.py:181] completed step: 2, seconds: 1.103, "
        "TFLOP/s/device: 200.000, Tokens/s/device: 29717.876, total_weights: 262144, loss: 11.500\n"
    )
    steps = parse_steps(sample)
    assert len(steps) == 3, steps
    assert steps[0]["step"] == 0 and steps[0]["loss"] == 12.260, steps[0]
    s = summarize(steps, tail_n=2, label="selftest")
    # tail-2 = steps 1,2 -> mean tflops (100+200)/2 = 150; mean loss (12+11.5)/2 = 11.75
    assert abs(s["mean_tflops_per_device"] - 150.0) < 1e-6, s
    assert abs(s["mean_loss"] - 11.75) < 1e-6, s
    assert s["final_loss"] == 11.5 and s["final_step"] == 2, s
    # gate: baseline 200 -> tail mean 150 is 25% below -> FAIL at 5% threshold.
    g = compare_baseline(
        dict(s),
        {"mean_tflops_per_device": 200.0, "mean_loss": 11.75},
        gate=True,
        gate_threshold=0.05,
        loss_tol=0.10,
    )
    assert g["gate_status"] == "FAIL", g
    # gate PASS when within threshold.
    p = compare_baseline(
        dict(s),
        {"mean_tflops_per_device": 151.0},
        gate=True,
        gate_threshold=0.05,
        loss_tol=0.10,
    )
    assert p["gate_status"] == "PASS", p
    print("parse_perf_log selftest OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", help="path to train.log")
    ap.add_argument("--out-json", help="write the JSON summary here")
    ap.add_argument("--tail-n", type=int, default=10, help="tail window (default 10)")
    ap.add_argument("--label", default="8b_fp4", help="leg label (default 8b_fp4)")
    ap.add_argument(
        "--baseline",
        default=None,
        help="baseline JSON (default: ci/perf/baseline_<label>.json next to this script, if it exists)",
    )
    ap.add_argument(
        "--gate",
        action="store_true",
        help="FAIL if tail-N TFLOP/s is >threshold below baseline (needs a baseline)",
    )
    ap.add_argument("--gate-threshold", type=float, default=0.05, help="throughput gate (default 0.05 = 5%%)")
    ap.add_argument("--loss-tol", type=float, default=0.10, help="soft loss drift tolerance (default 0.10)")
    ap.add_argument(
        "--step-summary",
        default=None,
        help="append the markdown snippet to this file (pass $GITHUB_STEP_SUMMARY in CI)",
    )
    ap.add_argument("--selftest", action="store_true", help="run the inline self-test and exit")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()
    if not args.log:
        ap.error("--log is required (or use --selftest)")

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"ERROR: log not found: {log_path}", file=sys.stderr)
        return 2
    steps = parse_steps(log_path.read_text())
    if not steps:
        print(f"ERROR: no 'completed step:' lines in {log_path}", file=sys.stderr)
        return 2

    summ = summarize(steps, args.tail_n, args.label)
    summ.setdefault("gate_status", "record-only")

    # Resolve baseline: explicit --baseline, else the conventional sibling path.
    baseline_path = args.baseline
    if baseline_path is None:
        default_bl = Path(__file__).resolve().parent / f"baseline_{args.label}.json"
        baseline_path = str(default_bl) if default_bl.exists() else None
    if baseline_path and Path(baseline_path).exists():
        baseline = json.loads(Path(baseline_path).read_text())
        summ["baseline_path"] = str(baseline_path)
        compare_baseline(summ, baseline, args.gate, args.gate_threshold, args.loss_tol)
    elif args.gate:
        print(
            "WARN: --gate set but no baseline found; running record-only.",
            file=sys.stderr,
        )

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summ, indent=2) + "\n")
        print(f"# wrote {out}", file=sys.stderr)

    md = render_markdown(summ)
    print(md)
    if args.step_summary:
        with open(args.step_summary, "a") as f:
            f.write(md + "\n")

    if summ.get("loss_warning"):
        print(f"WARN: {summ['loss_warning']}", file=sys.stderr)

    if summ.get("gate_status") == "FAIL":
        print(f"GATE FAIL: {summ.get('gate_detail', '')}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
