#!/usr/bin/env python3
"""Validate a complete MaxText training log and summarize its tail window."""

import argparse
import json
import math
import re
from pathlib import Path


STEP_RE = re.compile(r"completed step:\s*(\d+)")
FIELDS = (
    "seconds",
    "TFLOP/s/device",
    "Tokens/s/device",
    "loss",
    "lm_loss",
    "perplexity",
    "raw_grad_norm",
    "grad_norm",
    "param_norm",
    "lr",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--expected-last-step", type=int, default=49)
    parser.add_argument("--tail", type=int, default=10)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def metric(line: str, name: str) -> float:
    match = re.search(rf"(?:^|,\s*){re.escape(name)}:\s*([^,\s]+)", line)
    if not match:
        raise ValueError(f"missing {name}")
    value = float(match.group(1))
    if not math.isfinite(value):
        raise ValueError(f"non-finite {name}={match.group(1)}")
    return value


def main() -> None:
    args = parse_args()
    records: dict[int, dict[str, float]] = {}
    for line_number, line in enumerate(
        args.log.read_text(errors="replace").splitlines(), start=1
    ):
        step_match = STEP_RE.search(line)
        if not step_match:
            continue
        step = int(step_match.group(1))
        if step in records:
            raise SystemExit(f"duplicate completed step {step} at line {line_number}")
        try:
            records[step] = {name: metric(line, name) for name in FIELDS}
        except ValueError as exc:
            raise SystemExit(f"invalid step {step} at line {line_number}: {exc}") from exc

    expected = list(range(args.expected_last_step + 1))
    observed = sorted(records)
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        raise SystemExit(f"incomplete step sequence: missing={missing} extra={extra}")
    if not 1 <= args.tail <= len(expected):
        raise SystemExit(f"invalid tail length {args.tail}")

    tail_steps = expected[-args.tail :]
    summary = {
        "log": str(args.log),
        "steps": len(records),
        "first_step": expected[0],
        "last_step": expected[-1],
        "tail_steps": tail_steps,
        "tail_count": len(tail_steps),
        "tail_mean_seconds": sum(records[s]["seconds"] for s in tail_steps) / args.tail,
        "tail_mean_tflops_per_device": sum(
            records[s]["TFLOP/s/device"] for s in tail_steps
        )
        / args.tail,
        "tail_mean_tokens_per_device": sum(
            records[s]["Tokens/s/device"] for s in tail_steps
        )
        / args.tail,
        "last_metrics": records[expected[-1]],
        "finite": True,
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
