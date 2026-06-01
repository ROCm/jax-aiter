#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Unit test for ci/perf/parse_perf_log.py on a synthetic train.log snippet.

Runnable two ways:
  * pytest ci/perf/test_parse_perf_log.py
  * python3 ci/perf/test_parse_perf_log.py   (prints OK / raises)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parse_perf_log as ppl  # noqa: E402

# A few real-format MaxText lines (loss IS on the line), plus noise that the
# regex must ignore.
SAMPLE = (
    "I0601 22:14:54 metric_logger.py:181] completed step: 0, seconds: 56.766, "
    "TFLOP/s/device: 29.711, Tokens/s/device: 577.248, total_weights: 262144, loss: 12.260\n"
    "some unrelated warning line\n"
    "I0601 22:14:55 metric_logger.py:181] completed step: 1, seconds: 0.660, "
    "TFLOP/s/device: 2554.951, Tokens/s/device: 49639.234, total_weights: 262144, loss: 12.218\n"
    "I0601 22:14:56 metric_logger.py:181] completed step: 2, seconds: 1.103, "
    "TFLOP/s/device: 1529.591, Tokens/s/device: 29717.876, total_weights: 262144, loss: 12.181\n"
)


def test_parse_steps_extracts_loss():
    steps = ppl.parse_steps(SAMPLE)
    assert len(steps) == 3
    assert [s["step"] for s in steps] == [0, 1, 2]
    assert steps[0]["seconds"] == 56.766
    assert steps[0]["tflops_per_device"] == 29.711
    # loss must be captured even though it is not adjacent to TFLOP/s/device.
    assert [s["loss"] for s in steps] == [12.260, 12.218, 12.181]


def test_summarize_tail_window():
    steps = ppl.parse_steps(SAMPLE)
    s = ppl.summarize(steps, tail_n=2, label="8b_fp4")
    # tail-2 = steps 1,2.
    assert s["tail_n"] == 2
    assert s["tail_first_step"] == 1 and s["tail_last_step"] == 2
    assert abs(s["mean_tflops_per_device"] - (2554.951 + 1529.591) / 2) < 1e-6
    assert abs(s["mean_loss"] - (12.218 + 12.181) / 2) < 1e-6
    assert s["final_step"] == 2 and s["final_loss"] == 12.181


def test_summarize_tail_larger_than_steps():
    steps = ppl.parse_steps(SAMPLE)
    s = ppl.summarize(steps, tail_n=10, label="8b_fp4")
    assert s["tail_n"] == 3  # clamps to available steps


def test_gate_fail_and_pass():
    steps = ppl.parse_steps(SAMPLE)
    s = ppl.summarize(steps, tail_n=2, label="8b_fp4")
    tail_mean = s["mean_tflops_per_device"]
    # baseline 10% higher than tail -> >5% below -> FAIL.
    fail = ppl.compare_baseline(
        dict(s),
        {"mean_tflops_per_device": tail_mean * 1.10},
        gate=True,
        gate_threshold=0.05,
        loss_tol=0.10,
    )
    assert fail["gate_status"] == "FAIL"
    # baseline 2% higher -> within 5% -> PASS.
    ok = ppl.compare_baseline(
        dict(s),
        {"mean_tflops_per_device": tail_mean * 1.02},
        gate=True,
        gate_threshold=0.05,
        loss_tol=0.10,
    )
    assert ok["gate_status"] == "PASS"
    # record-only (no --gate) even with a baseline present.
    rec = ppl.compare_baseline(
        dict(s),
        {"mean_tflops_per_device": tail_mean * 1.10},
        gate=False,
        gate_threshold=0.05,
        loss_tol=0.10,
    )
    assert rec["gate_status"] == "record-only"


def test_loss_soft_warning():
    steps = ppl.parse_steps(SAMPLE)
    s = ppl.summarize(steps, tail_n=2, label="8b_fp4")
    warned = ppl.compare_baseline(
        dict(s),
        {"mean_tflops_per_device": s["mean_tflops_per_device"], "mean_loss": 1.0},
        gate=True,
        gate_threshold=0.05,
        loss_tol=0.10,
    )
    # huge loss divergence -> soft warning present, but gate still PASS.
    assert "loss_warning" in warned
    assert warned["gate_status"] == "PASS"


def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    # exercise the inline selftest too.
    assert ppl._selftest() == 0
    print("test_parse_perf_log: all checks OK")


if __name__ == "__main__":
    _run_all()
