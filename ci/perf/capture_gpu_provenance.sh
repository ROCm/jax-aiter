#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Record what machine a perf number came from, next to the number itself.
#
# Why: the nightly 8-GPU MXFP4 leg runs on an ephemeral ARC pool, so every run
# lands on a different physical host and the host is gone by the time anyone
# reads the result. The frozen 1840.6644 anchor was measured on a fixed local
# box; CI reproduces the recipe byte-for-byte (identical RESOLVED_RECIPE,
# XLA_FLAGS and COMMAND) yet reports a flat ~3.5% lower number with no
# excursions. Without machine state there is no way to tell a real kernel
# regression from a node with a lower power cap or a different VBIOS, and the
# gap can only be described with a label rather than a mechanism.
#
# Power cap is the first thing to check: MI355X parts ship at different Max
# Graphics Package Power values, and a lower cap produces exactly this
# signature -- a uniform offset with tight per-step spread.
#
# Never fails the caller. Missing provenance is worth a warning, not a red run.
set -uo pipefail

OUT="${1:-${PERF_OUT_DIR:-.}/gpu_provenance.txt}"
mkdir -p "$(dirname "$OUT")" 2>/dev/null || true

{
  echo "=== GPU_PROVENANCE_BEGIN ==="
  echo "runner_name=${RUNNER_NAME:-unknown}"
  echo "github_run_id=${GITHUB_RUN_ID:-n/a}"
  echo "hostname=$(cat /etc/hostname 2>/dev/null || hostname 2>/dev/null || echo unknown)"
  echo "kernel=$(uname -r 2>/dev/null || echo unknown)"
  echo "hip_visible_devices=${HIP_VISIBLE_DEVICES:-<unset>}"

  if command -v rocm-smi >/dev/null 2>&1; then
    # One line per card, so a diff between two runs points at the exact GPU.
    rocm-smi --showproductname --showvbios --showmaxpower --showdriverversion \
             --showserial --json 2>/dev/null | python3 -c '
import json, sys
try:
    cards = json.load(sys.stdin)
except Exception:
    sys.exit(0)
keys = [
    ("Card Series", "series"),
    ("Card Model", "model"),
    ("VBIOS version", "vbios"),
    ("Max Graphics Package Power (W)", "max_power_w"),
    ("Serial Number", "serial"),
]
for name, fields in sorted(cards.items()):
    if not isinstance(fields, dict):
        continue
    parts = [f"{short}={fields[full]}" for full, short in keys if full in fields]
    if parts:
        print(f"{name} " + " ".join(parts))
    for full, value in fields.items():
        if "Driver version" in full:
            print(f"driver_version={value}")
' | sort -u
  else
    echo "::warning::capture_gpu_provenance: rocm-smi not found"
  fi

  # Clocks at capture time. Taken before and after the recipe by the caller:
  # a cap that engages only under sustained load shows up as a delta.
  if command -v rocm-smi >/dev/null 2>&1; then
    echo "--- clocks ---"
    rocm-smi --showclocks 2>/dev/null \
      | grep -E "sclk|mclk|fclk" | sed 's/^[[:space:]]*//' || true
  fi
  echo "=== GPU_PROVENANCE_END ==="
} > "$OUT" 2>&1

echo "[ci/perf] gpu provenance -> $OUT"
cat "$OUT"
