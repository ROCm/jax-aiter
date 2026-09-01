#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Block until every visible GPU is idle, or give up after a timeout.
#
# Co-tenancy on the shared boxes is real: a timed run that starts against
# someone else's job produces a number that looks like a regression and is not
# one. Two mid-run excursions traced to exactly this forced a measured run to be
# retired during the wheel-vs-source analysis. Both nightly-perf.yml and
# weekly-convergence.yml call this before their timed leg; until now the file
# did not exist and both emitted "::warning:: absent; not gating on idle" and
# measured anyway.
#
# Exit status is deliberately 0 even on timeout -- the callers invoke this as a
# best-effort gate ("|| true") and a busy box should not fail the build. The
# decision the caller needs is in the log line, and on GitHub in the ::warning::.
set -uo pipefail

POLL_SECONDS="${JA_IDLE_POLL_SECONDS:-15}"
TIMEOUT_SECONDS="${JA_IDLE_TIMEOUT_SECONDS:-1800}"
# A GPU is "idle" below this utilisation. Not zero: rocm-smi occasionally reports
# a percent or two of residual activity on a quiescent card.
USE_THRESHOLD="${JA_IDLE_USE_THRESHOLD:-5}"
# VRAM matters independently of utilisation -- a parked process holding memory
# will not show as busy but will change our allocator behaviour at mem_fraction .97.
VRAM_THRESHOLD="${JA_IDLE_VRAM_THRESHOLD:-5}"

if ! command -v rocm-smi >/dev/null 2>&1; then
  echo "::warning::wait_for_idle_gpu: rocm-smi not found; not gating on idle"
  exit 0
fi

busy_report() {
  # Prints one "cardN use=X vram=Y" line per busy card; empty output means idle.
  # Honours HIP_VISIBLE_DEVICES so a 4-GPU leg does not wait on the other four.
  rocm-smi --showuse --showmemuse --json 2>/dev/null | python3 -c '
import json, os, re, sys

try:
    cards = json.load(sys.stdin)
except Exception:
    sys.exit(0)  # unparseable -> treat as idle, the gate is best-effort

visible = os.environ.get("HIP_VISIBLE_DEVICES", "").strip()
allowed = None
if visible:
    allowed = {f"card{i.strip()}" for i in visible.split(",") if i.strip() != ""}

use_max = float(os.environ["USE_THRESHOLD"])
vram_max = float(os.environ["VRAM_THRESHOLD"])

def num(value):
    m = re.search(r"[0-9]+(?:\.[0-9]+)?", str(value))
    return float(m.group()) if m else 0.0

for name, fields in sorted(cards.items()):
    if allowed is not None and name not in allowed:
        continue
    use = num(fields.get("GPU use (%)", 0))
    vram = num(fields.get("GPU Memory Allocated (VRAM%)", 0))
    if use > use_max or vram > vram_max:
        print(f"{name} use={use:g}% vram={vram:g}%")
'
}

export USE_THRESHOLD VRAM_THRESHOLD

deadline=$((SECONDS + TIMEOUT_SECONDS))
first=1
while :; do
  busy="$(busy_report)"
  if [[ -z "$busy" ]]; then
    [[ $first -eq 1 ]] \
      && echo "[wait_for_idle_gpu] all visible GPUs idle; proceeding" \
      || echo "[wait_for_idle_gpu] GPUs went idle after $((SECONDS))s; proceeding"
    exit 0
  fi
  if (( SECONDS >= deadline )); then
    echo "::warning::wait_for_idle_gpu: still busy after ${TIMEOUT_SECONDS}s; measuring anyway (number may be contended)"
    echo "$busy" | sed 's/^/[wait_for_idle_gpu] busy: /'
    exit 0
  fi
  if [[ $first -eq 1 ]]; then
    echo "[wait_for_idle_gpu] waiting up to ${TIMEOUT_SECONDS}s for:"
    echo "$busy" | sed 's/^/[wait_for_idle_gpu]   /'
    first=0
  fi
  sleep "$POLL_SECONDS"
done
