#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# One canonical MXFP4 regression process. The old script carried a second,
# divergent recipe (Llama 3, TE attention, GSPMD, scan=True, bf16 master state)
# that could not be compared with the documented JAX-AITER operating point.
# This wrapper installs pinned MaxText dependencies and delegates to the public
# recipe launcher instead.
set -euo pipefail

JA_ROOT_DIR="${JA_ROOT_DIR:-/jax-aiter}"
PERF_OUT_DIR="${PERF_OUT_DIR:-$JA_ROOT_DIR/ci_perf_out}"
MAXTEXT_DIR="${MAXTEXT_DIR:-/maxtext_perf}"
MAXTEXT_REPO="${MAXTEXT_REPO:-https://github.com/ROCm/maxtext.git}"
MAXTEXT_BRANCH="${MAXTEXT_BRANCH:-feature/jax-aiter-mxfp4-v26.6}"
MAXTEXT_COMMIT="${MAXTEXT_COMMIT:-b437942a5f33704f8438deb948488ad08164285c}"
MAXTEXT_REQUIREMENTS="${MAXTEXT_REQUIREMENTS:-src/dependencies/requirements/requirements_decoupled_rocm_jax_0_10_0.txt}"

echo "[ci/perf] MaxText: $MAXTEXT_REPO $MAXTEXT_BRANCH @ $MAXTEXT_COMMIT"
echo "[ci/perf] output:  $PERF_OUT_DIR"

if [[ ! -d "$MAXTEXT_DIR/.git" ]]; then
  git clone --filter=blob:none --branch "$MAXTEXT_BRANCH" \
    "$MAXTEXT_REPO" "$MAXTEXT_DIR"
fi
git -C "$MAXTEXT_DIR" fetch --depth=1 origin "$MAXTEXT_COMMIT"
git -C "$MAXTEXT_DIR" checkout --detach "$MAXTEXT_COMMIT"

# Install MaxText's decoupled dependencies without replacing the validated JAX
# 0.11 ROCm stack. The pinned branch's requirements predate alpha2 and contain
# a bare jax==0.10.0 line.
REQ_SOURCE="$MAXTEXT_DIR/$MAXTEXT_REQUIREMENTS"
REQ_FILTERED="$(mktemp)"
python3 - "$REQ_SOURCE" "$REQ_FILTERED" <<'PY'
from pathlib import Path
import sys

source, dest = map(Path, sys.argv[1:])
lines = [
    line for line in source.read_text().splitlines()
    if not line.strip().startswith(
        ("jax==", "jaxlib==", "jax-rocm7-plugin", "jax-rocm7-pjrt")
    )
]
dest.write_text("\n".join(lines) + "\n")
PY
python3 -m pip install --break-system-packages -r "$REQ_FILTERED"
rm -f "$REQ_FILTERED"
python3 -m pip install --break-system-packages --no-deps -e "$MAXTEXT_DIR"

# Restore and prove the exact runtime after dependency resolution.
JA_ROCM_PLUGIN_VERSION=0.11.0.post1 bash "$JA_ROOT_DIR/ci/setup_jax.sh" --jax-only
(
  cd /tmp
  env -u JA_ROOT_DIR -u AITER_ASM_DIR python3 - <<'PY'
import importlib.metadata as metadata
import os

import jax
import jax_aiter
from jax_aiter.mha import flash_attn_func

assert jax.__version__ == "0.11.0", jax.__version__
# The nightly builds its wheel from a moving AITER pin, so its version is not
# knowable here. Pin it only when the caller states an expectation (the release
# path does); otherwise just record what got installed.
expected = os.environ.get("JA_EXPECT_VERSION")
installed = metadata.version("jax-aiter")
if expected:
    assert installed == expected, f"{installed} != {expected}"
print("[ci/perf] jax", jax.__version__, "jax-aiter", jax_aiter.__version__)
print("[ci/perf] installed MHA import OK:", flash_attn_func)
PY
)

export PROJECT_ROOT=/
export JAX_AITER_ROOT="$JA_ROOT_DIR"
export MAXTEXT_ROOT="$MAXTEXT_DIR"
export JAX_AITER_RUNTIME=installed
# The MaxText checkout + dependency install above is the only part the weekly
# convergence workflow needs from this script; it drives the recipe itself
# through run_mlperf_ttc_8b.sh with a different dataset and step count. Stop
# here for that caller rather than making it duplicate the pinned-install logic.
if [[ "${JA_MAXTEXT_ONLY:-}" == "1" ]]; then
  echo "[ci/perf] JA_MAXTEXT_ONLY=1 -> MaxText installed at $MAXTEXT_COMMIT; not running the perf recipe."
  exit 0
fi

# Operating point. `blog_batch9` is the published MXFP4 point (FSDP8, batch 9,
# global 72, 8 GPUs, measurement window completed_steps_40_49) and is the ONLY
# configuration whose baseline is ~1840 TFLOP/s/device. The old default here was
# `ci_regression` (FSDP4, batch 4, global 16), whose correct baseline is ~1785 --
# a different operating point, not a regression against 1840. Do not compare
# a number produced under one profile with a baseline recorded under the other.
export RECIPE_PROFILE="${RECIPE_PROFILE:-blog_batch9}"
case "$RECIPE_PROFILE" in
  blog_batch9)
    export ICI_FSDP_PARALLELISM="${ICI_FSDP_PARALLELISM:-8}"
    export PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-9}"
    export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-72}"
    export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
    # Deliberately export NOTHING else. run_nvfp4_match_8b.sh enforces this
    # profile with require_value, and its own defaults ARE the contract:
    # autotune_level=4, weight/mu dtype bfloat16, use_iota_embed=False,
    # remat_policy=minimal_flash_save_fp4_wtcol. Those are the values the
    # frozen 1840.6644 run recorded in its train.log. Exporting the
    # ci_regression ladder here makes the launcher die on a require_value
    # mismatch before a single step runs.
    ;;
  ci_regression)
    export ICI_FSDP_PARALLELISM="${ICI_FSDP_PARALLELISM:-4}"
    export PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"
    export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
    export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3}"
    # ci_regression is unconstrained by require_value, and this ladder is the
    # one its ~1784.68 baseline was recorded under. It stays with the profile
    # it belongs to.
    export WEIGHT_DTYPE=float32 MU_DTYPE=float32
    export AUTOTUNE_LEVEL=5
    export REMAT_POLICY=minimal_flash_save_fp4col
    export JA_FP4_REMAT_SAVE_COL=both
    ;;
  *) echo "unknown RECIPE_PROFILE: $RECIPE_PROFILE" >&2; exit 2 ;;
esac
# Canonical for both profiles; the launcher refuses anything else.
export XLA_PYTHON_CLIENT_MEM_FRACTION=.97

cd "$JA_ROOT_DIR"
bash scripts/recipes/run_nvfp4_match_8b.sh \
  mxfp4 "$PERF_OUT_DIR" "${JA_PERF_STEPS:-50}"
