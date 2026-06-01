#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Build a jax-aiter wheel (lite or full) by REUSING the already-built AITER
# JIT libs in build/aiter_build/. This is a pure STAGING operation: it never
# invokes build_jit.py and never rebuilds the multi-GB MHA libs.
#
#   lite  -> drops libmha_fwd.so / libmha_bwd.so + their mha_*_ja.so shims
#            (and the stale moe_fwd_ja.so); carries a +lite local version tag.
#   full  -> ships everything (default of setup.py's JA_WHEEL_VARIANT).
#
# HARD GUARANTEE: build/aiter_build/*.so is snapshotted (sha256) before the
# build and re-checked after. If it changed, the script aborts non-zero --
# the MHA JIT libs must never be rebuilt by a wheel build.
#
# Run INSIDE the rv_aiter container (canonical env baked in):
#   docker exec rv_aiter bash -lc \
#     "cd /ruvaidya/aiter_proj/jax-aiter && bash scripts/build_wheel.sh --variant lite"
#
# Override the pip invocation extras (e.g. CI uses --break-system-packages):
#   PIP_EXTRA_ARGS=--break-system-packages bash scripts/build_wheel.sh --variant lite
#
# Usage: scripts/build_wheel.sh [--variant {lite|full}] [-h|--help]

set -euo pipefail

VARIANT="lite"
PIP_EXTRA_ARGS="${PIP_EXTRA_ARGS:-}"

usage() {
  cat >&2 <<'EOF'
usage: scripts/build_wheel.sh [--variant {lite|full}] [-h|--help]

  --variant lite   (default) drop MHA libs + shims; +lite local version tag
  --variant full   ship everything (no +lite tag)

Reuses build/aiter_build/*.so as-is (never rebuilds MHA). Aborts if those
libs change. Env override: PIP_EXTRA_ARGS (e.g. --break-system-packages).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant) VARIANT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "$VARIANT" != "lite" && "$VARIANT" != "full" ]]; then
  echo "ERROR: --variant must be 'lite' or 'full' (got '$VARIANT')" >&2
  exit 1
fi

# Step 1: announce.
echo "[build_wheel] variant=$VARIANT"

PRE_SHA="/tmp/ja_jitlibs_pre.sha"
POST_SHA="/tmp/ja_jitlibs_post.sha"
AITER_BUILD="build/aiter_build"
UMBRELLA="build/jax_aiter_build/libjax_aiter.so"

# Step 2: preflight -- the source-of-truth JIT libs must already exist.
# DO NOT auto-rebuild; surface the canonical pipeline to the user instead.
preflight_fail() {
  echo "ERROR: required AITER JIT lib missing or empty: $1" >&2
  echo "       This script REUSES prebuilt libs and never rebuilds them." >&2
  echo "       Run the canonical pipeline first (hours for MHA):" >&2
  echo "         make                                  # umbrella lib" >&2
  echo "         python3 jax_aiter/jit/build_jit.py    # AITER JIT libs" >&2
  echo "         make ja_mods                          # FFI shims" >&2
  exit 1
}

[[ -s "$AITER_BUILD/librmsnorm_fwd.so" ]] || preflight_fail "$AITER_BUILD/librmsnorm_fwd.so"
if [[ "$VARIANT" == "full" ]]; then
  [[ -s "$AITER_BUILD/libmha_fwd.so" ]] || preflight_fail "$AITER_BUILD/libmha_fwd.so"
  [[ -s "$AITER_BUILD/libmha_bwd.so" ]] || preflight_fail "$AITER_BUILD/libmha_bwd.so"
fi

# Step 3: snapshot sha256 of the JIT libs (the no-rebuild guard).
sha256sum "$AITER_BUILD"/*.so > "$PRE_SHA"
echo "[build_wheel] snapshotted $(wc -l < "$PRE_SHA") JIT libs -> $PRE_SHA"

# Step 4: clean ONLY the staging dirs (never build/aiter_build/).
echo "[build_wheel] make clean-stage"
make clean-stage

# Step 5: build the umbrella lib only if it is missing (cheap, ~5s).
if [[ ! -f "$UMBRELLA" ]]; then
  echo "[build_wheel] umbrella lib missing -> make"
  make
else
  echo "[build_wheel] umbrella lib present -> skip make"
fi

# Step 6: intentionally SKIP build_jit.py -- all AITER JIT libs are reused
# from build/aiter_build/ as-is (rebuilding MHA would cost hours).

# Step 7: build the thin FFI shims (incremental; hipcc no-ops if up-to-date).
if [[ "$VARIANT" == "lite" ]]; then
  echo "[build_wheel] make ja_mods_nomha (core shims only)"
  make ja_mods_nomha
else
  echo "[build_wheel] make ja_mods (core + MHA shims)"
  make ja_mods
fi

# Step 8: stale-artifact guard -- moe_fwd_ja.so has no source, must not ship.
rm -f build/jax_aiter_build/moe_fwd_ja.so

# Step 9: build the wheel. setup.py filters which existing *.so to stage by
# JA_WHEEL_VARIANT. --no-build-isolation reuses the container's setuptools
# (>=64) + wheel so no network fetch is needed.
echo "[build_wheel] pip wheel (JA_WHEEL_VARIANT=$VARIANT)"
JA_WHEEL_VARIANT="$VARIANT" python3 -m pip wheel . --no-deps --no-build-isolation \
  -w dist/ ${PIP_EXTRA_ARGS:-}

# Step 10: re-snapshot and assert the JIT libs are byte-identical.
sha256sum "$AITER_BUILD"/*.so > "$POST_SHA"
if ! diff -q "$PRE_SHA" "$POST_SHA" >/dev/null; then
  echo "ERROR: AITER JIT libs changed -- MHA must not be rebuilt" >&2
  echo "------ pre ------" >&2;  cat "$PRE_SHA"  >&2
  echo "------ post -----" >&2;  cat "$POST_SHA" >&2
  exit 1
fi
echo "[build_wheel] sha256 guard OK -- build/aiter_build/*.so unchanged"

# Step 11: report the wheel + total wall time.
echo "[build_wheel] wheel(s) in dist/:"
if [[ "$VARIANT" == "lite" ]]; then
  ls -lh dist/jax_aiter-*+lite-*.whl 2>/dev/null || ls -lh dist/
else
  ls -lh dist/
fi
echo "[build_wheel] DONE (variant=$VARIANT) in ${SECONDS}s wall time"
