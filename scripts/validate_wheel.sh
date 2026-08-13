#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# HOST-side validation of a jax-aiter wheel (lite or full) in a CLEAN room.
# Builds docker/validation/Dockerfile.lite (ROCm + JAX base, NO AITER source,
# NO MaxText, NO build toolchain), then pip-installs the wheel in a fresh
# SIBLING container and runs the variant's smoke test(s). This proves a
# downstream user can install and run the wheel with no source tree.
#
#   lite -> FP4 GEMM smoke; then MHA must fail with the fetch command named.
#   full -> FP4 GEMM smoke + MHA flash-attn fwd/bwd smoke (smoke_mha.py),
#           proving the bundled libmha_fwd/bwd.so load + run from the wheel.
#
# NOTE: runs on the HOST and spawns a sibling docker container -- NOT inside
# rv_aiter. Needs host docker + GPU access (/dev/kfd, /dev/dri).
#
# Usage: scripts/validate_wheel.sh [--variant {lite|full}] [path/to/wheel.whl]
#   default variant = lite; default wheel = newest matching dist/ wheel.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="jax-aiter-validate:clean"
DOCKERFILE="$REPO/docker/validation/Dockerfile.lite"

VARIANT="lite"
WHEEL=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant) VARIANT="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) WHEEL="$1"; shift ;;
  esac
done

if [[ "$VARIANT" != "lite" && "$VARIANT" != "full" ]]; then
  echo "ERROR: --variant must be 'lite' or 'full' (got '$VARIANT')" >&2
  exit 1
fi

# Resolve the wheel to test (explicit path wins; else newest matching dist/).
if [[ -z "$WHEEL" ]]; then
  if [[ "$VARIANT" == "lite" ]]; then
    WHEEL="$(python3 - "$REPO/dist" <<'PY'
from pathlib import Path
import sys

wheels = sorted(
    (p for p in Path(sys.argv[1]).glob("jax_aiter-*.whl") if "+full" not in p.name),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
print(wheels[0] if wheels else "")
PY
)"
  else
    WHEEL="$(ls -t "$REPO"/dist/jax_aiter-*+full-*.whl 2>/dev/null | head -n1 || true)"
  fi
fi
if [[ -z "$WHEEL" || ! -f "$WHEEL" ]]; then
  echo "ERROR: no $VARIANT wheel found in dist/." >&2
  echo "       Build one first: bash scripts/build_wheel.sh --variant $VARIANT" >&2
  exit 1
fi
WHEEL_BASE="$(basename "$WHEEL")"
echo "[validate_wheel] variant=$VARIANT wheel=$WHEEL_BASE"

# Build the clean validation image (cached across runs).
echo "[validate_wheel] docker build -> $IMAGE_TAG"
docker build -t "$IMAGE_TAG" -f "$DOCKERFILE" "$REPO/docker/validation"

# Smoke command per variant.
if [[ "$VARIANT" == "lite" ]]; then
  SMOKE="python /test/smoke_fp4_gemm.py && \
    python /test/check_mha_guard.py && \
    jax-aiter-fetch-mha && \
    python /test/smoke_mha.py"
else
  SMOKE="python /test/smoke_fp4_gemm.py && python /test/smoke_mha.py"
fi

# This container runs as a non-root user on purpose, which means it can only
# open the GPU device nodes if that user is in the groups owning them. Group
# NAMES resolve against the container's /etc/group, which need not agree with
# the host, so pass the host's numeric GIDs. Without them /dev/dri is mounted
# yet hipInit still reports HIP_ERROR_NoDevice, and whether it works depends on
# which groups happen to own the device nodes on a given runner.
mapfile -t DEV_GIDS < <(
  for dev in /dev/kfd /dev/dri/renderD*; do
    [[ -e "$dev" ]] || continue
    stat -c '%g' "$dev"
  done | sort -u
)
GROUP_ARGS=(--group-add video)
for gid in "${DEV_GIDS[@]}"; do
  GROUP_ARGS+=(--group-add "$gid")
done
echo "[validate_wheel] device groups: ${GROUP_ARGS[*]}"

# Run the smoke test(s) in a fresh sibling container with GPU access.
# One GPU is enough for these smokes, and it matches the 1x validation job.
echo "[validate_wheel] docker run smoke ($VARIANT)"
docker run --rm \
  --device=/dev/kfd --device=/dev/dri "${GROUP_ARGS[@]}" \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp/ja-home -e XDG_CACHE_HOME=/tmp/ja-home/.cache \
  -e GPU_ARCHS=gfx950 -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
  -v "$REPO/dist:/dist:ro" \
  -v "$REPO/docker/validation:/test:ro" \
  "$IMAGE_TAG" \
  bash -lc "mkdir -p /tmp/ja-home /tmp/ja-install && \
    python -m pip install --target /tmp/ja-install --no-deps /dist/$WHEEL_BASE && \
    export PYTHONPATH=/tmp/ja-install PATH=/tmp/ja-install/bin:\$PATH && $SMOKE"

echo "[validate_wheel] PASS ($VARIANT)"
