#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# HOST-side validation of the jax-aiter lite wheel. Builds the clean-room
# validation image (docker/validation/Dockerfile.lite -- ROCm + JAX base,
# NO AITER source, NO MaxText), then runs the FP4 GEMM smoke test inside a
# fresh SIBLING container with the wheel pip-installed at run time. This
# proves a downstream user can install and run the lite wheel with no source.
#
# NOTE: this runs on the HOST and spawns a sibling docker container -- it is
# NOT executed inside rv_aiter.
#
# Usage: scripts/validate_wheel.sh [path/to/wheel.whl]
#   default wheel = newest dist/jax_aiter-*+lite-*.whl

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="jax-aiter-validate:lite"
DOCKERFILE="$REPO/docker/validation/Dockerfile.lite"

WHEEL="${1:-}"
if [[ -z "$WHEEL" ]]; then
  WHEEL="$(ls -t "$REPO"/dist/jax_aiter-*+lite-*.whl 2>/dev/null | head -n1 || true)"
fi
if [[ -z "$WHEEL" || ! -f "$WHEEL" ]]; then
  echo "ERROR: no lite wheel found (looked for dist/jax_aiter-*+lite-*.whl)." >&2
  echo "       Build one first: bash scripts/build_wheel.sh --variant lite" >&2
  exit 1
fi
WHEEL_BASE="$(basename "$WHEEL")"
echo "[validate_wheel] wheel = $WHEEL_BASE"

# Build the clean validation image.
echo "[validate_wheel] docker build -> $IMAGE_TAG"
docker build -t "$IMAGE_TAG" -f "$DOCKERFILE" "$REPO/docker/validation"

# Run the smoke test in a fresh sibling container with GPU access.
echo "[validate_wheel] docker run FP4 GEMM smoke"
docker run --rm \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  -v "$REPO/dist:/dist:ro" \
  -v "$REPO/docker/validation:/test:ro" \
  "$IMAGE_TAG" \
  bash -lc "pip install --break-system-packages /dist/$WHEEL_BASE && python /test/smoke_fp4_gemm.py"

echo "[validate_wheel] PASS"
