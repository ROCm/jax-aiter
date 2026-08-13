#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Print the content-addressed reference of the clean-room validation image.
#
# Single source of truth for that reference: validation-image.yml builds and
# pushes it, scripts/validate_wheel.sh pulls it. The tag is derived from the
# Dockerfile, so editing the recipe yields a new tag and the next run rebuilds
# instead of silently validating against a stale environment.
#
# Env overrides:
#   JA_VALIDATE_IMAGE_REPO  registry/repository (default ghcr.io/rocm/jax-aiter-validate)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKERFILE="$REPO/docker/validation/Dockerfile.lite"
IMAGE_REPO="${JA_VALIDATE_IMAGE_REPO:-ghcr.io/rocm/jax-aiter-validate}"

if [[ ! -f "$DOCKERFILE" ]]; then
  echo "ERROR: missing $DOCKERFILE" >&2
  exit 1
fi

# 12 hex chars is plenty to separate recipes and keeps the tag readable.
RECIPE_HASH="$(sha256sum "$DOCKERFILE" | cut -c1-12)"
echo "${IMAGE_REPO}:v${RECIPE_HASH}"
