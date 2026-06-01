#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Install the build tooling + the JAX 0.9.0 ROCm runtime stack INSIDE the CI
# container (ghcr.io/rocm/jax-base-ubu24.rocm720). Shared by ci.yml,
# nightly-ci.yml, cache-refresh.yml and nightly-perf.yml so the proven
# install commands live in one place.
#
# The pinned JAX stack = jax==0.9.0 + the three rocm-jax v0.9.0-rc3 release
# wheels (jaxlib / jax_rocm7_plugin / jax_rocm7_pjrt). These URLs are the
# canonical recipe -- keep them in sync with the workflows.
#
# Modes:
#   (default)     apt tooling + pip build tools + JAX stack + pytest tooling
#                 + mark repos safe for in-container git.
#   --jax-only    ONLY (force-)reinstall the JAX 0.9.0 stack. Used by the perf
#                 leg to restore our pins after MaxText pulls its own deps.
#
# Usage:
#   bash ci/setup_jax.sh
#   bash ci/setup_jax.sh --jax-only
set -euxo pipefail

JAX_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --jax-only) JAX_ONLY=1 ;;
    *) echo "ERROR: unknown arg '$arg' (expected --jax-only or nothing)." >&2; exit 2 ;;
  esac
done

JA_ROOT_DIR="${JA_ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# --- canonical pinned JAX 0.9.0 ROCm stack (install order matters) ---
JAX_PIN="jax==0.9.0"
JAXLIB_WHL="https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.9.0-rc3/jaxlib-0.9.0+rocm7-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
PLUGIN_WHL="https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.9.0-rc3/jax_rocm7_plugin-0.9.0+rocm7.2.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
PJRT_WHL="https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.9.0-rc3/jax_rocm7_pjrt-0.9.0+rocm7.2.0-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"

install_jax_stack() {
  # ${FORCE} is empty in default mode and "--force-reinstall --no-deps" in
  # --jax-only mode (where we must clobber whatever MaxText pulled in).
  # shellcheck disable=SC2086
  python3 -m pip install --break-system-packages ${FORCE:-} "$JAX_PIN"
  # shellcheck disable=SC2086
  python3 -m pip install --break-system-packages ${FORCE:-} "$JAXLIB_WHL"
  # shellcheck disable=SC2086
  python3 -m pip install --break-system-packages ${FORCE:-} "$PLUGIN_WHL" "$PJRT_WHL"
}

if [[ "$JAX_ONLY" == "1" ]]; then
  FORCE="--force-reinstall --no-deps"
  install_jax_stack
  echo "[ci/setup_jax] --jax-only: restored pinned JAX 0.9.0 ROCm stack."
  exit 0
fi

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  git ca-certificates curl build-essential pkg-config
python3 -m pip install --break-system-packages cmake ninja pyyaml psutil pandas

# JAX must be installed before the build (Makefile needs jax.ffi.include_dir()).
install_jax_stack
python3 -m pip install --break-system-packages pytest pytest-rerunfailures

# Mark the checked-out repos safe so in-container git calls (AITER JIT) don't
# trip over "detected dubious ownership".
git config --global --add safe.directory "$JA_ROOT_DIR" || true
git config --global --add safe.directory "$JA_ROOT_DIR/third_party/aiter" || true

echo "[ci/setup_jax] tooling + JAX 0.9.0 ROCm stack installed."
