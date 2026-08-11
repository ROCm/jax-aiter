#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Install build tooling + the ROCm JAX runtime INSIDE the CI container
# (ghcr.io/rocm/jax-dev-ubu24.therock-7.14:7.14). Shared by every workflow so
# the install commands live in one place.
#
# The stack is plain PyPI: jax/jaxlib from upstream, and the ROCm backend from
# the jax-rocm7-{plugin,pjrt} wheels AMD publishes there. Those wheels are
# TheRock builds -- their RUNPATH resolves through the _rocm_sdk_* pip packages
# that the container already provides -- so nothing here installs ROCm itself.
#
# Modes:
#   (default)     apt tooling + pip build tools + JAX stack + pytest tooling
#                 + mark repos safe for in-container git.
#   --jax-only    ONLY (force-)reinstall the JAX stack. Used by the perf leg to
#                 restore our pins after MaxText pulls its own deps.
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

# Overridable so a bump is a workflow input, not a script edit.
JAX_VERSION="${JA_JAX_VERSION:-0.11.0}"
ROCM_PLUGIN_VERSION="${JA_ROCM_PLUGIN_VERSION:-0.11.0}"

install_jax_stack() {
  # ${FORCE} is empty in default mode and "--force-reinstall --no-deps" in
  # --jax-only mode, where we must clobber whatever MaxText pulled in.
  # shellcheck disable=SC2086
  python3 -m pip install --break-system-packages ${FORCE:-} \
    "jax==${JAX_VERSION}" "jaxlib==${JAX_VERSION}"
  # shellcheck disable=SC2086
  python3 -m pip install --break-system-packages ${FORCE:-} \
    "jax-rocm7-plugin==${ROCM_PLUGIN_VERSION}" "jax-rocm7-pjrt==${ROCM_PLUGIN_VERSION}"
}

if [[ "$JAX_ONLY" == "1" ]]; then
  FORCE="--force-reinstall --no-deps"
  install_jax_stack
  echo "[ci/setup_jax] --jax-only: restored jax ${JAX_VERSION} + rocm plugin ${ROCM_PLUGIN_VERSION}."
  exit 0
fi

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
  git ca-certificates curl build-essential pkg-config zstd
python3 -m pip install --break-system-packages cmake ninja pyyaml psutil pandas

# JAX must be installed before the build (Makefile needs jax.ffi.include_dir()).
install_jax_stack
python3 -m pip install --break-system-packages pytest pytest-rerunfailures

# Mark the checked-out repos safe so in-container git calls (AITER JIT) don't
# trip over "detected dubious ownership".
git config --global --add safe.directory "$JA_ROOT_DIR" || true
git config --global --add safe.directory "$JA_ROOT_DIR/third_party/aiter" || true

# Fail here rather than deep in a build if the ROCm backend did not resolve.
python3 - <<'PY'
import importlib
import jax, jaxlib
print(f"[ci/setup_jax] jax {jax.__version__} / jaxlib {jaxlib.__version__}")
for mod in ("jax_rocm7_plugin", "jax_rocm7_pjrt"):
    importlib.import_module(mod)
    print(f"[ci/setup_jax] {mod} importable")
PY

echo "[ci/setup_jax] tooling + jax ${JAX_VERSION} ROCm stack installed."
