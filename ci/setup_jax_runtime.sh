#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Install ONLY the ROCm JAX runtime, for legs that consume a prebuilt wheel and
# compile nothing (the nightly 8-GPU perf leg). No apt, no build tooling.
#
# Why this is a separate file rather than a flag on ci/setup_jax.sh:
# ci/setup_jax.sh is one of the inputs to compute_patch_hash() in
# ci/jit_libs_manifest.py, so editing it by even one byte rotates CACHE_ID and
# orphans every published JIT-libs release -- including the one the shipped
# alpha2 wheel fetches at runtime via jax_aiter/jit_assets.py. A change that
# provably cannot alter JIT library bytes must not invalidate them, so this
# lives outside that hash. The four pip lines below are duplicated from
# setup_jax.sh's install_jax_stack() for exactly that reason; keep them in
# sync, and keep the version defaults identical.
#
# The apt block in setup_jax.sh is what made this necessary: some nodes in the
# linux-jax-aiter-mi355-8 ARC pool deny even uid 0 a writable
# /var/lib/apt/lists inside docker exec, so `apt-get update` exits 100 (runs
# 33458425029, 33467884782). A probe on that same pool (33474028161) came back
# uid=0 with the directory writable, so the denial is per-node, not per-pool.
# This leg needs none of git/curl/build-essential/pkg-config/zstd beyond what
# the runtime image already ships, so the fix is to not ask for them.
#
# Usage:
#   bash ci/setup_jax_runtime.sh
set -euxo pipefail

JA_ROOT_DIR="${JA_ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

JAX_VERSION="${JA_JAX_VERSION:-0.11.0}"
ROCM_PLUGIN_VERSION="${JA_ROCM_PLUGIN_VERSION:-0.11.0}"

# Not --no-deps: a bare runtime container has no numpy, and the verify step
# below imports jax.
python3 -m pip install --break-system-packages \
  "jax==${JAX_VERSION}" "jaxlib==${JAX_VERSION}"
python3 -m pip install --break-system-packages \
  "jax-rocm7-plugin==${ROCM_PLUGIN_VERSION}" "jax-rocm7-pjrt==${ROCM_PLUGIN_VERSION}"

git config --global --add safe.directory "$JA_ROOT_DIR" || true
git config --global --add safe.directory "$JA_ROOT_DIR/third_party/aiter" || true

python3 - "$JAX_VERSION" "$ROCM_PLUGIN_VERSION" <<'PY'
import importlib
from importlib import metadata
import sys

import jax
import jaxlib

expected_jax, expected_rocm = sys.argv[1:]
assert jax.__version__ == expected_jax, (jax.__version__, expected_jax)
assert jaxlib.__version__ == expected_jax, (jaxlib.__version__, expected_jax)

for distribution, module in (
    ("jax-rocm7-plugin", "jax_rocm7_plugin"),
    ("jax-rocm7-pjrt", "jax_plugins.xla_rocm7"),
):
    version = metadata.version(distribution)
    assert version == expected_rocm, (distribution, version, expected_rocm)
    imported = importlib.import_module(module)
    print(f"[ci/setup_jax_runtime] {distribution} {version}: {module} -> {imported.__file__}")

print(f"[ci/setup_jax_runtime] jax {jax.__version__} / jaxlib {jaxlib.__version__}")
PY

echo "[ci/setup_jax_runtime] jax ${JAX_VERSION} ROCm stack installed (no apt, no build tooling)."
