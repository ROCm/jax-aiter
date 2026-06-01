#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Build JAX-AITER INSIDE the CI container. Shared by ci.yml, nightly-ci.yml,
# cache-refresh.yml and nightly-perf.yml.
#
# Steps:
#   1. make                          -- umbrella libjax_aiter.so
#   2. build_jit.py (CONDITIONAL)    -- the multi-GB MHA JIT libs. Only built
#      when JA_DO_JIT_BUILD is truthy (cache miss) OR no *.so are present in
#      build/aiter_build/. On a cache hit the restored libs are reused.
#   3. make ja_mods                  -- the FFI shim modules
#   4. pip install .                 -- the jax-aiter wheel
#
# Env:
#   JA_DO_JIT_BUILD   "true"/"1"/"yes" forces the JIT build (set by the
#                     workflow when the JIT cache missed). Anything else =
#                     skip the JIT build unless the libs are missing.
set -euxo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# 1. umbrella shared library
make
ls -lh build/jax_aiter_build/libjax_aiter.so

# 2. JIT libs -- only on a cache miss or when the libs are absent (they cost
#    hours + GBs to build, so we reuse the cached copies whenever possible).
need_jit=0
case "${JA_DO_JIT_BUILD:-}" in
  1|true|TRUE|yes|YES) need_jit=1 ;;
esac
if ! ls build/aiter_build/*.so >/dev/null 2>&1; then
  need_jit=1
fi

if [[ "$need_jit" == "1" ]]; then
  echo "[ci/build] building AITER JIT modules (JA_DO_JIT_BUILD='${JA_DO_JIT_BUILD:-}')."
  python3 jax_aiter/jit/build_jit.py --verbose
else
  echo "[ci/build] JIT libs present and JA_DO_JIT_BUILD not set -> skipping build_jit.py (cache reuse)."
fi
ls -lh build/aiter_build/*.so

# 3. FFI modules
make ja_mods
ls -lh build/jax_aiter_build/*.so

# 4. install jax-aiter
pip install --break-system-packages .

echo "[ci/build] build + install complete."
