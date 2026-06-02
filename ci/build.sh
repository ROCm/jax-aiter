#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Build JAX-AITER INSIDE the CI container. Shared by ci.yml, nightly-ci.yml,
# nightly-perf.yml and release-publish.yml.
#
# Steps:
#   1. make                          -- umbrella libjax_aiter.so
#   2. build_jit.py (CONDITIONAL)    -- the multi-GB MHA JIT libs. Only built
#      when JA_DO_JIT_BUILD is truthy OR no *.so are present in
#      build/aiter_build/. The normal CI path DOWNLOADS prebuilt libs first
#      (ci/fetch_jit_libs.sh) and sets JA_DO_JIT_BUILD=false, so this build is
#      the FALLBACK only (cold release / aiter bump / patch change).
#      JA_SKIP_JIT_BUILD short-circuits this step entirely (perf/lite path that
#      needs ZERO JIT libs -- see below).
#   3. make ja_mods[_nomha]          -- the FFI shim modules
#   4. pip install .                 -- the jax-aiter wheel
#
# Env:
#   JA_SKIP_JIT_BUILD "true"/"1"/"yes" SKIPS build_jit.py entirely (takes
#                     precedence over JA_DO_JIT_BUILD / lib presence). The
#                     MHA-independent MXFP4 perf/lite path sets this: its FP4
#                     GEMM + MXFP4 cast + BF16 GEMM shims are produced by
#                     `make` + `make ja_mods_nomha`, so it needs ZERO JIT libs
#                     and an empty build/aiter_build/ is NOT an error.
#   JA_DO_JIT_BUILD   "true"/"1"/"yes" forces the JIT build (set by the workflow
#                     when the prebuilt-lib fetch missed). Anything else = skip
#                     the JIT build unless the libs are missing. Ignored when
#                     JA_SKIP_JIT_BUILD is set.
#   JA_JIT_MODULES    comma-list passed to `build_jit.py --module` (e.g.
#                     "librmsnorm_fwd"). Empty = all 3. Ignored when
#                     JA_SKIP_JIT_BUILD is set.
#   JA_MODS_TARGET    make target for the FFI shims: "ja_mods" (default, core +
#                     MHA) or "ja_mods_nomha" (LITE: core shims only).
#   JA_WHEEL_VARIANT  "full" (default) or "lite" -- passed to the pip install
#                     so a lite build never expects the MHA libs/shims.
set -euxo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

JA_MODS_TARGET="${JA_MODS_TARGET:-ja_mods}"
JA_WHEEL_VARIANT="${JA_WHEEL_VARIANT:-full}"

# 1. umbrella shared library
make
ls -lh build/jax_aiter_build/libjax_aiter.so

# 2. JIT libs -- only when forced (fetch missed) or absent. They cost hours +
#    GBs, so the normal path reuses the prebuilt libs fetched from the release.
#    JA_SKIP_JIT_BUILD short-circuits this entirely: the MHA-independent MXFP4
#    perf/lite path needs ZERO JIT libs (its FP4 GEMM + MXFP4 cast + BF16 GEMM
#    shims all come from `make` + `make ja_mods_nomha`), so build_jit.py is
#    never invoked and an empty/absent build/aiter_build/ is not an error.
skip_jit=0
case "${JA_SKIP_JIT_BUILD:-}" in
  1|true|TRUE|yes|YES) skip_jit=1 ;;
esac

if [[ "$skip_jit" == "1" ]]; then
  echo "[ci/build] JA_SKIP_JIT_BUILD set -> skipping build_jit.py entirely (no JIT libs; FP4/BF16 shims come from make + ja_mods_nomha)."
else
  need_jit=0
  case "${JA_DO_JIT_BUILD:-}" in
    1|true|TRUE|yes|YES) need_jit=1 ;;
  esac
  if ! ls build/aiter_build/*.so >/dev/null 2>&1; then
    need_jit=1
  fi

  if [[ "$need_jit" == "1" ]]; then
    echo "[ci/build] building AITER JIT modules (JA_DO_JIT_BUILD='${JA_DO_JIT_BUILD:-}', modules='${JA_JIT_MODULES:-all}')."
    if [[ -n "${JA_JIT_MODULES:-}" ]]; then
      python3 jax_aiter/jit/build_jit.py --module "$JA_JIT_MODULES" --verbose
    else
      python3 jax_aiter/jit/build_jit.py --verbose
    fi
  else
    echo "[ci/build] prebuilt JIT libs present and JA_DO_JIT_BUILD not set -> skipping build_jit.py (release-asset reuse)."
  fi
fi

# List JIT libs when present; the no-JIT perf/lite path legitimately has none,
# so do not let a glob miss abort the script under `set -e`.
ls -lh build/aiter_build/*.so 2>/dev/null \
  || echo "[ci/build] no JIT libs in build/aiter_build/ (expected when JA_SKIP_JIT_BUILD is set)."

# 3. FFI modules (ja_mods = core + MHA; ja_mods_nomha = lite core-only).
make "$JA_MODS_TARGET"
ls -lh build/jax_aiter_build/*.so

# 4. install jax-aiter (variant gates which *.so are staged into the wheel).
JA_WHEEL_VARIANT="$JA_WHEEL_VARIANT" pip install --break-system-packages .

echo "[ci/build] build + install complete (variant=$JA_WHEEL_VARIANT, mods=$JA_MODS_TARGET)."
