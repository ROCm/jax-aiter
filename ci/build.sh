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
#   3b. make -f Makefile.kv ja_kv    -- the paged-KV shims (full path only)
#   3c. prebuild_pa_ragged.py        -- AOT paged-attention kernels they dlopen
#   4. python3 -m pip install .      -- the jax-aiter wheel
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
#   JA_SKIP_KV_BUILD  "true"/"1"/"yes" skips the paged-KV shims and their
#                     prebuilt paged-attention kernels.
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

# 3b. Paged-KV shims. They live in Makefile.kv rather than Makefile because
#     Makefile is one of the four compute_patch_hash() inputs behind CACHE_ID:
#     editing it rekeys the immutable JIT identity and orphans the multi-GB
#     prebuilt MHA libraries, so a one-line KV change would cost a 12 h rebuild.
#     That is an argument about which FILE the rules live in -- it was never a
#     reason not to BUILD them here. Skipping the target is why tests/test_*kv*
#     and tests/test_paged_* silently skipped in every CI run to date. The cold
#     build is 31 s wall (-j, see below) and the tests ~6 s, against a 12 h JIT
#     budget.
#     The lite path skips it: that variant exists to be small.
skip_kv=0
case "${JA_SKIP_KV_BUILD:-}" in
  1|true|TRUE|yes|YES) skip_kv=1 ;;
esac
if [[ "$JA_MODS_TARGET" == "ja_mods_nomha" ]]; then
  skip_kv=1
fi

if [[ "$skip_kv" == "1" ]]; then
  echo "[ci/build] skipping the paged-KV shims (lite path or JA_SKIP_KV_BUILD)."
else
  # -j is mandatory, not a nicety: the filtered ck_tile set is 257 translation
  # units and a measured cold build costs 87 CPU-minutes -- serially that is
  # past the 60 min gpu-job timeout on its own. Parallel it is 31 s wall on 256
  # cores. paged_prefill re-enters make and the jobserver is handed down, so
  # this one -j covers the inner build too.
  make -f Makefile.kv ja_kv -j"$(nproc)"
  ls -lh build/jax_aiter_build/*kv*.so build/jax_aiter_build/paged_*.so

  # 3c. paged_attention_ja.so resolves aiter's paged-attention kernel by dlopen
  #     from $HOME/.aiter/build and REFUSES to compile it on demand -- doing so
  #     would mean spawning a Python subprocess from inside an FFI handler. So
  #     the .so alone is not enough: without this, tests/test_paged_attention_ja
  #     fails with "kernel configuration ... is not built". The script's default
  #     set is by definition the one those tests exercise (bf16/fp16 x MHA/GQA/MQA,
  #     head 128, block 16), and a cold build of all 6 is ~24 s. AITER_ROOT_DIR
  #     is left unset deliberately: it is the only value the C++ and Python cache
  #     spellings agree on, and ci/test.sh runs in this same container.
  python3 scripts/prebuild_pa_ragged.py
fi

# 4. install jax-aiter (variant gates which *.so are staged into the wheel).
# `python3 -m pip`, never bare `pip`: the CI image ships a second, newer
# Python whose `pip` shadows the 3.12 one that setup_jax.sh installed into,
# and setup.py is `python_requires ==3.12.*`.
JA_WHEEL_VARIANT="$JA_WHEEL_VARIANT" python3 -m pip install --break-system-packages .

echo "[ci/build] build + install complete (variant=$JA_WHEEL_VARIANT, mods=$JA_MODS_TARGET)."
