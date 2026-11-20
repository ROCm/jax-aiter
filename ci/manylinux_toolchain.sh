# shellcheck shell=bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Point hipcc at the C++ standard library that the manylinux build image
# intends to be used. SOURCE this file inside the build container, before any
# `make` that compiles a shim:
#
#   source ci/manylinux_toolchain.sh
#
# Why. ghcr.io/rocm/jax-manylinux_2_28-therock-10.0 is AlmaLinux 8 with two GCC
# installations: the distribution's GCC 8 under /usr/lib/gcc, and gcc-toolset
# under /opt/rh, which is the one on PATH. ROCm's clang does not look at PATH.
# It searches /usr/lib/gcc, finds only GCC 8, and takes its libstdc++ headers.
# GCC 8 has no <bit> (added in GCC 9), so any shim that reaches a C++20 library
# header fails with
#
#   fatal error: 'bit' file not found
#
# which is what stopped mha_fwd_ja.so, and with it the nightly, once the OPUS
# launch header came into its include chain.
#
# gcc-toolset is the manylinux-supported way to use a newer libstdc++ on this
# base: the parts the system libstdc++.so.6 lacks are linked in statically, so
# the shims keep the same GLIBCXX requirement as before.
#
# This is an environment setting rather than a Makefile flag on purpose. The
# Makefile is an input to the JIT library cache key, and this only concerns
# one image.

_ja_gxx="$(command -v g++ || true)"
if [[ -z "$_ja_gxx" ]]; then
  echo "ERROR: ci/manylinux_toolchain.sh: no g++ on PATH." >&2
  return 1 2>/dev/null || exit 1
fi

# The directory of libgcc.a is what clang calls the "GCC installation".
_ja_gcc_dir="$(dirname "$("$_ja_gxx" -print-libgcc-file-name)")"
if [[ ! -d "$_ja_gcc_dir" ]]; then
  echo "ERROR: ci/manylinux_toolchain.sh: $_ja_gxx reports GCC installation" \
       "'$_ja_gcc_dir', which does not exist." >&2
  return 1 2>/dev/null || exit 1
fi

# hipcc appends these to every compile and link command it runs. Both are
# needed: the Makefile compiles and links each shim in a single hipcc call.
_ja_flag="--gcc-install-dir=$_ja_gcc_dir"
if [[ " ${HIPCC_COMPILE_FLAGS_APPEND:-} " != *" $_ja_flag "* ]]; then
  export HIPCC_COMPILE_FLAGS_APPEND="${HIPCC_COMPILE_FLAGS_APPEND:+$HIPCC_COMPILE_FLAGS_APPEND }$_ja_flag"
fi
if [[ " ${HIPCC_LINK_FLAGS_APPEND:-} " != *" $_ja_flag "* ]]; then
  export HIPCC_LINK_FLAGS_APPEND="${HIPCC_LINK_FLAGS_APPEND:+$HIPCC_LINK_FLAGS_APPEND }$_ja_flag"
fi

echo "[ci/manylinux_toolchain] hipcc uses the GCC installation at $_ja_gcc_dir"
unset _ja_gxx _ja_gcc_dir _ja_flag
