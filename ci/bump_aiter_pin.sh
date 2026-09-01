#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Move the consumed AITER pin to the tip of an upstream ref.
#
# The pin lives in TWO places that must never disagree:
#   1. the `third_party/aiter` submodule gitlink -- what actually gets built;
#   2. jax_aiter/jit_assets.py:AITER_SHA -- a CACHE_ID input, so it decides
#      which prebuilt JIT release ci/fetch_jit_libs.sh will accept.
# Updating only (1) yields a build of the new AITER that then loads JIT libs
# compiled against the old one, which is silently wrong rather than a failure.
#
# Writes `aiter_sha`/`previous_sha`/`moved` to $GITHUB_OUTPUT when set. Exits 0
# whether or not the pin moved -- "already current" is a normal nightly outcome,
# not an error; read `moved` to branch on it.
#
# Usage: ci/bump_aiter_pin.sh [--ref main] [--no-commit]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

AITER_REF="main"
DO_COMMIT=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ref) AITER_REF="$2"; shift 2 ;;
    --no-commit) DO_COMMIT=0; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

ASSETS="jax_aiter/jit_assets.py"
test -f "$ASSETS" || { echo "missing $ASSETS" >&2; exit 1; }

previous="$(git rev-parse HEAD:third_party/aiter)"
git -C third_party/aiter fetch --depth=1 origin "$AITER_REF"
target="$(git -C third_party/aiter rev-parse FETCH_HEAD)"

emit() {
  echo "[bump] previous=$previous target=$target moved=$1"
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    {
      echo "aiter_sha=$target"
      echo "previous_sha=$previous"
      echo "moved=$1"
    } >> "$GITHUB_OUTPUT"
  fi
}

if [[ "$previous" == "$target" ]]; then
  emit false
  exit 0
fi

git -C third_party/aiter checkout --detach "$target"
# Recurse: an AITER bump can move composable_kernel and the other nested
# submodules with it, and a stale nested checkout builds the wrong kernels.
git -C third_party/aiter submodule update --init --recursive --depth=1

# Rewrite the single AITER_SHA assignment. A line-anchored regex, not a blind
# string replace: the file also carries CACHE_ID and asset names that embed
# hashes, and clobbering those would break the JIT-release lookup.
python3 - "$ASSETS" "$target" <<'PY'
import re
import sys

path, target = sys.argv[1], sys.argv[2]
with open(path) as fh:
    text = fh.read()
new, n = re.subn(
    r'(?m)^(AITER_SHA\s*=\s*)"[0-9a-f]{40}"',
    lambda m: f'{m.group(1)}"{target}"',
    text,
)
if n != 1:
    raise SystemExit(f"expected exactly 1 AITER_SHA assignment, found {n}")
with open(path, "w") as fh:
    fh.write(new)
print(f"[bump] {path}: AITER_SHA -> {target}")
PY

git add third_party/aiter "$ASSETS"

if [[ "$DO_COMMIT" == "1" ]]; then
  git -c user.name="jax-aiter-nightly[bot]" \
      -c user.email="jax-aiter-nightly@users.noreply.github.com" \
      commit -m "chore(aiter): bump pin to ${target:0:12}

Automated nightly bump of third_party/aiter and jax_aiter/jit_assets.py.
Promoted only after the full test suite and the 8-GPU MXFP4 throughput gate
both passed against this pin.

Previous: $previous
New:      $target"
fi

emit true
