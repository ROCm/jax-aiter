#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# PRODUCER side of the prebuilt-JIT-lib release-asset flow.
#
# Compresses the 3 multi-GB AITER JIT libs from build/aiter_build/ + emits a
# manifest.json (keyed by aiter_sha + gpu_archs + rocm_version + patch_hash),
# round-trip-verifies the blobs, then uploads them to a GitHub release with
# `gh release upload --clobber`. Consumed by ci/fetch_jit_libs.sh.
#
# Compression: prefers `zstd -19 --long` (best ratio), else `xz`, else `gzip`.
# Each compressed blob is asserted < 2 GiB (a GitHub release-asset limit).
#
# `gh` auth: GITHUB_TOKEN (CI, needs `contents: write`) or a local PAT. A 403 on
# upload is reported with the exact manual command + staged paths instead of
# silently dropping the work (use --allow-upload-fail to make a 403 non-fatal).
#
# Usage:
#   ci/publish_jit_libs.sh [--tag TAG] [--repo OWNER/REPO] [--dist-dir DIR]
#                          [--no-upload] [--allow-upload-fail]
#
# Env overrides:
#   GH_RELEASE_TAG (default v0.1.0-alpha)   GH_REPO (default ROCm/jax-aiter)
#   GPU_ARCHS (default "gfx950")             ROCM_VERSION (recorded in manifest)
#   JA_ZSTD_LEVEL (default 19)               AITER_BUILD_DIR (default build/aiter_build)
#   JA_LIBS (space list; default the 3 JIT libs)   JA_DIST_DIR (default build/jit_libs_dist)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

REPO="${GH_REPO:-ROCm/jax-aiter}"
DIST_DIR="${JA_DIST_DIR:-build/jit_libs_dist}"
AITER_BUILD_DIR="${AITER_BUILD_DIR:-build/aiter_build}"
GPU_ARCHS="${GPU_ARCHS:-gfx950}"
ROCM_VERSION="${ROCM_VERSION:-7.14.0}"
TAG="${GH_RELEASE_TAG:-$(python3 ci/jit_libs_manifest.py cache-id \
  --repo-root "$REPO_ROOT" --gpu-archs "$GPU_ARCHS" \
  --rocm-version "$ROCM_VERSION" --prefix jit-libs)}"
ZSTD_LEVEL="${JA_ZSTD_LEVEL:-19}"
DO_UPLOAD=1
ALLOW_UPLOAD_FAIL=0
# shellcheck disable=SC2206  # intentional word-split into a list
LIBS=(${JA_LIBS:-librmsnorm_fwd.so libmha_fwd.so libmha_bwd.so})

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --dist-dir) DIST_DIR="$2"; shift 2 ;;
    --no-upload) DO_UPLOAD=0; shift ;;
    --allow-upload-fail) ALLOW_UPLOAD_FAIL=1; shift ;;
    -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "ERROR: unknown arg '$1'" >&2; exit 2 ;;
  esac
done

GIB=$((1024 * 1024 * 1024))
MAX_BLOB=$((2 * GIB))

# --- 1. preflight: the source-of-truth libs must already exist (NO rebuild) ---
for lib in "${LIBS[@]}"; do
  if [[ ! -s "$AITER_BUILD_DIR/$lib" ]]; then
    echo "ERROR: required JIT lib missing or empty: $AITER_BUILD_DIR/$lib" >&2
    echo "       publish_jit_libs.sh REUSES prebuilt libs; build them first via" >&2
    echo "       'python3 jax_aiter/jit/build_jit.py' (hours for the MHA libs)." >&2
    exit 1
  fi
done

# --- 2. choose the compressor (prefer zstd -19 --long) ---
if command -v zstd >/dev/null 2>&1; then
  COMPRESSION=zstd; SUFFIX=".zst"
elif command -v xz >/dev/null 2>&1; then
  COMPRESSION=xz; SUFFIX=".xz"
elif command -v gzip >/dev/null 2>&1; then
  COMPRESSION=gzip; SUFFIX=".gz"
else
  echo "ERROR: no compressor found (need zstd, xz or gzip)." >&2
  exit 1
fi
echo "[publish] compressor=$COMPRESSION  tag=$TAG  repo=$REPO  dist=$DIST_DIR"

mkdir -p "$DIST_DIR"

compress_one() {  # $1=src .so  $2=dst blob
  local src="$1" dst="$2"
  case "$COMPRESSION" in
    zstd) zstd "-${ZSTD_LEVEL}" --long=27 -T0 -f -q -o "$dst" "$src" ;;
    xz)   xz   -T0 -6 -f -c "$src" > "$dst" ;;
    gzip) gzip -6 -f -c "$src" > "$dst" ;;
  esac
}

# --- 3. compress each lib + size-check ---
for lib in "${LIBS[@]}"; do
  src="$AITER_BUILD_DIR/$lib"
  dst="$DIST_DIR/${lib}${SUFFIX}"
  raw_sz=$(stat -c '%s' "$src")
  echo "[publish] compressing $lib ($(numfmt --to=iec "$raw_sz")) -> ${lib}${SUFFIX} ..."
  compress_one "$src" "$dst"
  comp_sz=$(stat -c '%s' "$dst")
  echo "[publish]   -> $(numfmt --to=iec "$comp_sz") ($(awk "BEGIN{printf \"%.1f\", $raw_sz/$comp_sz}")x)"
  if (( comp_sz >= MAX_BLOB )); then
    echo "ERROR: $dst is $(numfmt --to=iec "$comp_sz") >= 2 GiB release-asset limit." >&2
    exit 1
  fi
done

# --- 4. emit manifest.json ---
echo "[publish] emitting manifest.json ..."
LIBS_CSV="$(IFS=,; echo "${LIBS[*]}")"
python3 ci/jit_libs_manifest.py emit \
  --repo-root "$REPO_ROOT" \
  --aiter-build-dir "$AITER_BUILD_DIR" \
  --dist-dir "$DIST_DIR" \
  --compression "$COMPRESSION" \
  --tag "$TAG" \
  --gpu-archs "$GPU_ARCHS" \
  ${ROCM_VERSION:+--rocm-version "$ROCM_VERSION"} \
  --libs "$LIBS_CSV" \
  --out "$DIST_DIR/manifest.json"

# --- 5. round-trip verify each blob (decompress + sha256 == manifest) ---
echo "[publish] round-trip verifying compressed blobs ..."
python3 - "$DIST_DIR/manifest.json" "$DIST_DIR" "$REPO_ROOT/ci/jit_libs_manifest.py" <<'PYEOF'
import sys, tempfile, os, importlib.util
manifest_path, dist, jlm_path = sys.argv[1], sys.argv[2], sys.argv[3]
spec = importlib.util.spec_from_file_location("jlm", jlm_path)
jlm = importlib.util.module_from_spec(spec); spec.loader.exec_module(jlm)
m = jlm.load_manifest(manifest_path)
comp = m["compression"]
fail = 0
for f in m["files"]:
    blob = os.path.join(dist, f["compressed_name"])
    got_csha = jlm.sha256_file(blob)
    if got_csha != f["compressed_sha256"]:
        print(f"  FAIL compressed sha mismatch: {f['compressed_name']}"); fail += 1; continue
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, f["name"])
        jlm.decompress_file(blob, out, comp)
        got = jlm.sha256_file(out)
        if got != f["sha256"]:
            print(f"  FAIL roundtrip sha mismatch: {f['name']}"); fail += 1
        elif os.path.getsize(out) != f["size"]:
            print(f"  FAIL roundtrip size mismatch: {f['name']}"); fail += 1
        else:
            print(f"  OK   {f['name']} <-> {f['compressed_name']} (sha256 verified)")
sys.exit(1 if fail else 0)
PYEOF
echo "[publish] round-trip verification passed."

# --- 6. report staged artifacts + the exact upload command ---
UPLOAD_FILES=("$DIST_DIR/manifest.json")
for lib in "${LIBS[@]}"; do UPLOAD_FILES+=("$DIST_DIR/${lib}${SUFFIX}"); done

echo "[publish] staged artifacts:"
ls -lh "${UPLOAD_FILES[@]}" | sed 's/^/    /'
UPLOAD_CMD="gh release upload \"$TAG\" --repo \"$REPO\" --clobber ${UPLOAD_FILES[*]}"

if [[ "$DO_UPLOAD" == "0" ]]; then
  echo "[publish] --no-upload: skipping upload. To upload manually run:"
  echo "    $UPLOAD_CMD"
  exit 0
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "[publish] WARN: gh not found; staged only. To upload run:" >&2
  echo "    $UPLOAD_CMD" >&2
  [[ "$ALLOW_UPLOAD_FAIL" == "1" ]] && exit 0 || exit 1
fi

# --- 7. upload (gh uses GITHUB_TOKEN in CI or a local PAT) ---
echo "[publish] uploading to release '$TAG' on '$REPO' ..."
if gh release upload "$TAG" --repo "$REPO" --clobber "${UPLOAD_FILES[@]}"; then
  echo "[publish] upload OK. Assets attached to release '$TAG':"
  for f in "${UPLOAD_FILES[@]}"; do echo "    $(basename "$f")"; done
  exit 0
fi

# Upload failed (commonly HTTP 403 = token lacks `contents: write`).
echo "" >&2
echo "[publish] ERROR: 'gh release upload' failed (often HTTP 403: the token" >&2
echo "          lacks 'contents: write', or the release/tag '$TAG' is missing)." >&2
echo "          The compressed libs + manifest ARE staged and intact at:" >&2
for f in "${UPLOAD_FILES[@]}"; do echo "            $f" >&2; done
echo "          Fix the PAT (grant contents:write) and re-run, or drag-drop the" >&2
echo "          files onto the release in the web UI. Manual upload command:" >&2
echo "            $UPLOAD_CMD" >&2
[[ "$ALLOW_UPLOAD_FAIL" == "1" ]] && exit 0 || exit 1
