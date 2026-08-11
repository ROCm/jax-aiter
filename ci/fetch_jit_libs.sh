#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# CONSUMER side of the prebuilt-JIT-lib release-asset flow.
#
# Downloads manifest.json + the compressed JIT libs from a GitHub release,
# VERIFIES the manifest against the current checkout (aiter submodule SHA +
# patch_hash + GPU archs are hard gates; ROCm is advisory), and on a match
# decompresses the libs into build/aiter_build/ and signals "skip build_jit".
# On any mismatch / missing asset / integrity failure it signals "rebuild" so
# ci/build.sh falls back to the (slow) source build.
#
# Decision is written to $GITHUB_OUTPUT as `rebuild=true|false` (+ `decision=`)
# and echoed. The script EXITS 0 for both skip-build and rebuild (a rebuild is
# an expected outcome, not an error); it exits non-zero only on a usage error.
#
# Release assets are public, so download uses plain curl (no token needed); it
# falls back to `gh release download` when curl can't reach the asset.
#
# Env overrides:
#   GH_RELEASE_TAG (default: immutable tag computed from current hard inputs)
#   GH_REPO (default ROCm/jax-aiter)
#   AITER_BUILD_DIR (default build/aiter_build)   GPU_ARCHS (default gfx950)
#   JA_MANIFEST_NAME (default manifest.json)
#   ROCM_VERSION + JA_FETCH_STRICT_ROCM=1 -> make the ROCm check a hard gate too
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

REPO="${GH_REPO:-ROCm/jax-aiter}"
AITER_BUILD_DIR="${AITER_BUILD_DIR:-build/aiter_build}"
GPU_ARCHS="${GPU_ARCHS:-gfx950}"
TAG="${GH_RELEASE_TAG:-$(python3 ci/jit_libs_manifest.py cache-id \
  --repo-root "$REPO_ROOT" --gpu-archs "$GPU_ARCHS" --prefix jit-libs)}"
MANIFEST_NAME="${JA_MANIFEST_NAME:-manifest.json}"
# Asset base URL. Defaults to the public release-download path; override
# JA_RELEASE_BASE_URL to point at a mirror/cache (assets served as <base>/<asset>).
BASE_URL="${JA_RELEASE_BASE_URL:-https://github.com/${REPO}/releases/download/${TAG}}"

signal() {  # $1=rebuild(true|false) $2=decision
  echo "[fetch] decision=$2 (rebuild=$1)"
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    { echo "rebuild=$1"; echo "decision=$2"; } >> "$GITHUB_OUTPUT"
  fi
}

rebuild() {  # $1=reason ; clean partial libs so build.sh definitely rebuilds
  echo "[fetch] REBUILD: $1"
  rm -f "$AITER_BUILD_DIR"/*.so 2>/dev/null || true
  signal true rebuild
  exit 0
}

download() {  # $1=asset-name $2=dest-path ; 0 on success
  local asset="$1" dest="$2"
  if curl -fsSL --retry 3 --retry-delay 2 -o "$dest" "${BASE_URL}/${asset}"; then
    return 0
  fi
  if command -v gh >/dev/null 2>&1; then
    echo "[fetch] curl failed for $asset; trying gh release download ..."
    if gh release download "$TAG" --repo "$REPO" --pattern "$asset" \
         --dir "$(dirname "$dest")" --clobber 2>/dev/null; then
      [[ "$(dirname "$dest")/$asset" != "$dest" ]] && mv -f "$(dirname "$dest")/$asset" "$dest" 2>/dev/null || true
      [[ -s "$dest" ]] && return 0
    fi
  fi
  return 1
}

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "[fetch] release=$REPO tag=$TAG  ->  $AITER_BUILD_DIR"
mkdir -p "$AITER_BUILD_DIR"

# --- 1. manifest ---
if ! download "$MANIFEST_NAME" "$TMP/manifest.json"; then
  rebuild "could not download $MANIFEST_NAME from release '$TAG' (asset missing?)"
fi

# --- 2. verify the manifest keys vs the current checkout ---
STRICT_ROCM_FLAG=""
[[ "${JA_FETCH_STRICT_ROCM:-0}" == "1" ]] && STRICT_ROCM_FLAG="--strict-rocm"
# Neutralize GITHUB_OUTPUT for the inner verify -- this script's signal()/rebuild()
# owns the single authoritative `rebuild=`/`decision=` line written at the end.
if GITHUB_OUTPUT="" python3 ci/jit_libs_manifest.py verify \
     --repo-root "$REPO_ROOT" \
     --manifest "$TMP/manifest.json" \
     --gpu-archs "$GPU_ARCHS" \
     ${ROCM_VERSION:+--rocm-version "$ROCM_VERSION"} \
     $STRICT_ROCM_FLAG; then
  echo "[fetch] manifest verify: PASS"
else
  rebuild "manifest verify failed (see reasons above)"
fi

# --- 3. download each compressed blob ---
mapfile -t BLOBS < <(python3 -c "import json,sys;print('\n'.join(f['compressed_name'] for f in json.load(open(sys.argv[1]))['files']))" "$TMP/manifest.json")
if [[ ${#BLOBS[@]} -eq 0 ]]; then
  rebuild "manifest lists no files"
fi
for blob in "${BLOBS[@]}"; do
  echo "[fetch] downloading $blob ..."
  if ! download "$blob" "$TMP/$blob"; then
    rebuild "could not download blob '$blob'"
  fi
done

# --- 4. integrity-check + decompress into AITER_BUILD_DIR ---
echo "[fetch] verifying + decompressing into $AITER_BUILD_DIR ..."
if python3 - "$TMP/manifest.json" "$TMP" "$AITER_BUILD_DIR" "$REPO_ROOT/ci/jit_libs_manifest.py" <<'PYEOF'
import sys, os, importlib.util
manifest_path, blobdir, outdir, jlm_path = sys.argv[1:5]
spec = importlib.util.spec_from_file_location("jlm", jlm_path)
jlm = importlib.util.module_from_spec(spec); spec.loader.exec_module(jlm)
m = jlm.load_manifest(manifest_path)
comp = m["compression"]
os.makedirs(outdir, exist_ok=True)
for f in m["files"]:
    blob = os.path.join(blobdir, f["compressed_name"])
    if not os.path.isfile(blob):
        print(f"  FAIL missing blob {f['compressed_name']}"); sys.exit(1)
    if jlm.sha256_file(blob) != f["compressed_sha256"]:
        print(f"  FAIL compressed sha mismatch {f['compressed_name']}"); sys.exit(1)
    out = os.path.join(outdir, f["name"])
    try:
        jlm.decompress_file(blob, out, comp)
    except Exception as e:
        print(f"  FAIL decompress {f['compressed_name']}: {e}"); sys.exit(1)
    if jlm.sha256_file(out) != f["sha256"] or os.path.getsize(out) != f["size"]:
        print(f"  FAIL roundtrip sha/size mismatch {f['name']}")
        try: os.remove(out)
        except OSError: pass
        sys.exit(1)
    print(f"  OK   {f['name']} ({os.path.getsize(out)} bytes, sha verified)")
print("[fetch] all libs verified + installed.")
PYEOF
then
  signal false skip-build
  echo "[fetch] prebuilt JIT libs ready -- ci/build.sh will skip build_jit.py."
  exit 0
else
  rebuild "blob integrity/decompress failed"
fi
