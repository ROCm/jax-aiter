# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Install AITER's prebuilt JIT libraries from a ROCm/jax-aiter jit-libs release.

Usage: python3 docker/mxfp4/fetch_aiter_jit_libs.py <release tag> <destination directory>

The release must be built from the AITER commit and GPU architecture that
jax_aiter/jit_assets.py pins. Its ROCm version and JIT recipe hash may differ from
this checkout's; both are printed so the image log records them.
"""
from __future__ import annotations

import ast
import importlib.util
import lzma
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = "ROCm/jax-aiter"


def _pins() -> dict:
    tree = ast.parse((ROOT / "jax_aiter/jit_assets.py").read_text())
    return {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"AITER_SHA", "GPU_ARCHS", "ROCM_VERSION", "JIT_RECIPE_HASH"}
    }


def _manifest_tools():
    spec = importlib.util.spec_from_file_location("jit_libs_manifest", ROOT / "ci/jit_libs_manifest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _download(url: str, dest: Path) -> None:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, open(dest, "wb") as out:
                shutil.copyfileobj(response, out, length=8 << 20)
            return
        except OSError as exc:
            if attempt == 3:
                raise SystemExit(f"could not download {url}: {exc}") from exc
            time.sleep(2**attempt)


def main() -> None:
    tag, dest = sys.argv[1], Path(sys.argv[2])
    pins = _pins()
    tools = _manifest_tools()
    base = f"https://github.com/{REPO}/releases/download/{tag}"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _download(f"{base}/manifest.json", tmp / "manifest.json")
        manifest = tools.load_manifest(tmp / "manifest.json")
        expected = {"tag": tag, "aiter_sha": pins["AITER_SHA"], "gpu_archs": pins["GPU_ARCHS"]}
        wrong = [
            f"{key}: release {manifest.get(key)!r}, this checkout {value!r}"
            for key, value in expected.items()
            if manifest.get(key) != value
        ]
        if wrong:
            raise SystemExit(f"{tag} does not match this checkout:\n  " + "\n  ".join(wrong))
        print(
            f"{tag}: AITER {manifest['aiter_sha']}, {manifest['gpu_archs']}, "
            f"built with ROCm {manifest.get('rocm_version')} (this checkout: {pins['ROCM_VERSION']}), "
            f"JIT recipe {manifest.get('patch_hash')} (this checkout: {pins['JIT_RECIPE_HASH']})",
            flush=True,
        )
        dest.mkdir(parents=True, exist_ok=True)
        for entry in manifest["files"]:
            blob = tmp / entry["compressed_name"]
            _download(f"{base}/{entry['compressed_name']}", blob)
            if tools.sha256_file(blob) != entry["compressed_sha256"]:
                raise SystemExit(f"checksum mismatch: {entry['compressed_name']}")
            library = dest / entry["name"]
            if manifest["compression"] == "xz":
                with lzma.open(blob) as source, open(library, "wb") as out:
                    shutil.copyfileobj(source, out, length=8 << 20)
            else:
                tools.decompress_file(blob, library, manifest["compression"])
            if tools.sha256_file(library) != entry["sha256"]:
                raise SystemExit(f"checksum mismatch after extraction: {entry['name']}")
            library.chmod(0o755)
            blob.unlink()
            print(f"  {entry['name']}: sha256 {entry['sha256']}", flush=True)
        shutil.copy2(tmp / "manifest.json", dest / "manifest.json")


if __name__ == "__main__":
    main()
