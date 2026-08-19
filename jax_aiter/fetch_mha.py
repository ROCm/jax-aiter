# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Download the prebuilt AITER JIT libraries that back flash attention.

Installed as the ``jax-aiter-fetch-mha`` command.

The attention kernels live in three libraries totalling ~2.6 GB uncompressed,
which is why the default wheel does not carry them: shipping them would take the
install from tens of megabytes to hundreds. Building them from source costs
2-3 hours, so this pulls the same artifacts CI uses instead.

Everything needed to validate a download is in the release's ``manifest.json``:
the architecture the libraries were built for, and a sha256 for both the
compressed blob and the extracted library.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import lzma
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from .jit_assets import (
    AITER_SHA as EXPECTED_AITER_SHA,
    ASSET_CONTRACT_VERSION as EXPECTED_ASSET_CONTRACT,
    GPU_ARCHS as EXPECTED_GPU_ARCHS,
    JIT_RECIPE_HASH as EXPECTED_JIT_RECIPE_HASH,
    ROCM_VERSION as EXPECTED_ROCM_VERSION,
    RELEASE_TAG as DEFAULT_TAG,
)

DEFAULT_REPO = "ROCm/jax-aiter"
MANIFEST_NAME = "manifest.json"

# urlopen without a timeout blocks forever, so a connection that stalls midway
# leaves the command hanging with no output and no recovery. Bound each socket
# operation and retry, since these assets are fetched over the public internet.
TIMEOUT_S = float(os.environ.get("JA_FETCH_TIMEOUT_S", "60"))
RETRIES = max(1, int(os.environ.get("JA_FETCH_RETRIES", "4")))


def _release_url(repo: str, tag: str, asset: str, base_url: str | None = None) -> str:
    if base_url:
        return f"{base_url.rstrip('/')}/{asset}"
    return f"https://github.com/{repo}/releases/download/{tag}/{asset}"


def _download(url: str, dest: Path, *, quiet: bool = False) -> None:
    asset = url.rsplit("/", 1)[-1]
    if not quiet:
        print(f"  fetching {asset}", flush=True)
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response, open(
                dest, "wb"
            ) as out:
                shutil.copyfileobj(response, out, length=8 * 1024 * 1024)
            return
        except urllib.error.HTTPError as exc:
            # A status code is a definite answer; retrying will not change it.
            raise SystemExit(
                f"error: could not download {url} ({exc.code} {exc.reason}).\n"
                f"       Check that release '{DEFAULT_TAG}' exists and has this asset."
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            dest.unlink(missing_ok=True)
            if attempt == RETRIES:
                raise SystemExit(
                    f"error: could not download {url} after {RETRIES} attempts "
                    f"({TIMEOUT_S:g}s timeout each): {exc}\n"
                    f"       Set JA_FETCH_TIMEOUT_S / JA_FETCH_RETRIES to adjust."
                ) from exc
            backoff = 2 ** (attempt - 1)
            print(
                f"  {asset}: attempt {attempt}/{RETRIES} failed ({exc}); "
                f"retrying in {backoff}s",
                flush=True,
            )
            time.sleep(backoff)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decompress(src: Path, dst: Path, compression: str) -> None:
    """Extract src to dst.

    gzip and xz go through the standard library. zstd is decoded by the
    ``zstandard`` wheel declared in install_requires, so users do not need a
    system package just to complete the jax-aiter installation.
    """
    if compression == "gzip":
        with gzip.open(src, "rb") as fin, open(dst, "wb") as fout:
            shutil.copyfileobj(fin, fout, length=8 * 1024 * 1024)
        return
    if compression == "xz":
        with lzma.open(src, "rb") as fin, open(dst, "wb") as fout:
            shutil.copyfileobj(fin, fout, length=8 * 1024 * 1024)
        return
    if compression == "zstd":
        try:
            import zstandard
        except ImportError as exc:
            raise SystemExit(
                "error: the required 'zstandard' Python package is missing.\n"
                "       Reinstall jax-aiter with dependencies and re-run."
            ) from exc
        with open(src, "rb") as fin, open(dst, "wb") as fout:
            dctx = zstandard.ZstdDecompressor(max_window_size=1 << 27)
            with dctx.stream_reader(fin) as reader:
                shutil.copyfileobj(reader, fout, length=8 * 1024 * 1024)
        return
    raise SystemExit(f"error: unknown compression '{compression}' in the manifest.")


def _target_dir() -> Path:
    from .ja_compat.config import get_downloaded_aiter_lib_dir

    return get_downloaded_aiter_lib_dir()


def _local_arch() -> str:
    from .ja_compat.chip_info import get_gfx

    arch = get_gfx()
    if not arch or arch == "native":
        raise RuntimeError(f"could not resolve GPU architecture: {arch!r}")
    return arch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="jax-aiter-fetch-mha",
        description="Download the prebuilt AITER flash-attention libraries.",
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"default: {DEFAULT_REPO}")
    parser.add_argument("--tag", default=DEFAULT_TAG, help=f"default: {DEFAULT_TAG}")
    parser.add_argument("--base-url", default=None,
                        help="serve assets from here instead of the GitHub release")
    parser.add_argument("--dest", default=None, help="override the install directory")
    parser.add_argument("--force", action="store_true",
                        help="re-download even if the libraries are already present")
    parser.add_argument("--skip-arch-check", action="store_true",
                        help="install even if the build arch differs from this GPU")
    args = parser.parse_args(argv)

    dest = Path(args.dest) if args.dest else _target_dir()
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"jax-aiter: fetching attention libraries into {dest}")

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        manifest_path = tmpdir / MANIFEST_NAME
        _download(_release_url(args.repo, args.tag, MANIFEST_NAME, args.base_url), manifest_path)
        manifest = json.loads(manifest_path.read_text())

        expected_keys = {
            "schema": 1,
            "asset_contract": EXPECTED_ASSET_CONTRACT,
            "tag": args.tag,
            "aiter_sha": EXPECTED_AITER_SHA,
            "gpu_archs": EXPECTED_GPU_ARCHS,
            "rocm_version": EXPECTED_ROCM_VERSION,
            # schema-v1 name; semantically this is the JIT recipe/ABI hash.
            "patch_hash": EXPECTED_JIT_RECIPE_HASH,
        }
        mismatches = [
            f"{key}: manifest={manifest.get(key)!r}, wheel={expected!r}"
            for key, expected in expected_keys.items()
            if manifest.get(key) != expected
        ]
        if mismatches:
            raise SystemExit(
                "error: JIT assets do not match this jax-aiter wheel:\n  "
                + "\n  ".join(mismatches)
            )

        built_for = manifest.get("gpu_archs", "unknown")
        try:
            local = _local_arch()
        except Exception as exc:
            if not args.skip_arch_check:
                raise SystemExit(
                    "error: could not detect the local GPU architecture; "
                    "refusing to install unchecked assets.\n"
                    "       Set GPU_ARCHS=gfx950 or explicitly pass "
                    "--skip-arch-check."
                ) from exc
            local = None
            print(f"  warning: architecture detection failed: {exc}")
        if local and built_for != "unknown" and local not in built_for.split(";"):
            message = (
                f"these libraries were built for '{built_for}' but this machine "
                f"reports '{local}'"
            )
            if not args.skip_arch_check:
                raise SystemExit(
                    f"error: {message}.\n"
                    f"       Re-run with --skip-arch-check to install anyway."
                )
            print(f"  warning: {message}")

        compression = manifest.get("compression", "zstd")
        files = manifest.get("files", [])
        if not files:
            raise SystemExit("error: manifest lists no files.")

        total = sum(entry.get("compressed_size", 0) for entry in files)
        print(f"  {len(files)} libraries, {total / 1e6:.0f} MB compressed "
              f"({compression}), built for {built_for}")

        if not args.force and (dest / MANIFEST_NAME).is_file():
            try:
                installed_manifest = json.loads(
                    (dest / MANIFEST_NAME).read_text()
                )
                installed_matches = installed_manifest == manifest and all(
                    (dest / entry["name"]).is_file()
                    and _sha256(dest / entry["name"]) == entry["sha256"]
                    for entry in files
                )
            except (OSError, ValueError, json.JSONDecodeError):
                installed_matches = False
            if installed_matches:
                for entry in files:
                    print(
                        f"  {entry['name']}: already present and matching, "
                        "skipping"
                    )
                return 0

        # Stage and verify the complete generation on the destination
        # filesystem. The active directory is untouched until every blob and
        # extracted library has passed validation.
        staging = Path(
            tempfile.mkdtemp(prefix=f".{dest.name}.staging-", dir=dest.parent)
        )
        backup = None
        try:
            for entry in files:
                name = entry["name"]
                blob = tmpdir / entry["compressed_name"]
                _download(
                    _release_url(
                        args.repo,
                        args.tag,
                        entry["compressed_name"],
                        args.base_url,
                    ),
                    blob,
                )

                actual = _sha256(blob)
                if actual != entry["compressed_sha256"]:
                    raise SystemExit(
                        f"error: checksum mismatch on {entry['compressed_name']}\n"
                        f"       expected {entry['compressed_sha256']}\n"
                        f"       got      {actual}"
                    )

                staged = staging / name
                _decompress(blob, staged, compression)
                actual = _sha256(staged)
                if actual != entry["sha256"]:
                    raise SystemExit(
                        f"error: checksum mismatch after extracting {name}\n"
                        f"       expected {entry['sha256']}\n"
                        f"       got      {actual}"
                    )
                staged.chmod(0o755)
                blob.unlink(missing_ok=True)
                print(f"  {name}: verified ({entry['size'] / 1e6:.0f} MB)")

            # The manifest is the completion marker checked by the loader.
            shutil.copy2(manifest_path, staging / MANIFEST_NAME)

            if dest.exists():
                backup = dest.parent / f".{dest.name}.backup-{os.getpid()}"
                if backup.exists():
                    shutil.rmtree(backup)
                dest.rename(backup)
            staging.rename(dest)
            staging = None
            if backup is not None:
                shutil.rmtree(backup)
                backup = None
        except BaseException:
            if backup is not None and backup.exists() and not dest.exists():
                backup.rename(dest)
                backup = None
            raise
        finally:
            if staging is not None and staging.exists():
                shutil.rmtree(staging)

    if args.dest:
        print(
            "  custom destination: set "
            f"JAX_AITER_LIB_DIR={dest.resolve()} before importing jax_aiter"
        )
    print("jax-aiter: done. `from jax_aiter.mha import flash_attn_func` should now work.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
