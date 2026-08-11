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
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_REPO = "ROCm/jax-aiter"
DEFAULT_TAG = "jit-libs"
MANIFEST_NAME = "manifest.json"


def _release_url(repo: str, tag: str, asset: str, base_url: str | None = None) -> str:
    if base_url:
        return f"{base_url.rstrip('/')}/{asset}"
    return f"https://github.com/{repo}/releases/download/{tag}/{asset}"


def _download(url: str, dest: Path, *, quiet: bool = False) -> None:
    if not quiet:
        print(f"  fetching {url.rsplit('/', 1)[-1]}", flush=True)
    try:
        with urllib.request.urlopen(url) as response, open(dest, "wb") as out:
            shutil.copyfileobj(response, out, length=8 * 1024 * 1024)
    except urllib.error.HTTPError as exc:
        raise SystemExit(
            f"error: could not download {url} ({exc.code} {exc.reason}).\n"
            f"       Check that release '{DEFAULT_TAG}' exists and has this asset."
        ) from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decompress(src: Path, dst: Path, compression: str) -> None:
    """Extract src to dst.

    gzip and xz go through the standard library. zstd has no stdlib decoder
    before Python 3.14, so it shells out and says so plainly if the binary is
    absent, rather than failing with an opaque traceback.
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
        if shutil.which("zstd") is None:
            raise SystemExit(
                "error: these assets are zstd-compressed and no 'zstd' binary was found.\n"
                "       Install it (apt-get install zstd) and re-run."
            )
        subprocess.run(
            ["zstd", "-d", "-f", "--long=27", "-o", str(dst), str(src)],
            check=True,
        )
        return
    raise SystemExit(f"error: unknown compression '{compression}' in the manifest.")


def _target_dir() -> Path:
    from .ja_compat.config import get_lib_root

    return Path(str(get_lib_root())) / "aiter_build"


def _local_arch() -> str | None:
    try:
        from .ja_compat.chip_info import get_gfx

        return get_gfx()
    except Exception:
        return None


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
    dest.mkdir(parents=True, exist_ok=True)

    print(f"jax-aiter: fetching attention libraries into {dest}")

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        manifest_path = tmpdir / MANIFEST_NAME
        _download(_release_url(args.repo, args.tag, MANIFEST_NAME, args.base_url), manifest_path)
        manifest = json.loads(manifest_path.read_text())

        built_for = manifest.get("gpu_archs", "unknown")
        local = _local_arch()
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

        for entry in files:
            name = entry["name"]
            final = dest / name
            if final.is_file() and not args.force:
                if _sha256(final) == entry["sha256"]:
                    print(f"  {name}: already present and matching, skipping")
                    continue
                print(f"  {name}: present but does not match the manifest, replacing")

            blob = tmpdir / entry["compressed_name"]
            _download(_release_url(args.repo, args.tag, entry["compressed_name"],
                                 args.base_url), blob)

            actual = _sha256(blob)
            if actual != entry["compressed_sha256"]:
                raise SystemExit(
                    f"error: checksum mismatch on {entry['compressed_name']}\n"
                    f"       expected {entry['compressed_sha256']}\n"
                    f"       got      {actual}"
                )

            staged = tmpdir / name
            _decompress(blob, staged, compression)

            actual = _sha256(staged)
            if actual != entry["sha256"]:
                raise SystemExit(
                    f"error: checksum mismatch after extracting {name}\n"
                    f"       expected {entry['sha256']}\n"
                    f"       got      {actual}"
                )

            # Move into place only once verified, so an interrupted run never
            # leaves a partial library that would fail at dlopen time.
            shutil.move(str(staged), str(final))
            final.chmod(0o755)
            blob.unlink(missing_ok=True)
            print(f"  {name}: installed ({entry['size'] / 1e6:.0f} MB)")

    print("jax-aiter: done. `from jax_aiter.mha import flash_attn_func` should now work.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
