# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only tests for the user-facing MHA library downloader."""
from __future__ import annotations

import gzip
import hashlib
import json
import lzma
from pathlib import Path

import pytest
import zstandard

from jax_aiter import fetch_mha
from jax_aiter.ja_compat import config


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _release(tmp_path: Path, compression: str, *, arch: str = "gfx950"):
    release = tmp_path / "release"
    release.mkdir()
    raw = b"jax-aiter-mha-fixture\0" * 10_000

    if compression == "gzip":
        compressed = gzip.compress(raw)
        suffix = ".gz"
    elif compression == "xz":
        compressed = lzma.compress(raw)
        suffix = ".xz"
    elif compression == "zstd":
        compressed = zstandard.ZstdCompressor(level=3).compress(raw)
        suffix = ".zst"
    else:  # pragma: no cover - helper misuse
        raise AssertionError(compression)

    blob_name = "libmha_fwd.so" + suffix
    (release / blob_name).write_bytes(compressed)
    manifest = {
        "schema": 1,
        "tag": "jit-libs",
        "aiter_sha": "fixture",
        "gpu_archs": arch,
        "rocm_version": "7.14.0",
        "patch_hash": "none",
        "compression": compression,
        "files": [
            {
                "name": "libmha_fwd.so",
                "size": len(raw),
                "sha256": _sha256(raw),
                "compressed_name": blob_name,
                "compressed_size": len(compressed),
                "compressed_sha256": _sha256(compressed),
            }
        ],
    }
    (release / "manifest.json").write_text(json.dumps(manifest))
    return release, raw, blob_name


@pytest.mark.parametrize("compression", ["gzip", "xz", "zstd"])
def test_fetch_install_and_idempotent(tmp_path, monkeypatch, capsys, compression):
    release, raw, _ = _release(tmp_path, compression)
    dest = tmp_path / "installed"
    monkeypatch.setattr(fetch_mha, "_local_arch", lambda: "gfx950")
    args = ["--base-url", release.as_uri(), "--dest", str(dest)]

    assert fetch_mha.main(args) == 0
    assert (dest / "libmha_fwd.so").read_bytes() == raw
    assert json.loads((dest / "manifest.json").read_text())["gpu_archs"] == "gfx950"

    # A second run verifies the installed hash and does not fetch the blob.
    assert fetch_mha.main(args) == 0
    assert "already present and matching" in capsys.readouterr().out


def test_fetch_rejects_wrong_arch_before_installing(tmp_path, monkeypatch):
    release, _, _ = _release(tmp_path, "gzip", arch="gfx950")
    dest = tmp_path / "installed"
    monkeypatch.setattr(fetch_mha, "_local_arch", lambda: "gfx942")

    with pytest.raises(SystemExit, match="built for 'gfx950'.*reports 'gfx942'"):
        fetch_mha.main(["--base-url", release.as_uri(), "--dest", str(dest)])
    assert not (dest / "libmha_fwd.so").exists()


def test_fetch_rejects_corrupt_blob_without_partial_install(tmp_path, monkeypatch):
    release, _, blob_name = _release(tmp_path, "gzip")
    blob = release / blob_name
    corrupt = bytearray(blob.read_bytes())
    corrupt[len(corrupt) // 2] ^= 0xFF
    blob.write_bytes(corrupt)
    dest = tmp_path / "installed"
    monkeypatch.setattr(fetch_mha, "_local_arch", lambda: "gfx950")

    with pytest.raises(SystemExit, match="checksum mismatch"):
        fetch_mha.main(["--base-url", release.as_uri(), "--dest", str(dest)])
    assert not (dest / "libmha_fwd.so").exists()


def test_default_download_dir_is_versioned_user_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("JA_ROOT_DIR", raising=False)
    monkeypatch.delenv("JAX_AITER_LIB_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    assert config.get_downloaded_aiter_lib_dir() == (
        tmp_path / "jax-aiter" / "0.1.0a2" / "aiter_build"
    )


def test_loader_uses_cache_only_when_all_jit_libs_are_present(tmp_path, monkeypatch):
    monkeypatch.delenv("JA_ROOT_DIR", raising=False)
    packaged = tmp_path / "package"
    packaged.mkdir()
    downloaded = tmp_path / "downloaded"
    downloaded.mkdir()
    monkeypatch.setattr(config, "get_lib_root", lambda: packaged)
    monkeypatch.setattr(config, "get_downloaded_aiter_lib_dir", lambda: downloaded)

    # A partial/interrupted fetch must not become the active library directory.
    (downloaded / "libmha_fwd.so").touch()
    assert config.get_aiter_lib_dir() == packaged / "aiter_build"

    for name in config._JIT_LIB_NAMES:
        (downloaded / name).touch()
    assert config.get_aiter_lib_dir() == downloaded
