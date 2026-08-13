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
from jax_aiter import jit_assets


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _release(tmp_path: Path, compression: str, *, arch: str = "gfx950"):
    release = tmp_path / "release"
    release.mkdir()
    raw_files = {
        name: (f"fixture:{name}\0".encode() * 10_000)
        for name in config._JIT_LIB_NAMES
    }
    entries = []
    blob_names = {}
    for name, raw in raw_files.items():
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
        blob_name = name + suffix
        blob_names[name] = blob_name
        (release / blob_name).write_bytes(compressed)
        entries.append(
            {
                "name": name,
                "size": len(raw),
                "sha256": _sha256(raw),
                "compressed_name": blob_name,
                "compressed_size": len(compressed),
                "compressed_sha256": _sha256(compressed),
            }
        )

    manifest = {
        "schema": 1,
        "asset_contract": jit_assets.ASSET_CONTRACT_VERSION,
        "tag": fetch_mha.DEFAULT_TAG,
        "aiter_sha": jit_assets.AITER_SHA,
        "gpu_archs": arch,
        "rocm_version": jit_assets.ROCM_VERSION,
        "patch_hash": jit_assets.JIT_RECIPE_HASH,
        "compression": compression,
        "files": entries,
    }
    (release / "manifest.json").write_text(json.dumps(manifest))
    return release, raw_files, blob_names


@pytest.mark.parametrize("compression", ["gzip", "xz", "zstd"])
def test_fetch_install_and_idempotent(tmp_path, monkeypatch, capsys, compression):
    release, raw_files, _ = _release(tmp_path, compression)
    dest = tmp_path / "installed"
    monkeypatch.setattr(fetch_mha, "_local_arch", lambda: "gfx950")
    args = ["--base-url", release.as_uri(), "--dest", str(dest)]

    assert fetch_mha.main(args) == 0
    for name, raw in raw_files.items():
        assert (dest / name).read_bytes() == raw
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


def test_fetch_fails_closed_when_arch_detection_fails(tmp_path, monkeypatch):
    release, _, _ = _release(tmp_path, "gzip")
    dest = tmp_path / "installed"

    def unavailable():
        raise RuntimeError("rocminfo unavailable")

    monkeypatch.setattr(fetch_mha, "_local_arch", unavailable)
    args = ["--base-url", release.as_uri(), "--dest", str(dest)]
    with pytest.raises(SystemExit, match="could not detect.*refusing"):
        fetch_mha.main(args)
    assert not dest.exists()

    assert fetch_mha.main(args + ["--skip-arch-check"]) == 0
    assert (dest / "manifest.json").is_file()


def test_fetch_rejects_assets_for_different_build_inputs(tmp_path, monkeypatch):
    release, _, _ = _release(tmp_path, "gzip")
    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["aiter_sha"] = "0" * 40
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(fetch_mha, "_local_arch", lambda: "gfx950")

    with pytest.raises(SystemExit, match=r"(?s)assets do not match.*aiter_sha"):
        fetch_mha.main(
            ["--base-url", release.as_uri(), "--dest", str(tmp_path / "installed")]
        )


def test_fetch_rejects_corrupt_blob_without_partial_install(tmp_path, monkeypatch):
    release, _, blob_names = _release(tmp_path, "gzip")
    blob = release / blob_names["libmha_bwd.so"]
    corrupt = bytearray(blob.read_bytes())
    corrupt[len(corrupt) // 2] ^= 0xFF
    blob.write_bytes(corrupt)
    dest = tmp_path / "installed"
    monkeypatch.setattr(fetch_mha, "_local_arch", lambda: "gfx950")

    with pytest.raises(SystemExit, match="checksum mismatch"):
        fetch_mha.main(["--base-url", release.as_uri(), "--dest", str(dest)])
    assert not (dest / "libmha_fwd.so").exists()


def test_failed_force_refresh_preserves_complete_active_generation(
    tmp_path, monkeypatch
):
    release, raw_files, blob_names = _release(tmp_path, "gzip")
    dest = tmp_path / "installed"
    monkeypatch.setattr(fetch_mha, "_local_arch", lambda: "gfx950")
    args = ["--base-url", release.as_uri(), "--dest", str(dest)]
    assert fetch_mha.main(args) == 0
    original_manifest = (dest / "manifest.json").read_bytes()

    blob = release / blob_names["libmha_bwd.so"]
    corrupt = bytearray(blob.read_bytes())
    corrupt[len(corrupt) // 2] ^= 0xFF
    blob.write_bytes(corrupt)

    with pytest.raises(SystemExit, match="checksum mismatch"):
        fetch_mha.main(args + ["--force"])
    assert (dest / "manifest.json").read_bytes() == original_manifest
    for name, raw in raw_files.items():
        assert (dest / name).read_bytes() == raw


def test_default_download_dir_is_versioned_user_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("JA_ROOT_DIR", raising=False)
    monkeypatch.delenv("JAX_AITER_LIB_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    assert config.get_downloaded_aiter_lib_dir() == (
        tmp_path
        / "jax-aiter"
        / "0.1.0a2"
        / jit_assets.CACHE_ID
        / "aiter_build"
    )


def test_loader_uses_cache_only_when_all_jit_libs_are_present(tmp_path, monkeypatch):
    monkeypatch.delenv("JA_ROOT_DIR", raising=False)
    packaged = tmp_path / "package"
    packaged_aiter = packaged / "aiter_build"
    packaged_aiter.mkdir(parents=True)
    downloaded = tmp_path / "downloaded"
    downloaded.mkdir()
    monkeypatch.setattr(config, "get_lib_root", lambda: packaged)
    monkeypatch.setattr(config, "get_downloaded_aiter_lib_dir", lambda: downloaded)

    # A partial/interrupted fetch must not become the active library directory.
    (downloaded / "libmha_fwd.so").write_bytes(b"partial")
    assert config.get_aiter_lib_dir() == packaged_aiter

    entries = []
    for name in config._JIT_LIB_NAMES:
        data = f"complete:{name}".encode()
        (downloaded / name).write_bytes(data)
        entries.append({"name": name, "size": len(data)})
    (downloaded / "manifest.json").write_text(
        json.dumps(
            {
                "aiter_sha": jit_assets.AITER_SHA,
                "asset_contract": jit_assets.ASSET_CONTRACT_VERSION,
                "gpu_archs": jit_assets.GPU_ARCHS,
                "patch_hash": jit_assets.JIT_RECIPE_HASH,
                "files": entries,
            }
        )
    )
    assert config.get_aiter_lib_dir() == downloaded

    # A +full wheel's bundled generation wins over any user cache.
    for name in config._JIT_LIB_NAMES:
        (packaged_aiter / name).write_bytes(b"bundled")
    assert config.get_aiter_lib_dir() == packaged_aiter
