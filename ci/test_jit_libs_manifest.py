#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Unit tests for ci/jit_libs_manifest.py.

Covers, with a couple of tiny fake ``.so`` files:
  * a real compress -> manifest -> decompress round-trip (sha256 + size match),
  * the verify-or-rebuild decision for matching vs mismatching aiter_sha /
    patch_hash / gpu_archs / rocm_version.

Run directly (``python3 ci/test_jit_libs_manifest.py``) or via pytest.
"""
from __future__ import annotations

import gzip
import importlib.util
import os
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("jlm", _HERE / "jit_libs_manifest.py")
jlm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(jlm)

AITER_SHA = "bd0534e9630f8f142f51689f5808c627460e35bf"
OTHER_SHA = "0000000000000000000000000000000000000000"
PATCH_HASH = "sha256:" + "ab" * 32
OTHER_PATCH = "sha256:" + "cd" * 32
ARCHS = "gfx942;gfx950"


def _make_fixture(tmp: Path):
    """Create 2 tiny fake libs, gzip them, build a manifest. Returns manifest."""
    aiter_build = tmp / "aiter_build"
    dist = tmp / "dist"
    aiter_build.mkdir()
    dist.mkdir()
    libs = ("librmsnorm_fwd.so", "libmha_fwd.so")
    for i, name in enumerate(libs):
        # Distinct, compressible content so the round-trip is meaningful.
        (aiter_build / name).write_bytes((b"AITER-FAKE-%d-" % i) * 4096)
        with open(aiter_build / name, "rb") as fi, gzip.open(dist / (name + ".gz"), "wb") as fo:
            fo.write(fi.read())
    manifest = jlm.build_manifest(
        aiter_build_dir=aiter_build,
        dist_dir=dist,
        compression="gzip",
        tag="v0.1.0-alpha",
        aiter_sha=AITER_SHA,
        gpu_archs=ARCHS,
        rocm_version="7.2.0",
        patch_hash=PATCH_HASH,
        libs=libs,
    )
    return manifest, aiter_build, dist, libs


def test_roundtrip_decompress_matches_manifest():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        manifest, aiter_build, dist, libs = _make_fixture(tmp)
        assert manifest["compression"] == "gzip"
        assert {f["name"] for f in manifest["files"]} == set(libs)
        for f in manifest["files"]:
            blob = dist / f["compressed_name"]
            assert jlm.sha256_file(blob) == f["compressed_sha256"]
            out = tmp / ("out_" + f["name"])
            jlm.decompress_file(blob, out, "gzip")
            assert jlm.sha256_file(out) == f["sha256"], f"sha mismatch for {f['name']}"
            assert out.stat().st_size == f["size"], f"size mismatch for {f['name']}"
    print("PASS test_roundtrip_decompress_matches_manifest")


def test_verify_matching_is_skip_build():
    with tempfile.TemporaryDirectory() as td:
        manifest, *_ = _make_fixture(Path(td))
        ok, decision, reasons = jlm.verify_manifest(
            manifest,
            current_aiter_sha=AITER_SHA,
            current_patch_hash=PATCH_HASH,
            expected_gpu_archs=ARCHS,
            current_rocm_version="7.2.0",
        )
        assert ok is True and decision == "skip-build", reasons
    print("PASS test_verify_matching_is_skip_build")


def test_verify_mismatched_aiter_sha_is_rebuild():
    with tempfile.TemporaryDirectory() as td:
        manifest, *_ = _make_fixture(Path(td))
        ok, decision, reasons = jlm.verify_manifest(
            manifest,
            current_aiter_sha=OTHER_SHA,
            current_patch_hash=PATCH_HASH,
            expected_gpu_archs=ARCHS,
            current_rocm_version="7.2.0",
        )
        assert ok is False and decision == "rebuild", reasons
        assert any("aiter_sha mismatch" in r for r in reasons)
    print("PASS test_verify_mismatched_aiter_sha_is_rebuild")


def test_verify_mismatched_patch_hash_is_rebuild():
    with tempfile.TemporaryDirectory() as td:
        manifest, *_ = _make_fixture(Path(td))
        ok, decision, reasons = jlm.verify_manifest(
            manifest,
            current_aiter_sha=AITER_SHA,
            current_patch_hash=OTHER_PATCH,
            expected_gpu_archs=ARCHS,
            current_rocm_version="7.2.0",
        )
        assert ok is False and decision == "rebuild", reasons
        assert any("patch_hash mismatch" in r for r in reasons)
    print("PASS test_verify_mismatched_patch_hash_is_rebuild")


def test_verify_gpu_archs_strict_vs_relaxed():
    with tempfile.TemporaryDirectory() as td:
        manifest, *_ = _make_fixture(Path(td))
        # strict (default): a differing arch set forces rebuild.
        ok_s, _, _ = jlm.verify_manifest(
            manifest, current_aiter_sha=AITER_SHA, current_patch_hash=PATCH_HASH,
            expected_gpu_archs="gfx942", current_rocm_version="7.2.0",
        )
        assert ok_s is False
        # relaxed: arch difference is advisory only.
        ok_r, _, _ = jlm.verify_manifest(
            manifest, current_aiter_sha=AITER_SHA, current_patch_hash=PATCH_HASH,
            expected_gpu_archs="gfx942", current_rocm_version="7.2.0",
            strict_gpu_archs=False,
        )
        assert ok_r is True
        # order-insensitive: same set, different order still matches.
        ok_o, _, _ = jlm.verify_manifest(
            manifest, current_aiter_sha=AITER_SHA, current_patch_hash=PATCH_HASH,
            expected_gpu_archs="gfx950;gfx942", current_rocm_version="7.2.0",
        )
        assert ok_o is True
    print("PASS test_verify_gpu_archs_strict_vs_relaxed")


def test_verify_rocm_advisory_by_default_strict_optional():
    with tempfile.TemporaryDirectory() as td:
        manifest, *_ = _make_fixture(Path(td))
        # advisory (default): rocm mismatch does NOT force rebuild.
        ok_a, _, reasons_a = jlm.verify_manifest(
            manifest, current_aiter_sha=AITER_SHA, current_patch_hash=PATCH_HASH,
            expected_gpu_archs=ARCHS, current_rocm_version="7.1.0",
        )
        assert ok_a is True
        assert any("rocm_version differs" in r for r in reasons_a)
        # strict: rocm mismatch forces rebuild.
        ok_s, _, _ = jlm.verify_manifest(
            manifest, current_aiter_sha=AITER_SHA, current_patch_hash=PATCH_HASH,
            expected_gpu_archs=ARCHS, current_rocm_version="7.1.0", strict_rocm=True,
        )
        assert ok_s is False
    print("PASS test_verify_rocm_advisory_by_default_strict_optional")


def test_normalize_archs_and_jit_input_hash():
    assert jlm.normalize_archs("gfx950;gfx942") == ["gfx942", "gfx950"]
    assert jlm.normalize_archs("gfx942, gfx950") == ["gfx942", "gfx950"]
    assert jlm.normalize_archs("") == []
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # No JIT recipe files -> "none".
        assert jlm.compute_patch_hash(td) == "none"

        jit = root / "jax_aiter" / "jit"
        jit.mkdir(parents=True)
        (jit / "build_jit.py").write_text("recipe = 1\n")
        (jit / "optCompilerConfig.json").write_text("{}\n")
        h1 = jlm.compute_patch_hash(td)
        assert h1.startswith("sha256:")
        assert h1 == jlm.compute_patch_hash(td)

        # Unrelated integration patches must not invalidate multi-GB JIT libs.
        sd = root / "scripts"
        sd.mkdir()
        (sd / "maxtext_aiter_fp4.patch").write_text("--- unrelated ---\n")
        assert jlm.compute_patch_hash(td) == h1

        # JIT recipe and explicitly named JIT patches are hard cache inputs.
        (jit / "optCompilerConfig.json").write_text('{"changed": true}\n')
        h2 = jlm.compute_patch_hash(td)
        assert h2 != h1
        (sd / "aiter_jit_example.patch").write_text("--- jit patch ---\n")
        assert jlm.compute_patch_hash(td) != h2
    print("PASS test_normalize_archs_and_jit_input_hash")


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"\nAll {len(fns)} manifest tests passed.")


if __name__ == "__main__":
    _run_all()
