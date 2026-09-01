# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only tests for optional JIT-library loading."""
from pathlib import Path

from jax_aiter.ffi import registry


def _touch(directory: Path, *names: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).touch()


def test_mha_shims_wait_for_jit_libs_and_load_after_fetch(tmp_path, monkeypatch):
    aiter_dir = tmp_path / "aiter_build"
    shim_dir = tmp_path / "jax_aiter_build"
    _touch(aiter_dir, "librmsnorm_fwd.so")
    _touch(shim_dir, "gemm_fwd_ja.so", "mha_fwd_ja.so", "mha_bwd_ja.so")

    loaded_paths = []

    def fake_cdll(path, mode=None):
        loaded_paths.append(Path(path).name)
        return object()

    monkeypatch.setattr(registry.ja_config, "get_aiter_lib_dir", lambda: aiter_dir)
    monkeypatch.setattr(registry.ja_config, "get_jax_aiter_lib_dir", lambda: shim_dir)
    monkeypatch.setattr(registry.ctypes, "CDLL", fake_cdll)
    monkeypatch.setattr(registry, "_umbrella_handle", object())
    monkeypatch.setattr(registry, "_module_handles", {})

    registry.load_thin_modules()
    assert set(registry._module_handles) == {
        "librmsnorm_fwd.so",
        "gemm_fwd_ja.so",
    }
    assert "mha_fwd_ja.so" not in loaded_paths
    assert "mha_bwd_ja.so" not in loaded_paths

    # Simulate jax-aiter-fetch-mha completing in the same Python process.
    _touch(aiter_dir, "libmha_fwd.so", "libmha_bwd.so")
    registry.load_thin_modules()
    assert set(registry._module_handles) == {
        "librmsnorm_fwd.so",
        "libmha_fwd.so",
        "libmha_bwd.so",
        "gemm_fwd_ja.so",
        "mha_fwd_ja.so",
        "mha_bwd_ja.so",
    }

    # Existing modules are not dlopened a second time.
    assert loaded_paths.count("librmsnorm_fwd.so") == 1
    assert loaded_paths.count("gemm_fwd_ja.so") == 1
