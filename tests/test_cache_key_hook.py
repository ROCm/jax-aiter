# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only tests for the persistent compilation cache key salt."""
from jax._src import cache_key

import jax_aiter  # noqa: F401  installs the hook
from jax_aiter.ja_compat import cache_key as ja_cache_key


def test_hook_is_installed_once():
    hook = cache_key.custom_hook
    assert getattr(hook, "_jax_aiter", False)
    ja_cache_key.install()
    assert cache_key.custom_hook is hook


def test_trace_time_switch_changes_salt(monkeypatch):
    monkeypatch.delenv("JA_MHA_BWD_ATOMIC_FP32", raising=False)
    unset = ja_cache_key.cache_key_salt()
    monkeypatch.setenv("JA_MHA_BWD_ATOMIC_FP32", "0")
    a16 = ja_cache_key.cache_key_salt()
    monkeypatch.setenv("JA_MHA_BWD_ATOMIC_FP32", "1")
    a32 = ja_cache_key.cache_key_salt()
    assert len({unset, a16, a32}) == 3
    assert cache_key.custom_hook().endswith(a32)


def test_locations_and_foreign_vars_do_not_change_salt(monkeypatch):
    base = ja_cache_key.cache_key_salt()
    monkeypatch.setenv("JA_PY_OVERLAY", "/some/overlay")
    monkeypatch.setenv("AITER_ASM_DIR", "/elsewhere/hsa/")
    monkeypatch.setenv("UNRELATED_SETTING", "1")
    assert ja_cache_key.cache_key_salt() == base
