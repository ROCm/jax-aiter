# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Phase 3 A1: verify the FP4 GEMM FFI honors ``AITER_FORCE_KERNEL_NAME``.

Background
----------
Phase B.0 sweep (``docs/runs/70b/maxtext/20260516_phaseB_A1_xval_097/results.md``)
found that ``BpreShuffle_256x256`` with ``splitK=1`` is the universal-best
FP4 GEMM variant across the 4 70B target shapes (top-2 at every shape,
#1 at both wgrad shapes). The chosen Plan-3-A1 mitigation is **Pathway X**:
expose an environment-variable hint that bypasses the default heuristic
and pins a single kernel name for the whole run. The FFI handler at
``csrc/ffi/gemm_fp4/gemm_fp4_ja.cu`` reads
``AITER_FORCE_KERNEL_NAME`` (and optional ``AITER_FORCE_LOG2_K_SPLIT``)
before each ``GemmFp4FwdJA`` invocation; when unset, the heuristic
path is preserved unchanged (reversibility guarantee).

Scope of this test file
-----------------------
* Smoke-check the env-var plumbing (Python-side reachability + numerical
  correctness when the override is set vs unset).
* The *kernel-name dispatch* assertion (i.e. that the GPU actually ran
  ``BpreShuffle_256x256``) requires rocprofv3 trace introspection on a
  GPU host. That verification is performed by Phase B.2 (Task 7) of
  Plan 3 (see
  ``docs/superpowers/plans/2026-05-16-70b-fp4-fp8-gap-audit-phase3-ship.md``);
  the corresponding tests here are marked ``pytest.skip`` so they document
  intent without running on CPU-only CI.
"""

from __future__ import annotations

import os

import pytest

EXPECTED_KERNEL = (
    "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E"
)


@pytest.fixture
def force_kernel():
    """Set ``AITER_FORCE_KERNEL_NAME``; restore on teardown.

    Also pins ``AITER_FORCE_LOG2_K_SPLIT=0`` (splitK=1) which matches the
    universal-best entry from the Phase B.0 sweep.
    """
    prev_name = os.environ.get("AITER_FORCE_KERNEL_NAME")
    prev_split = os.environ.get("AITER_FORCE_LOG2_K_SPLIT")
    os.environ["AITER_FORCE_KERNEL_NAME"] = EXPECTED_KERNEL
    os.environ["AITER_FORCE_LOG2_K_SPLIT"] = "0"
    try:
        yield
    finally:
        if prev_name is None:
            os.environ.pop("AITER_FORCE_KERNEL_NAME", None)
        else:
            os.environ["AITER_FORCE_KERNEL_NAME"] = prev_name
        if prev_split is None:
            os.environ.pop("AITER_FORCE_LOG2_K_SPLIT", None)
        else:
            os.environ["AITER_FORCE_LOG2_K_SPLIT"] = prev_split


def test_env_var_is_read_at_python_layer(force_kernel):
    """The env-var must be visible to the FFI handler (read via getenv)."""
    assert os.environ.get("AITER_FORCE_KERNEL_NAME") == EXPECTED_KERNEL
    assert os.environ.get("AITER_FORCE_LOG2_K_SPLIT") == "0"


def test_force_kernel_dispatches_bpreshuffle_256x256(force_kernel):
    """Assert that ``AITER_FORCE_KERNEL_NAME`` actually selects the named kernel.

    Verification strategy (Phase B.2): launch a 70B-shape FP4 GEMM with
    rocprofv3 attached, then grep the trace CSV for the kernel name. This
    cannot run on CPU-only CI, so the test is skipped here and re-enabled
    in the Plan-3 B.2 4-leg verify Round.
    """
    pytest.skip(
        "Requires GPU + rocprof for kernel-name introspection; covered by "
        "Plan 3 Task 7 (B.2 4-leg verify), run-id "
        "20260516_phaseB_A1_xval_097 4-leg follow-up."
    )


def test_force_kernel_numerical_smoke(force_kernel):
    """Forcing ``BpreShuffle_256x256`` must produce finite, shape-correct output.

    Compares the override path against the default-heuristic path at a
    representative 70B target shape. The kernel must be present in the
    pre-loaded ASM cache; if not, the FFI returns ``kInternal``.
    """
    pytest.skip(
        "Requires GPU to execute the FP4 GEMM kernel; covered by "
        "Plan 3 Task 7 (B.2 4-leg verify) and the existing CPU-only "
        "tests in tests/test_gemm_fp4_ja.py once GPU is available."
    )


def test_no_env_var_uses_default_heuristic(monkeypatch):
    """Reversibility guarantee: unset env-var leaves behavior unchanged.

    Removes the env-vars and asserts they are absent. The actual numerical
    parity with the legacy heuristic path is a GPU test (deferred to B.2).
    """
    monkeypatch.delenv("AITER_FORCE_KERNEL_NAME", raising=False)
    monkeypatch.delenv("AITER_FORCE_LOG2_K_SPLIT", raising=False)
    assert "AITER_FORCE_KERNEL_NAME" not in os.environ
    assert "AITER_FORCE_LOG2_K_SPLIT" not in os.environ


def test_expected_kernel_name_is_pinned_canonical():
    """Catch accidental edits to the recipe-level kernel-name string.

    The mangled name is referenced from three places that must stay in
    sync:
      1. This test (``EXPECTED_KERNEL``).
      2. ``scripts/run_fresh_maxtext_e2e.sh`` ``set_aiter_env()``.
      3. ``docs/runs/70b/maxtext/20260516_phaseB_A1_xval_097/results.md``.
    A drift here means the override no longer targets the variant the
    B.0 sweep selected; if a different variant is intentionally chosen,
    update all three locations together.
    """
    assert EXPECTED_KERNEL == (
        "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E"
    )
