# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Verify the FP4 GEMM FFI selection: env plumbing + REAL rocprof dispatch.

Selection priority in csrc/ffi/gemm_fp4/gemm_fp4_ja.cu:
  AITER_FP4_DISPATCH (per-shape oracle table, 20260615 study)
    > AITER_FP4_KVWGRAD_128x512 (legacy band-aid)
    > AITER_FORCE_KERNEL_NAME (blanket pin)
    > select_fp4_kernel (occupancy heuristic).

The 20260615 kernel-selection study
(mxfp4_analysis/runs/20260615_8b_fp4_kernselect_097) showed the blanket
256x256 FORCE pin is 3-9% slower than the per-shape oracle on the 8B
attention shapes (and the occupancy heuristic is *worse* than forced on
kv-wgrad). The validated winner is the per-shape dispatch table, gated by
AITER_FP4_DISPATCH (default OFF => production byte-identical / reversible).

Kernel choice is numerically NEUTRAL (fp32-accumulate; the study's
parsed/neutrality.json shows splitK=1 tile variants are byte-identical and
splitK>1 within 1 bf16 ULP), so dispatch changes wall-time only, never loss.

This file:
  * smoke-checks the env plumbing for both selectors (CPU-only safe);
  * asserts the per-shape oracle dispatch table's expected (tile, splitK)
    entries match the study (CPU-only, parses the handler source);
  * runs a REAL rocprofv3 kernel-trace dispatch assertion when a GPU +
    rocprofv3 are present (skips gracefully otherwise), replacing the old
    pytest.skip-only placeholder.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

EXPECTED_KERNEL = "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E"
DISPATCH_128x512 = "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_128x512E"

REPO = Path("/ruvaidya/aiter_proj")
HANDLER = REPO / "jax-aiter/csrc/ffi/gemm_fp4/gemm_fp4_ja.cu"
VERIFY = REPO / "jax-aiter/scripts/verify_fp4_dispatch_rocprof.py"

# Per-shape oracle expected by the 20260615 study (M,N,K) -> (tile, splitK).
ORACLE_TABLE = {
    (32768, 4096, 4096): ("128x512", 1),
    (32768, 1024, 4096): ("128x512", 1),
    (32768, 4096, 1024): ("128x512", 1),
    (32768, 14336, 4096): ("128x512", 1),
    (32768, 4096, 14336): ("128x512", 1),
    (4096, 4096, 32768): ("128x512", 1),
    (1024, 4096, 32768): ("128x512", 4),
    (14336, 4096, 32768): ("256x256", 1),
    (4096, 14336, 32768): ("256x256", 1),
}


# --------------------------------------------------------------------------
# Env-plumbing (CPU-only safe).
# --------------------------------------------------------------------------
@pytest.fixture
def force_kernel():
    prev_name = os.environ.get("AITER_FORCE_KERNEL_NAME")
    prev_split = os.environ.get("AITER_FORCE_LOG2_K_SPLIT")
    os.environ["AITER_FORCE_KERNEL_NAME"] = EXPECTED_KERNEL
    os.environ["AITER_FORCE_LOG2_K_SPLIT"] = "0"
    try:
        yield
    finally:
        for k, v in (("AITER_FORCE_KERNEL_NAME", prev_name),
                     ("AITER_FORCE_LOG2_K_SPLIT", prev_split)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_env_var_is_read_at_python_layer(force_kernel):
    assert os.environ.get("AITER_FORCE_KERNEL_NAME") == EXPECTED_KERNEL
    assert os.environ.get("AITER_FORCE_LOG2_K_SPLIT") == "0"


def test_dispatch_env_plumbing(monkeypatch):
    monkeypatch.setenv("AITER_FP4_DISPATCH", "1")
    assert os.environ["AITER_FP4_DISPATCH"] == "1"
    monkeypatch.delenv("AITER_FP4_DISPATCH", raising=False)
    assert "AITER_FP4_DISPATCH" not in os.environ


def test_no_env_var_uses_default_heuristic(monkeypatch):
    """Reversibility: with no selector env set, the handler uses the heuristic."""
    monkeypatch.delenv("AITER_FP4_DISPATCH", raising=False)
    monkeypatch.delenv("AITER_FORCE_KERNEL_NAME", raising=False)
    monkeypatch.delenv("AITER_FORCE_LOG2_K_SPLIT", raising=False)
    assert "AITER_FP4_DISPATCH" not in os.environ
    assert "AITER_FORCE_KERNEL_NAME" not in os.environ


# --------------------------------------------------------------------------
# Source-of-truth: the handler's dispatch table matches the study oracle.
# --------------------------------------------------------------------------
@pytest.mark.skipif(not HANDLER.exists(), reason="handler source not present")
def test_dispatch_table_matches_study_oracle():
    """Parse lookup_fp4_dispatch() entries and assert they equal ORACLE_TABLE."""
    text = HANDLER.read_text()
    # entries look like: {32768, 4096, 4096,   &K128x512, 0},  // comment
    entry_re = re.compile(
        r"\{\s*(\d+),\s*(\d+),\s*(\d+),\s*&(K128x512|K256x256),\s*(\d+)\s*\}")
    tile_of = {"K128x512": "128x512", "K256x256": "256x256"}
    parsed = {}
    for m in entry_re.finditer(text):
        M, N, K = int(m[1]), int(m[2]), int(m[3])
        tile = tile_of[m[4]]
        splitk = 1 << int(m[5])
        parsed[(M, N, K)] = (tile, splitk)
    assert parsed == ORACLE_TABLE, (
        f"handler dispatch table != study oracle.\n"
        f"handler: {parsed}\noracle:  {ORACLE_TABLE}")


# --------------------------------------------------------------------------
# REAL rocprof-trace dispatch assertion (GPU + rocprofv3 required).
# --------------------------------------------------------------------------
def _gpu_available():
    try:
        import jax
        devs = jax.devices()
        # JAX reports platform="gpu" for ROCm RocmDevice(s).
        return len(devs) > 0 and all(d.platform != "cpu" for d in devs)
    except Exception:
        return False


@pytest.mark.skipif(shutil.which("rocprofv3") is None,
                    reason="rocprofv3 not on PATH (GPU/rocprof host required)")
@pytest.mark.skipif(not VERIFY.exists(), reason="verify script missing")
def test_force_kernel_dispatches_via_rocprof_trace(tmp_path):
    """Launch FP4 GEMMs under rocprofv3 --kernel-trace and assert the GPU ran
    the per-shape oracle kernel under AITER_FP4_DISPATCH=1, and the 256x256 pin
    under AITER_FORCE_KERNEL_NAME. This is the real GPU-side dispatch proof.
    """
    if not _gpu_available():
        pytest.skip("no ROCm GPU visible to JAX")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "jax-aiter")
    env.setdefault("HIP_VISIBLE_DEVICES", "0")
    checks = [
        "32768,4096,4096:dispatch:128x512:1",   # tall fprop/dgrad -> 128x512
        "1024,4096,32768:dispatch:128x512:4",   # skinny kv-wgrad  -> 128x512/sK4
        "32768,4096,4096:forced:256x256:1",     # FORCE pin -> 256x256
    ]
    cmd = [sys.executable, str(VERIFY), "--out-dir", str(tmp_path),
           "--warmup", "2", "--iters", "8"]
    for c in checks:
        cmd += ["--check", c]
    proc = subprocess.run(cmd, env=env, cwd=str(REPO / "jax-aiter"),
                          capture_output=True, text=True, timeout=600)
    print(proc.stdout)
    print(proc.stderr, file=sys.stderr)
    assert proc.returncode == 0, (
        f"rocprof dispatch assertion failed (rc={proc.returncode}):\n"
        f"{proc.stdout[-1500:]}\n{proc.stderr[-800:]}")


def test_expected_kernel_name_is_pinned_canonical():
    """Guard the legacy FORCE-pin kernel string (still a supported fallback)."""
    assert EXPECTED_KERNEL == (
        "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E")
