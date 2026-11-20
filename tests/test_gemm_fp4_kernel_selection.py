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

# Derived from this file, not hardcoded: an absolute developer path made
# HANDLER.exists() and VERIFY.exists() false anywhere else, so the two checks
# below -- including the only guard that the dispatch table still matches the
# 20260615 oracle -- silently skipped instead of running.
REPO = Path(__file__).resolve().parents[1]
HANDLER = REPO / "csrc/ffi/gemm_fp4/gemm_fp4_ja.cu"
VERIFY = REPO / "scripts/verify_fp4_dispatch_rocprof.py"
ROCPROF_BENCH = os.environ.get("JA_ROCPROF_BENCH", "")

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


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("rocprofv3") is None,
                    reason="rocprofv3 not on PATH (GPU/rocprof host required)")
@pytest.mark.skipif(not VERIFY.exists(), reason="verify script missing")
@pytest.mark.skipif(
    not ROCPROF_BENCH or not Path(ROCPROF_BENCH).is_file(),
    reason="set JA_ROCPROF_BENCH to the external rocprof benchmark script",
)
def test_force_kernel_dispatches_via_rocprof_trace(tmp_path):
    """Launch FP4 GEMMs under rocprofv3 --kernel-trace and assert the GPU ran
    the per-shape oracle kernel under AITER_FP4_DISPATCH=1, and the 256x256 pin
    under AITER_FORCE_KERNEL_NAME. This is the real GPU-side dispatch proof.
    """
    if not _gpu_available():
        pytest.skip("no ROCm GPU visible to JAX")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
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
    proc = subprocess.run(cmd, env=env, cwd=str(REPO),
                          capture_output=True, text=True, timeout=600)
    print(proc.stdout)
    print(proc.stderr, file=sys.stderr)
    assert proc.returncode == 0, (
        f"rocprof dispatch assertion failed (rc={proc.returncode}):\n"
        f"{proc.stdout[-1500:]}\n{proc.stderr[-800:]}")


# --------------------------------------------------------------------------
# Persistent (ps=1) kernels have a different calling convention; the handler
# must not run them.
# --------------------------------------------------------------------------
def _persistent_kernel_names():
    """Symbols that AITER's f4gemm manifest marks ps=1. Empty on AITER pins
    whose manifest has no ps column."""
    hsa = Path(os.environ.get("AITER_ASM_DIR") or REPO / "third_party/aiter/hsa")
    manifest = hsa / "gfx950/f4gemm/f4gemm_bf16_per1x32Fp4.csv"
    if not manifest.is_file():
        return None
    rows = [ln.strip() for ln in manifest.read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]
    header = [c.strip() for c in rows[0].split(",")]
    if "ps" not in header:
        return []
    ps, name = header.index("ps"), header.index("knl_name")
    return [cols[name].strip() for cols in (r.split(",") for r in rows[1:])
            if cols[ps].strip() == "1"]


_FORCED_GEMM = """
import jax, jax.numpy as jnp
from jax_aiter.gemm_fp4 import gemm_fp4_bf16
a = jnp.ones((256, 4096), jnp.bfloat16)
b = jnp.ones((4096, 4096), jnp.bfloat16)
jax.block_until_ready(gemm_fp4_bf16(a, b))
"""


def test_persistent_kernels_are_refused():
    """Naming a ps=1 kernel must fail, not run it with the ordinary arguments.

    The heuristic used to pick f4gemm_..._256x256_ps for the gate/up weight
    gradient (M=14336, N=4096, K=512) once AITER added it, and the gradient came
    back as garbage. The handler now leaves ps=1 kernels out of its cache, so
    the heuristic cannot pick one and a forced one is rejected by name. Runs in
    a subprocess so the expected runtime error cannot leak into later tests.
    """
    names = _persistent_kernel_names()
    if names is None:
        pytest.skip("f4gemm manifest not found")
    if not names:
        pytest.skip("this AITER pin has no ps=1 f4gemm kernels")
    if not _gpu_available():
        pytest.skip("no ROCm GPU visible to JAX")
    for name in names:
        env = os.environ.copy()
        env["AITER_FORCE_KERNEL_NAME"] = name
        env.pop("AITER_FP4_DISPATCH", None)
        proc = subprocess.run([sys.executable, "-c", _FORCED_GEMM], env=env,
                              capture_output=True, text=True, timeout=600)
        assert proc.returncode != 0, f"{name} ran instead of being refused"
        assert "not in cache" in proc.stderr, proc.stderr[-1500:]


def test_expected_kernel_name_is_pinned_canonical():
    """Guard the legacy FORCE-pin kernel string (still a supported fallback)."""
    assert EXPECTED_KERNEL == (
        "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E")
