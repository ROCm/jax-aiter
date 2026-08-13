#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Smoke imports + a pytest tier, INSIDE the CI container.
#
# JA_TEST_TIER selects what runs, so the workflows no longer hand-maintain a
# list of test files that drifts from what exists:
#   pr       (default) everything except the exhaustive sweeps and the
#            multi-device cases -- what a 1-GPU PR job should run
#   nightly  the full suite including the slow sweeps, on >= 4 devices
#   multigpu only the cases that need >= 4 visible devices
set -euxo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

TIER="${JA_TEST_TIER:-pr}"
case "$TIER" in
  pr)       MARKS="not slow and not multigpu" ;;
  nightly)  MARKS="" ;;
  multigpu) MARKS="multigpu" ;;
  *) echo "[ci/test] unknown JA_TEST_TIER '$TIER' (pr|nightly|multigpu)" >&2; exit 2 ;;
esac

# --- smoke imports ---
python3 -c 'from jax_aiter.mha import flash_attn_func, flash_attn_varlen; from jax_aiter.rmsnorm import rms_norm; from jax_aiter.gemm import gemm; from jax_aiter.gemm_fp4 import gemm_fp4_bf16, MXFP4Quantizer, WeightWorkspace; print("jax-aiter import OK")'
python3 tests/smoke_gemm_all_test.py

# --- unit tier ---
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_nccl_comm_splitting=false --xla_gpu_enable_command_buffer="
if [ -n "$MARKS" ]; then
  python3 -m pytest -v --reruns 2 -m "$MARKS"
else
  python3 -m pytest -v --reruns 2
fi

echo "[ci/test] smoke + '$TIER' tier passed."
