#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Smoke imports + the 1-GPU unit pytest subset, INSIDE the CI container.
# Shared by ci.yml and nightly-ci.yml. Commands preserved verbatim from the
# pre-refactor ci.yml.
set -euxo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# --- smoke imports ---
python3 -c 'from jax_aiter.mha import flash_attn_func, flash_attn_varlen; from jax_aiter.rmsnorm import rms_norm; from jax_aiter.gemm import gemm; from jax_aiter.gemm_fp4 import gemm_fp4_bf16, MXFP4Quantizer, WeightWorkspace; print("jax-aiter import OK")'
python3 tests/smoke_gemm_test.py
python3 tests/smoke_gemm_all_test.py

# --- unit subset (1 GPU, ~90 seconds) ---
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_nccl_comm_splitting=false --xla_gpu_enable_command_buffer="
python3 -m pytest -v --reruns 2 \
  tests/test_mha_ja.py tests/test_rmsnorm_ja.py tests/test_gemm_ja.py \
  tests/test_gemm_fp4_ja.py tests/test_silu_and_mul_ja.py \
  tests/test_mxfp4_quantizer.py tests/test_weight_workspace.py \
  tests/test_fp4_wgrad.py

echo "[ci/test] smoke + unit subset passed."
