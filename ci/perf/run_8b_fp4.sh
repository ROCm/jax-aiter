#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Nightly MXFP4 8B FP4 perf leg (all 8 GPUs, FSDP=8). Runs INSIDE the CI
# container. Env + MaxText args are ported from the 8b_fp4 leg of
# scripts/run_fresh_maxtext_e2e.sh (the canonical perf recipe).
#
# What it does:
#   1. Clone ROCm/maxtext @ session24/70b-gate-up-fusion, checkout 83c0ba54
#      (NO patch -- the aiter_fp4 plumbing is committed on that branch).
#   2. Install TransformerEngine from the hosted wheel with --no-deps (its
#      metadata pins jax==0.8.2, which must NOT disturb our jax 0.9.0 stack).
#   3. Install MaxText's other deps WITHOUT clobbering our pinned
#      jax/jaxlib/jax_rocm7_plugin/jax_rocm7_pjrt 0.9.0 (strip the file's own
#      0.8.2 jax pins, install the rest under a jax-0.9.0 pip constraint, then
#      hard-restore the exact stack via `ci/setup_jax.sh --jax-only`).
#   4. Gate on an idle GPU node (best effort; the shared self-hosted node).
#   5. Run 5 discarded warmup steps + 50 timed steps -> train.log.
#
# Overridable env (sensible CI defaults below):
#   PERF_OUT_DIR  MAXTEXT_DIR  MAXTEXT_REPO  MAXTEXT_BRANCH  MAXTEXT_COMMIT
#   MAXTEXT_REQUIREMENTS  TE_WHEEL_URL  STEPS  WARMUP_STEPS
#   WAIT_FOR_IDLE_GPU_SCRIPT
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

JA_ROOT_DIR="${JA_ROOT_DIR:-$REPO}"
LEG="8b_fp4"
PERF_OUT_DIR="${PERF_OUT_DIR:-$REPO/ci_perf_out}"
OUTDIR="$PERF_OUT_DIR/$LEG"
LOGFILE="$OUTDIR/train.log"
WARMUP_LOGFILE="$OUTDIR/warmup.log"

STEPS="${STEPS:-50}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"

MAXTEXT_DIR="${MAXTEXT_DIR:-/maxtext_perf}"
MAXTEXT_REPO="${MAXTEXT_REPO:-https://github.com/ROCm/maxtext.git}"
MAXTEXT_BRANCH="${MAXTEXT_BRANCH:-session24/70b-gate-up-fusion}"
MAXTEXT_COMMIT="${MAXTEXT_COMMIT:-83c0ba54}"
MAXTEXT_REQUIREMENTS="${MAXTEXT_REQUIREMENTS:-dependencies/requirements/requirements_decoupled_rocm_jax_0_8_2.txt}"

# The ONLY jax-0.9.0-compatible TransformerEngine wheel (our validated
# 72aab8e3, hosted on the v0.1.0-alpha prerelease). %2B == '+' (URL-encoded).
TE_WHEEL_URL="${TE_WHEEL_URL:-https://github.com/ROCm/jax-aiter/releases/download/v0.1.0-alpha/transformer_engine-2.12.0.dev0%2B72aab8e3-cp312-cp312-linux_x86_64.whl}"

# Host-side idle-GPU gate. Default is the canonical dev-box path; inside the
# CI container this path is typically absent (the perf workflow gates on the
# host runner instead), so the in-script gate cleanly skips.
WAIT_FOR_IDLE_GPU_SCRIPT="${WAIT_FOR_IDLE_GPU_SCRIPT:-/home/ruvaidya/aiter_proj/scripts/wait_for_idle_gpu.sh}"

mkdir -p "$OUTDIR"

echo "=========================================================================="
echo "[ci/perf] leg=${LEG} steps=${STEPS} warmup=${WARMUP_STEPS}"
echo "  maxtext: ${MAXTEXT_REPO} @ ${MAXTEXT_BRANCH} (${MAXTEXT_COMMIT}) -> ${MAXTEXT_DIR}"
echo "  out:     ${OUTDIR}"
echo "=========================================================================="

# ---------------------------------------------------------------------------
# 1. Clone MaxText at the pinned commit (no patch).
# ---------------------------------------------------------------------------
if [[ ! -d "$MAXTEXT_DIR/.git" ]]; then
  git clone --branch "$MAXTEXT_BRANCH" "$MAXTEXT_REPO" "$MAXTEXT_DIR"
fi
git config --global --add safe.directory "$MAXTEXT_DIR" || true
git -C "$MAXTEXT_DIR" fetch --depth 50 origin "$MAXTEXT_BRANCH" || true
git -C "$MAXTEXT_DIR" checkout "$MAXTEXT_COMMIT"
echo "[ci/perf] MaxText HEAD: $(git -C "$MAXTEXT_DIR" rev-parse HEAD)"

# ---------------------------------------------------------------------------
# 2. TransformerEngine (no-deps; its metadata pins jax 0.8.2).
# ---------------------------------------------------------------------------
python3 -m pip install --break-system-packages --no-deps "$TE_WHEEL_URL"

# ---------------------------------------------------------------------------
# 3. MaxText deps, decoupled from our jax 0.9.0 pins.
# ---------------------------------------------------------------------------
REQ_PATH="$MAXTEXT_DIR/$MAXTEXT_REQUIREMENTS"
if [[ ! -f "$REQ_PATH" ]]; then
  echo "ERROR: MaxText requirements not found: $REQ_PATH" >&2
  exit 1
fi

# pip constraint that pins the whole jax stack to 0.9.0 (local-version tags
# like +rocm7 are ignored by '==0.9.0' matching, so our installed wheels
# satisfy these without reinstalling).
PINS_FILE="$(mktemp)"
cat > "$PINS_FILE" <<'EOF'
jax==0.9.0
jaxlib==0.9.0
jax-rocm7-plugin==0.9.0
jax-rocm7-pjrt==0.9.0
EOF

# Strip the requirements file's OWN jax 0.8.2 pins (the 3 rocm-jax 0.8.2
# wheel URLs + the bare `jax==0.8.2` line) so they cannot downgrade us; the
# pip constraint above would otherwise hard-conflict with `jax==0.8.2`.
REQ_NOJAX="$(mktemp)"
grep -vE '(rocm-jax/releases|^[[:space:]]*jax==[0-9])' "$REQ_PATH" > "$REQ_NOJAX"
echo "[ci/perf] installing MaxText deps (jax pins stripped, constrained to 0.9.0)"
PIP_CONSTRAINT="$PINS_FILE" python3 -m pip install --break-system-packages -r "$REQ_NOJAX"

# Hard guard: restore the EXACT jax 0.9.0 ROCm stack regardless of what the
# dep resolution did, then assert it.
bash "$JA_ROOT_DIR/ci/setup_jax.sh" --jax-only
python3 - <<'PYEOF'
import importlib
import jax
assert jax.__version__ == "0.9.0", f"jax drifted to {jax.__version__}"
for mod in ("jax_rocm7_plugin", "jax_rocm7_pjrt"):
    importlib.import_module(mod)
print(f"[ci/perf] jax stack OK: jax {jax.__version__} + rocm7 plugin/pjrt importable")
PYEOF

# ---------------------------------------------------------------------------
# 4. Idle-GPU gate (best effort; avoids colliding on the shared node).
# ---------------------------------------------------------------------------
if [[ -n "$WAIT_FOR_IDLE_GPU_SCRIPT" && -r "$WAIT_FOR_IDLE_GPU_SCRIPT" ]] \
   && command -v rocm-smi >/dev/null 2>&1; then
  echo "[ci/perf] gating on idle GPU via $WAIT_FOR_IDLE_GPU_SCRIPT"
  bash "$WAIT_FOR_IDLE_GPU_SCRIPT"
else
  echo "[ci/perf] WARN: GPU idle gate skipped (script '$WAIT_FOR_IDLE_GPU_SCRIPT' not readable or rocm-smi missing)."
fi

# ---------------------------------------------------------------------------
# 5. Env + recipe (ported from scripts/run_fresh_maxtext_e2e.sh 8b_fp4 leg).
# ---------------------------------------------------------------------------
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=.97
export NVTE_USE_HIPBLASLT=1
export NVTE_CK_USES_BWD_V3=1 NVTE_CK_USES_FWD_V3=1
export NVTE_CK_IS_V3_ATOMIC_FP32=0 NVTE_CK_HOW_V3_BF16_CVT=2
export NVTE_FUSED_ATTN=1 NVTE_FUSED_ATTN_CK=1 NVTE_FUSED_ATTN_AOTRITON=0
export GPU_MAX_HW_QUEUES=2
export HIP_FORCE_DEV_KERNARG=1 HSA_FORCE_FINE_GRAIN_PCIE=1
export JAX_PLATFORMS=rocm
export DECOUPLE_GCLOUD=TRUE
export PYTHONPATH="$MAXTEXT_DIR/src:${PYTHONPATH:-}"
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# AITER env (FP4 leg / use_jax_aiter=True).
export JA_ROOT_DIR="$JA_ROOT_DIR"
export AITER_ASM_DIR="$JA_ROOT_DIR/third_party/aiter/hsa/"
export AITER_SYMBOL_VISIBLE=1
export GPU_ARCHS=gfx950
export AITER_FP4_ATTN=1
# Universal-best FP4 GEMM kernel from the 70B variant sweep (8B unaffected).
export AITER_FORCE_KERNEL_NAME=_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E
export AITER_FORCE_LOG2_K_SPLIT=0

XLA_BASE='--xla_gpu_memory_limit_slop_factor=95 --xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592 --xla_gpu_enable_command_buffer= --xla_gpu_enable_latency_hiding_scheduler=True --xla_gpu_all_gather_combine_threshold_bytes=8589934592 --xla_gpu_enable_triton_gemm=False --xla_gpu_enable_cublaslt=True --xla_gpu_autotune_level=4 --xla_gpu_enable_all_gather_combine_by_dim=FALSE --xla_gpu_enable_nccl_comm_splitting=false'

cd "$MAXTEXT_DIR"

MODEL="src/maxtext/configs/base.yml \
  hardware=gpu model_name=llama3-8b attention=cudnn_flash_te \
  enable_checkpointing=False \
  ici_fsdp_parallelism=8 ici_data_parallelism=1 ici_expert_parallelism=1 \
  remat_policy=minimal_flash scan_layers=True dataset_type=synthetic \
  logits_dot_in_fp32=False dtype=bfloat16 weight_dtype=bfloat16 \
  per_device_batch_size=4 max_target_length=8192 shardy=False packing=True max_segments_per_seq=32 \
  base_output_directory=${OUTDIR} param_scan_axis=1 use_iota_embed=True"

# ---- discarded warmup leg (same graph; absorbs autotune/cache-warm cost) ----
if [[ "$WARMUP_STEPS" -gt 0 ]]; then
  echo "[ci/perf] warmup: ${WARMUP_STEPS} steps (discarded) -> ${WARMUP_LOGFILE}"
  echo "=== ${LEG} WARMUP START $(date -u) ===" >> "$WARMUP_LOGFILE"
  set +e
  XLA_FLAGS="${XLA_BASE}" \
    python3 -m maxtext.trainers.pre_train.train ${MODEL} \
      steps="${WARMUP_STEPS}" use_jax_aiter=True run_name="${LEG}_warmup" \
      quantization=aiter_fp4 aiter_attention=False \
      >> "${WARMUP_LOGFILE}" 2>&1
  WRC=$?
  set -e
  echo "=== ${LEG} WARMUP EXIT ${WRC} $(date -u) ===" >> "$WARMUP_LOGFILE"
  echo "[ci/perf] warmup exit: ${WRC}"
fi

# ---- timed leg (parsed by ci/perf/parse_perf_log.py) ----
echo "[ci/perf] timed: ${STEPS} steps -> ${LOGFILE}"
export XLA_FLAGS="${XLA_BASE}"
echo "=== ${LEG} START $(date -u) ===" >> "$LOGFILE"
set +e
python3 -m maxtext.trainers.pre_train.train ${MODEL} \
  steps="${STEPS}" use_jax_aiter=True run_name="${LEG}" \
  quantization=aiter_fp4 aiter_attention=False \
  2>&1 | tee -a "${LOGFILE}"
RC="${PIPESTATUS[0]}"
set -e
echo "=== ${LEG} EXIT ${RC} $(date -u) ===" >> "$LOGFILE"

echo "[ci/perf] last 'completed step' lines:"
grep "completed step" "${LOGFILE}" | tail -3 | sed 's/^/    /' || true
echo "[ci/perf] timed leg exit: ${RC}  log: ${LOGFILE}"
exit "${RC}"
