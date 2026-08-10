#!/usr/bin/env bash
# Single-process Llama 3.1 8B publication throughput runner.
#
# Usage:
#   bash scripts/recipes/run_nvfp4_match_8b.sh \
#     mxfp4|te_fp8_currentscaling|bf16 OUT_ROOT [50]
#
# Canonical in-container roots remain /ruvaidya/aiter_proj. Colleagues using a
# different mount can set PROJECT_ROOT, JAX_AITER_ROOT, and/or MAXTEXT_ROOT.
# RECIPE_DRY_RUN=1 resolves and prints the recipe without importing JAX or
# launching MaxText.
set -euo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
is_true() { [[ "${1,,}" == "1" || "${1,,}" == "true" || "${1,,}" == "yes" ]]; }

[[ $# -ge 2 && $# -le 3 ]] || die "usage: $0 MODE OUT_ROOT [50]"
MODE="$1"
OUT_ROOT="$2"
STEPS="${3:-50}"
[[ "$STEPS" == "50" ]] || die "publication performance runs are exactly 50 steps"

PROJECT_ROOT="${PROJECT_ROOT:-/ruvaidya/aiter_proj}"
JAX_AITER_ROOT="${JAX_AITER_ROOT:-${PROJECT_ROOT}/jax-aiter}"
MAXTEXT_ROOT="${MAXTEXT_ROOT:-${PROJECT_ROOT}/maxtext}"
MAXTEXT_CONFIG="${MAXTEXT_CONFIG:-src/maxtext/configs/base.yml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MEMFRAC="${XLA_PYTHON_CLIENT_MEM_FRACTION:-.97}"
[[ "$MEMFRAC" == ".97" ]] || die "canonical mem fraction is .97; stop instead of lowering it"

MODEL_NAME="${MODEL_NAME:-llama3.1-8b}"
MODEL_CONTROLS="${MODEL_CONTROLS:-default}"
FSDP="${ICI_FSDP_PARALLELISM:-8}"
SHARDY="${SHARDY:-True}"
SCAN_LAYERS="${SCAN_LAYERS:-False}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-8192}"
AUTOTUNE_LEVEL="${AUTOTUNE_LEVEL:-5}"
COMBINE_TH_BYTES="${COMBINE_TH_BYTES:-67108864}"
ENABLE_NNX="${ENABLE_NNX:-False}"
USE_IOTA_EMBED="${USE_IOTA_EMBED:-False}"
WEIGHT_DTYPE="${WEIGHT_DTYPE:-float32}"
MU_DTYPE="${MU_DTYPE:-float32}"
INIT_WEIGHTS_SEED="${INIT_WEIGHTS_SEED:-0}"

MODEL_ARGS=()
case "$MODEL_CONTROLS" in
  default) ;;
  llama31_mlperf)
    [[ "$MODEL_NAME" == "llama3.1-8b" ]] ||
      die "MODEL_CONTROLS=llama31_mlperf requires MODEL_NAME=llama3.1-8b"
    MODEL_ARGS=(
      query_pre_attn_scalar=0.08838834764831843
      rope_use_scale=False
      normalize_embedding_logits=False
      megatron_init_std=0.02
      megatron_residual_scale=True
      num_vocab_tiling=1
    )
    ;;
  *) die "bad MODEL_CONTROLS '$MODEL_CONTROLS' (expected default or llama31_mlperf)" ;;
esac

case "$MODE" in
  mxfp4)
    LABEL="MXFP4"; QUANTIZATION="aiter_fp4"; USE_AITER=True
    SR_KEY_MODE=maxtext_runtime_params_rng
    ATTENTION="${ATTENTION:-aiter_flash}"
    REMAT_POLICY="${REMAT_POLICY:-minimal_flash_save_fp4col}"
    ;;
  te_fp8_currentscaling)
    LABEL="TE FP8 current scaling"; QUANTIZATION="te_fp8_currentscaling"; USE_AITER=False
    SR_KEY_MODE=n/a
    ATTENTION="${ATTENTION:-cudnn_flash_te}"
    REMAT_POLICY="${REMAT_POLICY:-minimal_flash}"
    ;;
  bf16)
    LABEL="BF16"; QUANTIZATION=""; USE_AITER=False
    SR_KEY_MODE=n/a
    ATTENTION="${ATTENTION:-cudnn_flash_te}"
    REMAT_POLICY="${REMAT_POLICY:-minimal_flash}"
    ;;
  *) die "bad mode '$MODE' (expected mxfp4, te_fp8_currentscaling, or bf16)" ;;
esac

# Required ROCm/JAX safeguards. This publication runner is intentionally not a
# profiler wrapper; profiling requires a separate, explicitly labeled process.
export XLA_PYTHON_CLIENT_MEM_FRACTION="$MEMFRAC"
export JAX_PLATFORMS=rocm
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HIP_FORCE_DEV_KERNARG=1 HSA_FORCE_FINE_GRAIN_PCIE=1
export ROCPROFILER_QUEUE_INTERPOSITION=0 ROCPROFILER_REGISTER_ENABLED=0
export HSA_NO_SCRATCH_RECLAIM=1 GPU_MAX_HW_QUEUES=2
export RCCL_WARP_SPEED_AUTO=0 NCCL_DEBUG="${NCCL_DEBUG:-VERSION}"
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 NVTE_USE_HIPBLASLT=1
export NVTE_FUSED_ATTN=1 NVTE_FUSED_ATTN_CK=1 NVTE_FUSED_ATTN_AOTRITON=0
export NVTE_CK_USES_FWD_V3=1 NVTE_CK_USES_BWD_V3=1
export NVTE_CK_IS_V3_ATOMIC_FP32=1 NVTE_CK_HOW_V3_BF16_CVT=2
export DECOUPLE_GCLOUD=TRUE
export JA_ROOT_DIR="${JA_ROOT_DIR:-$JAX_AITER_ROOT}"
export AITER_ASM_DIR="${AITER_ASM_DIR:-${JAX_AITER_ROOT}/third_party/aiter/hsa/}"
export AITER_SYMBOL_VISIBLE=1 GPU_ARCHS="${GPU_ARCHS:-gfx950}"
export PYTHONPATH="${MAXTEXT_ROOT}/src:${JAX_AITER_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "$MODE" == "mxfp4" ]]; then
  export FP4_SELECT="${FP4_SELECT:-dispatch}" AITER_FP4_ATTN=1
  case "$FP4_SELECT" in
    dispatch) export AITER_FP4_DISPATCH=1; unset AITER_FORCE_KERNEL_NAME AITER_FORCE_LOG2_K_SPLIT ;;
    forced)
      unset AITER_FP4_DISPATCH
      export AITER_FORCE_KERNEL_NAME=_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E
      export AITER_FORCE_LOG2_K_SPLIT=0
      ;;
    heuristic) unset AITER_FP4_DISPATCH AITER_FORCE_KERNEL_NAME AITER_FORCE_LOG2_K_SPLIT ;;
    *) die "bad FP4_SELECT '$FP4_SELECT' (expected dispatch, forced, or heuristic)" ;;
  esac
  export JA_FP4_HADAMARD_PASSES="${JA_FP4_HADAMARD_PASSES:-wgrad}"
  export JA_FP4_SR_PASSES="${JA_FP4_SR_PASSES:-wgrad_col}"
  export JA_FP4_DGRAD_PARTITION="${JA_FP4_DGRAD_PARTITION:-gather_packed}"
  export JA_FP4_DGRAD_REUSE_FWD_COL="${JA_FP4_DGRAD_REUSE_FWD_COL:-1}"
  export JA_FP4_REMAT_SAVE_COL="${JA_FP4_REMAT_SAVE_COL:-both}"
  export JA_FP4_PACK_GATEUP_AG="${JA_FP4_PACK_GATEUP_AG:-1}"
  export JA_MHA_BWD_ATOMIC_FP32=0 JA_MHA_BWD_BF16_CVT=2
  export JA_MHA_FUSE_GQA_REDUCE=1 JA_MHA_ZERO_PAD=1
  unset AITER_FP4_SR AITER_FUSED_QUANT_HADAMARD AITER_FP4_HADAMARD_OFF
else
  unset FP4_SELECT AITER_FP4_DISPATCH AITER_FP4_ATTN
  unset JA_FP4_HADAMARD_PASSES JA_FP4_SR_PASSES JA_FP4_DGRAD_PARTITION
  unset JA_FP4_DGRAD_REUSE_FWD_COL JA_FP4_REMAT_SAVE_COL JA_FP4_PACK_GATEUP_AG
  unset JA_MHA_BWD_ATOMIC_FP32 JA_MHA_BWD_BF16_CVT JA_MHA_FUSE_GQA_REDUCE JA_MHA_ZERO_PAD
fi

MODE_ARGS=()
if [[ "$USE_AITER" == "True" && "$ATTENTION" != "aiter_flash" ]]; then
  MODE_ARGS+=(aiter_attention=False)
elif [[ "$USE_AITER" != "True" && "$ATTENTION" == "aiter_flash" ]]; then
  MODE_ARGS+=(aiter_gemm=False)
fi

XLA_BASE="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false --xla_gpu_enable_cublaslt=True --xla_gpu_autotune_level=${AUTOTUNE_LEVEL}"
XLA_SCHED="--xla_gpu_all_reduce_combine_threshold_bytes=${COMBINE_TH_BYTES} --xla_gpu_all_gather_combine_threshold_bytes=${COMBINE_TH_BYTES} --xla_gpu_reduce_scatter_combine_threshold_bytes=${COMBINE_TH_BYTES} --xla_gpu_enable_pipelined_all_gather=true --xla_gpu_enable_pipelined_reduce_scatter=true --xla_gpu_enable_pipelined_all_reduce=true --xla_gpu_enable_while_loop_double_buffering=false --xla_gpu_enable_all_gather_combine_by_dim=false --xla_gpu_enable_reduce_scatter_combine_by_dim=false --xla_disable_hlo_passes=rematerialization"
XLA_SCAN_FALSE="--xla_gpu_enable_command_buffer= --xla_gpu_experimental_enable_fusion_autotuner=false --xla_gpu_enable_allocator_spatial_partitioning=false"
export XLA_FLAGS="${XLA_BASE} ${XLA_SCHED} ${XLA_SCAN_FALSE}${EXTRA_XLA:+ ${EXTRA_XLA}}"

OUTDIR="${OUT_ROOT}/${MODE}"
LOGFILE="${OUTDIR}/train.log"
CMD=("$PYTHON_BIN" -m maxtext.trainers.pre_train.train "$MAXTEXT_CONFIG"
  hardware=gpu "model_name=${MODEL_NAME}" "attention=${ATTENTION}"
  enable_checkpointing=False "ici_fsdp_parallelism=${FSDP}" ici_data_parallelism=1 ici_expert_parallelism=1
  "remat_policy=${REMAT_POLICY}" "scan_layers=${SCAN_LAYERS}" "weight_dtype=${WEIGHT_DTYPE}"
  "mu_dtype=${MU_DTYPE}" "use_iota_embed=${USE_IOTA_EMBED}" dataset_type=synthetic
  logits_dot_in_fp32=False dtype=bfloat16 "per_device_batch_size=${PER_DEVICE_BATCH}"
  "global_batch_size_to_train_on=${GLOBAL_BATCH_SIZE}" "init_weights_seed=${INIT_WEIGHTS_SEED}"
  "${MODEL_ARGS[@]}"
  "max_target_length=${MAX_TARGET_LENGTH}" "shardy=${SHARDY}" packing=True max_segments_per_seq=32
  "steps=${STEPS}" "use_jax_aiter=${USE_AITER}" "run_name=${MODE}"
  "quantization=${QUANTIZATION}" "${MODE_ARGS[@]}" "enable_nnx=${ENABLE_NNX}" "pure_nnx=${ENABLE_NNX}"
  "pure_nnx_decoder=${ENABLE_NNX}" "base_output_directory=${OUTDIR}")

print_recipe() {
  printf '%s\n' "=== RESOLVED_RECIPE_BEGIN ==="
  printf 'runner=performance mode=%s label=%s model=%s model_controls=%s processes=1 steps=%s measurement_window=completed_steps_40_49\n' "$MODE" "$LABEL" "$MODEL_NAME" "$MODEL_CONTROLS" "$STEPS"
  printf 'quantization=%s attention=%s use_jax_aiter=%s scan_layers=%s remat_policy=%s\n' "${QUANTIZATION:-none}" "$ATTENTION" "$USE_AITER" "$SCAN_LAYERS" "$REMAT_POLICY"
  printf 'autotune=%s weight_dtype=%s mu_dtype=%s use_iota_embed=%s batch=%s global_batch=%s sequence=%s init_seed=%s mem_fraction=%s\n' "$AUTOTUNE_LEVEL" "$WEIGHT_DTYPE" "$MU_DTYPE" "$USE_IOTA_EMBED" "$PER_DEVICE_BATCH" "$GLOBAL_BATCH_SIZE" "$MAX_TARGET_LENGTH" "$INIT_WEIGHTS_SEED" "$MEMFRAC"
  printf 'fsdp=%s shardy=%s scheduler=nvidia combine_threshold=%s enable_nnx=%s\n' "$FSDP" "$SHARDY" "$COMBINE_TH_BYTES" "$ENABLE_NNX"
  printf 'hadamard_passes=%s sr_passes=%s dgrad_partition=%s dgrad_reuse_fwd_col=%s remat_save_col=%s pack_gateup_ag=%s\n' "${JA_FP4_HADAMARD_PASSES:-n/a}" "${JA_FP4_SR_PASSES:-n/a}" "${JA_FP4_DGRAD_PARTITION:-n/a}" "${JA_FP4_DGRAD_REUSE_FWD_COL:-n/a}" "${JA_FP4_REMAT_SAVE_COL:-n/a}" "${JA_FP4_PACK_GATEUP_AG:-n/a}"
  printf 'fp4_select=%s fp4_attention_gemm=%s sr_key_mode=%s mha_fuse_gqa_reduce=%s mha_zero_pad=%s\n' "${FP4_SELECT:-n/a}" "${AITER_FP4_ATTN:-n/a}" "$SR_KEY_MODE" "${JA_MHA_FUSE_GQA_REDUCE:-n/a}" "${JA_MHA_ZERO_PAD:-n/a}"
  printf 'te_atomic_fp32=%s te_bf16_cvt=%s ja_mha_atomic_fp32=%s ja_mha_bf16_cvt=%s\n' "$NVTE_CK_IS_V3_ATOMIC_FP32" "$NVTE_CK_HOW_V3_BF16_CVT" "${JA_MHA_BWD_ATOMIC_FP32:-n/a}" "${JA_MHA_BWD_BF16_CVT:-n/a}"
  printf 'rocm_safeguards=jax_platforms:%s,queue_interposition:%s,register_enabled:%s,no_scratch_reclaim:%s,dev_kernarg:%s,fine_grain_pcie:%s\n' "$JAX_PLATFORMS" "$ROCPROFILER_QUEUE_INTERPOSITION" "$ROCPROFILER_REGISTER_ENABLED" "$HSA_NO_SCRATCH_RECLAIM" "$HIP_FORCE_DEV_KERNARG" "$HSA_FORCE_FINE_GRAIN_PCIE"
  printf 'project_root=%s jax_aiter_root=%s maxtext_root=%s oom_policy=stop_and_report_no_memfrac_fallback\n' "$PROJECT_ROOT" "$JAX_AITER_ROOT" "$MAXTEXT_ROOT"
  printf '%s\n' "=== RESOLVED_RECIPE_END ==="
}

if is_true "${RECIPE_DRY_RUN:-0}"; then
  print_recipe
  printf 'XLA_FLAGS=%s\n' "$XLA_FLAGS"
  printf 'RESOLVED_ARG=%s\n' "${CMD[@]}"
  exit 0
fi

[[ -d "$MAXTEXT_ROOT" ]] || die "MAXTEXT_ROOT does not exist: $MAXTEXT_ROOT"
mkdir -p "$OUTDIR"
[[ ! -e "$LOGFILE" ]] || die "train.log already exists; use a fresh output root: $LOGFILE"
LAUNCH_MARKER="${OUTDIR}/.launch_once"
if ! (set -o noclobber; printf '%s mode=%s pid=%s\n' "$(date -u)" "$MODE" "$$" >"$LAUNCH_MARKER") 2>/dev/null; then
  die "launch already attempted for $OUTDIR; use a fresh output root"
fi
{ print_recipe; printf 'XLA_FLAGS=%s\n' "$XLA_FLAGS"; printf 'COMMAND='; printf ' %q' "${CMD[@]}"; printf '\n'; } | tee "$LOGFILE"
start_ts="$(date +%s)"
set +e
(cd "$MAXTEXT_ROOT" && "${CMD[@]}") 2>&1 | tee -a "$LOGFILE"
rc=${PIPESTATUS[0]}
set -e
printf '=== %s EXIT %s wall=%ss %s ===\n' "$MODE" "$rc" "$(( $(date +%s) - start_ts ))" "$(date -u)" | tee -a "$LOGFILE"
exit "$rc"
