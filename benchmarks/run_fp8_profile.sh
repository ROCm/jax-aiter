#!/bin/bash
# Collect jax.profiler (xplane) traces for FP8 baseline vs AITER BF16.
#
# Captures 2 training steps after 3 warmup steps, producing TensorBoard-
# compatible traces for per-op timing breakdown.
#
# Usage (inside container):
#   bash benchmarks/run_fp8_profile.sh [8b|70b]
set -uo pipefail

MODEL_SIZE="${1:-8b}"

cd /ruvaidya/aiter_proj/maxtext

# ── Shared env (MAD-aligned) ──────────────────────────────────────────
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=.97
export NVTE_USE_HIPBLASLT=1
export NVTE_CK_USES_BWD_V3=1
export NVTE_CK_USES_FWD_V3=1
export NVTE_CK_IS_V3_ATOMIC_FP32=0
export NVTE_CK_HOW_V3_BF16_CVT=2
export NVTE_FUSED_ATTN=1
export NVTE_FUSED_ATTN_CK=1
export NVTE_FUSED_ATTN_AOTRITON=0
export GPU_MAX_HW_QUEUES=2
export HIP_FORCE_DEV_KERNARG=1
export HSA_FORCE_FINE_GRAIN_PCIE=1
export JAX_PLATFORMS=rocm
export DECOUPLE_GCLOUD=TRUE
export PYTHONPATH=/ruvaidya/aiter_proj/maxtext/src:${PYTHONPATH:-}
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

XLA_BASE='--xla_gpu_memory_limit_slop_factor=95 --xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592 --xla_gpu_enable_command_buffer= --xla_gpu_enable_latency_hiding_scheduler=True --xla_gpu_all_gather_combine_threshold_bytes=8589934592 --xla_gpu_enable_triton_gemm=False --xla_gpu_enable_cublaslt=True --xla_gpu_autotune_level=4 --xla_gpu_enable_all_gather_combine_by_dim=FALSE --xla_gpu_enable_nccl_comm_splitting=false'

PROFILE_DIR=/ruvaidya/aiter_proj/jax-aiter/hlo_dumps/profiles
mkdir -p "$PROFILE_DIR"

LOGDIR=/ruvaidya/aiter_proj/docs/logs/fp8_profiles
mkdir -p "$LOGDIR"

case "$MODEL_SIZE" in
    8b)
        MAXTEXT_MODEL="src/maxtext/configs/base.yml \
          hardware=gpu model_name=llama3-8b attention=cudnn_flash_te \
          enable_checkpointing=False \
          ici_fsdp_parallelism=8 ici_data_parallelism=1 ici_expert_parallelism=1 \
          remat_policy=minimal_flash scan_layers=True dataset_type=synthetic \
          logits_dot_in_fp32=False dtype=bfloat16 weight_dtype=bfloat16 \
          per_device_batch_size=9 max_target_length=8192 shardy=False packing=False \
          base_output_directory=${PROFILE_DIR}"
        ;;
    70b)
        MAXTEXT_MODEL="src/maxtext/configs/base.yml \
          hardware=gpu model_name=llama3.3-70b attention=cudnn_flash_te \
          enable_checkpointing=False \
          ici_fsdp_parallelism=8 ici_data_parallelism=1 ici_expert_parallelism=1 \
          remat_policy=full scan_layers=True param_scan_axis=1 dataset_type=synthetic \
          logits_dot_in_fp32=False dtype=bfloat16 weight_dtype=bfloat16 \
          per_device_batch_size=10 max_target_length=8192 \
          use_iota_embed=True shardy=False packing=False \
          base_output_directory=${PROFILE_DIR}"
        ;;
    *)
        echo "Usage: $0 {8b|70b}"
        exit 1
        ;;
esac

PROFILE_ARGS="profiler=xplane skip_first_n_steps_for_profiler=3 profiler_steps=2"

echo "======================================================================"
echo "FP8 Profiler Trace Collection — ${MODEL_SIZE^^}"
echo "  Profile dir: $PROFILE_DIR"
echo "  Capturing steps 4-5 (after 3 warmup)"
echo "======================================================================"

# ── Run 1: FP8 baseline ──────────────────────────────────────────────
echo ""
echo "── Run 1: FP8 baseline (quantization=fp8) ──"
export XLA_FLAGS="$XLA_BASE"

set +e
python3 -m maxtext.trainers.pre_train.train $MAXTEXT_MODEL \
    steps=6 quantization=fp8 use_jax_aiter=False \
    run_name=profile_fp8_${MODEL_SIZE} \
    $PROFILE_ARGS \
    2>&1 | tee "$LOGDIR/profile_fp8_${MODEL_SIZE}.log" | grep -E 'completed step|profiler'
RC_FP8=$?
set -e
echo "  Exit: $RC_FP8"

# ── Run 2: AITER BF16 ────────────────────────────────────────────────
echo ""
echo "── Run 2: AITER BF16 (use_jax_aiter=True) ──"
export JA_ROOT_DIR=/ruvaidya/aiter_proj/jax-aiter
export AITER_ASM_DIR=/ruvaidya/aiter_proj/jax-aiter/third_party/aiter/hsa/
export AITER_SYMBOL_VISIBLE=1
export GPU_ARCHS=gfx950
export XLA_FLAGS="$XLA_BASE"

set +e
python3 -m maxtext.trainers.pre_train.train $MAXTEXT_MODEL \
    steps=6 use_jax_aiter=True aiter_attention=False \
    run_name=profile_aiter_bf16_${MODEL_SIZE} \
    $PROFILE_ARGS \
    2>&1 | tee "$LOGDIR/profile_aiter_bf16_${MODEL_SIZE}.log" | grep -E 'completed step|profiler'
RC_AITER=$?
set -e
echo "  Exit: $RC_AITER"

# ── Run 3: BF16 baseline ─────────────────────────────────────────────
echo ""
echo "── Run 3: BF16 baseline ──"
unset JA_ROOT_DIR AITER_ASM_DIR AITER_SYMBOL_VISIBLE GPU_ARCHS 2>/dev/null || true
export XLA_FLAGS="$XLA_BASE"

set +e
python3 -m maxtext.trainers.pre_train.train $MAXTEXT_MODEL \
    steps=6 use_jax_aiter=False \
    run_name=profile_bf16_${MODEL_SIZE} \
    $PROFILE_ARGS \
    2>&1 | tee "$LOGDIR/profile_bf16_${MODEL_SIZE}.log" | grep -E 'completed step|profiler'
RC_BF16=$?
set -e
echo "  Exit: $RC_BF16"

echo ""
echo "======================================================================"
echo "Profile traces saved under: $PROFILE_DIR/"
echo "View with: tensorboard --logdir=$PROFILE_DIR"
echo "======================================================================"
