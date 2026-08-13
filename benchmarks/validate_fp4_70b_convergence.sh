#!/bin/bash
# FP4 loss convergence validation -- Llama3.3-70B.
# Reports step-by-step loss for fp4 vs fp8_baseline to verify the FP4 recipe
# converges similarly to native FP8.
#
# Usage (inside container):
#   bash tests/validate_fp4_70b_convergence.sh          # 50 steps
#   bash tests/validate_fp4_70b_convergence.sh 200      # 200 steps
set -uo pipefail

STEPS="${1:-50}"

cd /ruvaidya/aiter_proj/maxtext

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
export PYTHONPATH=/ruvaidya/aiter_proj/maxtext/src:${PYTHONPATH:-}
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

XLA_BASE='--xla_gpu_memory_limit_slop_factor=95 --xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592 --xla_gpu_enable_command_buffer= --xla_gpu_enable_latency_hiding_scheduler=True --xla_gpu_all_gather_combine_threshold_bytes=8589934592 --xla_gpu_enable_triton_gemm=False --xla_gpu_enable_cublaslt=True --xla_gpu_autotune_level=4 --xla_gpu_enable_all_gather_combine_by_dim=FALSE --xla_gpu_enable_nccl_comm_splitting=false'

LOGDIR=/ruvaidya/aiter_proj/docs/logs/fp4_70b_convergence
mkdir -p "$LOGDIR"

MODEL_70B="src/maxtext/configs/base.yml \
  hardware=gpu model_name=llama3.3-70b attention=cudnn_flash_te \
  enable_checkpointing=False \
  ici_fsdp_parallelism=8 ici_data_parallelism=1 ici_expert_parallelism=1 \
  remat_policy=full scan_layers=True param_scan_axis=1 dataset_type=synthetic \
  logits_dot_in_fp32=False dtype=bfloat16 weight_dtype=bfloat16 \
  per_device_batch_size=10 max_target_length=8192 \
  use_iota_embed=True shardy=False packing=False \
  base_output_directory=/tmp/maxtext_output"

set_aiter_env() {
    export JA_ROOT_DIR=/ruvaidya/aiter_proj/jax-aiter
    export AITER_ASM_DIR=/ruvaidya/aiter_proj/jax-aiter/third_party/aiter/hsa/
    export AITER_SYMBOL_VISIBLE=1
    export GPU_ARCHS=gfx950
    export AITER_FP4_ATTN=1
}

clean_env() {
    unset JA_ROOT_DIR AITER_ASM_DIR AITER_SYMBOL_VISIBLE GPU_ARCHS \
          AITER_FP4_ATTN AITER_FUSED_QUANT_HADAMARD AITER_KERNEL_SEL 2>/dev/null || true
}

echo "======================================================================"
echo "FP4 vs FP8 Convergence -- Llama3.3-70B, ${STEPS} steps, 8x MI355X"
echo "======================================================================"

# --- Run 1: fp8_baseline (cool GPUs first for thermal hygiene) ---
echo ""
echo "-- fp8_baseline (native hipBLASLt FP8) --"
clean_env
export XLA_FLAGS="${XLA_BASE}"
set +e
python3 -m maxtext.trainers.pre_train.train $MODEL_70B \
    steps=$STEPS use_jax_aiter=False quantization=fp8 \
    run_name=conv_fp8_70b \
    2>&1 | tee "$LOGDIR/fp8_baseline.log" | grep -E 'completed step' | tail -5
RC1=$?
set -e

# --- Run 2: fp4 (the new default; Hadamard ON for grad cast) ---
echo ""
echo "-- fp4 (AITER FP4 / MXFP4 with Hadamard grad quant) --"
clean_env
set_aiter_env
export XLA_FLAGS="${XLA_BASE}"
set +e
python3 -m maxtext.trainers.pre_train.train $MODEL_70B \
    steps=$STEPS use_jax_aiter=True aiter_attention=False \
    quantization=aiter_fp4 run_name=conv_fp4_70b \
    2>&1 | tee "$LOGDIR/fp4.log" | grep -E 'completed step' | tail -5
RC2=$?
set -e

echo ""
echo "======================================================================"
echo "Loss Trajectory -- ${STEPS} steps"
echo "======================================================================"
echo ""
printf "  %-6s  %-12s  %-12s  %-10s\n" "Step" "FP8" "FP4" "|Delta|"
printf "  %-6s  %-12s  %-12s  %-10s\n" "----" "---" "---" "-------"

for step in 1 5 9 19 29 39 49 99 149 199 $STEPS; do
    [[ $step -gt $STEPS ]] && continue
    fp8_loss=$(grep "completed step: $step[^0-9]" "$LOGDIR/fp8_baseline.log" 2>/dev/null | \
        tail -1 | grep -oP 'loss: \K[0-9.nanif]+' || echo "N/A")
    fp4_loss=$(grep "completed step: $step[^0-9]" "$LOGDIR/fp4.log" 2>/dev/null | \
        tail -1 | grep -oP 'loss: \K[0-9.nanif]+' || echo "N/A")
    if [[ "$fp8_loss" =~ ^[0-9.]+$ ]] && [[ "$fp4_loss" =~ ^[0-9.]+$ ]]; then
        delta=$(awk "BEGIN {printf \"%.4f\", ($fp4_loss - $fp8_loss) < 0 ? -($fp4_loss - $fp8_loss) : ($fp4_loss - $fp8_loss)}")
    else
        delta="N/A"
    fi
    printf "  %-6s  %-12s  %-12s  %-10s\n" "$step" "$fp8_loss" "$fp4_loss" "$delta"
done

echo ""
echo "  Pass criteria: |fp4_loss - fp8_loss| < 0.05 at step $STEPS (and trending stable / down)."
echo "======================================================================"
echo ""
echo "Logs: $LOGDIR/"
