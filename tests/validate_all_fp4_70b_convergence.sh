#!/bin/bash
# All-FP4 loss convergence validation — Llama3.3-70B, 50 steps.
# Mirrors session-18 convergence check used for the hybrid production recipe.
# Reports step-by-step loss for hybrid_prod and all_fp4 to verify the all-FP4
# recipe converges to within 0.03 of hybrid by step 49.
#
# Usage (inside container):
#   bash tests/validate_all_fp4_70b_convergence.sh          # 50 steps
#   bash tests/validate_all_fp4_70b_convergence.sh 100      # 100 steps
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

LOGDIR=/ruvaidya/aiter_proj/docs/logs/all_fp4_70b_convergence
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
}

clean_aiter() {
    unset AITER_FP4_DA AITER_FP4_DB AITER_FP8_DB AITER_FUSED_QUANT \
          AITER_FP4_ATTN AITER_ALL_FP4 2>/dev/null || true
}

echo "======================================================================"
echo "All-FP4 Convergence — Llama3.3-70B, ${STEPS} steps, 8x MI355X"
echo "======================================================================"

# --- Run 1: hybrid_prod ---
echo ""
echo "── hybrid_prod (FP4 fwd + FP4 dA + FP8 dB) ──"
clean_aiter; set_aiter_env
export AITER_FP4_DA=1 AITER_FP4_DB=0 AITER_FP8_DB=1 AITER_FUSED_QUANT=1 AITER_FP4_ATTN=1 AITER_ALL_FP4=0
export XLA_FLAGS="${XLA_BASE}"
set +e
python3 -m maxtext.trainers.pre_train.train $MODEL_70B \
    steps=$STEPS use_jax_aiter=True aiter_attention=False \
    quantization=aiter_mxfp4 run_name=conv_hybrid_70b \
    2>&1 | tee "$LOGDIR/hybrid_prod.log" | grep -E 'completed step' | tail -5
RC1=$?
set -e

# --- Run 2: all_fp4 ---
echo ""
echo "── all_fp4 (FP4 fwd + FP4 dA + FP4 dB wgrad-sharded) ──"
clean_aiter; set_aiter_env
export AITER_FP4_DA=1 AITER_FUSED_QUANT=1 AITER_FP4_ATTN=1 AITER_ALL_FP4=1
export XLA_FLAGS="${XLA_BASE}"
set +e
python3 -m maxtext.trainers.pre_train.train $MODEL_70B \
    steps=$STEPS use_jax_aiter=True aiter_attention=False \
    quantization=aiter_mxfp4 run_name=conv_all_fp4_70b \
    2>&1 | tee "$LOGDIR/all_fp4.log" | grep -E 'completed step' | tail -5
RC2=$?
set -e

echo ""
echo "======================================================================"
echo "Loss Trajectory — ${STEPS} steps"
echo "======================================================================"
echo ""
printf "  %-6s  %-12s  %-12s  %-10s\n" "Step" "Hybrid" "All-FP4" "|Delta|"
printf "  %-6s  %-12s  %-12s  %-10s\n" "----" "------" "-------" "-------"

for step in 1 5 9 19 29 39 49 $STEPS; do
    [[ $step -gt $STEPS ]] && continue
    hyb_loss=$(grep "completed step: $step[^0-9]" "$LOGDIR/hybrid_prod.log" 2>/dev/null | \
        tail -1 | grep -oP 'loss: \K[0-9.nanif]+' || echo "N/A")
    all_loss=$(grep "completed step: $step[^0-9]" "$LOGDIR/all_fp4.log" 2>/dev/null | \
        tail -1 | grep -oP 'loss: \K[0-9.nanif]+' || echo "N/A")
    if [[ "$hyb_loss" =~ ^[0-9.]+$ ]] && [[ "$all_loss" =~ ^[0-9.]+$ ]]; then
        delta=$(awk "BEGIN {printf \"%.4f\", ($all_loss - $hyb_loss) < 0 ? -($all_loss - $hyb_loss) : ($all_loss - $hyb_loss)}")
    else
        delta="N/A"
    fi
    printf "  %-6s  %-12s  %-12s  %-10s\n" "$step" "$hyb_loss" "$all_loss" "$delta"
done

echo ""
echo "  Promotion gate: |all_fp4_loss - hybrid_loss| < 0.03 at step $STEPS"
echo "======================================================================"
echo ""
echo "Logs: $LOGDIR/"
