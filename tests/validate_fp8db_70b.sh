#!/bin/bash
# FP8 dB 70B Validation — 20-step, MXFP4 baseline vs MXFP4+FP8_dB
#
# Tests whether the +4.5% FP8 dB gain at 8B holds or grows at 70B.
# Prior: MXFP4+FP4_dA = 1,262 T/s (+28.9% vs BF16 979 T/s)
#
# Usage (inside container):
#   bash tests/validate_fp8db_70b.sh           # 20 steps
#   bash tests/validate_fp8db_70b.sh 10        # 10 steps (quick)
set -uo pipefail

STEPS="${1:-20}"

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

LOGDIR=/ruvaidya/aiter_proj/docs/logs/fp8db_spike
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

extract_metrics() {
    local logfile="$1" tail_n="${2:-10}"
    if [[ ! -f "$logfile" ]]; then echo "N/A N/A N/A N/A"; return; fi
    local tflops toks loss first_nan
    tflops=$(grep 'completed step' "$logfile" 2>/dev/null | tail -"$tail_n" | grep -oP 'TFLOP/s/device: \K[0-9.]+' | awk '{s+=$1;n++} END{if(n>0) printf "%.1f",s/n; else print "N/A"}')
    toks=$(grep 'completed step' "$logfile" 2>/dev/null | tail -"$tail_n" | grep -oP 'Tokens/s/device: \K[0-9.]+' | awk '{s+=$1;n++} END{if(n>0) printf "%.1f",s/n; else print "N/A"}')
    loss=$(grep 'completed step' "$logfile" 2>/dev/null | tail -1 | grep -oP 'loss: \K[0-9.nanif]+' || echo "N/A")
    first_nan=$(grep 'completed step' "$logfile" 2>/dev/null | grep -oP 'step: \K\d+.*loss: (nan|inf)' | head -1 | grep -oP '^\d+' || echo "none")
    echo "$tflops $toks $loss $first_nan"
}

echo "======================================================================"
echo "FP8 dB 70B Validation — Llama3.3-70B, ${STEPS} steps"
echo "======================================================================"

# --- Run 1: MXFP4 + FP4 dA (current baseline) ---
echo ""
echo "── 70B MXFP4 + FP4 dA (baseline) ──"
set_aiter_env
export AITER_FP4_DA=1
unset AITER_FP8_DB 2>/dev/null || true
export XLA_FLAGS="${XLA_BASE}"
set +e
python3 -m maxtext.trainers.pre_train.train $MODEL_70B \
    steps=$STEPS use_jax_aiter=True aiter_attention=False \
    quantization=aiter_mxfp4 run_name=fp8db_70b_baseline \
    2>&1 | tee "$LOGDIR/70b_baseline.log" | grep -E 'completed step' | tail -5
RC1=$?
set -e
echo "  Exit: $RC1"

# --- Run 2: MXFP4 + FP4 dA + FP8 dB ---
echo ""
echo "── 70B MXFP4 + FP4 dA + FP8 dB ──"
set_aiter_env
export AITER_FP4_DA=1
export AITER_FP8_DB=1
export XLA_FLAGS="${XLA_BASE}"
set +e
python3 -m maxtext.trainers.pre_train.train $MODEL_70B \
    steps=$STEPS use_jax_aiter=True aiter_attention=False \
    quantization=aiter_mxfp4 run_name=fp8db_70b_test \
    2>&1 | tee "$LOGDIR/70b_fp8db.log" | grep -E 'completed step' | tail -5
RC2=$?
set -e
echo "  Exit: $RC2"
unset AITER_FP8_DB

# --- Results ---
echo ""
echo "======================================================================"
echo "FP8 dB 70B RESULTS — ${STEPS} steps, packing=False"
echo "======================================================================"
printf "  %-30s  %-10s  %-14s  %-8s  %-6s  %s\n" "Config" "TFLOP/s" "Tok/s/GPU" "NaN" "Exit" "Loss"
printf "  %-30s  %-10s  %-14s  %-8s  %-6s  %s\n" "------" "-------" "---------" "---" "----" "----"

for tag_rc in "70b_baseline:$RC1" "70b_fp8db:$RC2"; do
    tag="${tag_rc%%:*}"; rc="${tag_rc##*:}"
    read tflops toks loss nan <<< "$(extract_metrics "$LOGDIR/${tag}.log")"
    printf "  %-30s  %-10s  %-14s  %-8s  %-6s  %s\n" "$tag" "$tflops" "$toks" "$nan" "$rc" "$loss"
done

echo ""
echo "  Reference (prior measurements):"
echo "    70B BF16 baseline:      ~979 TFLOP/s   (~2,179 tok/s)"
echo "    70B MXFP4 + FP4 dA:    ~1,262 TFLOP/s  (~2,810 tok/s, +28.9%)"
echo "    70B TE FP8:             ~1,664 TFLOP/s  (~3,703 tok/s, +70%)"
echo "    8B FP8 dB gain:        +4.5% (1,268 -> 1,326)"
echo "======================================================================"
echo ""
echo "Logs: $LOGDIR/"
