#!/bin/bash
# FP4 Attention 70B Benchmark — Llama3.3-70B, 10 steps
#
# Measures the impact of AITER_FP4_ATTN=1 at 70B scale.
# Prior 70B MXFP4 (BF16 attn) = 1,462 T/s (-9.6% vs FP8).
# 8B FP4 attn = 1,440 T/s (+2.8% vs FP8).
#
# Configurations:
#   1. MXFP4 + FP4 attn + FP8 dB  (new best candidate)
#   2. MXFP4 + FP8 dB             (prior production, BF16 attn)
#   3. Native FP8                  (same-day baseline)
#
# Usage (inside container):
#   bash tests/bench_fp4attn_70b.sh [steps]
set -uo pipefail

STEPS="${1:-10}"

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

LOGDIR=/ruvaidya/aiter_proj/docs/logs/fp4attn_70b
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

clean_aiter_env() {
    unset JA_ROOT_DIR AITER_ASM_DIR AITER_SYMBOL_VISIBLE GPU_ARCHS 2>/dev/null || true
    unset AITER_FP4_ATTN AITER_FUSED_QUANT_HADAMARD AITER_KERNEL_SEL 2>/dev/null || true
}

set_aiter_env() {
    export JA_ROOT_DIR=/ruvaidya/aiter_proj/jax-aiter
    export AITER_ASM_DIR=/ruvaidya/aiter_proj/jax-aiter/third_party/aiter/hsa/
    export AITER_SYMBOL_VISIBLE=1
    export GPU_ARCHS=gfx950
}

get_metrics() {
    local log="$1"
    local tflops toks loss mem first_nan
    tflops=$(grep 'completed step' "$log" 2>/dev/null | tail -5 | grep -oP 'TFLOP/s/device: \K[0-9.]+' | awk '{s+=$1;n++} END{if(n>0) printf "%.1f",s/n; else print "N/A"}')
    toks=$(grep 'completed step' "$log" 2>/dev/null | tail -5 | grep -oP 'Tokens/s/device: \K[0-9.]+' | awk '{s+=$1;n++} END{if(n>0) printf "%.1f",s/n; else print "N/A"}')
    loss=$(grep 'completed step' "$log" 2>/dev/null | tail -1 | grep -oP 'loss: \K[0-9.nanif]+' || echo "N/A")
    first_nan=$(grep 'completed step' "$log" 2>/dev/null | grep -oP 'step: \K\d+.*loss: (nan|inf)' | head -1 | grep -oP '^\d+' || echo "none")
    echo "$tflops $toks $loss $first_nan"
}

echo "======================================================================"
echo "FP4 Attention 70B Benchmark — Llama3.3-70B, $STEPS steps, 8x MI355X"
echo "======================================================================"

# --- Run 1: FP4 with FP4 attention ---
echo ""
echo "── Run 1/3: FP4 with FP4 attention (AITER_FP4_ATTN=1) ──"
clean_aiter_env; set_aiter_env
export AITER_FP4_ATTN=1
export XLA_FLAGS="${XLA_BASE}"
set +e
python3 -m maxtext.trainers.pre_train.train $MODEL_70B \
    steps=$STEPS use_jax_aiter=True aiter_attention=False \
    quantization=aiter_fp4 run_name=fp4attn_70b \
    2>&1 | tee "$LOGDIR/70b_fp4_attn_on.log" | grep -E 'completed step' | tail -5
RC1=$?
set -e
echo "  Exit: $RC1"

# --- Run 2: FP4 with BF16 attention (AITER_FP4_ATTN=0) ---
echo ""
echo "── Run 2/3: FP4 with BF16 attention (AITER_FP4_ATTN=0) ──"
clean_aiter_env; set_aiter_env
export AITER_FP4_ATTN=0
export XLA_FLAGS="${XLA_BASE}"
set +e
python3 -m maxtext.trainers.pre_train.train $MODEL_70B \
    steps=$STEPS use_jax_aiter=True aiter_attention=False \
    quantization=aiter_fp4 run_name=fp4_attn_off_70b \
    2>&1 | tee "$LOGDIR/70b_fp4_attn_off.log" | grep -E 'completed step' | tail -5
RC2=$?
set -e
echo "  Exit: $RC2"

# --- Run 3: Native FP8 (same-day baseline) ---
echo ""
echo "── Run 3/3: Native FP8 (same-day baseline) ──"
clean_aiter_env
export XLA_FLAGS="${XLA_BASE}"
set +e
python3 -m maxtext.trainers.pre_train.train $MODEL_70B \
    steps=$STEPS use_jax_aiter=False \
    quantization=fp8 run_name=fp8_baseline_70b \
    2>&1 | tee "$LOGDIR/70b_fp8_baseline.log" | grep -E 'completed step' | tail -5
RC3=$?
set -e
echo "  Exit: $RC3"

# --- Results ---
echo ""
echo "======================================================================"
echo "FP4 ATTENTION 70B RESULTS — $STEPS steps, 8x MI355X"
echo "======================================================================"
printf "  %-40s  %-10s  %-14s  %-8s  %-6s  %s\n" "Config" "TFLOP/s" "Tok/s/GPU" "NaN" "Exit" "Loss"
printf "  %-40s  %-10s  %-14s  %-8s  %-6s  %s\n" "------" "-------" "---------" "---" "----" "----"

for tag_rc in "70b_fp4attn_fp8db:$RC1" "70b_mxfp4_fp8db:$RC2" "70b_fp8_baseline:$RC3"; do
    tag="${tag_rc%%:*}"; rc="${tag_rc##*:}"
    read tflops toks loss nan <<< "$(get_metrics "$LOGDIR/${tag}.log")"
    printf "  %-40s  %-10s  %-14s  %-8s  %-6s  %s\n" "$tag" "$tflops" "$toks" "$nan" "$rc" "$loss"
done

fp4attn_t=$(grep 'completed step' "$LOGDIR/70b_fp4attn_fp8db.log" 2>/dev/null | tail -5 | grep -oP 'TFLOP/s/device: \K[0-9.]+' | awk '{s+=$1;n++} END{if(n>0) printf "%.1f",s/n; else print "0"}')
mxfp4_t=$(grep 'completed step' "$LOGDIR/70b_mxfp4_fp8db.log" 2>/dev/null | tail -5 | grep -oP 'TFLOP/s/device: \K[0-9.]+' | awk '{s+=$1;n++} END{if(n>0) printf "%.1f",s/n; else print "0"}')
fp8_t=$(grep 'completed step' "$LOGDIR/70b_fp8_baseline.log" 2>/dev/null | tail -5 | grep -oP 'TFLOP/s/device: \K[0-9.]+' | awk '{s+=$1;n++} END{if(n>0) printf "%.1f",s/n; else print "0"}')

echo ""
if [[ "$fp4attn_t" != "0" && "$fp8_t" != "0" ]]; then
    gain=$(awk "BEGIN {printf \"%.1f\", ($fp4attn_t/$fp8_t - 1)*100}")
    echo "  FP4-attn+FP8-dB vs FP8:    ${fp4attn_t} / ${fp8_t} = ${gain}%"
fi
if [[ "$mxfp4_t" != "0" && "$fp8_t" != "0" ]]; then
    gain=$(awk "BEGIN {printf \"%.1f\", ($mxfp4_t/$fp8_t - 1)*100}")
    echo "  MXFP4 (BF16 attn) vs FP8:  ${mxfp4_t} / ${fp8_t} = ${gain}%"
fi
if [[ "$fp4attn_t" != "0" && "$mxfp4_t" != "0" ]]; then
    gain=$(awk "BEGIN {printf \"%.1f\", ($fp4attn_t/$mxfp4_t - 1)*100}")
    echo "  FP4-attn vs BF16-attn:      ${fp4attn_t} / ${mxfp4_t} = ${gain}%"
fi

echo ""
echo "  Reference (prior measurements):"
echo "    70B MXFP4 (BF16 attn):  1,462 T/s  (-9.6% vs FP8)"
echo "    70B Hybrid FP4+FP8:     1,507 T/s  (-2.7% vs FP8)"
echo "    70B FP8 baseline:       1,548 T/s"
echo "    8B FP4-attn+FP8-dB:     1,440 T/s  (+2.8% vs FP8)"
echo "======================================================================"
echo ""
echo "Logs: $LOGDIR/"
