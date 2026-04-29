#!/bin/bash
# FP4 training validation -- Llama3.3-70B.
# Compares:
#   1. fp8_baseline -- native Flax FP8 reference
#   2. fp4          -- AITER FP4 (MXFP4) recipe (default for jax-aiter on MI350/MI355X)
#
# Reports tail-20 avg to capture steady-state perf.
#
# Usage (inside container):
#   bash tests/validate_fp4_e2e_70b.sh          # 30 steps
#   bash tests/validate_fp4_e2e_70b.sh 40       # 40 steps
set -uo pipefail

STEPS="${1:-30}"

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
# Append additional XLA flags via env var (e.g., for command-buffer experiments):
#   XLA_FLAGS_EXTRA='--xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL' bash ...
XLA_BASE="${XLA_BASE} ${XLA_FLAGS_EXTRA:-}"

LOGDIR=/ruvaidya/aiter_proj/docs/logs/fp4_70b
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

declare -a CFG_NAME=( "fp8_baseline" "fp4"      )
declare -a CFG_AITER=( "False"        "True"     )
declare -a CFG_QUANT=( "fp8"          "aiter_fp4" )

# Skip FP8 baseline if already measured: SKIP_FP8=1 bash tests/validate_fp4_e2e_70b.sh ...
# (FP8 numbers are stable; re-running wastes ~10 min GPU.)
if [[ "${SKIP_FP8:-0}" == "1" ]]; then
    declare -a CFG_NAME=( "fp4" )
    declare -a CFG_AITER=( "True" )
    declare -a CFG_QUANT=( "aiter_fp4" )
fi

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
echo "FP4 Validation -- Llama3.3-70B, ${STEPS} steps, 8x MI355X"
echo "======================================================================"

declare -a ALL_TAGS=()
declare -a ALL_RC=()

for ((c=0; c<${#CFG_NAME[@]}; c++)); do
    NAME="${CFG_NAME[$c]}"
    USE_AITER="${CFG_AITER[$c]}"
    QUANT="${CFG_QUANT[$c]}"
    TAG="${NAME}"
    LOGFILE="${LOGDIR}/${TAG}.log"
    ALL_TAGS+=("$TAG")

    echo ""
    echo "-- ${TAG} --"

    clean_env
    if [[ "$USE_AITER" == "True" ]]; then
        set_aiter_env
    fi
    export XLA_FLAGS="${XLA_BASE}"

    EXTRA_ARGS="steps=$STEPS use_jax_aiter=$USE_AITER run_name=fp4_70b_${TAG} quantization=$QUANT"
    if [[ "$USE_AITER" == "True" ]]; then
        EXTRA_ARGS+=" aiter_attention=False"
    fi

    set +e
    python3 -m maxtext.trainers.pre_train.train $MODEL_70B \
        $EXTRA_ARGS \
        2>&1 | tee "$LOGFILE" | grep -E 'completed step' | tail -3
    RC=$?
    set -e

    ALL_RC+=($RC)
    echo "  Exit: $RC"
done

echo ""
echo "======================================================================"
echo "FP4 70B RESULTS -- ${STEPS} steps"
echo "======================================================================"
printf "  %-30s  %-10s  %-14s  %-10s  %-6s  %s\n" \
    "Config" "TFLOP/s" "Tokens/s/GPU" "Step(s)" "Exit" "Loss"
printf "  %-30s  %-10s  %-14s  %-10s  %-6s  %s\n" \
    "------" "-------" "------------" "-------" "----" "----"

for ((i=0; i<${#ALL_TAGS[@]}; i++)); do
    TAG="${ALL_TAGS[$i]}"
    LOGFILE="${LOGDIR}/${TAG}.log"

    loss_last=$(grep 'completed step' "$LOGFILE" 2>/dev/null | tail -1 | \
        grep -oP 'loss: \K[0-9.nanif]+' || echo "N/A")
    tflops=$(grep 'completed step' "$LOGFILE" 2>/dev/null | tail -20 | \
        grep -oP 'TFLOP/s/device: \K[0-9.]+' | \
        awk '{sum+=$1; n++} END {if(n>0) printf "%.1f", sum/n; else print "N/A"}')
    toks=$(grep 'completed step' "$LOGFILE" 2>/dev/null | tail -20 | \
        grep -oP 'Tokens/s/device: \K[0-9.]+' | \
        awk '{sum+=$1; n++} END {if(n>0) printf "%.1f", sum/n; else print "N/A"}')
    steptime=$(grep 'completed step' "$LOGFILE" 2>/dev/null | tail -20 | \
        grep -oP 'seconds: \K[0-9.]+' | \
        awk '{sum+=$1; n++} END {if(n>0) printf "%.3f", sum/n; else print "N/A"}')

    printf "  %-30s  %-10s  %-14s  %-10s  %-6s  %s\n" \
        "$TAG" "$tflops" "$toks" "${steptime}s" "${ALL_RC[$i]}" "$loss_last"
done

echo ""
echo "  Reference (session 20, 2026-04-20, 8x MI355X, 30-step):"
echo "    fp8_baseline:  ~1,619 TFLOP/s/dev"
echo "    fp4:           ~1,716 TFLOP/s/dev (+6.0% vs FP8)"
echo "======================================================================"
