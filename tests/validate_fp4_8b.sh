#!/bin/bash
# FP4 Validation — Llama3.1-8B, 4 configs x 3 repeats x 100 steps
#
# Configs:
#   1. bf16_baseline   — no AITER, no quantization
#   2. aiter_bf16      — AITER BF16 ASM GEMM
#   3. aiter_mxfp4     — AITER FP4 MLP forward + BF16 backward
#   4. fp8_baseline    — native Flax FP8 (hipBLASLt FP8)
#
# Usage (inside container):
#   bash tests/validate_fp4_8b.sh           # 100 steps, 3 repeats
#   bash tests/validate_fp4_8b.sh 200       # 200 steps, 3 repeats
#   bash tests/validate_fp4_8b.sh 100 1     # 100 steps, 1 repeat (quick)
set -uo pipefail

STEPS="${1:-100}"
REPEATS="${2:-3}"

cd /ruvaidya/aiter_proj/maxtext

# ── Shared env (MAD-aligned) ──────────────────────────────────────────
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

LOGDIR=/ruvaidya/aiter_proj/docs/logs/fp4_validation
mkdir -p "$LOGDIR"

MODEL="src/maxtext/configs/base.yml \
  hardware=gpu model_name=llama3-8b attention=cudnn_flash_te \
  enable_checkpointing=False \
  ici_fsdp_parallelism=8 ici_data_parallelism=1 ici_expert_parallelism=1 \
  remat_policy=minimal_flash scan_layers=True dataset_type=synthetic \
  logits_dot_in_fp32=False dtype=bfloat16 weight_dtype=bfloat16 \
  per_device_batch_size=9 max_target_length=8192 shardy=False packing=False \
  base_output_directory=/tmp/maxtext_output"

# ── Config definitions ────────────────────────────────────────────────
declare -a CFG_NAME=( "bf16_baseline" "aiter_bf16" "aiter_mxfp4" "fp8_baseline" )
declare -a CFG_AITER=( "False"         "True"       "True"        "False"        )
declare -a CFG_QUANT=( ""              ""           "aiter_mxfp4" "fp8"          )
NUM_CFGS=${#CFG_NAME[@]}

clean_aiter_env() {
    unset JA_ROOT_DIR AITER_ASM_DIR AITER_SYMBOL_VISIBLE GPU_ARCHS 2>/dev/null || true
}

set_aiter_env() {
    export JA_ROOT_DIR=/ruvaidya/aiter_proj/jax-aiter
    export AITER_ASM_DIR=/ruvaidya/aiter_proj/jax-aiter/third_party/aiter/hsa/
    export AITER_SYMBOL_VISIBLE=1
    export GPU_ARCHS=gfx950
}

echo "======================================================================"
echo "FP4 Validation — Llama3.1-8B"
echo "  Steps: $STEPS   Repeats: $REPEATS   Configs: $NUM_CFGS"
echo "  Logs:  $LOGDIR/"
echo "======================================================================"

# ── Run loop ──────────────────────────────────────────────────────────
declare -a ALL_TAGS=()
declare -a ALL_RC=()

for ((c=0; c<NUM_CFGS; c++)); do
    NAME="${CFG_NAME[$c]}"
    USE_AITER="${CFG_AITER[$c]}"
    QUANT="${CFG_QUANT[$c]}"

    for ((r=1; r<=REPEATS; r++)); do
        TAG="${NAME}_r${r}"
        LOGFILE="${LOGDIR}/${TAG}.log"
        ALL_TAGS+=("$TAG")

        echo ""
        echo "── ${TAG} (config=$NAME, repeat=$r/$REPEATS) ──"
        echo "   aiter=$USE_AITER  quant=${QUANT:-none}"

        clean_aiter_env
        if [[ "$USE_AITER" == "True" ]]; then
            set_aiter_env
        fi
        export XLA_FLAGS="${XLA_BASE}"

        EXTRA_ARGS="steps=$STEPS use_jax_aiter=$USE_AITER run_name=val_${TAG}"
        if [[ -n "$QUANT" ]]; then
            EXTRA_ARGS+=" quantization=$QUANT"
        fi
        if [[ "$USE_AITER" == "True" ]]; then
            EXTRA_ARGS+=" aiter_attention=False"
        fi

        set +e
        python3 -m maxtext.trainers.pre_train.train $MODEL \
            $EXTRA_ARGS \
            2>&1 | tee "$LOGFILE" | grep -E 'completed step' | tail -3
        RC=$?
        set -e

        ALL_RC+=($RC)
        echo "  Exit: $RC"
    done
done

# ── Summary table ─────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "FP4 VALIDATION RESULTS — 8B, ${STEPS} steps, ${REPEATS} repeats"
echo "======================================================================"
printf "  %-25s  %-10s  %-14s  %-10s  %-10s  %-10s  %s\n" \
    "Config" "TFLOP/s" "Tokens/s/GPU" "Step(s)" "Exit" "1st NaN" "Loss"
printf "  %-25s  %-10s  %-14s  %-10s  %-10s  %-10s  %s\n" \
    "------" "-------" "------------" "-------" "----" "-------" "----"

for ((i=0; i<${#ALL_TAGS[@]}; i++)); do
    TAG="${ALL_TAGS[$i]}"
    LOGFILE="${LOGDIR}/${TAG}.log"

    if [[ ! -f "$LOGFILE" ]]; then
        printf "  %-25s  (no log)\n" "$TAG"
        continue
    fi

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
    first_nan=$(grep 'completed step' "$LOGFILE" 2>/dev/null | \
        grep -oP 'step: \K\d+.*loss: (nan|inf)' | head -1 | \
        grep -oP '^\d+' || echo "none")

    printf "  %-25s  %-10s  %-14s  %-10s  %-10s  %-10s  %s\n" \
        "$TAG" "$tflops" "$toks" "${steptime}s" "${ALL_RC[$i]}" "$first_nan" "$loss_last"
done

echo ""
echo "  Reference (prior measurements):"
echo "    BF16 baseline:  ~1,060 TFLOP/s  (~20,594 tok/s)"
echo "    AITER BF16:     ~1,074 TFLOP/s  (~20,849 tok/s, +1.3%)"
echo "    AITER MXFP4:    ~1,165 TFLOP/s  (~22,620 tok/s, +9.9%)  [10-step smoke]"
echo "    Native FP8:     ~1,409 TFLOP/s  (~27,378 tok/s, +32%)   [session 3]"
echo "======================================================================"
