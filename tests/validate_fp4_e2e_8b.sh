#!/bin/bash
# FP4 training validation -- Llama3-8B.
# Compares:
#   1. fp8_baseline -- native Flax FP8 reference (hipBLASLt FP8 dot_general)
#   2. fp4          -- AITER FP4 (MXFP4) recipe (default for jax-aiter on MI350/MI355X)
#
# Usage (inside container):
#   bash tests/validate_fp4_e2e_8b.sh            # 100 steps, 1 repeat
#   bash tests/validate_fp4_e2e_8b.sh 20 1       # 20 steps, 1 repeat (quick)
#   bash tests/validate_fp4_e2e_8b.sh 100 3      # 100 steps, 3 repeats
set -uo pipefail

STEPS="${1:-100}"
REPEATS="${2:-1}"

cd /ruvaidya/aiter_proj/maxtext

# Common env (MAD-aligned).
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

LOGDIR=/ruvaidya/aiter_proj/docs/logs/fp4_8b
mkdir -p "$LOGDIR"

MODEL="src/maxtext/configs/base.yml \
  hardware=gpu model_name=llama3-8b attention=cudnn_flash_te \
  enable_checkpointing=False \
  ici_fsdp_parallelism=8 ici_data_parallelism=1 ici_expert_parallelism=1 \
  remat_policy=minimal_flash scan_layers=True dataset_type=synthetic \
  logits_dot_in_fp32=False dtype=bfloat16 weight_dtype=bfloat16 \
  per_device_batch_size=9 max_target_length=8192 shardy=False packing=False \
  base_output_directory=/tmp/maxtext_output"

# Order: FP8 first on cool GPUs (FP8 is unusually thermal/state-sensitive at
# 8B MLP shapes -- see docs/logs/session20_e2e/SUMMARY.md "FP8 baseline").
declare -a CFG_NAME=( "fp8_baseline" "fp4"      )
declare -a CFG_AITER=( "False"        "True"     )
declare -a CFG_QUANT=( "fp8"          "aiter_fp4" )

set_aiter_env() {
    export JA_ROOT_DIR=/ruvaidya/aiter_proj/jax-aiter
    export AITER_ASM_DIR=/ruvaidya/aiter_proj/jax-aiter/third_party/aiter/hsa/
    export AITER_SYMBOL_VISIBLE=1
    export GPU_ARCHS=gfx950
    # Apply FP4 to attention projections too (recommended on MI355X).
    export AITER_FP4_ATTN=1
}

clean_env() {
    unset JA_ROOT_DIR AITER_ASM_DIR AITER_SYMBOL_VISIBLE GPU_ARCHS \
          AITER_FP4_ATTN AITER_FUSED_QUANT_HADAMARD AITER_KERNEL_SEL 2>/dev/null || true
}

echo "======================================================================"
echo "FP4 Validation -- Llama3-8B"
echo "  Steps: $STEPS   Repeats: $REPEATS"
echo "  Logs:  $LOGDIR/"
echo "======================================================================"

declare -a ALL_TAGS=()
declare -a ALL_RC=()

for ((c=0; c<${#CFG_NAME[@]}; c++)); do
    NAME="${CFG_NAME[$c]}"
    USE_AITER="${CFG_AITER[$c]}"
    QUANT="${CFG_QUANT[$c]}"

    for ((r=1; r<=REPEATS; r++)); do
        TAG="${NAME}_r${r}"
        LOGFILE="${LOGDIR}/${TAG}.log"
        ALL_TAGS+=("$TAG")

        echo ""
        echo "-- ${TAG} (config=$NAME, repeat=$r/$REPEATS) --"
        echo "   aiter=$USE_AITER quant=${QUANT}"

        clean_env
        if [[ "$USE_AITER" == "True" ]]; then
            set_aiter_env
        fi
        export XLA_FLAGS="${XLA_BASE}"

        EXTRA_ARGS="steps=$STEPS use_jax_aiter=$USE_AITER run_name=fp4_8b_${TAG} quantization=$QUANT"
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

echo ""
echo "======================================================================"
echo "FP4 8B RESULTS -- ${STEPS} steps x ${REPEATS} repeats"
echo "======================================================================"
printf "  %-30s  %-10s  %-14s  %-10s  %-6s  %-8s  %s\n" \
    "Config" "TFLOP/s" "Tokens/s/GPU" "Step(s)" "Exit" "1st NaN" "Loss"
printf "  %-30s  %-10s  %-14s  %-10s  %-6s  %-8s  %s\n" \
    "------" "-------" "------------" "-------" "----" "-------" "----"

for ((i=0; i<${#ALL_TAGS[@]}; i++)); do
    TAG="${ALL_TAGS[$i]}"
    LOGFILE="${LOGDIR}/${TAG}.log"
    if [[ ! -f "$LOGFILE" ]]; then
        printf "  %-30s  (no log)\n" "$TAG"
        continue
    fi

    loss_last=$(grep 'completed step' "$LOGFILE" 2>/dev/null | tail -1 | \
        grep -oP 'loss: \K[0-9.nanif]+' || echo "N/A")
    # Tail-15 average (stable steady-state, skipping JIT compile/warmup).
    tflops=$(grep 'completed step' "$LOGFILE" 2>/dev/null | tail -15 | \
        grep -oP 'TFLOP/s/device: \K[0-9.]+' | \
        awk '{sum+=$1; n++} END {if(n>0) printf "%.1f", sum/n; else print "N/A"}')
    toks=$(grep 'completed step' "$LOGFILE" 2>/dev/null | tail -15 | \
        grep -oP 'Tokens/s/device: \K[0-9.]+' | \
        awk '{sum+=$1; n++} END {if(n>0) printf "%.1f", sum/n; else print "N/A"}')
    steptime=$(grep 'completed step' "$LOGFILE" 2>/dev/null | tail -15 | \
        grep -oP 'seconds: \K[0-9.]+' | \
        awk '{sum+=$1; n++} END {if(n>0) printf "%.3f", sum/n; else print "N/A"}')
    first_nan=$(grep 'completed step' "$LOGFILE" 2>/dev/null | \
        grep -oP 'step: \K\d+.*loss: (nan|inf)' | head -1 | \
        grep -oP '^\d+' || echo "none")

    printf "  %-30s  %-10s  %-14s  %-10s  %-6s  %-8s  %s\n" \
        "$TAG" "$tflops" "$toks" "${steptime}s" "${ALL_RC[$i]}" "$first_nan" "$loss_last"
done

echo ""
echo "  Reference (session 22, 2026-04-23, 8x MI355X, 6-step xplane profile):"
echo "    fp8_baseline:  ~1,419 TFLOP/s/dev (~27,570 Tokens/s/dev)"
echo "    fp4:           ~1,591 TFLOP/s/dev (~30,908 Tokens/s/dev), +12.1% vs FP8"
echo ""
echo "  Pass criteria: fp4 TFLOP/s/dev >= 1.10 * fp8 (i.e., >= 10% E2E gain)."
echo "======================================================================"
