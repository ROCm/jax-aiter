#!/bin/bash
# All-FP4 training validation — Llama3.3-70B.
# Compares:
#   1. hybrid_prod  — current production (FP4 fwd + FP4 dA + FP8 dB)
#   2. all_fp4      — new TE-parity recipe (FP4 fwd + FP4 dA + FP4 dB via wgrad sharding)
#   3. fp8_baseline — native Flax FP8 reference
#
# Reports tail-20 avg to capture steady-state perf.
#
# Usage (inside container):
#   bash tests/validate_all_fp4_70b.sh          # 40 steps, pack mode (tail-20 avg)
#   bash tests/validate_all_fp4_70b.sh 20       # 20 steps
set -uo pipefail

STEPS="${1:-40}"

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

LOGDIR=/ruvaidya/aiter_proj/docs/logs/all_fp4_70b
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

# Config order: FP8 first so its measurement is thermally clean. Repeated
# MaxText training in one bash session accumulates GPU thermal load that
# disproportionately slows the FP8 path at smaller MLP shapes. See
# docs/logs/session20_e2e/SUMMARY.md for evidence. 70B is less sensitive
# than 8B (larger per-step work, smaller relative dispatch overhead) but
# applying the same ordering for consistency.
declare -a CFG_NAME=( "fp8_baseline" "hybrid_prod"  "all_fp4"      )
declare -a CFG_AITER=( "False"        "True"         "True"         )
declare -a CFG_QUANT=( "fp8"          "aiter_mxfp4"  "aiter_mxfp4"  )
declare -a CFG_ENV=(
  ""
  "AITER_FP4_DA=1;AITER_FP4_DB=0;AITER_FP8_DB=1;AITER_FUSED_QUANT=1;AITER_FP4_ATTN=1;AITER_ALL_FP4=0"
  "AITER_FP4_DA=1;AITER_FUSED_QUANT=1;AITER_FP4_ATTN=1;AITER_ALL_FP4=1"
)
NUM_CFGS=${#CFG_NAME[@]}

clean_aiter_env() {
    unset JA_ROOT_DIR AITER_ASM_DIR AITER_SYMBOL_VISIBLE GPU_ARCHS 2>/dev/null || true
    unset AITER_FP4_DA AITER_FP4_DB AITER_FP8_DB AITER_FUSED_QUANT \
          AITER_FP4_ATTN AITER_ALL_FP4 AITER_COMPOSITE_FP4 \
          AITER_LAZY_WEIGHT_COL AITER_FP4_RESIDUALS 2>/dev/null || true
}

set_aiter_env() {
    export JA_ROOT_DIR=/ruvaidya/aiter_proj/jax-aiter
    export AITER_ASM_DIR=/ruvaidya/aiter_proj/jax-aiter/third_party/aiter/hsa/
    export AITER_SYMBOL_VISIBLE=1
    export GPU_ARCHS=gfx950
}

apply_cfg_env() {
    local env_str="$1"
    [[ -z "$env_str" ]] && return
    IFS=';' read -ra kvs <<< "$env_str"
    for kv in "${kvs[@]}"; do
        export "$kv"
    done
}

echo "======================================================================"
echo "All-FP4 Validation — Llama3.3-70B, ${STEPS} steps, 8x MI355X"
echo "======================================================================"

declare -A ALL_RC=()

for ((c=0; c<NUM_CFGS; c++)); do
    NAME="${CFG_NAME[$c]}"
    USE_AITER="${CFG_AITER[$c]}"
    QUANT="${CFG_QUANT[$c]}"
    EXTRA_ENV="${CFG_ENV[$c]}"

    TAG="${NAME}"
    LOGFILE="${LOGDIR}/${TAG}.log"

    echo ""
    echo "── ${TAG} ──"
    echo "   aiter=$USE_AITER quant=${QUANT:-none} env=[${EXTRA_ENV}]"

    clean_aiter_env
    if [[ "$USE_AITER" == "True" ]]; then
        set_aiter_env
    fi
    apply_cfg_env "$EXTRA_ENV"
    export XLA_FLAGS="${XLA_BASE}"

    EXTRA_ARGS="steps=$STEPS use_jax_aiter=$USE_AITER run_name=allfp4_70b_${TAG}"
    if [[ -n "$QUANT" ]]; then
        EXTRA_ARGS+=" quantization=$QUANT"
    fi
    if [[ "$USE_AITER" == "True" ]]; then
        EXTRA_ARGS+=" aiter_attention=False"
    fi

    set +e
    python3 -m maxtext.trainers.pre_train.train $MODEL_70B \
        $EXTRA_ARGS \
        2>&1 | tee "$LOGFILE" | grep -E 'completed step' | tail -5
    RC=$?
    set -e

    ALL_RC["$TAG"]=$RC
    echo "  Exit: $RC"
done

echo ""
echo "======================================================================"
echo "ALL-FP4 70B RESULTS — ${STEPS} steps (tail-20 avg for steady state)"
echo "======================================================================"
printf "  %-30s  %-10s  %-14s  %-10s  %-6s  %-8s  %s\n" \
    "Config" "TFLOP/s" "Tokens/s/GPU" "Step(s)" "Exit" "1st NaN" "Loss"
printf "  %-30s  %-10s  %-14s  %-10s  %-6s  %-8s  %s\n" \
    "------" "-------" "------------" "-------" "----" "-------" "----"

for ((c=0; c<NUM_CFGS; c++)); do
    NAME="${CFG_NAME[$c]}"
    LOGFILE="${LOGDIR}/${NAME}.log"
    RC="${ALL_RC[$NAME]:-?}"

    if [[ ! -f "$LOGFILE" ]]; then
        printf "  %-30s  (no log)\n" "$NAME"
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

    printf "  %-30s  %-10s  %-14s  %-10s  %-6s  %-8s  %s\n" \
        "$NAME" "$tflops" "$toks" "${steptime}s" "$RC" "$first_nan" "$loss_last"
done

# Per-config deltas vs FP8 baseline.
fp8_t=$(grep 'completed step' "$LOGDIR/fp8_baseline.log" 2>/dev/null | tail -20 | grep -oP 'TFLOP/s/device: \K[0-9.]+' | awk '{s+=$1;n++} END{if(n>0) printf "%.1f",s/n; else print "0"}')
hyb_t=$(grep 'completed step' "$LOGDIR/hybrid_prod.log" 2>/dev/null | tail -20 | grep -oP 'TFLOP/s/device: \K[0-9.]+' | awk '{s+=$1;n++} END{if(n>0) printf "%.1f",s/n; else print "0"}')
all_t=$(grep 'completed step' "$LOGDIR/all_fp4.log" 2>/dev/null | tail -20 | grep -oP 'TFLOP/s/device: \K[0-9.]+' | awk '{s+=$1;n++} END{if(n>0) printf "%.1f",s/n; else print "0"}')

echo ""
if [[ "$hyb_t" != "0" && "$fp8_t" != "0" ]]; then
    g=$(awk "BEGIN {printf \"%+.1f\", ($hyb_t/$fp8_t - 1)*100}")
    echo "  Hybrid vs FP8:   ${hyb_t} / ${fp8_t} = ${g}%"
fi
if [[ "$all_t" != "0" && "$fp8_t" != "0" ]]; then
    g=$(awk "BEGIN {printf \"%+.1f\", ($all_t/$fp8_t - 1)*100}")
    echo "  All-FP4 vs FP8:  ${all_t} / ${fp8_t} = ${g}%"
fi
if [[ "$all_t" != "0" && "$hyb_t" != "0" ]]; then
    g=$(awk "BEGIN {printf \"%+.1f\", ($all_t/$hyb_t - 1)*100}")
    echo "  All-FP4 vs Hybrid: ${all_t} / ${hyb_t} = ${g}%"
fi

echo ""
echo "  Reference (session 18 production, 2026-04-11):"
echo "    Hybrid (FP4 attn + FP8 dB):  1,624 TFLOP/s  (+1.4% vs FP8)"
echo "    FP8 baseline:                 1,602 TFLOP/s"
echo ""
echo "  Promotion gate: all_fp4 must tie or beat hybrid_prod within 1% noise."
echo "======================================================================"
