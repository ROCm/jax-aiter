#!/bin/bash
# FP4 Validation — Llama3.3-70B, 3 configs x 2 packing modes x 20 steps
#
# Configs:
#   1. bf16_baseline   — no AITER, no quantization
#   2. aiter_bf16      — AITER BF16 ASM GEMM
#   3. aiter_mxfp4     — AITER FP4 MLP forward + BF16 backward
#
# Pass 1: packing=False (apples-to-apples with 8B validation)
# Pass 2: packing=True  (MAD-aligned production config)
#
# Usage (inside container):
#   bash tests/validate_fp4_70b.sh           # 20 steps, both packing modes
#   bash tests/validate_fp4_70b.sh 20        # custom steps, both packing modes
#   bash tests/validate_fp4_70b.sh 20 nopack # packing=False only
#   bash tests/validate_fp4_70b.sh 20 pack   # packing=True only
set -uo pipefail

STEPS="${1:-20}"
PACKING_MODE="${2:-both}"

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

LOGDIR=/ruvaidya/aiter_proj/docs/logs/fp4_validation_70b
mkdir -p "$LOGDIR"

MODEL_BASE="src/maxtext/configs/base.yml \
  hardware=gpu model_name=llama3.3-70b attention=cudnn_flash_te \
  enable_checkpointing=False \
  ici_fsdp_parallelism=8 ici_data_parallelism=1 ici_expert_parallelism=1 \
  remat_policy=full scan_layers=True param_scan_axis=1 dataset_type=synthetic \
  logits_dot_in_fp32=False dtype=bfloat16 weight_dtype=bfloat16 \
  per_device_batch_size=10 max_target_length=8192 \
  use_iota_embed=True shardy=False \
  base_output_directory=/tmp/maxtext_output"

# ── Config definitions ────────────────────────────────────────────────
declare -a CFG_NAME=( "bf16_baseline" "aiter_bf16" "aiter_mxfp4" )
declare -a CFG_AITER=( "False"         "True"       "True"        )
declare -a CFG_QUANT=( ""              ""           "aiter_mxfp4" )
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

# ── Packing passes ────────────────────────────────────────────────────
declare -a PASS_NAME=()
declare -a PASS_PACKING=()
declare -a PASS_EXTRA=()

case "$PACKING_MODE" in
    nopack)
        PASS_NAME+=("nopack")
        PASS_PACKING+=("False")
        PASS_EXTRA+=("")
        ;;
    pack)
        PASS_NAME+=("pack")
        PASS_PACKING+=("True")
        PASS_EXTRA+=("max_segments_per_seq=32")
        ;;
    both)
        PASS_NAME+=("nopack" "pack")
        PASS_PACKING+=("False" "True")
        PASS_EXTRA+=("" "max_segments_per_seq=32")
        ;;
    *)
        echo "Usage: $0 [steps] [nopack|pack|both]"
        exit 1
        ;;
esac
NUM_PASSES=${#PASS_NAME[@]}

echo "======================================================================"
echo "FP4 Validation — Llama3.3-70B"
echo "  Steps: $STEPS   Configs: $NUM_CFGS   Packing passes: $NUM_PASSES"
echo "  Logs:  $LOGDIR/"
echo "======================================================================"

# ── Summary extraction function ───────────────────────────────────────
extract_metrics() {
    local logfile="$1"
    local tail_n="${2:-10}"
    if [[ ! -f "$logfile" ]]; then
        echo "N/A N/A N/A N/A N/A"
        return
    fi
    local loss tflops toks steptime first_nan
    loss=$(grep 'completed step' "$logfile" 2>/dev/null | tail -1 | \
        grep -oP 'loss: \K[0-9.nanif]+' || echo "N/A")
    tflops=$(grep 'completed step' "$logfile" 2>/dev/null | tail -"$tail_n" | \
        grep -oP 'TFLOP/s/device: \K[0-9.]+' | \
        awk '{sum+=$1; n++} END {if(n>0) printf "%.1f", sum/n; else print "N/A"}')
    toks=$(grep 'completed step' "$logfile" 2>/dev/null | tail -"$tail_n" | \
        grep -oP 'Tokens/s/device: \K[0-9.]+' | \
        awk '{sum+=$1; n++} END {if(n>0) printf "%.1f", sum/n; else print "N/A"}')
    steptime=$(grep 'completed step' "$logfile" 2>/dev/null | tail -"$tail_n" | \
        grep -oP 'seconds: \K[0-9.]+' | \
        awk '{sum+=$1; n++} END {if(n>0) printf "%.3f", sum/n; else print "N/A"}')
    first_nan=$(grep 'completed step' "$logfile" 2>/dev/null | \
        grep -oP 'step: \K\d+.*loss: (nan|inf)' | head -1 | \
        grep -oP '^\d+' || echo "none")
    echo "$tflops $toks $steptime $first_nan $loss"
}

# ── Run loop ──────────────────────────────────────────────────────────
declare -A ALL_RC=()

for ((p=0; p<NUM_PASSES; p++)); do
    PNAME="${PASS_NAME[$p]}"
    PPACKING="${PASS_PACKING[$p]}"
    PEXTRA="${PASS_EXTRA[$p]}"

    echo ""
    echo "======================================================================"
    echo "  PASS: $PNAME (packing=$PPACKING)"
    echo "======================================================================"

    for ((c=0; c<NUM_CFGS; c++)); do
        NAME="${CFG_NAME[$c]}"
        USE_AITER="${CFG_AITER[$c]}"
        QUANT="${CFG_QUANT[$c]}"

        TAG="${PNAME}_${NAME}"
        LOGFILE="${LOGDIR}/${TAG}.log"

        echo ""
        echo "── ${TAG} (packing=$PPACKING, aiter=$USE_AITER, quant=${QUANT:-none}) ──"

        clean_aiter_env
        if [[ "$USE_AITER" == "True" ]]; then
            set_aiter_env
        fi
        export XLA_FLAGS="${XLA_BASE}"

        EXTRA_ARGS="steps=$STEPS use_jax_aiter=$USE_AITER packing=$PPACKING run_name=fp4_70b_${TAG}"
        if [[ -n "$QUANT" ]]; then
            EXTRA_ARGS+=" quantization=$QUANT"
        fi
        if [[ "$USE_AITER" == "True" ]]; then
            EXTRA_ARGS+=" aiter_attention=False"
        fi
        if [[ -n "$PEXTRA" ]]; then
            EXTRA_ARGS+=" $PEXTRA"
        fi

        set +e
        python3 -m maxtext.trainers.pre_train.train $MODEL_BASE \
            $EXTRA_ARGS \
            2>&1 | tee "$LOGFILE" | grep -E 'completed step' | tail -5
        RC=$?
        set -e

        ALL_RC["$TAG"]=$RC
        echo "  Exit: $RC"
    done
done

# ── Summary table per pass ────────────────────────────────────────────
for ((p=0; p<NUM_PASSES; p++)); do
    PNAME="${PASS_NAME[$p]}"
    PPACKING="${PASS_PACKING[$p]}"

    echo ""
    echo "======================================================================"
    echo "FP4 VALIDATION — 70B, ${STEPS} steps, packing=${PPACKING}"
    echo "======================================================================"
    printf "  %-25s  %-10s  %-14s  %-10s  %-6s  %-8s  %s\n" \
        "Config" "TFLOP/s" "Tokens/s/GPU" "Step(s)" "Exit" "1st NaN" "Loss"
    printf "  %-25s  %-10s  %-14s  %-10s  %-6s  %-8s  %s\n" \
        "------" "-------" "------------" "-------" "----" "-------" "----"

    for ((c=0; c<NUM_CFGS; c++)); do
        NAME="${CFG_NAME[$c]}"
        TAG="${PNAME}_${NAME}"
        LOGFILE="${LOGDIR}/${TAG}.log"
        RC="${ALL_RC[$TAG]:-?}"

        read tflops toks steptime first_nan loss <<< "$(extract_metrics "$LOGFILE" 10)"

        printf "  %-25s  %-10s  %-14s  %-10s  %-6s  %-8s  %s\n" \
            "$NAME" "$tflops" "$toks" "${steptime}s" "$RC" "$first_nan" "$loss"
    done

    # Compute FP4 vs BF16 gain
    BF16_LOG="${LOGDIR}/${PNAME}_bf16_baseline.log"
    FP4_LOG="${LOGDIR}/${PNAME}_aiter_mxfp4.log"
    if [[ -f "$BF16_LOG" && -f "$FP4_LOG" ]]; then
        bf16_t=$(grep 'completed step' "$BF16_LOG" 2>/dev/null | tail -10 | \
            grep -oP 'TFLOP/s/device: \K[0-9.]+' | \
            awk '{sum+=$1; n++} END {if(n>0) printf "%.1f", sum/n; else print "0"}')
        fp4_t=$(grep 'completed step' "$FP4_LOG" 2>/dev/null | tail -10 | \
            grep -oP 'TFLOP/s/device: \K[0-9.]+' | \
            awk '{sum+=$1; n++} END {if(n>0) printf "%.1f", sum/n; else print "0"}')
        if [[ "$bf16_t" != "0" && "$fp4_t" != "0" ]]; then
            gain=$(awk "BEGIN {printf \"%.1f\", ($fp4_t/$bf16_t - 1)*100}")
            echo ""
            echo "  >>> FP4 vs BF16 (packing=$PPACKING): ${fp4_t} / ${bf16_t} = +${gain}%"
        fi
    fi
done

echo ""
echo "======================================================================"
echo "  Reference (8B validated, 100 steps, 3 repeats):"
echo "    BF16 baseline:  1,048 TFLOP/s  (20,361 tok/s)"
echo "    AITER BF16:     1,058 TFLOP/s  (20,550 tok/s, +0.9%)"
echo "    AITER MXFP4:    1,150 TFLOP/s  (22,350 tok/s, +9.8%)"
echo ""
echo "  Reference (70B prior, packing=False):"
echo "    BF16 baseline:  ~983 TFLOP/s   (~2,189 tok/s)"
echo "    AITER BF16:    ~1,024 TFLOP/s  (~2,280 tok/s, +4.2%)"
echo ""
echo "  Key question: does FP4 +9.8% (8B) grow at 70B?"
echo "    If BF16 AITER scales +0.9% -> +4.0%, FP4 may reach +15-20%"
echo "======================================================================"
