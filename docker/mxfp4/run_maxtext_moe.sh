#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# MaxText MoE training with the BF16 baseline or JAX-AITER's MXFP4 routed experts, for
# DeepSeek-V3 671B, Qwen3.5-397B-A17B or GPT-OSS-120B: expert parallelism within a node, FSDP
# across nodes, and Primus-Turbo DeepEP dropless dispatch. The defaults are the settings of the
# ROCm blog "Multi-Node MXFP4 MoE Training in JAX on ROCm with JAX-AITER".
#
# Run once on every node, as root, inside the image built from Dockerfile, Dockerfile.te and
# Dockerfile.deepep (README.md):
#
#   MODEL=deepseek PRECISION=mxfp4 NNODES=8 NODE_RANK=<0..7> COORDINATOR=<node 0 address> \
#   OUTPUT_DIR=/work/out/deepseek_mxfp4_b4 bash /opt/jax-aiter/docker/mxfp4/run_maxtext_moe.sh
#
# Required:
#   MODEL          deepseek | qwen | gptoss
#   PRECISION      bf16 (Transformer Engine grouped GEMM) | mxfp4 (JAX-AITER MXFP4 experts)
#   NNODES         node count (8 in the blog); NODE_RANK is this node's rank, 0 on the coordinator
#   COORDINATOR    address of node 0, reachable from every node on COORDINATOR_PORT (default 1234)
#   OUTPUT_DIR     new directory for logs, metrics and the compilation cache; per node
# Optional:
#   DATASET        synthetic (default) or c4. c4 also needs C4_FILES, a glob of allenai/c4 en
#                  train shards (the blog used c4-train.00000-of-01024 to 00063-of-01024), and
#                  TOKENIZER, a Hugging Face tokenizer directory: DeepSeek-V3-Base,
#                  Qwen3.5-397B-A17B or gpt-oss-20b (the GPT-OSS-120B tokenizer)
#   BATCH          per-GPU batch, default 4. STEPS, default 30 (1000 for the convergence runs)
#   LAYERS         default: the full model (61, 60, 36)
#   PAD_V_HEAD     deepseek with DATASET=c4 and ATTENTION=cudnn_flash_te: true (default) pads the
#                  128-wide value heads to 192 and uses 32-bit query-gradient atomics; false uses
#                  neither, as in the blog's comparison without padding
#   GPUS_PER_NODE  default 8
#   MAXTEXT_DIR    default /opt/maxtext
# Further MXFP4 changes, which the BF16 baseline does not use:
#   GDN_KERNEL     qwen: jax (default, MaxText's) or jax_aiter_fla (FLA Gated DeltaNet)
#   SAVES          full (default, full rematerialization) or selected (keep the gathered weights)
#   ATTENTION      cudnn_flash_te (default, Transformer Engine) or jax_aiter_flash (AITER)
#   LAUNCH         per_gpu (default) or per_node (one process drives all GPUs of a node)
#   EXTRA_ARGS     more MaxText key=value settings, appended last so they override the above,
#                  e.g. balance_packed_rows_by_attention_work=true, which needs LAUNCH=per_node
#   DRY_RUN=1      write the command and environment to OUTPUT_DIR/command-node<NODE_RANK>.txt
#                  and exit without training
# The cluster network is not set here: for multi-node runs export NCCL_NET=IB, NCCL_IB_HCA,
# NCCL_IB_GID_INDEX and NCCL_SOCKET_IFNAME for your RoCE/InfiniBand fabric. The blog runs used
# one NIC per GPU (NCCL_IB_HCA=rdma0,...,rdma7, NCCL_IB_GID_INDEX=1).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${MODEL:?set MODEL to deepseek, qwen or gptoss}"
: "${PRECISION:?set PRECISION to bf16 or mxfp4}"
: "${NNODES:?set NNODES}"
: "${NODE_RANK:?set NODE_RANK}"
: "${COORDINATOR:?set COORDINATOR to the address of node 0}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
COORDINATOR_PORT="${COORDINATOR_PORT:-1234}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MAXTEXT_DIR="${MAXTEXT_DIR:-/opt/maxtext}"
DATASET="${DATASET:-synthetic}"
BATCH="${BATCH:-4}"
STEPS="${STEPS:-30}"
SAVES="${SAVES:-full}"
ATTENTION="${ATTENTION:-cudnn_flash_te}"
LAUNCH="${LAUNCH:-per_gpu}"
PAD_V_HEAD="${PAD_V_HEAD:-true}"
[[ "${PAD_V_HEAD}" == true || "${PAD_V_HEAD}" == false ]] ||
  { echo "PAD_V_HEAD must be true or false; got ${PAD_V_HEAD}" >&2; exit 2; }

MODEL_ARGS=()
NVTE_CK_IS_V3_ATOMIC_FP32=0
case "${MODEL}" in
  deepseek)
    MODEL_NAME=deepseek3-671b LAYERS="${LAYERS:-61}" ADD_BOS=true
    if [[ "${DATASET}" == c4 && "${ATTENTION}" == cudnn_flash_te && "${PAD_V_HEAD}" == true ]]; then
      # Packed MLA rows reach AITER's 192/192 group-mode backward only with V padded to 192.
      MODEL_ARGS+=(cudnn_te_pad_v_head_dim=true)
      NVTE_CK_IS_V3_ATOMIC_FP32=1
    fi
    ;;
  qwen) MODEL_NAME=qwen3.5-397b-a17b LAYERS="${LAYERS:-60}" ADD_BOS=false ;;
  gptoss) MODEL_NAME=gpt-oss-120b LAYERS="${LAYERS:-36}" ADD_BOS=true ;;
  *) echo "MODEL must be deepseek, qwen or gptoss; got ${MODEL}" >&2; exit 2 ;;
esac
if [[ -n "${GDN_KERNEL:-}" ]]; then
  [[ "${MODEL}" == qwen ]] || { echo "GDN_KERNEL applies to MODEL=qwen only" >&2; exit 2; }
  MODEL_ARGS+=("gdn_kernel=${GDN_KERNEL}")
fi

case "${PRECISION}" in
  bf16)
    EXPERT_ARGS=(moe_local_expert_backend=default use_te_grouped_gemm=true)
    if [[ "${MODEL}" == deepseek ]]; then
      # hipBLASLt can pick a Stream-K GEMM whose workspace exceeds TE's 64 MiB for DeepSeek-V3's
      # expert down-projection gradient; this raises it without changing TE.
      export MXFP4_PROJ_TE_GEMM_WORKSPACE_MIB=128
      test -r "${HERE}/te_workspace_override/usercustomize.py"
      export PYTHONPATH="${HERE}/te_workspace_override${PYTHONPATH:+:${PYTHONPATH}}"
    fi
    ;;
  mxfp4)
    EXPERT_ARGS=(moe_local_expert_backend=jax_aiter_mxfp4 jax_aiter_mxfp4_gathered_quant=true
      jax_aiter_mxfp4_dual_grad_cast=true jax_aiter_mxfp4_fsdp_weight_gather=true)
    ;;
  *) echo "PRECISION must be bf16 or mxfp4; got ${PRECISION}" >&2; exit 2 ;;
esac

case "${SAVES}" in
  full) REMAT_ARGS=(remat_policy=full) ;;
  selected)
    REMAT_ARGS=(remat_policy=custom decoder_layer_input=remat dense_retain_gathered_weights=true
      gathered_weight=device moe_gathered_weight=device moe_gathered_wo_bwd=device)
    ;;
  *) echo "SAVES must be full or selected; got ${SAVES}" >&2; exit 2 ;;
esac

case "${ATTENTION}" in
  cudnn_flash_te) export JA_MHA_BWD_ATOMIC_FP32=0 ;;
  jax_aiter_flash) export JA_MHA_BWD_ATOMIC_FP32=1 ;;
  *) echo "ATTENTION must be cudnn_flash_te or jax_aiter_flash; got ${ATTENTION}" >&2; exit 2 ;;
esac

case "${LAUNCH}" in
  per_gpu) ;;
  per_node)
    # TE's grouped GEMM syncs group sizes to the host inside its FFI, which deadlocks when one
    # process drives several GPUs.
    [[ "${PRECISION}" == mxfp4 ]] || { echo "LAUNCH=per_node needs PRECISION=mxfp4" >&2; exit 2; }
    ;;
  *) echo "LAUNCH must be per_gpu or per_node; got ${LAUNCH}" >&2; exit 2 ;;
esac

case "${DATASET}" in
  synthetic) DATA_ARGS=(dataset_type=synthetic hf_path=allenai/c4 hf_data_dir=en) ;;
  c4)
    : "${C4_FILES:?set C4_FILES to a glob of C4 train shards}"
    : "${TOKENIZER:?set TOKENIZER to a Hugging Face tokenizer directory}"
    compgen -G "${C4_FILES}" >/dev/null || { echo "no file matches C4_FILES=${C4_FILES}" >&2; exit 2; }
    test -r "${TOKENIZER}/tokenizer.json"
    DATA_ARGS=(dataset_type=hf hf_path=json "hf_train_files=${C4_FILES}" tokenizer_type=huggingface
      "tokenizer_path=${TOKENIZER}" enable_data_shuffling=false "add_bos=${ADD_BOS}" add_eos=true)
    ;;
  *) echo "DATASET must be synthetic or c4; got ${DATASET}" >&2; exit 2 ;;
esac

read -r -a USER_ARGS <<< "${EXTRA_ARGS:-}"

first_rank=$((NODE_RANK * GPUS_PER_NODE))
[[ "${LAUNCH}" == per_node ]] && first_rank="${NODE_RANK}"
if [[ -e "${OUTPUT_DIR}/rank-${first_rank}/train.log" ]]; then
  echo "${OUTPUT_DIR} already holds a run; choose a new OUTPUT_DIR" >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}/jax_cache"

export PYTHONPATH="${MAXTEXT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_FLAGS="--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false \
--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false \
--xla_gpu_disable_async_collectives=RAGGEDALLTOALL,ALLTOALL \
--xla_gpu_allow_ragged_all_to_all_nccl_send_recv_fallback=true \
--xla_gpu_enable_command_buffer= --xla_gpu_command_buffer_update_mode=ALWAYS_UPDATE \
--xla_gpu_all_gather_combine_threshold_bytes=8589934592 \
--xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592 \
--xla_gpu_enable_cub_radix_sort=false --xla_gpu_enable_latency_hiding_scheduler=true \
--xla_gpu_enable_pipelined_all_gather=true --xla_gpu_experimental_parallel_collective_overlap_limit=2 \
--xla_gpu_enable_nccl_comm_splitting=false --xla_gpu_force_compilation_parallelism=8 \
--xla_enable_hlo_sharding_v3=false"
export JAX_PLATFORMS=rocm,cpu
export JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=xla_gpu_per_fusion_autotune_cache_dir
export JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS=false JAX_SHARE_BINARY_BETWEEN_HOSTS=false
export XLA_PYTHON_CLIENT_ALLOCATOR=bfc XLA_PYTHON_CLIENT_MEM_FRACTION=.90 XLA_PYTHON_CLIENT_PREALLOCATE=true
export HSA_NO_SCRATCH_RECLAIM=1 HSA_FORCE_FINE_GRAIN_PCIE=1 HIP_FORCE_DEV_KERNARG=1 GPU_MAX_HW_QUEUES=2
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}" NCCL_LAUNCH_ORDER_IMPLICIT=0 NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_DISABLE=0 NCCL_IB_MERGE_NICS=0 NCCL_CROSS_NIC=0
export RCCL_DIRECT_REDUCE_SCATTER_DISABLE=1
export NVTE_FRAMEWORK=jax NVTE_ALLOW_NONDETERMINISTIC_ALGO=1 NVTE_FUSED_ATTN=1 NVTE_FUSED_ATTN_CK=1
export NVTE_FUSED_ATTN_AOTRITON=0 NVTE_CK_USES_BWD_V3=1 NVTE_CK_USES_FWD_V3=1
export NVTE_CK_IS_V3_ATOMIC_FP32 NVTE_CK_HOW_V3_BF16_CVT=2
export JA_MHA_FWD_USE_ASM_V3=1 AITER_DISABLE_FMHA_OPUS=0

python3 - <<'PY'
import importlib.metadata as m
v = {d: m.version(d) for d in ("jax", "jaxlib", "jax-rocm10-plugin", "jax-rocm10-pjrt")}
print("jax versions:", v, flush=True)
raise SystemExit(0 if set(v.values()) == {"0.11.2"} else "this recipe needs JAX 0.11.2")
PY
python3 -c 'import maxtext, sys; sys.exit(0 if maxtext.__file__.startswith(sys.argv[1]) else "MAXTEXT_DIR is not the MaxText on the path")' "${MAXTEXT_DIR}/src/"

TRAIN=(
  python3 -u -m maxtext.trainers.pre_train.train "${MAXTEXT_DIR}/src/maxtext/configs/base.yml"
  "run_name=$(basename "${OUTPUT_DIR}")" "base_output_directory=${OUTPUT_DIR}/maxtext"
  "jax_cache_dir=${OUTPUT_DIR}/jax_cache" jax_distributed_initialization_timeout=1800 hardware=gpu
  enable_checkpointing=false save_checkpoint_on_completion=false async_checkpointing=false
  log_period=1 log_config=false "model_name=${MODEL_NAME}" "base_num_decoder_layers=${LAYERS}"
  "steps=${STEPS}" "per_device_batch_size=${BATCH}" "${REMAT_ARGS[@]}"
  override_model_config=true use_iota_embed=true "${DATA_ARGS[@]}"
  packing=true max_segments_per_seq=32 optimizer_memory_host_offload=false scan_layers=true
  param_scan_axis=1 logits_dot_in_fp32=false dtype=bfloat16 weight_dtype=bfloat16
  max_target_length=4096 learning_rate_schedule_steps=2000
  ici_expert_parallelism="${GPUS_PER_NODE}" ici_fsdp_parallelism=1 ici_data_parallelism=1
  dcn_fsdp_parallelism="${NNODES}" dcn_data_parallelism=1
  "attention=${ATTENTION}" shardy=true sparse_matmul=true megablox=false capacity_factor=-1
  "${EXPERT_ARGS[@]}"
  jax_aiter_mxfp4_dedup_dispatch=true jax_aiter_mxfp4_dedup_deepep=true
  jax_aiter_mxfp4_dropless_tiers=true ragged_buffer_factor=2.0 jax_aiter_mxfp4_overflow_buffer_factor=1.0
  jax_aiter_mxfp4_dedup_recompute_tier_switch=true jax_aiter_mxfp4_fresh_a2a_buffers=true
  jax_aiter_mxfp4_dedup_skip_unread_zeroing=true jax_aiter_mxfp4_check_route_capacity=false
  "${MODEL_ARGS[@]}" "${USER_ARGS[@]}"
)
{
  date -u
  echo "host=$(hostname) node=${NODE_RANK}/${NNODES} model=${MODEL} precision=${PRECISION}" \
    "dataset=${DATASET} batch=${BATCH} steps=${STEPS} launch=${LAUNCH}"
  printf 'command='; printf '%q ' "${TRAIN[@]}"; printf '\n'
  env | sort | grep -E '^(XLA_|NCCL_|RCCL_|MSCCLPP_|JAX_|MAXTEXT_|JA_|AITER_|HSA_|HIP_|DEBUG_HIP_|GPU_|CUDA_DEVICE_|NVTE_|TF_CPP_|MXFP4_|LD_LIBRARY_PATH|PYTHONPATH)'
} | tee "${OUTPUT_DIR}/command-node${NODE_RANK}.txt"
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  exit 0
fi

cd "${MAXTEXT_DIR}"
if [[ "${LAUNCH}" == per_node ]]; then
  distributed=(skip_jax_distributed_system=true)
  if ((NNODES > 1)); then
    distributed=(skip_jax_distributed_system=false)
    export JAX_COORDINATOR_IP="${COORDINATOR}" JAX_COORDINATOR_PORT="${COORDINATOR_PORT}"
    export NNODES NODE_RANK
  fi
  mkdir -p "${OUTPUT_DIR}/rank-${NODE_RANK}"
  status=0
  PYTHONUNBUFFERED=1 "${TRAIN[@]}" "${distributed[@]}" \
    "metrics_file=${OUTPUT_DIR}/rank-${NODE_RANK}/metrics.txt" 2>&1 \
    | tee "${OUTPUT_DIR}/rank-${NODE_RANK}/train.log" || status=$?
  echo "node ${NODE_RANK} exit ${status}; log in ${OUTPUT_DIR}/rank-${NODE_RANK}/train.log"
  exit "${status}"
fi

# One process per GPU. MaxText maps SLURM_STEP_GPUS to the process's local device;
# CUDA_VISIBLE_DEVICES would hide the other GPUs on ROCm.
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
CPUS_PER_RANK=$(($(nproc) / GPUS_PER_NODE))
pids=()
for ((local_rank = 0; local_rank < GPUS_PER_NODE; local_rank++)); do
  rank=$((NODE_RANK * GPUS_PER_NODE + local_rank))
  hsaco_cache="${TF_XLA_HSACO_CACHE_DIR:-${OUTPUT_DIR}/hsaco}/g${rank}"
  mkdir -p "${OUTPUT_DIR}/rank-${rank}" "${hsaco_cache}"
  first_cpu=$((local_rank * CPUS_PER_RANK))
  (
    unset CUDA_VISIBLE_DEVICES
    JAX_COORDINATOR_IP="${COORDINATOR}" JAX_COORDINATOR_PORT="${COORDINATOR_PORT}" \
    NNODES="${WORLD_SIZE}" NODE_RANK="${rank}" SLURM_STEP_GPUS="${local_rank}" PYTHONUNBUFFERED=1 \
    TF_XLA_HSACO_CACHE_DIR="${hsaco_cache}" \
      taskset -c "${first_cpu}-$((first_cpu + CPUS_PER_RANK - 1))" \
      "${TRAIN[@]}" skip_jax_distributed_system=false \
      "metrics_file=${OUTPUT_DIR}/rank-${rank}/metrics.txt" 2>&1 \
      | tee "${OUTPUT_DIR}/rank-${rank}/train.log" | sed -u "s/^/[g${rank}] /"
  ) &
  pids+=("$!")
done

status=0
for ((local_rank = 0; local_rank < GPUS_PER_NODE; local_rank++)); do
  if ! wait "${pids[local_rank]}"; then
    echo "global rank $((NODE_RANK * GPUS_PER_NODE + local_rank)) failed" >&2
    status=1
  fi
done
echo "node ${NODE_RANK} exit ${status}; logs in ${OUTPUT_DIR}/rank-*/train.log"
exit "${status}"
