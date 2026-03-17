# JAX-AITER: Next Task Context

## Where We Are

We integrated AITER's hand-tuned ASM GEMM kernels into MaxText Llama2-7B training via
`quantization=aiter_bf16` (and `aiter_fp8`). The final architecture uses AITER ASM for
the forward GEMM and `lax.dot_general` (hipBLASLt) for the backward GEMM.

**Current performance (Llama2-7B, batch=10, seq=4096, MI355X):**

| Config | 1-GPU TFLOP/s | 8-GPU TFLOP/s |
|--------|:---:|:---:|
| AITER fwd + hipBLASLt bwd | 909 (+0.6%) | 858 (+3.2%) |
| hipBLASLt baseline | 903 | 831 |

Loss converges normally. Profile confirmed AITER kernels are called (10.2% of GPU time).

## What's Been Built

### jax-aiter changes (all in `jax-aiter/`):
- `jax_aiter/gemm/gemm.py` -- BF16 GEMM with `custom_vjp` (AITER fwd, `lax.dot_general` bwd) + `custom_partitioning`
- `jax_aiter/gemm_fp8/gemm_fp8_mi350.py` -- FP8 GEMM with `custom_vjp` + delayed scaling + `custom_partitioning`
- `csrc/ffi/gemm_fwd/gemm_fwd_ja.cu` -- per-device BF16 kernel cache + eager preload
- `csrc/ffi/gemm_fp8_mi350/gemm_fp8_mi350_ja.cu` -- per-device FP8 kernel cache
- `jax_aiter/ffi/registry.py` -- preload hooks for both BF16 and FP8

### MaxText changes (all in `maxtext/`):
- `src/maxtext/layers/quantizations.py` -- `AiterBf16DotGeneralOp`, `AiterFp8DotGeneralOp`, `AiterBf16Quantization`, `AiterFp8Quantization`
- `src/maxtext/configs/types.py` -- `AITER_BF16`, `AITER_FP8` enum values

## Commands

### Run hybrid (AITER fwd + hipBLASLt bwd) on 8-GPU:
```bash
./tools/in_container.sh "cd /ruvaidya/aiter_proj/maxtext && \
  JA_ROOT_DIR=/ruvaidya/aiter_proj/jax-aiter \
  AITER_ASM_DIR=/ruvaidya/aiter_proj/jax-aiter/third_party/aiter/hsa/ \
  XLA_FLAGS='--xla_gpu_enable_command_buffer= --xla_gpu_enable_triton_gemm=False --xla_gpu_enable_cublaslt=True --xla_gpu_autotune_level=4 --xla_gpu_enable_nccl_comm_splitting=false' \
  JAX_PLATFORMS=rocm DECOUPLE_GCLOUD=TRUE \
  PYTHONPATH=/ruvaidya/aiter_proj/maxtext/src:\$PYTHONPATH \
  HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python3 -m maxtext.trainers.pre_train.train \
    src/maxtext/configs/base.yml \
    run_name=8gpu_hybrid hardware=gpu steps=20 \
    model_name=llama2-7b attention=cudnn_flash_te \
    enable_checkpointing=False \
    ici_expert_parallelism=1 ici_fsdp_parallelism=-1 ici_data_parallelism=1 \
    remat_policy=full scan_layers=True dataset_type=synthetic \
    logits_dot_in_fp32=False dtype=bfloat16 weight_dtype=bfloat16 \
    per_device_batch_size=10 max_target_length=4096 \
    shardy=False packing=False \
    base_output_directory=/tmp/maxtext_output \
    quantization=aiter_bf16"
```

### Run hipBLASLt baseline on 8-GPU:
```bash
./tools/in_container.sh "cd /ruvaidya/aiter_proj/maxtext && \
  XLA_FLAGS='--xla_gpu_enable_command_buffer= --xla_gpu_enable_triton_gemm=False --xla_gpu_enable_cublaslt=True --xla_gpu_autotune_level=4 --xla_gpu_enable_nccl_comm_splitting=false' \
  JAX_PLATFORMS=rocm DECOUPLE_GCLOUD=TRUE \
  PYTHONPATH=/ruvaidya/aiter_proj/maxtext/src:\$PYTHONPATH \
  HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python3 -m maxtext.trainers.pre_train.train \
    src/maxtext/configs/base.yml \
    run_name=8gpu_baseline hardware=gpu steps=20 \
    model_name=llama2-7b attention=cudnn_flash_te \
    enable_checkpointing=False \
    ici_expert_parallelism=1 ici_fsdp_parallelism=-1 ici_data_parallelism=1 \
    remat_policy=full scan_layers=True dataset_type=synthetic \
    logits_dot_in_fp32=False dtype=bfloat16 weight_dtype=bfloat16 \
    per_device_batch_size=10 max_target_length=4096 \
    shardy=False packing=False \
    base_output_directory=/tmp/maxtext_output"
```

### For 1-GPU: change `HIP_VISIBLE_DEVICES=0`, `ici_fsdp_parallelism=1`, remove `nccl_comm_splitting`

### Build GEMM .so:
```bash
./tools/in_container.sh "cd /ruvaidya/aiter_proj/jax-aiter && \
  JA_ROOT_DIR=\$PWD AITER_SYMBOL_VISIBLE=1 GPU_ARCHS=gfx950 \
  AITER_ASM_DIR=\$PWD/third_party/aiter/hsa/ \
  make build/jax_aiter_build/gemm_fwd_ja.so"
```

### Profile with rocprofv3:
```bash
# Add before python3: rocprofv3 --kernel-trace -o <output_dir> --
# Analyze with: python3 docs/logs/analyze_all_profiles.py
```

### Parse benchmark results (skip first 5 warmup steps):
```python
import re
def parse_steps(logfile, skip=5):
    steps = []
    with open(logfile) as f:
        for line in f:
            m = re.search(r'completed step: (\d+), seconds: ([\d.]+), TFLOP/s/device: ([\d.]+)', line)
            if m:
                step, secs, tflops = int(m.group(1)), float(m.group(2)), float(m.group(3))
                if step >= skip:
                    steps.append((step, secs, tflops))
    return len(steps), sum(s[1] for s in steps)/len(steps), sum(s[2] for s in steps)/len(steps)
```

## What To Do Next (ordered by impact)

### 1. Fused Add+RMSNorm in MaxText (READY -- just needs wiring)
The kernel is already built: `jax_aiter.rmsnorm.rms_norm_with_add(x, residual, gamma, eps)`.
Profiling shows XLA fusions (RMSNorm, SiLU, optimizer) take 15% of GPU time.
Fused Add+RMSNorm saves one full memory pass over the hidden state.

**What to do:**
- Wire `rms_norm_with_add` into MaxText's `normalizations.py` (replace the JAX-computed RMSNorm)
- Benchmark the impact on the 15% XLA fusions bucket
- This stacks with the GEMM improvement (they're independent)

### 2. 8-GPU Consistency Runs
The +3.2% on 8-GPU needs verification with back-to-back runs (like we did for 1-GPU).
Run baseline -> hybrid -> baseline -> hybrid on 8-GPU to control for thermal effects.

### 3. FP8 Large-Batch Stability
FP8 training works at small batch but NaN at large batch. Needs:
- Per-layer scale calibration (different layers have different activation ranges)
- Or: use the Flax `in_qdq`/`out_qdq` pattern with AITER FP8 kernel
- The FP8 kernel is correct (0.9% relative error) -- it's purely a scaling issue

### 4. Fused MoE GEMM (for Mixtral/DeepSeek)
AITER's `fused_moe` combines token routing + expert GEMMs in one kernel.
This is where AITER adds unique value that hipBLASLt can't provide.
Needs FFI handler + MaxText MoE layer integration.

### 5. Inference/Decode Path
AITER ASM GEMM genuinely wins at small M (decode shapes, M=1..32).
Different MaxText config needed (batch=1, decode mode).
Also: split-KV attention, paged attention for serving.

## Key Lessons

1. **Always compare against hipBLASLt with proper flags** (`--xla_gpu_enable_cublaslt=True --xla_gpu_autotune_level=4`), not Triton GEMM
2. **hipBLASLt on MI350 uses mandatory Stream-K** -- can't be disabled
3. **FFI custom_calls can't get XLA layout optimization** -- use `lax.dot_general` for ops that need transposed layouts
4. **The hybrid approach works** -- use AITER where it wins (forward TN layout), hipBLASLt where it wins (backward with arbitrary layouts)
5. **Profile with rocprofv3** to verify kernels are actually being called and measure overhead
6. **Skip first 5 steps** when benchmarking (compilation + autotuning warmup)
7. **Run multiple times** to control for GPU thermal effects

## Repository Layout
```
/home/ruvaidya/aiter_proj/
  jax-aiter/          -- our code (edit here)
  maxtext/            -- MaxText (integration target)
  aiter/              -- reference AITER (read-only)
  rocm-libraries/     -- hipBLASLt source (read-only)
  jax/                -- JAX source (read-only, for API reference)
  docs/logs/          -- all benchmark logs and profiles
  .cursor/memory_bank/ -- project context
```

## Memory Bank
Read `.cursor/memory_bank/activeContext.md` for full context including all benchmark
numbers, technical findings, and the evolution of the approach.
