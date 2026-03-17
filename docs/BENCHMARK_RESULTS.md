# JAX-AITER GEMM Benchmark Results

**Date:** 2026-03-17
**Author:** Ruvaid Yaqoob + AI assistant

## Test Environment

| Component | Version / Details |
|-----------|------------------|
| GPU | 8x AMD Instinct MI355X (gfx950, CDNA4) |
| VRAM per GPU | 288 GB (309,220,868,096 bytes) |
| ROCm | 7.2.0 |
| HIP | 7.2.26015-fc0010cf6a |
| Python | 3.12.3 |
| JAX | 0.9.0 |
| jaxlib | 0.9.0+rocm7 |
| Flax | 0.12.5 |
| Container | `ghcr.io/rocm/jax-base-ubu24.rocm720:latest` (name: `rv_aiter`) |
| AITER | Submodule at `jax-aiter/third_party/aiter/` (commit `3baf198`) |
| MaxText | Local clone at `/home/ruvaidya/aiter_proj/maxtext/` |
| hipBLASLt | From ROCm 7.2.0, Tensile with mandatory Origami + Stream-K on MI350 |

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Model | Llama2-7B (6.738B params) |
| Attention | `cudnn_flash_te` (TE -> AITER CK kernels) |
| Precision | BF16 (dtype=bfloat16, weight_dtype=bfloat16) |
| Dataset | Synthetic |
| Remat | full |
| Scan layers | True |
| Packing | False |
| Shardy | False |

## Workload Configurations

| Config | per_device_batch_size | max_target_length | M per device | Total tokens/step |
|--------|:---:|:---:|:---:|:---:|
| Large (8-GPU) | 10 | 4096 | 40,960 | 327,680 |
| Large (1-GPU) | 10 | 4096 | 40,960 | 40,960 |
| Small (1-GPU) | 2 | 1024 | 2,048 | 2,048 |

## XLA Flags

### hipBLASLt Baseline
```
--xla_gpu_enable_command_buffer=
--xla_gpu_enable_triton_gemm=False
--xla_gpu_enable_cublaslt=True
--xla_gpu_autotune_level=4
```

### AITER BF16 / Hybrid (AITER fwd + hipBLASLt bwd)
```
--xla_gpu_enable_command_buffer=
--xla_gpu_enable_triton_gemm=False
--xla_gpu_enable_cublaslt=True
--xla_gpu_autotune_level=4
--xla_gpu_enable_nccl_comm_splitting=false  (8-GPU only)
```
Plus: `JA_ROOT_DIR`, `AITER_ASM_DIR`, `quantization=aiter_bf16`

### AITER BF16 (all-AITER backward, old)
```
--xla_gpu_enable_command_buffer=
--xla_gpu_enable_nccl_comm_splitting=false  (8-GPU only)
```
No hipBLASLt flags (backward uses AITER GEMM with explicit transposes).

### nanoo_fp8
```
--xla_gpu_enable_command_buffer=
--xla_gpu_enable_nccl_comm_splitting=false  (8-GPU only)
```
Plus: `quantization=nanoo_fp8`

## Results: 1-GPU (Large Workload)

Steps 5-19 average (skip first 5 for warmup/compilation).

| Backend | TFLOP/s | Step (s) | Runs | Notes |
|---------|:---:|:---:|:---:|-------|
| **AITER fwd + hipBLASLt bwd** | **909** | **1.93** | 3 | `quantization=aiter_bf16` + hipBLASLt flags |
| hipBLASLt BF16 (AT=4) | 903 | 1.94 | 4 | Pure baseline, no quantization flag |
| AITER BF16 (all AITER bwd) | 828 | 2.12 | 1 | 31% transpose overhead in backward |
| AITER FP8 | 800 | 2.19 | 1 | FP8 fwd + BF16 bwd, loss=NaN (broken quant) |
| nanoo_fp8 | 672 | 2.61 | 1 | Flax NANOOFp8DotGeneralOp |

### 1-GPU Individual Run Data

| Run | Config | TFLOP/s | Step (s) |
|-----|--------|:---:|:---:|
| Baseline R1 (cold) | hipBLASLt | 890.9 | 1.971 |
| Baseline R2 | hipBLASLt | 904.2 | 1.942 |
| Baseline R3 | hipBLASLt | 910.2 | 1.929 |
| Baseline R4 (after hybrid) | hipBLASLt | 907.5 | 1.935 |
| Hybrid R1 | AITER fwd + hipBLASLt bwd | 906.7 | 1.937 |
| Hybrid R2 | AITER fwd + hipBLASLt bwd | 909.3 | 1.931 |
| Hybrid R3 (before R4) | AITER fwd + hipBLASLt bwd | 910.8 | 1.928 |

**Observation:** GPU thermal warmup causes ~2% variation between cold and warm runs.
Back-to-back comparison (R3 hybrid 910.8 vs R4 baseline 907.5) shows +0.4%.

## Results: 8-GPU FSDP-8 (Large Workload)

Steps 5-19 average.

| Backend | TFLOP/s | Step (s) | Notes |
|---------|:---:|:---:|-------|
| **AITER fwd + hipBLASLt bwd** | **858** | **2.05** | +3.2% vs baseline |
| hipBLASLt BF16 (AT=4) | 831 | 2.11 | Baseline |
| nanoo_fp8 | 646 | 2.72 | Flax FP8 |
| AITER FP8 (all AITER) | 221 | 7.96 | FP8 fwd + BF16 bwd, loss=NaN |
| AITER BF16 (all AITER) | 185 | 9.50 | custom_partitioning, no comm overlap |

**Note:** 8-GPU hybrid needs consistency verification with back-to-back runs.

## Results: 1-GPU FP8 Training Stability

| Config | batch | seq | Steps before NaN | Final loss | TFLOP/s |
|--------|:---:|:---:|:---:|:---:|:---:|
| FP8 delayed scaling | 2 | 1024 | 20+ (stable) | 10.810 | 248 |
| FP8 static scaling | 2 | 1024 | 6 | 8.066 | ~350 |
| FP8 static scaling | 1 | 4096 | 2 | 10.485 | ~530 |
| FP8 delayed scaling | 10 | 4096 | 0 | NaN | 787 |
| FP8 (broken, no shuffle) | 10 | 4096 | 0 | NaN | 800 |

## Profiling: Kernel Time Breakdown

### rocprofv3 --kernel-trace, 1-GPU, 5 steps

**Hybrid (AITER fwd + hipBLASLt bwd), batch=10, seq=4096:**

| Category | Time (ms) | % | Count |
|----------|-----------|---|-------|
| hipBLASLt Tensile (bwd) | 24,323 | 79.1% | 5,093 |
| AITER ASM (fwd) | 3,128 | 10.2% | 2,080 |
| Other XLA | 1,693 | 5.5% | 22,129 |
| Attention (CK) | 1,363 | 4.4% | 800 |
| Transpose/Copy | 153 | 0.5% | 5,372 |
| Memset/Fill | 77 | 0.2% | 6,064 |

**hipBLASLt Baseline, batch=10, seq=4096:**

| Category | Time (ms) | % | Count |
|----------|-----------|---|-------|
| hipBLASLt Tensile | 30,534 | 89.7% | 8,205 |
| Attention (CK) | 1,351 | 4.0% | 800 |
| Other XLA | 1,169 | 3.4% | 23,093 |
| FP8 Overhead (autotuning) | 642 | 1.9% | 1,964 |
| Transpose/Copy | 260 | 0.8% | 8,001 |

**AITER BF16 (all AITER bwd), batch=1, seq=1024 (earlier profile):**

| Category | Time (ms) | % | Count |
|----------|-----------|---|-------|
| AITER ASM GEMM | 1,216 | 48.8% | 2,592 |
| Triton GEMM (residual) | 404 | 16.2% | 33 |
| Transpose/Copy | 340 | 13.7% | 2,825 |
| Other XLA | 337 | 13.5% | 8,346 |
| Attention (CK) | 114 | 4.6% | 480 |

**Key finding:** Transpose overhead dropped from 13.7% (all-AITER bwd) to 0.5% (hybrid)
by using `lax.dot_general` for backward instead of AITER GEMM with explicit transposes.

### 8-GPU Profiling (AITER custom_partitioning path, batch=10, seq=4096)

Per-device per-step breakdown:

| Category | Time (ms) | % |
|----------|-----------|---|
| AITER GEMM | 4,122 | 43.6% |
| Communication (RCCL) | 4,046 | 42.8% |
| Transpose/Copy | 747 | 7.9% |
| Attention (CK) | 272 | 2.9% |
| Other XLA | 182 | 1.9% |

**Key finding:** Zero compute/communication overlap in AITER path (everything serialized).
hipBLASLt baseline achieves ~20% overlap via XLA's native collective-matmul fusion.

## AITER Kernel Details

### BF16 Kernels Used

| Shape (M, N, K) | Kernel | Tile | splitK | Notes |
|-----------------|--------|------|--------|-------|
| (40960, 4096, 4096) | `bf16gemm_bf16_tn_256x256` | 256x256 | 1 | q/k/v/o projections |
| (40960, 11008, 4096) | `bf16gemm_fp32bf16_tn_*x64_splitk_clean` | *x64 | >1 | gate/up proj (11008%256!=0) |
| (40960, 4096, 11008) | `bf16gemm_bf16_tn_256x256` | 256x256 | 1 | down projection |

### FP8 Kernels Used

| Condition | Kernel | Notes |
|-----------|--------|-------|
| M > 32 | `f8_block_scale_mi350_x128` | Training shapes |
| M <= 32 | `f8_block_scale_mi350_x32` | Decode shapes |

FP8 kernel requires weight pre-shuffle: `reshape(N//16,16,K//32,2,16).transpose(0,2,3,1,4).reshape(N,K)`

### Kernel Selection Heuristic

The `select_kernel` function in `gemm_fwd_ja.cu` iterates over 24 kernel configs and picks
based on: fewest CU rounds > fewer empty CUs > less out-of-bounds > better compute-to-memory ratio.
Filters: `N % tileN == 0` and `bPreshuffle == 0` (pre-shuffle variants skipped).

## hipBLASLt on MI350

- Uses **Origami + Stream-K** scheduling (mandatory, `TENSILE_SOLUTION_SELECTION_METHOD` has no effect)
- Stream-K distributes work evenly across all 256 CUs regardless of shape
- Tensile kernels support all layout combinations natively: `Ailk`, `Alik`, `Bjlk`, `Bljk`
- XLA fuses bias addition into Tensile kernel (`Cijk_*_Bias_*`)
- Autotuning level 4 tries multiple kernel configs per shape at compile time
- Source: `rocm-libraries/projects/hipblaslt/tensilelite/`

## Log Files

All logs stored in `docs/logs/`:

| File | Description |
|------|-------------|
| `bench_1gpu_hipblaslt.log` | 1-GPU hipBLASLt baseline R1 |
| `bench_1gpu_baseline_r2.log` | 1-GPU hipBLASLt baseline R2 |
| `bench_1gpu_baseline_r3.log` | 1-GPU hipBLASLt baseline R3 |
| `bench_1gpu_baseline_r4.log` | 1-GPU hipBLASLt baseline R4 |
| `bench_1gpu_hybrid_r2.log` | 1-GPU hybrid R2 |
| `bench_1gpu_hybrid_r3.log` | 1-GPU hybrid R3 |
| `bench_1gpu_aiter_bf16.log` | 1-GPU AITER BF16 (all AITER bwd) |
| `bench_1gpu_aiter_fp8.log` | 1-GPU AITER FP8 (broken quant) |
| `bench_1gpu_nanoo_fp8.log` | 1-GPU nanoo_fp8 |
| `bench_8gpu_aiter_bf16.log` | 8-GPU AITER BF16 (all AITER bwd) |
| `bench_8gpu_nanoo_fp8.log` | 8-GPU nanoo_fp8 |
| `bench_8gpu_hybrid_laxbwd.log` | 8-GPU hybrid |
| `8gpu_pure_baseline_20.log` | 8-GPU hipBLASLt baseline |
| `profiles/` | rocprofv3 `.db` files for all configurations |
| `analyze_all_profiles.py` | Script to analyze profile DBs |
| `analyze_profile.py` | Script to analyze kernel trace CSVs |
