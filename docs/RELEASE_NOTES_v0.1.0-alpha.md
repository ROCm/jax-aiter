<!-- SPDX-License-Identifier: MIT -->
<!-- Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved. -->

# jax-aiter v0.1.0-alpha (LITE / MXFP4)

First public alpha of `jax-aiter`. This release ships the **lite** wheel:
the AMD AITER **MXFP4 (FP4) training path** for JAX on ROCm gfx950 (MI350 /
MI355X), built on **AITER v0.1.14**.

- **Wheel:** `jax_aiter-0.1.0a0+lite-cp312-cp312-linux_x86_64.whl` (~43 MiB)
- **AITER pin:** `v0.1.14` (`bd0534e9630f8f142f51689f5808c627460e35bf`)
- **Target:** ROCm 7.2+, Python 3.12, gfx950 (MI350 / MI355X)
- **Runtime:** JAX 0.9.x + ROCm pjrt/plugin. No PyTorch at runtime.

## Scope

The lite wheel is the MXFP4-focused subset. It deliberately **excludes the
multi-GB MHA flash-attention JIT libraries** (`libmha_fwd.so` /
`libmha_bwd.so`) and their FFI shims, keeping the wheel at ~43 MiB instead of
~513 MiB. The canonical FP4 training recipe routes attention through
TransformerEngine (`attention=cudnn_flash_te`), so AITER MHA is **not** needed
for the FP4 path — the lite wheel is sufficient for the recipe below.

### Included ops

| Op | API | Notes |
|----|-----|-------|
| FP4 GEMM (training) | `gemm_fp4_bf16(a, b)` | BF16 in/out, MXFP4 internally, `custom_vjp`, 35 ASM kernels (`_hsa/gfx950/f4gemm`). |
| MXFP4 cast | `CastMxfp4JA` / `CastMxfp4DualJA` | BF16 → E2M1 + E8M0 block scales. |
| FP4 GEMM (low-level) | `gemm_fp4(a, b, a_scale, b_scale)` | Pre-quantized fp4x2 inputs. |
| BF16 GEMM (training) | `gemm(a, b)` | 24 ASM kernels, `custom_vjp`. |
| RMSNorm | `rms_norm`, `rms_norm_with_add` | See limitation note on `librmsnorm_fwd.so` below. |
| SiLU-and-Mul | `silu_and_mul(x)` | Fused activation. |

### Excluded (use the full wheel)

- `flash_attn_func` / `flash_attn_varlen` (AITER MHA). Importing
  `jax_aiter.mha` from the lite wheel raises a clear `ModuleNotFoundError`
  pointing to the full variant.

## Install

```bash
pip install jax_aiter-0.1.0a0+lite-cp312-cp312-linux_x86_64.whl
```

**ROCm JAX runtime prerequisite.** The lite wheel needs a ROCm-enabled JAX.
Obtain one either way:

- `pip install jax jax-rocm7-pjrt jax-rocm7-plugin` — currently resolves to
  **0.9.1**, which is ABI-compatible with the FP4 FFI and is the stack the
  clean-container smoke validated; or
- pull the exact build-matched **`0.9.0+rocm7.2.0`** wheels (`jax`, `jaxlib`,
  `jax-rocm7-pjrt`, `jax-rocm7-plugin`) from the
  [rocm-jax v0.9.0 release assets](https://github.com/ROCm/rocm-jax/releases/tag/rocm-jax-v0.9.0-rc3).
  These are published there as GitHub release assets (they are simply not on
  the PyPI / radeon pip indexes).

The wheel bundles the gfx950 FP4 ASM kernels under `jax_aiter/_hsa/`.
`AITER_ASM_DIR` auto-resolves to the packaged `_hsa/` as long as
`JA_ROOT_DIR` is **unset** (leave it unset for installed-wheel use; it is only
needed for source/dev trees).

## Validation (MXFP4-only)

Validated standalone in a **clean sibling container** with no AITER source
tree, no MaxText, and no build toolchain (`scripts/validate_wheel.sh` →
`docker/validation/Dockerfile.lite` + `smoke_fp4_gemm.py`):

- **Base:** `ghcr.io/rocm/jax-base-ubu24.rocm720` + ROCm JAX stack
  (`jax` / `jax-rocm7-pjrt` / `jax-rocm7-plugin` `0.9.1`, installed from
  `repo.radeon.com/rocm/manylinux/rocm-rel-7.2/`).
- **Smoke:** `import jax_aiter` (prints `__version__ = 0.1.0a0`), then
  `gemm_fp4_bf16` on a `1024×4096 @ 4096×4096` BF16 GEMM; asserts shape +
  finiteness.
- **Result:** **PASS** — ROCm detected 8× gfx950, "Loaded 35 FP4 GEMM kernels"
  from the wheel's packaged `_hsa/`, `shape=(1024, 4096) dtype=bfloat16`.

> **Validated on two jax stacks.** The lite wheel's FP4 path is validated on
> **both** `jax 0.9.0` and `jax 0.9.1`:
> - **0.9.0** + `jax-rocm7-pjrt/plugin 0.9.0+rocm7.2.0` — the `rv_aiter` build
>   env, where the pre-bump and post-bump MaxText FP4 E2E ran (the
>   build-matched stack).
> - **0.9.1** — the clean-container FP4 GEMM smoke above (what `pip install`
>   resolves to today).
>
> The FP4 FFI handlers use the version-stable XLA FFI C API, so `0.9.0` and
> `0.9.1` are ABI-compatible. The exact `0.9.0+rocm7.2.0` wheels are published
> as assets on the [rocm-jax v0.9.0 release](https://github.com/ROCm/rocm-jax/releases/tag/rocm-jax-v0.9.0-rc3)
> (they are just not on the PyPI/radeon pip indexes), so the build env is
> fully reproducible.

FP8 and full-E2E MaxText runs were intentionally **not** part of this
release validation (MXFP4-only).

## Known limitations / caveats

- **No flash attention in lite.** MHA is excluded by design. Use the full
  wheel for `flash_attn_func` / `flash_attn_varlen`.
- **Full wheel deferred (MHA-shim CK header conflict).** Building the full
  wheel's MHA FFI shims (`mha_fwd_ja.so` / `mha_bwd_ja.so`) currently hits a
  Composable Kernel (CK) header conflict against the v0.1.14 AITER/CK headers.
  The lite wheel sidesteps this by excluding the MHA shims; the full-wheel
  build is a tracked follow-up.
- **`librmsnorm_fwd.so` is stale (off the MXFP4 path).** The bundled
  `librmsnorm_fwd.so` is the pre-bump build (from AITER pin `3baf198`, not
  rebuilt against v0.1.14). RMSNorm is **off** the MXFP4 FP4 GEMM/cast path, so
  this does **not** affect the FP4 validation above. Rebuilding it against
  v0.1.14 is a follow-up if RMSNorm is exercised in production.

## MaxText FP4 integration

### Apply the integration patch

`scripts/maxtext_aiter_fp4.patch` carries the MaxText-side FP4 plumbing
(AITER quantization modes incl. `aiter_fp4`, FP4 dispatch). It is scoped to
`src/maxtext/{configs,layers}/` (4 files, +1047/−20) and applies cleanly onto
`origin/rocm-main`:

```bash
cd /path/to/maxtext
git apply --check /path/to/jax-aiter/scripts/maxtext_aiter_fp4.patch   # verify
git apply         /path/to/jax-aiter/scripts/maxtext_aiter_fp4.patch   # apply
```

### Canonical FP4 recipe (env + flags)

Authoritative source: `scripts/run_fresh_maxtext_e2e.sh` (parent project root).

MaxText args (FP4 leg):

```text
src/maxtext/configs/base.yml \
  hardware=gpu model_name=llama3-8b           # or llama3.3-70b
  quantization=aiter_fp4 use_jax_aiter=True aiter_attention=False \
  attention=cudnn_flash_te \
  ici_fsdp_parallelism=8 ici_data_parallelism=1 ici_expert_parallelism=1 \
  remat_policy=minimal_flash                  # 70B: remat_policy=full (forced; minimal_flash OOMs at 70B)
  per_device_batch_size=4                     # 70B: 7
  max_target_length=8192 scan_layers=True param_scan_axis=1 use_iota_embed=True \
  dtype=bfloat16 weight_dtype=bfloat16 logits_dot_in_fp32=False
```

Environment:

```bash
# Canonical (see .cursor/rules/35-policies.mdc).
export XLA_PYTHON_CLIENT_MEM_FRACTION=.97
export JAX_PLATFORMS=rocm
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# AITER FP4 (gfx950).
export GPU_ARCHS=gfx950
export AITER_SYMBOL_VISIBLE=1
export AITER_FP4_ATTN=1
# Source/dev tree only — for the installed wheel leave BOTH unset so
# AITER_ASM_DIR auto-resolves to the packaged jax_aiter/_hsa/:
#   export JA_ROOT_DIR=/path/to/jax-aiter
#   export AITER_ASM_DIR=$JA_ROOT_DIR/third_party/aiter/hsa/

# TransformerEngine attention (cudnn_flash_te) for the canonical FP4 leg.
export NVTE_FUSED_ATTN=1 NVTE_FUSED_ATTN_CK=1 NVTE_FUSED_ATTN_AOTRITON=0
export NVTE_CK_USES_FWD_V3=1 NVTE_CK_USES_BWD_V3=1
export NVTE_CK_IS_V3_ATOMIC_FP32=0 NVTE_CK_HOW_V3_BF16_CVT=2
export NVTE_USE_HIPBLASLT=1 NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export GPU_MAX_HW_QUEUES=2 HIP_FORCE_DEV_KERNARG=1 HSA_FORCE_FINE_GRAIN_PCIE=1
```

`XLA_FLAGS` (baseline):

```text
--xla_gpu_memory_limit_slop_factor=95
--xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592
--xla_gpu_all_gather_combine_threshold_bytes=8589934592
--xla_gpu_enable_command_buffer=
--xla_gpu_enable_latency_hiding_scheduler=True
--xla_gpu_enable_triton_gemm=False
--xla_gpu_enable_cublaslt=True
--xla_gpu_autotune_level=4
--xla_gpu_enable_all_gather_combine_by_dim=FALSE
--xla_gpu_enable_nccl_comm_splitting=false
```

> The `AITER_FORCE_KERNEL_NAME=...BpreShuffle_256x256` /
> `AITER_FORCE_LOG2_K_SPLIT=0` Phase-B pins are now **redundant**: the in-tree
> splitK=1 heuristic fix (gemm_fp4 search-axis) selects `BpreShuffle_256x256`
> splitK=1 automatically. They remain available only as an optional override.

## Reproduce

```bash
# Build the lite wheel (inside the rv_aiter container; reuses prebuilt
# AITER JIT libs, never rebuilds MHA — sha256-guarded).
docker exec rv_aiter bash -lc \
  "cd /ruvaidya/aiter_proj/jax-aiter && bash scripts/build_wheel.sh --variant lite"

# Validate in a clean sibling container (host-side; GPU smoke).
bash scripts/validate_wheel.sh
```
