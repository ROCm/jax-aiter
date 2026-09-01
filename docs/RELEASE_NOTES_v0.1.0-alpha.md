<!-- SPDX-License-Identifier: MIT -->
<!-- Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved. -->

# jax-aiter v0.1.0-alpha (MXFP4)

First public alpha of `jax-aiter`: the AMD AITER **MXFP4 (FP4) training path**
for JAX on ROCm gfx950 (MI350 / MI355X), built on **AITER v0.1.14**. Two wheels
are published — pick one:

- **lite** `jax_aiter-0.1.0a0+lite-cp312-cp312-linux_x86_64.whl` (~54 MiB) —
  MXFP4 path only; **no** AITER flash-attention. Sufficient for the canonical
  FP4 recipe (attention routes through TransformerEngine).
- **full** `jax_aiter-0.1.0a0-cp312-cp312-linux_x86_64.whl` (~739 MiB) —
  everything in lite **plus** AITER MHA flash-attention (`flash_attn_func` /
  `flash_attn_varlen`; bundles `libmha_fwd.so` + `libmha_bwd.so`).

- **AITER pin:** `v0.1.14` (`bd0534e9630f8f142f51689f5808c627460e35bf`)
- **Target:** ROCm 7.2+, Python 3.12, gfx950 (MI350 / MI355X)
- **Runtime:** JAX 0.9.x + ROCm pjrt/plugin. No PyTorch at runtime.

## Scope

Both wheels share the MXFP4 core (FP4 GEMM/cast, BF16 GEMM, RMSNorm,
SiLU-and-mul). They differ only in AITER flash-attention:

- **lite** excludes the multi-GB MHA JIT libs (`libmha_fwd.so` /
  `libmha_bwd.so`) + their FFI shims, keeping the wheel ~54 MiB. The canonical
  FP4 training recipe routes attention through TransformerEngine
  (`attention=cudnn_flash_te`), so AITER MHA is **not** needed for the FP4 path
  — lite is sufficient for the recipe below.
- **full** adds the AITER MHA flash-attention path (forward + backward),
  bundling the v0.1.14 `libmha_fwd.so` (1.81 GB) + `libmha_bwd.so` (1.45 GB);
  the wheel is ~739 MiB (deflate-compressed).

### Included ops

| Op | API | Notes |
|----|-----|-------|
| FP4 GEMM (training) | `gemm_fp4_bf16(a, b)` | BF16 in/out, MXFP4 internally, `custom_vjp`, 35 ASM kernels (`_hsa/gfx950/f4gemm`). |
| MXFP4 cast | `CastMxfp4JA` / `CastMxfp4DualJA` | BF16 → E2M1 + E8M0 block scales. |
| FP4 GEMM (low-level) | `gemm_fp4(a, b, a_scale, b_scale)` | Pre-quantized fp4x2 inputs. |
| BF16 GEMM (training) | `gemm(a, b)` | 24 ASM kernels, `custom_vjp`. |
| RMSNorm | `rms_norm`, `rms_norm_with_add` | See limitation note on `librmsnorm_fwd.so` below. |
| SiLU-and-Mul | `silu_and_mul(x)` | Fused activation. |

### MHA flash-attention (full wheel only)

- `flash_attn_func` / `flash_attn_varlen` (AITER MHA, `custom_vjp` fwd+bwd).
  Shipped in the **full** wheel. Importing `jax_aiter.mha` from the **lite**
  wheel raises a clear `ModuleNotFoundError` pointing to the full variant.

## Install

```bash
# lite (MXFP4 only):
pip install jax_aiter-0.1.0a0+lite-cp312-cp312-linux_x86_64.whl
# full (MXFP4 + AITER MHA flash-attention):
pip install jax_aiter-0.1.0a0-cp312-cp312-linux_x86_64.whl
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

## Validation

**Both** wheels are validated standalone in a **clean sibling container** with
no AITER source tree, no MaxText, and no build toolchain
(`scripts/validate_wheel.sh --variant {lite|full}` →
`docker/validation/Dockerfile.lite` + the smoke scripts):

- **Base:** `ghcr.io/rocm/jax-base-ubu24.rocm720` + ROCm JAX stack
  (`jax` / `jax-rocm7-pjrt` / `jax-rocm7-plugin` `0.9.1`, installed from
  `repo.radeon.com/rocm/manylinux/rocm-rel-7.2/`).
- **lite (`smoke_fp4_gemm.py`):** `import jax_aiter` (prints
  `__version__ = 0.1.0a0`), then `gemm_fp4_bf16` on a `1024×4096 @ 4096×4096`
  BF16 GEMM; asserts shape + finiteness. **PASS** — 8× gfx950, "Loaded 35 FP4
  GEMM kernels" from the packaged `_hsa/`, `shape=(1024, 4096) dtype=bfloat16`.
- **full (`smoke_fp4_gemm.py` + `smoke_mha.py`):** the FP4 GEMM smoke above
  **plus** `flash_attn_func` forward **and** backward (via `jax.grad`) on a
  `(2, 256, 4, 64)` bf16 tensor; asserts fwd + dq/dk/dv shapes + finiteness.
  **PASS** (`MHA smoke PASS fwd+bwd shape=(2, 256, 4, 64)`) — proves the bundled
  `libmha_fwd.so` + `libmha_bwd.so` load and run from the wheel alone.

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

- **No flash attention in the lite wheel** (by design). Use the **full** wheel
  for `flash_attn_func` / `flash_attn_varlen`.
- **Full wheel size (~739 MiB).** It embeds the v0.1.14 MHA JIT libs
  (`libmha_fwd.so` 1.81 GB + `libmha_bwd.so` 1.45 GB, deflate-compressed in the
  wheel); still well under the 2 GiB GitHub release-asset limit.
- **alpha quality.** APIs may change; gfx950-only; validated on the smoke
  surface above (FP4 GEMM + MHA fwd/bwd), not the full op matrix. All 3 JIT
  libs (incl. `librmsnorm_fwd.so`) are rebuilt against v0.1.14.

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
# Build both wheels (inside the rv_aiter container; reuses prebuilt AITER JIT
# libs, never rebuilds them — sha256-guarded).
docker exec rv_aiter bash -lc \
  "cd /ruvaidya/aiter_proj/jax-aiter && \
   bash scripts/build_wheel.sh --variant lite && \
   bash scripts/build_wheel.sh --variant full"

# Validate each in a clean sibling container (host-side; GPU smoke).
bash scripts/validate_wheel.sh --variant lite   # FP4 GEMM
bash scripts/validate_wheel.sh --variant full   # FP4 GEMM + MHA fwd/bwd
```
