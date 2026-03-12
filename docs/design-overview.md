# JAX-AITER Design Overview

This document describes how `jax_aiter` is structured, how calls flow from JAX
into ROCm-native kernels, and how the project is built and packaged.

## Goals

- Expose high-performance AITER kernels as JAX-native APIs.
- Keep JAX autodiff behavior by using `custom_vjp` wrappers.
- Select kernel implementations dynamically based on runtime shapes/options.
- Avoid unnecessary copies by using JAX FFI and direct buffer passing.

## High-Level Architecture

```
User JAX code
  -> jax_aiter.mha / gemm_a8w8 / wv_splitkq Python APIs
  -> custom_vjp forward/backward wrappers (for attention)
  -> kernel dispatch logic (CK vs FMHA v3, varlen vs dense, etc.)
  -> jax.ffi.ffi_call("TargetName", ...)
  -> jax_aiter.ffi.registry resolves + registers symbol pointers
  -> libjax_aiter.so + module_*.so (AITER/JAX bridge shared objects)
  -> ROCm kernels on GPU
```

The repository is a bridge layer. It does not reimplement core kernel math in
Python; it orchestrates configuration, dispatch, registration, and ABI plumbing.

## Main Runtime Components

### 1) Package bootstrap and environment setup

- `jax_aiter/__init__.py`
  - Calls `set_aiter_asm_dir()` early.
  - Uses lazy submodule loading for public top-level modules.
- `jax_aiter/ja_compat/config.py`
  - Resolves library roots from `JA_ROOT_DIR` (dev) or packaged `_lib` (wheel).
  - Sets `AITER_ASM_DIR` to architecture-specific HSA kernel directories.

### 2) FFI registry and symbol resolution

- `jax_aiter/ffi/registry.py`
  - Owns symbol-to-module mapping (`SYMBOL_TO_MODULE_MAP`).
  - Loads umbrella library (`libjax_aiter.so`) first.
  - Loads thin modules from `build/aiter_build` and `build/jax_aiter_build`.
  - Resolves symbol pointers via `ctypes` and registers targets with
    `jax.ffi.register_ffi_target(...)`.

This is the central handshake between Python/JAX and compiled shared objects.

### 3) Attention APIs and autodiff integration

- Dense attention: `jax_aiter/mha/mha.py`
  - Public API: `flash_attn_func(...)`
  - Uses `@jax.custom_vjp` to keep forward/backward under JAX autodiff.
  - Pads head dimensions to multiples of 8 when needed.
  - Dispatches between:
    - Standard CK-style path (`MhaFwdJA`, `MhaBwdJA`)
    - FMHA v3 path (`FmhaV3FwdJA`, `FmhaV3BwdJA`)
- Variable-length attention: `jax_aiter/mha/mha_varlen.py`
  - Public API: `flash_attn_varlen(...)`
  - Also uses `@jax.custom_vjp`.
  - Accepts `cu_seqlens_*`-based unpadded layouts and supports paged/block-table
    style inputs.
  - Dispatches between varlen CK and varlen FMHA v3 kernels.

Both modules cache `jax.jit(jax.ffi.ffi_call(...))` wrappers by signature to
avoid recreating call objects repeatedly.

### 4) Additional operator wrappers

- `jax_aiter/gemm_a8w8/asm_gemm_a8w8.py`
  - Quantized GEMM wrapper (`int8/int8` + scales -> bf16).
  - Reads tuned config from CSV via `ja_compat.tuning`.
  - Registers and invokes `GemmA8W8JA` FFI target.
- `jax_aiter/wv_splitkq/wv_splitkq.py`
  - FP8 split-KQ wrapper.
  - Registers and invokes `WvSplitKQJA`.

### 5) Hardware and dtype compatibility helpers

- `jax_aiter/ja_compat/chip_info.py`
  - Detects GPU arch (`gfx*`) and CU count using `rocminfo`.
- `jax_aiter/ja_compat/dtypes.py`
  - Normalizes dtype choices (including arch-dependent FP8 type).
- `jax_aiter/ja_compat/tuning.py`
  - Loads tuned shape configs (for ASM GEMM) from AITER config CSV files.

## Dense Attention Execution Flow

For `flash_attn_func(q, k, v, ...)`:

1. Normalize defaults (`softmax_scale`, `window_size`, optional tensors).
2. Pad `q/k/v` head dimensions to align with kernel constraints.
3. Run capability checks for FMHA v3 eligibility (dtype, dims, dropout, bias,
   masking mode, architecture constraints).
4. Select kernel family (FMHA v3 vs CK-style MHA).
5. Call jitted FFI function.
6. Unpad output to original head dimensions.
7. During backward, `custom_vjp` executes dispatch-aware backward kernels and
   returns gradients for JAX autodiff.

## Varlen Attention Execution Flow

For `flash_attn_varlen(...)`:

1. Inputs are unpadded token-major tensors + cumulative sequence lengths.
2. Optional physical padding metadata may be passed (`cu_seqlens_*_padded`).
3. Dispatch logic selects varlen FMHA v3 when constraints allow; otherwise
   uses varlen CK kernels.
4. Forward returns output and optional LSE/probability tensors.
5. Backward consumes saved residuals and returns token-major gradients.

## Build and Packaging Model

### Build artifacts

The runtime expects two primary shared-object roots:

- `build/aiter_build/*.so` (AITER-produced modules)
- `build/jax_aiter_build/*.so` (JAX-facing bridge modules, incl. umbrella lib)

### Build orchestration

- `make`
  - Produces umbrella JAX-AITER library (`libjax_aiter.so`) and JA bridge mods.
- `python jax_aiter/jit/build_jit.py [...]`
  - Reuses AITER JIT logic with JAX-specific patching.
  - Reads module definitions from `jax_aiter/jit/optCompilerConfig.json`.
  - Redirects JIT outputs to `build/aiter_build`.
  - Injects include/link settings needed by JAX FFI and static PyTorch bits.

### Packaging

- `setup.py`
  - Copies built `.so` files into `jax_aiter/_lib/...` during build/develop.
  - Copies HSA kernel assets into `jax_aiter/_hsa/...`.
  - Marks wheel as non-pure because native binaries are bundled.

## Configuration Surfaces

Common environment variables:

- `JA_ROOT_DIR`: enables local/dev artifact lookup under `build/`.
- `GPU_ARCHS`: architecture selection (for build/runtime behavior).
- `AITER_ASM_DIR`: explicit kernel code object directory override.
- `CU_NUM`: optional CU override for specific operators.

## Testing Strategy (Current Repository)

- `tests/test_mha_ja.py`: dense attention correctness + gradient checks against
  JAX baseline implementation.
- `tests/test_mha_varlen_ja.py`: varlen correctness + gradient checks.
- `tests/test_mha_torch_ja.py`: PyTorch parity checks using DLPack conversion.
- `tests/test_gemm_a8w8_ja.py`: quantized GEMM and split-KQ parity/perf checks.

The test suite primarily validates numerical parity and dispatch behavior under
many shape/dtype/masking combinations.

## Repository Map (Design-Relevant)

- `jax_aiter/ffi/` - FFI registration and symbol loading.
- `jax_aiter/mha/` - dense and varlen attention APIs + custom VJP logic.
- `jax_aiter/gemm_a8w8/` - quantized ASM GEMM wrapper.
- `jax_aiter/wv_splitkq/` - FP8 split-KQ wrapper.
- `jax_aiter/ja_compat/` - architecture/dtype/tuning helpers.
- `jax_aiter/jit/` - JIT build integration with AITER tooling.
- `tests/` - parity/correctness/perf-oriented verification.

## Known Constraints

- ROCm-focused path (`platform="ROCM"` in FFI registration).
- Native shared libraries and HSA assets must exist in expected build/package
  locations.
- Some advanced combinations (for example certain varlen bias-gradient paths)
  are intentionally limited in current tests/implementation.
