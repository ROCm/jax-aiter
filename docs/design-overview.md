# JAX-AITER Design Overview

This document explains how `jax-aiter` is structured and how a user API call reaches ROCm kernels through JAX FFI.

## 1. Purpose and Scope

`jax-aiter` provides JAX-native wrappers around AITER GPU kernels, with:

- JAX-facing Python APIs (`flash_attn_func`, `flash_attn_varlen`, quantized GEMM wrappers).
- Runtime kernel dispatch between CK and FMHA-v3 variants.
- Custom VJP integration for differentiable training paths.
- A C++/HIP FFI bridge layer that binds XLA FFI symbols to AITER implementations.

Primary areas in this repository:

- `jax_aiter/` - Python APIs, dispatch logic, FFI registration, compatibility helpers.
- `csrc/` - XLA FFI bridge handlers and shared HIP/C++ utilities.
- `tests/` - correctness and smoke tests.
- `benchmarks/` - performance comparison tools.

## 2. High-Level Architecture

```text
User JAX code
    |
    v
jax_aiter Python API
  - mha/mha.py
  - mha/mha_varlen.py
  - gemm_a8w8 / wv_splitkq
    |
    v
FFI registry (jax_aiter/ffi/registry.py)
  - load umbrella lib
  - load module .so files
  - resolve symbol pointers
  - jax.ffi.register_ffi_target(...)
    |
    v
jax.ffi.ffi_call(...)
    |
    v
XLA FFI handlers in csrc/ffi/*.cu
  - validate buffers/attrs
  - build kernel args
  - call aiter::* ops
    |
    v
AITER kernels on ROCm
```

Design pattern: keep Python layer focused on API shape/dtype semantics and dispatch policy, while native bridge layer handles low-level validation, stream-aware launch setup, and calls into AITER.

## 3. Component Responsibilities

### 3.1 Python package (`jax_aiter/`)

- `jax_aiter/__init__.py`
  - Sets `AITER_ASM_DIR` early via `set_aiter_asm_dir()`.
  - Exposes lazy-loaded submodules.

- `jax_aiter/mha/mha.py`
  - Fixed-length MHA API (`flash_attn_func`) and custom VJP.
  - Dispatches between `Mha*` and `FmhaV3*` kernels based on constraints.
  - Pads head dimensions to multiples of 8, then unpads outputs/gradients.

- `jax_aiter/mha/mha_varlen.py`
  - Variable-length API (`flash_attn_varlen`) and custom VJP.
  - Uses `cu_seqlens_*` and max/min sequence metadata.
  - Dispatches between `MhaVarlen*` and `FmhaV3Varlen*` paths.

- `jax_aiter/ffi/registry.py`
  - Central loader/registrar for FFI symbols.
  - Loads umbrella library (`libjax_aiter.so`) first, then thin modules from:
    - `build/aiter_build/*.so`
    - `build/jax_aiter_build/*.so`
  - Resolves symbol pointer and registers it with `jax.ffi.register_ffi_target`.

- `jax_aiter/ja_compat/`
  - `config.py`: resolves build/package library roots and runtime HSA path setup.
  - `chip_info.py`: GPU arch and CU discovery from ROCm tooling.
  - `dtypes.py`: device-aware dtype aliases and conversion helpers.
  - `tuning.py`: reads tuned GEMM CSV config tables.

- `jax_aiter/gemm_a8w8/asm_gemm_a8w8.py`
  - Wrapper for `GemmA8W8JA` with tuned tile/splitK config lookup.

- `jax_aiter/wv_splitkq/wv_splitkq.py`
  - Wrapper for `WvSplitKQJA` FP8 path.

### 3.2 Native bridge layer (`csrc/`)

- `csrc/ffi/*/*.cu`
  - Defines XLA FFI handler symbols via `XLA_FFI_DEFINE_HANDLER_SYMBOL`.
  - Examples:
    - `MhaFwdJA`, `MhaBwdJA`
    - `FmhaV3FwdJA`, `FmhaV3BwdJA`
    - `MhaVarlenFwdJA`, `MhaVarlenBwdJA`
    - `FmhaV3VarlenFwdJA`, `FmhaV3VarlenBwdJA`
    - `GemmA8W8JA`, `WvSplitKQJA`

- `csrc/common/mha_common_utils.*`
  - Shared utility logic used by attention bridges (RNG state setup, reduction helpers, stream config helpers).

### 3.3 Build and packaging

- `Makefile`
  - Builds:
    - `build/jax_aiter_build/libjax_aiter.so` (umbrella library)
    - per-op JAX FFI modules (`*_ja.so`) in `build/jax_aiter_build/`.

- `jax_aiter/jit/build_jit.py`
  - Patches AITER JIT core behavior for JAX-AITER layout.
  - Redirects user JIT output to `build/aiter_build`.
  - Injects JAX FFI and local include/link settings from `optCompilerConfig.json`.

- `setup.py`
  - Copies built `.so` files into `jax_aiter/_lib/{aiter_build,jax_aiter_build}`.
  - Copies HSA kernel assets into `jax_aiter/_hsa`.

## 4. Runtime Flow: `flash_attn_func`

1. User calls `jax_aiter.mha.flash_attn_func(...)`.
2. Python layer normalizes defaults (e.g., `softmax_scale`), pads head dims if needed.
3. Forward path runs `_flash_attn_forward(...)`, which decides CK vs FMHA-v3 kernel family.
4. Selected wrapper (`mha_fwd` or `fmha_v3_fwd`) ensures FFI target is registered:
   - `register_ffi_target("MhaFwdJA" | "FmhaV3FwdJA", "ROCM")`.
5. FFI call executes (`jax.ffi.ffi_call(...)`) with static attrs (dropout, causal, window sizes, etc.).
6. Native bridge validates dimensions/types and calls into `aiter::mha_*`.
7. Custom VJP backward path dispatches similarly and returns gradients (`dq`, `dk`, `dv`, optional `dbias`).

Varlen follows the same layered pattern but uses cumulative sequence tensors and varlen symbols.

## 5. Dispatch Strategy

Dispatch is explicit and conservative:

- FMHA-v3 paths are selected only when shape/dtype/masking constraints are met.
- CK paths are fallback for unsupported combinations (including many padded or feature-rich cases).
- Both fixed-length and varlen paths normalize window semantics and guard unsupported cases.

This strategy favors correctness and broad compatibility first, then performance via optimized kernels when safe.

## 6. FFI Registration and Symbol Resolution

`jax_aiter/ffi/registry.py` uses a symbol-to-module mapping (`SYMBOL_TO_MODULE_MAP`) to:

- determine which `.so` should contain each symbol,
- ensure libraries are loaded in the right order,
- resolve function pointer with `ctypes`,
- register the pointer with JAX as an FFI target.

Registration is lazy and idempotent (per symbol), which avoids expensive startup work at import time.

## 7. Build Artifacts and Expected Layout

During development builds:

- `build/aiter_build/*.so` - modules produced by AITER JIT path.
- `build/jax_aiter_build/libjax_aiter.so` - umbrella shared library.
- `build/jax_aiter_build/*_ja.so` - JAX FFI bridge modules.

In packaged installs:

- `jax_aiter/_lib/aiter_build/*.so`
- `jax_aiter/_lib/jax_aiter_build/*.so`
- `jax_aiter/_hsa/<gfx>/*` (HSA kernel files/assets)

Library root selection is controlled by:

- `JA_ROOT_DIR` for local builds, or
- installed package resources when running from wheel/site-packages.

## 8. Testing and Validation Strategy

Testing is focused on numerical correctness versus pure-JAX references:

- `tests/test_mha_ja.py`
  - broad parameterized coverage for forward/backward, dtype/layout variants, causal/local, padding via `cu_seqlens`.
- `tests/test_mha_varlen_ja.py`
  - varlen-specific coverage and gradient comparisons.
- `tests/smoke_mha_test.py`
  - fast, sharded smoke runner for representative subsets.
- `benchmarks/benchmark_mha.py`
  - compares runtime of JAX-AITER vs baseline reference implementation.

## 9. Extending the System (New Op)

Typical integration sequence:

1. Implement native bridge in `csrc/ffi/<op>/<op>_ja.cu`.
2. Export symbol with `XLA_FFI_DEFINE_HANDLER_SYMBOL`.
3. Add build target in `Makefile` and/or JIT config (`optCompilerConfig.json`).
4. Map symbol to module in `jax_aiter/ffi/registry.py`.
5. Add Python wrapper that:
   - ensures target registration,
   - defines `jax.ffi.ffi_call(...)`,
   - handles shape/dtype conventions.
6. Add correctness tests and, if relevant, benchmark coverage.

This keeps the same layered contract and minimizes coupling between user API and kernel internals.
