# JAX-AITER Design Overview

## Purpose

JAX-AITER integrates AMD's AITER GPU kernels into JAX through XLA FFI so that
JAX programs can call high-performance ROCm kernels with native JAX APIs and
autodiff support.

At a high level, the project provides:

- Python user-facing ops (`jax_aiter/*`) such as MHA, RMSNorm, and GEMM.
- A runtime FFI registry that loads shared objects and registers FFI symbols.
- C++/HIP FFI bridge handlers (`csrc/ffi/*`) that translate XLA buffers/attrs
  into AITER kernel argument structs and kernel launches.
- Build tooling for both umbrella/shared FFI modules and AITER JIT modules.

---

## Repository design (major areas)

- `jax_aiter/`
  - Python public API and wrappers around `jax.ffi.ffi_call`.
  - Custom gradients via `jax.custom_vjp` for trainable ops.
  - Runtime compatibility helpers (`ja_compat/`) for GPU detection and paths.
- `csrc/common/`
  - Shared C++/HIP utilities and headers used by FFI modules.
- `csrc/ffi/`
  - One FFI bridge implementation per op family (`mha_*`, `gemm_*`, `rmsnorm`).
- `jax_aiter/jit/`
  - JIT build integration with AITER's compilation system and config.
- `tests/`
  - Operator tests and GPU-arch-aware skip markers (`gfx942` vs `gfx950`).
- `Makefile`
  - Builds umbrella lib and per-op FFI shared objects (`make`, `make ja_mods`).
- `setup.py`
  - Copies built `.so` libraries and HSA `.co` files into package data.

---

## Runtime architecture

### 1) Import and environment bootstrap

When `jax_aiter` is imported:

1. `jax_aiter/__init__.py` calls `set_aiter_asm_dir()`.
2. `ja_compat/config.py` sets `AITER_ASM_DIR` if unset, selecting either:
   - development path: `<JA_ROOT_DIR>/third_party/aiter/hsa/`
   - installed path: `<site-packages>/jax_aiter/_hsa/`
3. GPU architecture detection (for sanity checks) comes from
   `ja_compat/chip_info.py`.

### 2) Lazy FFI registration model

Each Python op wrapper calls `register_ffi_target(...)` on first use.

`jax_aiter/ffi/registry.py` then:

1. Initializes JAX backend (`jax.devices()`), once.
2. Loads umbrella shared lib (`libjax_aiter.so`).
3. Loads thin op modules from:
   - `build/aiter_build/*.so`
   - `build/jax_aiter_build/*.so`
4. Resolves symbol pointers using `ctypes`.
5. Registers symbols via `jax.ffi.register_ffi_target(..., platform="ROCM")`.

This avoids eager loading all symbols at import time and keeps per-op setup
lightweight.

### 3) JAX call path

For a typical op call:

```text
Python API (jax_aiter/<op>) 
  -> jax.ffi.ffi_call(target_name, output_specs, attrs...)
  -> XLA executes FFI target on ROCm stream
  -> C++/HIP bridge in csrc/ffi/<op>/*_ja.cu
  -> AITER kernel entrypoint / ASM launcher
  -> output buffers returned to JAX
```

Attributes such as dropout probability, causal flags, and window sizes are
passed as FFI attrs, while tensors are passed as device buffers.

---

## Operator design patterns

### Pattern A: FFI forward + custom VJP backward

Used by trainable ops where backward must be defined explicitly.

- `mha/mha.py`:
  - forward and backward both call unified AITER bridges (`MhaFwdUnifiedJA`,
    `MhaBwdUnifiedJA`).
  - exposes `flash_attn_func` and `flash_attn_varlen` with `custom_vjp`.
  - handles shape padding to head-dim multiples required by kernels.
- `gemm/gemm.py`:
  - forward uses `GemmFwdJA`.
  - backward composes additional GEMM calls to compute `dA` and `dB`.
- `rmsnorm/rmsnorm.py`:
  - forward uses `RmsnormFwdJA`.
  - backward is implemented in JAX (no CK backward kernel yet).

### Pattern B: FFI forward-only inference kernels

Used by specialized quantized kernels with no backward wiring here:

- `gemm_fp8/gemm_fp8_mi350.py` -> `GemmFp8Mi350FwdJA`
- `gemm_i8/gemm_i8.py` -> `GemmI8FwdJA`
- `gemm_fp4/gemm_fp4.py` -> `GemmFp4FwdJA`
- `flatmm_fp8/flatmm_fp8.py` -> `FlatmmFp8FwdJA`

---

## FFI bridge design (C++/HIP)

Each bridge follows the same structure:

1. Validate tensor ranks/shapes/dtypes.
2. Parse XLA attrs and optional buffers.
3. Compute strides/layout metadata expected by backend kernel ABI.
4. Build kernel argument structs (for AITER APIs or direct ASM launchers).
5. Launch on provided HIP stream.
6. Return `ffi::Error::Success()` or a descriptive error.

Examples:

- `csrc/ffi/mha_fwd/mha_fwd_ja.cu`
  - Supports both batch and varlen by rank inspection (4D vs 3D).
  - Handles bias/ALiBi/dropout/rng state plumbing.
  - Calls `aiter::mha_fwd(args, stream_config)`.
- `csrc/ffi/mha_bwd/mha_bwd_ja.cu`
  - Handles deterministic and split accumulation paths.
  - Supports MQA/GQA expansion-and-reduction logic.
  - Calls `aiter::mha_bwd(args, stream_config)`.
- `csrc/ffi/gemm_fwd/gemm_fwd_ja.cu`
  - Uses generated ASM kernel config tables.
  - Heuristically selects kernel and split-K.
  - Launches with packed ABI-compatible argument struct.

---

## Build and packaging design

### Build stages

1. **Umbrella library**
   - `make`
   - Produces `build/jax_aiter_build/libjax_aiter.so`.
2. **JIT modules (AITER build integration)**
   - `python3 jax_aiter/jit/build_jit.py`
   - Uses `optCompilerConfig.json` and patched AITER core logic.
3. **FFI modules**
   - `make ja_mods`
   - Produces `*_ja.so` files in `build/jax_aiter_build`.

### Packaging behavior

`setup.py` copies runtime artifacts into package directories:

- `jax_aiter/_lib/**` for `.so` modules
- `jax_aiter/_hsa/**` for `.co` and related kernel data

This enables installed wheels to run without relying on repo-local build paths.

---

## Architecture constraints and assumptions

- ROCm-only runtime (`platform="ROCM"` for registered FFI targets).
- Python 3.12 requirement.
- GPU-specific kernels have arch constraints (`gfx942` vs `gfx950`).
- Several kernels require specific divisibility constraints (for example GEMM K
  alignment), validated in bridge or wrapper code.
- Tests are architecture-aware and skip unsupported paths automatically.

---

## Current extension points

- Add new FFI symbol:
  1. Implement bridge under `csrc/ffi/<new_op>/`.
  2. Build `.so` via `Makefile`.
  3. Add symbol-to-module entry in `ffi/registry.py`.
  4. Add Python wrapper in `jax_aiter/<new_op>/`.
  5. Add tests under `tests/`.
- Add new JIT-built module:
  - Extend `jax_aiter/jit/optCompilerConfig.json`.
  - Ensure `build_jit.py` patch logic includes required includes/ldflags.

---

## Notes on partially wired wrappers

There are Python wrappers in-tree (`gemm_a8w8`, `wv_splitkq`) that invoke
`register_ffi_target(...)`, but their symbols are not currently listed in
`ffi/registry.py`'s `SYMBOL_TO_MODULE_MAP` and corresponding FFI modules are not
built by the top-level `Makefile` targets shown above. Treat these as
in-progress integrations unless/until they are fully wired.
