# JAX-AITER Design Overview

This document explains how `jax-aiter` is structured, how a call flows from Python to ROCm kernels, and where to extend the system safely.

## 1. Goals

`jax-aiter` integrates AMD AITER kernels with JAX through XLA FFI:

- JAX-native APIs (`flash_attn_func`, `flash_attn_varlen`, `rms_norm`, `rms_norm_with_add`)
- ROCm execution with zero-copy device buffers across the Python/C++ boundary
- Gradient support through `custom_vjp` wrappers
- No runtime PyTorch dependency in the Python API

Current CI-covered ops are:

- Flash Attention (forward + backward, batch + varlen)
- RMSNorm forward (AITER kernel) + backward (JAX math)

## 2. Layered architecture

```text
User JAX code
  -> jax_aiter.mha / jax_aiter.rmsnorm (Python APIs, custom_vjp)
    -> jax.ffi.ffi_call(...) (JAX FFI call sites)
      -> jax_aiter.ffi.registry (lazy loading + symbol registration)
        -> build/jax_aiter_build/*.so thin bridge modules
          -> AITER kernels from build/aiter_build/*.so + HSA blobs
            -> ROCm GPU execution
```

Key modules:

- `jax_aiter/mha/mha.py`: unified MHA API and VJP plumbing
- `jax_aiter/rmsnorm/rmsnorm.py`: RMSNorm API and VJP plumbing
- `jax_aiter/ffi/registry.py`: dynamic loading and `jax.ffi.register_ffi_target`
- `csrc/ffi/*/*.cu`: XLA FFI bridge handlers and argument packing
- `jax_aiter/jit/build_jit.py`: JIT build orchestration for AITER modules
- `jax_aiter/jit/optCompilerConfig.json`: module build config

## 3. Runtime data flow

### 3.1 Flash Attention path

1. Python API (`flash_attn_func` / `flash_attn_varlen`) normalizes shapes and options.
2. Head dims are padded to multiples of 8 when needed; outputs are sliced back.
3. A cached `jax.ffi.ffi_call` is built with static attrs (dropout, causal/window, kernel toggles, etc.).
4. `register_ffi_target("MhaFwdUnifiedJA" or "MhaBwdUnifiedJA")` ensures symbols are loaded once.
5. C++ bridge (`csrc/ffi/mha_fwd/mha_fwd_ja.cu`, `csrc/ffi/mha_bwd/mha_bwd_ja.cu`) validates inputs and packs `aiter::mha_*_args`.
6. AITER executes CK/ASM kernel paths internally; results are returned to JAX buffers.
7. Backward uses `custom_vjp` residuals and calls unified backward FFI.

Design choice: Python avoids CK-vs-ASM branching logic and delegates dispatch to AITER internals, keeping API logic simpler.

### 3.2 RMSNorm path

1. Python API calls `register_ffi_target("RmsnormFwdJA")`.
2. `jax.ffi.ffi_call` invokes the C++ bridge in `csrc/ffi/rmsnorm/rmsnorm_fwd_ja.cu`.
3. Bridge flattens leading dims to 2D (`m x n`) and calls `rmsnorm2d_fwd`.
4. Backward is implemented in Python/JAX (`custom_vjp`) because a CK backward kernel is not wired here yet.

## 4. Build and packaging model

Artifacts are produced in two stages plus packaging:

1. **Umbrella + thin bridge libs** (`make`, `make ja_mods`)
   - `build/jax_aiter_build/libjax_aiter.so`
   - `build/jax_aiter_build/mha_fwd_ja.so`, `mha_bwd_ja.so`, `rmsnorm_fwd_ja.so`
2. **AITER JIT modules** (`python jax_aiter/jit/build_jit.py`)
   - `build/aiter_build/libmha_fwd.so`, `libmha_bwd.so`, `librmsnorm_fwd.so`
3. **Wheel/install copy step** (`setup.py`)
   - Copies `.so` files into `jax_aiter/_lib/**`
   - Copies HSA kernel assets into `jax_aiter/_hsa/**`

At runtime, `ffi/registry.py` loads umbrella first, then thin modules from:

- `<JA_ROOT_DIR>/build/{aiter_build,jax_aiter_build}` in dev mode
- packaged `jax_aiter/_lib/**` in installed mode

## 5. Configuration and environment

Important env vars:

- `JA_ROOT_DIR`: repository root for dev/build mode
- `GPU_ARCHS`: target gfx architectures (e.g., `gfx942`, `gfx950`)
- `AITER_ASM_DIR`: base HSA directory (set automatically if not provided)

`jax_aiter/__init__.py` calls `set_aiter_asm_dir()` early to align runtime assembly-kernel lookup with AITER conventions.

## 6. Contracts and constraints

- Platform: ROCm only (`platform="ROCM"` for FFI registration)
- MHA/RMSNorm input dtypes: fp16 and bf16 in current bridge implementations
- MHA head dims: bridge enforces `<=256` and multiple-of-8 at FFI boundary (Python pads and unpads as needed)
- Varlen mode is inferred by rank (3D packed tensors) in unified MHA bridges

## 7. Testing strategy

Primary CI tests:

- `tests/test_mha_ja.py`: large matrix of forward/backward configs, dropout, SWA, bias/ALiBi, varlen, regression guards
- `tests/test_rmsnorm_ja.py`: forward/backward accuracy vs JAX refs, fused add+RMSNorm coverage

CI flow in `.github/workflows/ci.yml` builds all required libraries, installs the package, runs smoke import, then executes test suites.

## 8. Extension guide (adding a new FFI op)

When adding an op, keep changes inside this repository:

1. Add C++ bridge handler under `csrc/ffi/<op>/...` and export symbol via `XLA_FFI_DEFINE_HANDLER_SYMBOL`.
2. Build the bridge `.so` from the `Makefile`.
3. Add/ensure required AITER JIT module config in `jax_aiter/jit/optCompilerConfig.json`.
4. Expose Python wrapper with `jax.ffi.ffi_call` and (if needed) `custom_vjp`.
5. Register symbol-to-module mapping in `jax_aiter/ffi/registry.py`.
6. Add tests under `tests/` and include them in CI as appropriate.

This keeps the integration boundary explicit: Python API -> FFI symbol -> bridge .so -> AITER kernel library.
