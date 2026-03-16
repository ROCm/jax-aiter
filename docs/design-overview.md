# JAX-AITER Design Overview

This document explains how `jax-aiter` is structured, how data flows from Python to ROCm kernels, and how to extend the project with new operators.

## 1) Goals and scope

`jax-aiter` integrates ROCm-optimized AITER kernels with JAX using XLA FFI. The design focuses on:

- Exposing AITER kernels through JAX-native Python APIs.
- Zero-copy GPU execution through XLA FFI buffers.
- Training support via `jax.custom_vjp` where backward is available.
- No PyTorch dependency at runtime for end users.

This repository contains the JAX integration layer. AITER kernel implementations remain in AITER-generated modules and kernel artifacts.

## 2) High-level architecture

At runtime, the stack is:

1. Python API layer (`jax_aiter/<op>/<op>.py`)
2. FFI registry and symbol loader (`jax_aiter/ffi/registry.py`)
3. XLA FFI bridge handlers (`csrc/ffi/*/*_ja.cu`)
4. AITER kernels (CK/ASM) and `.co` artifacts

```text
JAX program
  -> jax_aiter Python op (custom_vjp / jit wrapper)
  -> register_ffi_target("...JA")
  -> jax.ffi.ffi_call(...)
  -> XLA FFI handler symbol in *_ja.so
  -> aiter::... kernel launch path
  -> ROCm GPU kernel (.co / CK / ASM)
```

## 3) Repository layout (design-relevant)

- `jax_aiter/`
  - Public Python APIs (`mha`, `rmsnorm`, `gemm`, `gemm_fp8`, `gemm_i8`, etc.)
  - Compatibility/config helpers in `ja_compat/`
  - FFI module loader + JAX FFI target registration in `ffi/registry.py`
  - JIT build integration in `jit/`
- `csrc/`
  - `ffi/*`: C++/HIP XLA FFI handlers for each op
  - `common/*`: shared helpers (logging, stream/device helpers, utilities)
- `build/` (generated)
  - `aiter_build/*.so` (AITER JIT modules)
  - `jax_aiter_build/*.so` (JAX FFI bridge modules + umbrella library)
- `third_party/aiter/hsa/` (kernel artifacts used in development mode)

## 4) Runtime design

### 4.1 Initialization and environment

On import, `jax_aiter/__init__.py` calls `set_aiter_asm_dir()` (`jax_aiter/ja_compat/config.py`) to set `AITER_ASM_DIR` if the user did not set it:

- Dev mode (`JA_ROOT_DIR` set): `${JA_ROOT_DIR}/third_party/aiter/hsa/`
- Installed package mode: `${site-packages}/jax_aiter/_hsa/`

This keeps assembly kernel lookup consistent between source builds and wheels.

### 4.2 FFI symbol loading and registration

`jax_aiter/ffi/registry.py` is the central loader/registrar:

1. Loads umbrella shared library (`libjax_aiter.so`) first.
2. Loads module `.so` files from:
   - `build/aiter_build/`
   - `build/jax_aiter_build/`
3. Maps JAX target names (for example `MhaFwdUnifiedJA`) to module files.
4. Resolves symbol pointers and registers them with `jax.ffi.register_ffi_target(...)`.

Registration is lazy and cached, so each target is only registered once per process.

### 4.3 Python op layer pattern

Most ops follow this pattern:

- Ensure FFI target registration (`register_ffi_target(...)`).
- Build a `jax.ffi.ffi_call(...)` with output shape/dtype contracts.
- Wrap invocation in `jax.jit` for compiled execution.
- Optionally define gradients with `jax.custom_vjp`.

Examples:

- `mha/mha.py`: forward/backward both through unified FFI handlers.
- `rmsnorm/rmsnorm.py`: forward via FFI, backward computed in JAX.
- `gemm/gemm.py`: forward and backward both routed through GEMM FFI calls.

### 4.4 Native FFI handler responsibilities

Each `csrc/ffi/*/*_ja.cu` handler:

- Accepts XLA buffers + attrs from FFI binding macros.
- Validates tensor rank, dtype, and shape constraints.
- Computes/normalizes strides and metadata.
- Packs operator-specific argument structures.
- Launches AITER kernel path (CK/ASM), often with architecture-specific heuristics.
- Returns `ffi::Error::Success()` or detailed errors.

Examples:

- `mha_fwd_ja.cu`: batch/varlen detection by rank, mask handling, RNG/dropout wiring, then `aiter::mha_fwd(...)`.
- `gemm_fwd_ja.cu`: kernel selection heuristic and launch for BF16 `A @ B^T`.

## 5) Build and packaging design

Build is split into two artifact groups:

1. **AITER JIT modules** (operator backends)
2. **JAX FFI bridge modules** (XLA entry points)

### 5.1 Build flow

Typical sequence:

1. `make`
   - Builds umbrella lib: `build/jax_aiter_build/libjax_aiter.so`
2. `python3 jax_aiter/jit/build_jit.py`
   - Builds AITER modules into `build/aiter_build/`
   - Patches AITER build config using `jax_aiter/jit/optCompilerConfig.json`
3. `make ja_mods`
   - Builds bridge modules such as `mha_fwd_ja.so`, `gemm_fwd_ja.so`, etc.

### 5.2 Packaging flow

`setup.py` copies runtime artifacts into the package:

- `.so` files to `jax_aiter/_lib/{aiter_build,jax_aiter_build}`
- kernel artifacts from `third_party/aiter/hsa` to `jax_aiter/_hsa`

Runtime lookup in `ja_compat/config.py` then works the same for installed wheels.

## 6) Operator design patterns

The repository currently uses three common differentiation patterns:

1. **FFI forward + FFI backward**
   - Example: Flash Attention
2. **FFI forward + JAX backward**
   - Example: RMSNorm
3. **FFI forward reused in backward math**
   - Example: GEMM gradient routes through the same GEMM FFI entry point

This makes it possible to ship performant kernels incrementally while maintaining a consistent Python API.

## 7) Adding a new operator (design checklist)

1. Add native FFI bridge in `csrc/ffi/<op>/..._ja.cu`
2. Export symbol with `XLA_FFI_DEFINE_HANDLER_SYMBOL(...)`
3. Add module build target in `Makefile` (and generated config if needed)
4. Add symbol mapping in `jax_aiter/ffi/registry.py` (`SYMBOL_TO_MODULE_MAP`)
5. Add Python API wrapper under `jax_aiter/<op>/`
6. Add `custom_vjp` if gradient is required
7. Add tests in `tests/` for forward + backward behavior

## 8) Design tradeoffs and constraints

- Runtime intentionally avoids PyTorch coupling; some build paths still reuse AITER/PyTorch headers for compilation compatibility.
- Many kernels are architecture-specific (`gfx942` vs `gfx950`), so guardrails and fallback behavior are required.
- FFI contracts must stay in strict sync between Python call signatures and native bindings.
- Some ops currently provide forward-only support on specific architectures.

## 9) Debugging notes

- If symbols fail to register, inspect loaded modules and symbol map in `ffi/registry.py`.
- If kernels are missing, verify `AITER_ASM_DIR` and that expected `.co` files are present.
- If shape issues occur, check native handlers first; validation is implemented there and errors are returned through FFI.

---

For user-facing setup and supported-op details, see the root `README.md`.
