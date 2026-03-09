# JAX-AITER Design Overview

This document explains how `jax-aiter` is structured, how execution flows from Python into ROCm kernels, and where to extend the system.

## Goals

- Expose AITER operators as JAX-native functions.
- Keep runtime overhead low via JAX FFI and direct device-buffer interop.
- Support training by wiring custom forward/backward paths (`custom_vjp`).
- Preserve access to architecture-specific ROCm assembly kernels.

## High-level architecture

`jax-aiter` is organized in four runtime layers:

1. **Python API layer (`jax_aiter/*`)**
   - Public APIs such as `flash_attn_func` and `flash_attn_varlen`.
   - Input normalization (padding head dims to multiples of 8, window normalization).
   - Kernel capability checks and dispatch decisions.
2. **JAX autodiff + FFI call layer**
   - Forward/backward functions are wrapped with `jax.custom_vjp`.
   - Actual GPU calls are emitted via `jax.ffi.ffi_call(...)`, then `jax.jit(...)`.
3. **FFI registry/load layer (`jax_aiter/ffi/registry.py`)**
   - Lazily loads shared objects and registers symbols into JAX (`register_ffi_target`).
   - Resolves symbol -> module mapping and tracks registration state.
4. **Native bridge and kernel layer (`csrc/ffi/*`, AITER JIT outputs)**
   - Thin JA bridge `.so` modules expose C symbols used by JAX FFI.
   - AITER-generated modules provide backend kernel implementations.

## Repository map (design-relevant paths)

- `jax_aiter/mha/mha.py`: fixed-length attention API, dispatch, custom VJP.
- `jax_aiter/mha/mha_varlen.py`: variable-length attention API and VJP.
- `jax_aiter/ffi/registry.py`: dynamic loading and JAX FFI target registration.
- `jax_aiter/ja_compat/config.py`: build/install path resolution; `AITER_ASM_DIR` setup.
- `jax_aiter/ja_compat/chip_info.py`: GPU arch detection (`gfx942`, `gfx950`, etc.).
- `jax_aiter/jit/build_jit.py`: orchestrates AITER JIT builds using JA config.
- `jax_aiter/jit/optCompilerConfig.json`: module-level compiler/build configuration.
- `csrc/common/*`: shared native utilities and umbrella library sources.
- `csrc/ffi/*/*_ja.cu`: per-operator JA bridge implementations.
- `Makefile`: umbrella + JA module compilation entry points.
- `setup.py`: wheel packaging of `.so` and HSA artifacts.

## End-to-end execution flow

The common attention flow:

1. User calls `flash_attn_func(...)` or `flash_attn_varlen(...)`.
2. Python layer validates/pads shapes and computes normalized runtime parameters.
3. Dispatch checks decide standard MHA vs FMHA v3 kernels.
4. Wrapper ensures FFI symbol registration (`register_ffi_target(...)`).
5. Registry loads:
   - umbrella library (`build/jax_aiter_build/libjax_aiter.so`)
   - AITER modules (`build/aiter_build/*.so`)
   - JA thin modules (`build/jax_aiter_build/*.so`)
6. JAX executes `ffi_call` in compiled graph form.
7. Backward path reuses saved residuals and dispatches its own kernel choice.

## Kernel dispatch strategy

The attention modules use capability predicates before selecting kernels. Typical checks include:

- dtype constraints (for example bf16-only paths),
- head dimension restrictions,
- causal/windowed mode compatibility,
- dropout and bias support,
- architecture checks (`gfx942` vs `gfx950`),
- sequence-shape constraints (including varlen/padded behavior).

When constraints are not met, code falls back to broader-coverage kernels.

## Build and artifact model

There are two main native build products:

1. **Umbrella + JA bridge modules (`make`, `make ja_mods`)**
   - Output directory: `build/jax_aiter_build/`
   - Includes `libjax_aiter.so` and JA bridge `.so` modules.
2. **AITER JIT modules (`python jax_aiter/jit/build_jit.py`)**
   - Output directory: `build/aiter_build/`
   - Includes per-kernel modules expected by registry loading.

At packaging time (`setup.py`):

- build outputs are copied into `jax_aiter/_lib/**`,
- architecture HSA files are copied into `jax_aiter/_hsa/**`,
- runtime path resolution in `ja_compat/config.py` supports both:
  - local dev mode (`JA_ROOT_DIR`),
  - installed wheel mode.

## Configuration and environment contracts

Important environment variables:

- `JA_ROOT_DIR`: repository root for dev-mode path resolution and build scripts.
- `GPU_ARCHS`: architecture selection and autodetection hints.
- `AITER_ASM_DIR`: location of architecture-specific assembly kernels.
  - If not provided, it is derived at import time from detected GPU and mode.

## Extending the design (new operator/kernel)

To add a new operator end-to-end:

1. Add native bridge source under `csrc/ffi/<op>/<op>_ja.cu`.
2. Add build rule and output target in `Makefile` (JA module side).
3. Add AITER JIT build config in `jax_aiter/jit/optCompilerConfig.json` if needed.
4. Add symbol mapping in `jax_aiter/ffi/registry.py` (`SYMBOL_TO_MODULE_MAP`).
5. Add Python wrapper that:
   - calls `register_ffi_target(<Symbol>, "ROCM")`,
   - defines cached `jax.ffi.ffi_call`,
   - wraps in `jax.jit` and (if needed) `custom_vjp`.
6. Export through package `__init__.py` and add tests.

## Design trade-offs

- **Pros**
  - Good separation between Python API, symbol registration, and native kernels.
  - Lazy loading/registration keeps startup overhead down.
  - Supports both source builds and packaged runtime artifacts.
- **Costs**
  - Multiple artifact types and directories increase build complexity.
  - Kernel dispatch logic is necessarily hardware- and shape-specific.
  - Tight coordination is needed between symbol names, build outputs, and registry map.
