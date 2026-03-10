# JAX-AITER Design Overview

This document explains how `jax-aiter` is structured and how execution flows from a JAX API call to ROCm kernel execution.

## 1) Goals and scope

`jax-aiter` integrates AMD AITER operators into JAX through:

- A JAX-native Python API
- JAX FFI registration and invocation
- `custom_vjp` wrappers for training (forward + backward)
- ROCm-oriented kernel dispatch and packaging

Primary operator families in this repository:

- Attention:
  - dense/padded flash attention (`jax_aiter.mha.flash_attn_func`)
  - variable-length flash attention (`jax_aiter.mha.flash_attn_varlen`)
- GEMM:
  - int8/weight-shuffled path (`jax_aiter.gemm_a8w8.gemm_a8w8_ASM`)
  - split-KQ fp8 path (`jax_aiter.wv_splitkq.wv_splitkq_fp8`)

## 2) Repository architecture

Top-level layout:

- `jax_aiter/`: Python package, public APIs, dispatch, FFI registry, compatibility/config.
- `csrc/`: Native bridge code (XLA FFI handlers) for JAX <-> AITER.
- `tests/`: Correctness and integration tests (including sharded/multi-GPU support).
- `benchmarks/`: Performance comparisons (JAX baseline vs JAX-AITER).
- `scripts/`, `Makefile`, `setup.py`: Build and packaging orchestration.

Within the Python package:

- `jax_aiter/mha/`: Attention APIs + dispatch logic + custom VJP.
- `jax_aiter/ffi/registry.py`: Shared library loading + symbol resolution + `jax.ffi.register_ffi_target`.
- `jax_aiter/gemm_a8w8/`, `jax_aiter/wv_splitkq/`: GEMM wrappers using FFI.
- `jax_aiter/ja_compat/`: Environment/config helpers (library roots, GPU arch, dtype helpers, tuning data).
- `jax_aiter/jit/`: JIT module build integration for AITER components.
- `jax_aiter/baseline/`: Reference implementation used by tests/benchmarks.

## 3) Layered runtime model

At runtime, the project is organized as four layers:

1. **JAX API layer**
   - User-facing functions are called from Python (`flash_attn_func`, `flash_attn_varlen`, GEMM wrappers).

2. **Dispatch/autodiff layer**
   - Input normalization, capability checks, kernel selection, and `custom_vjp` forward/backward wiring.

3. **FFI registry layer**
   - `jax_aiter.ffi.registry` loads shared objects, resolves symbol pointers, and registers FFI targets with JAX.

4. **Native bridge layer**
   - XLA FFI C++/CUDA handlers in `csrc/ffi/*` validate inputs, marshal arguments, and call into AITER implementations.

## 4) End-to-end execution flow

### 4.1 Import-time setup

When `jax_aiter` is imported:

- `jax_aiter/__init__.py` calls `set_aiter_asm_dir()` early.
- `jax_aiter/ja_compat/config.py` determines `AITER_ASM_DIR` from:
  - development mode (`JA_ROOT_DIR` set), or
  - installed-package mode (`jax_aiter/_hsa/<gfx_arch>/`).

This ensures architecture-specific kernel assets are locatable before operators execute.

### 4.2 Attention path (`flash_attn_func`)

High-level sequence:

1. User calls `flash_attn_func(...)`.
2. Python dispatch logic in `jax_aiter/mha/mha.py`:
   - validates/normalizes shapes and options,
   - handles head-dim padding when needed,
   - chooses implementation (for example CK vs FMHA v3) via capability checks.
3. Wrapper ensures FFI target registration (`register_ffi_target(...)`).
4. `jax.ffi.ffi_call(...)` launches the selected target.
5. Native bridge symbol in `csrc/ffi/...` receives tensors/attrs, validates contract, and invokes AITER kernel entry points.
6. For backward pass, `custom_vjp` routes gradients through matching native backward kernels.

### 4.3 Variable-length attention path (`flash_attn_varlen`)

`jax_aiter/mha/mha_varlen.py` follows the same architecture with varlen-specific metadata (sequence boundary information and related attributes), plus custom VJP handling for gradient flow.

### 4.4 GEMM paths

- `gemm_a8w8_ASM` and `wv_splitkq_fp8` wrappers:
  - register one FFI target,
  - build a cached `ffi_call` for a given signature,
  - dispatch to corresponding bridge symbols.

## 5) FFI registration and shared-library loading

`jax_aiter/ffi/registry.py` centralizes dynamic loading and registration:

- Defines a symbol-to-module map (example: `MhaFwdJA` -> `ck_fused_attn_fwd_ja.so`).
- Loads umbrella library first (`libjax_aiter.so`) to preserve shared runtime context.
- Loads thin modules from:
  - `build/aiter_build/`
  - `build/jax_aiter_build/`
- Resolves symbol addresses with `ctypes` and registers them with JAX.
- Tracks loaded modules and already-registered targets to avoid redundant work.

This isolates linkage complexity from API modules and gives one place to inspect runtime registration status.

## 6) Build and packaging design

Build pipeline (source tree / CI):

1. Build static PyTorch components needed for linkage (`scripts/build_static_pytorch.sh`).
2. Build umbrella shared library (`make` -> `build/jax_aiter_build/libjax_aiter.so`).
3. Build AITER JIT modules (`python jax_aiter/jit/build_jit.py`).
4. Build JAX-AITER frontend bridge modules (`make ja_mods`).

Packaging pipeline (`setup.py`):

- Copies built shared objects into `jax_aiter/_lib/**`.
- Copies architecture-specific HSA artifacts into `jax_aiter/_hsa/**`.
- Includes both as package data for installation/runtime use.

## 7) Configuration and environment model

Important environment variables:

- `JA_ROOT_DIR`: repository root for local/dev builds.
- `AITER_ASM_DIR`: architecture-specific kernel directory (auto-set when possible).
- `GPU_ARCHS`: target GPU arch list for builds.
- `AITER_SYMBOL_VISIBLE`: enables expected symbol visibility behavior for integration flows.

`ja_compat` modules also provide chip/dtype/tuning helpers that are used by dispatch and runtime checks.

## 8) Testing and performance validation

Testing (`tests/`):

- MHA dense coverage (`test_mha_ja.py`, `test_mha_torch_ja.py`).
- MHA varlen coverage (`test_mha_varlen_ja.py`).
- GEMM coverage (`test_gemm_a8w8_ja.py`).
- Optional sharded execution via `PYTEST_SHARD_TOTAL` and `PYTEST_SHARD_INDEX` in `tests/conftest.py`.

Benchmarking (`benchmarks/benchmark_mha.py`):

- Compares reference JAX attention (`jax_aiter.baseline.mha_attn.attention_ref`) against JAX-AITER attention.
- Runs across multiple sequence/head settings and reports timing statistics.

## 9) Design principles reflected in code

- **Separation of concerns:** API/dispatch code is independent from library-loading mechanics and from native bridge implementations.
- **Runtime adaptability:** kernel selection is data- and hardware-aware.
- **Training compatibility:** `custom_vjp` keeps forward/backward behavior explicit and testable.
- **Deployability:** build artifacts are staged into package-local runtime directories for wheel installs.

## 10) Practical call graph (attention)

`flash_attn_func` (Python API)  
-> dispatch + capability checks (`jax_aiter/mha/mha.py`)  
-> `register_ffi_target("...JA")` (`jax_aiter/ffi/registry.py`)  
-> `jax.ffi.ffi_call(...)`  
-> XLA FFI handler symbol in `csrc/ffi/*_ja.*`  
-> AITER kernel implementation

