# JAX-AITER Design Overview

## Purpose

`jax-aiter` integrates AMD's AITER operator stack into JAX through XLA FFI bindings.  
The project focuses on high-performance ROCm kernels (especially attention) while preserving a JAX-first API and autodiff behavior.

---

## Architecture at a glance

The codebase is organized into five layers:

1. **Python API layer** (`jax_aiter/*`)
   - User-facing JAX functions:
     - `jax_aiter.mha.flash_attn_func`
     - `jax_aiter.mha.flash_attn_varlen`
     - `jax_aiter.gemm_a8w8.gemm_a8w8_ASM`
     - `jax_aiter.wv_splitkq.wv_splitkq_fp8`
   - Uses `jax.custom_vjp` for forward/backward integration.
   - Handles input normalization (padding to multiples of 8, optional tensors, window normalization).

2. **FFI registration/runtime loader layer** (`jax_aiter/ffi/registry.py`)
   - Loads shared libraries and registers function pointers with `jax.ffi.register_ffi_target`.
   - Uses a symbol-to-module map (for example, `MhaFwdJA -> ck_fused_attn_fwd_ja.so`).
   - Ensures symbols are registered lazily, on first call.

3. **C++/HIP FFI bridge layer** (`csrc/ffi/**/*.cu`, `csrc/common/*`)
   - Implements XLA FFI handlers (`XLA_FFI_DEFINE_HANDLER_SYMBOL(...)`).
   - Validates shapes/dtypes/options, manages optional arguments, and dispatches into AITER kernels.
   - Provides shared helpers for mask logic, RNG/dropout state, and MQA/GQA reductions.

4. **Kernel implementation layer** (AITER-generated modules)
   - Built artifacts in:
     - `build/aiter_build/*.so` (AITER JIT modules)
     - `build/jax_aiter_build/*.so` (JAX-AITER bridge modules + umbrella lib)
   - Includes CK and ASM-backed attention/GEMM kernels.

5. **Build and packaging layer**
   - `Makefile`: builds umbrella and JAX bridge modules.
   - `jax_aiter/jit/build_jit.py`: builds AITER JIT modules with JAX-specific patching.
   - `setup.py`: packages `.so` and HSA artifacts into `jax_aiter/_lib` and `jax_aiter/_hsa`.

---

## Core runtime flow (attention path)

Example call:

1. User calls `jax_aiter.mha.flash_attn_func(...)`.
2. Python wrapper:
   - Normalizes parameters.
   - Pads head dimensions when needed.
   - Chooses kernel family (standard MHA vs FMHA v3) based on shape/dtype/options/hardware.
3. Wrapper ensures FFI target is registered:
   - `register_ffi_target("MhaFwdJA" | "FmhaV3FwdJA" | ...)`
   - Registry loads `libjax_aiter.so`, then module `.so` files, then registers symbol pointers with JAX.
4. JAX executes `jax.ffi.ffi_call(...)` with typed output signatures.
5. C++ bridge receives buffers and attributes, validates contract, prepares masks/RNG/dropout info, then calls underlying AITER kernel entry points.
6. Outputs are returned to JAX; custom VJP backward path triggers corresponding backward bridge target.

---

## Attention design details

### Standard attention (`mha.py`)

- Public API: `flash_attn_func` (with `custom_vjp`).
- Kernel wrappers:
  - Forward: `MhaFwdJA` / `FmhaV3FwdJA`
  - Backward: `MhaBwdJA` / `FmhaV3BwdJA`
- Dispatch checks include:
  - dtype and head-dim support
  - dropout/bias/alibi restrictions
  - causal/window mode
  - hardware constraints (`gfx942`/`gfx950`)
  - sequence length heuristics

### Variable-length attention (`mha_varlen.py`)

- Public API: `flash_attn_varlen` (with `custom_vjp`).
- Uses cumulative sequence length arrays (`cu_seqlens_*`) and unpadded token layout.
- Separate FFI targets:
  - `MhaVarlenFwdJA`, `MhaVarlenBwdJA`
  - `FmhaV3VarlenFwdJA`, `FmhaV3VarlenBwdJA`
- Supports padded-sequence metadata (`cu_seqlens_*_padded`) on non-v3 paths.

---

## FFI registry and library loading model

`jax_aiter/ffi/registry.py` centralizes symbol registration:

- Tracks global state:
  - umbrella handle
  - module handles
  - already-registered targets
- Load order:
  1. `libjax_aiter.so` (umbrella)
  2. `build/aiter_build/*.so`
  3. `build/jax_aiter_build/*.so`
- Symbol pointers are resolved from loaded module handles and registered to ROCm backend via JAX FFI.

This design keeps registration lazy (fast startup) and avoids repetitive registration work.

---

## Build and artifact model

### Makefile-driven artifacts

- `make`:
  - builds `build/jax_aiter_build/libjax_aiter.so`
- `make ja_mods`:
  - builds JAX bridge modules under `build/jax_aiter_build/` (for example `ck_fused_attn_fwd_ja.so`)

### AITER JIT artifacts

- `python3 jax_aiter/jit/build_jit.py`
  - patches AITER JIT behavior to:
    - use JAX-AITER build locations
    - inject JAX FFI and project include paths
    - wire linker flags against `libjax_aiter`
  - emits modules into `build/aiter_build/`

### Packaging behavior

- `setup.py` copies runtime artifacts into package data:
  - `.so` files into `jax_aiter/_lib/...`
  - HSA kernels (`.co`, related files) into `jax_aiter/_hsa/...`
- Runtime lookup (`ja_compat/config.py`) prefers:
  1. `JA_ROOT_DIR/build` in development
  2. packaged `_lib` in installed mode

---

## Environment and configuration strategy

- `JA_ROOT_DIR`: root for local build/runtime discovery.
- `AITER_ASM_DIR`: set early at package import (`jax_aiter/__init__.py`) based on detected GPU arch when not pre-set.
- `GPU_ARCHS`: influences architecture-specific behavior and build targeting.

This keeps local development and installed-wheel usage aligned with the same Python APIs.

---

## Testing strategy

Main tests are under `tests/`:

- `test_mha_ja.py`
  - compares forward/backward against pure-JAX baseline (`baseline/mha_attn.py`)
  - covers dtype/layout/causal/local/dropout/bias/alibi/MHA-MQA-GQA combinations
- `test_mha_varlen_ja.py`
  - validates varlen forward/backward against baseline with random padding masks
- `test_gemm_a8w8_ja.py`
  - exercises GEMM and split-KQ wrappers (JAX vs AITER/PyTorch references)
- `smoke_mha_test.py`
  - focused smoke runner with sharding across available GPUs

---

## Design tradeoffs

- **Pros**
  - JAX-native API with custom VJP integration.
  - Lazy FFI registration and modular shared-library loading.
  - Supports both dense and varlen attention flows.
  - Reuses AITER build stack while adapting for JAX.

- **Tradeoffs**
  - Runtime depends on correctly built external artifacts (`.so`, HSA files).
  - Kernel capability checks are complex and hardware-specific.
  - Feature support differs across kernel families (for example, some v3 constraints).

