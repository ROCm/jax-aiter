# JAX-AITER Design Overview

## Purpose

JAX-AITER integrates AMD AITER GPU operator kernels into JAX through XLA FFI.
The design goal is:

- keep user APIs JAX-native,
- avoid runtime PyTorch dependencies,
- pass device buffers directly through FFI without host copies.

At a high level, Python wrappers build `jax.ffi.ffi_call(...)` invocations, and
native ROCm handlers launch AITER/ASM kernels on the current HIP stream.

## Architectural layers

### 1) Public Python APIs (user-facing)

Primary operator modules:

- `jax_aiter/mha/mha.py` - flash attention (`flash_attn_func`, `flash_attn_varlen`) with `custom_vjp`.
- `jax_aiter/rmsnorm/rmsnorm.py` - RMSNorm forward via FFI, backward in JAX.
- `jax_aiter/gemm/gemm.py` - BF16 GEMM forward via FFI, backward via GEMM-based transpose math.
- `jax_aiter/gemm_fp8/gemm_fp8_mi350.py`, `gemm_i8/gemm_i8.py`, `gemm_fp4/gemm_fp4.py`, `flatmm_fp8/flatmm_fp8.py` - forward-only FFI wrappers.

Patterns:

- `custom_vjp` is used where gradients are implemented.
- `jax.jit` wraps FFI call closures with static attrs.
- lightweight shape/dtype checks and padding happen before dispatch.

### 2) FFI registry and dynamic loading

`jax_aiter/ffi/registry.py` is the registration hub:

- loads `libjax_aiter.so` umbrella library first,
- then loads module `.so` files from:
  - `build/aiter_build` (AITER JIT modules),
  - `build/jax_aiter_build` (JAX-AITER FFI modules),
- resolves handler symbol pointers and registers them through:
  `jax.ffi.register_ffi_target(..., platform="ROCM")`.

`SYMBOL_TO_MODULE_MAP` controls which Python target name maps to which shared object.

### 3) Native FFI bridge handlers (C++/HIP)

`csrc/ffi/**` contains one bridge per exported FFI target, for example:

- `MhaFwdUnifiedJA` and `MhaBwdUnifiedJA`
- `RmsnormFwdJA`
- `GemmFwdJA`
- `GemmFp8Mi350FwdJA`, `GemmI8FwdJA`, `GemmFp4FwdJA`, `FlatmmFp8FwdJA`

Handlers are exported with `XLA_FFI_DEFINE_HANDLER_SYMBOL(...)`.
Each bridge:

1. validates shapes/dtypes/attrs,
2. unpacks pointers from XLA buffers,
3. prepares kernel arguments (and helper state like RNG or semaphores),
4. launches AITER CK/ASM kernels via HIP on the provided stream.

### 4) Build + packaging pipeline

- `Makefile` compiles:
  - umbrella shared library: `build/jax_aiter_build/libjax_aiter.so`,
  - thin FFI modules: `build/jax_aiter_build/*_ja.so`.
- GEMM config headers are generated from AITER codegen.
- `jax_aiter/jit/build_jit.py` builds AITER JIT modules into `build/aiter_build`.
- `setup.py` copies built `.so` and HSA `.co` artifacts into package-local:
  - `jax_aiter/_lib/**`
  - `jax_aiter/_hsa/**`

This enables both local-development mode (`JA_ROOT_DIR`) and installed-wheel mode.

## End-to-end execution flow

```text
User JAX code
  -> jax_aiter op wrapper (Python)
  -> _ensure_registered(target)
  -> ffi.registry loads .so + registers FFI target
  -> jax.ffi.ffi_call(target, outputs, attrs)
  -> XLA FFI dispatch
  -> C++/HIP bridge handler (csrc/ffi/*)
  -> AITER CK/ASM kernel launch (HIP stream)
  -> output buffers written in-place
  -> JAX receives outputs
  -> (if training) custom_vjp backward path executes
```

## Operator gradient model

- **MHA**: forward and backward both use ROCm FFI handlers.
- **RMSNorm**: forward uses CK FFI kernel; backward is pure JAX math.
- **BF16 GEMM**: forward uses ASM FFI; backward reuses GEMM FFI via transpose-based calls.
- **FP8/INT8/FP4/FlatMM variants**: currently forward-only wrappers.

## Runtime configuration and environment

Important environment variables:

- `JA_ROOT_DIR` - points to repository root in development mode; used to resolve `build/`.
- `AITER_ASM_DIR` - path to base `hsa/` directory; set automatically on import when possible.
- `GPU_ARCHS` - target architectures for build/codegen (for example `gfx942`, `gfx950`).

Selected runtime constraints enforced by wrappers/handlers include:

- MHA: fp16/bf16, head dimensions multiple of 8 and up to 256.
- BF16 GEMM: `K % 64 == 0`.
- Additional shape constraints for FP8/INT8/FP4 paths depend on kernel family.

## Repository boundaries

- Editable project code lives in `jax_aiter/`, `csrc/`, `tests/`, `benchmarks/`, and top-level build scripts.
- `third_party/` contains vendored dependencies and should not be modified for routine repo changes.

## Current design notes

- Registry-based lazy loading keeps startup light and avoids loading all modules unless needed.
- The umbrella library is loaded before thin modules to preserve expected HIP symbol/runtime behavior.
- The package top-level export is intentionally minimal (`jax_aiter.mha` via lazy import); other operators are imported from their subpackages directly.
