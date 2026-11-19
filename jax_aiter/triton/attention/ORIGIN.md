# AITER Triton varlen MHA — origin and ABI lock

jax-aiter does **not** vendor these kernel sources. They are imported from the
read-only AITER checkout at compile/lowering time.

- AITER kernels:
  - `aiter.ops.triton._triton_kernels.attention.mha._attn_fwd`
  - `aiter.ops.triton._triton_kernels.attention.mha_onekernel_bwd._bwd_preprocess`
  - `aiter.ops.triton._triton_kernels.attention.mha_onekernel_bwd.bwd_kernel_causal`
- Public Torch wrappers (`flash_attn_varlen_func`, `_FlashAttnVarlenFunc.apply`)
  are **not** used. JAX `custom_vjp` launches the JIT kernels above.
- Config: `aiter/ops/triton/configs/gfx950-MHA-DEFAULT.json`
- Triton: **3.8.0** (ROCm `triton-rocm`). Other versions fail loudly.

## HIP launch ABI (gfx950, Triton 3.8.0)

Measured in `rv_rock` by compiling `_attn_fwd` (VARLEN, QK=192/V=128, causal):

- Binary: HSACO (`CompiledKernel.asm["hsaco"]`)
- `hipModuleLoadData` + `hipModuleGetFunction(metadata.name)`
- `hipModuleLaunchKernel(..., kernelParams, extra=nullptr)`
- `block = warp_size * num_warps` with `warp_size=64`
- `kernelParams`: `void**` of per-argument storage in runtime-arg order,
  then **two** scratch pointers (`global_scratch`, `profile_scratch`).
  v1 requires both scratch sizes to be 0 and passes nullptr.
- Pointer args: `hipDeviceptr_t` (8 bytes). Empty JAX buffers (0 bytes) become
  nullptr for unused optional tensors.
- `i32` / `fp32` stored in the low bytes of an 8-byte slot (little-endian).

Constexprs are baked into the binary. Runtime args for `_attn_fwd` are the
pointer/scalar list excluding `IS_CAUSAL` … `HEAD_STRIDE_ALIGNED_8` except
`BATCH`, which is a runtime `i32`.

When an AITER wrapper would pass Python ``None`` (no alibi, no dropout mask,
no FP8 descale, no sink), specialize that tensor argument as constexpr
``None`` at compile time. Passing a null ``hipDeviceptr_t`` is not the same:
Triton then treats the pointer as present and loads address 0.
