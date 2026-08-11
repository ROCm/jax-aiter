# JAX-AITER architecture

This document describes how repository-owned JAX-AITER code connects JAX to
AITER. It intentionally avoids performance claims.

## Layers

```text
JAX public API
  custom_vjp / custom_partitioning
    raw JAX FFI operation
      thin C++/HIP FFI shim
        AITER JIT library or packaged HSA kernel
          AMD GPU
```

The public APIs live under `jax_aiter/{gemm,gemm_fp4,mha,rmsnorm,activation}`.
Raw FFI wrappers live under `jax_aiter/ops`. Thin handlers live under
`csrc/ffi`.

## Library loading

`jax_aiter/ffi/registry.py` loads libraries in this order:

1. `libjax_aiter.so`, the umbrella library that owns shared HIP helpers;
2. AITER JIT libraries such as `libmha_fwd.so`;
3. thin FFI shims such as `mha_fwd_ja.so`;
4. FFI symbols registered with `jax.ffi.register_ffi_target`.

Development checkouts load both sets from `$JA_ROOT_DIR/build`.

Installed default wheels combine two roots:

- thin FFI shims from `jax_aiter/_lib/jax_aiter_build`;
- downloaded JIT libraries from
  `$XDG_CACHE_HOME/jax-aiter/<version>/<cache-id>/aiter_build`.

The cache becomes active only when all three expected JIT libraries are
present. This prevents an interrupted fetch from becoming the runtime source.
The `+full` GitHub wheel packages both roots and needs no download.

## MXFP4 training path

`gemm_fp4_bf16(a, b)` accepts BF16 matrices and returns BF16 output.

Forward:

```text
A BF16 -> rowwise MXFP4
B BF16 -> rowwise + columnwise MXFP4
row(A) x row(B) -> GemmFp4FwdJA -> BF16 output
```

Backward:

```text
dA = row(dY) x col(B) -> GemmFp4FwdJA
dB = col(dY) x col(A) -> GemmFp4FwdJA
```

The dB/wgrad partition contracts the batch/token dimension. When that
dimension is FSDP-sharded, the partition emits `jax.lax.psum` so every rank
receives the reduced weight gradient.

The high-level implementation is in
`jax_aiter/gemm_fp4/gemm_fp4.py`; raw casts and GEMMs are in
`jax_aiter/ops/gemm_fp4.py`.

## Flash attention

The batch and variable-length APIs share unified forward/backward FFI targets:

- `MhaFwdUnifiedJA`
- `MhaBwdUnifiedJA`

AITER chooses CK or ASM-v3 implementations internally based on the shape and
configuration. JAX-AITER owns the `custom_vjp`, packed metadata, sharding
rules, residual naming for rematerialization, and MQA/GQA gradient reduction.

The default wheel includes the MHA shims but not the large JIT libraries.
`jax-aiter-fetch-mha` downloads the exact manifest-backed libraries for
`gfx950`.

## Wheel variants

| Variant | Version form | MHA shims | MHA JIT libraries | Distribution |
|---|---|---:|---:|---|
| Default | `0.1.0a2` | Yes | Downloaded on demand | GitHub, then PyPI after approval |
| Full | `0.1.0a2+full` | Yes | Packaged | GitHub release only |

Both variants contain only `gfx950` HSA files for alpha2.

## Runtime dependency boundary

JAX-AITER does not import PyTorch at runtime. The public package depends on:

- JAX;
- the separately installed ROCm PJRT/plugin stack;
- `zstandard` for downloading JIT-library assets.

The C++/HIP shims use JAX FFI and AITER headers directly.
