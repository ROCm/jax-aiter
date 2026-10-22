# Third-party notices

## Primus-Turbo MXFP4 kernels

The FlyDSL MXFP4 cast and grouped-GEMM sources under
`jax_aiter/flydsl/kernels/primus_turbo/` are vendored from
`AMD-AGI/Primus-Turbo` commit `e2d9f1d7285f3e77e9f02bdc2ad802f9322ab5f0`;
`primus_turbo/ORIGIN.md` lists the files, their source hashes and the import
rewrites. The FlyDSL-derived kernel files are Apache-2.0
(`primus_turbo/LICENSE-APACHE`) and `primus_turbo/common/` is MIT
(`primus_turbo/LICENSE`).

`primus_turbo/utils/buffer_ops.py` is `ROCm/aiter`
`aiter/ops/flydsl/kernels/buffer_ops.py` at
`08d73a1921fa221e5b0873621bdf65da2238b623`, followed by `extract_base_index`
and `create_llvm_ptr` from `ROCm/FlyDSL` `kernels/common/buffer_ops.py` at
`7dac0826746b158a3d44a7fd323faf2ed02aa2db`. Both are Apache-2.0 (FlyDSL
Project Contributors, with AMD modifications); see `licenses/Apache-2.0.txt`.

## Primus-Turbo DeepEP image layer

`docker/mxfp4/deepep/bindings_deepep_jax.cpp` is the DeepEP subset of
`AMD-AGI/Primus-Turbo` `csrc/jax/bindings_jax.cpp`, and
`docker/mxfp4/deepep/primitive__init__.py` is its
`primus_turbo/jax/primitive/__init__.py` reduced to the MoE primitives, both at
commit `e2d9f1d7285f3e77e9f02bdc2ad802f9322ab5f0` (MIT, Advanced Micro
Devices, Inc.; Primus-Turbo `LICENSE`). `Dockerfile.deepep` compiles the
remaining DeepEP sources from that Primus-Turbo commit and copies its `LICENSE`
and `LICENSE-APACHE` into the image.

## MXFP4 casts

`jax_aiter/flydsl/kernels/primus_turbo_mxfp4_quant.py` applies Primus-Turbo's
cast math (Apache-2.0) with the flat row mapping of AITER's
`per_1x32_mx_quant` (MIT, Advanced Micro Devices, Inc.). Its column-wise
staging and layout derive from `pytorch/ao#4357` at `fbefeb0`, copyright Meta
Platforms, Inc. and affiliates, under BSD-3-Clause; see
`licenses/BSD-3-Clause.txt`.

## JAX–FlyDSL bridge architecture

The first-party bridge under `jax_aiter/flydsl/` was implemented using the
Apache-2.0 `AMD-ROCm-Internal/jax-flydsl` project at commit
`44e587016b1a08fcfbfdf69165477b0392cafe37` as architectural reference. It
uses JAX-AITER's typed XLA FFI conventions and does not depend on or import the
`jax_flydsl` package.
