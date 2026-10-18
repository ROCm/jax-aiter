# Primus-Turbo MXFP4 kernel provenance

Vendored from https://github.com/AMD-AGI/Primus-Turbo at `e2d9f1d7285f3e77e9f02bdc2ad802f9322ab5f0` by
`vendor_primus_turbo.sh`. Only imports are rewritten: `primus_turbo.*` to this package, and
`flydsl.expr.buffer_ops`, which FlyDSL 0.3.0 no longer exports, to `utils/buffer_ops.py`.
`utils/prims.py` also reads `fx.ArithValue` as the `ArithValue` it already imports.
FlyDSL-derived kernel files are Apache-2.0 (`LICENSE-APACHE`); `common/` is MIT (`LICENSE`).

| file | source sha256 |
| --- | --- |
| `common/constants.py` | `a15957af601053fb1cc893e469baf601aeb9cee5625ce45192101b7ceda9f0ad` |
| `common/logger.py` | `df0b29783b0abe9c4ede714bdffae3b99343dd91dfbc29e840601f046bace2fd` |
| `gemm/gemm_mxfp4_kernel.py` | `9a22107e1c9e56853a682ee620b96111308d2d750ec6b1dde1ce4f6e48ac5c6c` |
| `grouped_gemm/grouped_gemm_fp8_kernel.py` | `b36c4468ef715e19c0530bc4ae486f1c2499ddf400bb84e4624be6009f7d59c4` |
| `grouped_gemm/grouped_gemm_mxfp4_kernel.py` | `8f06536f4b531312af36262258c7cbf4bb19ede21c816dbe6781b6952276f282` |
| `grouped_gemm/grouped_gemm_mxfp8_kernel.py` | `7ebbcfa8f4298575fb9634852a8a8db520401d4335d7829e4ba7bc44fae181f6` |
| `quantization/mxfp4_grouped_quant.py` | `696cf3641872d07b1f5911a2fed37efe6dc0a84cfd767e0b6a9e604def65530e` |
| `quantization/mxfp4_quant_kernel.py` | `0f32bbcc9c9d386f74615dcb6cc0429eab680959ac1020045d70e6c922632148` |
| `quantization/mxfp8_quant_flydsl.py` | `4bb0e43036b95d2ac257b41a4650448947a88640053b0967aa2f3a8b8d961463` |
| `utils/gemm_epilogue_helper.py` | `51ad94791a04b3d488a8df8fc81b379fc84d790165aca8f00d8225fece7d7fb3` |
| `utils/gemm_helper.py` | `b73bf4a8ea9aeb364575e224dfd5ffb854b04029141937773ae96ebc2f211dbc` |
| `utils/prims.py` | `305fa8e045061e47fd8ff46ef6bb66f494e8e1d0074de750c041f8e7985ecbcd` |

## `utils/buffer_ops.py`

Apache-2.0 (FlyDSL Project Contributors, with AMD modifications). It is
https://github.com/ROCm/aiter `aiter/ops/flydsl/kernels/buffer_ops.py` at `08d73a1921fa221e5b0873621bdf65da2238b623`
(sha256 `d95ceaa4d526d7b018b38cfc2f42885cae9083792511890559da1127cdf7800e`), unchanged, followed by `extract_base_index` and
`create_llvm_ptr` verbatim from https://github.com/ROCm/FlyDSL `kernels/common/buffer_ops.py` at
`7dac0826746b158a3d44a7fd323faf2ed02aa2db` (sha256 `ae7ffa2669e073cf010b293eb473cd5a929e336fc1a3945f00e48e0a866c4fb8`).
