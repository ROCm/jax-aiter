Integration Notes

1) Integration for gemm_a8w8 and gemm_a16w16 is shown. To run a) Start a docker container with JAX and JAX-Triton 2) python3 [gemm_a16w16.py|gemm_a8w8.py]
2) _aiter_triton folder is created when AITER repo is imported as third party. This folder only contains AITER Triton stuff needed for JAX Triton
3) The following are the paths in AITER repo copied into _aiter_triton a) aiter/ops/triton/_triton_kernels b) aiter/ops/triton/config c) aiter/ops/triton/utils

