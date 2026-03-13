// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// FP8 flat matmul FFI handler using single AITER ASM kernel.
// gfx942 (MI300) only -- .co file not available for gfx950.
// Out[M,N] fp16 = dequant(A[M,K], a_scale) @ dequant(B[N,K], b_scale)^T
// Constraints: N % 256 == 0, K % 128 == 0.

#include <cstring>
#include <hip/hip_runtime.h>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "aiter_hip_common.h"

namespace ffi = xla::ffi;

namespace jax_aiter {

struct __attribute__((packed)) FlatmmKernelArgs {
  const void* a_ptr;
  const void* b_ptr;
  const void* c_ptr;
  const void* sa_ptr;
  const void* sb_ptr;
  void* d_ptr;
  void* d_f16_ptr;
  void* dbg_int_ptr;
  void* dbg_fp8_ptr;
  void* dbg_f16_ptr;
  void* dbg_fp32_ptr;
  int hidden_size;
  int intermediate_size;
  int num_tokens;
  int num_experts;
  int topk;
  int stride_token;
};

ffi::Error
FlatmmFp8Fwd_Bridge(
    hipStream_t stream,
    ffi::AnyBuffer xq,
    ffi::AnyBuffer wq,
    ffi::AnyBuffer x_scale,
    ffi::AnyBuffer w_scale,
    ffi::Result<ffi::AnyBuffer> out) {

  auto xq_dims = xq.dimensions();
  auto wq_dims = wq.dimensions();

  if (xq_dims.size() != 2 || wq_dims.size() != 2) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "FlatMM requires 2D inputs: A[M,K], B[N,K]");
  }

  int M = static_cast<int>(xq_dims[0]);
  int K = static_cast<int>(xq_dims[1]);
  int N = static_cast<int>(wq_dims[0]);

  constexpr int TileM = 128, TileN = 256, TileK = 128;

  if (N % TileN != 0 || K % TileK != 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "N must be divisible by 256 and K by 128 for FlatMM");
  }

  static AiterAsmKernel kernel(
      "flatmm_uk_gfx9_f16f8_128x256x128_1x4x1_16x16x32",
      "flatmm_uk_gfx9_f16f8_128x256x128_1x4x1_16x16x32.co");

  FlatmmKernelArgs args = {};
  args.a_ptr  = xq.untyped_data();
  args.b_ptr  = wq.untyped_data();
  args.c_ptr  = nullptr;
  args.sa_ptr = x_scale.untyped_data();
  args.sb_ptr = w_scale.untyped_data();
  args.d_ptr  = nullptr;
  args.d_f16_ptr = out->untyped_data();
  args.num_tokens = M;
  args.intermediate_size = N;
  args.hidden_size = K;
  args.num_experts = 1;
  args.topk = 1;
  args.stride_token = K;

  int gdx = (N + TileN - 1) / TileN;
  int gdy = (M + TileM - 1) / TileM;

  size_t arg_size = sizeof(args);
  kernel.launch_kernel({&args, &arg_size, gdx, gdy, 1, 256, 1, 1, stream});

  return ffi::Error::Success();
}

} // namespace jax_aiter

#pragma GCC visibility push(default)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FlatmmFp8FwdJA, jax_aiter::FlatmmFp8Fwd_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>(),
    {xla::ffi::Traits::kCmdBufferCompatible});

#pragma GCC visibility pop
