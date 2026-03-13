// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// MI350 FP8 block-scale GEMM FFI handler.
// Computes Out = dequant(A, a_scale) @ dequant(B, b_scale)^T
// where A:[M,K] fp8, B:[N,K] fp8, a_scale:[K/128,M] f32, b_scale:[K/128,N/128] f32.
// Output: [M,N] bf16.
// Uses 2 hardcoded AITER ASM kernels: x128 (M>32) and x32 (M<=32).

#include <cstring>
#include <hip/hip_runtime.h>
#include <string>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "aiter_hip_common.h"

namespace ffi = xla::ffi;

namespace jax_aiter {

struct __attribute__((packed)) Fp8Mi350KernelArgs {
  void *ptr_C;     p2 _p0;   // output [M, N] bf16
  void *ptr_X;     p2 _p1;   // input A [M, K] fp8
  void *ptr_W;     p2 _p2;   // input B [N, K] fp8
  void *ptr_XC;    p2 _p3;   // unused
  void *ptr_XQ;    p2 _p4;   // a_scale [K/128, M] f32
  void *ptr_WQ;    p2 _p5;   // b_scale [K/128, N/128] f32
  void *ptr_tmp0;  p2 _p6;   // unused
  void *ptr_tmp1;  p2 _p7;   // unused
  void *ptr_tmp2;  p2 _p8;   // unused
  unsigned int K;   p3 _p9;
  unsigned int N;   p3 _p10;
  unsigned int M;   p3 _p11;
  unsigned int eprt_cnt; p3 _p12;
  unsigned int Xs;  p3 _p13;  // stride A in bytes
  unsigned int Ws;  p3 _p14;  // stride B in bytes
  unsigned int Cs;  p3 _p15;  // stride C in bytes
  unsigned int tmp0; p3 _p16;
  unsigned int tmp1; p3 _p17;
  unsigned int tmp2; p3 _p18;
  unsigned int tmp3; p3 _p19;
  unsigned int splitk; p3 _p20;
  unsigned int activation; p3 _p21;
  void *ptr_tmp3;  p2 _p22;  // unused
};

ffi::Error
GemmFp8Mi350Fwd_Bridge(
    hipStream_t stream,
    ffi::AnyBuffer xq,       // A: [M, K] fp8
    ffi::AnyBuffer wq,       // B: [N, K] fp8
    ffi::AnyBuffer x_scale,  // [K/128, M] f32
    ffi::AnyBuffer w_scale,  // [K/128, N/128] f32
    ffi::Result<ffi::AnyBuffer> out) {  // [M, N] bf16

  auto xq_dims = xq.dimensions();
  auto wq_dims = wq.dimensions();

  if (xq_dims.size() != 2 || wq_dims.size() != 2) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "FP8 GEMM requires 2D inputs: A[M,K], B[N,K]");
  }

  int M = static_cast<int>(xq_dims[0]);
  int K = static_cast<int>(xq_dims[1]);
  int N = static_cast<int>(wq_dims[0]);

  if (static_cast<int>(wq_dims[1]) != K) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "K dimension mismatch: A[M,K] vs B[N,K]");
  }

  constexpr int TileN = 128;
  constexpr int TileK = 128;
  constexpr int GridTileN = TileN * 2;  // kernel processes 2 N-tiles per workgroup

  if (N % GridTileN != 0 || K % TileK != 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "N must be divisible by 256 and K by 128 for MI350 FP8 GEMM");
  }
  if (M < 16) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "M must be >= 16 for MI350 FP8 GEMM");
  }
  if (K < 512) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "K must be >= 512 for MI350 FP8 GEMM");
  }

  int TileM = (M <= 32) ? 32 : 128;

  static AiterAsmKernel kernel_x128("f8_block_scale_mi350_x128",
                                     "f8_block_scale_mi350_x128.co");
  static AiterAsmKernel kernel_x32("f8_block_scale_mi350_x32",
                                    "f8_block_scale_mi350_x32.co");
  AiterAsmKernel* impl = (M <= 32) ? &kernel_x32 : &kernel_x128;

  const int fp8_elem_size = 1;

  Fp8Mi350KernelArgs args = {};
  args.ptr_C    = out->untyped_data();
  args.ptr_X    = const_cast<void*>(xq.untyped_data());
  args.ptr_W    = const_cast<void*>(wq.untyped_data());
  args.ptr_XQ   = const_cast<void*>(x_scale.untyped_data());
  args.ptr_WQ   = const_cast<void*>(w_scale.untyped_data());
  args.K        = K;
  args.N        = N;
  args.M        = M;
  args.eprt_cnt = 1;
  args.Xs       = K * fp8_elem_size;
  args.Ws       = K * fp8_elem_size;
  args.Cs       = N * 2;  // bf16 output = 2 bytes
  args.splitk   = 0;
  args.activation = 0;

  int gdx = (N + TileN * 2 - 1) / (TileN * 2);
  int gdy = (M + TileM - 1) / TileM;

  size_t arg_size = sizeof(args);
  impl->launch_kernel({&args, &arg_size, gdx, gdy, 1, 256, 1, 1, stream});

  return ffi::Error::Success();
}

} // namespace jax_aiter

#pragma GCC visibility push(default)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    GemmFp8Mi350FwdJA, jax_aiter::GemmFp8Mi350Fwd_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()   // XQ: [M, K] fp8
        .Arg<ffi::AnyBuffer>()   // WQ: [N, K] fp8
        .Arg<ffi::AnyBuffer>()   // x_scale: [K/128, M] f32
        .Arg<ffi::AnyBuffer>()   // w_scale: [K/128, N/128] f32
        .Ret<ffi::AnyBuffer>(),  // Out: [M, N] bf16
    {xla::ffi::Traits::kCmdBufferCompatible});

#pragma GCC visibility pop
