// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>
#include <torch/all.h>

#include "xla/ffi/api/ffi.h"

#include "hip_utils.h"
#include "logging.h"
#include "torch_utils.h"
#include "custom.h"

// Forward declarations for some of the unimplemented functions.
namespace aiter {
  void LLZZ(at::Tensor in_a, at::Tensor in_b, at::Tensor out_c, const int64_t solidx);
  void MMCustomGPU(at::Tensor& in_a, at::Tensor& in_b, at::Tensor& out_c);
}

namespace ffi = xla::ffi;

namespace jax_aiter {

ffi::Error
LLMM1_Bridge(hipStream_t stream, ffi::AnyBuffer in_a,
            ffi::AnyBuffer in_b,
            ffi::Result<ffi::AnyBuffer> out_c,
            int32_t rows_per_block) {
  
  // Validate input types match.
  if (in_a.element_type() != in_b.element_type()) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Input tensor types must match");
  }
  
  // Validate supported types
  auto dtype = in_a.element_type();
  if (dtype != ffi::DataType::F16 && dtype != ffi::DataType::BF16) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, 
                     "LLMM1 only supports F16 and BF16 types");
  }

  auto dims_a = in_a.dimensions();
  auto dims_c = out_c->dimensions();
  int64_t m = dims_a[0];
  int64_t k = dims_a[1];
  int64_t n = dims_c[1];

  JA_LOG("LLMM1 (%ld*%ld) . (%ld*%ld) -> (%ld*%ld) rows_per_block=%d", 
         m, k, n, k, m, n, rows_per_block);

  const int dev_idx = ::jax_aiter::device_from_ptr(in_a.untyped_data());

  // Create tensors using the new AnyBuffer utilities
  auto A_t = wrap_any_buffer(in_a, dev_idx);
  auto B_t = wrap_any_buffer(in_b, dev_idx);
  auto out_t = wrap_any_buffer(*out_c, dev_idx);

  aiter::LLMM1(A_t, B_t, out_t, static_cast<int64_t>(rows_per_block));
  return ffi::Error::Success();
}

ffi::Error
WvSplitK_Bridge(hipStream_t stream, ffi::AnyBuffer in_a,
               ffi::AnyBuffer in_b,
               ffi::Result<ffi::AnyBuffer> out_c,
               int32_t N, int32_t CuCount) {
  
  // Validate input types match
  if (in_a.element_type() != in_b.element_type()) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Input tensor types must match");
  }
  
  // Validate supported types
  auto dtype = in_a.element_type();
  if (dtype != ffi::DataType::F16 && dtype != ffi::DataType::BF16) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, 
                     "WvSplitK only supports F16 and BF16 types");
  }

  auto dims_a = in_a.dimensions();
  auto dims_c = out_c->dimensions();
  int64_t m = dims_a[0];
  int64_t k = dims_a[1];
  int64_t n = dims_c[1];

  JA_LOG("WvSplitK (%ld*%ld) . (%ld*%ld) -> (%ld*%ld) N=%d CuCount=%d", 
         m, k, k, n, m, n, N, CuCount);

  const int dev_idx = ::jax_aiter::device_from_ptr(in_a.untyped_data());

  // Create tensors using the new AnyBuffer utilities
  auto A_t = wrap_any_buffer(in_a, dev_idx);
  auto B_t = wrap_any_buffer(in_b, dev_idx);
  auto out_t = wrap_any_buffer(*out_c, dev_idx);

  aiter::wvSpltK(A_t, B_t, out_t, static_cast<int64_t>(N), static_cast<int64_t>(CuCount));
  return ffi::Error::Success();
}

ffi::Error
WvSplitKSmall_Bridge(hipStream_t stream, ffi::AnyBuffer in_a,
                    ffi::AnyBuffer in_b,
                    ffi::Result<ffi::AnyBuffer> out_c,
                    int32_t N, int32_t CuCount) {
  
  // Validate input types match
  if (in_a.element_type() != in_b.element_type()) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Input tensor types must match");
  }
  
  // Validate supported types
  auto dtype = in_a.element_type();
  if (dtype != ffi::DataType::F16 && dtype != ffi::DataType::BF16) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, 
                     "WvSplitKSmall only supports F16 and BF16 types");
  }

  auto dims_a = in_a.dimensions();
  auto dims_c = out_c->dimensions();
  int64_t m = dims_a[0];
  int64_t k = dims_a[1];
  int64_t n = dims_c[1];

  JA_LOG("WvSplitKSmall (%ld*%ld) . (%ld*%ld) -> (%ld*%ld) N=%d CuCount=%d", 
         m, k, k, n, m, n, N, CuCount);

  const int dev_idx = ::jax_aiter::device_from_ptr(in_a.untyped_data());

  // Create tensors using the new AnyBuffer utilities
  auto A_t = wrap_any_buffer(in_a, dev_idx);
  auto B_t = wrap_any_buffer(in_b, dev_idx);
  auto out_t = wrap_any_buffer(*out_c, dev_idx);

  aiter::wv_splitk_small_fp16_bf16_wrapper(A_t, B_t, out_t, static_cast<int64_t>(N), static_cast<int64_t>(CuCount));
  return ffi::Error::Success();
}

ffi::Error
WvSplitKQ_Bridge(hipStream_t stream, ffi::AnyBuffer in_a,
                ffi::AnyBuffer in_b,
                ffi::Result<ffi::AnyBuffer> out_c,
                ffi::AnyBuffer scale_a,
                ffi::AnyBuffer scale_b,
                int32_t CuCount) {
  
  auto dims_a = in_a.dimensions();
  auto dims_c = out_c->dimensions();
  int64_t m = dims_a[0];
  int64_t k = dims_a[1];
  int64_t n = dims_c[1];

  JA_LOG("WvSplitKQ (%ld*%ld) . (%ld*%ld) -> (%ld*%ld) CuCount=%d", 
         m, k, k, n, m, n, CuCount);

  // Validate input types
  if (in_a.element_type() != ffi::DataType::F8E4M3FNUZ || in_b.element_type() != ffi::DataType::F8E4M3FNUZ) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, 
                     "WvSplitKQ requires F8E4M3FNUZ input types");
  }
  
  if (scale_a.element_type() != ffi::DataType::F32 || scale_b.element_type() != ffi::DataType::F32) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, 
                     "WvSplitKQ requires F32 scale types");
  }

  auto out_dtype = out_c->element_type();
  if (out_dtype != ffi::DataType::F16 && out_dtype != ffi::DataType::BF16) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, 
                     "WvSplitKQ output must be F16 or BF16");
  }

  const int dev_idx = ::jax_aiter::device_from_ptr(in_a.untyped_data());

  // Create tensors using the new AnyBuffer utilities
  auto A_t = wrap_any_buffer(in_a, dev_idx);
  auto B_t = wrap_any_buffer(in_b, dev_idx);
  auto out_t = wrap_any_buffer(*out_c, dev_idx);
  auto scale_a_t = wrap_any_buffer(scale_a, dev_idx);
  auto scale_b_t = wrap_any_buffer(scale_b, dev_idx);

  aiter::wvSplitKQ(A_t, B_t, out_t, scale_a_t, scale_b_t, static_cast<int64_t>(CuCount));
  return ffi::Error::Success();
}

ffi::Error
LLZZ_Bridge(hipStream_t stream, ffi::AnyBuffer in_a,
           ffi::AnyBuffer in_b,
           ffi::Result<ffi::AnyBuffer> out_c,
           int32_t solidx) {
  
  auto dims_a = in_a.dimensions();
  auto dims_c = out_c->dimensions();
  int64_t m = dims_a[0];
  int64_t k = dims_a[1];
  int64_t n = dims_c[1];

  JA_LOG("LLZZ (%ld*%ld) . (%ld*%ld) -> (%ld*%ld) solidx=%d", 
         m, k, k, n, m, n, solidx);

  // Validate F32 types
  if (in_a.element_type() != ffi::DataType::F32 || 
      in_b.element_type() != ffi::DataType::F32 || 
      out_c->element_type() != ffi::DataType::F32) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, 
                     "LLZZ requires F32 types");
  }

  const int dev_idx = ::jax_aiter::device_from_ptr(in_a.untyped_data());

  // Create tensors using the new AnyBuffer utilities
  auto A_t = wrap_any_buffer(in_a, dev_idx);
  auto B_t = wrap_any_buffer(in_b, dev_idx);
  auto out_t = wrap_any_buffer(*out_c, dev_idx);

  aiter::LLZZ(A_t, B_t, out_t, static_cast<int64_t>(solidx));
  return ffi::Error::Success();
}

ffi::Error
MMCustomGPU_Bridge(hipStream_t stream, ffi::AnyBuffer in_a,
                  ffi::AnyBuffer in_b,
                  ffi::Result<ffi::AnyBuffer> out_c) {
  
  auto dims_a = in_a.dimensions();
  auto dims_c = out_c->dimensions();
  int64_t m = dims_a[0];
  int64_t k = dims_a[1];
  int64_t n = dims_c[1];

  JA_LOG("MMCustomGPU (%ld*%ld) . (%ld*%ld) -> (%ld*%ld)", 
         m, k, k, n, m, n);

  // Validate F32 types
  if (in_a.element_type() != ffi::DataType::F32 || 
      in_b.element_type() != ffi::DataType::F32 || 
      out_c->element_type() != ffi::DataType::F32) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, 
                     "MMCustomGPU requires F32 types");
  }

  const int dev_idx = ::jax_aiter::device_from_ptr(in_a.untyped_data());

  // Create tensors using the new AnyBuffer utilities
  auto A_t = wrap_any_buffer(in_a, dev_idx);
  auto B_t = wrap_any_buffer(in_b, dev_idx);
  auto out_t = wrap_any_buffer(*out_c, dev_idx);

  aiter::MMCustomGPU(A_t, B_t, out_t);
  return ffi::Error::Success();
}

} // namespace jax_aiter

#pragma GCC visibility push(default)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
  LLMM1, jax_aiter::LLMM1_Bridge,
  ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<hipStream_t>>()
      .Arg<ffi::AnyBuffer>()      // A: [M,K]
      .Arg<ffi::AnyBuffer>()      // B: [N,K]
      .Ret<ffi::AnyBuffer>()      // out: [M,N]
      .Attr<int32_t>("rows_per_block"),
  {xla::ffi::Traits::kCmdBufferCompatible}
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
  WvSplitK, jax_aiter::WvSplitK_Bridge,
  ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<hipStream_t>>()
      .Arg<ffi::AnyBuffer>()      // A: [M,K]
      .Arg<ffi::AnyBuffer>()      // B: [K,N]
      .Ret<ffi::AnyBuffer>()      // out: [M,N]
      .Attr<int32_t>("N")
      .Attr<int32_t>("CuCount"),
  {xla::ffi::Traits::kCmdBufferCompatible}
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
  WvSplitKSmall, jax_aiter::WvSplitKSmall_Bridge,
  ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<hipStream_t>>()
      .Arg<ffi::AnyBuffer>()      // A: [M,K]
      .Arg<ffi::AnyBuffer>()      // B: [K,N]
      .Ret<ffi::AnyBuffer>()      // out: [M,N]
      .Attr<int32_t>("N")
      .Attr<int32_t>("CuCount"),
  {xla::ffi::Traits::kCmdBufferCompatible}
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
  WvSplitKQ, jax_aiter::WvSplitKQ_Bridge,
  ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<hipStream_t>>()
      .Arg<ffi::AnyBuffer>()      // A: [M,K] (FP8)
      .Arg<ffi::AnyBuffer>()      // B: [K,N] (FP8)
      .Ret<ffi::AnyBuffer>()      // out: [M,N] (F16/BF16)
      .Arg<ffi::AnyBuffer>()      // scale_a: [M,1] (F32)
      .Arg<ffi::AnyBuffer>()      // scale_b: [1,N] (F32)
      .Attr<int32_t>("CuCount"),
  {xla::ffi::Traits::kCmdBufferCompatible}
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
  LLZZ, jax_aiter::LLZZ_Bridge,
  ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<hipStream_t>>()
      .Arg<ffi::AnyBuffer>()      // A: [M,K] (F32)
      .Arg<ffi::AnyBuffer>()      // B: [K,N] (F32)
      .Ret<ffi::AnyBuffer>()      // out: [M,N] (F32)
      .Attr<int32_t>("solidx"),
  {xla::ffi::Traits::kCmdBufferCompatible}
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
  MMCustomGPU, jax_aiter::MMCustomGPU_Bridge,
  ffi::Ffi::Bind()
      .Ctx<ffi::PlatformStream<hipStream_t>>()
      .Arg<ffi::AnyBuffer>()      // A: [M,K] (F32)
      .Arg<ffi::AnyBuffer>()      // B: [K,N] (F32)
      .Ret<ffi::AnyBuffer>()      // out: [M,N] (F32)
);

#pragma GCC visibility pop
