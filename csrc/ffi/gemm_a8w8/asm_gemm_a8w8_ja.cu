// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>
#include <torch/all.h>

#include "asm_gemm_a8w8.h"
#include "xla/ffi/api/ffi.h"

#include "hip_utils.h"
#include "logging.h"
#include "torch_utils.h"

namespace ffi = xla::ffi;

namespace jax_aiter {

ffi::Error GemmA8W8_Bridge(hipStream_t stream, ffi::AnyBuffer A,
                           ffi::AnyBuffer B, ffi::AnyBuffer A_scale,
                           ffi::AnyBuffer B_scale,
                           ffi::Result<ffi::AnyBuffer> out, ffi::AnyBuffer bias,
                           int32_t sub_m = 128, int32_t sub_n = 128,
                           int32_t pad_a = 0, int32_t pad_b = 0,
                           int32_t pad_c = 0, int32_t splitK = 0) {
  auto dims_A = A.dimensions();
  auto dims_out = out->dimensions();
  int64_t m = dims_A[0];
  int64_t k = dims_A[1];
  int64_t n = dims_out[1];

  JA_LOG("GEMM (%ld*%ld) . (%ld*%ld) -> (%ld*%ld)  splitK=%d", m, k, k, n, m, n,
         splitK);

  // Be precise and use the "current_device".
  const int dev_idx = ::jax_aiter::device_from_ptr(A.untyped_data());

  // Wrap JAX buffers into torch Tensors.
  auto A_t = wrap_any_buffer(A, dev_idx);
  auto B_t = wrap_any_buffer(B, dev_idx);
  auto A_scale_t = wrap_any_buffer(A_scale, dev_idx);
  auto B_scale_t = wrap_any_buffer(B_scale, dev_idx);
  auto out_t = wrap_any_buffer(*out, dev_idx);
  auto bias_t = wrap_any_buffer(bias, dev_idx);

  gemm_a8w8_asm(A_t, B_t, A_scale_t, B_scale_t, out_t, bias_t, sub_m, sub_n,
                pad_a, pad_b, pad_c, splitK);

  return ffi::Error::Success();
}

} // namespace jax_aiter

#pragma GCC visibility push(default)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    GemmA8W8JA, jax_aiter::GemmA8W8_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>() // A: [m, k] (S8)
        .Arg<ffi::AnyBuffer>() // B: [n, k] (S8)
        .Arg<ffi::AnyBuffer>() // A_scale: [m, 1] (F32)
        .Arg<ffi::AnyBuffer>() // B_scale: [1, n] (F32)
        .Ret<ffi::AnyBuffer>() // out: [m, n] (BF16)
        .Arg<ffi::AnyBuffer>() // bias: [1, n] (F32)
        .Attr<int32_t>("sub_m")
        .Attr<int32_t>("sub_n")
        .Attr<int32_t>("pad_a")
        .Attr<int32_t>("pad_b")
        .Attr<int32_t>("pad_c")
        .Attr<int32_t>("splitK"),
    {xla::ffi::Traits::kCmdBufferCompatible});

#pragma GCC visibility pop
