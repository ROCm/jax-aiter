// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// NEOX RoPE forward FFI handler (self-contained, torch-free).
// Computes out = x * cos + rotate_half(x) * sin, matching MaxText's
// RotaryEmbedding.apply_rotary. Uses rope_kernel.cu's launcher.
//
// Inputs:
//   x:   [B, S, N, D] bf16  (BSHD; query or key after projection)
//   cos: [B, S, D]    bf16  (full-width, shared across the N heads)
//   sin: [B, S, D]    bf16
// Output:
//   out: [B, S, N, D] bf16
//
// Command-buffer compatible: only hipModuleLaunch via the kernel launch.

#include <hip/hip_runtime.h>
#include <cstdint>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace ja_rope {
extern "C" void launch_rope_neox_fwd_bf16(
    const void* x, const void* cos, const void* sin, void* out,
    int64_t n_rows, int n_heads, int D, hipStream_t stream);
}

namespace jax_aiter {

ffi::Error
RopeFwd_Bridge(
    hipStream_t stream,
    ffi::AnyBuffer x,
    ffi::AnyBuffer cos,
    ffi::AnyBuffer sin,
    ffi::Result<ffi::AnyBuffer> out) {

  if (!x.untyped_data() || !cos.untyped_data() || !sin.untyped_data()) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "RoPE: required input buffer (x/cos/sin) is null");
  }

  auto xd = x.dimensions();
  if (xd.size() != 4) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "RoPE expects x of rank 4 [B, S, N, D]");
  }
  if (x.element_type() != ffi::DataType::BF16 ||
      cos.element_type() != ffi::DataType::BF16 ||
      sin.element_type() != ffi::DataType::BF16) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "RoPE only supports bf16 x/cos/sin");
  }

  int64_t B = xd[0], S = xd[1], N = xd[2], D = xd[3];
  if ((D & 1) != 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "RoPE head_dim D must be even");
  }

  // cos/sin must be [B, S, D] (full-width, shared across heads).
  auto cd = cos.dimensions();
  if (cd.size() != 3 || cd[0] != B || cd[1] != S || cd[2] != D) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "RoPE cos must be [B, S, D] matching x");
  }
  auto sd = sin.dimensions();
  if (sd.size() != 3 || sd[0] != B || sd[1] != S || sd[2] != D) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "RoPE sin must be [B, S, D] matching x");
  }

  int64_t n_rows = B * S * N;
  ja_rope::launch_rope_neox_fwd_bf16(
      x.untyped_data(), cos.untyped_data(), sin.untyped_data(),
      out->untyped_data(), n_rows, static_cast<int>(N),
      static_cast<int>(D), stream);

  return ffi::Error::Success();
}

}  // namespace jax_aiter

#pragma GCC visibility push(default)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RopeFwdJA, jax_aiter::RopeFwd_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()   // x:   [B, S, N, D] bf16
        .Arg<ffi::AnyBuffer>()   // cos: [B, S, D] bf16
        .Arg<ffi::AnyBuffer>()   // sin: [B, S, D] bf16
        .Ret<ffi::AnyBuffer>(),  // out: [B, S, N, D] bf16
    {xla::ffi::Traits::kCmdBufferCompatible});

#pragma GCC visibility pop
