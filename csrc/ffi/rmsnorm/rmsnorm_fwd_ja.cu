// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// RMSNorm forward FFI handler calling rmsnorm2d_fwd from CK.
// Input: x [m, n], gamma [n]. Output: y [m, n], inv_rms [m].

#include <hip/hip_runtime.h>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "hip_utils.h"
#include "rmsnorm2d_fwd.hpp"

namespace ffi = xla::ffi;

namespace jax_aiter {

static std::string dtype_to_prec(ffi::DataType dtype) {
  switch (dtype) {
  case ffi::DataType::F16:  return "fp16";
  case ffi::DataType::BF16: return "bf16";
  case ffi::DataType::F32:  return "fp32";
  default:
    return "fp16";
  }
}

ffi::Error
RmsnormFwd_Bridge(
    hipStream_t stream,
    ffi::AnyBuffer x,
    ffi::AnyBuffer gamma,
    ffi::Result<ffi::AnyBuffer> y,
    ffi::Result<ffi::AnyBuffer> inv_rms,
    float epsilon,
    bool save_rms) {

  if (!x.untyped_data() || !gamma.untyped_data()) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "Required input buffer (x/gamma) is null");
  }

  auto x_dims = x.dimensions();
  auto x_dtype = x.element_type();

  if (x_dtype != ffi::DataType::F16 && x_dtype != ffi::DataType::BF16) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "RMSNorm only supports fp16 and bf16 input");
  }

  // Flatten leading dims to m.
  int64_t n = x_dims.back();
  int64_t m = 1;
  for (size_t i = 0; i < x_dims.size() - 1; ++i) {
    m *= x_dims[i];
  }

  auto gamma_dims = gamma.dimensions();
  int64_t gamma_n = gamma_dims.back();
  if (gamma_n != n) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "gamma size must match last dim of x");
  }

  std::string prec = dtype_to_prec(x_dtype);

  ck_tile::stream_config stream_config{stream, false, 0, -1, -1};

  rmsnorm2d_fwd_traits traits{
      .prec_i = prec,
      .prec_o = prec,
      .prec_sm = prec,
      .prec_sy = prec,
      .save_rms = save_rms,
      .save_unquant = false,
      .fused_add = 0,
      .fused_quant = 0,
      .use_model_sensitive_rmsnorm = 0
  };

  rmsnorm2d_fwd_args args{};
  args.p_x         = x.untyped_data();
  args.p_x_residual = nullptr;
  args.p_sm_scale  = nullptr;
  args.p_gamma     = gamma.untyped_data();
  args.p_y         = y->untyped_data();
  args.p_y_residual = nullptr;
  args.p_y_scale   = nullptr;
  args.p_invRms    = save_rms ? inv_rms->untyped_data() : nullptr;
  args.p_y_unquant = nullptr;
  args.epsilon     = epsilon;
  args.m           = static_cast<ck_tile::index_t>(m);
  args.n           = static_cast<ck_tile::index_t>(n);
  args.x_stride    = static_cast<ck_tile::index_t>(n);
  args.xr_stride   = 0;
  args.y_stride    = static_cast<ck_tile::index_t>(n);
  args.yr_stride   = 0;

  float elapsed = rmsnorm2d_fwd(traits, args, stream_config);

  if (elapsed < 0) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      "rmsnorm2d_fwd failed - unsupported configuration");
  }

  return ffi::Error::Success();
}

} // namespace jax_aiter

#pragma GCC visibility push(default)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RmsnormFwdJA, jax_aiter::RmsnormFwd_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()   // x: [m, n]
        .Arg<ffi::AnyBuffer>()   // gamma: [n]
        .Ret<ffi::AnyBuffer>()   // y: [m, n]
        .Ret<ffi::AnyBuffer>()   // inv_rms: [m]
        .Attr<float>("epsilon")
        .Attr<bool>("save_rms"),
    {xla::ffi::Traits::kCmdBufferCompatible});

#pragma GCC visibility pop
