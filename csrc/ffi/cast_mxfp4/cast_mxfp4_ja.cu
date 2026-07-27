// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Fused BF16 -> MXFP4 cast + transpose + shuffle FFI handlers.
// Uses cast_transpose_mxfp4_kernel_shuffled.cu for JAX FFI.
//
// CastMxfp4JA:     Rowwise-only output (activation + weight quantization).
//                   Generic calls retain the baseline int64 addressing.
// CastMxfp4DualJA: Rowwise + columnwise activation/weight/gradient output.
//                   Auto may select guarded uint32 only for the two approved
//                   production templates; every fallback retains baseline int64.

#include <hip/hip_runtime.h>
#include <cstdint>

#include "cast_mxfp4_offset_guard.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace mxfp4 {
extern "C" void launch_cast_transpose_mxfp4_shuffled(
    const void* input,
    void* rowwise_fp4,
    void* rowwise_scale,
    void* colwise_fp4,
    void* colwise_scale,
    int M, int N,
    bool use_rowwise,
    bool use_colwise,
    bool shuffle_scales,
    bool use_hadamard_row,
    bool use_hadamard_col,
    bool shuffle_rowwise_fp4,
    bool shuffle_colwise_fp4,
    bool use_sr_row,
    bool use_sr_col,
    int rowwise_scale_stride,
    int colwise_scale_stride,
    int rowwise_scale_N,
    int rowwise_scale_M_pad,
    int rowwise_scale_N_pad,
    int colwise_scale_M,
    int colwise_scale_N,
    int colwise_scale_M_pad,
    int colwise_scale_N_pad,
    int scale_margin,
    int scale_mode,
    bool use_2d_scale,
    offset_guard::OffsetType offset_type,
    hipStream_t stream
);
}

namespace jax_aiter {

// ---------------------------------------------------------------------------
// CastMxfp4JA: Rowwise output (single kernel launch for any M)
// ---------------------------------------------------------------------------
ffi::Error CastMxfp4_Bridge(
    hipStream_t stream,
    ffi::AnyBuffer input,
    bool shuffle_fp4,
    bool shuffle_scales,
    bool use_hadamard,
    bool use_sr,
    int scale_margin,
    int scale_mode,
    bool use_2d_scale,
    int offset_mode,
    ffi::Result<ffi::AnyBuffer> rowwise_fp4_out,
    ffi::Result<ffi::AnyBuffer> rowwise_scale_out
) {
  auto dims = input.dimensions();
  if (dims.size() != 2) {
    return ffi::Error(
        ffi::ErrorCode::kInvalidArgument,
        "cast_mxfp4: input must be rank 2");
  }
  const int64_t M_wide = dims[0];
  const int64_t K_wide = dims[1];
  const mxfp4::offset_guard::LayoutFlags layout{
      true, false, shuffle_scales, shuffle_fp4, false};
  const auto guard =
      mxfp4::offset_guard::evaluate_offset_guard(M_wide, K_wide, layout);
  if (guard.status == mxfp4::offset_guard::GuardStatus::kDimensionMisaligned) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
        "cast_mxfp4: M,K must be multiples of 32 and input 8B-aligned");
  }
  if (guard.status != mxfp4::offset_guard::GuardStatus::kOk) {
    return ffi::Error(
        ffi::ErrorCode::kInvalidArgument,
        "cast_mxfp4: dimensions and derived strides must fit the positive "
        "int kernel contract without arithmetic overflow");
  }
  if (reinterpret_cast<uintptr_t>(input.untyped_data()) % 8) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
        "cast_mxfp4: M,K must be multiples of 32 and input 8B-aligned");
  }
  const auto selection = mxfp4::offset_guard::select_offset_type(
      offset_mode, guard.u32_safe, false);
  if (!selection.valid_mode) {
    return ffi::Error(
        ffi::ErrorCode::kInvalidArgument,
        "cast_mxfp4: offset_mode must be 0 (off), 1 (auto), or 2 "
        "(force64)");
  }

  const int M = static_cast<int>(M_wide);
  const int K = static_cast<int>(K_wide);
  const int scale_N = static_cast<int>(guard.row_scale_n);
  const int scale_M_pad = static_cast<int>(guard.m_pad);
  const int scale_N_pad = static_cast<int>(guard.row_scale_n_pad);

  mxfp4::launch_cast_transpose_mxfp4_shuffled(
      input.untyped_data(),
      rowwise_fp4_out->untyped_data(),
      rowwise_scale_out->untyped_data(),
      nullptr, nullptr,          // no colwise
      M, K,
      true, false,               // rowwise only
      shuffle_scales,
      use_hadamard,              // rowwise Hadamard
      false,                     // colwise Hadamard (unused; colwise off)
      shuffle_fp4,
      false,                     // no colwise shuffle
      use_sr,                    // rowwise SR (default false = RNE)
      false,                     // colwise SR (unused; colwise off)
      scale_N_pad,
      0,                         // colwise stride (unused)
      scale_N, scale_M_pad, scale_N_pad,
      0, 0, 0, 0,               // colwise params (unused)
      scale_margin,              // E8M0 under-flush headroom (default 0 = legacy)
      scale_mode,                // 0 = round-nearest (legacy); 1 = OAS floor
      use_2d_scale,              // ignored for rowwise-only (no colwise to share)
      selection.type,            // rowwise/generic calls have no u32 specialization
      stream);

  return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
// CastMxfp4DualJA: Rowwise + columnwise output (weight dual quant).
// Only used for weight quantization where M = N_weight (FSDP-sharded).
// With 8-way FSDP, max M = 28672/8 = 3584.  No int64 addressing needed.
// ---------------------------------------------------------------------------
ffi::Error CastMxfp4Dual_Bridge(
    hipStream_t stream,
    ffi::AnyBuffer input,
    bool shuffle_fp4,
    bool shuffle_colwise_fp4,
    bool shuffle_scales,      // shuffle E8M0 scale layout (both directions)
    bool use_hadamard,        // rowwise-direction Hadamard
    bool use_hadamard_col,    // colwise-direction Hadamard (independent)
    bool use_sr,              // rowwise-direction SR
    bool use_sr_col,          // colwise-direction SR (independent)
    int scale_margin,
    int scale_mode,
    bool use_2d_scale,
    int offset_mode,
    ffi::Result<ffi::AnyBuffer> rowwise_fp4_out,
    ffi::Result<ffi::AnyBuffer> rowwise_scale_out,
    ffi::Result<ffi::AnyBuffer> colwise_fp4_out,
    ffi::Result<ffi::AnyBuffer> colwise_scale_out
) {
  auto dims = input.dimensions();
  if (dims.size() != 2) {
    return ffi::Error(
        ffi::ErrorCode::kInvalidArgument,
        "cast_mxfp4: input must be rank 2");
  }
  const int64_t M_wide = dims[0];
  const int64_t K_wide = dims[1];
  const mxfp4::offset_guard::LayoutFlags layout{
      true, true, shuffle_scales, shuffle_fp4, shuffle_colwise_fp4};
  const auto guard =
      mxfp4::offset_guard::evaluate_offset_guard(M_wide, K_wide, layout);
  if (guard.status == mxfp4::offset_guard::GuardStatus::kDimensionMisaligned) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
        "cast_mxfp4: M,K must be multiples of 32 and input 8B-aligned");
  }
  if (guard.status != mxfp4::offset_guard::GuardStatus::kOk) {
    return ffi::Error(
        ffi::ErrorCode::kInvalidArgument,
        "cast_mxfp4: dimensions and derived strides must fit the positive "
        "int kernel contract without arithmetic overflow");
  }
  if (reinterpret_cast<uintptr_t>(input.untyped_data()) % 8) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
        "cast_mxfp4: M,K must be multiples of 32 and input 8B-aligned");
  }

  // use_2d_scale shares ONE 32x32-tile E8M0 scale across the rowwise + colwise
  // casts so both emit identical FP4 codes (W_fprop == W_dgrad). That guarantee
  // holds ONLY under deterministic RNE on the raw values, so it is mutually
  // exclusive with Hadamard and SR:
  //   * Hadamard (#2): the 2D tile-amax is reduced from the RAW pre-Hadamard
  //     values, but the quantized values are post-Hadamard. Hadamard preserves
  //     L2 but not L-inf, so the shared scale mis-normalizes; and row/col
  //     Hadamard are independent, so a shared scale cannot yield matching codes.
  //   * SR (#3): stochastic rounding draws independent per-direction dither, so
  //     rowwise and colwise diverge even with an identical scale and values.
  // Reject the combos rather than silently emit a wrong / asymmetric cast.
  if (use_2d_scale &&
      (use_hadamard || use_hadamard_col || use_sr || use_sr_col)) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
        "cast_mxfp4: use_2d_scale is incompatible with Hadamard or SR "
        "(2D tile-amax is pre-Hadamard; SR breaks rowwise==colwise codes)");
  }

  const mxfp4::offset_guard::TemplateFlags template_flags{
      true, true, shuffle_scales, use_hadamard, use_hadamard_col,
      shuffle_fp4, shuffle_colwise_fp4, use_sr, use_sr_col};
  const auto selection = mxfp4::offset_guard::select_offset_type(
      offset_mode, guard.u32_safe,
      mxfp4::offset_guard::is_u32_specialized_template(template_flags));
  if (!selection.valid_mode) {
    return ffi::Error(
        ffi::ErrorCode::kInvalidArgument,
        "cast_mxfp4: offset_mode must be 0 (off), 1 (auto), or 2 "
        "(force64)");
  }

  const int M = static_cast<int>(M_wide);
  const int K = static_cast<int>(K_wide);
  const int rowwise_scale_N = static_cast<int>(guard.row_scale_n);
  const int rowwise_scale_M_pad = static_cast<int>(guard.m_pad);
  const int rowwise_scale_N_pad =
      static_cast<int>(guard.row_scale_n_pad);
  const int colwise_scale_M = K;
  const int colwise_scale_N = static_cast<int>(guard.col_scale_n);
  const int colwise_scale_M_pad = static_cast<int>(guard.k_pad);
  const int colwise_scale_N_pad =
      static_cast<int>(guard.col_scale_n_pad);
  const int rowwise_scale_stride = rowwise_scale_N_pad;
  const int colwise_scale_stride = colwise_scale_N_pad;

  mxfp4::launch_cast_transpose_mxfp4_shuffled(
      input.untyped_data(),
      rowwise_fp4_out->untyped_data(),
      rowwise_scale_out->untyped_data(),
      colwise_fp4_out->untyped_data(),
      colwise_scale_out->untyped_data(),
      M, K,
      true,              // use_rowwise
      true,              // use_colwise
      shuffle_scales,    // shuffle_scales (plumbed; was hardcoded true). Passing
                         // false emits LINEAR scales for BOTH directions so a
                         // per-shard colwise scale concatenates cleanly for the
                         // packed dgrad all-gather (Fix 2). Default true keeps
                         // every existing caller byte-identical.
      use_hadamard,          // rowwise Hadamard
      use_hadamard_col,      // colwise Hadamard (independent of row)
      shuffle_fp4,           // shuffle_rowwise_fp4
      shuffle_colwise_fp4,   // shuffle_colwise_fp4
      use_sr,                // rowwise SR (default false = RNE)
      use_sr_col,            // colwise SR (independent of row)
      rowwise_scale_stride,
      colwise_scale_stride,
      rowwise_scale_N,
      rowwise_scale_M_pad,
      rowwise_scale_N_pad,
      colwise_scale_M,
      colwise_scale_N,
      colwise_scale_M_pad,
      colwise_scale_N_pad,
      scale_margin,              // E8M0 under-flush headroom (default 0 = legacy)
      scale_mode,                // 0 = round-nearest (legacy); 1 = OAS floor
      use_2d_scale,              // share one 32x32-tile scale across row+col
      selection.type,
      stream
  );

  return ffi::Error::Success();
}

}  // namespace jax_aiter

#pragma GCC visibility push(default)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CastMxfp4JA, jax_aiter::CastMxfp4_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()        // input: [M, K] bf16
        .Attr<bool>("shuffle_fp4")    // shuffle rowwise FP4 data
        .Attr<bool>("shuffle_scales") // shuffle E8M0 scale layout
        .Attr<bool>("use_hadamard")   // apply Hadamard transform
        .Attr<bool>("use_sr")         // stochastic rounding (default false = RNE)
        .Attr<int>("scale_margin")    // E8M0 under-flush headroom (default 0 = legacy exp-2)
        .Attr<int>("scale_mode")      // 0 = round-nearest (legacy); 1 = OAS floor
        .Attr<bool>("use_2d_scale")   // share one 32x32-tile scale (no-op rowwise-only)
        .Attr<int>("offset_mode")     // 0=off, 1=auto, 2=force64
        .Ret<ffi::AnyBuffer>()        // rowwise_fp4: [M, K/2] uint8
        .Ret<ffi::AnyBuffer>(),       // rowwise_scale: [M_pad, scale_N_pad] uint8
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    CastMxfp4DualJA, jax_aiter::CastMxfp4Dual_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()        // input: [M, K] bf16
        .Attr<bool>("shuffle_fp4")    // shuffle rowwise FP4 data
        .Attr<bool>("shuffle_colwise_fp4")  // shuffle colwise FP4 data
        .Attr<bool>("shuffle_scales") // shuffle E8M0 scale layout (both directions)
        .Attr<bool>("use_hadamard")   // rowwise Hadamard
        .Attr<bool>("use_hadamard_col")  // colwise Hadamard (independent of row)
        .Attr<bool>("use_sr")         // rowwise SR (default false = RNE)
        .Attr<bool>("use_sr_col")     // colwise SR (independent of row)
        .Attr<int>("scale_margin")    // E8M0 under-flush headroom (default 0 = legacy exp-2)
        .Attr<int>("scale_mode")      // 0 = round-nearest (legacy); 1 = OAS floor
        .Attr<bool>("use_2d_scale")   // share one 32x32-tile scale across row+col
        .Attr<int>("offset_mode")     // 0=off, 1=auto, 2=force64
        .Ret<ffi::AnyBuffer>()        // rowwise_fp4:  [M, K/2] uint8
        .Ret<ffi::AnyBuffer>()        // rowwise_scale: [M_pad, rscale_N_pad] uint8
        .Ret<ffi::AnyBuffer>()        // colwise_fp4:  [K, M/2] uint8
        .Ret<ffi::AnyBuffer>(),       // colwise_scale: [K_pad, cscale_N_pad] uint8
    {xla::ffi::Traits::kCmdBufferCompatible});

#pragma GCC visibility pop
