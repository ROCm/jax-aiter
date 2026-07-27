// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Fused BF16 -> MXFP4 cast + transpose + shuffle FFI handlers.
// Uses cast_transpose_mxfp4_kernel_shuffled.cu for JAX FFI.
//
// CastMxfp4JA:     Rowwise-only output (activation + weight quantization).
//                   Single kernel launch for any M.  The kernel uses int64
//                   addressing to handle M*N > INT32_MAX (e.g. 70B batch=10).
// CastMxfp4DualJA: Rowwise + columnwise output in one launch. Used by the
//                   weight, activation AND gradient dual casts, so M is the
//                   token count for the latter two and int64 addressing applies.

#include <hip/hip_runtime.h>
#include <cstdint>

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
    hipStream_t stream
);
}

namespace jax_aiter {

static inline int cdiv(int a, int b) { return (a + b - 1) / b; }

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
    ffi::Result<ffi::AnyBuffer> rowwise_fp4_out,
    ffi::Result<ffi::AnyBuffer> rowwise_scale_out
) {
  auto dims = input.dimensions();
  int M = static_cast<int>(dims[0]);
  int K = static_cast<int>(dims[1]);

  constexpr int BLOCK_SIZE = 32;

  // #8: host-side shape/alignment guard (once per launch, no per-thread cost).
  // The MXFP4 32-block layout requires both dims to be multiples of 32, and the
  // vectorized bf16 loads require 8-byte input alignment. Fail before launch.
  if (M % BLOCK_SIZE || K % BLOCK_SIZE ||
      (reinterpret_cast<uintptr_t>(input.untyped_data()) % 8)) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
        "cast_mxfp4: M,K must be multiples of 32 and input 8B-aligned");
  }

  int scale_N = cdiv(K, BLOCK_SIZE);
  int scale_M_pad = cdiv(M, 256) * 256;
  int scale_N_pad = cdiv(scale_N, 8) * 8;

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
      stream);

  return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
// CastMxfp4DualJA: Rowwise + columnwise output.
// Serves three callers, not just weights: the weight dual cast (M = N_weight,
// FSDP-sharded, so max M = 28672/8 = 3584), plus the activation and gradient
// dual casts where M is the local token count (32768 at the 8B matched recipe).
// ---------------------------------------------------------------------------
ffi::Error CastMxfp4Dual_Bridge(
    hipStream_t stream,
    ffi::AnyBuffer input,
    bool shuffle_fp4,
    bool shuffle_colwise_fp4,
    bool use_hadamard,        // rowwise-direction Hadamard
    bool use_hadamard_col,    // colwise-direction Hadamard (independent)
    bool use_sr,              // rowwise-direction SR
    bool use_sr_col,          // colwise-direction SR (independent)
    int scale_margin,
    int scale_mode,
    bool use_2d_scale,
    ffi::Result<ffi::AnyBuffer> rowwise_fp4_out,
    ffi::Result<ffi::AnyBuffer> rowwise_scale_out,
    ffi::Result<ffi::AnyBuffer> colwise_fp4_out,
    ffi::Result<ffi::AnyBuffer> colwise_scale_out
) {
  auto dims = input.dimensions();
  int M = static_cast<int>(dims[0]);
  int K = static_cast<int>(dims[1]);

  constexpr int BLOCK_SIZE = 32;

  // #8: host-side shape/alignment guard (once per launch, no per-thread cost).
  // The MXFP4 32-block layout requires both dims to be multiples of 32, and the
  // vectorized bf16 loads require 8-byte input alignment. Fail before launch.
  if (M % BLOCK_SIZE || K % BLOCK_SIZE ||
      (reinterpret_cast<uintptr_t>(input.untyped_data()) % 8)) {
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

  int rowwise_scale_N = cdiv(K, BLOCK_SIZE);
  int rowwise_scale_M_pad = cdiv(M, 256) * 256;
  int rowwise_scale_N_pad = cdiv(rowwise_scale_N, 8) * 8;

  int colwise_scale_M = K;
  int colwise_scale_N = cdiv(M, BLOCK_SIZE);
  int colwise_scale_M_pad = cdiv(K, 256) * 256;
  int colwise_scale_N_pad = cdiv(colwise_scale_N, 8) * 8;

  int rowwise_scale_stride = rowwise_scale_N_pad;
  int colwise_scale_stride = colwise_scale_N_pad;

  mxfp4::launch_cast_transpose_mxfp4_shuffled(
      input.untyped_data(),
      rowwise_fp4_out->untyped_data(),
      rowwise_scale_out->untyped_data(),
      colwise_fp4_out->untyped_data(),
      colwise_scale_out->untyped_data(),
      M, K,
      true,              // use_rowwise
      true,              // use_colwise
      true,              // shuffle_scales
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
        .Attr<bool>("use_hadamard")   // rowwise Hadamard
        .Attr<bool>("use_hadamard_col")  // colwise Hadamard (independent of row)
        .Attr<bool>("use_sr")         // rowwise SR (default false = RNE)
        .Attr<bool>("use_sr_col")     // colwise SR (independent of row)
        .Attr<int>("scale_margin")    // E8M0 under-flush headroom (default 0 = legacy exp-2)
        .Attr<int>("scale_mode")      // 0 = round-nearest (legacy); 1 = OAS floor
        .Attr<bool>("use_2d_scale")   // share one 32x32-tile scale across row+col
        .Ret<ffi::AnyBuffer>()        // rowwise_fp4:  [M, K/2] uint8
        .Ret<ffi::AnyBuffer>()        // rowwise_scale: [M_pad, rscale_N_pad] uint8
        .Ret<ffi::AnyBuffer>()        // colwise_fp4:  [K, M/2] uint8
        .Ret<ffi::AnyBuffer>(),       // colwise_scale: [K_pad, cscale_N_pad] uint8
    {xla::ffi::Traits::kCmdBufferCompatible});

#pragma GCC visibility pop
