// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Fused SiLU-and-Mul forward FFI handler.
// Input: [M, 2*D] (bf16/fp16).  Output: [M, D] (same dtype).
// Computes: out[i] = silu(input[i, :D]) * input[i, D:]
//
// Standalone kernel — no torch or CK tile dependency.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "hip_utils.h"

namespace ffi = xla::ffi;

namespace jax_aiter {

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ float silu_f32(float x) {
  return x / (1.0f + __expf(-x));
}

// ---------------------------------------------------------------------------
// BF16 fused SiLU-and-Mul kernel with v_pk_mul_f32 ASM
// ---------------------------------------------------------------------------

__global__ void silu_and_mul_bf16_kernel(
    __hip_bfloat16* __restrict__ out,         // [M, D]
    const __hip_bfloat16* __restrict__ input, // [M, 2*D]
    const int D) {

  const int64_t row = blockIdx.x;
  const __hip_bfloat16* gate = input + row * 2 * D;
  const __hip_bfloat16* up   = input + row * 2 * D + D;
  __hip_bfloat16* out_row    = out + row * D;

  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float g = __bfloat162float(gate[i]);
    float u = __bfloat162float(up[i]);
    float s = silu_f32(g);
    out_row[i] = __float2bfloat16(s * u);
  }
}

// ---------------------------------------------------------------------------
// FP16 fused SiLU-and-Mul kernel
// ---------------------------------------------------------------------------

__global__ void silu_and_mul_fp16_kernel(
    __half* __restrict__ out,         // [M, D]
    const __half* __restrict__ input, // [M, 2*D]
    const int D) {

  const int64_t row = blockIdx.x;
  const __half* gate = input + row * 2 * D;
  const __half* up   = input + row * 2 * D + D;
  __half* out_row    = out + row * D;

  for (int i = threadIdx.x; i < D; i += blockDim.x) {
    float g = __half2float(gate[i]);
    float u = __half2float(up[i]);
    float s = silu_f32(g);
    out_row[i] = __float2half(s * u);
  }
}

// ---------------------------------------------------------------------------
// Kernel launch helpers
// ---------------------------------------------------------------------------

static void launch_silu_and_mul_bf16(__hip_bfloat16* out,
                                     const __hip_bfloat16* input,
                                     int64_t M, int D,
                                     hipStream_t stream) {
  // Use enough threads to cover D elements per row.
  int block = (D < 1024) ? ((D + 63) / 64 * 64) : 1024;
  if (block < 64) block = 64;
  dim3 grid(M);
  dim3 blk(block);
  silu_and_mul_bf16_kernel<<<grid, blk, 0, stream>>>(out, input, D);
}

static void launch_silu_and_mul_fp16(__half* out,
                                     const __half* input,
                                     int64_t M, int D,
                                     hipStream_t stream) {
  int block = (D < 1024) ? ((D + 63) / 64 * 64) : 1024;
  if (block < 64) block = 64;
  dim3 grid(M);
  dim3 blk(block);
  silu_and_mul_fp16_kernel<<<grid, blk, 0, stream>>>(out, input, D);
}

// ---------------------------------------------------------------------------
// FFI Bridge
// ---------------------------------------------------------------------------

ffi::Error SiluAndMul_Bridge(
    hipStream_t stream,
    ffi::AnyBuffer input,               // [M, 2*D] bf16 or fp16
    ffi::Result<ffi::AnyBuffer> out) {  // [M, D]

  if (!input.untyped_data()) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "SiluAndMul: input buffer is null");
  }

  auto dims  = input.dimensions();
  auto dtype = input.element_type();

  if (dtype != ffi::DataType::BF16 && dtype != ffi::DataType::F16) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "SiluAndMul: only bf16 and fp16 supported");
  }

  // Last dim is 2*D.
  int64_t two_d = dims.back();
  if (two_d % 2 != 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "SiluAndMul: last dim must be even (2*D)");
  }
  int D = static_cast<int>(two_d / 2);

  // Flatten leading dims into M.
  int64_t M = 1;
  for (size_t i = 0; i < dims.size() - 1; ++i) {
    M *= dims[i];
  }

  if (dtype == ffi::DataType::BF16) {
    auto* in_ptr  = reinterpret_cast<const __hip_bfloat16*>(input.untyped_data());
    auto* out_ptr = reinterpret_cast<__hip_bfloat16*>(out->untyped_data());
    launch_silu_and_mul_bf16(out_ptr, in_ptr, M, D, stream);
  } else {
    auto* in_ptr  = reinterpret_cast<const __half*>(input.untyped_data());
    auto* out_ptr = reinterpret_cast<__half*>(out->untyped_data());
    launch_silu_and_mul_fp16(out_ptr, in_ptr, M, D, stream);
  }

  return ffi::Error::Success();
}

} // namespace jax_aiter

// ---------------------------------------------------------------------------
// XLA FFI handler registration
// ---------------------------------------------------------------------------

#pragma GCC visibility push(default)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    SiluAndMulJA, jax_aiter::SiluAndMul_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()   // input: [M, 2*D]
        .Ret<ffi::AnyBuffer>(),  // out: [M, D]
    {xla::ffi::Traits::kCmdBufferCompatible});

#pragma GCC visibility pop
