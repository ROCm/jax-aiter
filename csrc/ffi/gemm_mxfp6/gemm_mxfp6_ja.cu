// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Torch-free JAX FFI for gfx950 A6W6 MXFP6 GEMM.  The quantizer and ASM ABI
// are shared with AITER's gemm_op_a6w6.py.

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

#include <hip/hip_runtime.h>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "asm_f6gemm_configs.hpp"
#include "hip_utils.h"
#include "quant.h"

namespace ffi = xla::ffi;

namespace jax_aiter {
namespace {

constexpr int64_t kTile = 256;
constexpr int64_t kKTile = 128;
constexpr int64_t kKGuardTiles = 2;
constexpr int64_t kPackedTileBytes = 24576;
constexpr int64_t kScaleTileBytes = 1024;
constexpr int kMaxDevices = 16;
constexpr char kDefaultKernel[] = "_ZN5aiter29f6gemm_bf16_per1x32Fp6_dmabigE";

int64_t ceil_div(int64_t value, int64_t divisor) {
  return (value + divisor - 1) / divisor;
}

struct __attribute__((packed)) Mxfp6GemmKernelArgs {
  void* ptr_D;
  p2 _p0;
  void* ptr_C;
  p2 _p1;
  void* ptr_A;
  p2 _p2;
  void* ptr_B;
  p2 _p3;
  float alpha;
  p3 _p4;
  float beta;
  p3 _p5;
  unsigned int stride_D0;
  p3 _p6;
  unsigned int stride_D1;
  p3 _p7;
  unsigned int stride_C0;
  p3 _p8;
  unsigned int stride_C1;
  p3 _p9;
  unsigned int stride_A0;
  p3 _p10;
  unsigned int stride_A1;
  p3 _p11;
  unsigned int stride_B0;
  p3 _p12;
  unsigned int stride_B1;
  p3 _p13;
  unsigned int M;
  p3 _p14;
  unsigned int N;
  p3 _p15;
  unsigned int K;
  p3 _p16;
  void* ptr_ScaleA;
  p2 _p17;
  void* ptr_ScaleB;
  p2 _p18;
  unsigned int stride_ScaleA0;
  p3 _p19;
  unsigned int stride_ScaleA1;
  p3 _p20;
  unsigned int stride_ScaleB0;
  p3 _p21;
  unsigned int stride_ScaleB1;
  p3 _p22;
  int log2_k_split;
  p3 _p23;
};
static_assert(sizeof(Mxfp6GemmKernelArgs) == 0x180, "A6W6 kernarg ABI");

aiter_tensor_t tensor_from(void* ptr, std::initializer_list<int64_t> dims,
                           AiterDtype dtype, int device) {
  aiter_tensor_t tensor{};
  tensor.ptr = ptr;
  tensor.dtype_ = dtype;
  tensor.device_id = device;
  tensor.ndim = static_cast<int>(dims.size());
  tensor.numel_ = 1;
  int axis = 0;
  for (int64_t dim : dims) {
    tensor.shape[axis++] = dim;
    tensor.numel_ *= static_cast<size_t>(dim);
  }
  int64_t stride = 1;
  for (int axis = tensor.ndim - 1; axis >= 0; --axis) {
    tensor.strides[axis] = stride;
    stride *= tensor.shape[axis];
  }
  return tensor;
}

struct StreamGuard {
  hipStream_t previous;
  explicit StreamGuard(hipStream_t stream) : previous(aiter::getCurrentHIPStream()) {
    aiter::setCurrentHIPStream(stream);
  }
  ~StreamGuard() { aiter::setCurrentHIPStream(previous); }
};

bool u8_vector(const ffi::AnyBuffer& buffer, int64_t expected) {
  const auto dims = buffer.dimensions();
  return buffer.element_type() == ffi::DataType::U8 && dims.size() == 1 &&
         dims[0] == expected && static_cast<int64_t>(buffer.size_bytes()) == expected;
}

bool bf16_matrix(const ffi::AnyBuffer& buffer, int64_t rows, int64_t cols) {
  const auto dims = buffer.dimensions();
  return buffer.element_type() == ffi::DataType::BF16 && dims.size() == 2 &&
         dims[0] == rows && dims[1] == cols &&
         static_cast<int64_t>(buffer.size_bytes()) == rows * cols * 2;
}

struct KernelEntry {
  std::unique_ptr<AiterAsmKernel> kernel;
  int tile_m;
  int tile_n;
  int block_size;
  int swizzle_max_m;
  int swizzle_max_n;
  int swizzle_max_k;
  int exact_m;
  int exact_n;
  int exact_k;
};

using KernelCache = std::unordered_map<std::string, KernelEntry>;

KernelCache& kernel_cache(int device) {
  static KernelCache caches[kMaxDevices];
  static std::once_flag flags[kMaxDevices];
  std::call_once(flags[device], [device]() {
    HIP_CALL(hipSetDevice(device));
    auto& cache = caches[device];
    const std::string arch = get_gpu_arch();
    for (const auto& item : cfg_f6gemm_bf16_per1x32Fp6) {
      if (item.first.rfind(arch, 0) != 0) continue;
      const auto& cfg = item.second;
      if (cfg.splitK != 0 || cfg.pack_layout != "mxfp6_c0c1_256_padk2" ||
          cache.count(cfg.knl_name)) {
        continue;
      }
      cache.emplace(cfg.knl_name,
                    KernelEntry{std::make_unique<AiterAsmKernel>(cfg.knl_name.c_str(),
                                                                   cfg.co_name.c_str()),
                                cfg.tile_M, cfg.tile_N, cfg.block_size,
                                cfg.swizzle_max_M, cfg.swizzle_max_N, cfg.swizzle_max_K,
                                cfg.exact_M, cfg.exact_N, cfg.exact_K});
    }
  });
  return caches[device];
}

ffi::Error Mxfp6GemmQuant_Bridge(hipStream_t stream, ffi::AnyBuffer input,
                                  ffi::Result<ffi::AnyBuffer> packed,
                                  ffi::Result<ffi::AnyBuffer> scales) {
  const auto dims = input.dimensions();
  if (input.element_type() != ffi::DataType::BF16 || dims.size() != 2 || dims[0] <= 0 ||
      dims[1] <= 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "MXFP6 GEMM quantization requires a non-empty bfloat16 [rows, K]");
  }
  const int64_t rows = dims[0];
  const int64_t k = dims[1];
  const int64_t padded_rows = ceil_div(rows, kTile) * kTile;
  const int64_t padded_k = ceil_div(k, kKTile) * kKTile;
  const int64_t tiles = padded_rows / kTile * (padded_k / kKTile + kKGuardTiles);
  const int64_t packed_size = tiles * kPackedTileBytes;
  const int64_t scale_size = tiles * kScaleTileBytes;
  if (!u8_vector(*packed, packed_size) || !u8_vector(*scales, scale_size)) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "MXFP6 GEMM packed buffers have an invalid shape or size");
  }
  const int device = device_from_ptr(input.untyped_data());
  if (device < 0 || device >= kMaxDevices || hipSetDevice(device) != hipSuccess) {
    return ffi::Error(ffi::ErrorCode::kInternal, "Unable to select the MXFP6 GEMM input device");
  }
  try {
    StreamGuard guard(stream);
    auto src = tensor_from(const_cast<void*>(input.untyped_data()), {rows, k}, AITER_DTYPE_bf16,
                           device);
    auto dst = tensor_from(packed->untyped_data(), {packed_size}, AITER_DTYPE_u8, src.device_id);
    auto scale =
        tensor_from(scales->untyped_data(), {scale_size}, AITER_DTYPE_u8, src.device_id);
    aiter::quant_mxfp6_gemm_hip(src, dst, scale);
  } catch (const std::exception& error) {
    return ffi::Error(ffi::ErrorCode::kInternal, error.what());
  }
  return ffi::Error::Success();
}

ffi::Error Mxfp6GemmFwd_Bridge(hipStream_t stream, ffi::AnyBuffer a, ffi::AnyBuffer b,
                                ffi::AnyBuffer a_scale, ffi::AnyBuffer b_scale,
                                ffi::Result<ffi::AnyBuffer> out, int64_t m, int64_t n,
                                int64_t k, std::string_view kernel_name_attr) {
  if (m <= 0 || n <= 0 || k <= 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "MXFP6 GEMM dimensions must be positive");
  }
  const int64_t padded_m = ceil_div(m, kTile) * kTile;
  const int64_t padded_n = ceil_div(n, kTile) * kTile;
  const int64_t padded_k = ceil_div(k, kKTile) * kKTile;
  const int64_t a_tiles = padded_m / kTile * (padded_k / kKTile + kKGuardTiles);
  const int64_t b_tiles = padded_n / kTile * (padded_k / kKTile + kKGuardTiles);
  if (!u8_vector(a, a_tiles * kPackedTileBytes) ||
      !u8_vector(b, b_tiles * kPackedTileBytes) ||
      !u8_vector(a_scale, a_tiles * kScaleTileBytes) ||
      !u8_vector(b_scale, b_tiles * kScaleTileBytes) ||
      !bf16_matrix(*out, padded_m, padded_n)) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "MXFP6 GEMM packed operands or padded output have invalid shapes");
  }
  int device = device_from_ptr(a.untyped_data());
  if (device < 0 || device >= kMaxDevices || hipSetDevice(device) != hipSuccess) {
    return ffi::Error(ffi::ErrorCode::kInternal, "Unable to select the MXFP6 GEMM input device");
  }
  if (device_from_ptr(b.untyped_data()) != device ||
      device_from_ptr(a_scale.untyped_data()) != device ||
      device_from_ptr(b_scale.untyped_data()) != device ||
      device_from_ptr(out->untyped_data()) != device) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "MXFP6 GEMM operands and output must be on one device");
  }
  const std::string kernel_name =
      kernel_name_attr.empty() ? kDefaultKernel : std::string(kernel_name_attr);
  auto& cache = kernel_cache(device);
  const auto it = cache.find(kernel_name);
  if (it == cache.end()) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "Unknown or unsupported MXFP6 GEMM kernel: " + kernel_name);
  }
  const auto& entry = it->second;
  if (entry.exact_k > 0 &&
      (padded_m != entry.exact_m || padded_n != entry.exact_n || padded_k != entry.exact_k)) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "MXFP6 GEMM kernel is restricted to a different exact padded shape");
  }
  if (entry.swizzle_max_k > 0 && padded_k <= entry.swizzle_max_k &&
      (padded_m > entry.swizzle_max_m || padded_n > entry.swizzle_max_n)) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "MXFP6 GEMM shape exceeds the selected kernel's swizzle bounds");
  }
  Mxfp6GemmKernelArgs args{};
  args.ptr_D = out->untyped_data();
  args.ptr_A = const_cast<void*>(a.untyped_data());
  args.ptr_B = const_cast<void*>(b.untyped_data());
  args.alpha = 1.0f;
  args.stride_D0 = static_cast<unsigned int>(padded_n);
  args.M = static_cast<unsigned int>(padded_m);
  args.N = static_cast<unsigned int>(padded_n);
  args.K = static_cast<unsigned int>(padded_k);
  args.ptr_ScaleA = const_cast<void*>(a_scale.untyped_data());
  args.ptr_ScaleB = const_cast<void*>(b_scale.untyped_data());
  args.stride_ScaleA0 = static_cast<unsigned int>(padded_k / 32);
  args.stride_ScaleB0 = static_cast<unsigned int>(padded_k / 32);
  size_t arg_size = sizeof(args);
  try {
    entry.kernel->launch_kernel({&args, &arg_size, static_cast<int>(padded_n / entry.tile_n),
                                 static_cast<int>(padded_m / entry.tile_m), 1,
                                 entry.block_size, 1, 1, stream});
  } catch (const std::exception& error) {
    return ffi::Error(ffi::ErrorCode::kInternal, error.what());
  }
  return ffi::Error::Success();
}

}  // namespace
}  // namespace jax_aiter

#pragma GCC visibility push(default)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Mxfp6GemmQuantJA, jax_aiter::Mxfp6GemmQuant_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>(),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Mxfp6GemmFwdJA, jax_aiter::Mxfp6GemmFwd_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Attr<int64_t>("m")
        .Attr<int64_t>("n")
        .Attr<int64_t>("k")
        .Attr<std::string_view>("kernel_name"),
    {xla::ffi::Traits::kCmdBufferCompatible});

#pragma GCC visibility pop
