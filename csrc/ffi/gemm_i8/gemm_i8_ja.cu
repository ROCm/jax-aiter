// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// INT8 GEMM FFI handler using AITER ASM kernels.
// gfx942 (MI300) only -- no gfx950 kernels.
// Out[M,N] bf16 = dequant(A[M,K] i8, a_scale) @ dequant(B[N,K] i8, b_scale)^T
// Requires asm_i8gemm_configs.hpp from codegen.

#include <cstring>
#include <hip/hip_runtime.h>
#include <string>
#include <tuple>
#include <unordered_map>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "aiter_hip_common.h"
#include "asm_i8gemm_configs.hpp"

namespace ffi = xla::ffi;

namespace jax_aiter {

struct __attribute__((packed)) I8GemmKernelArgs {
  void* ptr_c;    p2 _p0;
  void* ptr_a;    p2 _p1;
  void* ptr_b;    p2 _p2;
  void* ptr_sa;   p2 _p3;
  void* ptr_sb;   p2 _p4;
  void* ptr_bias; p2 _p5;
  unsigned int m;   p3 _p12;
  unsigned int n;   p3 _p13;
  unsigned int k;   p3 _p14;
  unsigned int lda; p3 _p15;
  unsigned int ldb; p3 _p16;
  unsigned int ldc; p3 _p17;
  unsigned int ks;  p3 _p18;
};

static std::tuple<std::string, int>
select_i8_kernel(int M, int N, int K, const std::string& arch_id, CFG* cfgs) {
  hipDevice_t dev;
  hipDeviceProp_t dev_prop;
  HIP_CALL(hipGetDevice(&dev));
  HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
  uint32_t num_cu = dev_prop.multiProcessorCount;

  uint32_t empty_cu = num_cu;
  uint32_t round = 0xffffffff;
  float compute2mem_effi = 1.0;
  std::string selectedKernelName = "";
  int selectedsplitK = 1;

  for (const auto& el : *cfgs) {
    if (el.first.find(arch_id) != 0) continue;
    const auto& cfg = el.second;
    if (cfg.bpreshuffle != 1) continue;
    if ((N % cfg.tile_n) != 0) continue;

    std::vector<int> splitK_list = cfg.splitK
        ? std::vector<int>{1, 2, 4, 8}
        : std::vector<int>{1};

    for (int splitK : splitK_list) {
      int tg_num_M = (M + cfg.tile_m - 1) / cfg.tile_m;
      int tg_num_N = (N + cfg.tile_n - 1) / cfg.tile_n;
      uint32_t tg_num = tg_num_M * tg_num_N * splitK;
      uint32_t local_round = (tg_num + num_cu - 1) / num_cu;
      float local_c2m = static_cast<float>(cfg.tile_m * cfg.tile_n) /
                        (cfg.tile_m + cfg.tile_n);

      bool is_earlier = (local_round < round);
      bool is_same = (local_round == round);
      bool fewer_empty = (empty_cu > (local_round * num_cu - tg_num));
      bool better_c2m = (local_c2m > compute2mem_effi);

      if (is_earlier || (is_same && (fewer_empty || better_c2m))) {
        round = local_round;
        empty_cu = local_round * num_cu - tg_num;
        compute2mem_effi = local_c2m;
        selectedKernelName = el.first;
        selectedsplitK = splitK;
      }
    }
  }
  return {selectedKernelName, selectedsplitK};
}

static AiterAsmKernel*
load_i8_kernel(const std::string& name, CFG* config_map,
               int& SUBM, int& SUBN) {
  static std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> cache;

  auto it = config_map->find(name);
  if (it == config_map->end()) return nullptr;

  const auto& cfg = it->second;
  SUBM = cfg.tile_m;
  SUBN = cfg.tile_n;

  auto result = cache.emplace(cfg.knl_name, nullptr);
  if (result.second)
    result.first->second = std::make_unique<AiterAsmKernel>(
        cfg.knl_name.c_str(), cfg.co_name.c_str());

  return result.first->second.get();
}

ffi::Error
GemmI8Fwd_Bridge(
    hipStream_t stream,
    ffi::AnyBuffer a,       // [M, K] int8
    ffi::AnyBuffer b,       // [N, K] int8 (shuffled layout)
    ffi::AnyBuffer a_scale, // [M, 1] f32
    ffi::AnyBuffer b_scale, // [1, N] f32
    ffi::AnyBuffer bias,    // [1, N] f32
    ffi::Result<ffi::AnyBuffer> out) { // [M, N] bf16

  auto a_dims = a.dimensions();
  auto b_dims = b.dimensions();

  if (a_dims.size() != 2 || b_dims.size() != 2) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "INT8 GEMM requires 2D inputs");
  }

  int M = static_cast<int>(a_dims[0]);
  int K = static_cast<int>(a_dims[1]);
  int N = static_cast<int>(b_dims[0]);

  std::string arch_id = get_gpu_arch();
  CFG* config_map = &cfg_i8gemm_bf16_perTokenI8;

  auto [name, splitK] = select_i8_kernel(M, N, K, arch_id, config_map);
  if (name.empty()) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "No suitable INT8 GEMM kernel found");
  }

  int SUBM = 0, SUBN = 0;
  AiterAsmKernel* impl = load_i8_kernel(name, config_map, SUBM, SUBN);
  if (!impl) {
    return ffi::Error(ffi::ErrorCode::kInternal, "Failed to load INT8 GEMM kernel");
  }

  I8GemmKernelArgs args = {};
  args.ptr_c    = out->untyped_data();
  args.ptr_a    = const_cast<void*>(a.untyped_data());
  args.ptr_b    = const_cast<void*>(b.untyped_data());
  args.ptr_sa   = const_cast<void*>(a_scale.untyped_data());
  args.ptr_sb   = const_cast<void*>(b_scale.untyped_data());
  args.ptr_bias = const_cast<void*>(bias.untyped_data());
  args.m   = M;
  args.n   = N;
  args.k   = K;
  args.lda = K;         // int8 stride = K elements
  args.ldb = K;         // int8 stride = K elements
  args.ldc = N * 2;     // bf16 stride = N * 2 bytes
  args.ks  = splitK;

  if (splitK > 1) {
    (void)hipMemsetAsync(out->untyped_data(), 0, M * N * 2, stream);
  }

  int gdx = (N / SUBN) * 256;
  int gdy = (M + SUBM - 1) / SUBM;
  gdx = (gdx / 256) * splitK;

  size_t arg_size = sizeof(args);
  impl->launch_kernel({&args, &arg_size, gdx, gdy, 1, 256, 1, 1, stream});

  return ffi::Error::Success();
}

} // namespace jax_aiter

#pragma GCC visibility push(default)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    GemmI8FwdJA, jax_aiter::GemmI8Fwd_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()   // A: [M, K] int8
        .Arg<ffi::AnyBuffer>()   // B: [N, K] int8
        .Arg<ffi::AnyBuffer>()   // a_scale: [M, 1] f32
        .Arg<ffi::AnyBuffer>()   // b_scale: [1, N] f32
        .Arg<ffi::AnyBuffer>()   // bias: [1, N] f32
        .Ret<ffi::AnyBuffer>(),  // Out: [M, N] bf16
    {xla::ffi::Traits::kCmdBufferCompatible});

#pragma GCC visibility pop
