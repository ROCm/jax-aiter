// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// FP4 GEMM FFI handler using AITER ASM kernels.
// Out[M,N] bf16 = A[M,K/2] fp4x2 @ B[N,K/2] fp4x2 with e8m0 block scales.

#include <cstdio>
#include <hip/hip_runtime.h>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "aiter_hip_common.h"
#include "asm_f4gemm_configs.hpp"

namespace ffi = xla::ffi;

namespace jax_aiter {

static constexpr int kMaxDevices = 16;

struct __attribute__((packed)) Fp4GemmKernelArgs {
  void* ptr_D;         p2 _p0;
  void* ptr_C;         p2 _p1;
  void* ptr_A;         p2 _p2;
  void* ptr_B;         p2 _p3;
  float alpha;          p3 _p4;
  float beta;           p3 _p5;
  unsigned int stride_D0; p3 _p6;
  unsigned int stride_D1; p3 _p7;
  unsigned int stride_C0; p3 _p8;
  unsigned int stride_C1; p3 _p9;
  unsigned int stride_A0; p3 _p10;
  unsigned int stride_A1; p3 _p11;
  unsigned int stride_B0; p3 _p12;
  unsigned int stride_B1; p3 _p13;
  unsigned int M;         p3 _p14;
  unsigned int N;         p3 _p15;
  unsigned int K;         p3 _p16;
  void* ptr_ScaleA;    p2 _p17;
  void* ptr_ScaleB;    p2 _p18;
  unsigned int stride_ScaleA0; p3 _p19;
  unsigned int stride_ScaleA1; p3 _p20;
  unsigned int stride_ScaleB0; p3 _p21;
  unsigned int stride_ScaleB1; p3 _p22;
  int log2_k_split;
};

// ---------------------------------------------------------------------------
// Per-device kernel cache with thread-safe eager loading (same pattern as BF16).
// ---------------------------------------------------------------------------
struct Fp4KernelEntry {
  std::unique_ptr<AiterAsmKernel> impl;
  int tile_M;
  int tile_N;
  int splitK_capable;
};

using Fp4KernelMap = std::unordered_map<std::string, Fp4KernelEntry>;

static Fp4KernelMap& get_fp4_kernel_cache(int dev_id) {
  static Fp4KernelMap caches[kMaxDevices];
  static std::once_flag flags[kMaxDevices];

  std::call_once(flags[dev_id], [dev_id]() {
    HIP_CALL(hipSetDevice(dev_id));

    std::string arch_id = get_gpu_arch();
    CFG* cfgs = &cfg_f4gemm_bf16_per1x32Fp4;

    int count = 0;
    for (const auto& el : *cfgs) {
      if (el.first.find(arch_id) != 0) continue;
      const auto& cfg = el.second;
      if (caches[dev_id].count(cfg.knl_name)) continue;

      Fp4KernelEntry entry;
      entry.impl = std::make_unique<AiterAsmKernel>(
          cfg.knl_name.c_str(), cfg.co_name.c_str());
      entry.tile_M = cfg.tile_M;
      entry.tile_N = cfg.tile_N;
      entry.splitK_capable = cfg.splitK;
      caches[dev_id].emplace(cfg.knl_name, std::move(entry));
      count++;
    }
    AITER_LOG_INFO("Loaded " << count << " FP4 GEMM kernels for "
                   << arch_id << " on device " << dev_id);
  });

  return caches[dev_id];
}

// ---------------------------------------------------------------------------
// Kernel selection heuristic.
// ---------------------------------------------------------------------------
static std::tuple<std::string, int>
select_fp4_kernel(int M, int N, int K, int dev_id, CFG* cfgs) {
  hipDeviceProp_t dev_prop;
  HIP_CALL(hipGetDeviceProperties(&dev_prop, dev_id));
  uint32_t num_cu = dev_prop.multiProcessorCount;

  std::string arch_id = get_gpu_arch();
  uint32_t empty_cu = num_cu;
  uint32_t round = 0xffffffff;
  float compute2mem_effi = 1.0;
  std::string selected_knl_name = "";
  int selectedsplitK = 0;

  for (const auto& el : *cfgs) {
    if (el.first.find(arch_id) != 0) continue;
    const auto& cfg = el.second;
    if (cfg.bpreshuffle != 1) continue;

    if (cfg.tile_M != 128 || cfg.tile_N != 512 || (N % cfg.tile_N) == 0) {
      std::vector<int> splitK_list = cfg.splitK
          ? std::vector<int>{2, 4, 8, 16}
          : std::vector<int>{1};

      for (int splitK : splitK_list) {
        int tg_num_M = (M + cfg.tile_M - 1) / cfg.tile_M;
        int tg_num_N = (N + cfg.tile_N - 1) / cfg.tile_N;
        uint32_t tg_num = tg_num_M * tg_num_N * splitK;
        uint32_t local_round = (tg_num + num_cu - 1) / num_cu;
        float local_c2m = static_cast<float>(cfg.tile_M * cfg.tile_N) /
                          (cfg.tile_M + cfg.tile_N);

        bool is_earlier = (local_round < round);
        bool is_same = (local_round == round);
        bool fewer_empty = (empty_cu > (local_round * num_cu - tg_num));
        bool better_c2m = (local_c2m > compute2mem_effi);

        if (is_earlier || (is_same && (fewer_empty || better_c2m))) {
          round = local_round;
          empty_cu = local_round * num_cu - tg_num;
          compute2mem_effi = local_c2m;
          selected_knl_name = cfg.knl_name;
          int log2 = 0;
          int tmp = splitK;
          while (tmp >>= 1) ++log2;
          selectedsplitK = log2;
        }
      }
    }
  }
  return {selected_knl_name, selectedsplitK};
}

// ---------------------------------------------------------------------------
// FFI bridge
// ---------------------------------------------------------------------------
ffi::Error
GemmFp4Fwd_Bridge(
    hipStream_t stream,
    ffi::AnyBuffer a,
    ffi::AnyBuffer b,
    ffi::AnyBuffer a_scale,
    ffi::AnyBuffer b_scale,
    ffi::Result<ffi::AnyBuffer> out) {

  auto a_dims = a.dimensions();
  auto b_dims = b.dimensions();

  if (a_dims.size() != 2 || b_dims.size() != 2) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "FP4 GEMM requires 2D inputs: A[M,K/2], B[N,K/2]");
  }

  int M = static_cast<int>(a_dims[0]);
  int K = static_cast<int>(a_dims[1]) * 2;
  int N = static_cast<int>(b_dims[0]);

  int dev_id;
  HIP_CALL(hipGetDevice(&dev_id));
  CFG* config_map = &cfg_f4gemm_bf16_per1x32Fp4;

  auto [knl_name, log2_split] = select_fp4_kernel(M, N, K, dev_id, config_map);
  if (knl_name.empty()) {
    char msg[256];
    snprintf(msg, sizeof(msg),
             "No suitable FP4 GEMM kernel for M=%d N=%d K=%d", M, N, K);
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, msg);
  }

  auto& cache = get_fp4_kernel_cache(dev_id);
  auto it = cache.find(knl_name);
  if (it == cache.end()) {
    char msg[512];
    snprintf(msg, sizeof(msg),
             "FP4 kernel '%s' not in cache for dev %d (M=%d N=%d K=%d)",
             knl_name.c_str(), dev_id, M, N, K);
    return ffi::Error(ffi::ErrorCode::kInternal, msg);
  }

  auto& entry = it->second;
  int SUBM = entry.tile_M;
  int SUBN = entry.tile_N;

  auto a_scale_dims = a_scale.dimensions();

  Fp4GemmKernelArgs args = {};
  args.ptr_D       = out->untyped_data();
  args.ptr_C       = nullptr;
  args.ptr_A       = const_cast<void*>(a.untyped_data());
  args.ptr_B       = const_cast<void*>(b.untyped_data());
  args.alpha       = 1.0f;
  args.beta        = 0.0f;
  args.stride_C0   = N;
  args.stride_A0   = static_cast<int>(a_dims[1]) * 2;
  args.stride_B0   = static_cast<int>(b_dims[1]) * 2;
  args.M           = M;
  args.N           = N;
  args.K           = K;
  args.ptr_ScaleA  = const_cast<void*>(a_scale.untyped_data());
  args.ptr_ScaleB  = const_cast<void*>(b_scale.untyped_data());
  args.stride_ScaleA0 = static_cast<int>(a_scale_dims[1]);
  args.stride_ScaleB0 = static_cast<int>(b_scale.dimensions()[1]);

  int gdx = (N + SUBN - 1) / SUBN;
  int gdy = (M + SUBM - 1) / SUBM;
  int gdz = 1;

  if (entry.splitK_capable && log2_split > 0) {
    args.log2_k_split = log2_split;
    int k_num = 1 << log2_split;
    int k_per_tg = K / k_num;
    k_per_tg = ((k_per_tg + 255) / 256) * 256;
    gdz = (K + k_per_tg - 1) / k_per_tg;
    HIP_CALL(hipMemsetAsync(out->untyped_data(), 0,
                            M * N * sizeof(uint16_t), stream));
  } else {
    args.log2_k_split = 0;
  }

  size_t arg_size = sizeof(args);
  entry.impl->launch_kernel({&args, &arg_size, gdx, gdy, gdz, 256, 1, 1, stream});

  return ffi::Error::Success();
}

} // namespace jax_aiter

#pragma GCC visibility push(default)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    GemmFp4FwdJA, jax_aiter::GemmFp4Fwd_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()   // A: [M, K/2] uint8 packed fp4
        .Arg<ffi::AnyBuffer>()   // B: [N, K/2] uint8 packed fp4
        .Arg<ffi::AnyBuffer>()   // a_scale: [M, K/32] uint8 e8m0
        .Arg<ffi::AnyBuffer>()   // b_scale: [N, K/32] uint8 e8m0
        .Ret<ffi::AnyBuffer>(),  // Out: [M, N] bf16
    {xla::ffi::Traits::kCmdBufferCompatible});

#pragma GCC visibility pop
