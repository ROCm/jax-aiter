// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "aiter_hip_common.h"
#include "asm_f4gemm_configs.hpp" // CFG map + kernel metadata

#include <cstdint>
#include <cstring>

namespace ffi = xla::ffi;

struct __attribute__((packed)) KernelArgs {
  void *ptr_D;
  p2 _p0;
  void *ptr_C;
  p2 _p1;
  void *ptr_A;
  p2 _p2;
  void *ptr_B;
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
  void *ptr_ScaleA;
  p2 _p17;
  void *ptr_ScaleB;
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
};

// Quick hash so we can memoize heuristic selections.
using DictKey =
    std::tuple<int, int, int, int /*log2_k_split or -1*/, bool /*bpreshuffle*/>;
struct SimpleHash {
  size_t operator()(const DictKey &k) const {
    auto [m, n, kdim, log2s, shuffle] = k;
    return std::hash<int>()(m) ^ std::hash<int>()(n) ^ std::hash<int>()(kdim) ^
           std::hash<int>()(log2s) ^ std::hash<bool>()(shuffle);
  }
};

int64_t strides_from_shape(ffi::Span<const int64_t> shape, int dim) {
  int64_t stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i > dim; --i) {
    stride *= shape[i];
  }
  return stride;
}

template <typename InBuf, typename OutBuf>
static CFG *GetCfg(const InBuf &, const OutBuf &) {
  return &cfg_f4gemm_bf16_per1x32Fp4; // TODO: (Ruturaj4) only variant for
                                      // fp4‑>bf16 today
}

static std::tuple<std::string /*kernel name*/, int /*log2 split‑K*/>
PickKernel(int M, int N, int K, std::optional<int> log2_k_split,
           bool bpreshuffle, CFG *cfgs) {
  hipDevice_t dev;
  hipDeviceProp_t prop;
  HIP_CALL(hipGetDevice(&dev));
  HIP_CALL(hipGetDeviceProperties(&prop, dev));
  uint32_t num_cu = prop.multiProcessorCount;

  uint32_t best_round = 0xffffffff, best_empty_cu = 0xffffffff;
  float best_eff = 0.0f;
  std::string best_name;
  int best_split = 1;

  for (const auto &e : *cfgs) {
    const auto &cfg = e.second;
    if (cfg.bpreshuffle != (bpreshuffle ? 1 : 0))
      continue;
    if (cfg.splitK && !log2_k_split)
      continue; // split‑K requested but none supplied
    if ((N % cfg.tile_N) != 0)
      continue;

    std::vector<int> splits;
    if (cfg.splitK) {
      if (log2_k_split) {
        splits = {1 << *log2_k_split};
      } else {
        splits = {2, 4, 8, 16};
      }
    } else {
      splits = {1};
    }

    for (int split : splits) {
      int tg_M = (M + cfg.tile_M - 1) / cfg.tile_M;
      int tg_N = (N + cfg.tile_N - 1) / cfg.tile_N;
      uint32_t tg_num = tg_M * tg_N * split;
      uint32_t round = (tg_num + num_cu - 1) / num_cu;
      float eff = static_cast<float>(cfg.tile_M * cfg.tile_N) /
                  (cfg.tile_M + cfg.tile_N);

      bool replace = false;
      if (round < best_round)
        replace = true;
      else if (round == best_round) {
        uint32_t empty_cu = round * num_cu - tg_num;
        if (empty_cu < best_empty_cu)
          replace = true;
        else if (empty_cu == best_empty_cu && eff > best_eff)
          replace = true;
      }

      if (replace) {
        best_round = round;
        best_empty_cu = round * num_cu - tg_num;
        best_eff = eff;
        best_name = e.first;
        best_split = split;
      }
    }
  }

  if (best_name.empty()) {
    throw std::runtime_error(
        "gemm_a4w4_asm_jax: heuristic failed – no kernel found");
  }
  int log2_split = 0;
  while ((best_split >>= 1) > 0)
    ++log2_split;
  return {best_name, log2_split};
}

static std::unordered_map<DictKey, std::tuple<std::string, int>, SimpleHash>
    kCache;
static std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>>
    kImplPool;

// A4W4 asm gemm kernel
// D=A*B*alpha+beta*C
ffi::Error
gemm_a4w4_asm_jax(hipStream_t stream,
                  ffi::Buffer<ffi::S8> A,        // A:[M, K/2] f4x2
                  ffi::Buffer<ffi::S8> B,        // B:[N, K/2] f4x2
                  ffi::Buffer<ffi::F32> A_scale, // A_scale:[M, K/32] e8m0 paded
                  ffi::Buffer<ffi::F32> B_scale, // B_scale:[N, K/32] e8m0 paded
                  ffi::ResultBuffer<ffi::BF16> out,          // Out:[M, N] bf16
                  std::optional<ffi::Buffer<ffi::F32>> bias, // bias:[M, N] f32
                  float_t alpha = 1.0f, float_t beta = 0.0f,
                  bool bpreshuffle = true,
                  std::optional<int> log2_k_split = std::nullopt) {
  auto a_dims = A.dimensions();
  auto b_dims = B.dimensions();
  auto out_dims = out->dimensions();
  auto a_scale_dims = A_scale.dimensions();
  auto b_scale_dims = B_scale.dimensions();

  int Mdim = a_dims[0];
  int Ndim = b_dims[0];
  int Kdim = a_dims[1] * 2; // always fp4_x2F

  KernelArgs args;
  args.ptr_D = out->untyped_data();
  args.ptr_C = bias.has_value() ? bias->untyped_data() : nullptr;
  args.ptr_A = A.untyped_data();
  args.ptr_B = B.untyped_data();

  args.alpha = alpha;
  args.beta = beta;
  args.stride_C0 = strides_from_shape(out_dims, 0);
  args.stride_A0 = strides_from_shape(a_dims, 0) * 2; // always fp4_x2
  args.stride_B0 = strides_from_shape(b_dims, 0) * 2; // always fp4_x2
  args.M = Mdim;
  args.N = Ndim;
  args.K = Kdim;

  args.ptr_ScaleA = A_scale.untyped_data();
  args.ptr_ScaleB = B_scale.untyped_data();
  args.stride_ScaleA0 = strides_from_shape(a_scale_dims, 0);
  args.stride_ScaleB0 = strides_from_shape(b_scale_dims, 0);
  args.log2_k_split = 0;

  // ----- Select implementation --------------------------------------------
  CFG *cfg_map = GetCfg(A, *out);
  if (cfg_map->empty())
    return ffi::Error::InvalidArgument("no kernels for this arch");
  DictKey key{Mdim, Ndim, Kdim, log2_k_split.value_or(-1), bpreshuffle};
  auto it = kCache.find(key);
  std::string kernel_name;
  int log2_split;
  if (it != kCache.end()) {
    std::tie(kernel_name, log2_split) = it->second;
  } else {
    std::tie(kernel_name, log2_split) =
        PickKernel(Mdim, Ndim, Kdim, log2_k_split, bpreshuffle, cfg_map);
    kCache.emplace(key, std::make_tuple(kernel_name, log2_split));
  }
  const auto &cfg = cfg_map->at(kernel_name);

  // ----- Launch parameters -------------------------------------------------
  int SUBM = cfg.tile_M;
  int SUBN = cfg.tile_N;
  int gdx = (Ndim + SUBN - 1) / SUBN;
  int gdy = (Mdim + SUBM - 1) / SUBM;
  int gdz = 1;
  if (cfg.splitK) {
    int splits = 1 << log2_split;
    int k_per_tg = ((Kdim / splits + 256 - 1) / 256) * 256;
    gdz = (Kdim + k_per_tg - 1) / k_per_tg;
  }
  args.log2_k_split = cfg.splitK ? log2_split : 0;

  // TODO should get from kernel
  int tg_group_size = 32;

  // --- get Aiter kernel ---------------------------------------------------
  AiterAsmKernel *impl;
  auto impl_it = kImplPool.find(kernel_name);
  if (impl_it == kImplPool.end()) {
    impl_it =
        kImplPool
            .emplace(kernel_name, std::make_unique<AiterAsmKernel>(
                                      cfg.name.c_str(), cfg.co_name.c_str()))
            .first;
  }
  impl = impl_it->second.get();

  size_t arg_size = sizeof(args);

  impl->launch_kernel({&args, &arg_size,
                       gdx, // gdx
                       gdy, // gdy
                       gdz, // gdz
                       256, // bdx: 4 wv64
                       1,   // bdy
                       1,   // bdz
                       stream});
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(GemmA4W4, gemm_a4w4_asm_jax,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<hipStream_t>>()
                                  .Arg<ffi::Buffer<ffi::S8>>()   // A
                                  .Arg<ffi::Buffer<ffi::S8>>()   // B
                                  .Arg<ffi::Buffer<ffi::F32>>()  // A_scale
                                  .Arg<ffi::Buffer<ffi::F32>>()  // B_scale
                                  .Ret<ffi::Buffer<ffi::BF16>>() // Out
                                  .Arg<ffi::Buffer<ffi::F32>>() // Optional bias
                                  .Attr<float_t>("alpha")
                                  .Attr<float_t>("beta")
                                  .Attr<bool>("bpreshuffle")
                                  .Attr<int64_t>("log2_k_split"),
                              {xla::ffi::Traits::kCmdBufferCompatible});
