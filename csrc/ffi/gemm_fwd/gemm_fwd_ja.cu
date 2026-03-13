// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// BF16 GEMM forward FFI handler using AITER hand-tuned ASM kernels.
// Computes Out = A @ B^T where A:[M,K] bf16, B:[N,K] bf16, Out:[M,N] bf16.
// Kernel selection via heuristic from asm_bf16gemm_configs.hpp.

#include <cstring>
#include <hip/hip_runtime.h>
#include <string>
#include <tuple>
#include <unordered_map>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "aiter_hip_common.h"
#include "asm_bf16gemm_configs.hpp"

namespace ffi = xla::ffi;

namespace jax_aiter {

// Packed kernel args -- must match the AITER ASM kernel ABI exactly.
struct __attribute__((packed)) GemmKernelArgs {
  void* ptr_D;          p2 _p0;   // output [M, N]
  void* ptr_C;          p2 _p1;   // unused (set to nullptr)
  void* ptr_A;          p2 _p2;   // input A [M, K]
  void* ptr_B;          p2 _p3;   // input B [N, K]
  float alpha;           p3 _p4;
  float beta;            p3 _p5;
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
  unsigned int splitk;    p3 _p17;
  unsigned int is_out_b16; p3 _p18;
  void* ptr_Bias;        p2 _p19;
  unsigned int add_bias;  p3 _p20;
  void* ptr_semaphore;   p2 _p21;
};

static std::tuple<std::string, int>
select_kernel(int M, int N, int K, CFG* cfgs, const std::string& arch_id) {
  if (K % 64 != 0) {
    return {"", -1};
  }

  hipDevice_t dev;
  hipDeviceProp_t dev_prop;
  HIP_CALL(hipGetDevice(&dev));
  HIP_CALL(hipGetDeviceProperties(&dev_prop, dev));
  uint32_t num_cu = dev_prop.multiProcessorCount;

  uint32_t empty_cu      = num_cu;
  uint32_t pure_tg_num   = 0;
  uint32_t round         = 0xffffffff;
  float compute2mem_effi = 1.0;
  int oob                = M;

  std::string selectedKernelName = "";
  int selectedsplitK             = 1;

  for (const auto& el : *cfgs) {
    if (el.first.find(arch_id) != 0)
      continue;
    const auto& cfg = el.second;

    if (N % cfg.tileN != 0 || cfg.bPreshuffle != 0)
      continue;

    int split_K = 1;
    pure_tg_num = ((M + cfg.tileM - 1) / cfg.tileM) * (N / cfg.tileN);
    if (cfg.splitK == 1 && K / cfg.subK >= 2) {
      int max_splitk = std::min(std::min(static_cast<int>(num_cu / pure_tg_num), 16),
                                static_cast<int>(K / cfg.subK));
      split_K = std::max(2, max_splitk);
    }

    uint32_t tg_num      = pure_tg_num * split_K;
    uint32_t local_round = (tg_num + num_cu - 1) / num_cu;
    float local_c2m = static_cast<float>(cfg.tileM * cfg.tileN) / (cfg.tileM + cfg.tileN);
    bool is_earlier_round     = (local_round < round);
    bool is_same_round        = (local_round == round);
    bool has_fewer_empty_cu   = (empty_cu > (local_round * num_cu - tg_num));
    bool has_same_empty_cu    = (empty_cu == (local_round * num_cu - tg_num));
    bool has_better_c2m       = (local_c2m > compute2mem_effi);
    bool less_oob = (M % cfg.tileM == 0) ? (oob > 0)
                                          : (cfg.tileM - M % cfg.tileM < static_cast<unsigned>(oob));
    bool has_same_oob = (static_cast<int>(cfg.tileM - (M % cfg.tileM)) == oob);

    if (is_earlier_round || (is_same_round && (has_fewer_empty_cu || less_oob)) ||
        (is_same_round && has_same_empty_cu && has_same_oob && has_better_c2m)) {
      round            = local_round;
      empty_cu         = local_round * num_cu - tg_num;
      compute2mem_effi = local_c2m;
      oob              = (M % cfg.tileM == 0) ? 0 : cfg.tileM - (M % cfg.tileM);
      selectedKernelName = el.first;
      selectedsplitK     = split_K;
    }
  }
  return {selectedKernelName, selectedsplitK};
}

static AiterAsmKernel*
load_kernel(const std::string& name, CFG* config_map,
            unsigned int& SUBM, unsigned int& SUBN) {
  static std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> cache;

  auto it = config_map->find(name);
  if (it == config_map->end()) return nullptr;

  const auto& cfg = it->second;
  SUBM = cfg.tileM;
  SUBN = cfg.tileN;

  auto result = cache.emplace(cfg.knl_name, nullptr);
  if (result.second)
    result.first->second = std::make_unique<AiterAsmKernel>(
        cfg.knl_name.c_str(), cfg.co_name.c_str());

  return result.first->second.get();
}

ffi::Error
GemmFwd_Bridge(
    hipStream_t stream,
    ffi::AnyBuffer a,
    ffi::AnyBuffer b,
    ffi::Result<ffi::AnyBuffer> out,
    ffi::Result<ffi::AnyBuffer> semaphore) {

  auto a_dims = a.dimensions();
  auto b_dims = b.dimensions();

  if (a_dims.size() != 2 || b_dims.size() != 2) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "GEMM requires 2D inputs: A[M,K], B[N,K]");
  }

  int M = static_cast<int>(a_dims[0]);
  int K = static_cast<int>(a_dims[1]);
  int N = static_cast<int>(b_dims[0]);

  if (static_cast<int>(b_dims[1]) != K) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "K dimension mismatch: A[M,K] vs B[N,K]");
  }

  if (K % 64 != 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "K must be divisible by 64 for ASM GEMM");
  }

  std::string arch_id = get_gpu_arch();
  CFG* config_map = &cfg_bf16gemm_fp32bf16;

  auto [name, split] = select_kernel(M, N, K, config_map, arch_id);
  if (name.empty()) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "No suitable GEMM kernel found for these dimensions");
  }

  unsigned int SUBM = 32, SUBN = 64;
  AiterAsmKernel* impl = load_kernel(name, config_map, SUBM, SUBN);
  if (!impl) {
    return ffi::Error(ffi::ErrorCode::kInternal, "Failed to load GEMM kernel");
  }

  const int elem_size_in  = 2;  // bf16
  const int elem_size_out = 2;  // bf16 output

  GemmKernelArgs args = {};
  args.ptr_D       = out->untyped_data();
  args.ptr_C       = nullptr;
  args.ptr_A       = const_cast<void*>(a.untyped_data());
  args.ptr_B       = const_cast<void*>(b.untyped_data());
  args.alpha       = 1.0f;
  args.beta        = 0.0f;
  args.stride_A0   = K * elem_size_in;
  args.stride_B0   = K * elem_size_in;
  args.stride_D0   = N * elem_size_out;
  args.stride_C0   = N * elem_size_out;
  args.M           = M;
  args.N           = N;
  args.K           = K;
  args.splitk      = split;
  args.is_out_b16  = 1;
  args.ptr_Bias    = nullptr;
  args.add_bias    = 0;

  args.ptr_semaphore = semaphore->untyped_data();
  (void)hipMemsetAsync(semaphore->untyped_data(), 0,
                       16 * 64 * sizeof(uint32_t), stream);

  int gdx = (N + SUBN - 1) / SUBN;
  int gdy = (M + SUBM - 1) / SUBM;
  int gdz = split;

  size_t arg_size = sizeof(args);
  impl->launch_kernel({&args, &arg_size, gdx, gdy, gdz, 256, 1, 1, stream});

  return ffi::Error::Success();
}

} // namespace jax_aiter

#pragma GCC visibility push(default)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    GemmFwdJA, jax_aiter::GemmFwd_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()   // A: [M, K] bf16
        .Arg<ffi::AnyBuffer>()   // B: [N, K] bf16
        .Ret<ffi::AnyBuffer>()   // Out: [M, N] bf16
        .Ret<ffi::AnyBuffer>(),  // Semaphore: [16, 64] u32
    {xla::ffi::Traits::kCmdBufferCompatible});

#pragma GCC visibility pop
