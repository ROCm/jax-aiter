// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// BF16 GEMM forward FFI handler using AITER hand-tuned ASM kernels.
// Computes Out = A @ B^T where A:[M,K] bf16, B:[N,K] bf16, Out:[M,N] bf16.
// Kernel selection via heuristic from asm_bf16gemm_configs.hpp.
//
// All HIP module loads happen eagerly at first use (before any graph capture).
// GemmFwd_Bridge is fully command-buffer-compatible: only hipMemsetAsync +
// hipModuleLaunchKernel are called during execution.

#include <cstring>
#include <hip/hip_runtime.h>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "aiter_hip_common.h"
#include "asm_bf16gemm_configs.hpp"

namespace ffi = xla::ffi;

namespace jax_aiter {

struct __attribute__((packed)) GemmKernelArgs {
  void* ptr_D;          p2 _p0;
  void* ptr_C;          p2 _p1;
  void* ptr_A;          p2 _p2;
  void* ptr_B;          p2 _p3;
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

// ---------------------------------------------------------------------------
// Cached device properties -- queried once per device.
// hipModule_t / hipFunction_t handles are device-specific in HIP, so we need
// a separate kernel cache per device.
// ---------------------------------------------------------------------------
struct DeviceInfo {
  std::string arch_id;
  uint32_t num_cu;
};

static constexpr int kMaxDevices = 16;

static DeviceInfo& get_device_info(int dev_id) {
  static DeviceInfo infos[kMaxDevices];
  static std::once_flag flags[kMaxDevices];

  std::call_once(flags[dev_id], [dev_id]() {
    hipDeviceProp_t dev_prop;
    HIP_CALL(hipGetDeviceProperties(&dev_prop, dev_id));
    infos[dev_id].arch_id = get_gpu_arch();
    infos[dev_id].num_cu = dev_prop.multiProcessorCount;
  });
  return infos[dev_id];
}

// ---------------------------------------------------------------------------
// Per-device kernel cache.
// hipModuleLoad binds to the current HIP context (device), so each device
// needs its own loaded modules.  Thread-safe via per-device std::once_flag.
// After init, lookups are lock-free (read-only map).
// ---------------------------------------------------------------------------
struct KernelEntry {
  std::unique_ptr<AiterAsmKernel> impl;
  unsigned int tileM;
  unsigned int tileN;
};

using KernelMap = std::unordered_map<std::string, KernelEntry>;

static KernelMap& get_kernel_cache_for_device(int dev_id) {
  static KernelMap caches[kMaxDevices];
  static std::once_flag flags[kMaxDevices];

  std::call_once(flags[dev_id], [dev_id]() {
    HIP_CALL(hipSetDevice(dev_id));

    const auto& di = get_device_info(dev_id);
    CFG* cfgs = &cfg_bf16gemm_fp32bf16;

    AITER_LOG_INFO("Eagerly loading all BF16 GEMM kernels for "
                   << di.arch_id << " on device " << dev_id);
    int count = 0;
    for (const auto& el : *cfgs) {
      if (el.first.find(di.arch_id) != 0)
        continue;
      const auto& cfg = el.second;
      if (caches[dev_id].count(cfg.knl_name))
        continue;

      KernelEntry entry;
      entry.impl = std::make_unique<AiterAsmKernel>(
          cfg.knl_name.c_str(), cfg.co_name.c_str());
      entry.tileM = cfg.tileM;
      entry.tileN = cfg.tileN;
      caches[dev_id].emplace(cfg.knl_name, std::move(entry));
      count++;
    }
    AITER_LOG_INFO("Loaded " << count << " BF16 GEMM kernels for "
                   << di.arch_id << " on device " << dev_id);
  });

  return caches[dev_id];
}

// ---------------------------------------------------------------------------
// Kernel selection heuristic -- pure CPU, no HIP calls.
// ---------------------------------------------------------------------------
static std::tuple<std::string, int>
select_kernel(int M, int N, int K, int dev_id, CFG* cfgs) {
  if (K % 64 != 0) {
    return {"", -1};
  }

  const auto& di = get_device_info(dev_id);
  uint32_t num_cu = di.num_cu;

  uint32_t empty_cu      = num_cu;
  uint32_t pure_tg_num   = 0;
  uint32_t round         = 0xffffffff;
  float compute2mem_effi = 1.0;
  int oob                = M;

  std::string selectedKernelName = "";
  int selectedsplitK             = 1;

  for (const auto& el : *cfgs) {
    if (el.first.find(di.arch_id) != 0)
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

// ---------------------------------------------------------------------------
// Look up a pre-loaded kernel for the given device. No HIP calls.
// ---------------------------------------------------------------------------
static const KernelEntry*
lookup_kernel(const std::string& config_name, int dev_id, CFG* config_map) {
  auto cfg_it = config_map->find(config_name);
  if (cfg_it == config_map->end()) return nullptr;

  auto& cache = get_kernel_cache_for_device(dev_id);
  auto it = cache.find(cfg_it->second.knl_name);
  if (it == cache.end()) return nullptr;
  return &it->second;
}

// ---------------------------------------------------------------------------
// FFI bridge -- only stream-recordable HIP calls (memset + launch).
// ---------------------------------------------------------------------------
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

  int dev_id = 0;
  HIP_CALL(hipGetDevice(&dev_id));

  CFG* config_map = &cfg_bf16gemm_fp32bf16;

  auto [name, split] = select_kernel(M, N, K, dev_id, config_map);
  if (name.empty()) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "No suitable GEMM kernel found for these dimensions"
                      " (M=" + std::to_string(M) + " N=" + std::to_string(N)
                      + " K=" + std::to_string(K) + ")");
  }

  const KernelEntry* entry = lookup_kernel(name, dev_id, config_map);
  if (!entry || !entry->impl) {
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

  unsigned int SUBM = entry->tileM;
  unsigned int SUBN = entry->tileN;
  int gdx = (N + SUBN - 1) / SUBN;
  int gdy = (M + SUBM - 1) / SUBM;
  int gdz = split;

  size_t arg_size = sizeof(args);
  entry->impl->launch_kernel({&args, &arg_size, gdx, gdy, gdz, 256, 1, 1, stream});

  return ffi::Error::Success();
}

} // namespace jax_aiter

// ---------------------------------------------------------------------------
// Eagerly pre-load kernels for ALL visible devices.
// Called from Python after .so is loaded (avoids static init order issues).
// ---------------------------------------------------------------------------
extern "C" __attribute__((visibility("default")))
void gemm_fwd_ja_preload_kernels() {
  int num_devices = 0;
  auto err = hipGetDeviceCount(&num_devices);
  if (err != hipSuccess || num_devices <= 0) return;

  int orig_dev = 0;
  hipGetDevice(&orig_dev);

  for (int d = 0; d < num_devices && d < jax_aiter::kMaxDevices; d++) {
    jax_aiter::get_kernel_cache_for_device(d);
  }

  hipSetDevice(orig_dev);
}

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
