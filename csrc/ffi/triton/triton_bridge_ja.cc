// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Advanced Micro Devices, Inc.
//
// Typed XLA FFI trampoline for Triton HIP HSACO. Python lowering compiles
// kernels with triton.compiler and registers the binary here; execution
// launches on the XLA HIP stream with no Python on the dispatch path.
//
// ABI lock (Triton 3.8.0 AMD / gfx950):
//   hipModuleLaunchKernel(function, grid, block=warp_size*num_warps, shared,
//                         stream, kernelParams, extra=nullptr)
//   kernelParams is void** of per-argument storage, then two scratch pointers
//   (global, profile). Pointer args are hipDeviceptr_t; i32/fp32 as native.

#include <hip/hip_runtime.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace jax_aiter::triton_bridge {

namespace {

constexpr int kMaxArgs = 128;
constexpr int kMaxDevices = 16;
constexpr int kMaxCompiles = 1024;

constexpr int32_t kInputBuffer = 0;
constexpr int32_t kOutputBuffer = 1;
constexpr int32_t kScalar = 2;

constexpr int32_t kAbiPtr = 0;
constexpr int32_t kAbiI32 = 1;
constexpr int32_t kAbiI64 = 2;
constexpr int32_t kAbiF32 = 3;
constexpr int32_t kAbiF64 = 4;

struct CompiledKernel {
  std::array<hipModule_t, kMaxDevices> modules{};
  std::array<hipFunction_t, kMaxDevices> functions{};
  int32_t num_devices = 0;
  int32_t num_warps = 0;
  int32_t num_ctas = 0;
  int32_t shared = 0;
  int32_t warp_size = 64;
  int32_t num_params = 0;
  int64_t compile_key = 0;
  std::array<int32_t, kMaxArgs> abi_kinds{};
  std::string name;
};

std::array<CompiledKernel, kMaxCompiles> g_compiles;
std::array<std::atomic<uint64_t>, kMaxCompiles> g_dispatch_counts{};
std::atomic<int32_t> g_compile_count{0};
std::mutex g_compile_mutex;

ffi::Error InvalidArgument(std::string message) {
  return ffi::Error(ffi::ErrorCode::kInvalidArgument, std::move(message));
}

ffi::Error HipError(const char* what, hipError_t status) {
  return ffi::Error(
      ffi::ErrorCode::kInternal,
      std::string("TritonDispatchJA: ") + what + ": " +
          hipGetErrorString(status));
}

int32_t AbiSize(int32_t kind) {
  switch (kind) {
    case kAbiPtr:
    case kAbiI64:
    case kAbiF64:
      return 8;
    case kAbiI32:
    case kAbiF32:
      return 4;
    default:
      return 0;
  }
}

}  // namespace

extern "C" __attribute__((visibility("default"))) int32_t
TritonBridgeRegister(const uint8_t* hsaco, uint64_t hsaco_len,
                     const char* kernel_name, int32_t num_devices,
                     int32_t num_warps, int32_t num_ctas, int32_t shared,
                     int32_t warp_size, int64_t compile_key,
                     const int32_t* abi_kinds, int32_t num_params) {
  if (hsaco == nullptr || hsaco_len == 0 || kernel_name == nullptr) {
    return -1;
  }
  if (num_devices < 1 || num_devices > kMaxDevices) {
    return -1;
  }
  if (num_params < 0 || num_params > kMaxArgs - 2) {
    return -1;
  }
  if (num_ctas != 1) {
    return -4;
  }
  if (warp_size <= 0 || num_warps <= 0) {
    return -1;
  }

  std::vector<uint8_t> blob(hsaco, hsaco + hsaco_len);

  std::lock_guard<std::mutex> lock(g_compile_mutex);
  const int32_t index = g_compile_count.load(std::memory_order_relaxed);
  if (index >= kMaxCompiles) {
    return -2;
  }

  auto& entry = g_compiles[index];
  entry.num_devices = num_devices;
  entry.num_warps = num_warps;
  entry.num_ctas = num_ctas;
  entry.shared = shared;
  entry.warp_size = warp_size;
  entry.num_params = num_params;
  entry.compile_key = compile_key;
  entry.name = kernel_name;
  for (int32_t i = 0; i < num_params; ++i) {
    entry.abi_kinds[i] = abi_kinds[i];
    if (AbiSize(abi_kinds[i]) == 0) {
      return -1;
    }
  }

  int saved = -1;
  hipError_t status = hipGetDevice(&saved);
  if (status != hipSuccess) {
    return -3;
  }

  for (int32_t device = 0; device < num_devices; ++device) {
    status = hipSetDevice(device);
    if (status != hipSuccess) {
      (void)hipSetDevice(saved);
      return -3;
    }
    hipModule_t module = nullptr;
    status = hipModuleLoadData(&module, blob.data());
    if (status != hipSuccess) {
      (void)hipSetDevice(saved);
      return -3;
    }
    hipFunction_t function = nullptr;
    status = hipModuleGetFunction(&function, module, kernel_name);
    if (status != hipSuccess) {
      (void)hipModuleUnload(module);
      (void)hipSetDevice(saved);
      return -3;
    }
    entry.modules[device] = module;
    entry.functions[device] = function;
  }
  (void)hipSetDevice(saved);
  g_compile_count.store(index + 1, std::memory_order_release);
  return index;
}

extern "C" __attribute__((visibility("default"))) int32_t
TritonBridgeMaxArgs() {
  return kMaxArgs;
}
extern "C" __attribute__((visibility("default"))) int32_t
TritonBridgeMaxDevices() {
  return kMaxDevices;
}
extern "C" __attribute__((visibility("default"))) int32_t
TritonBridgeMaxCompiles() {
  return kMaxCompiles;
}
extern "C" __attribute__((visibility("default"))) uint64_t
TritonBridgeDispatchCount(int32_t compile_index) {
  const int32_t count = g_compile_count.load(std::memory_order_acquire);
  if (compile_index < 0 || compile_index >= count) {
    return 0;
  }
  return g_dispatch_counts[compile_index].load(std::memory_order_relaxed);
}

ffi::Error TritonDispatch(
    hipStream_t stream, int32_t device_ordinal, ffi::RemainingArgs args,
    ffi::RemainingRets rets, int64_t compile_index, int64_t compile_key,
    int64_t grid_x, int64_t grid_y, int64_t grid_z,
    ffi::Span<const int32_t> arg_kinds, ffi::Span<const int32_t> arg_indices,
    ffi::Span<const int64_t> scalar_values) {
  const size_t num_args = arg_kinds.size();
  if (arg_indices.size() != num_args || scalar_values.size() != num_args) {
    return InvalidArgument(
        "TritonDispatchJA: argument metadata arrays must have equal lengths");
  }
  if (num_args > kMaxArgs - 2) {
    return InvalidArgument("TritonDispatchJA: too many packed arguments");
  }
  const int32_t compile_count =
      g_compile_count.load(std::memory_order_acquire);
  if (compile_index < 0 || compile_index >= compile_count) {
    return InvalidArgument(
        "TritonDispatchJA: compiled kernel index is not registered");
  }

  auto& compiled = g_compiles[compile_index];
  if (compiled.compile_key != compile_key) {
    return InvalidArgument(
        "TritonDispatchJA: compiled kernel key does not match registry");
  }
  if (static_cast<int32_t>(num_args) != compiled.num_params) {
    return InvalidArgument(
        "TritonDispatchJA: packed argument count does not match ABI");
  }
  if (device_ordinal < 0 || device_ordinal >= compiled.num_devices) {
    return InvalidArgument(
        "TritonDispatchJA: device ordinal has no registered module");
  }
  hipFunction_t function = compiled.functions[device_ordinal];
  if (function == nullptr) {
    return InvalidArgument("TritonDispatchJA: registered HIP function is null");
  }
  if (grid_x < 0 || grid_y < 0 || grid_z < 0) {
    return InvalidArgument("TritonDispatchJA: grid dimensions must be >= 0");
  }
  if (grid_x * grid_y * grid_z == 0) {
    return ffi::Error::Success();
  }

  alignas(8) std::array<uint64_t, kMaxArgs> storage{};
  std::array<void*, kMaxArgs> params{};
  const int32_t num_params = compiled.num_params;

  for (int32_t index = 0; index < num_params; ++index) {
    const int32_t abi = compiled.abi_kinds[index];
    const int32_t kind = arg_kinds[index];
    const int32_t source_index = arg_indices[index];
    uint64_t value = 0;

    if (kind == kInputBuffer) {
      if (source_index < 0 ||
          static_cast<size_t>(source_index) >= args.size()) {
        return InvalidArgument(
            "TritonDispatchJA: input buffer index is out of range");
      }
      auto decoded = args.get<ffi::AnyBuffer>(source_index);
      if (!decoded) {
        return std::move(decoded).error();
      }
      if (decoded->size_bytes() == 0) {
        value = 0;
      } else {
        value = reinterpret_cast<uintptr_t>(decoded->untyped_data());
      }
      if (abi != kAbiPtr) {
        return InvalidArgument(
            "TritonDispatchJA: buffer argument does not match pointer ABI");
      }
    } else if (kind == kOutputBuffer) {
      if (source_index < 0 ||
          static_cast<size_t>(source_index) >= rets.size()) {
        return InvalidArgument(
            "TritonDispatchJA: output buffer index is out of range");
      }
      auto decoded = rets.get<ffi::AnyBuffer>(source_index);
      if (!decoded) {
        return std::move(decoded).error();
      }
      ffi::Result<ffi::AnyBuffer> result = *decoded;
      if (result->size_bytes() == 0) {
        value = 0;
      } else {
        value = reinterpret_cast<uintptr_t>(result->untyped_data());
      }
      if (abi != kAbiPtr) {
        return InvalidArgument(
            "TritonDispatchJA: buffer argument does not match pointer ABI");
      }
    } else if (kind == kScalar) {
      value = static_cast<uint64_t>(scalar_values[index]);
      if (abi == kAbiPtr) {
        return InvalidArgument(
            "TritonDispatchJA: scalar argument does not match pointer ABI");
      }
    } else {
      return InvalidArgument("TritonDispatchJA: unknown packed argument kind");
    }

    storage[index] = value;
    params[index] = &storage[index];
  }

  // Triton AMD always appends global_scratch and profile_scratch pointers.
  storage[num_params] = 0;
  storage[num_params + 1] = 0;
  params[num_params] = &storage[num_params];
  params[num_params + 1] = &storage[num_params + 1];

  const unsigned block = static_cast<unsigned>(compiled.warp_size) *
                         static_cast<unsigned>(compiled.num_warps);
  hipError_t status = hipModuleLaunchKernel(
      function, static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y),
      static_cast<unsigned>(grid_z), block, 1, 1,
      static_cast<unsigned>(compiled.shared), stream, params.data(), nullptr);
  if (status != hipSuccess) {
    return HipError("hipModuleLaunchKernel", status);
  }
  g_dispatch_counts[compile_index].fetch_add(1, std::memory_order_relaxed);
  return ffi::Error::Success();
}

}  // namespace jax_aiter::triton_bridge

#pragma GCC visibility push(default)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    TritonDispatchJA, jax_aiter::triton_bridge::TritonDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Ctx<ffi::DeviceOrdinal>()
        .RemainingArgs()
        .RemainingRets()
        .Attr<int64_t>("compile_index")
        .Attr<int64_t>("compile_key")
        .Attr<int64_t>("grid_x")
        .Attr<int64_t>("grid_y")
        .Attr<int64_t>("grid_z")
        .Attr<ffi::Span<const int32_t>>("arg_kinds")
        .Attr<ffi::Span<const int32_t>>("arg_indices")
        .Attr<ffi::Span<const int64_t>>("scalar_values"));

#pragma GCC visibility pop
