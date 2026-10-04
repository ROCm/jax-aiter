// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Advanced Micro Devices, Inc.
//
// Generic typed-XLA-FFI trampoline for FlyDSL CompiledArtifact entry points.
// FlyDSL exposes a packed C interface `void(void**)`; this handler repacks
// XLA-owned input/result buffers, scalar attributes, and the XLA HIP stream
// into that interface. Compiled function pointers are process-local and are
// registered by jax_aiter.flydsl.bridge during JAX lowering.

#include <hip/hip_runtime.h>

#include <atomic>
#include <array>
#include <cstdint>
#include <mutex>
#include <string>
#include <utility>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace jax_aiter::flydsl {

namespace {

constexpr int kMaxArgs = 128;
constexpr int kMaxDevices = 16;
constexpr int kMaxCompiles = 1024;

constexpr int32_t kInputBuffer = 0;
constexpr int32_t kOutputBuffer = 1;
constexpr int32_t kScalar = 2;

using FlydslFunction = void (*)(void**);
static_assert(sizeof(void*) == sizeof(uint64_t));

struct CompiledFunctions {
  std::array<FlydslFunction, kMaxDevices> functions{};
  int32_t num_devices = 0;
  int64_t compile_key = 0;
};

std::array<CompiledFunctions, kMaxCompiles> g_compiles;
std::array<std::atomic<uint64_t>, kMaxCompiles> g_dispatch_counts{};
std::atomic<int32_t> g_compile_count{0};
std::mutex g_compile_mutex;

ffi::Error InvalidArgument(std::string message) {
  return ffi::Error(ffi::ErrorCode::kInvalidArgument, std::move(message));
}

}  // namespace

extern "C" __attribute__((visibility("default"))) int32_t
FlydslBridgeRegister(void** function_ptrs, int32_t num_devices,
                     int64_t compile_key) {
  if (function_ptrs == nullptr || num_devices < 1 ||
      num_devices > kMaxDevices) {
    return -1;
  }

  std::lock_guard<std::mutex> lock(g_compile_mutex);
  const int32_t index = g_compile_count.load(std::memory_order_relaxed);
  if (index >= kMaxCompiles) {
    return -2;
  }

  auto& entry = g_compiles[index];
  entry.num_devices = num_devices;
  entry.compile_key = compile_key;
  for (int32_t device = 0; device < num_devices; ++device) {
    entry.functions[device] =
        reinterpret_cast<FlydslFunction>(function_ptrs[device]);
    if (entry.functions[device] == nullptr) {
      return -3;
    }
  }
  g_compile_count.store(index + 1, std::memory_order_release);
  return index;
}

extern "C" __attribute__((visibility("default"))) int32_t
FlydslBridgeMaxArgs() {
  return kMaxArgs;
}
extern "C" __attribute__((visibility("default"))) int32_t
FlydslBridgeMaxDevices() {
  return kMaxDevices;
}
extern "C" __attribute__((visibility("default"))) int32_t
FlydslBridgeMaxCompiles() {
  return kMaxCompiles;
}
extern "C" __attribute__((visibility("default"))) uint64_t
FlydslBridgeDispatchCount(int32_t compile_index) {
  const int32_t count = g_compile_count.load(std::memory_order_acquire);
  if (compile_index < 0 || compile_index >= count) {
    return 0;
  }
  return g_dispatch_counts[compile_index].load(std::memory_order_relaxed);
}

ffi::Error FlydslDispatch(
    hipStream_t stream, int32_t device_ordinal, ffi::RemainingArgs args,
    ffi::RemainingRets rets, int64_t compile_index, int64_t compile_key,
    ffi::Span<const int32_t> arg_kinds,
    ffi::Span<const int32_t> arg_indices,
    ffi::Span<const int64_t> scalar_values) {
  const size_t num_args = arg_kinds.size();
  if (arg_indices.size() != num_args || scalar_values.size() != num_args) {
    return InvalidArgument(
        "FlydslDispatchJA: argument metadata arrays must have equal lengths");
  }
  if (num_args > kMaxArgs) {
    return InvalidArgument("FlydslDispatchJA: too many packed arguments");
  }
  const int32_t compile_count =
      g_compile_count.load(std::memory_order_acquire);
  if (compile_index < 0 || compile_index >= compile_count) {
    return InvalidArgument(
        "FlydslDispatchJA: compiled function index is not registered");
  }

  auto& compiled = g_compiles[compile_index];
  if (compiled.compile_key != compile_key) {
    return InvalidArgument(
        "FlydslDispatchJA: compiled function key does not match registry");
  }
  if (device_ordinal < 0 || device_ordinal >= compiled.num_devices) {
    return InvalidArgument(
        "FlydslDispatchJA: device ordinal has no registered function");
  }
  FlydslFunction function = compiled.functions[device_ordinal];
  if (function == nullptr) {
    return InvalidArgument(
        "FlydslDispatchJA: registered FlyDSL function is null");
  }

  // Each packed entry points to storage containing either a device pointer or
  // the scalar's bit pattern. FlyDSL's stream is the final packed argument.
  std::array<uint64_t, kMaxArgs + 1> storage{};
  std::array<void*, kMaxArgs + 1> packed{};

  for (size_t index = 0; index < num_args; ++index) {
    const int32_t kind = arg_kinds[index];
    const int32_t source_index = arg_indices[index];

    if (kind == kInputBuffer) {
      if (source_index < 0 ||
          static_cast<size_t>(source_index) >= args.size()) {
        return InvalidArgument(
            "FlydslDispatchJA: input buffer index is out of range");
      }
      auto decoded = args.get<ffi::AnyBuffer>(source_index);
      if (!decoded) {
        return std::move(decoded).error();
      }
      storage[index] = reinterpret_cast<uintptr_t>(decoded->untyped_data());
    } else if (kind == kOutputBuffer) {
      if (source_index < 0 ||
          static_cast<size_t>(source_index) >= rets.size()) {
        return InvalidArgument(
            "FlydslDispatchJA: output buffer index is out of range");
      }
      auto decoded = rets.get<ffi::AnyBuffer>(source_index);
      if (!decoded) {
        return std::move(decoded).error();
      }
      ffi::Result<ffi::AnyBuffer> result = *decoded;
      storage[index] = reinterpret_cast<uintptr_t>(result->untyped_data());
    } else if (kind == kScalar) {
      storage[index] = static_cast<uint64_t>(scalar_values[index]);
    } else {
      return InvalidArgument(
          "FlydslDispatchJA: unknown packed argument kind");
    }
    packed[index] = &storage[index];
  }

  storage[num_args] = reinterpret_cast<uintptr_t>(stream);
  packed[num_args] = &storage[num_args];
  g_dispatch_counts[compile_index].fetch_add(1, std::memory_order_relaxed);
  function(packed.data());
  return ffi::Error::Success();
}

}  // namespace jax_aiter::flydsl

#pragma GCC visibility push(default)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FlydslDispatchJA, jax_aiter::flydsl::FlydslDispatch,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Ctx<ffi::DeviceOrdinal>()
        .RemainingArgs()
        .RemainingRets()
        .Attr<int64_t>("compile_index")
        .Attr<int64_t>("compile_key")
        .Attr<ffi::Span<const int32_t>>("arg_kinds")
        .Attr<ffi::Span<const int32_t>>("arg_indices")
        .Attr<ffi::Span<const int64_t>>("scalar_values"));

#pragma GCC visibility pop
