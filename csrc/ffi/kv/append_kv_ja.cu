// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// append_kv. Writes new K/V into a paged KV pool at slot_mapping, in place,
// by bridging aiter::reshape_and_cache_flash through XLA FFI.
//
// This is a shim, not a reimplementation: aiter's cache.h is already torch-free
// and takes aiter_tensor_t (a POD of pointer, shape, strides, dtype), so the
// handler populates descriptors from FFI buffer metadata and calls the existing
// symbol. It does not replicate the kernel launch.
//
// Layout, per the v1 recommendation: NHD, K and V in separate pools, no
// x-packing. That is exactly what reshape_and_cache_flash consumes:
//
//   k_new / v_new  [num_tokens, num_kv_heads, head_dim]
//   k_pool / v_pool [num_pages, tokens_per_page, num_kv_heads, head_dim]
//   slot_mapping   [num_tokens]  int32, absolute token slot; negative = skip
//
// The pools are bound as both .Arg and .Ret and must be aliased by the caller.
// Mutation is therefore expressed as a value, so the op stays pure and needs no
// has_side_effect: paged attention consumes the returned pool, and that data
// dependence is what stops XLA reordering the read before the write.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <string>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "aiter_interop.h"
#include "cache.h"

namespace ffi = xla::ffi;

namespace jax_aiter {

ffi::Error AppendKv_Bridge(hipStream_t stream, int32_t device_ordinal,
                           ffi::AnyBuffer k_new, ffi::AnyBuffer v_new,
                           ffi::AnyBuffer slot_mapping, ffi::AnyBuffer k_scale,
                           ffi::AnyBuffer v_scale, ffi::AnyBuffer k_pool_in,
                           ffi::AnyBuffer v_pool_in,
                           ffi::Result<ffi::AnyBuffer> k_pool_out,
                           ffi::Result<ffi::AnyBuffer> v_pool_out,
                           std::string_view kv_cache_dtype) {
  // The aliasing contract proven at M0, enforced here so a misconfigured caller
  // cannot silently write into a buffer that is about to be discarded.
  if (k_pool_in.untyped_data() != k_pool_out->untyped_data() ||
      v_pool_in.untyped_data() != v_pool_out->untyped_data()) {
    return ffi::Error(
        ffi::ErrorCode::kFailedPrecondition,
        "AppendKvJA: input_output_aliases was not honoured -- a KV pool "
        "resolved to different device pointers as .Arg and .Ret, so the write "
        "would be lost. Pass input_output_aliases={5: 0, 6: 1}.");
  }

  auto k_new_dims = k_new.dimensions();
  auto pool_dims = k_pool_in.dimensions();

  if (k_new_dims.size() != 3) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "AppendKvJA: k_new must be [num_tokens, num_kv_heads, head_dim]");
  }
  if (pool_dims.size() != 4) {
    return ffi::Error(
        ffi::ErrorCode::kInvalidArgument,
        "AppendKvJA: pool must be [num_pages, tokens_per_page, num_kv_heads, head_dim]");
  }
  if (v_new.dimensions().size() != 3 || v_pool_in.dimensions().size() != 4) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "AppendKvJA: v_new / v_pool rank mismatch with k side");
  }
  if (k_new.element_type() != k_pool_in.element_type()) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "AppendKvJA: k_new and k_pool dtype must match "
                      "(quantised pools are out of scope at M1)");
  }
  // int32 is what the control plane produces and what JAX emits without x64;
  // int64 is accepted too so a caller sharing vLLM-shaped metadata needs no
  // conversion. aiter picks the width off the descriptor dtype either way.
  if (slot_mapping.element_type() != ffi::DataType::S32 &&
      slot_mapping.element_type() != ffi::DataType::S64) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "AppendKvJA: slot_mapping must be int32 or int64");
  }
  if (k_scale.element_type() != ffi::DataType::F32 ||
      v_scale.element_type() != ffi::DataType::F32) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "AppendKvJA: k_scale / v_scale must be float32");
  }

  const int64_t num_tokens = k_new_dims[0];
  const int64_t num_kv_heads = k_new_dims[1];
  const int64_t head_dim = k_new_dims[2];

  if (pool_dims[2] != num_kv_heads || pool_dims[3] != head_dim) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "AppendKvJA: pool head count / head_dim disagree with k_new");
  }
  if (static_cast<int64_t>(slot_mapping.element_count()) != num_tokens) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "AppendKvJA: slot_mapping length must equal num_tokens");
  }
  if (num_tokens == 0) {
    return ffi::Error::Success();
  }

  const int device_id = static_cast<int>(device_ordinal);

  aiter_tensor_t t_key, t_value, t_key_cache, t_value_cache, t_slots, t_kscale,
      t_vscale;

  if (auto e = MakeAiterTensor(k_new, device_id, &t_key, "k_new"); e.failure()) return e;
  if (auto e = MakeAiterTensor(v_new, device_id, &t_value, "v_new"); e.failure()) return e;
  // Describe the pools through the aliased .Ret pointers: same memory, but it
  // keeps the "we write to the output" intent explicit.
  if (auto e = MakeAiterTensor(k_pool_out->untyped_data(), pool_dims,
                               k_pool_in.element_type(), device_id, &t_key_cache,
                               "k_pool");
      e.failure())
    return e;
  if (auto e = MakeAiterTensor(v_pool_out->untyped_data(), v_pool_in.dimensions(),
                               v_pool_in.element_type(), device_id,
                               &t_value_cache, "v_pool");
      e.failure())
    return e;

  // Passed straight through as int32: aiter derives the index width from the
  // descriptor's dtype, so no widening pass is needed.
  if (auto e = MakeAiterTensor(slot_mapping, device_id, &t_slots, "slot_mapping");
      e.failure())
    return e;
  if (auto e = MakeAiterTensor(k_scale, device_id, &t_kscale, "k_scale"); e.failure()) return e;
  if (auto e = MakeAiterTensor(v_scale, device_id, &t_vscale, "v_scale"); e.failure()) return e;

  // aiter reads its stream from thread-local state, so install XLA's for the
  // duration of the call or the write races the surrounding graph.
  ScopedAiterStream stream_guard(stream);

  const std::string dtype_str(kv_cache_dtype);
  aiter::reshape_and_cache_flash(t_key, t_value, t_key_cache, t_value_cache,
                                 t_slots, dtype_str, t_kscale, t_vscale);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      std::string("AppendKvJA: reshape_and_cache_flash failed: ") +
                          hipGetErrorString(err));
  }

  return ffi::Error::Success();
}

} // namespace jax_aiter

#pragma GCC visibility push(default)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    AppendKvJA, jax_aiter::AppendKv_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Ctx<ffi::DeviceOrdinal>()
        .Arg<ffi::AnyBuffer>() // k_new        [num_tokens, num_kv_heads, head_dim]
        .Arg<ffi::AnyBuffer>() // v_new        [num_tokens, num_kv_heads, head_dim]
        .Arg<ffi::AnyBuffer>() // slot_mapping [num_tokens] int32
        .Arg<ffi::AnyBuffer>() // k_scale      [1] f32
        .Arg<ffi::AnyBuffer>() // v_scale      [1] f32
        .Arg<ffi::AnyBuffer>() // k_pool       aliased -> result 0
        .Arg<ffi::AnyBuffer>() // v_pool       aliased -> result 1
        .Ret<ffi::AnyBuffer>() // k_pool'
        .Ret<ffi::AnyBuffer>() // v_pool'
        .Attr<std::string_view>("kv_cache_dtype"));

#pragma GCC visibility pop
