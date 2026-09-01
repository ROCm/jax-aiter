// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Shared glue for driving aiter's torch-free C++ entry points from an XLA FFI
// handler. Used by the paged-KV shims (append_kv, paged attention).
//
// Two things here are load-bearing and easy to get wrong:
//
//  1. aiter's kernels do not take a stream. They read a *thread-local* stream
//     via aiter::getCurrentHIPStream(), so every handler must install XLA's
//     stream for the duration of the call. Setting it once at registration is
//     not enough, because XLA may call handlers from several threads, and
//     getting it wrong means the kernel runs unordered against XLA's own work
//     -- a race that shows up as intermittently stale KV rather than a crash.
//
//  2. aiter_tensor_t carries explicit strides. They are computed here from the
//     buffer dimensions rather than assumed, because the packed pool layouts are
//     stride-driven and a wrong stride is silent.
//
// The descriptor carries dtype, but reshape_and_cache_flash still reads
// slot_mapping as int64_t*. AppendKvJA widens int32 slots before that call.

#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "xla/ffi/api/ffi.h"

#include "aiter_enum.h"
#include "aiter_stream.h"
#include "aiter_tensor.h"

namespace ffi = xla::ffi;

namespace jax_aiter {

// Installs `stream` as aiter's current stream and restores the previous value on
// scope exit.
class ScopedAiterStream {
public:
  explicit ScopedAiterStream(hipStream_t stream)
      : prev_(aiter::getCurrentHIPStream()) {
    aiter::setCurrentHIPStream(stream);
  }
  ~ScopedAiterStream() { aiter::setCurrentHIPStream(prev_); }

  ScopedAiterStream(const ScopedAiterStream &) = delete;
  ScopedAiterStream &operator=(const ScopedAiterStream &) = delete;

private:
  hipStream_t prev_;
};

inline bool FfiDtypeToAiter(ffi::DataType dt, AiterDtype *out) {
  switch (dt) {
  case ffi::DataType::F32: *out = AITER_DTYPE_fp32; return true;
  case ffi::DataType::F16: *out = AITER_DTYPE_fp16; return true;
  case ffi::DataType::BF16: *out = AITER_DTYPE_bf16; return true;
  case ffi::DataType::S32: *out = AITER_DTYPE_i32; return true;
  case ffi::DataType::S64: *out = AITER_DTYPE_i64; return true;
  case ffi::DataType::S8: *out = AITER_DTYPE_i8; return true;
  case ffi::DataType::U8: *out = AITER_DTYPE_u8; return true;
  case ffi::DataType::U32: *out = AITER_DTYPE_u32; return true;
  default: return false;
  }
}

// Fills `out` to describe a dense row-major buffer. XLA hands FFI handlers
// buffers in descending-minor (row-major) layout, so strides follow from the
// dimensions; they are still written out explicitly.
inline ffi::Error MakeAiterTensor(void *data, ffi::Span<const int64_t> dims,
                                  ffi::DataType dtype, int device_id,
                                  aiter_tensor_t *out, const char *name) {
  if (dims.size() > 8) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      std::string(name) + ": rank > 8 unsupported by aiter");
  }

  AiterDtype adt;
  if (!FfiDtypeToAiter(dtype, &adt)) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      std::string(name) + ": dtype not representable as AiterDtype");
  }

  *out = aiter_tensor_t{};
  out->ptr = data;
  out->ndim = static_cast<int>(dims.size());
  out->dtype_ = adt;
  out->device_id = device_id;

  int64_t numel = 1;
  for (size_t i = 0; i < dims.size(); ++i) {
    out->shape[i] = dims[i];
    numel *= dims[i];
  }
  out->numel_ = static_cast<size_t>(numel);

  if (out->ndim > 0) {
    out->strides[out->ndim - 1] = 1;
    for (int d = out->ndim - 2; d >= 0; --d) {
      out->strides[d] = out->strides[d + 1] * out->shape[d + 1];
    }
  }

  return ffi::Error::Success();
}

inline ffi::Error MakeAiterTensor(ffi::AnyBuffer buf, int device_id,
                                  aiter_tensor_t *out, const char *name) {
  return MakeAiterTensor(buf.untyped_data(), buf.dimensions(),
                         buf.element_type(), device_id, out, name);
}

} // namespace jax_aiter
