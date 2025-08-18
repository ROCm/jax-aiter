// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include "hip_utils.h"
#include "logging.h"
#include <torch/extension.h>

namespace jax_aiter {

// dtype helpers.

template <typename T> constexpr at::ScalarType torch_dtype();
template <> constexpr at::ScalarType torch_dtype<int8_t>() {
  return torch::kInt8;
}
template <> constexpr at::ScalarType torch_dtype<float>() {
  return torch::kFloat32;
}
template <> constexpr at::ScalarType torch_dtype<at::BFloat16>() {
  return torch::kBFloat16;
}

// Non-owning view that never frees the memory.  Works for any scalar
// type known to torch_dtype<T>(). `strides` are element-strides.

template <typename T>
inline at::Tensor
wrap_buffer(void *data, at::IntArrayRef shape, at::IntArrayRef strides,
            int device_idx = current_device(), bool requires_grad = false) {
  auto opts = at::TensorOptions()
                  .dtype(torch_dtype<T>())
                  .device(at::kCUDA, device_idx)
                  .requires_grad(requires_grad);

  // Null deleter - i.e. we don't own the buffer.
  return at::from_blob(
      data, shape, strides, [](void *) {}, opts);
}

} // namespace jax_aiter
