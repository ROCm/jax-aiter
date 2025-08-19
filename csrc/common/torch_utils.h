// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include "hip_utils.h"
#include "logging.h"
#include <torch/extension.h>
#include "xla/ffi/api/ffi.h"

namespace jax_aiter {

// dtype helpers.
template <typename T> constexpr at::ScalarType torch_dtype();
template <> constexpr at::ScalarType torch_dtype<int8_t>() {
  return torch::kInt8;
}
template <> constexpr at::ScalarType torch_dtype<uint8_t>() {
  return torch::kUInt8;
}
template <> constexpr at::ScalarType torch_dtype<float>() {
  return torch::kFloat32;
}
template <> constexpr at::ScalarType torch_dtype<at::Half>() {
  return torch::kFloat16;
}
template <> constexpr at::ScalarType torch_dtype<at::BFloat16>() {
  return torch::kBFloat16;
}

// Helper to map XLA DataType to PyTorch scalar types
inline at::ScalarType xla_to_torch_dtype(xla::ffi::DataType xla_type) {
  switch (xla_type) {
    case xla::ffi::DataType::F16: return torch::kFloat16;
    case xla::ffi::DataType::BF16: return torch::kBFloat16;
    case xla::ffi::DataType::F32: return torch::kFloat32;
    case xla::ffi::DataType::S8: return torch::kInt8;
    case xla::ffi::DataType::U8: return torch::kUInt8;
    case xla::ffi::DataType::F8E4M3FNUZ: return torch::kUInt8; // Treat as uint8 for kernel interface
    default:
      throw std::runtime_error("Unsupported XLA DataType");
  }
}

// Helper to compute row-major strides from shape
inline std::vector<int64_t> compute_row_major_strides(at::IntArrayRef shape) {
  std::vector<int64_t> strides(shape.size());
  if (shape.empty()) return strides;
  
  strides.back() = 1;
  for (int i = shape.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
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

// Overload with automatic row-major stride computation
template <typename T>
inline at::Tensor
wrap_buffer(void *data, at::IntArrayRef shape,
            int device_idx = current_device(), bool requires_grad = false) {
  auto strides = compute_row_major_strides(shape);
  return wrap_buffer<T>(data, shape, strides, device_idx, requires_grad);
}

// Type-erased wrapper using AnyBuffer - extracts shape, type, and strides automatically
inline at::Tensor
wrap_any_buffer(xla::ffi::AnyBuffer buffer,
                int device_idx = current_device(), bool requires_grad = false) {
  // Extract shape directly from AnyBuffer dimensions
  auto dims = buffer.dimensions();
  std::vector<int64_t> shape_vec(dims.begin(), dims.end());
  at::IntArrayRef shape(shape_vec);
  
  auto strides = compute_row_major_strides(shape);
  auto torch_dtype = xla_to_torch_dtype(buffer.element_type());
  
  auto opts = at::TensorOptions()
                  .dtype(torch_dtype)
                  .device(at::kCUDA, device_idx)
                  .requires_grad(requires_grad);

  // Null deleter - i.e. we don't own the buffer.
  return at::from_blob(
      buffer.untyped_data(), shape, strides, [](void *) {}, opts);
}

// Type-erased wrapper using AnyBuffer with explicit shape override
inline at::Tensor
wrap_any_buffer(xla::ffi::AnyBuffer buffer, at::IntArrayRef shape,
                int device_idx = current_device(), bool requires_grad = false) {
  auto strides = compute_row_major_strides(shape);
  auto torch_dtype = xla_to_torch_dtype(buffer.element_type());
  
  auto opts = at::TensorOptions()
                  .dtype(torch_dtype)
                  .device(at::kCUDA, device_idx)
                  .requires_grad(requires_grad);

  // Null deleter - i.e. we don't own the buffer.
  return at::from_blob(
      buffer.untyped_data(), shape, strides, [](void *) {}, opts);
}

// Type-erased wrapper using AnyBuffer with explicit shape and strides
inline at::Tensor
wrap_any_buffer(xla::ffi::AnyBuffer buffer, at::IntArrayRef shape, at::IntArrayRef strides,
                int device_idx = current_device(), bool requires_grad = false) {
  auto torch_dtype = xla_to_torch_dtype(buffer.element_type());
  
  auto opts = at::TensorOptions()
                  .dtype(torch_dtype)
                  .device(at::kCUDA, device_idx)
                  .requires_grad(requires_grad);

  // Null deleter - i.e. we don't own the buffer.
  return at::from_blob(
      buffer.untyped_data(), shape, strides, [](void *) {}, opts);
}

} // namespace jax_aiter
