// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include "hip_utils.h"
#include "logging.h"
#include "xla/ffi/api/ffi.h"
#include <torch/extension.h>

namespace jax_aiter {

// Data type helpers.
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

// Maps XLA DataType to PyTorch scalar types.
inline at::ScalarType xla_to_torch_dtype(xla::ffi::DataType xla_type) {
  JA_LOG("xla_to_torch_dtype called with XLA DataType: %d (hex: 0x%x)",
         static_cast<int>(xla_type), static_cast<int>(xla_type));

  switch (xla_type) {
  case xla::ffi::DataType::INVALID:
    throw std::runtime_error("Invalid XLA DataType");
  case xla::ffi::DataType::PRED:
    return torch::kBool;
  case xla::ffi::DataType::S1:
    return torch::kInt8; // Map to closest available type
  case xla::ffi::DataType::S2:
    return torch::kInt8; // Map to closest available type
  case xla::ffi::DataType::S4:
    return torch::kInt8; // Map to closest available type
  case xla::ffi::DataType::S8:
    return torch::kInt8;
  case xla::ffi::DataType::S16:
    return torch::kInt16;
  case xla::ffi::DataType::S32:
    return torch::kInt32;
  case xla::ffi::DataType::S64:
    return torch::kInt64;
  case xla::ffi::DataType::U1:
    return torch::kUInt8; // Map to closest available type
  case xla::ffi::DataType::U2:
    return torch::kUInt8; // Map to closest available type
  case xla::ffi::DataType::U4:
    return torch::kUInt8; // Map to closest available type
  case xla::ffi::DataType::U8:
    return torch::kUInt8;
  case xla::ffi::DataType::U16:
    return torch::kUInt16;
  case xla::ffi::DataType::U32:
    return torch::kUInt32;
  case xla::ffi::DataType::U64:
    return torch::kUInt64;
  case xla::ffi::DataType::F16:
    return torch::kFloat16;
  case xla::ffi::DataType::F32:
    return torch::kFloat32;
  case xla::ffi::DataType::F64:
    return torch::kFloat64;
  case xla::ffi::DataType::BF16:
    return torch::kBFloat16;
  case xla::ffi::DataType::C64:
    return torch::kComplexFloat;
  case xla::ffi::DataType::C128:
    return torch::kComplexDouble;
  case xla::ffi::DataType::F8E4M3:
    return torch::kFloat8_e4m3fn; // Map to closest available type
  case xla::ffi::DataType::F8E4M3FN:
    return torch::kFloat8_e4m3fn;
  case xla::ffi::DataType::F8E5M2:
    return torch::kFloat8_e5m2;
  case xla::ffi::DataType::F8E4M3FNUZ:
    return torch::kFloat8_e4m3fnuz;
  case xla::ffi::DataType::F8E5M2FNUZ:
    return torch::kFloat8_e5m2fnuz;
  case xla::ffi::DataType::F8E4M3B11FNUZ:
    return torch::kFloat8_e4m3fnuz; // Map to closest available type
  case xla::ffi::DataType::F8E3M4:
    return torch::kFloat8_e4m3fn; // Map to closest available type
  case xla::ffi::DataType::F4E2M1FN:
    return torch::kFloat16; // Map to closest available type
  case xla::ffi::DataType::F8E8M0FNU:
    return torch::kFloat8_e5m2; // Map to closest available type
  case xla::ffi::DataType::TOKEN:
    throw std::runtime_error(
        "TOKEN DataType not supported for tensor conversion");
  default:
    throw std::runtime_error("Unsupported XLA DataType: " +
                             std::to_string(static_cast<int>(xla_type)));
  }
}

// Computes row-major strides from shape.
inline std::vector<int64_t> compute_row_major_strides(at::IntArrayRef shape) {
  std::vector<int64_t> strides(shape.size());
  if (shape.empty())
    return strides;

  strides.back() = 1;
  for (int i = shape.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }
  return strides;
}

// Creates non-owning tensor view with custom strides.
template <typename T>
inline at::Tensor
wrap_buffer(void *data, at::IntArrayRef shape, at::IntArrayRef strides,
            int device_idx = current_device(), bool requires_grad = false) {
  if (!data) {
    throw std::runtime_error("wrap_buffer: null data pointer");
  }

  auto opts = at::TensorOptions()
                  .dtype(torch_dtype<T>())
                  .device(at::kCUDA, device_idx)
                  .requires_grad(requires_grad);

  // Non-owning tensor - null deleter.
  return at::from_blob(
      data, shape, strides, [](void *) {}, opts);
}

// Creates non-owning tensor view with row-major strides.
template <typename T>
inline at::Tensor wrap_buffer(void *data, at::IntArrayRef shape,
                              int device_idx = current_device(),
                              bool requires_grad = false) {
  auto strides = compute_row_major_strides(shape);
  return wrap_buffer<T>(data, shape, strides, device_idx, requires_grad);
}

// Creates tensor from XLA buffer with auto-detected shape and type.
inline at::Tensor wrap_any_buffer(xla::ffi::AnyBuffer buffer,
                                  int device_idx = current_device(),
                                  bool requires_grad = false) {
  // Extract shape from buffer dimensions.
  auto dims = buffer.dimensions();
  std::vector<int64_t> shape_vec(dims.begin(), dims.end());
  at::IntArrayRef shape(shape_vec);

  JA_LOG("wrap_any_buffer called with buffer size: %zu bytes, element_type: %d "
         "(hex: 0x%x)",
         buffer.size_bytes(), static_cast<int>(buffer.element_type()),
         static_cast<int>(buffer.element_type()));

  auto strides = compute_row_major_strides(shape);
  auto torch_dtype = xla_to_torch_dtype(buffer.element_type());

  // Validate inputs before tensor creation.
  if (!buffer.untyped_data()) {
    throw std::runtime_error("wrap_any_buffer: null buffer data");
  }
  if (device_idx < 0) {
    throw std::runtime_error("wrap_any_buffer: invalid device index: " +
                             std::to_string(device_idx));
  }
  for (auto dim : shape) {
    if (dim <= 0) {
      throw std::runtime_error("wrap_any_buffer: invalid shape dimension: " +
                               std::to_string(dim));
    }
  }

  auto opts = at::TensorOptions()
                  .dtype(torch_dtype)
                  .device(at::kCUDA, device_idx)
                  .requires_grad(requires_grad);

  // Non-owning tensor - null deleter.
  return at::from_blob(
      buffer.untyped_data(), shape, strides, [](void *) {}, opts);
}

// Creates tensor from XLA buffer with explicit shape override.
inline at::Tensor wrap_any_buffer(xla::ffi::AnyBuffer buffer,
                                  at::IntArrayRef shape,
                                  int device_idx = current_device(),
                                  bool requires_grad = false) {
  if (!buffer.untyped_data()) {
    throw std::runtime_error("wrap_any_buffer: null buffer data");
  }

  auto strides = compute_row_major_strides(shape);
  auto torch_dtype = xla_to_torch_dtype(buffer.element_type());

  auto opts = at::TensorOptions()
                  .dtype(torch_dtype)
                  .device(at::kCUDA, device_idx)
                  .requires_grad(requires_grad);

  // Non-owning tensor - null deleter.
  return at::from_blob(
      buffer.untyped_data(), shape, strides, [](void *) {}, opts);
}

// Creates tensor from XLA buffer with explicit shape and strides.
inline at::Tensor wrap_any_buffer(xla::ffi::AnyBuffer buffer,
                                  at::IntArrayRef shape,
                                  at::IntArrayRef strides,
                                  int device_idx = current_device(),
                                  bool requires_grad = false) {
  if (!buffer.untyped_data()) {
    throw std::runtime_error("wrap_any_buffer: null buffer data");
  }

  auto torch_dtype = xla_to_torch_dtype(buffer.element_type());

  auto opts = at::TensorOptions()
                  .dtype(torch_dtype)
                  .device(at::kCUDA, device_idx)
                  .requires_grad(requires_grad);

  // Non-owning tensor - null deleter.
  return at::from_blob(
      buffer.untyped_data(), shape, strides, [](void *) {}, opts);
}

} // namespace jax_aiter
