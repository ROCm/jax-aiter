// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>
#include <vector>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "hip_utils.h"
#include "mha_bwd.h"
#include "mha_common_utils.cu"
#include "mha_common_utils.h"

namespace ffi = xla::ffi;

namespace jax_aiter {

static size_t compute_dq_acc_size(int64_t batch_size, int64_t seqlen_q,
                                  int64_t seqlen_k, int64_t num_heads,
                                  int64_t head_size, bool deterministic,
                                  bool is_v3_atomic_fp32,
                                  xla::ffi::DataType q_dtype,
                                  std::vector<int64_t> &out_shape) {

  size_t element_size = 4;

  if (!deterministic) {
    if (is_v3_atomic_fp32) {
      out_shape = {1, batch_size, num_heads, seqlen_q, head_size};
      element_size = 4;
    } else {
      int64_t seqlen_q_padded = ((seqlen_q + 15) / 16) * 16;
      out_shape = {1, batch_size, num_heads, seqlen_q_padded, 128};
      element_size = (q_dtype == xla::ffi::DataType::F16 ||
                      q_dtype == xla::ffi::DataType::BF16)
                         ? 2
                         : 4;
    }
  } else {
    const ck_tile::index_t kN0 = head_size <= 128 ? 128 : 64;
    const ck_tile::index_t nsplits =
        ck_tile::integer_divide_ceil(seqlen_k, kN0);
    out_shape = {nsplits, batch_size, num_heads, seqlen_q, head_size};
    element_size = 4;
  }

  size_t total_elements = 1;
  for (auto dim : out_shape) {
    total_elements *= dim;
  }

  return total_elements * element_size;
}

ffi::Error FmhaV3Bwd_Bridge(
    hipStream_t stream, ffi::AnyBuffer dout, ffi::AnyBuffer q, ffi::AnyBuffer k,
    ffi::AnyBuffer v, ffi::AnyBuffer out, ffi::AnyBuffer softmax_lse,
    std::optional<ffi::AnyBuffer> dq_, std::optional<ffi::AnyBuffer> dk_,
    std::optional<ffi::AnyBuffer> dv_,
    std::optional<ffi::AnyBuffer> alibi_slopes_,
    std::optional<ffi::AnyBuffer> rng_state_,
    std::optional<ffi::AnyBuffer> gen_, ffi::Result<ffi::AnyBuffer> dq,
    ffi::Result<ffi::AnyBuffer> dk, ffi::Result<ffi::AnyBuffer> dv,
    ffi::Result<ffi::AnyBuffer> softmax_d, float dropout_p, float softmax_scale,
    bool is_causal, int window_size_left, int window_size_right,
    bool deterministic, bool is_v3_atomic_fp32, int how_v3_bf16_cvt) {

  if (!q.untyped_data() || !k.untyped_data() || !v.untyped_data() ||
      !out.untyped_data() || !softmax_lse.untyped_data() ||
      !dout.untyped_data()) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "Required input buffer (q/k/v/out/lse/dout) is null");
  }

  const int dev_idx = ::jax_aiter::device_from_ptr(q.untyped_data());
  if (dev_idx < 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "bad device from q");
  }

  auto q_dims = q.dimensions();
  auto k_dims = k.dimensions();
  auto v_dims = v.dimensions();
  auto dout_dims = dout.dimensions();
  auto out_dims = out.dimensions();
  auto lse_dims = softmax_lse.dimensions();

  int64_t batch_size = q_dims[0];
  int64_t seqlen_q = q_dims[1];
  int64_t num_heads_q = q_dims[2];
  int64_t head_size_q = q_dims[3];

  int64_t seqlen_k = k_dims[1];
  int64_t num_heads_k = k_dims[2];
  int64_t head_size_v = v_dims[3];

  if (seqlen_q == 0) {
    if (dq->size_bytes() > 0) {
      HIP_CHECK(
          hipMemsetAsync(dq->untyped_data(), 0, dq->size_bytes(), stream));
    }
    if (dk->size_bytes() > 0) {
      HIP_CHECK(
          hipMemsetAsync(dk->untyped_data(), 0, dk->size_bytes(), stream));
    }
    if (dv->size_bytes() > 0) {
      HIP_CHECK(
          hipMemsetAsync(dv->untyped_data(), 0, dv->size_bytes(), stream));
    }
    if (softmax_d->size_bytes() > 0) {
      HIP_CHECK(hipMemsetAsync(softmax_d->untyped_data(), 0,
                               softmax_d->size_bytes(), stream));
    }
    return ffi::Error::Success();
  }

  try {
    mha_utils::validate_mha_bwd_inputs(dout, q, k, v, out, softmax_lse,
                                       head_size_q, head_size_v, num_heads_q,
                                       num_heads_k);
  } catch (const std::exception &e) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, e.what());
  }

  if (num_heads_q != num_heads_k) {
    return ffi::Error(ffi::ErrorCode::kUnimplemented,
                      "MQA/GQA not yet supported in v3 backward path");
  }

  std::string dtype_str = mha_utils::dtype_to_string(q.element_type());

  auto mask = mha_utils::create_mask_info(
      is_causal, window_size_left, window_size_right, seqlen_q, seqlen_k);

  const void *alibi_ptr = nullptr;
  ck_tile::index_t stride_alibi = 0;

  if (alibi_slopes_.has_value() && mha_utils::is_valid_buffer(*alibi_slopes_)) {
    alibi_ptr = alibi_slopes_->untyped_data();
    auto alibi_dims = alibi_slopes_->dimensions();
    stride_alibi =
        alibi_dims.size() >= 2 ? mha_utils::calculate_stride(alibi_dims, 0) : 0;
  }

  bias_enum bias_type =
      (alibi_ptr != nullptr) ? bias_enum::alibi : bias_enum::no_bias;

  uint64_t *seed_ptr = nullptr;
  uint64_t *offset_ptr = nullptr;

  try {
    auto [seed, offset] =
        mha_utils::get_rng_seed_offset_ptrs(rng_state_, dropout_p);
    seed_ptr = seed;
    offset_ptr = offset;
  } catch (const std::exception &e) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, e.what());
  }

  std::vector<int64_t> dq_acc_shape;
  size_t dq_acc_bytes = compute_dq_acc_size(
      batch_size, seqlen_q, seqlen_k, num_heads_q, head_size_v, deterministic,
      is_v3_atomic_fp32, q.element_type(), dq_acc_shape);

  void *dq_acc_ptr = nullptr;
  HIP_CHECK(hipMalloc(&dq_acc_ptr, dq_acc_bytes));
  HIP_CHECK(hipMemsetAsync(dq_acc_ptr, 0, dq_acc_bytes, stream));

  ck_tile::index_t split_stride_dq_acc = 1;
  ck_tile::index_t batch_stride_dq_acc = 1;
  ck_tile::index_t nhead_stride_dq_acc = 1;
  ck_tile::index_t stride_dq_acc = 1;

  if (dq_acc_shape.size() == 5) {
    std::vector<ck_tile::index_t> strides(5);
    strides[4] = 1;
    for (int i = 3; i >= 0; i--) {
      strides[i] = strides[i + 1] * dq_acc_shape[i + 1];
    }

    split_stride_dq_acc = strides[0];
    batch_stride_dq_acc = strides[1];
    nhead_stride_dq_acc = strides[2];
    stride_dq_acc = strides[3];
  }

  ck_tile::index_t stride_q = mha_utils::calculate_stride(q_dims, 1);
  ck_tile::index_t stride_k = mha_utils::calculate_stride(k_dims, 1);
  ck_tile::index_t stride_v = mha_utils::calculate_stride(v_dims, 1);
  ck_tile::index_t stride_o = mha_utils::calculate_stride(out_dims, 1);
  ck_tile::index_t stride_do = mha_utils::calculate_stride(dout_dims, 1);

  auto dq_dims = dq->dimensions();
  auto dk_dims = dk->dimensions();
  auto dv_dims = dv->dimensions();

  ck_tile::index_t stride_dq = mha_utils::calculate_stride(dq_dims, 1);
  ck_tile::index_t stride_dk = mha_utils::calculate_stride(dk_dims, 1);
  ck_tile::index_t stride_dv = mha_utils::calculate_stride(dv_dims, 1);

  ck_tile::index_t nhead_stride_q = mha_utils::calculate_stride(q_dims, 2);
  ck_tile::index_t nhead_stride_k = mha_utils::calculate_stride(k_dims, 2);
  ck_tile::index_t nhead_stride_v = mha_utils::calculate_stride(v_dims, 2);
  ck_tile::index_t nhead_stride_o = mha_utils::calculate_stride(out_dims, 2);
  ck_tile::index_t nhead_stride_do = mha_utils::calculate_stride(dout_dims, 2);
  ck_tile::index_t nhead_stride_lse = mha_utils::calculate_stride(lse_dims, 1);
  ck_tile::index_t nhead_stride_dq = mha_utils::calculate_stride(dq_dims, 2);
  ck_tile::index_t nhead_stride_dk = mha_utils::calculate_stride(dk_dims, 2);
  ck_tile::index_t nhead_stride_dv = mha_utils::calculate_stride(dv_dims, 2);

  ck_tile::index_t batch_stride_q = mha_utils::calculate_stride(q_dims, 0);
  ck_tile::index_t batch_stride_k = mha_utils::calculate_stride(k_dims, 0);
  ck_tile::index_t batch_stride_v = mha_utils::calculate_stride(v_dims, 0);
  ck_tile::index_t batch_stride_o = mha_utils::calculate_stride(out_dims, 0);
  ck_tile::index_t batch_stride_do = mha_utils::calculate_stride(dout_dims, 0);
  ck_tile::index_t batch_stride_lse = mha_utils::calculate_stride(lse_dims, 0);
  ck_tile::index_t batch_stride_dq = mha_utils::calculate_stride(dq_dims, 0);
  ck_tile::index_t batch_stride_dk = mha_utils::calculate_stride(dk_dims, 0);
  ck_tile::index_t batch_stride_dv = mha_utils::calculate_stride(dv_dims, 0);

  float p_undrop = mha_utils::calculate_p_undrop(dropout_p);

  auto args = fmha_bwd_args{q.untyped_data(),
                            k.untyped_data(),
                            v.untyped_data(),
                            alibi_ptr,
                            out.untyped_data(),
                            softmax_lse.untyped_data(),
                            dout.untyped_data(),
                            softmax_d->untyped_data(),
                            nullptr,
                            dq->untyped_data(),
                            dk->untyped_data(),
                            dv->untyped_data(),
                            nullptr,
                            dq_acc_ptr,
                            nullptr,
                            nullptr,
                            nullptr,
                            static_cast<ck_tile::index_t>(seqlen_q),
                            static_cast<ck_tile::index_t>(seqlen_k),
                            static_cast<ck_tile::index_t>(batch_size),
                            static_cast<ck_tile::index_t>(seqlen_q),
                            static_cast<ck_tile::index_t>(seqlen_k),
                            static_cast<ck_tile::index_t>(head_size_q),
                            static_cast<ck_tile::index_t>(head_size_v),
                            static_cast<ck_tile::index_t>(num_heads_q),
                            static_cast<ck_tile::index_t>(num_heads_k),
                            softmax_scale,
                            stride_q,
                            stride_k,
                            stride_v,
                            stride_alibi,
                            stride_o,
                            0,
                            stride_do,
                            stride_dq_acc,
                            stride_dq,
                            stride_dk,
                            stride_dv,
                            0,
                            nhead_stride_q,
                            nhead_stride_k,
                            nhead_stride_v,
                            0,
                            nhead_stride_o,
                            0,
                            nhead_stride_do,
                            nhead_stride_lse,
                            nhead_stride_dq_acc,
                            nhead_stride_dq,
                            nhead_stride_dk,
                            nhead_stride_dv,
                            0,
                            batch_stride_q,
                            batch_stride_k,
                            batch_stride_v,
                            0,
                            batch_stride_o,
                            0,
                            batch_stride_do,
                            batch_stride_lse,
                            batch_stride_dq_acc,
                            batch_stride_dq,
                            batch_stride_dk,
                            batch_stride_dv,
                            0,
                            split_stride_dq_acc,
                            mask.left,
                            mask.right,
                            static_cast<ck_tile::index_t>(mask.type),
                            dropout_p,
                            p_undrop,
                            std::make_pair(seed_ptr, offset_ptr)};

  auto stream_config = mha_utils::create_stream_config(stream);

  float runtime = aiter::mha_bwd(
      args, stream_config, dtype_str, false, mask.type, bias_type, false, false,
      deterministic, true, is_v3_atomic_fp32, how_v3_bf16_cvt);

  hipFree(dq_acc_ptr);

  if (runtime < 0) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      "aiter::mha_bwd failed - invalid arguments or "
                      "unsupported configuration");
  }

  return ffi::Error::Success();
}

} // namespace jax_aiter

#pragma GCC visibility push(default)

XLA_FFI_DEFINE_HANDLER_SYMBOL(FmhaV3BwdJA, jax_aiter::FmhaV3Bwd_Bridge,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<hipStream_t>>()
                                  .Arg<ffi::AnyBuffer>() // dout
                                  .Arg<ffi::AnyBuffer>() // q
                                  .Arg<ffi::AnyBuffer>() // k
                                  .Arg<ffi::AnyBuffer>() // v
                                  .Arg<ffi::AnyBuffer>() // o
                                  .Arg<ffi::AnyBuffer>() // lse
                                  .Arg<ffi::AnyBuffer>() // dq_provided
                                  .Arg<ffi::AnyBuffer>() // dk_provided
                                  .Arg<ffi::AnyBuffer>() // dv_provided
                                  .Arg<ffi::AnyBuffer>() // alibi_slopes
                                  .Arg<ffi::AnyBuffer>() // rng_state_provided
                                  .Arg<ffi::AnyBuffer>() // gen
                                  .Ret<ffi::AnyBuffer>() // dq
                                  .Ret<ffi::AnyBuffer>() // dk
                                  .Ret<ffi::AnyBuffer>() // dv
                                  .Ret<ffi::AnyBuffer>() // softmax_d
                                  .Attr<float>("dropout_p")
                                  .Attr<float>("softmax_scale")
                                  .Attr<bool>("is_causal")
                                  .Attr<int>("window_size_left")
                                  .Attr<int>("window_size_right")
                                  .Attr<bool>("deterministic")
                                  .Attr<bool>("is_v3_atomic_fp32")
                                  .Attr<int>("how_v3_bf16_cvt"),
                              {xla::ffi::Traits::kCmdBufferCompatible});

#pragma GCC visibility pop
