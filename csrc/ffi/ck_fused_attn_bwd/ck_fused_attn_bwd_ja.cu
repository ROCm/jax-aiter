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

static size_t compute_dq_acc_size_ck(int64_t batch_size, int64_t seqlen_q,
                                     int64_t seqlen_k, int64_t num_heads,
                                     int64_t head_size, bool deterministic,
                                     mask_enum mask_type,
                                     std::vector<int64_t> &out_shape) {

  size_t element_size = 4;

  if (!deterministic) {
    out_shape = {1, batch_size, seqlen_q, num_heads, head_size};
  } else {
    const ck_tile::index_t kN0 = head_size <= 128 ? 128 : 64;
    const ck_tile::index_t nsplits =
        ck_tile::integer_divide_ceil(seqlen_k, kN0);
    out_shape = {nsplits, batch_size, seqlen_q, num_heads, head_size};
  }

  size_t total_elements = 1;
  for (auto dim : out_shape) {
    total_elements *= dim;
  }

  return total_elements * element_size;
}

ffi::Error MhaBwd_Bridge(
    hipStream_t stream,
    ffi::AnyBuffer dout,                         // [b, sq, hq, d_v]
    ffi::AnyBuffer q,                            // [b, sq, hq, d]
    ffi::AnyBuffer k,                            // [b, sk, hk, d]
    ffi::AnyBuffer v,                            // [b, sk, hk, d_v]
    ffi::AnyBuffer out,                          // [b, sq, hq, d_v]
    ffi::AnyBuffer softmax_lse,                  // [b, hq, sq]
    std::optional<ffi::AnyBuffer> dq_,           // [b, sq, hq, d]
    std::optional<ffi::AnyBuffer> dk_,           // [b, sk, hk, d]
    std::optional<ffi::AnyBuffer> dv_,           // [b, sk, hk, d_v]
    std::optional<ffi::AnyBuffer> bias_,         // [sq, sk]
    std::optional<ffi::AnyBuffer> alibi_slopes_, // [hq] or [b, hq]
    std::optional<ffi::AnyBuffer> rng_state_,
    std::optional<ffi::AnyBuffer> gen_, ffi::Result<ffi::AnyBuffer> dq_ret,
    ffi::Result<ffi::AnyBuffer> dk_ret, ffi::Result<ffi::AnyBuffer> dv_ret,
    ffi::Result<ffi::AnyBuffer> softmax_d_ret,
    ffi::Result<ffi::AnyBuffer> dbias_ret, float dropout_p, float softmax_scale,
    bool is_causal, int window_size_left, int window_size_right,
    bool deterministic) {

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
  int64_t num_heads = q_dims[2];
  int64_t head_size_q = q_dims[3];
  int64_t seqlen_k = k_dims[1];
  int64_t num_heads_k = k_dims[2];
  int64_t head_size_v = v_dims[3];

  if (seqlen_q == 0) {
    if (dq_ret->size_bytes() > 0) {
      HIP_CHECK(hipMemsetAsync(dq_ret->untyped_data(), 0, dq_ret->size_bytes(),
                               stream));
    }
    if (dk_ret->size_bytes() > 0) {
      HIP_CHECK(hipMemsetAsync(dk_ret->untyped_data(), 0, dk_ret->size_bytes(),
                               stream));
    }
    if (dv_ret->size_bytes() > 0) {
      HIP_CHECK(hipMemsetAsync(dv_ret->untyped_data(), 0, dv_ret->size_bytes(),
                               stream));
    }
    if (softmax_d_ret->size_bytes() > 0) {
      HIP_CHECK(hipMemsetAsync(softmax_d_ret->untyped_data(), 0,
                               softmax_d_ret->size_bytes(), stream));
    }
    if (dbias_ret->size_bytes() > 0) {
      HIP_CHECK(hipMemsetAsync(dbias_ret->untyped_data(), 0,
                               dbias_ret->size_bytes(), stream));
    }
    return ffi::Error::Success();
  }

  try {
    mha_utils::validate_mha_bwd_inputs(dout, q, k, v, out, softmax_lse,
                                       head_size_q, head_size_v, num_heads,
                                       num_heads_k);
  } catch (const std::exception &e) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, e.what());
  }

  if (num_heads % num_heads_k != 0) {
    return ffi::Error(
        ffi::ErrorCode::kInvalidArgument,
        "Number of heads in q must be divisible by number of heads in k/v");
  }

  bool is_mqa_gqa = (num_heads != num_heads_k);

  std::string dtype_str = mha_utils::dtype_to_string(q.element_type());

  auto mask = mha_utils::create_mask_info(
      is_causal, window_size_left, window_size_right, seqlen_q, seqlen_k);

  const void *bias_ptr = nullptr;
  ck_tile::index_t stride_bias = 0;
  bool has_bias = bias_.has_value() && mha_utils::is_valid_buffer(*bias_);
  bool has_alibi =
      alibi_slopes_.has_value() && mha_utils::is_valid_buffer(*alibi_slopes_);

  if (has_bias) {
    bias_ptr = bias_->untyped_data();
    auto bias_dims = bias_->dimensions();
    stride_bias =
        bias_dims.size() >= 2 ? mha_utils::calculate_stride(bias_dims, 0) : 0;
  } else if (has_alibi) {
    bias_ptr = alibi_slopes_->untyped_data();
    auto alibi_dims = alibi_slopes_->dimensions();
    stride_bias =
        alibi_dims.size() >= 2 ? mha_utils::calculate_stride(alibi_dims, 0) : 0;
  }

  bias_enum bias_type = mha_utils::get_bias_type(has_bias, has_alibi);

  // Allocate dbias_expanded if bias present; Python will reduce to final shape.
  bool has_dbias = has_bias && (dbias_ret->size_bytes() > 0);
  void *dbias_expanded_ptr = nullptr;
  ck_tile::index_t stride_dbias = 0;
  ck_tile::index_t nhead_stride_dbias = 0;
  ck_tile::index_t batch_stride_dbias = 0;

  if (has_dbias) {
    // Allocate dbias_expanded [batch, seqlen_q, num_heads, seqlen_k] (match
    // Torch)
    size_t dbias_expanded_size = batch_size * seqlen_q * num_heads * seqlen_k *
                                 mha_utils::dtype_size(q.element_type());
    HIP_CHECK(hipMalloc(&dbias_expanded_ptr, dbias_expanded_size));
    HIP_CHECK(
        hipMemsetAsync(dbias_expanded_ptr, 0, dbias_expanded_size, stream));

    stride_dbias = num_heads * seqlen_k;
    nhead_stride_dbias = seqlen_k;
    batch_stride_dbias = seqlen_q * num_heads * seqlen_k;
  }
  void *dbias_ptr = dbias_expanded_ptr;

  uint64_t *seed_ptr = nullptr;
  uint64_t *offset_ptr = nullptr;
  uint64_t *dummy_rng = nullptr;

  if (dropout_p > 0.0f) {
    if (rng_state_.has_value() && mha_utils::is_valid_buffer(*rng_state_)) {
      try {
        auto [seed, offset] =
            mha_utils::get_rng_seed_offset_ptrs(rng_state_, dropout_p);
        seed_ptr = seed;
        offset_ptr = offset;
      } catch (const std::exception &e) {
        HIP_CHECK(hipMalloc(&dummy_rng, 2 * sizeof(uint64_t)));
        HIP_CHECK(hipMemsetAsync(dummy_rng, 0, 2 * sizeof(uint64_t), stream));
        seed_ptr = dummy_rng;
        offset_ptr = dummy_rng + 1;
      }
    } else {
      HIP_CHECK(hipMalloc(&dummy_rng, 2 * sizeof(uint64_t)));
      HIP_CHECK(hipMemsetAsync(dummy_rng, 0, 2 * sizeof(uint64_t), stream));
      seed_ptr = dummy_rng;
      offset_ptr = dummy_rng + 1;
    }
  } else {
    HIP_CHECK(hipMalloc(&dummy_rng, 2 * sizeof(uint64_t)));
    HIP_CHECK(hipMemsetAsync(dummy_rng, 0, 2 * sizeof(uint64_t), stream));
    seed_ptr = dummy_rng;
    offset_ptr = dummy_rng + 1;
  }

  std::vector<int64_t> dq_acc_shape;
  size_t dq_acc_bytes = compute_dq_acc_size_ck(
      batch_size, seqlen_q, seqlen_k, num_heads, head_size_v, deterministic,
      mask.type, dq_acc_shape);

  void *dq_acc_ptr = nullptr;
  HIP_CHECK(hipMalloc(&dq_acc_ptr, dq_acc_bytes));

  if (!deterministic || mask.type != mask_enum::no_mask) {
    HIP_CHECK(hipMemsetAsync(dq_acc_ptr, 0, dq_acc_bytes, stream));
  }

  ck_tile::index_t split_stride_dq_acc = 1;
  ck_tile::index_t batch_stride_dq_acc = 1;
  ck_tile::index_t stride_dq_acc = 1;
  ck_tile::index_t nhead_stride_dq_acc = 1;

  if (dq_acc_shape.size() == 5) {
    std::vector<ck_tile::index_t> strides(5);
    strides[4] = 1;
    for (int i = 3; i >= 0; i--) {
      strides[i] = strides[i + 1] * dq_acc_shape[i + 1];
    }

    split_stride_dq_acc = strides[0];
    batch_stride_dq_acc = strides[1];
    stride_dq_acc = strides[2];
    nhead_stride_dq_acc = strides[3];
  }

  auto dq_dims = dq_ret->dimensions();
  auto dk_dims = dk_ret->dimensions();
  auto dv_dims = dv_ret->dimensions();
  auto softmax_d_dims = softmax_d_ret->dimensions();

  void *dk_expanded_ptr = nullptr;
  void *dv_expanded_ptr = nullptr;
  void *dk_final = dk_ret->untyped_data();
  void *dv_final = dv_ret->untyped_data();

  if (is_mqa_gqa) {
    size_t dk_expanded_size = batch_size * seqlen_k * num_heads * head_size_q *
                              mha_utils::dtype_size(q.element_type());
    size_t dv_expanded_size = batch_size * seqlen_k * num_heads * head_size_v *
                              mha_utils::dtype_size(v.element_type());

    HIP_CHECK(hipMalloc(&dk_expanded_ptr, dk_expanded_size));
    HIP_CHECK(hipMalloc(&dv_expanded_ptr, dv_expanded_size));

    dk_final = dk_expanded_ptr;
    dv_final = dv_expanded_ptr;
  }

  float p_undrop = mha_utils::calculate_p_undrop(dropout_p);

  ck_tile::index_t stride_q = mha_utils::calculate_stride(q_dims, 1);
  ck_tile::index_t stride_k = mha_utils::calculate_stride(k_dims, 1);
  ck_tile::index_t stride_v = mha_utils::calculate_stride(v_dims, 1);
  ck_tile::index_t stride_o = mha_utils::calculate_stride(out_dims, 1);
  ck_tile::index_t stride_do = mha_utils::calculate_stride(dout_dims, 1);
  ck_tile::index_t stride_dq = mha_utils::calculate_stride(dq_dims, 1);

  ck_tile::index_t stride_dk, stride_dv;
  if (is_mqa_gqa) {
    stride_dk = num_heads * head_size_q;
    stride_dv = num_heads * head_size_v;
  } else {
    stride_dk = mha_utils::calculate_stride(dk_dims, 1);
    stride_dv = mha_utils::calculate_stride(dv_dims, 1);
  }

  ck_tile::index_t nhead_stride_q = mha_utils::calculate_stride(q_dims, 2);
  ck_tile::index_t nhead_stride_k = mha_utils::calculate_stride(k_dims, 2);
  ck_tile::index_t nhead_stride_v = mha_utils::calculate_stride(v_dims, 2);
  ck_tile::index_t nhead_stride_o = mha_utils::calculate_stride(out_dims, 2);
  ck_tile::index_t nhead_stride_do = mha_utils::calculate_stride(dout_dims, 2);
  ck_tile::index_t nhead_stride_lse = mha_utils::calculate_stride(lse_dims, 1);
  ck_tile::index_t nhead_stride_dq = mha_utils::calculate_stride(dq_dims, 2);
  ck_tile::index_t nhead_stride_dk =
      is_mqa_gqa ? head_size_q : mha_utils::calculate_stride(dk_dims, 2);
  ck_tile::index_t nhead_stride_dv =
      is_mqa_gqa ? head_size_v : mha_utils::calculate_stride(dv_dims, 2);

  ck_tile::index_t batch_stride_q = mha_utils::calculate_stride(q_dims, 0);
  ck_tile::index_t batch_stride_k = mha_utils::calculate_stride(k_dims, 0);
  ck_tile::index_t batch_stride_v = mha_utils::calculate_stride(v_dims, 0);
  ck_tile::index_t batch_stride_o = mha_utils::calculate_stride(out_dims, 0);
  ck_tile::index_t batch_stride_do = mha_utils::calculate_stride(dout_dims, 0);
  ck_tile::index_t batch_stride_lse = mha_utils::calculate_stride(lse_dims, 0);
  ck_tile::index_t batch_stride_dq = mha_utils::calculate_stride(dq_dims, 0);
  ck_tile::index_t batch_stride_dk =
      is_mqa_gqa ? seqlen_k * num_heads * head_size_q
                 : mha_utils::calculate_stride(dk_dims, 0);
  ck_tile::index_t batch_stride_dv =
      is_mqa_gqa ? seqlen_k * num_heads * head_size_v
                 : mha_utils::calculate_stride(dv_dims, 0);

  auto args = fmha_bwd_args{q.untyped_data(),
                            k.untyped_data(),
                            v.untyped_data(),
                            bias_ptr,
                            out.untyped_data(),
                            softmax_lse.untyped_data(),
                            dout.untyped_data(),
                            softmax_d_ret->untyped_data(),
                            nullptr, // rand_val_ptr
                            dq_ret->untyped_data(),
                            dk_final,
                            dv_final,
                            dbias_ptr,
                            dq_acc_ptr,
                            nullptr, // seqstart_q_ptr
                            nullptr, // seqstart_k_ptr
                            nullptr, // seqlen_k_ptr
                            static_cast<ck_tile::index_t>(seqlen_q),
                            static_cast<ck_tile::index_t>(seqlen_k),
                            static_cast<ck_tile::index_t>(batch_size),
                            static_cast<ck_tile::index_t>(seqlen_q),
                            static_cast<ck_tile::index_t>(seqlen_k),
                            static_cast<ck_tile::index_t>(head_size_q),
                            static_cast<ck_tile::index_t>(head_size_v),
                            static_cast<ck_tile::index_t>(num_heads),
                            static_cast<ck_tile::index_t>(num_heads_k),
                            softmax_scale,
                            stride_q,
                            stride_k,
                            stride_v,
                            stride_bias,
                            stride_o,
                            0, // stride_randval
                            stride_do,
                            stride_dq_acc,
                            stride_dq,
                            stride_dk,
                            stride_dv,
                            stride_dbias,
                            nhead_stride_q,
                            nhead_stride_k,
                            nhead_stride_v,
                            0, // nhead_stride_bias
                            nhead_stride_o,
                            0, // nhead_stride_randval
                            nhead_stride_do,
                            nhead_stride_lse,
                            nhead_stride_dq_acc,
                            nhead_stride_dq,
                            nhead_stride_dk,
                            nhead_stride_dv,
                            nhead_stride_dbias,
                            batch_stride_q,
                            batch_stride_k,
                            batch_stride_v,
                            0, // batch_stride_bias
                            batch_stride_o,
                            0, // batch_stride_randval
                            batch_stride_do,
                            batch_stride_lse,
                            batch_stride_dq_acc,
                            batch_stride_dq,
                            batch_stride_dk,
                            batch_stride_dv,
                            batch_stride_dbias,
                            split_stride_dq_acc,
                            mask.left,
                            mask.right,
                            static_cast<ck_tile::index_t>(mask.type),
                            dropout_p,
                            p_undrop,
                            std::make_pair(seed_ptr, offset_ptr)};

  auto stream_config = mha_utils::create_stream_config(stream);

  float runtime = aiter::mha_bwd(args, stream_config, dtype_str,
                                 false, // is_group_mode
                                 mask.type, bias_type, has_dbias,
                                 false, // is_store_randval
                                 deterministic,
                                 false, // use_ext_asm
                                 false, // is_v3_atomic_fp32
                                 0      // how_v3_bf16_cvt
  );

  if (runtime < 0) {
    if (dq_acc_ptr)
      hipFree(dq_acc_ptr);
    if (dk_expanded_ptr)
      hipFree(dk_expanded_ptr);
    if (dv_expanded_ptr)
      hipFree(dv_expanded_ptr);
    if (dummy_rng)
      hipFree(dummy_rng);
    return ffi::Error(ffi::ErrorCode::kInternal,
                      "aiter::mha_bwd failed - invalid arguments or "
                      "unsupported configuration");
  }

  if (is_mqa_gqa) {
    int64_t groups = num_heads / num_heads_k;

    mha_utils::launch_mqa_gqa_reduction(
        dk_expanded_ptr, dk_ret->untyped_data(), batch_size, seqlen_k,
        num_heads, num_heads_k, head_size_q, groups, q.element_type(), stream);

    mha_utils::launch_mqa_gqa_reduction(
        dv_expanded_ptr, dv_ret->untyped_data(), batch_size, seqlen_k,
        num_heads, num_heads_k, head_size_v, groups, v.element_type(), stream);

    HIP_CHECK(hipFree(dk_expanded_ptr));
    HIP_CHECK(hipFree(dv_expanded_ptr));
  }

  // Copy dbias_expanded to output; Python will sum over batch and heads.
  if (has_dbias && dbias_expanded_ptr) {
    size_t dbias_expanded_size = batch_size * seqlen_q * num_heads * seqlen_k *
                                 mha_utils::dtype_size(q.element_type());
    HIP_CHECK(hipMemcpyAsync(dbias_ret->untyped_data(), dbias_expanded_ptr,
                             dbias_expanded_size, hipMemcpyDeviceToDevice,
                             stream));
    HIP_CHECK(hipFree(dbias_expanded_ptr));
  }

  hipFree(dq_acc_ptr);
  if (dummy_rng != nullptr) {
    hipFree(dummy_rng);
  }

  return ffi::Error::Success();
}

} // namespace jax_aiter

#pragma GCC visibility push(default)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MhaBwdJA, jax_aiter::MhaBwd_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>() // dout
        .Arg<ffi::AnyBuffer>() // q
        .Arg<ffi::AnyBuffer>() // k
        .Arg<ffi::AnyBuffer>() // v
        .Arg<ffi::AnyBuffer>() // out
        .Arg<ffi::AnyBuffer>() // softmax_lse
        .Arg<ffi::AnyBuffer>() // dq_ (optional)
        .Arg<ffi::AnyBuffer>() // dk_ (optional)
        .Arg<ffi::AnyBuffer>() // dv_ (optional)
        .Arg<ffi::AnyBuffer>() // bias_ (optional)
        .Arg<ffi::AnyBuffer>() // alibi_slopes_ (optional)
        .Arg<ffi::AnyBuffer>() // rng_state_ (optional)
        .Arg<ffi::AnyBuffer>() // gen_ (optional)
        .Ret<ffi::AnyBuffer>() // dq_ret
        .Ret<ffi::AnyBuffer>() // dk_ret
        .Ret<ffi::AnyBuffer>() // dv_ret
        .Ret<ffi::AnyBuffer>() // softmax_d_ret
        .Ret<ffi::AnyBuffer>() // dbias_ret
        .Attr<float>("dropout_p")
        .Attr<float>("softmax_scale")
        .Attr<bool>("is_causal")
        .Attr<int>("window_size_left")
        .Attr<int>("window_size_right")
        .Attr<bool>("deterministic"),
    {xla::ffi::Traits::kCmdBufferCompatible});

#pragma GCC visibility pop
