// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <chrono>
#include <cstring>
#include <hip/hip_runtime.h>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "hip_utils.h"
#include "mha_common_utils.h"
#include "mha_fwd.h"

namespace ffi = xla::ffi;

namespace jax_aiter {

ffi::Error
MhaFwd_Bridge(hipStream_t stream,
              ffi::AnyBuffer q, // [b, sq, hq, d]
              ffi::AnyBuffer k, // [b, sk, hk, d]
              ffi::AnyBuffer v, // [b, sk, hk, d_v]
              std::optional<ffi::AnyBuffer>
                  cu_seqlens_q_, // [b+1] cumulative sequence lengths
              std::optional<ffi::AnyBuffer>
                  cu_seqlens_kv_, // [b+1] cumulative sequence lengths
              std::optional<ffi::AnyBuffer> out_,          // [b, sq, hq, d_v]
              std::optional<ffi::AnyBuffer> bias_,         // [sq, sk]
              std::optional<ffi::AnyBuffer> alibi_slopes_, // [hq] or [b, hq]
              std::optional<ffi::AnyBuffer> gen_,    // [2] seed and offset
              ffi::Result<ffi::AnyBuffer> o,         // [b, sq, hq, d_v]
              ffi::Result<ffi::AnyBuffer> lse,       // [b, hq, sq]
              ffi::Result<ffi::AnyBuffer> p,         // [b, hq, sq, sk]
              ffi::Result<ffi::AnyBuffer> rng_state, // [2]
              float dropout_p, float softmax_scale, bool is_causal,
              int window_size_left, int window_size_right,
              bool return_softmax_lse, bool return_dropout_randval) {

  if (!q.untyped_data() || !k.untyped_data() || !v.untyped_data()) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "Required input buffer (q/k/v) is null");
  }

  const int dev_idx = ::jax_aiter::device_from_ptr(q.untyped_data());
  if (dev_idx < 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "bad device from q");
  }

  auto q_dims = q.dimensions();
  auto k_dims = k.dimensions();
  auto v_dims = v.dimensions();

  mha_utils::validate_mha_dimensions(q_dims, k_dims, v_dims);

  const int64_t batch_size = q_dims[0];
  const int64_t seqlen_q = q_dims[1];
  const int64_t num_heads = q_dims[2];
  const int64_t head_size_q = q_dims[3];
  const int64_t seqlen_k = k_dims[1];
  const int64_t num_heads_k = k_dims[2];
  const int64_t head_size_v = v_dims[3];

  // Validate dtypes match Torch requirements.
  auto q_dtype = q.element_type();
  if (q_dtype != xla::ffi::DataType::F16 &&
      q_dtype != xla::ffi::DataType::BF16) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "FlashAttention only supports fp16 and bf16 data type");
  }
  if (k.element_type() != q_dtype) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "query and key must have the same dtype");
  }
  if (v.element_type() != q_dtype) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "query and value must have the same dtype");
  }

  if (batch_size <= 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "batch size must be positive");
  }
  if (head_size_q > 256 || head_size_v > 256) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "CK only supports head dimension at most 256");
  }
  if (head_size_q % 8 != 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "query head_size must be a multiple of 8");
  }
  if (head_size_v % 8 != 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "value head_size must be a multiple of 8");
  }
  if (num_heads % num_heads_k != 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "Number of heads in query must be divisible by number of "
                      "heads in key/value");
  }

  // Dropout requires valid randval buffer.
  if (return_dropout_randval && dropout_p <= 0.0f) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "return_dropout_randval requires dropout_p > 0");
  }

  std::string dtype_str = mha_utils::dtype_to_string(q.element_type());

  const void *bias_ptr = nullptr;
  ck_tile::index_t stride_bias = 0;
  bool has_bias = bias_.has_value() && mha_utils::is_valid_buffer(*bias_);
  bool has_alibi =
      alibi_slopes_.has_value() && mha_utils::is_valid_buffer(*alibi_slopes_);

  if (has_bias && has_alibi) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "cannot apply both bias and alibi at the same time");
  }

  if (has_bias) {
    bias_ptr = bias_->untyped_data();
    auto bias_dims = bias_->dimensions();
    if (bias_dims.size() != 2 || bias_dims[0] != seqlen_q ||
        bias_dims[1] != seqlen_k) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                        "bias shape should be [seqlen_q, seqlen_k]");
    }
    stride_bias = mha_utils::calculate_stride(bias_dims, 0);
  } else if (has_alibi) {
    bias_ptr = alibi_slopes_->untyped_data();
    auto alibi_dims = alibi_slopes_->dimensions();
    bool valid_1d = (alibi_dims.size() == 1 && alibi_dims[0] == num_heads);
    bool valid_2d = (alibi_dims.size() == 2 && alibi_dims[0] == batch_size &&
                     alibi_dims[1] == num_heads);
    if (!valid_1d && !valid_2d) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                        "alibi_slopes shape should be [num_heads] or "
                        "[batch_size, num_heads]");
    }
    stride_bias =
        alibi_dims.size() == 2 ? mha_utils::calculate_stride(alibi_dims, 0) : 0;
  }

  bias_enum bias_type = mha_utils::get_bias_type(has_bias, has_alibi);

  // Normalize window sizes and causal flag per Torch semantics.
  int w_left = window_size_left;
  int w_right = window_size_right;
  if (w_left >= seqlen_k) {
    w_left = -1;
  }
  if (w_right >= seqlen_k) {
    w_right = -1;
  }

  bool effective_is_causal = is_causal;
  if (seqlen_q == 1 && !has_alibi) {
    effective_is_causal = false;
  }

  mask_info mask = mha_utils::create_mask_info(effective_is_causal, w_left,
                                               w_right, seqlen_q, seqlen_k);

  // Empty key sequences require zeroed output and +inf LSE.
  if (seqlen_k == 0) {
    HIP_CHECK(hipMemsetAsync(o->untyped_data(), 0, o->size_bytes(), stream));
    if (return_softmax_lse) {
      std::vector<float> inf_buffer(lse->element_count(),
                                    std::numeric_limits<float>::infinity());
      HIP_CHECK(hipMemcpyAsync(lse->untyped_data(), inf_buffer.data(),
                               inf_buffer.size() * sizeof(float),
                               hipMemcpyHostToDevice, stream));
    }
    return ffi::Error::Success();
  }

  mha_utils::RngStatePointers rng_ptrs;
  auto rng_err = mha_utils::prepare_rng_state_for_fwd(
      stream, dropout_p, dev_idx, batch_size, num_heads, gen_, rng_state,
      rng_ptrs);
  if (!rng_err.success()) {
    return rng_err;
  }

  const ck_tile::index_t *cu_seqlen_q_ptr = nullptr;
  const ck_tile::index_t *cu_seqlen_kv_ptr = nullptr;

  if (cu_seqlens_q_.has_value() && mha_utils::is_valid_buffer(*cu_seqlens_q_)) {
    cu_seqlen_q_ptr = reinterpret_cast<const ck_tile::index_t *>(
        cu_seqlens_q_->untyped_data());
  }
  if (cu_seqlens_kv_.has_value() &&
      mha_utils::is_valid_buffer(*cu_seqlens_kv_)) {
    cu_seqlen_kv_ptr = reinterpret_cast<const ck_tile::index_t *>(
        cu_seqlens_kv_->untyped_data());
  }

  auto o_dims = o->dimensions();
  auto lse_dims = lse->dimensions();
  auto p_dims = p->dimensions();

  ck_tile::index_t stride_q = mha_utils::calculate_stride(q_dims, 1);
  ck_tile::index_t stride_k = mha_utils::calculate_stride(k_dims, 1);
  ck_tile::index_t stride_v = mha_utils::calculate_stride(v_dims, 1);
  ck_tile::index_t stride_o = mha_utils::calculate_stride(o_dims, 1);
  ck_tile::index_t stride_randval =
      return_dropout_randval ? mha_utils::calculate_stride(p_dims, 2) : 0;

  ck_tile::index_t nhead_stride_q = mha_utils::calculate_stride(q_dims, 2);
  ck_tile::index_t nhead_stride_k = mha_utils::calculate_stride(k_dims, 2);
  ck_tile::index_t nhead_stride_v = mha_utils::calculate_stride(v_dims, 2);
  ck_tile::index_t nhead_stride_o = mha_utils::calculate_stride(o_dims, 2);
  ck_tile::index_t nhead_stride_lse =
      return_softmax_lse ? mha_utils::calculate_stride(lse_dims, 1) : 0;
  ck_tile::index_t nhead_stride_randval =
      return_dropout_randval ? mha_utils::calculate_stride(p_dims, 1) : 0;

  ck_tile::index_t batch_stride_q = mha_utils::calculate_stride(q_dims, 0);
  ck_tile::index_t batch_stride_k = mha_utils::calculate_stride(k_dims, 0);
  ck_tile::index_t batch_stride_v = mha_utils::calculate_stride(v_dims, 0);
  ck_tile::index_t batch_stride_o = mha_utils::calculate_stride(o_dims, 0);
  ck_tile::index_t batch_stride_lse =
      return_softmax_lse ? mha_utils::calculate_stride(lse_dims, 0) : 0;
  ck_tile::index_t batch_stride_randval =
      return_dropout_randval ? mha_utils::calculate_stride(p_dims, 0) : 0;

  auto args =
      fmha_fwd_args{q.untyped_data(),
                    k.untyped_data(),
                    v.untyped_data(),
                    bias_ptr,
                    return_dropout_randval ? p->untyped_data() : nullptr,
                    return_softmax_lse ? lse->untyped_data() : nullptr,
                    o->untyped_data(),
                    cu_seqlen_q_ptr,
                    cu_seqlen_kv_ptr,
                    nullptr, // seqstart_q
                    nullptr, // seqstart_k
                    nullptr, // seqlen_k_ptr
                    nullptr, // seqstart_padded_q_ptr
                    nullptr, // seqstart_padded_k_ptr
                    static_cast<ck_tile::index_t>(seqlen_q),
                    static_cast<ck_tile::index_t>(seqlen_k),
                    static_cast<ck_tile::index_t>(batch_size),
                    static_cast<ck_tile::index_t>(seqlen_q), // max_seqlen_q
                    static_cast<ck_tile::index_t>(head_size_q),
                    static_cast<ck_tile::index_t>(head_size_v),
                    static_cast<ck_tile::index_t>(num_heads),
                    static_cast<ck_tile::index_t>(num_heads_k),
                    softmax_scale,
                    1.0f, // scale_p
                    1.0f, // scale_o
                    0.0f, // logits_soft_cap
                    stride_q,
                    stride_k,
                    stride_v,
                    stride_bias,
                    stride_randval,
                    stride_o,
                    nhead_stride_q,
                    nhead_stride_k,
                    nhead_stride_v,
                    0, // nhead_stride_bias
                    nhead_stride_randval,
                    nhead_stride_lse,
                    nhead_stride_o,
                    batch_stride_q,
                    batch_stride_k,
                    batch_stride_v,
                    0, // batch_stride_bias
                    batch_stride_randval,
                    batch_stride_lse,
                    batch_stride_o,
                    mask.left,
                    mask.right,
                    static_cast<ck_tile::index_t>(mask.type),
                    0, // min_seqlen_q
                    dropout_p,
                    return_dropout_randval,
                    std::make_pair(rng_ptrs.seed, rng_ptrs.offset)};

  auto stream_config = mha_utils::create_stream_config(stream);

  float elapsed_time = aiter::mha_fwd(args, stream_config, dtype_str,
                                      false, // is_group_mode
                                      mask.type, bias_type, return_softmax_lse,
                                      false // use_ext_asm
  );

  if (elapsed_time < 0) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      "aiter::mha_fwd failed - invalid arguments or "
                      "unsupported configuration");
  }

  return ffi::Error::Success();
}

} // namespace jax_aiter

#pragma GCC visibility push(default)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MhaFwdJA, jax_aiter::MhaFwd_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>() // q: [batch, seqlen_q, nhead, hdim_q]
        .Arg<ffi::AnyBuffer>() // k: [batch, seqlen_kv, nhead, hdim_q]
        .Arg<ffi::AnyBuffer>() // v: [batch, seqlen_kv, nhead, hdim_v]
        .Arg<ffi::AnyBuffer>() // cu_seqlens_q (optional, for varlen)
        .Arg<ffi::AnyBuffer>() // cu_seqlens_kv (optional, for varlen)
        .Arg<ffi::AnyBuffer>() // out_provided (optional)
        .Arg<ffi::AnyBuffer>() // bias (optional)
        .Arg<ffi::AnyBuffer>() // alibi_slopes (optional)
        .Arg<ffi::AnyBuffer>() // gen (optional)
        .Ret<ffi::AnyBuffer>() // o: [batch, seqlen_q, nhead, hdim_v]
        .Ret<ffi::AnyBuffer>() // lse: [batch, nhead, seqlen_q]
        .Ret<ffi::AnyBuffer>() // p: dropout mask
        .Ret<ffi::AnyBuffer>() // rng_state
        .Attr<float>("dropout_p")
        .Attr<float>("softmax_scale")
        .Attr<bool>("is_causal")
        .Attr<int>("window_size_left")
        .Attr<int>("window_size_right")
        .Attr<bool>("return_softmax_lse")
        .Attr<bool>("return_dropout_randval"),
    {xla::ffi::Traits::kCmdBufferCompatible});

#pragma GCC visibility pop
