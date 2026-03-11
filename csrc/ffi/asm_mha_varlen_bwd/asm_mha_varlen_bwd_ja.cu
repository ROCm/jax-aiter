// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>
#include <vector>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "hip_utils.h"
#include "logging.h"
#include "mha_bwd.h"
#include "mha_common_utils.h"

namespace ffi = xla::ffi;

namespace jax_aiter {

// Compute dq_acc size for v3 varlen layout (ASM-specific).
static size_t compute_dq_acc_size_v3_varlen(
    int64_t batch_size, int64_t total_q, int64_t max_seqlen_q,
    int64_t max_seqlen_k, int64_t num_heads, int64_t head_size,
    bool deterministic, bool is_v3_atomic_fp32, xla::ffi::DataType q_dtype,
    std::vector<int64_t> &out_shape) {

  size_t element_size = 4; // float32 by default.

  if (!deterministic) {
    if (is_v3_atomic_fp32) {
      // dq_acc shape is {1, num_heads, total_q, head_size} when using atomic
      // fp32.
      out_shape = {1, num_heads, total_q, head_size};
      element_size = 4; // float32.
    } else {
      // dq_acc shape is {1, batch, num_heads, round_up(max_seqlen_q, 16), 128}
      // for atomic16 kernels.
      int64_t seqlen_q_padded = ((max_seqlen_q + 15) / 16) * 16;
      out_shape = {1, batch_size, num_heads, seqlen_q_padded, 128};
      // Use same dtype as q (fp16/bf16).
      element_size = (q_dtype == xla::ffi::DataType::F16 ||
                      q_dtype == xla::ffi::DataType::BF16)
                         ? 2
                         : 4;
    }
  } else {
    // Deterministic path uses [nsplits, num_heads, total_q, head_size].
    const ck_tile::index_t kN0 = head_size <= 128 ? 128 : 64;
    const ck_tile::index_t nsplits =
        ck_tile::integer_divide_ceil(max_seqlen_k, kN0);
    out_shape = {nsplits, num_heads, total_q, head_size};
    element_size = 4; // float32.
  }

  size_t total_elements = 1;
  for (auto dim : out_shape) {
    total_elements *= dim;
  }

  return total_elements * element_size;
}

ffi::Error FmhaV3VarlenBwd_Bridge(
    hipStream_t stream, ffi::AnyBuffer dout, ffi::AnyBuffer q, ffi::AnyBuffer k,
    ffi::AnyBuffer v, ffi::AnyBuffer out, ffi::AnyBuffer softmax_lse,
    ffi::AnyBuffer cu_seqlens_q, ffi::AnyBuffer cu_seqlens_k,
    std::optional<ffi::AnyBuffer> dq_, std::optional<ffi::AnyBuffer> dk_,
    std::optional<ffi::AnyBuffer> dv_,
    std::optional<ffi::AnyBuffer> alibi_slopes_,
    std::optional<ffi::AnyBuffer> rng_state_,
    std::optional<ffi::AnyBuffer> gen_, ffi::Result<ffi::AnyBuffer> dq,
    ffi::Result<ffi::AnyBuffer> dk, ffi::Result<ffi::AnyBuffer> dv,
    ffi::Result<ffi::AnyBuffer> softmax_d, int max_seqlen_q, int max_seqlen_k,
    float p_dropout, float softmax_scale, bool zero_tensors, bool is_causal,
    int window_size_left, int window_size_right, bool deterministic,
    bool is_v3_atomic_fp32, int how_v3_bf16_cvt) {

  if (!q.untyped_data() || !k.untyped_data() || !v.untyped_data() ||
      !out.untyped_data() || !softmax_lse.untyped_data() ||
      !dout.untyped_data() || !cu_seqlens_q.untyped_data() ||
      !cu_seqlens_k.untyped_data()) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "Required input buffer is null");
  }

  const int dev_idx = ::jax_aiter::device_from_ptr(q.untyped_data());
  if (dev_idx < 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "bad device from q");
  }

  try {
    auto q_dims = q.dimensions();
    auto k_dims = k.dimensions();
    auto v_dims = v.dimensions();
    auto cu_q_dims = cu_seqlens_q.dimensions();

    // Varlen layout is [total, nheads, d].
    int64_t total_q = q_dims[0];
    int64_t num_heads = q_dims[1];
    int64_t head_size_q = q_dims[2];

    int64_t total_k = k_dims[0];
    int64_t num_heads_k = k_dims[1];
    int64_t head_size_v = v_dims[2];

    int64_t batch_size = cu_q_dims[0] - 1;

    // Validate input shapes and dtypes.
    auto q_dtype = q.element_type();
    if (q_dtype != xla::ffi::DataType::F16 &&
        q_dtype != xla::ffi::DataType::BF16) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                        "FlashAttention only supports fp16/bf16");
    }

    if (k.element_type() != q_dtype || v.element_type() != q_dtype ||
        out.element_type() != q_dtype || dout.element_type() != q_dtype) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                        "Q, K, V, OUT, DOUT must have same dtype");
    }

    if (softmax_lse.element_type() != xla::ffi::DataType::F32) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                        "softmax_lse must be float32");
    }

    if (head_size_q % 8 != 0 || head_size_v % 8 != 0) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                        "head dimensions must be multiples of 8");
    }

    if (head_size_q > 256 || head_size_v > 256) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                        "CK FlashAttention backward only supports head "
                        "dimension at most 256");
    }

    if (num_heads % num_heads_k != 0) {
      return ffi::Error(
          ffi::ErrorCode::kInvalidArgument,
          "Number of heads in query must divide number of heads in key/value");
    }

    if (num_heads != num_heads_k) {
      return ffi::Error(ffi::ErrorCode::kUnimplemented,
                        "MQA/GQA not yet supported in v3 backward path");
    }

    std::string dtype_str = mha_utils::dtype_to_string(q.element_type());

    // Normalize attention window sizes.
    if (is_causal) {
      window_size_right = 0;
    }
    if (window_size_left >= max_seqlen_k) {
      window_size_left = -1;
    }
    if (window_size_right >= max_seqlen_k) {
      window_size_right = -1;
    }

    auto mask = mha_utils::create_mask_info(is_causal, window_size_left,
                                            window_size_right, max_seqlen_q,
                                            max_seqlen_k);

    // Handle ALiBi bias.
    const void *alibi_ptr = nullptr;
    ck_tile::index_t stride_alibi = 0;

    if (alibi_slopes_.has_value() &&
        mha_utils::is_valid_buffer(*alibi_slopes_)) {
      alibi_ptr = alibi_slopes_->untyped_data();
      auto alibi_dims = alibi_slopes_->dimensions();
      stride_alibi = alibi_dims.size() >= 2
                         ? mha_utils::calculate_stride(alibi_dims, 0)
                         : 0;
    }

    bias_enum bias_type =
        (alibi_ptr != nullptr) ? bias_enum::alibi : bias_enum::no_bias;

    // Initialize RNG state for dropout.
    uint64_t *seed_ptr = nullptr;
    uint64_t *offset_ptr = nullptr;
    uint64_t *dummy_rng = nullptr;

    if (p_dropout > 0.0f) {
      if (rng_state_.has_value() && mha_utils::is_valid_buffer(*rng_state_)) {
        try {
          auto [seed, offset] =
              mha_utils::get_rng_seed_offset_ptrs(rng_state_, p_dropout);
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

    // Allocate dq_acc buffer with v3-specific layout.
    std::vector<int64_t> dq_acc_shape;
    size_t dq_acc_bytes = compute_dq_acc_size_v3_varlen(
        batch_size, total_q, max_seqlen_q, max_seqlen_k, num_heads, head_size_q,
        deterministic, is_v3_atomic_fp32, q.element_type(), dq_acc_shape);

    void *dq_acc_ptr = nullptr;
    HIP_CHECK(hipMalloc(&dq_acc_ptr, dq_acc_bytes));
    HIP_CHECK(hipMemsetAsync(dq_acc_ptr, 0, dq_acc_bytes, stream));

    // Compute dq_acc strides from dq_acc_shape.
    ck_tile::index_t split_stride_dq_acc = 1;
    ck_tile::index_t batch_stride_dq_acc = 0;
    ck_tile::index_t nhead_stride_dq_acc = 1;
    ck_tile::index_t stride_dq_acc = 1;

    if (is_v3_atomic_fp32) {
      // Layout [1, nheads, total_q, head_dim].
      if (dq_acc_shape.size() == 4) {
        std::vector<ck_tile::index_t> strides(4);
        strides[3] = 1;
        for (int i = 2; i >= 0; i--) {
          strides[i] = strides[i + 1] * dq_acc_shape[i + 1];
        }
        split_stride_dq_acc = strides[0];
        nhead_stride_dq_acc = strides[1];
        stride_dq_acc = strides[2];
        batch_stride_dq_acc = 0;
      }
    } else {
      // [1, batch, nheads, padded_seqlen, 128] for atomic16
      // or [nsplits, nheads, total_q, hdim] for deterministic
      if (dq_acc_shape.size() == 5) {
        // atomic16 layout case.
        std::vector<ck_tile::index_t> strides(5);
        strides[4] = 1;
        for (int i = 3; i >= 0; i--) {
          strides[i] = strides[i + 1] * dq_acc_shape[i + 1];
        }
        split_stride_dq_acc = strides[0];
        batch_stride_dq_acc = strides[1];
        nhead_stride_dq_acc = strides[2];
        stride_dq_acc = strides[3];
      } else if (dq_acc_shape.size() == 4) {
        // Deterministic layout case [nsplits, nheads, total_q, head_dim].
        std::vector<ck_tile::index_t> strides(4);
        strides[3] = 1;
        for (int i = 2; i >= 0; i--) {
          strides[i] = strides[i + 1] * dq_acc_shape[i + 1];
        }
        split_stride_dq_acc = strides[0];
        nhead_stride_dq_acc = strides[1];
        stride_dq_acc = strides[2];
        batch_stride_dq_acc = 0;
      }
    }

    // Handle MQA/GQA head expansion.
    bool is_mqa_gqa = (num_heads != num_heads_k);
    void *dk_expanded_ptr = nullptr;
    void *dv_expanded_ptr = nullptr;
    void *dk_final = dk->untyped_data();
    void *dv_final = dv->untyped_data();

    if (is_mqa_gqa) {
      size_t dk_expanded_size = total_k * num_heads * head_size_q *
                                mha_utils::dtype_size(q.element_type());
      size_t dv_expanded_size = total_k * num_heads * head_size_v *
                                mha_utils::dtype_size(v.element_type());

      HIP_CHECK(hipMalloc(&dk_expanded_ptr, dk_expanded_size));
      HIP_CHECK(hipMalloc(&dv_expanded_ptr, dv_expanded_size));

      dk_final = dk_expanded_ptr;
      dv_final = dv_expanded_ptr;
    }

    // Zero-initialize outputs when zero_tensors is set.
    if (zero_tensors) {
      HIP_CHECK(
          hipMemsetAsync(dq->untyped_data(), 0, dq->size_bytes(), stream));
      HIP_CHECK(hipMemsetAsync(dk_final, 0,
                               is_mqa_gqa
                                   ? (total_k * num_heads * head_size_q *
                                      mha_utils::dtype_size(q.element_type()))
                                   : dk->size_bytes(),
                               stream));
      HIP_CHECK(hipMemsetAsync(dv_final, 0,
                               is_mqa_gqa
                                   ? (total_k * num_heads * head_size_v *
                                      mha_utils::dtype_size(v.element_type()))
                                   : dv->size_bytes(),
                               stream));
      HIP_CHECK(hipMemsetAsync(softmax_d->untyped_data(), 0,
                               softmax_d->size_bytes(), stream));
    }

    if (max_seqlen_q == 0) {
      if (dk_expanded_ptr)
        hipFree(dk_expanded_ptr);
      if (dv_expanded_ptr)
        hipFree(dv_expanded_ptr);
      if (dummy_rng)
        hipFree(dummy_rng);
      if (dq_acc_ptr)
        hipFree(dq_acc_ptr);
      return ffi::Error::Success();
    }

    // Compute strides for varlen layout [total, nheads, d].
    ck_tile::index_t stride_q = mha_utils::calculate_stride(q_dims, 0);
    ck_tile::index_t stride_k = mha_utils::calculate_stride(k_dims, 0);
    ck_tile::index_t stride_v = mha_utils::calculate_stride(v_dims, 0);

    auto out_dims = out.dimensions();
    auto dout_dims = dout.dimensions();
    ck_tile::index_t stride_o = mha_utils::calculate_stride(out_dims, 0);
    ck_tile::index_t stride_do = mha_utils::calculate_stride(dout_dims, 0);

    auto dq_dims = dq->dimensions();
    ck_tile::index_t stride_dq = mha_utils::calculate_stride(dq_dims, 0);

    ck_tile::index_t stride_dk =
        is_mqa_gqa ? num_heads * head_size_q
                   : mha_utils::calculate_stride(dk->dimensions(), 0);
    ck_tile::index_t stride_dv =
        is_mqa_gqa ? num_heads * head_size_v
                   : mha_utils::calculate_stride(dv->dimensions(), 0);

    ck_tile::index_t nhead_stride_q = mha_utils::calculate_stride(q_dims, 1);
    ck_tile::index_t nhead_stride_k = mha_utils::calculate_stride(k_dims, 1);
    ck_tile::index_t nhead_stride_v = mha_utils::calculate_stride(v_dims, 1);
    ck_tile::index_t nhead_stride_o = mha_utils::calculate_stride(out_dims, 1);
    ck_tile::index_t nhead_stride_do =
        mha_utils::calculate_stride(dout_dims, 1);

    // softmax_lse uses 2D layout [nheads, total_q] for varlen.
    auto lse_dims = softmax_lse.dimensions();
    ck_tile::index_t nhead_stride_lse =
        mha_utils::calculate_stride(lse_dims, 0);

    ck_tile::index_t nhead_stride_dq = mha_utils::calculate_stride(dq_dims, 1);
    ck_tile::index_t nhead_stride_dk =
        is_mqa_gqa ? head_size_q
                   : mha_utils::calculate_stride(dk->dimensions(), 1);
    ck_tile::index_t nhead_stride_dv =
        is_mqa_gqa ? head_size_v
                   : mha_utils::calculate_stride(dv->dimensions(), 1);

    // Get cu_seqlens pointers.
    const ck_tile::index_t *cu_seqlens_q_ptr =
        reinterpret_cast<const ck_tile::index_t *>(cu_seqlens_q.untyped_data());
    const ck_tile::index_t *cu_seqlens_k_ptr =
        reinterpret_cast<const ck_tile::index_t *>(cu_seqlens_k.untyped_data());

    float p_undrop = mha_utils::calculate_p_undrop(p_dropout);

    auto args = aiter::mha_bwd_args{
        .use_asm_v3 = true,
        .v3_atomic_fp32 = is_v3_atomic_fp32,
        .v3_bf16_cvt = how_v3_bf16_cvt,
        .v3_api_check = false,
        .hdim_q = static_cast<int>(head_size_q),
        .hdim_v = static_cast<int>(head_size_v),
        .data_type = dtype_str,
        .is_group_mode = true,
        .mask_type = static_cast<int>(mask.type),
        .bias_type = static_cast<int>(bias_type),
        .has_dbias = false,
        .has_dropout = (p_dropout > 0.0f),
        .is_store_randval = false,
        .is_deterministic = deterministic,

        .q_ptr = q.untyped_data(),
        .k_ptr = k.untyped_data(),
        .v_ptr = v.untyped_data(),
        .bias_ptr = alibi_ptr,
        .o_ptr = out.untyped_data(),
        .lse_ptr = softmax_lse.untyped_data(),
        .do_ptr = dout.untyped_data(),
        .d_ptr = softmax_d->untyped_data(),
        .rand_val_ptr = nullptr,
        .dq_ptr = dq->untyped_data(),
        .dk_ptr = dk_final,
        .dv_ptr = dv_final,
        .dbias_ptr = nullptr,
        .dq_acc_ptr = dq_acc_ptr,

        .seqstart_q_ptr = cu_seqlens_q_ptr,
        .seqstart_k_ptr = cu_seqlens_k_ptr,

        .seqlen_q = static_cast<int>(total_q),
        .seqlen_k = static_cast<int>(total_k),
        .batch = static_cast<int>(batch_size),
        .max_seqlen_q = static_cast<int>(max_seqlen_q),
        .max_seqlen_k = static_cast<int>(max_seqlen_k),
        .nhead_q = static_cast<int>(num_heads),
        .nhead_k = static_cast<int>(num_heads_k),
        .scale = softmax_scale,
        .stride_q = static_cast<int>(stride_q),
        .stride_k = static_cast<int>(stride_k),
        .stride_v = static_cast<int>(stride_v),
        .stride_bias = static_cast<int>(stride_alibi),
        .stride_o = static_cast<int>(stride_o),
        .stride_randval = 0,
        .stride_do = static_cast<int>(stride_do),
        .stride_dq_acc = static_cast<int>(stride_dq_acc),
        .stride_dq = static_cast<int>(stride_dq),
        .stride_dk = static_cast<int>(stride_dk),
        .stride_dv = static_cast<int>(stride_dv),
        .stride_dbias = 0,
        .nhead_stride_q = static_cast<int>(nhead_stride_q),
        .nhead_stride_k = static_cast<int>(nhead_stride_k),
        .nhead_stride_v = static_cast<int>(nhead_stride_v),
        .nhead_stride_bias = 0,
        .nhead_stride_o = static_cast<int>(nhead_stride_o),
        .nhead_stride_randval = 0,
        .nhead_stride_do = static_cast<int>(nhead_stride_do),
        .nhead_stride_lsed = static_cast<int>(nhead_stride_lse),
        .nhead_stride_dq_acc = static_cast<int64_t>(nhead_stride_dq_acc),
        .nhead_stride_dq = static_cast<int>(nhead_stride_dq),
        .nhead_stride_dk = static_cast<int>(nhead_stride_dk),
        .nhead_stride_dv = static_cast<int>(nhead_stride_dv),
        .nhead_stride_dbias = 0,
        .batch_stride_q = 0,
        .batch_stride_k = 0,
        .batch_stride_v = 0,
        .batch_stride_bias = 0,
        .batch_stride_o = 0,
        .batch_stride_randval = 0,
        .batch_stride_do = 0,
        .batch_stride_lsed = 0,
        .batch_stride_dq_acc = static_cast<int64_t>(batch_stride_dq_acc),
        .batch_stride_dq = 0,
        .batch_stride_dk = 0,
        .batch_stride_dv = 0,
        .batch_stride_dbias = 0,
        .split_stride_dq_acc = static_cast<int>(split_stride_dq_acc),
        .window_size_left = static_cast<int>(mask.left),
        .window_size_right = static_cast<int>(mask.right),
        .p_drop = p_dropout,
        .p_undrop = p_undrop,
        .drop_seed_offset = std::make_pair(seed_ptr, offset_ptr)
    };

    auto stream_config = mha_utils::create_stream_config(stream);

    float runtime = aiter::mha_bwd(args, stream_config);

    if (runtime < 0) {
      if (dq_acc_ptr)
        hipFree(dq_acc_ptr);
      if (dk_expanded_ptr)
        hipFree(dk_expanded_ptr);
      if (dv_expanded_ptr)
        hipFree(dv_expanded_ptr);
      if (dummy_rng)
        hipFree(dummy_rng);
      return ffi::Error(ffi::ErrorCode::kInternal, "aiter::mha_bwd failed");
    }

    // Reduce expanded MQA/GQA gradients back to grouped layout.
    if (is_mqa_gqa) {
      int64_t groups = num_heads / num_heads_k;

      mha_utils::launch_mqa_gqa_reduction(
          dk_expanded_ptr, dk->untyped_data(), total_k, 1, num_heads,
          num_heads_k, head_size_q, groups, q.element_type(), stream);

      mha_utils::launch_mqa_gqa_reduction(
          dv_expanded_ptr, dv->untyped_data(), total_k, 1, num_heads,
          num_heads_k, head_size_v, groups, v.element_type(), stream);

      HIP_CHECK(hipFree(dk_expanded_ptr));
      HIP_CHECK(hipFree(dv_expanded_ptr));
    }

    if (dq_acc_ptr)
      hipFree(dq_acc_ptr);
    if (dummy_rng)
      hipFree(dummy_rng);

    JA_LOG("FmhaV3VarlenBwd completed successfully, elapsed time: %.3f ms",
           runtime);
    return ffi::Error::Success();

  } catch (const std::exception &e) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      std::string("fmha_v3_varlen_bwd: ") + e.what());
  }
}

} // namespace jax_aiter

#pragma GCC visibility push(default)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FmhaV3VarlenBwdJA, jax_aiter::FmhaV3VarlenBwd_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>() // dout
        .Arg<ffi::AnyBuffer>() // q
        .Arg<ffi::AnyBuffer>() // k
        .Arg<ffi::AnyBuffer>() // v
        .Arg<ffi::AnyBuffer>() // out
        .Arg<ffi::AnyBuffer>() // softmax_lse
        .Arg<ffi::AnyBuffer>() // cu_seqlens_q
        .Arg<ffi::AnyBuffer>() // cu_seqlens_k
        .Arg<ffi::AnyBuffer>() // dq_provided (optional)
        .Arg<ffi::AnyBuffer>() // dk_provided (optional)
        .Arg<ffi::AnyBuffer>() // dv_provided (optional)
        .Arg<ffi::AnyBuffer>() // alibi_slopes (optional)
        .Arg<ffi::AnyBuffer>() // rng_state (optional)
        .Arg<ffi::AnyBuffer>() // gen (optional)
        .Ret<ffi::AnyBuffer>() // dq
        .Ret<ffi::AnyBuffer>() // dk
        .Ret<ffi::AnyBuffer>() // dv
        .Ret<ffi::AnyBuffer>() // softmax_d
        .Attr<int>("max_seqlen_q")
        .Attr<int>("max_seqlen_k")
        .Attr<float>("p_dropout")
        .Attr<float>("softmax_scale")
        .Attr<bool>("zero_tensors")
        .Attr<bool>("is_causal")
        .Attr<int>("window_size_left")
        .Attr<int>("window_size_right")
        .Attr<bool>("deterministic")
        .Attr<bool>("is_v3_atomic_fp32")
        .Attr<int>("how_v3_bf16_cvt"),
    {xla::ffi::Traits::kCmdBufferCompatible});

#pragma GCC visibility pop
