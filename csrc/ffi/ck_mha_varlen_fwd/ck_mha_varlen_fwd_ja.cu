// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>
#include <vector>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "hip_utils.h"
#include "logging.h"
#include "mha_common_utils.h"
#include "mha_fwd.h"

namespace ffi = xla::ffi;

namespace jax_aiter {

ffi::Error MhaVarlenFwd_Bridge(
    hipStream_t stream, ffi::AnyBuffer q, ffi::AnyBuffer k, ffi::AnyBuffer v,
    ffi::AnyBuffer cu_seqlens_q, std::optional<ffi::AnyBuffer> cu_seqlens_k,
    std::optional<ffi::AnyBuffer> out_provided,
    std::optional<ffi::AnyBuffer> block_table_,
    std::optional<ffi::AnyBuffer> bias_,
    std::optional<ffi::AnyBuffer> alibi_slopes_,
    std::optional<ffi::AnyBuffer> gen_,
    std::optional<ffi::AnyBuffer> cu_seqlens_q_padded_,
    std::optional<ffi::AnyBuffer> cu_seqlens_k_padded_,
    ffi::Result<ffi::AnyBuffer> out, ffi::Result<ffi::AnyBuffer> softmax_lse,
    ffi::Result<ffi::AnyBuffer> p, ffi::Result<ffi::AnyBuffer> rng_state,
    int max_seqlen_q, int max_seqlen_k, int min_seqlen_q, float p_dropout,
    float softmax_scale, float logits_soft_cap, bool zero_tensors,
    bool is_causal, int window_size_left, int window_size_right,
    bool return_softmax_lse, bool return_dropout_randval) {

  if (!q.untyped_data() || !k.untyped_data() || !v.untyped_data() ||
      !cu_seqlens_q.untyped_data()) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "Required input buffer (q/k/v/cu_seqlens_q) is null");
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

    // Varlen layout is [total_q, nheads, d].
    int64_t total_q = q_dims[0];
    int64_t num_heads = q_dims[1];
    int64_t head_size_q = q_dims[2];

    int64_t total_k = k_dims[0];
    int64_t num_heads_k = k_dims[1];
    int64_t head_size_v = v_dims[2];

    int64_t batch_size = cu_q_dims[0] - 1;

    // Validate dtypes.
    auto q_dtype = q.element_type();
    if (q_dtype != xla::ffi::DataType::F16 &&
        q_dtype != xla::ffi::DataType::BF16) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                        "FlashAttention only supports fp16/bf16");
    }

    if (k.element_type() != q_dtype || v.element_type() != q_dtype) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                        "Q, K, V must have same dtype");
    }

    if (cu_seqlens_q.element_type() != xla::ffi::DataType::S32) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                        "cu_seqlens_q must be int32");
    }

    // Basic shape checks.
    if (batch_size <= 0) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                        "batch size must be positive");
    }
    if (head_size_q > 256 || head_size_v > 256) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                        "CK only supports head dimension at most 256");
    }
    if (head_size_q % 8 != 0 || head_size_v % 8 != 0) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                        "head dimensions must be multiples of 8");
    }
    if (num_heads % num_heads_k != 0) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                        "Number of heads in query must be divisible by number "
                        "of heads in key/value");
    }

    std::string dtype_str = mha_utils::dtype_to_string(q.element_type());

    // Handle optional bias and ALiBi inputs.
    const void *bias_ptr = nullptr;
    const void *alibi_ptr = nullptr;
    bool has_bias = bias_.has_value() && mha_utils::is_valid_buffer(*bias_);
    bool has_alibi =
        alibi_slopes_.has_value() && mha_utils::is_valid_buffer(*alibi_slopes_);

    if (has_bias && has_alibi) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                        "cannot apply both bias and alibi");
    }

    if (has_bias) {
      bias_ptr = bias_->untyped_data();
    }
    if (has_alibi) {
      alibi_ptr = alibi_slopes_->untyped_data();
    }

    bias_enum bias_type = mha_utils::get_bias_type(has_bias, has_alibi);
    const void *final_bias_ptr = has_alibi ? alibi_ptr : bias_ptr;

    // Create attention mask metadata.
    if (max_seqlen_q == 1 && !has_alibi) {
      is_causal = false;
    }

    // Normalize attention window sizes.
    if (window_size_left >= max_seqlen_k) {
      window_size_left = -1;
    }
    if (window_size_right >= max_seqlen_k) {
      window_size_right = -1;
    }

    mask_info mask = mha_utils::create_mask_info(is_causal, window_size_left,
                                                 window_size_right,
                                                 max_seqlen_q, max_seqlen_k);

    // Prepare RNG state for dropout.
    mha_utils::RngStatePointers rng_ptrs;
    auto rng_err = mha_utils::prepare_rng_state_for_fwd(
        stream, p_dropout, dev_idx, batch_size, num_heads, gen_, rng_state,
        rng_ptrs);
    if (!rng_err.success()) {
      return rng_err;
    }

    // Get cu_seqlens pointers.
    const ck_tile::index_t *cu_seqlens_q_ptr =
        reinterpret_cast<const ck_tile::index_t *>(cu_seqlens_q.untyped_data());
    const ck_tile::index_t *cu_seqlens_k_ptr = nullptr;

    if (cu_seqlens_k.has_value() && mha_utils::is_valid_buffer(*cu_seqlens_k)) {
      cu_seqlens_k_ptr = reinterpret_cast<const ck_tile::index_t *>(
          cu_seqlens_k->untyped_data());
    }

    // Zero-initialize outputs when zero_tensors is set.
    if (zero_tensors) {
      HIP_CHECK(
          hipMemsetAsync(out->untyped_data(), 0, out->size_bytes(), stream));
      if (return_softmax_lse && softmax_lse->size_bytes() > 0) {
        // Initialize softmax_lse buffer.
        float neg_inf = -std::numeric_limits<float>::infinity();
        // Use memset to initialize softmax_lse.
        HIP_CHECK(hipMemsetAsync(softmax_lse->untyped_data(), 0,
                                 softmax_lse->size_bytes(), stream));
      }
      if (return_dropout_randval && p->size_bytes() > 0) {
        HIP_CHECK(
            hipMemsetAsync(p->untyped_data(), 0, p->size_bytes(), stream));
      }
    }

    // Check for paged KV attention.
    bool is_paged_kv =
        block_table_.has_value() && mha_utils::is_valid_buffer(*block_table_);

    if (is_paged_kv) {
      // Paged KV attention is not supported in varlen forward.
      return ffi::Error(
          ffi::ErrorCode::kUnimplemented,
          "Paged KV attention not yet supported in Torch-free varlen forward");
    }

    if (max_seqlen_k == 0) {
      // Handle empty-key case by zeroing outputs.
      HIP_CHECK(
          hipMemsetAsync(out->untyped_data(), 0, out->size_bytes(), stream));
      if (return_softmax_lse && softmax_lse->size_bytes() > 0) {
        float inf_val = std::numeric_limits<float>::infinity();
        // Initialize softmax_lse buffer for empty sequences.
        HIP_CHECK(hipMemsetAsync(softmax_lse->untyped_data(), 0,
                                 softmax_lse->size_bytes(), stream));
      }
      return ffi::Error::Success();
    }

    // Compute strides for varlen layout [total, nheads, d].
    ck_tile::index_t stride_q = mha_utils::calculate_stride(q_dims, 0);
    ck_tile::index_t stride_k = mha_utils::calculate_stride(k_dims, 0);
    ck_tile::index_t stride_v = mha_utils::calculate_stride(v_dims, 0);

    auto out_dims = out->dimensions();
    ck_tile::index_t stride_o = mha_utils::calculate_stride(out_dims, 0);

    ck_tile::index_t nhead_stride_q = mha_utils::calculate_stride(q_dims, 1);
    ck_tile::index_t nhead_stride_k = mha_utils::calculate_stride(k_dims, 1);
    ck_tile::index_t nhead_stride_v = mha_utils::calculate_stride(v_dims, 1);
    ck_tile::index_t nhead_stride_o = mha_utils::calculate_stride(out_dims, 1);

    ck_tile::index_t nhead_stride_lse = 0;
    ck_tile::index_t stride_randval = 0;
    ck_tile::index_t nhead_stride_randval = 0;

    if (return_softmax_lse) {
      auto lse_dims = softmax_lse->dimensions();
      nhead_stride_lse = mha_utils::calculate_stride(lse_dims, 0);
    }

    if (return_dropout_randval) {
      auto p_dims = p->dimensions();
      stride_randval = mha_utils::calculate_stride(p_dims, 1);
      nhead_stride_randval = mha_utils::calculate_stride(p_dims, 0);
    }

    // Compute bias and ALiBi strides.
    ck_tile::index_t stride_bias = 0;
    if (final_bias_ptr != nullptr) {
      if (has_bias && bias_->dimensions().size() >= 2) {
        stride_bias = mha_utils::calculate_stride(bias_->dimensions(), 0);
      } else if (has_alibi && alibi_slopes_->dimensions().size() >= 2) {
        stride_bias =
            mha_utils::calculate_stride(alibi_slopes_->dimensions(), 0);
      }
    }

    // Construct fmha_fwd_args for varlen; is_group_mode=true.
    auto args = fmha_fwd_args{
        q.untyped_data(),
        k.untyped_data(),
        v.untyped_data(),
        final_bias_ptr,
        return_dropout_randval ? p->untyped_data() : nullptr,
        return_softmax_lse ? softmax_lse->untyped_data() : nullptr,
        out->untyped_data(),
        nullptr,          // cu_seqlen_q_ptr (batch mode)
        nullptr,          // cu_seqlen_kv_ptr (batch mode)
        cu_seqlens_q_ptr, // seqstart_q (group mode)
        cu_seqlens_k_ptr, // seqstart_k (group mode)
        nullptr,          // seqlen_k_ptr
        nullptr,          // seqstart_padded_q_ptr
        nullptr,          // seqstart_padded_k_ptr
        static_cast<ck_tile::index_t>(total_q),
        static_cast<ck_tile::index_t>(total_k),
        static_cast<ck_tile::index_t>(batch_size),
        static_cast<ck_tile::index_t>(max_seqlen_q),
        static_cast<ck_tile::index_t>(head_size_q),
        static_cast<ck_tile::index_t>(head_size_v),
        static_cast<ck_tile::index_t>(num_heads),
        static_cast<ck_tile::index_t>(num_heads_k),
        softmax_scale,
        1.0f, // scale_p
        1.0f, // scale_o
        logits_soft_cap,
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
        0, // batch_stride_q (varlen uses 0)
        0, // batch_stride_k
        0, // batch_stride_v
        0, // batch_stride_bias
        0, // batch_stride_randval
        0, // batch_stride_lse
        0, // batch_stride_o
        mask.left,
        mask.right,
        static_cast<ck_tile::index_t>(mask.type),
        min_seqlen_q,
        p_dropout,
        return_dropout_randval,
        std::make_pair(rng_ptrs.seed, rng_ptrs.offset)};

    auto stream_config = mha_utils::create_stream_config(stream);

    float runtime = aiter::mha_fwd(args, stream_config, dtype_str,
                                   true, // is_group_mode (varlen)
                                   mask.type, bias_type, return_softmax_lse,
                                   false // use_ext_asm (CK kernels)
    );

    if (runtime < 0) {
      return ffi::Error(ffi::ErrorCode::kInternal,
                        "aiter::mha_fwd failed - invalid arguments or "
                        "unsupported configuration");
    }

    JA_LOG("MhaVarlenFwd completed successfully, elapsed time: %.3f ms",
           runtime);
    return ffi::Error::Success();

  } catch (const std::exception &e) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      std::string("mha_varlen_fwd: ") + e.what());
  }
}

} // namespace jax_aiter

#pragma GCC visibility push(default)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MhaVarlenFwdJA, jax_aiter::MhaVarlenFwd_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>() // q: [total_q, hq, d]
        .Arg<ffi::AnyBuffer>() // k: [total_k, hk, d]
        .Arg<ffi::AnyBuffer>() // v: [total_k, hk, d]
        .Arg<ffi::AnyBuffer>() // cu_seqlens_q: [b+1]
        .Arg<ffi::AnyBuffer>() // cu_seqlens_k: [b+1]
        .Arg<ffi::AnyBuffer>() // out_provided: [total_q, hq, d] (optional)
        .Arg<ffi::AnyBuffer>() // block_table: [hq] or [b, hq] (optional)
        .Arg<ffi::AnyBuffer>() // bias: [total_q, max_seqlen_k] (optional)
        .Arg<ffi::AnyBuffer>() // alibi_slopes: [hq] or [b, hq] (optional)
        .Arg<ffi::AnyBuffer>() // gen: generator (optional)
        .Arg<ffi::AnyBuffer>() // cu_seqlens_q_padded: [b+1] physical starts
                               // with PAD (optional)
        .Arg<ffi::AnyBuffer>() // cu_seqlens_k_padded: [b+1] physical starts
                               // with PAD (optional)
        .Ret<ffi::AnyBuffer>() // out: [total_q, hq, d]
        .Ret<ffi::AnyBuffer>() // softmax_lse: [hq, total_q]
        .Ret<ffi::AnyBuffer>() // p: [hq, total_q, max_seqlen_k] (dropout mask)
        .Ret<ffi::AnyBuffer>() // rng_state: [2]
        .Attr<int>("max_seqlen_q")
        .Attr<int>("max_seqlen_k")
        .Attr<int>("min_seqlen_q")
        .Attr<float>("p_dropout")
        .Attr<float>("softmax_scale")
        .Attr<float>("logits_soft_cap")
        .Attr<bool>("zero_tensors")
        .Attr<bool>("is_causal")
        .Attr<int>("window_size_left")
        .Attr<int>("window_size_right")
        .Attr<bool>("return_softmax_lse")
        .Attr<bool>("return_dropout_randval"),
    {xla::ffi::Traits::kCmdBufferCompatible});

#pragma GCC visibility pop
