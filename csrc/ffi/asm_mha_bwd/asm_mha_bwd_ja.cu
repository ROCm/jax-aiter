// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
#include <hip/hip_runtime.h>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include <ATen/hip/HIPContext.h>
#include <torch/all.h>

#include "hip_utils.h"
#include "logging.h"
#include "torch_utils.h"

// Forward declaration for aiter torch interface.
// Returns { dq, dk, dv, softmax_d }.
namespace aiter {
namespace torch_itfs {
std::vector<at::Tensor>
fmha_v3_bwd(const at::Tensor &dout,        // [b, sq, hq, d_v]
            const at::Tensor &q,           // [b, sq, hq, d]
            const at::Tensor &k,           // [b, sk, hk, d]
            const at::Tensor &v,           // [b, sk, hk, d_v]
            const at::Tensor &out,         // [b, sq, hq, d_v]
            const at::Tensor &softmax_lse, // [b, hq, sq]
            float p_dropout, float softmax_scale, bool is_causal,
            int window_size_left, int window_size_right, bool deterministic,
            bool is_v3_atomic_fp32, int how_v3_bf16_cvt,
            std::optional<at::Tensor> dq_, std::optional<at::Tensor> dk_,
            std::optional<at::Tensor> dv_,
            std::optional<const at::Tensor> alibi_slopes_, // [hq] or [b, hq]
            std::optional<const at::Tensor> rng_state_,
            std::optional<at::Generator> gen_);
}
} // namespace aiter

namespace ffi = xla::ffi;

namespace jax_aiter {

ffi::Error FmhaV3Bwd_Bridge(
    hipStream_t stream,
    ffi::AnyBuffer dout, // [batch, seqlen_q, nhead, hdim_v]
    ffi::AnyBuffer q,    // [batch, seqlen_q, nhead, hdim_q]
    ffi::AnyBuffer k,    // [batch, seqlen_kv, nhead, hdim_q]
    ffi::AnyBuffer v,    // [batch, seqlen_kv, nhead, hdim_v]
    ffi::AnyBuffer out,  // [batch, seqlen_q, nhead, hdim_v] (from forward pass)
    ffi::AnyBuffer softmax_lse,            // [batch, nhead, seqlen_q]
    std::optional<ffi::AnyBuffer> dq_,
    std::optional<ffi::AnyBuffer> dk_, std::optional<ffi::AnyBuffer> dv_,
    std::optional<ffi::AnyBuffer> alibi_slopes_,
    std::optional<ffi::AnyBuffer> rng_state_,
    std::optional<ffi::AnyBuffer> gen_,
    ffi::Result<ffi::AnyBuffer> dq,        // [batch, seqlen_q, nhead, hdim_q]
    ffi::Result<ffi::AnyBuffer> dk,        // [batch, seqlen_kv, nhead, hdim_q]
    ffi::Result<ffi::AnyBuffer> dv,        // [batch, seqlen_kv, nhead, hdim_v]
    ffi::Result<ffi::AnyBuffer> softmax_d, // [batch, nhead, seqlen_q]
    float dropout_p, float softmax_scale, bool is_causal, int window_size_left,
    int window_size_right, bool deterministic, bool is_v3_atomic_fp32,
    int how_v3_bf16_cvt) {
  const int dev_idx = ::jax_aiter::device_from_ptr(q.untyped_data());

  // Wrap input tensors.
  auto dout_tensor = ::jax_aiter::wrap_any_buffer(dout, dev_idx);
  auto q_tensor = ::jax_aiter::wrap_any_buffer(q, dev_idx);
  auto k_tensor = ::jax_aiter::wrap_any_buffer(k, dev_idx);
  auto v_tensor = ::jax_aiter::wrap_any_buffer(v, dev_idx);
  auto out_tensor = ::jax_aiter::wrap_any_buffer(out, dev_idx);
  auto softmax_lse_tensor = ::jax_aiter::wrap_any_buffer(softmax_lse, dev_idx);

  // Handle optional parameters by checking buffer size.
  std::optional<at::Tensor> dq_provided_opt = std::nullopt;
  if (dq_.has_value() && dq_->size_bytes() > 0) {
    dq_provided_opt =
        std::make_optional(::jax_aiter::wrap_any_buffer(*dq_, dev_idx));
  }

  std::optional<at::Tensor> dk_provided_opt = std::nullopt;
  if (dk_.has_value() && dk_->size_bytes() > 0) {
    dk_provided_opt =
        std::make_optional(::jax_aiter::wrap_any_buffer(*dk_, dev_idx));
  }

  std::optional<at::Tensor> dv_provided_opt = std::nullopt;
  if (dv_.has_value() && dv_->size_bytes() > 0) {
    dv_provided_opt =
        std::make_optional(::jax_aiter::wrap_any_buffer(*dv_, dev_idx));
  }

  std::optional<const at::Tensor> alibi_slopes_opt =
      (alibi_slopes_.has_value() && alibi_slopes_->size_bytes() > 0)
          ? std::make_optional<const at::Tensor>(
                ::jax_aiter::wrap_any_buffer(*alibi_slopes_, dev_idx))
          : std::nullopt;

  std::optional<const at::Tensor> rng_state_opt =
      (rng_state_.has_value() && rng_state_->size_bytes() > 0)
          ? std::make_optional<const at::Tensor>(
                ::jax_aiter::wrap_any_buffer(*rng_state_, dev_idx))
          : std::nullopt;

  std::optional<at::Generator> gen_opt = std::nullopt;

  try {
    auto results = aiter::torch_itfs::fmha_v3_bwd(
        dout_tensor,        // dout: [b, sq, hq, d_v]
        q_tensor,           // q: [b, sq, hq, d]
        k_tensor,           // k: [b, sk, hk, d]
        v_tensor,           // v: [b, sk, hk, d_v]
        out_tensor,         // out: [b, sq, hq, d_v]
        softmax_lse_tensor, // softmax_lse: [b, hq, sq]
        dropout_p,          // p_dropout
        softmax_scale,      // softmax_scale
        is_causal,          // is_causal
        window_size_left,   // window_size_left
        window_size_right,  // window_size_right
        deterministic,      // deterministic
        is_v3_atomic_fp32,  // is_v3_atomic_fp32
        how_v3_bf16_cvt,    // how_v3_bf16_cvt
        dq_provided_opt,    // dq_ (optional pre-allocated)
        dk_provided_opt,    // dk_ (optional pre-allocated)
        dv_provided_opt,    // dv_ (optional pre-allocated)
        alibi_slopes_opt,   // alibi_slopes_ (optional)
        rng_state_opt,      // rng_state_ (optional)
        gen_opt             // gen_ (optional generator)
    );

    // Copy results back to JAX output buffers.
    // Results: {dq, dk, dv, softmax_d}.
    if (results.size() >= 4) {
      try {
        // Wrap JAX output buffers as PyTorch tensors.
        JA_LOG("Wrapping backward output buffers");
        auto dq_tensor = ::jax_aiter::wrap_any_buffer(*dq, dev_idx);
        auto dk_tensor = ::jax_aiter::wrap_any_buffer(*dk, dev_idx);
        auto dv_tensor = ::jax_aiter::wrap_any_buffer(*dv, dev_idx);
        auto softmax_d_tensor =
            ::jax_aiter::wrap_any_buffer(*softmax_d, dev_idx);

        // Copy gradient tensors to output buffers.
        dq_tensor.copy_(results[0]);
        dk_tensor.copy_(results[1]);
        dv_tensor.copy_(results[2]);
        softmax_d_tensor.copy_(results[3]);

        JA_LOG("Successfully copied all backward gradients");
      } catch (const std::exception &e) {
        return ffi::Error(ffi::ErrorCode::kInternal,
                          std::string("Output buffer wrapping failed: ") +
                              e.what());
      }
    }
    return ffi::Error::Success();
  } catch (const std::exception &e) {
    return ffi::Error(ffi::ErrorCode::kInternal, e.what());
  }
}

} // namespace jax_aiter

#pragma GCC visibility push(default)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FmhaV3BwdJA, jax_aiter::FmhaV3Bwd_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>() // dout: [batch, seqlen_q, nhead, hdim_v]
        .Arg<ffi::AnyBuffer>() // q: [batch, seqlen_q, nhead, hdim_q]
        .Arg<ffi::AnyBuffer>() // k: [batch, seqlen_kv, nhead, hdim_q]
        .Arg<ffi::AnyBuffer>() // v: [batch, seqlen_kv, nhead, hdim_v]
        .Arg<ffi::AnyBuffer>() // o: [batch, seqlen_q, nhead, hdim_v]
        .Arg<ffi::AnyBuffer>() // lse: [batch, nhead, seqlen_q]
        .Arg<ffi::AnyBuffer>()  // dq_provided (optional)
        .Arg<ffi::AnyBuffer>()  // dk_provided (optional)
        .Arg<ffi::AnyBuffer>()  // dv_provided (optional)
        .Arg<ffi::AnyBuffer>()  // alibi_slopes (optional)
        .Arg<ffi::AnyBuffer>()  // rng_state_provided (optional)
        .Arg<ffi::AnyBuffer>() // gen (optional)
        .Ret<ffi::AnyBuffer>() // dq: [batch, seqlen_q, nhead, hdim_q]
        .Ret<ffi::AnyBuffer>() // dk: [batch, seqlen_kv, nhead, hdim_q]
        .Ret<ffi::AnyBuffer>() // dv: [batch, seqlen_kv, nhead, hdim_v]
        .Ret<ffi::AnyBuffer>() // softmax_d: [batch, nhead, seqlen_q]
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
