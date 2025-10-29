// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
#include <hip/hip_runtime.h>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include <ATen/hip/HIPContext.h>
#include <ATen/hip/HIPGeneratorImpl.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <torch/all.h>

#include "hip_utils.h"
#include "logging.h"
#include "torch_utils.h"

// Forward declare the aiter torch interface.
// Returns { dq, dk, dv, softmax_d }.
namespace aiter {
namespace torch_itfs {
std::vector<at::Tensor>
mha_bwd(const at::Tensor &dout,        // [b, sq, hq, d_v]
        const at::Tensor &q,           // [b, sq, hq, d]
        const at::Tensor &k,           // [b, sk, hk, d]
        const at::Tensor &v,           // [b, sk, hk, d_v]
        const at::Tensor &out,         // [b, sq, hq, d_v]
        const at::Tensor &softmax_lse, // [b, hq, sq]
        float p_dropout, float softmax_scale, bool is_causal,
        int window_size_left, int window_size_right, bool deterministic,
        std::optional<at::Tensor> dq_,                 // [b, sq, hq, d]
        std::optional<at::Tensor> dk_,                 // [b, sk, hk, d]
        std::optional<at::Tensor> dv_,                 // [b, sk, hk, d]
        std::optional<at::Tensor> dbias_,              // [sq, sk]
        std::optional<const at::Tensor> bias_,         // [sq, sk]
        std::optional<const at::Tensor> alibi_slopes_, // [hq] or [b, hq]
        std::optional<const at::Tensor> rng_state_,
        std::optional<at::Generator> gen_);
}
} // namespace aiter

namespace ffi = xla::ffi;

namespace jax_aiter {

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
    std::optional<ffi::AnyBuffer> dv_,           // [b, sk, hk, d]
    std::optional<ffi::AnyBuffer> bias_,         // [sq, sk]
    std::optional<ffi::AnyBuffer> alibi_slopes_, // [hq] or [b, hq]
    std::optional<ffi::AnyBuffer> rng_state_,
    std::optional<ffi::AnyBuffer> gen_, ffi::Result<ffi::AnyBuffer> dq_ret,
    ffi::Result<ffi::AnyBuffer> dk_ret, ffi::Result<ffi::AnyBuffer> dv_ret,
    ffi::Result<ffi::AnyBuffer> softmax_d_ret,
    ffi::Result<ffi::AnyBuffer> dbias_, float dropout_p, float softmax_scale,
    bool is_causal, int window_size_left, int window_size_right,
    bool deterministic) {
  if (!q.untyped_data() || !k.untyped_data() || !v.untyped_data()) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "Required input buffer (q/k/v) is null");
  }

  const int dev_idx = ::jax_aiter::device_from_ptr(q.untyped_data());
  if (dev_idx < 0)
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "bad device from q");

  auto dout_tensor = ::jax_aiter::wrap_any_buffer(dout, dev_idx);
  auto q_tensor = ::jax_aiter::wrap_any_buffer(q, dev_idx);
  auto k_tensor = ::jax_aiter::wrap_any_buffer(k, dev_idx);
  auto v_tensor = ::jax_aiter::wrap_any_buffer(v, dev_idx);
  auto out_tensor = ::jax_aiter::wrap_any_buffer(out, dev_idx);
  auto softmax_lse_tensor = ::jax_aiter::wrap_any_buffer(softmax_lse, dev_idx);

  std::optional<at::Tensor> dbias_tensor = std::nullopt;
  if (dbias_->size_bytes() > 0) {
    dbias_tensor = ::jax_aiter::wrap_any_buffer(*dbias_, dev_idx);
  }

  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(
      device_of(q_tensor));

  const c10::hip::HIPStreamMasqueradingAsCUDA ext_stream =
      c10::hip::getStreamFromExternalMasqueradingAsCUDA(stream, dev_idx);
  const c10::hip::HIPStreamGuardMasqueradingAsCUDA stream_guard{ext_stream};

  // Handle optional parameters (check for None by buffer size).
  std::optional<at::Tensor> dq_tensor = std::nullopt;
  if (dq_.has_value() && dq_->size_bytes() > 0) {
    dq_tensor = ::jax_aiter::wrap_any_buffer(*dq_, dev_idx);
  }

  std::optional<at::Tensor> dk_tensor = std::nullopt;
  if (dk_.has_value() && dk_->size_bytes() > 0) {
    dk_tensor = ::jax_aiter::wrap_any_buffer(*dk_, dev_idx);
  }

  std::optional<at::Tensor> dv_tensor = std::nullopt;
  if (dv_.has_value() && dv_->size_bytes() > 0) {
    dv_tensor = ::jax_aiter::wrap_any_buffer(*dv_, dev_idx);
  }

  std::optional<const at::Tensor> bias_opt =
      (bias_.has_value() && bias_->size_bytes() > 0)
          ? std::make_optional<const at::Tensor>(
                ::jax_aiter::wrap_any_buffer(*bias_, dev_idx))
          : std::nullopt;

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
  if (gen_.has_value() &&
      gen_->size_bytes() >= static_cast<size_t>(2 * sizeof(int64_t))) {
    // Extract seed and offset from JAX buffer.
    const auto *gen_data = static_cast<const int64_t *>(gen_->untyped_data());
    const uint64_t seed = static_cast<uint64_t>(gen_data[0]);
    const uint64_t offset = static_cast<uint64_t>(gen_data[1]);

    // Create PyTorch generator with the provided seed.
    auto gen = at::make_generator<at::CUDAGeneratorImpl>(dev_idx);
    auto *impl = gen.get<at::CUDAGeneratorImpl>();
    impl->set_current_seed(seed);
    impl->set_offset(offset);
    gen_opt = gen;
  }

  std::vector<at::Tensor> results;
  try {
    // PyTorch function writes directly to the provided output buffers
    // and returns { dq, dk, dv, softmax_d }.
    results = aiter::torch_itfs::mha_bwd(dout_tensor,        // dout
                                         q_tensor,           // q
                                         k_tensor,           // k
                                         v_tensor,           // v
                                         out_tensor,         // out
                                         softmax_lse_tensor, // softmax_lse
                                         dropout_p,          // dropout_p
                                         softmax_scale,      // softmax_scale
                                         is_causal,          // is_causal
                                         window_size_left,   // window_size_left
                                         window_size_right, // window_size_right
                                         deterministic,     // deterministic
                                         dq_tensor,         // dq_
                                         dk_tensor,         // dk_
                                         dv_tensor,         // dv_
                                         dbias_tensor,      // dbias_
                                         bias_opt,          // bias_
                                         alibi_slopes_opt,  // alibi_slopes_
                                         rng_state_opt,     // rng_state_
                                         gen_opt            // gen_
    );
  } catch (const c10::Error &e) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      std::string("mha_bwd PyTorch error: ") +
                          e.what_without_backtrace());
  } catch (const std::exception &e) {
    const char *what_msg = e.what();
    std::fprintf(stderr, "[MHA_BWD] Exception caught - type: %s, message: %s\n",
                 typeid(e).name(), what_msg ? what_msg : "null");
    std::fflush(stderr);
    return ffi::Error(ffi::ErrorCode::kInternal,
                      std::string("mha_bwd: ") +
                          (what_msg ? what_msg : "unknown error"));
  }

  if (results.size() < 4) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      "Kernel returned insufficient results");
  }

  // Copy results back to JAX output buffers
  // results = { dq, dk, dv, softmax_d }.
  try {
    // Wrap JAX output buffers as PyTorch tensors for copying.
    auto dq_tensor = ::jax_aiter::wrap_any_buffer(*dq_ret, dev_idx);
    dq_tensor.copy_(results[0], /*non_blocking=*/true);

    if (results[1].numel() > 0) {
      auto dk_tensor = ::jax_aiter::wrap_any_buffer(*dk_ret, dev_idx);
      dk_tensor.copy_(results[1], /*non_blocking=*/true);
    }
    if (results[2].numel() > 0) {
      auto dv_tensor = ::jax_aiter::wrap_any_buffer(*dv_ret, dev_idx);
      dv_tensor.copy_(results[2], /*non_blocking=*/true);
    }
    if (results[3].numel() > 0) {
      auto softmax_d_tensor =
          ::jax_aiter::wrap_any_buffer(*softmax_d_ret, dev_idx);
      softmax_d_tensor.copy_(results[3], /*non_blocking=*/true);
    }
  } catch (const c10::Error &e) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      std::string("Output copy PyTorch error: ") +
                          e.what_without_backtrace());
  } catch (const std::exception &e) {
    const char *what_msg = e.what();
    return ffi::Error(ffi::ErrorCode::kInternal,
                      std::string("Output copy failed: ") +
                          (what_msg ? what_msg : "unknown error"));
  }
  return ffi::Error::Success();
}

} // namespace jax_aiter

#pragma GCC visibility push(default)

XLA_FFI_DEFINE_HANDLER_SYMBOL(MhaBwdJA, jax_aiter::MhaBwd_Bridge,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<hipStream_t>>()
                                  .Arg<ffi::AnyBuffer>() // dout
                                  .Arg<ffi::AnyBuffer>() // q
                                  .Arg<ffi::AnyBuffer>() // k
                                  .Arg<ffi::AnyBuffer>() // v
                                  .Arg<ffi::AnyBuffer>() // out
                                  .Arg<ffi::AnyBuffer>() // softmax_lse
                                  .Arg<ffi::AnyBuffer>() // dq_
                                  .Arg<ffi::AnyBuffer>() // dk_
                                  .Arg<ffi::AnyBuffer>() // dv_
                                  .Arg<ffi::AnyBuffer>() // bias_
                                  .Arg<ffi::AnyBuffer>() // alibi_slopes_
                                  .Arg<ffi::AnyBuffer>() // rng_state_
                                  .Arg<ffi::AnyBuffer>() // gen_
                                  .Ret<ffi::AnyBuffer>() // dq_ret
                                  .Ret<ffi::AnyBuffer>() // dk_ret
                                  .Ret<ffi::AnyBuffer>() // dv_ret
                                  .Ret<ffi::AnyBuffer>() // softmax_d_ret
                                  .Ret<ffi::AnyBuffer>() // dbias_
                                  .Attr<float>("dropout_p")
                                  .Attr<float>("softmax_scale")
                                  .Attr<bool>("is_causal")
                                  .Attr<int>("window_size_left")
                                  .Attr<int>("window_size_right")
                                  .Attr<bool>("deterministic"),
                              {xla::ffi::Traits::kCmdBufferCompatible});

#pragma GCC visibility pop
