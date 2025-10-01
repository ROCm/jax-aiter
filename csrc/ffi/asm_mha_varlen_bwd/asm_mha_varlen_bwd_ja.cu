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

// Forward declare the exact aiter torch interface
namespace aiter {
namespace torch_itfs {
std::vector<at::Tensor> fmha_v3_varlen_bwd(
    const at::Tensor &dout,         // [total_q, hq, d_v]
    const at::Tensor &q,            // [total_q, hq, d_q]
    const at::Tensor &k,            // [total_k, hk, d_q]
    const at::Tensor &v,            // [total_k, hk, d_v]
    const at::Tensor &out,          // [total_q, hq, d_v]
    const at::Tensor &softmax_lse,  // [b, hq, sq]
    const at::Tensor &cu_seqlens_q, // [b+1]
    const at::Tensor &cu_seqlens_k, // [b+1]
    const int max_seqlen_q, const int max_seqlen_k, const float p_dropout,
    const float softmax_scale, const bool zero_tensors, const bool is_causal,
    int window_size_left, int window_size_right, const bool deterministic,
    bool is_v3_atomic_fp32, int how_v3_bf16_cvt,
    std::optional<at::Tensor> dq_,                 // [total_q, hq, d_q]
    std::optional<at::Tensor> dk_,                 // [total_k, hk, d_q]
    std::optional<at::Tensor> dv_,                 // [total_k, hk, d_v]
    std::optional<const at::Tensor> alibi_slopes_, // [hq] or [b, hq]
    std::optional<const at::Tensor> rng_state_,
    std::optional<at::Generator> gen_);
}
} // namespace aiter

namespace ffi = xla::ffi;

namespace jax_aiter {

ffi::Error FmhaV3VarlenBwd_Bridge(
    hipStream_t stream,
    ffi::AnyBuffer dout,                        // [total_q, hq, d_v]
    ffi::AnyBuffer q,                           // [total_q, hq, d_q]
    ffi::AnyBuffer k,                           // [total_k, hk, d_q]
    ffi::AnyBuffer v,                           // [total_k, hk, d_v]
    ffi::AnyBuffer out,                         // [total_q, hq, d_v]
    ffi::AnyBuffer softmax_lse,                 // [b, hq, sq]
    ffi::AnyBuffer cu_seqlens_q,                // [b+1]
    ffi::AnyBuffer cu_seqlens_k,                // [b+1]
    std::optional<ffi::AnyBuffer> dq_provided,  // [total_q, hq, d_q] (optional)
    std::optional<ffi::AnyBuffer> dk_provided,  // [total_k, hk, d_q] (optional)
    std::optional<ffi::AnyBuffer> dv_provided,  // [total_k, hk, d_v] (optional)
    std::optional<ffi::AnyBuffer> alibi_slopes, // [hq] or [b, hq] (optional)
    std::optional<ffi::AnyBuffer> rng_state,    // [2] (optional)
    std::optional<ffi::AnyBuffer> gen_,          // generator (optional)
    ffi::Result<ffi::AnyBuffer> dq,             // [total_q, hq, d_q]
    ffi::Result<ffi::AnyBuffer> dk,             // [total_k, hk, d_q]
    ffi::Result<ffi::AnyBuffer> dv,             // [total_k, hk, d_v]
    ffi::Result<ffi::AnyBuffer> softmax_d,      // [b, hq, max_seqlen_q]
    int max_seqlen_q, int max_seqlen_k, double p_dropout,
    double softmax_scale, bool zero_tensors, bool is_causal,
    int window_size_left, int window_size_right, bool deterministic,
    bool is_v3_atomic_fp32, int how_v3_bf16_cvt) {
  // Get device index for tensor creation.
  const int dev_idx = ::jax_aiter::device_from_ptr(q.untyped_data());

  // Create tensor views from the JAX buffers for inputs
  auto dout_tensor = ::jax_aiter::wrap_any_buffer(dout, dev_idx);
  auto q_tensor = ::jax_aiter::wrap_any_buffer(q, dev_idx);
  auto k_tensor = ::jax_aiter::wrap_any_buffer(k, dev_idx);
  auto v_tensor = ::jax_aiter::wrap_any_buffer(v, dev_idx);
  auto out_tensor = ::jax_aiter::wrap_any_buffer(out, dev_idx);
  auto softmax_lse_tensor = ::jax_aiter::wrap_any_buffer(softmax_lse, dev_idx);
  auto cu_seqlens_q_tensor =
      ::jax_aiter::wrap_any_buffer(cu_seqlens_q, dev_idx);
  auto cu_seqlens_k_tensor =
      ::jax_aiter::wrap_any_buffer(cu_seqlens_k, dev_idx);

  // Set device and stream guards
  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(
      device_of(q_tensor));

  const c10::hip::HIPStreamMasqueradingAsCUDA ext_stream =
      c10::hip::getStreamFromExternalMasqueradingAsCUDA(stream, dev_idx);
  const c10::hip::HIPStreamGuardMasqueradingAsCUDA stream_guard{ext_stream};

  // Handle optional parameters (check for None by buffer size).
  std::optional<const at::Tensor> dq_provided_opt =
      (dq_provided.has_value() && dq_provided->size_bytes() > 0)
          ? std::make_optional<const at::Tensor>(
                ::jax_aiter::wrap_any_buffer(*dq_provided, dev_idx))
          : std::nullopt;

  std::optional<const at::Tensor> dk_provided_opt =
      (dk_provided.has_value() && dk_provided->size_bytes() > 0)
          ? std::make_optional<const at::Tensor>(
                ::jax_aiter::wrap_any_buffer(*dk_provided, dev_idx))
          : std::nullopt;

  std::optional<const at::Tensor> dv_provided_opt =
      (dv_provided.has_value() && dv_provided->size_bytes() > 0)
          ? std::make_optional<const at::Tensor>(
                ::jax_aiter::wrap_any_buffer(*dv_provided, dev_idx))
          : std::nullopt;

  std::optional<const at::Tensor> alibi_slopes_opt =
      (alibi_slopes.has_value() && alibi_slopes->size_bytes() > 0)
          ? std::make_optional<const at::Tensor>(
                ::jax_aiter::wrap_any_buffer(*alibi_slopes, dev_idx))
          : std::nullopt;

  std::optional<const at::Tensor> rng_state_opt =
      (rng_state.has_value() && rng_state->size_bytes() > 0)
          ? std::make_optional<const at::Tensor>(
                ::jax_aiter::wrap_any_buffer(*rng_state, dev_idx))
          : std::nullopt;

  // Handle generator parameter for dropout.
  std::optional<at::Generator> gen_opt = std::nullopt;
  if (gen_.has_value() &&
      gen_->size_bytes() >= static_cast<size_t>(2 * sizeof(int64_t))) {
    // Extract seed and offset from JAX generator buffer.
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

  try {
    auto results = aiter::torch_itfs::fmha_v3_varlen_bwd(
        dout_tensor,                         // dout: [total_q, hq, d_v]
        q_tensor,                            // q: [total_q, hq, d_q]
        k_tensor,                            // k: [total_k, hk, d_q]
        v_tensor,                            // v: [total_k, hk, d_v]
        out_tensor,                          // out: [total_q, hq, d_v]
        softmax_lse_tensor,                  // softmax_lse: [b, hq, sq]
        cu_seqlens_q_tensor,                 // cu_seqlens_q: [b+1]
        cu_seqlens_k_tensor,                 // cu_seqlens_k: [b+1]
        max_seqlen_q,                        // max_seqlen_q
        max_seqlen_k,                        // max_seqlen_k
        p_dropout,                           // p_dropout
        softmax_scale,                       // softmax_scale
        zero_tensors,                        // zero_tensors
        is_causal,                           // is_causal
        window_size_left,                    // window_size_left
        window_size_right,                   // window_size_right
        deterministic,                       // deterministic
        is_v3_atomic_fp32,                   // is_v3_atomic_fp32
        how_v3_bf16_cvt,                     // how_v3_bf16_cvt
        dq_provided_opt,                     // dq_ (optional pre-allocated dq)
        dk_provided_opt,                     // dk_ (optional pre-allocated dk)
        dv_provided_opt,                     // dv_ (optional pre-allocated dv)
        alibi_slopes_opt,                    // alibi_slopes_ (optional)
        rng_state_opt,                       // rng_state_ (optional)
        gen_opt                              // gen_ (optional generator)
    );
    // Copy results back to JAX output buffers
    // results = {dq, dk, dv, softmax_d}
    if (results.size() >= 4) {
      // Create output tensor views for copying results back.
      auto dq_tensor = ::jax_aiter::wrap_any_buffer(*dq, dev_idx);
      dq_tensor.copy_(results[0], /*non_blocking=*/true); // dq gradient

      if (results[1].numel() > 0) {
        auto dk_tensor = ::jax_aiter::wrap_any_buffer(*dk, dev_idx);
        dk_tensor.copy_(results[1], /*non_blocking=*/true); // dk gradient
      }

      if (results[2].numel() > 0) {
        auto dv_tensor = ::jax_aiter::wrap_any_buffer(*dv, dev_idx);
        dv_tensor.copy_(results[2], /*non_blocking=*/true); // dv gradient
      }

      if (results[3].numel() > 0) {
        auto softmax_d_tensor =
            ::jax_aiter::wrap_any_buffer(*softmax_d, dev_idx);
        softmax_d_tensor.copy_(results[3], /*non_blocking=*/true); // softmax_d
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
    FmhaV3VarlenBwdJA, jax_aiter::FmhaV3VarlenBwd_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>() // dout: [total_q, hq, d_v]
        .Arg<ffi::AnyBuffer>() // q: [total_q, hq, d_q]
        .Arg<ffi::AnyBuffer>() // k: [total_k, hk, d_q]
        .Arg<ffi::AnyBuffer>() // v: [total_k, hk, d_v]
        .Arg<ffi::AnyBuffer>() // out: [total_q, hq, d_v]
        .Arg<ffi::AnyBuffer>() // softmax_lse: [b, hq, sq]
        .Arg<ffi::AnyBuffer>() // cu_seqlens_q: [b+1]
        .Arg<ffi::AnyBuffer>() // cu_seqlens_k: [b+1]
        .Arg<ffi::AnyBuffer>() // dq_provided: [total_q, hq, d_q] (optional)
        .Arg<ffi::AnyBuffer>() // dk_provided: [total_k, hk, d_q] (optional)
        .Arg<ffi::AnyBuffer>() // dv_provided: [total_k, hk, d_v] (optional)
        .Arg<ffi::AnyBuffer>() // alibi_slopes: [hq] or [b, hq] (optional)
        .Arg<ffi::AnyBuffer>() // rng_state: [2] (optional)
        .Arg<ffi::AnyBuffer>() // gen: generator (optional)
        .Ret<ffi::AnyBuffer>() // dq: [total_q, hq, d_q]
        .Ret<ffi::AnyBuffer>() // dk: [total_k, hk, d_q]
        .Ret<ffi::AnyBuffer>() // dv: [total_k, hk, d_v]
        .Ret<ffi::AnyBuffer>() // softmax_d: [b, hq, max_seqlen_q]
        .Attr<int>("max_seqlen_q")
        .Attr<int>("max_seqlen_k")
        .Attr<double>("p_dropout")
        .Attr<double>("softmax_scale")
        .Attr<bool>("zero_tensors")
        .Attr<bool>("is_causal")
        .Attr<int>("window_size_left")
        .Attr<int>("window_size_right")
        .Attr<bool>("deterministic")
        .Attr<bool>("is_v3_atomic_fp32")
        .Attr<int>("how_v3_bf16_cvt"),
    {xla::ffi::Traits::kCmdBufferCompatible});

#pragma GCC visibility pop
