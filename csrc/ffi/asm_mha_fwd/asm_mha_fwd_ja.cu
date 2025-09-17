#include <hip/hip_runtime.h>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include <ATen/hip/HIPContext.h>
#include <torch/all.h>

#include "hip_utils.h"
#include "logging.h"
#include "torch_utils.h"

// Forward declaration for aiter torch interface.
// Returns {out, softmax_lse, p, rng_state}.
namespace aiter {
namespace torch_itfs {
std::vector<at::Tensor>
fmha_v3_fwd(at::Tensor &q,       // [b, sq, hq, d]
            const at::Tensor &k, // [b, sk, hk, d]
            const at::Tensor &v, // [b, sk, hk, d_v]
            float p_dropout, float softmax_scale, bool is_causal,
            int window_size_left, int window_size_right,
            bool return_softmax_lse, bool return_dropout_randval,
            std::optional<at::Tensor> out_,                // [b, sq, hq, d_v]
            std::optional<const at::Tensor> bias_,         // [sq, sk]
            std::optional<const at::Tensor> alibi_slopes_, // [hq] or [b, hq]
            std::optional<at::Generator> gen_);
}
} // namespace aiter

namespace ffi = xla::ffi;

namespace jax_aiter {

ffi::Error FmhaV3Fwd_Bridge(
    hipStream_t stream,
    ffi::AnyBuffer q,                            // [b, sq, hq, d]
    ffi::AnyBuffer k,                            // [b, sk, hk, d]
    ffi::AnyBuffer v,                            // [b, sk, hk, d_v]
    std::optional<ffi::AnyBuffer> out_,          // [b, sq, hq, d_v]
    std::optional<ffi::AnyBuffer> bias_,         // [sq, sk]
    std::optional<ffi::AnyBuffer> alibi_slopes_, // [hq] or [b, hq]
    std::optional<ffi::AnyBuffer> gen_, ffi::Result<ffi::AnyBuffer> o,
    ffi::Result<ffi::AnyBuffer> lse, ffi::Result<ffi::AnyBuffer> p,
    ffi::Result<ffi::AnyBuffer> rng_state, float dropout_p, float softmax_scale,
    bool is_causal, int64_t window_size_left, int64_t window_size_right,
    bool return_softmax_lse, bool return_dropout_randval) {
  const int dev_idx = ::jax_aiter::device_from_ptr(q.untyped_data());

  // Wrap input tensors.
  auto q_tensor = ::jax_aiter::wrap_any_buffer(q, dev_idx);
  auto k_tensor = ::jax_aiter::wrap_any_buffer(k, dev_idx);
  auto v_tensor = ::jax_aiter::wrap_any_buffer(v, dev_idx);

  // Handle optional parameters by checking buffer size.
  std::optional<at::Tensor> out_provided_opt = std::nullopt;
  if (out_.has_value() && out_->size_bytes() > 0) {
    out_provided_opt =
        std::make_optional(::jax_aiter::wrap_any_buffer(*out_, dev_idx));
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

  std::optional<at::Generator> gen_opt = std::nullopt;

  try {
    auto results = aiter::torch_itfs::fmha_v3_fwd(
        q_tensor,               // q: [b, sq, hq, d]
        k_tensor,               // k: [b, sk, hk, d]
        v_tensor,               // v: [b, sk, hk, d_v]
        dropout_p,              // p_dropout
        softmax_scale,          // softmax_scale
        is_causal,              // is_causal
        window_size_left,       // window_size_left
        window_size_right,      // window_size_right
        return_softmax_lse,     // return_softmax_lse
        return_dropout_randval, // return_dropout_randval
        out_provided_opt,       // out_ (optional pre-allocated output)
        bias_opt,               // bias_ (optional)
        alibi_slopes_opt,       // alibi_slopes_ (optional)
        gen_opt                 // gen_ (optional generator)
    );

    // Copy results back to JAX output buffers.
    // Results: {out, softmax_lse, p, rng_state}.
    if (results.size() >= 4) {
      try {
        // Wrap JAX output buffers as PyTorch tensors.
        JA_LOG("Wrapping output buffers");
        auto o_tensor = ::jax_aiter::wrap_any_buffer(*o, dev_idx);
        auto lse_tensor = ::jax_aiter::wrap_any_buffer(*lse, dev_idx);

        // Copy main outputs.
        o_tensor.copy_(results[0]);
        
        // Only copy LSE if requested (and if result has valid data)
        if (return_softmax_lse && results[1].numel() > 0) {
          lse_tensor.copy_(results[1]);
        }

        // Copy optional dropout mask if valid.
        if (p->untyped_data() && p->size_bytes() > 0) {
          auto p_tensor = ::jax_aiter::wrap_any_buffer(*p, dev_idx);
          p_tensor.copy_(results[2]);
        }

        // Copy RNG state if valid.
        if (rng_state->untyped_data() && rng_state->size_bytes() > 0) {
          JA_LOG(
              "rng_state buffer element_type: %d (hex: 0x%x), size: %zu bytes",
              static_cast<int>(rng_state->element_type()),
              static_cast<int>(rng_state->element_type()),
              rng_state->size_bytes());
          auto rng_tensor = ::jax_aiter::wrap_any_buffer(*rng_state, dev_idx);
          rng_tensor.copy_(results[3]);
        }
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
    FmhaV3FwdJA, jax_aiter::FmhaV3Fwd_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>() // q
        .Arg<ffi::AnyBuffer>() // k
        .Arg<ffi::AnyBuffer>() // v
        .Arg<ffi::AnyBuffer>() // out_provided (optional)
        .Arg<ffi::AnyBuffer>() // bias (optional)
        .Arg<ffi::AnyBuffer>() // alibi_slopes (optional)
        .Arg<ffi::AnyBuffer>() // gen (optional)
        .Ret<ffi::AnyBuffer>() // o
        .Ret<ffi::AnyBuffer>() // lse
        .Ret<ffi::AnyBuffer>() // p (dropout mask)
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
