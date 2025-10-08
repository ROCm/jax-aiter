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
    bool is_causal, int window_size_left, int window_size_right,
    bool return_softmax_lse, bool return_dropout_randval) {
  if (!q.untyped_data() || !k.untyped_data() || !v.untyped_data()) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "Required input buffer (q/k/v) is null");
  }

  const int dev_idx = ::jax_aiter::device_from_ptr(q.untyped_data());
  if (dev_idx < 0)
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "bad device from q");

  // Wrap input tensors.
  auto q_tensor = ::jax_aiter::wrap_any_buffer(q, dev_idx);
  auto k_tensor = ::jax_aiter::wrap_any_buffer(k, dev_idx);
  auto v_tensor = ::jax_aiter::wrap_any_buffer(v, dev_idx);

  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(
      device_of(q_tensor));

  const c10::hip::HIPStreamMasqueradingAsCUDA ext_stream =
      c10::hip::getStreamFromExternalMasqueradingAsCUDA(stream, dev_idx);
  const c10::hip::HIPStreamGuardMasqueradingAsCUDA stream_guard{ext_stream};

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
  // Handle generator parameter - extract from JAX buffer if provided.
  if (gen_.has_value() &&
      gen_->size_bytes() >= static_cast<size_t>(2 * sizeof(int64_t))) {
    // Expecting exactly 2 int64 values: [seed, offset].
    const auto *gen_data = static_cast<const int64_t *>(gen_->untyped_data());
    const uint64_t seed = static_cast<uint64_t>(gen_data[0]);
    const uint64_t offset = static_cast<uint64_t>(gen_data[1]);

    // Create PyTorch generator with the provided seed.
    auto gen = at::make_generator<at::CUDAGeneratorImpl>(dev_idx);
    auto *impl = gen.get<at::CUDAGeneratorImpl>();
    impl->set_current_seed(seed);
    impl->set_offset(offset);
    gen_opt = gen;
    JA_LOG("Using generator with seed: %llu, offset: %llu", seed, offset);
  }

  std::vector<at::Tensor> results;
  try {
    results = aiter::torch_itfs::fmha_v3_fwd(
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
  } catch (const c10::Error &e) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      std::string("fmha_v3_fwd PyTorch error: ") +
                          e.what_without_backtrace());
  } catch (const std::exception &e) {
    const char *what_msg = e.what();
    std::fprintf(stderr,
                 "[FMHA_FWD] Exception caught - type: %s, message: %s\n",
                 typeid(e).name(), what_msg ? what_msg : "null");
    std::fflush(stderr);
    return ffi::Error(ffi::ErrorCode::kInternal,
                      std::string("fmha_v3_fwd: ") +
                          (what_msg ? what_msg : "unknown error"));
  }

  if (results.size() < 4) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      "Kernel returned insufficient results");
  }
  // Copy results back to JAX output buffers.
  // Results: {out, softmax_lse, p, rng_state}.
  try {
    // Wrap JAX output buffers as PyTorch tensors.
    auto o_tensor = ::jax_aiter::wrap_any_buffer(*o, dev_idx);

    // Copy main outputs.
    o_tensor.copy_(results[0], /*non_blocking=*/true);

    // Only copy LSE if requested (and if result has valid data)
    if (return_softmax_lse && results[1].numel() > 0) {
      auto lse_tensor = ::jax_aiter::wrap_any_buffer(*lse, dev_idx);
      lse_tensor.copy_(results[1], /*non_blocking=*/true);
    }

    // Copy optional dropout mask if valid.
    if (return_dropout_randval && results[2].numel() > 0) {
      auto p_tensor = ::jax_aiter::wrap_any_buffer(*p, dev_idx);
      p_tensor.copy_(results[2], /*non_blocking=*/true);
    }

    // Copy RNG state if valid.
    if (results.size() > 3 && results[3].numel() > 0) {
      auto rng_tensor = ::jax_aiter::wrap_any_buffer(*rng_state, dev_idx);
      rng_tensor.copy_(results[3], /*non_blocking=*/true);
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
