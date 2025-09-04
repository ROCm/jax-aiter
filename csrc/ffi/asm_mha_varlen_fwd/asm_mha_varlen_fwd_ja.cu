#include <hip/hip_runtime.h>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include <ATen/hip/HIPContext.h>
#include <torch/all.h>

#include "hip_utils.h"
#include "logging.h"
#include "torch_utils.h"

// Forward declare the exact aiter torch interface.
namespace aiter {
namespace torch_itfs {
std::vector<at::Tensor> fmha_v3_varlen_fwd(
    const at::Tensor &q,            // [total_q, hq, d_q]
    const at::Tensor &k,            // [total_k, hk, d_q]
    const at::Tensor &v,            // [total_k, hk, d_v]
    const at::Tensor &cu_seqlens_q, // [b+1]
    const at::Tensor &cu_seqlens_k, // [b+1]
    const int max_seqlen_q, const int max_seqlen_k, const float p_dropout,
    const float softmax_scale, const bool zero_tensors, const bool is_causal,
    int window_size_left, int window_size_right, const bool return_softmax_lse,
    const bool return_dropout_randval,
    std::optional<at::Tensor> out_,        // [total_q, hq, d_v]
    std::optional<const at::Tensor> bias_, // [max_seqlen_q, max_seqlen_k]
    std::optional<const at::Tensor> alibi_slopes_, // [hq] or [b, hq]
    std::optional<at::Generator> gen_);
}
} // namespace aiter

namespace ffi = xla::ffi;

namespace jax_aiter {

ffi::Error FmhaV3VarlenFwd_Bridge(
    hipStream_t stream,
    ffi::AnyBuffer q,                        // [total_q, hq, d_q]
    ffi::AnyBuffer k,                        // [total_k, hk, d_q]
    ffi::AnyBuffer v,                        // [total_k, hk, d_v]
    ffi::AnyBuffer cu_seqlens_q,             // [b+1]
    ffi::AnyBuffer cu_seqlens_k,             // [b+1]
    ffi::Result<ffi::AnyBuffer> out,         // [total_q, hq, d_v]
    ffi::Result<ffi::AnyBuffer> softmax_lse, // [b, hq, max_seqlen_q]
    ffi::Result<ffi::AnyBuffer>
        p, // [b, hq, max_seqlen_q, max_seqlen_k] (dropout mask)
    ffi::Result<ffi::AnyBuffer> rng_state, // [2]
    int64_t max_seqlen_q, int64_t max_seqlen_k, float p_dropout,
    float softmax_scale, bool zero_tensors, bool is_causal,
    int64_t window_size_left, int64_t window_size_right,
    bool return_softmax_lse, bool return_dropout_randval,
    ffi::AnyBuffer out_provided, // [total_q, hq, d_v] (optional)
    ffi::AnyBuffer bias,         // [max_seqlen_q, max_seqlen_k] (optional)
    ffi::AnyBuffer alibi_slopes, // [hq] or [b, hq] (optional)
    ffi::AnyBuffer gen           // generator (optional)
) {
  JA_LOG("FMHA_V3_VARLEN_FWD max_seqlen_q=%ld max_seqlen_k=%ld dropout=%f "
         "scale=%f causal=%d window=(%ld,%ld) lse=%d randval=%d",
         max_seqlen_q, max_seqlen_k, p_dropout, softmax_scale, is_causal,
         window_size_left, window_size_right, return_softmax_lse,
         return_dropout_randval);

  // Get device index for tensor creation.
  const int dev_idx = ::jax_aiter::device_from_ptr(q.untyped_data());

  // Create tensor views from the JAX buffers for inputs.
  auto q_tensor = ::jax_aiter::wrap_any_buffer(q, dev_idx);
  auto k_tensor = ::jax_aiter::wrap_any_buffer(k, dev_idx);
  auto v_tensor = ::jax_aiter::wrap_any_buffer(v, dev_idx);
  auto cu_seqlens_q_tensor =
      ::jax_aiter::wrap_any_buffer(cu_seqlens_q, dev_idx);
  auto cu_seqlens_k_tensor =
      ::jax_aiter::wrap_any_buffer(cu_seqlens_k, dev_idx);

  // Handle optional parameters (check for None by buffer size).
  std::optional<at::Tensor> out_provided_opt = std::nullopt;
  if (out_provided.size_bytes() > 0) {
    out_provided_opt =
        std::make_optional(::jax_aiter::wrap_any_buffer(out_provided, dev_idx));
  }

  std::optional<const at::Tensor> bias_opt =
      bias.size_bytes() > 0 ? std::make_optional<const at::Tensor>(
                                  ::jax_aiter::wrap_any_buffer(bias, dev_idx))
                            : std::nullopt;

  std::optional<const at::Tensor> alibi_slopes_opt =
      alibi_slopes.size_bytes() > 0
          ? std::make_optional<const at::Tensor>(
                ::jax_aiter::wrap_any_buffer(alibi_slopes, dev_idx))
          : std::nullopt;

  // Generator is always None for now.
  std::optional<at::Generator> gen_opt = std::nullopt;

  try {
    // Call the aiter FMHA V3 varlen forward PyTorch kernel with exact signature
    // match.
    auto results = aiter::torch_itfs::fmha_v3_varlen_fwd(
        q_tensor,                            // q: [total_q, hq, d_q]
        k_tensor,                            // k: [total_k, hk, d_q]
        v_tensor,                            // v: [total_k, hk, d_v]
        cu_seqlens_q_tensor,                 // cu_seqlens_q: [b+1]
        cu_seqlens_k_tensor,                 // cu_seqlens_k: [b+1]
        static_cast<int>(max_seqlen_q),      // max_seqlen_q
        static_cast<int>(max_seqlen_k),      // max_seqlen_k
        p_dropout,                           // p_dropout
        softmax_scale,                       // softmax_scale
        zero_tensors,                        // zero_tensors
        is_causal,                           // is_causal
        static_cast<int>(window_size_left),  // window_size_left
        static_cast<int>(window_size_right), // window_size_right
        return_softmax_lse,                  // return_softmax_lse
        return_dropout_randval,              // return_dropout_randval
        out_provided_opt, // out_ (optional pre-allocated output)
        bias_opt,         // bias_ (optional)
        alibi_slopes_opt, // alibi_slopes_ (optional)
        gen_opt           // gen_ (optional generator)
    );

    // Copy results back to JAX output buffers.
    // results = {out, softmax_lse, p, rng_state}
    if (results.size() >= 4) {
      // Copy output tensor
      size_t output_bytes = results[0].numel() * results[0].element_size();
      if (output_bytes <= out->size_bytes()) {
        hipMemcpy(out->untyped_data(), results[0].data_ptr(), output_bytes,
                  hipMemcpyDeviceToDevice);
      }

      // Copy softmax_lse if requested.
      if (return_softmax_lse && results[1].numel() > 0 &&
          softmax_lse->size_bytes() > 0) {
        size_t lse_bytes = results[1].numel() * results[1].element_size();
        if (lse_bytes <= softmax_lse->size_bytes()) {
          hipMemcpy(softmax_lse->untyped_data(), results[1].data_ptr(),
                    lse_bytes, hipMemcpyDeviceToDevice);
        }
      }

      // Copy dropout mask if requested.
      if (return_dropout_randval && results[2].numel() > 0 &&
          p->size_bytes() > 0) {
        size_t p_bytes = results[2].numel() * results[2].element_size();
        if (p_bytes <= p->size_bytes()) {
          hipMemcpy(p->untyped_data(), results[2].data_ptr(), p_bytes,
                    hipMemcpyDeviceToDevice);
        }
      }

      // Copy rng_state if present.
      if (results[3].numel() > 0 && rng_state->size_bytes() > 0) {
        size_t rng_bytes = results[3].numel() * results[3].element_size();
        if (rng_bytes <= rng_state->size_bytes()) {
          hipMemcpy(rng_state->untyped_data(), results[3].data_ptr(), rng_bytes,
                    hipMemcpyDeviceToDevice);
        }
      }
    }

    JA_LOG("FMHA_V3_VARLEN_FWD completed successfully");

    return ffi::Error::Success();
  } catch (const std::exception &e) {
    JA_LOG("FMHA_V3_VARLEN_FWD failed: %s", e.what());
    return ffi::Error(ffi::ErrorCode::kInternal, e.what());
  }
}

} // namespace jax_aiter

#pragma GCC visibility push(default)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FmhaV3VarlenFwdJA, jax_aiter::FmhaV3VarlenFwd_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>() // q: [total_q, hq, d_q]
        .Arg<ffi::AnyBuffer>() // k: [total_k, hk, d_q]
        .Arg<ffi::AnyBuffer>() // v: [total_k, hk, d_v]
        .Arg<ffi::AnyBuffer>() // cu_seqlens_q: [b+1]
        .Arg<ffi::AnyBuffer>() // cu_seqlens_k: [b+1]
        .Ret<ffi::AnyBuffer>() // out: [total_q, hq, d_v]
        .Ret<ffi::AnyBuffer>() // softmax_lse: [b, hq, max_seqlen_q]
        .Ret<ffi::AnyBuffer>() // p: [b, hq, max_seqlen_q, max_seqlen_k]
                               // (dropout mask)
        .Ret<ffi::AnyBuffer>() // rng_state: [2]
        .Attr<int64_t>("max_seqlen_q")
        .Attr<int64_t>("max_seqlen_k")
        .Attr<float>("p_dropout")
        .Attr<float>("softmax_scale")
        .Attr<bool>("zero_tensors")
        .Attr<bool>("is_causal")
        .Attr<int64_t>("window_size_left")
        .Attr<int64_t>("window_size_right")
        .Attr<bool>("return_softmax_lse")
        .Attr<bool>("return_dropout_randval")
        .Arg<ffi::AnyBuffer>()  // out_provided: [total_q, hq, d_v] (optional)
        .Arg<ffi::AnyBuffer>()  // bias: [max_seqlen_q, max_seqlen_k] (optional)
        .Arg<ffi::AnyBuffer>()  // alibi_slopes: [hq] or [b, hq] (optional)
        .Arg<ffi::AnyBuffer>(), // gen: generator (optional)
    {xla::ffi::Traits::kCmdBufferCompatible});

#pragma GCC visibility pop
