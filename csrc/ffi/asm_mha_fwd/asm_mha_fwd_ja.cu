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

ffi::Error
FmhaV3Fwd_Bridge(hipStream_t stream, ffi::AnyBuffer q, ffi::AnyBuffer k,
                 ffi::AnyBuffer v, ffi::Result<ffi::AnyBuffer> o,
                 ffi::Result<ffi::AnyBuffer> lse, ffi::Result<ffi::AnyBuffer> p,
                 ffi::Result<ffi::AnyBuffer> rng_state, float dropout_p,
                 float softmax_scale, bool is_causal, int64_t window_size_left,
                 int64_t window_size_right, bool return_softmax_lse,
                 bool return_dropout_randval, ffi::AnyBuffer out_provided,
                 ffi::AnyBuffer bias, ffi::AnyBuffer alibi_slopes,
                 ffi::AnyBuffer gen) {
  printf("[FMHA_V3_FWD] dropout=%f scale=%f causal=%d window=(%ld,%ld) "
         "return_lse=%d return_randval=%d\n",
         dropout_p, softmax_scale, is_causal, window_size_left,
         window_size_right, return_softmax_lse, return_dropout_randval);

  printf("[FMHA_V3_FWD] Getting device index...\n");
  const int dev_idx = ::jax_aiter::device_from_ptr(q.untyped_data());
  printf("[FMHA_V3_FWD] Device index: %d\n", dev_idx);

  printf("[FMHA_V3_FWD] Wrapping input tensors...\n");
  auto q_tensor = ::jax_aiter::wrap_any_buffer(q, dev_idx);
  printf("[FMHA_V3_FWD] Wrapped q_tensor\n");
  auto k_tensor = ::jax_aiter::wrap_any_buffer(k, dev_idx);
  printf("[FMHA_V3_FWD] Wrapped k_tensor\n");
  auto v_tensor = ::jax_aiter::wrap_any_buffer(v, dev_idx);
  printf("[FMHA_V3_FWD] Wrapped v_tensor\n");

  // Debug: Print input tensor shapes
  printf("[FMHA_V3_FWD] q_tensor: shape=[%ld,%ld,%ld,%ld], numel=%ld\n",
         q_tensor.size(0), q_tensor.size(1), q_tensor.size(2), q_tensor.size(3),
         q_tensor.numel());
  printf("[FMHA_V3_FWD] k_tensor: shape=[%ld,%ld,%ld,%ld], numel=%ld\n",
         k_tensor.size(0), k_tensor.size(1), k_tensor.size(2), k_tensor.size(3),
         k_tensor.numel());
  printf("[FMHA_V3_FWD] v_tensor: shape=[%ld,%ld,%ld,%ld], numel=%ld\n",
         v_tensor.size(0), v_tensor.size(1), v_tensor.size(2), v_tensor.size(3),
         v_tensor.numel());

  // Check result buffer sizes
  printf("[FMHA_V3_FWD] Result buffer sizes: o=%zu, lse=%zu, p=%zu, rng=%zu\n",
         o->size_bytes(), lse->size_bytes(), p->size_bytes(),
         rng_state->size_bytes());

  // Handle optional parameters (check for None by buffer size).
  std::optional<at::Tensor> out_provided_opt = std::nullopt;
  if (out_provided.size_bytes() > 0) {
    out_provided_opt =
        std::make_optional(::jax_aiter::wrap_any_buffer(out_provided, dev_idx));
    printf("[FMHA_V3_FWD] out_provided: provided\n");
  } else {
    printf("[FMHA_V3_FWD] out_provided: null\n");
  }

  std::optional<at::Tensor> bias_opt = std::nullopt;
  if (bias.size_bytes() > 0) {
    bias_opt = std::make_optional(::jax_aiter::wrap_any_buffer(bias, dev_idx));
    printf("[FMHA_V3_FWD] bias: provided\n");
  } else {
    printf("[FMHA_V3_FWD] bias: null\n");
  }

  std::optional<at::Tensor> alibi_slopes_opt = std::nullopt;
  if (alibi_slopes.size_bytes() > 0) {
    alibi_slopes_opt =
        std::make_optional(::jax_aiter::wrap_any_buffer(alibi_slopes, dev_idx));
    printf("[FMHA_V3_FWD] alibi_slopes: provided\n");
  } else {
    printf("[FMHA_V3_FWD] alibi_slopes: null\n");
  }

  // Generator is always None for now.
  std::optional<at::Generator> gen_opt = std::nullopt;

  try {
    printf("[FMHA_V3_FWD] Calling aiter::torch_itfs::fmha_v3_fwd\n");
    auto results = aiter::torch_itfs::fmha_v3_fwd(
        q_tensor,                            // q: [b, sq, hq, d]
        k_tensor,                            // k: [b, sk, hk, d]
        v_tensor,                            // v: [b, sk, hk, d_v]
        dropout_p,                           // p_dropout
        softmax_scale,                       // softmax_scale
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

    printf("[FMHA_V3_FWD] PyTorch kernel returned %zu tensors\n",
           results.size());
    for (size_t i = 0; i < results.size(); i++) {
      std::string shape_str = "";
      for (int64_t j = 0; j < results[i].dim(); j++) {
        shape_str += std::to_string(results[i].size(j));
        if (j < results[i].dim() - 1)
          shape_str += ",";
      }
      printf("[FMHA_V3_FWD] Tensor %zu: shape=[%s], numel=%ld, dtype=%s\n", i,
             shape_str.c_str(), results[i].numel(), results[i].dtype().name());
    }

    // Copy results back to JAX output buffers.
    // results = {out, softmax_lse, p, rng_state}.
    if (results.size() >= 4) {
      printf("[FMHA_V3_FWD] Copying tensor 0 (output) with numel=%ld\n",
             results[0].numel());
      // Copy PyTorch tensor data to JAX output buffer
      size_t output_bytes = results[0].numel() * results[0].element_size();
      if (output_bytes <= o->size_bytes()) {
        hipMemcpy(o->untyped_data(), results[0].data_ptr(), output_bytes,
                  hipMemcpyDeviceToDevice);
      }

      printf("[FMHA_V3_FWD] Copying tensor 1 (lse) with numel=%ld\n",
             results[1].numel());
      size_t lse_bytes = results[1].numel() * results[1].element_size();
      if (lse_bytes <= lse->size_bytes()) {
        hipMemcpy(lse->untyped_data(), results[1].data_ptr(), lse_bytes,
                  hipMemcpyDeviceToDevice);
      }

      if (return_dropout_randval && results[2].numel() > 0 &&
          p->size_bytes() > 0) {
        printf("[FMHA_V3_FWD] Copying tensor 2 (dropout mask) with numel=%ld\n",
               results[2].numel());
        size_t p_bytes = results[2].numel() * results[2].element_size();
        if (p_bytes <= p->size_bytes()) {
          hipMemcpy(p->untyped_data(), results[2].data_ptr(), p_bytes,
                    hipMemcpyDeviceToDevice);
        }
      }

      if (results[3].numel() > 0 && rng_state->size_bytes() > 0) {
        printf("[FMHA_V3_FWD] Copying tensor 3 (rng_state) with numel=%ld\n",
               results[3].numel());
        size_t rng_bytes = results[3].numel() * results[3].element_size();
        if (rng_bytes <= rng_state->size_bytes()) {
          hipMemcpy(rng_state->untyped_data(), results[3].data_ptr(), rng_bytes,
                    hipMemcpyDeviceToDevice);
        }
      }
    }

    printf("[FMHA_V3_FWD] completed successfully\n");

    return ffi::Error::Success();
  } catch (const std::exception &e) {
    printf("[FMHA_V3_FWD] failed: %s\n", e.what());
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
        .Ret<ffi::AnyBuffer>() // o
        .Ret<ffi::AnyBuffer>() // lse
        .Ret<ffi::AnyBuffer>() // p (dropout mask)
        .Ret<ffi::AnyBuffer>() // rng_state
        .Attr<float>("dropout_p")
        .Attr<float>("softmax_scale")
        .Attr<bool>("is_causal")
        .Attr<int64_t>("window_size_left")
        .Attr<int64_t>("window_size_right")
        .Attr<bool>("return_softmax_lse")
        .Attr<bool>("return_dropout_randval")
        .Arg<ffi::AnyBuffer>()  // out_provided (optional)
        .Arg<ffi::AnyBuffer>()  // bias (optional)
        .Arg<ffi::AnyBuffer>()  // alibi_slopes (optional)
        .Arg<ffi::AnyBuffer>(), // gen (optional)
    {xla::ffi::Traits::kCmdBufferCompatible});

#pragma GCC visibility pop
