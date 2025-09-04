#include <hip/hip_runtime.h>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include <ATen/hip/HIPContext.h>
#include <torch/all.h>

#include "hip_utils.h"
#include "logging.h"
#include "torch_utils.h"

// Forward declare the aiter torch interface
namespace aiter {
namespace torch_itfs {
std::vector<at::Tensor>
mha_fwd(at::Tensor &q,       // [b, sq, hq, d]
        const at::Tensor &k, // [b, sk, hk, d]
        const at::Tensor &v, // [b, sk, hk, d_v]
        float p_dropout, float softmax_scale, bool is_causal,
        int window_size_left, int window_size_right, bool return_softmax_lse,
        bool return_dropout_randval,
        std::optional<at::Tensor> out_,                // [b, sq, hq, d_v]
        std::optional<const at::Tensor> bias_,         // [sq, sk]
        std::optional<const at::Tensor> alibi_slopes_, // [hq] or [b, hq]
        std::optional<at::Generator> gen_);
}
} // namespace aiter

namespace ffi = xla::ffi;

namespace jax_aiter {

ffi::Error MhaFwd_Bridge(hipStream_t stream, ffi::AnyBuffer q, ffi::AnyBuffer k,
                         ffi::AnyBuffer v, ffi::Result<ffi::AnyBuffer> o,
                         ffi::Result<ffi::AnyBuffer> lse,
                         ffi::Result<ffi::AnyBuffer> p,
                         ffi::Result<ffi::AnyBuffer> rng_state, float dropout_p,
                         float softmax_scale, bool is_causal,
                         int64_t window_size_left, int64_t window_size_right,
                         bool return_softmax_lse, bool return_dropout_randval,
                         ffi::AnyBuffer out_provided, ffi::AnyBuffer bias,
                         ffi::AnyBuffer alibi_slopes, ffi::AnyBuffer gen) {
  const int dev_idx = ::jax_aiter::device_from_ptr(q.untyped_data());

  auto q_tensor = ::jax_aiter::wrap_any_buffer(q, dev_idx);
  auto k_tensor = ::jax_aiter::wrap_any_buffer(k, dev_idx);
  auto v_tensor = ::jax_aiter::wrap_any_buffer(v, dev_idx);

  auto o_tensor = ::jax_aiter::wrap_any_buffer(*o, dev_idx);
  auto lse_tensor = ::jax_aiter::wrap_any_buffer(*lse, dev_idx);
  auto p_tensor = ::jax_aiter::wrap_any_buffer(*p, dev_idx);
  auto rng_tensor = ::jax_aiter::wrap_any_buffer(*rng_state, dev_idx);

  // Handle optional parameters (check for None by buffer size)
  std::optional<at::Tensor> out_provided_opt = std::nullopt;
  if (out_provided.size_bytes() > 0) {
    out_provided_opt =
        std::make_optional(::jax_aiter::wrap_any_buffer(out_provided, dev_idx));
  }

  std::optional<at::Tensor> bias_opt = std::nullopt;
  if (bias.size_bytes() > 0) {
    bias_opt = std::make_optional(::jax_aiter::wrap_any_buffer(bias, dev_idx));
  }

  std::optional<at::Tensor> alibi_slopes_opt = std::nullopt;
  if (alibi_slopes.size_bytes() > 0) {
    alibi_slopes_opt =
        std::make_optional(::jax_aiter::wrap_any_buffer(alibi_slopes, dev_idx));
  }

  std::optional<at::Generator> gen_opt = std::nullopt;

  // Debug: Print input tensor shapes and parameters
  printf("[MHA_DEBUG] === MHA Forward Debug ===\n");
  printf("[MHA_DEBUG] q_tensor: shape=[%ld,%ld,%ld,%ld], numel=%ld\n",
         q_tensor.size(0), q_tensor.size(1), q_tensor.size(2), q_tensor.size(3),
         q_tensor.numel());
  printf("[MHA_DEBUG] k_tensor: shape=[%ld,%ld,%ld,%ld], numel=%ld\n",
         k_tensor.size(0), k_tensor.size(1), k_tensor.size(2), k_tensor.size(3),
         k_tensor.numel());
  printf("[MHA_DEBUG] v_tensor: shape=[%ld,%ld,%ld,%ld], numel=%ld\n",
         v_tensor.size(0), v_tensor.size(1), v_tensor.size(2), v_tensor.size(3),
         v_tensor.numel());
  printf("[MHA_DEBUG] dropout_p: %f, softmax_scale: %f\n", dropout_p,
         softmax_scale);
  printf("[MHA_DEBUG] is_causal: %s, window_size: [%d, %d]\n",
         is_causal ? "true" : "false", static_cast<int>(window_size_left),
         static_cast<int>(window_size_right));
  printf("[MHA_DEBUG] return_softmax_lse: %s, return_dropout_randval: %s\n",
         return_softmax_lse ? "true" : "false",
         return_dropout_randval ? "true" : "false");
  printf("[MHA_DEBUG] out_provided_opt: %s\n",
         out_provided_opt.has_value() ? "provided" : "null");
  printf("[MHA_DEBUG] bias_opt: %s\n",
         bias_opt.has_value() ? "provided" : "null");
  printf("[MHA_DEBUG] alibi_slopes_opt: %s\n",
         alibi_slopes_opt.has_value() ? "provided" : "null");

  try {
    auto results = aiter::torch_itfs::mha_fwd(
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

    // Debug: Print what PyTorch kernel returned
    printf("[MHA_DEBUG] PyTorch kernel returned %zu tensors\n", results.size());
    for (size_t i = 0; i < results.size(); i++) {
      std::string shape_str = "";
      for (int64_t j = 0; j < results[i].dim(); j++) {
        shape_str += std::to_string(results[i].size(j));
        if (j < results[i].dim() - 1)
          shape_str += ",";
      }
      printf("[MHA_DEBUG] Tensor %zu: shape=[%s], numel=%ld, dtype=%s\n", i,
             shape_str.c_str(), results[i].numel(), results[i].dtype().name());
    }

    // Copy results from PyTorch vector to JAX output buffers
    // results = {out, softmax_lse, p, rng_state}
    if (results.size() >= 4) {
      printf("[MHA_DEBUG] Copying tensor 0 (output) with numel=%ld\n",
             results[0].numel());
      o_tensor.copy_(results[0]); // Copy output
      printf("[MHA_DEBUG] Copying tensor 1 (lse) with numel=%ld\n",
             results[1].numel());
      lse_tensor.copy_(results[1]); // Copy LSE

      if (results[2].numel() > 0) {
        printf("[MHA_DEBUG] Copying tensor 2 (dropout mask) with numel=%ld\n",
               results[2].numel());
        p_tensor.copy_(results[2]); // Copy dropout mask
      }
      if (results[3].numel() > 0) {
        printf("[MHA_DEBUG] Copying tensor 3 (rng_state) with numel=%ld\n",
               results[3].numel());
        rng_tensor.copy_(results[3]); // Copy RNG state
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
    MhaFwdJA, jax_aiter::MhaFwd_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>() // q: [batch, seqlen_q, nhead, hdim_q]
        .Arg<ffi::AnyBuffer>() // k: [batch, seqlen_kv, nhead, hdim_q]
        .Arg<ffi::AnyBuffer>() // v: [batch, seqlen_kv, nhead, hdim_v]
        .Ret<ffi::AnyBuffer>() // o: [batch, seqlen_q, nhead, hdim_v]
        .Ret<ffi::AnyBuffer>() // lse: [batch, nhead, seqlen_q]
        .Ret<ffi::AnyBuffer>() // p: dropout mask
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
