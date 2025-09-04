#include <hip/hip_runtime.h>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include <ATen/hip/HIPContext.h>
#include <torch/all.h>

#include "hip_utils.h"
#include "logging.h"
#include "torch_utils.h"

// Forward declare the exact aiter torch interface
namespace aiter {
namespace torch_itfs {
std::vector<at::Tensor>
mha_varlen_fwd(at::Tensor &q,                  // [total_q, hq, d]
               const at::Tensor &k,            // [total_k, hk, d]
               const at::Tensor &v,            // [total_k, hk, d]
               const at::Tensor &cu_seqlens_q, // [b+1]
               std::optional<const at::Tensor> &cu_seqlens_k, // [b+1]
               int max_seqlen_q, int max_seqlen_k, int min_seqlen_q,
               float p_dropout, float softmax_scale, float logits_soft_cap,
               bool zero_tensors, bool is_causal, int window_size_left,
               int window_size_right, bool return_softmax_lse,
               bool return_dropout_randval,
               std::optional<at::Tensor> out_,               // [total_q, hq, d]
               std::optional<const at::Tensor> block_table_, // [hq] or [b, hq]
               std::optional<const at::Tensor> bias_, // [total_q, max_seqlen_k]
               std::optional<const at::Tensor> alibi_slopes_, // [hq] or [b, hq]
               std::optional<at::Generator> gen_);
}
} // namespace aiter

namespace ffi = xla::ffi;

namespace jax_aiter {

ffi::Error MhaVarlenFwd_Bridge(
    hipStream_t stream,
    ffi::AnyBuffer q,                        // [total_q, hq, d]
    ffi::AnyBuffer k,                        // [total_k, hk, d]
    ffi::AnyBuffer v,                        // [total_k, hk, d]
    ffi::AnyBuffer cu_seqlens_q,             // [b+1]
    ffi::Result<ffi::AnyBuffer> out,         // [total_q, hq, d]
    ffi::Result<ffi::AnyBuffer> softmax_lse, // [hq, total_q]
    ffi::Result<ffi::AnyBuffer> p, // [hq, total_q, max_seqlen_k] (dropout mask)
    ffi::Result<ffi::AnyBuffer> rng_state, // [2]
    int64_t max_seqlen_q, int64_t max_seqlen_k, int64_t min_seqlen_q,
    float p_dropout, float softmax_scale, float logits_soft_cap,
    bool zero_tensors, bool is_causal, int64_t window_size_left,
    int64_t window_size_right, bool return_softmax_lse,
    bool return_dropout_randval,
    ffi::AnyBuffer cu_seqlens_k, // [b+1] (optional)
    ffi::AnyBuffer out_provided, // [total_q, hq, d] (optional)
    ffi::AnyBuffer block_table,  // [hq] or [b, hq] (optional)
    ffi::AnyBuffer bias,         // [total_q, max_seqlen_k] (optional)
    ffi::AnyBuffer alibi_slopes, // [hq] or [b, hq] (optional)
    ffi::AnyBuffer gen           // generator (optional)
) {
  JA_LOG("MHA_VARLEN_FWD max_seqlen_q=%ld max_seqlen_k=%ld dropout=%f scale=%f "
         "causal=%d window=(%ld,%ld)",
         max_seqlen_q, max_seqlen_k, p_dropout, softmax_scale, is_causal,
         window_size_left, window_size_right);

  // Get device index for tensor creation
  const int dev_idx = ::jax_aiter::device_from_ptr(q.untyped_data());

  // Create tensor views from the JAX buffers
  auto q_tensor = ::jax_aiter::wrap_any_buffer(q, dev_idx);
  auto k_tensor = ::jax_aiter::wrap_any_buffer(k, dev_idx);
  auto v_tensor = ::jax_aiter::wrap_any_buffer(v, dev_idx);
  auto cu_seqlens_q_tensor =
      ::jax_aiter::wrap_any_buffer(cu_seqlens_q, dev_idx);

  // Create output tensor views for copying results back
  auto out_tensor = ::jax_aiter::wrap_any_buffer(*out, dev_idx);
  auto lse_tensor = ::jax_aiter::wrap_any_buffer(*softmax_lse, dev_idx);
  auto p_tensor = ::jax_aiter::wrap_any_buffer(*p, dev_idx);
  auto rng_tensor = ::jax_aiter::wrap_any_buffer(*rng_state, dev_idx);

  // Handle optional parameters - initialize directly since const optionals
  // can't be assigned
  std::optional<const at::Tensor> cu_seqlens_k_opt =
      cu_seqlens_k.size_bytes() > 0
          ? std::make_optional<const at::Tensor>(
                ::jax_aiter::wrap_any_buffer(cu_seqlens_k, dev_idx))
          : std::nullopt;

  std::optional<at::Tensor> out_provided_opt =
      out_provided.size_bytes() > 0
          ? std::make_optional(
                ::jax_aiter::wrap_any_buffer(out_provided, dev_idx))
          : std::nullopt;

  std::optional<const at::Tensor> block_table_opt =
      block_table.size_bytes() > 0
          ? std::make_optional<const at::Tensor>(
                ::jax_aiter::wrap_any_buffer(block_table, dev_idx))
          : std::nullopt;

  std::optional<const at::Tensor> bias_opt =
      bias.size_bytes() > 0 ? std::make_optional<const at::Tensor>(
                                  ::jax_aiter::wrap_any_buffer(bias, dev_idx))
                            : std::nullopt;

  std::optional<const at::Tensor> alibi_slopes_opt =
      alibi_slopes.size_bytes() > 0
          ? std::make_optional<const at::Tensor>(
                ::jax_aiter::wrap_any_buffer(alibi_slopes, dev_idx))
          : std::nullopt;

  std::optional<at::Generator> gen_opt = std::nullopt;

  try {
    // Call the aiter MHA varlen forward PyTorch kernel with exact signature
    // match
    auto results = aiter::torch_itfs::mha_varlen_fwd(
        q_tensor,                            // q: [total_q, hq, d]
        k_tensor,                            // k: [total_k, hk, d]
        v_tensor,                            // v: [total_k, hk, d]
        cu_seqlens_q_tensor,                 // cu_seqlens_q: [b+1]
        cu_seqlens_k_opt,                    // cu_seqlens_k: [b+1] (optional)
        static_cast<int>(max_seqlen_q),      // max_seqlen_q
        static_cast<int>(max_seqlen_k),      // max_seqlen_k
        static_cast<int>(min_seqlen_q),      // min_seqlen_q
        p_dropout,                           // p_dropout
        softmax_scale,                       // softmax_scale
        logits_soft_cap,                     // logits_soft_cap
        zero_tensors,                        // zero_tensors
        is_causal,                           // is_causal
        static_cast<int>(window_size_left),  // window_size_left
        static_cast<int>(window_size_right), // window_size_right
        return_softmax_lse,                  // return_softmax_lse
        return_dropout_randval,              // return_dropout_randval
        out_provided_opt, // out_ (optional pre-allocated output)
        block_table_opt,  // block_table_ (optional for paged attention)
        bias_opt,         // bias_ (optional)
        alibi_slopes_opt, // alibi_slopes_ (optional)
        gen_opt           // gen_ (optional generator)
    );

    // Copy results back to JAX output buffers
    // results = {out, softmax_lse, p, rng_state}
    if (results.size() >= 4) {
      out_tensor.copy_(results[0]); // output
      lse_tensor.copy_(results[1]); // softmax_lse
      if (return_dropout_randval && results[2].numel() > 0) {
        p_tensor.copy_(results[2]); // dropout mask
      }
      rng_tensor.copy_(results[3]); // rng_state
    }

    JA_LOG("MHA_VARLEN_FWD completed successfully");

    return ffi::Error::Success();
  } catch (const std::exception &e) {
    JA_LOG("MHA_VARLEN_FWD failed: %s", e.what());
    return ffi::Error(ffi::ErrorCode::kInternal, e.what());
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
        .Ret<ffi::AnyBuffer>() // out: [total_q, hq, d]
        .Ret<ffi::AnyBuffer>() // softmax_lse: [hq, total_q]
        .Ret<ffi::AnyBuffer>() // p: [hq, total_q, max_seqlen_k] (dropout mask)
        .Ret<ffi::AnyBuffer>() // rng_state: [2]
        .Attr<int64_t>("max_seqlen_q")
        .Attr<int64_t>("max_seqlen_k")
        .Attr<int64_t>("min_seqlen_q")
        .Attr<float>("p_dropout")
        .Attr<float>("softmax_scale")
        .Attr<float>("logits_soft_cap")
        .Attr<bool>("zero_tensors")
        .Attr<bool>("is_causal")
        .Attr<int64_t>("window_size_left")
        .Attr<int64_t>("window_size_right")
        .Attr<bool>("return_softmax_lse")
        .Attr<bool>("return_dropout_randval")
        .Arg<ffi::AnyBuffer>()  // cu_seqlens_k: [b+1] (optional)
        .Arg<ffi::AnyBuffer>()  // out_provided: [total_q, hq, d] (optional)
        .Arg<ffi::AnyBuffer>()  // block_table: [hq] or [b, hq] (optional)
        .Arg<ffi::AnyBuffer>()  // bias: [total_q, max_seqlen_k] (optional)
        .Arg<ffi::AnyBuffer>()  // alibi_slopes: [hq] or [b, hq] (optional)
        .Arg<ffi::AnyBuffer>(), // gen: generator (optional)
    {xla::ffi::Traits::kCmdBufferCompatible});

#pragma GCC visibility pop
