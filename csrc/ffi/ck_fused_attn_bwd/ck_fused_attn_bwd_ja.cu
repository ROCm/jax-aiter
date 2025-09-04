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
    ffi::AnyBuffer dout, // [batch, seqlen_q, nhead, hdim_v]
    ffi::AnyBuffer q,    // [batch, seqlen_q, nhead, hdim_q]
    ffi::AnyBuffer k,    // [batch, seqlen_kv, nhead, hdim_q]
    ffi::AnyBuffer v,    // [batch, seqlen_kv, nhead, hdim_v]
    ffi::AnyBuffer o,    // [batch, seqlen_q, nhead, hdim_v] (from forward pass)
    ffi::AnyBuffer lse,  // [batch, nhead, seqlen_q]
    ffi::Result<ffi::AnyBuffer> dq, // [batch, seqlen_q, nhead, hdim_q]
    ffi::Result<ffi::AnyBuffer> dk, // [batch, seqlen_kv, nhead, hdim_q]
    ffi::Result<ffi::AnyBuffer> dv, // [batch, seqlen_kv, nhead, hdim_v]
    int64_t batch, int64_t nhead, int64_t seqlen_q, int64_t seqlen_kv,
    int64_t hdim_q, int64_t hdim_v, float scale, float dropout_prob) {
  JA_LOG("MHA_BWD batch=%ld nhead=%ld seqlen_q=%ld seqlen_kv=%ld hdim_q=%ld "
         "hdim_v=%ld scale=%f dropout=%f",
         batch, nhead, seqlen_q, seqlen_kv, hdim_q, hdim_v, scale,
         dropout_prob);

  // Get device index for tensor creation
  const int dev_idx = ::jax_aiter::device_from_ptr(q.untyped_data());

  // Create tensor views from the JAX buffers using torch_utils helpers
  // Layout: [batch, seqlen, nhead, hdim]
  auto dout_tensor = ::jax_aiter::wrap_any_buffer(
      dout, {batch, seqlen_q, nhead, hdim_v}, dev_idx);

  auto q_tensor = ::jax_aiter::wrap_any_buffer(
      q, {batch, seqlen_q, nhead, hdim_q}, dev_idx);

  auto k_tensor = ::jax_aiter::wrap_any_buffer(
      k, {batch, seqlen_kv, nhead, hdim_q}, dev_idx);

  auto v_tensor = ::jax_aiter::wrap_any_buffer(
      v, {batch, seqlen_kv, nhead, hdim_v}, dev_idx);

  auto o_tensor = ::jax_aiter::wrap_any_buffer(
      o, {batch, seqlen_q, nhead, hdim_v}, dev_idx);

  // LSE tensor: [batch, nhead, seqlen_q]
  auto lse_tensor =
      ::jax_aiter::wrap_any_buffer(lse, {batch, nhead, seqlen_q}, dev_idx);

  // Output tensors
  auto dq_tensor = ::jax_aiter::wrap_any_buffer(
      *dq, {batch, seqlen_q, nhead, hdim_q}, dev_idx);

  auto dk_tensor = ::jax_aiter::wrap_any_buffer(
      *dk, {batch, seqlen_kv, nhead, hdim_q}, dev_idx);

  auto dv_tensor = ::jax_aiter::wrap_any_buffer(
      *dv, {batch, seqlen_kv, nhead, hdim_v}, dev_idx);

  try {
    // Call the aiter PyTorch kernel
    auto results =
        aiter::torch_itfs::mha_bwd(dout_tensor,  // dout: [b, sq, hq, d_v]
                                   q_tensor,     // q: [b, sq, hq, d]
                                   k_tensor,     // k: [b, sk, hk, d]
                                   v_tensor,     // v: [b, sk, hk, d_v]
                                   o_tensor,     // out: [b, sq, hq, d_v]
                                   lse_tensor,   // softmax_lse: [b, hq, sq]
                                   dropout_prob, // p_dropout
                                   scale,        // softmax_scale
                                   false,        // is_causal
                                   -1,    // window_size_left (-1 = infinite)
                                   -1,    // window_size_right (-1 = infinite)
                                   false, // deterministic
                                   std::make_optional(dq_tensor), // dq_
                                   std::make_optional(dk_tensor), // dk_
                                   std::make_optional(dv_tensor), // dv_
                                   std::nullopt,                  // dbias_
                                   std::nullopt,                  // bias_
                                   std::nullopt, // alibi_slopes_
                                   std::nullopt, // rng_state_
                                   std::nullopt  // gen_
        );

    // The function should populate dq_tensor, dk_tensor, dv_tensor in-place
    // results[0] = dq, results[1] = dk, results[2] = dv, results[3] = softmax_d

    JA_LOG("MHA_BWD completed successfully");

    return ffi::Error::Success();
  } catch (const std::exception &e) {
    JA_LOG("MHA_BWD failed: %s", e.what());
    return ffi::Error(ffi::ErrorCode::kInternal, e.what());
  }
}

} // namespace jax_aiter

#pragma GCC visibility push(default)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MhaBwdJA, jax_aiter::MhaBwd_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>() // dout: [batch, seqlen_q, nhead, hdim_v]
        .Arg<ffi::AnyBuffer>() // q: [batch, seqlen_q, nhead, hdim_q]
        .Arg<ffi::AnyBuffer>() // k: [batch, seqlen_kv, nhead, hdim_q]
        .Arg<ffi::AnyBuffer>() // v: [batch, seqlen_kv, nhead, hdim_v]
        .Arg<ffi::AnyBuffer>() // o: [batch, seqlen_q, nhead, hdim_v]
        .Arg<ffi::AnyBuffer>() // lse: [batch, nhead, seqlen_q]
        .Ret<ffi::AnyBuffer>() // dq: [batch, seqlen_q, nhead, hdim_q]
        .Ret<ffi::AnyBuffer>() // dk: [batch, seqlen_kv, nhead, hdim_q]
        .Ret<ffi::AnyBuffer>() // dv: [batch, seqlen_kv, nhead, hdim_v]
        .Attr<int64_t>("batch")
        .Attr<int64_t>("nhead")
        .Attr<int64_t>("seqlen_q")
        .Attr<int64_t>("seqlen_kv")
        .Attr<int64_t>("hdim_q")
        .Attr<int64_t>("hdim_v")
        .Attr<float>("scale")
        .Attr<float>("dropout_prob"),
    {xla::ffi::Traits::kCmdBufferCompatible});

#pragma GCC visibility pop
