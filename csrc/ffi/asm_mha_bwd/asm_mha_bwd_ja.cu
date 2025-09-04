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
    ffi::AnyBuffer o,    // [batch, seqlen_q, nhead, hdim_v] (from forward pass)
    ffi::AnyBuffer lse,  // [batch, nhead, seqlen_q]
    ffi::Result<ffi::AnyBuffer> dq,        // [batch, seqlen_q, nhead, hdim_q]
    ffi::Result<ffi::AnyBuffer> dk,        // [batch, seqlen_kv, nhead, hdim_q]
    ffi::Result<ffi::AnyBuffer> dv,        // [batch, seqlen_kv, nhead, hdim_v]
    ffi::Result<ffi::AnyBuffer> softmax_d, // [batch, nhead, seqlen_q]
    float dropout_p, float softmax_scale, bool is_causal,
    int64_t window_size_left, int64_t window_size_right, bool deterministic,
    bool is_v3_atomic_fp32, int64_t how_v3_bf16_cvt, ffi::AnyBuffer dq_provided,
    ffi::AnyBuffer dk_provided, ffi::AnyBuffer dv_provided,
    ffi::AnyBuffer alibi_slopes, ffi::AnyBuffer rng_state_provided,
    ffi::AnyBuffer gen) {
  JA_LOG("FMHA_V3_BWD dropout=%f scale=%f causal=%d window=(%ld,%ld) "
         "deterministic=%d atomic_fp32=%d bf16_cvt=%ld",
         dropout_p, softmax_scale, is_causal, window_size_left,
         window_size_right, deterministic, is_v3_atomic_fp32, how_v3_bf16_cvt);

  // Get device index for tensor creation.
  const int dev_idx = ::jax_aiter::device_from_ptr(q.untyped_data());

  // Create tensor views from the JAX buffers using torch_utils helpers.
  auto dout_tensor = ::jax_aiter::wrap_any_buffer(dout, dev_idx);
  auto q_tensor = ::jax_aiter::wrap_any_buffer(q, dev_idx);
  auto k_tensor = ::jax_aiter::wrap_any_buffer(k, dev_idx);
  auto v_tensor = ::jax_aiter::wrap_any_buffer(v, dev_idx);
  auto o_tensor = ::jax_aiter::wrap_any_buffer(o, dev_idx);
  auto lse_tensor = ::jax_aiter::wrap_any_buffer(lse, dev_idx);

  // Output tensors.
  auto dq_tensor = ::jax_aiter::wrap_any_buffer(*dq, dev_idx);
  auto dk_tensor = ::jax_aiter::wrap_any_buffer(*dk, dev_idx);
  auto dv_tensor = ::jax_aiter::wrap_any_buffer(*dv, dev_idx);
  auto softmax_d_tensor = ::jax_aiter::wrap_any_buffer(*softmax_d, dev_idx);

  // Handle optional parameters (check for None by buffer size).
  std::optional<at::Tensor> dq_provided_opt = std::nullopt;
  if (dq_provided.size_bytes() > 0) {
    dq_provided_opt =
        std::make_optional(::jax_aiter::wrap_any_buffer(dq_provided, dev_idx));
  }

  std::optional<at::Tensor> dk_provided_opt = std::nullopt;
  if (dk_provided.size_bytes() > 0) {
    dk_provided_opt =
        std::make_optional(::jax_aiter::wrap_any_buffer(dk_provided, dev_idx));
  }

  std::optional<at::Tensor> dv_provided_opt = std::nullopt;
  if (dv_provided.size_bytes() > 0) {
    dv_provided_opt =
        std::make_optional(::jax_aiter::wrap_any_buffer(dv_provided, dev_idx));
  }

  std::optional<at::Tensor> alibi_slopes_opt = std::nullopt;
  if (alibi_slopes.size_bytes() > 0) {
    alibi_slopes_opt =
        std::make_optional(::jax_aiter::wrap_any_buffer(alibi_slopes, dev_idx));
  }

  std::optional<at::Tensor> rng_state_opt = std::nullopt;
  if (rng_state_provided.size_bytes() > 0) {
    rng_state_opt = std::make_optional(
        ::jax_aiter::wrap_any_buffer(rng_state_provided, dev_idx));
  }

  // Generator is always None for now.
  std::optional<at::Generator> gen_opt = std::nullopt;

  try {
    // Call the aiter FMHA v3 backward PyTorch kernel with exact signature match.
    auto results = aiter::torch_itfs::fmha_v3_bwd(
        dout_tensor,                         // dout: [b, sq, hq, d_v]
        q_tensor,                            // q: [b, sq, hq, d]
        k_tensor,                            // k: [b, sk, hk, d]
        v_tensor,                            // v: [b, sk, hk, d_v]
        o_tensor,                            // out: [b, sq, hq, d_v]
        lse_tensor,                          // softmax_lse: [b, hq, sq]
        dropout_p,                           // p_dropout
        softmax_scale,                       // softmax_scale
        is_causal,                           // is_causal
        static_cast<int>(window_size_left),  // window_size_left
        static_cast<int>(window_size_right), // window_size_right
        deterministic,                       // deterministic
        is_v3_atomic_fp32,                   // is_v3_atomic_fp32
        static_cast<int>(how_v3_bf16_cvt),   // how_v3_bf16_cvt
        dq_provided_opt,                     // dq_ (optional pre-allocated)
        dk_provided_opt,                     // dk_ (optional pre-allocated)
        dv_provided_opt,                     // dv_ (optional pre-allocated)
        alibi_slopes_opt,                    // alibi_slopes_ (optional)
        rng_state_opt,                       // rng_state_ (optional)
        gen_opt                              // gen_ (optional generator)
    );

    // Copy results back to JAX output buffers.
    // results = {dq, dk, dv, softmax_d}
    if (results.size() >= 4) {
      dq_tensor.copy_(results[0]);        // dq
      dk_tensor.copy_(results[1]);        // dk
      dv_tensor.copy_(results[2]);        // dv
      softmax_d_tensor.copy_(results[3]); // softmax_d
    }

    JA_LOG("FMHA_V3_BWD completed successfully");

    return ffi::Error::Success();
  } catch (const std::exception &e) {
    JA_LOG("FMHA_V3_BWD failed: %s", e.what());
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
        .Ret<ffi::AnyBuffer>() // dq: [batch, seqlen_q, nhead, hdim_q]
        .Ret<ffi::AnyBuffer>() // dk: [batch, seqlen_kv, nhead, hdim_q]
        .Ret<ffi::AnyBuffer>() // dv: [batch, seqlen_kv, nhead, hdim_v]
        .Ret<ffi::AnyBuffer>() // softmax_d: [batch, nhead, seqlen_q]
        .Attr<float>("dropout_p")
        .Attr<float>("softmax_scale")
        .Attr<bool>("is_causal")
        .Attr<int64_t>("window_size_left")
        .Attr<int64_t>("window_size_right")
        .Attr<bool>("deterministic")
        .Attr<bool>("is_v3_atomic_fp32")
        .Attr<int64_t>("how_v3_bf16_cvt")
        .Arg<ffi::AnyBuffer>()  // dq_provided (optional)
        .Arg<ffi::AnyBuffer>()  // dk_provided (optional)
        .Arg<ffi::AnyBuffer>()  // dv_provided (optional)
        .Arg<ffi::AnyBuffer>()  // alibi_slopes (optional)
        .Arg<ffi::AnyBuffer>()  // rng_state_provided (optional)
        .Arg<ffi::AnyBuffer>(), // gen (optional)
    {xla::ffi::Traits::kCmdBufferCompatible});

#pragma GCC visibility pop
