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

// Forward declare the exact aiter torch interface.
// returns {out, softmax_lse, p, rng_state};
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
    ffi::AnyBuffer q,                           // [total_q, hq, d]
    ffi::AnyBuffer k,                           // [total_k, hk, d]
    ffi::AnyBuffer v,                           // [total_k, hk, d]
    ffi::AnyBuffer cu_seqlens_q,                // [b+1]
    std::optional<ffi::AnyBuffer> cu_seqlens_k, // [b+1]
    std::optional<ffi::AnyBuffer> out_provided, // [total_q, hq, d] (optional)
    std::optional<ffi::AnyBuffer> block_table,  // [hq] or [b, hq] (optional)
    std::optional<ffi::AnyBuffer> bias, // [total_q, max_seqlen_k] (optional)
    std::optional<ffi::AnyBuffer> alibi_slopes, // [hq] or [b, hq] (optional)
    std::optional<ffi::AnyBuffer> gen,          // generator (optional)
    ffi::Result<ffi::AnyBuffer> out,            // [total_q, hq, d]
    ffi::Result<ffi::AnyBuffer> softmax_lse,    // [hq, total_q]
    ffi::Result<ffi::AnyBuffer> p, // [hq, total_q, max_seqlen_k] (dropout mask)
    ffi::Result<ffi::AnyBuffer> rng_state, // [2]
    int max_seqlen_q, int max_seqlen_k, int min_seqlen_q, float p_dropout,
    float softmax_scale, float logits_soft_cap, bool zero_tensors,
    bool is_causal, int window_size_left, int window_size_right,
    bool return_softmax_lse, bool return_dropout_randval) {
  // Get device index for tensor creation.
  const int dev_idx = ::jax_aiter::device_from_ptr(q.untyped_data());

  // Create tensor views from the JAX buffers.
  auto q_tensor = ::jax_aiter::wrap_any_buffer(q, dev_idx);
  auto k_tensor = ::jax_aiter::wrap_any_buffer(k, dev_idx);
  auto v_tensor = ::jax_aiter::wrap_any_buffer(v, dev_idx);
  auto cu_seqlens_q_tensor =
      ::jax_aiter::wrap_any_buffer(cu_seqlens_q, dev_idx);

  const c10::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(
      device_of(q_tensor));

  const c10::hip::HIPStreamMasqueradingAsCUDA ext_stream =
      c10::hip::getStreamFromExternalMasqueradingAsCUDA(stream, dev_idx);
  const c10::hip::HIPStreamGuardMasqueradingAsCUDA stream_guard{ext_stream};

  // Handle optional parameters (check for None by buffer size).
  std::optional<const at::Tensor> cu_seqlens_k_opt =
      (cu_seqlens_k.has_value() && cu_seqlens_k->size_bytes() > 0)
          ? std::make_optional<const at::Tensor>(
                ::jax_aiter::wrap_any_buffer(*cu_seqlens_k, dev_idx))
          : std::nullopt;

  std::optional<at::Tensor> out_provided_opt =
      (out_provided.has_value() && out_provided->size_bytes() > 0)
          ? std::make_optional(
                ::jax_aiter::wrap_any_buffer(*out_provided, dev_idx))
          : std::nullopt;

  std::optional<const at::Tensor> block_table_opt =
      (block_table.has_value() && block_table->size_bytes() > 0)
          ? std::make_optional<const at::Tensor>(
                ::jax_aiter::wrap_any_buffer(*block_table, dev_idx))
          : std::nullopt;

  std::optional<const at::Tensor> bias_opt =
      (bias.has_value() && bias->size_bytes() > 0)
          ? std::make_optional<const at::Tensor>(
                ::jax_aiter::wrap_any_buffer(*bias, dev_idx))
          : std::nullopt;

  std::optional<const at::Tensor> alibi_slopes_opt =
      (alibi_slopes.has_value() && alibi_slopes->size_bytes() > 0)
          ? std::make_optional<const at::Tensor>(
                ::jax_aiter::wrap_any_buffer(*alibi_slopes, dev_idx))
          : std::nullopt;

  std::optional<at::Generator> gen_opt = std::nullopt;
  // Handle generator parameter - extract from JAX buffer if provided.
  if (gen.has_value() &&
      gen->size_bytes() >= static_cast<size_t>(2 * sizeof(int64_t))) {
    // Extract seed and offset from JAX buffer.
    const auto *gen_data = static_cast<const int64_t *>(gen->untyped_data());
    const uint64_t seed = static_cast<uint64_t>(gen_data[0]);
    const uint64_t offset = static_cast<uint64_t>(gen_data[1]);

    // Create PyTorch generator with the provided seed.
    auto gen_torch = at::make_generator<at::CUDAGeneratorImpl>(dev_idx);
    auto *impl = gen_torch.get<at::CUDAGeneratorImpl>();
    impl->set_current_seed(seed);
    impl->set_offset(offset);
    gen_opt = gen_torch;
  }

  std::vector<at::Tensor> results;
  try {
    results = aiter::torch_itfs::mha_varlen_fwd(
        q_tensor,               // q: [total_q, hq, d]
        k_tensor,               // k: [total_k, hk, d]
        v_tensor,               // v: [total_k, hk, d]
        cu_seqlens_q_tensor,    // cu_seqlens_q: [b+1]
        cu_seqlens_k_opt,       // cu_seqlens_k: [b+1] (optional)
        max_seqlen_q,           // max_seqlen_q
        max_seqlen_k,           // max_seqlen_k
        min_seqlen_q,           // min_seqlen_q
        p_dropout,              // p_dropout
        softmax_scale,          // softmax_scale
        logits_soft_cap,        // logits_soft_cap
        zero_tensors,           // zero_tensors
        is_causal,              // is_causal
        window_size_left,       // window_size_left
        window_size_right,      // window_size_right
        return_softmax_lse,     // return_softmax_lse
        return_dropout_randval, // return_dropout_randval
        out_provided_opt,       // out_ (optional pre-allocated output)
        block_table_opt,        // block_table_ (optional for paged attention)
        bias_opt,               // bias_ (optional)
        alibi_slopes_opt,       // alibi_slopes_ (optional)
        gen_opt                 // gen_ (optional generator)
    );
  } catch (const c10::Error &e) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      std::string("mha_varlen_fwd PyTorch error: ") +
                          e.what_without_backtrace());
  } catch (const std::exception &e) {
    const char *what_msg = e.what();
    std::fprintf(stderr,
                 "[mha_varlen_fwd] Exception caught - type: %s, message: %s\n",
                 typeid(e).name(), what_msg ? what_msg : "null");
    std::fflush(stderr);
    return ffi::Error(ffi::ErrorCode::kInternal,
                      std::string("mha_varlen_fwd: ") +
                          (what_msg ? what_msg : "unknown error"));
  }

  if (results.size() < 4) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      "Kernel returned insufficient results");
  }

  // Copy results back to JAX output buffers
  // results = {out, softmax_lse, p, rng_state}.
  try {
    // Create output tensor views for copying results back.
    auto out_tensor = ::jax_aiter::wrap_any_buffer(*out, dev_idx);

    // Always copy main output
    out_tensor.copy_(results[0], /*non_blocking=*/true);

    // Only copy LSE if requested (and if result has valid data)
    if (return_softmax_lse && results[1].numel() > 0) {
      auto lse_tensor = ::jax_aiter::wrap_any_buffer(*softmax_lse, dev_idx);
      lse_tensor.copy_(results[1], /*non_blocking=*/true);
    }

    // Only copy dropout mask if requested and valid
    if (return_dropout_randval && results[2].numel() > 0) {
      auto p_tensor = ::jax_aiter::wrap_any_buffer(*p, dev_idx);
      p_tensor.copy_(results[2], /*non_blocking=*/true);
    }

    // Copy RNG state if valid
    if (results[3].numel() > 0) {
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
    MhaVarlenFwdJA, jax_aiter::MhaVarlenFwd_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>() // q: [total_q, hq, d]
        .Arg<ffi::AnyBuffer>() // k: [total_k, hk, d]
        .Arg<ffi::AnyBuffer>() // v: [total_k, hk, d]
        .Arg<ffi::AnyBuffer>() // cu_seqlens_q: [b+1]
        .Arg<ffi::AnyBuffer>() // cu_seqlens_k: [b+1]
        .Arg<ffi::AnyBuffer>() // out_provided: [total_q, hq, d] (optional)
        .Arg<ffi::AnyBuffer>() // block_table: [hq] or [b, hq] (optional)
        .Arg<ffi::AnyBuffer>() // bias: [total_q, max_seqlen_k] (optional)
        .Arg<ffi::AnyBuffer>() // alibi_slopes: [hq] or [b, hq] (optional)
        .Arg<ffi::AnyBuffer>() // gen: generator (optional)
        .Ret<ffi::AnyBuffer>() // out: [total_q, hq, d]
        .Ret<ffi::AnyBuffer>() // softmax_lse: [hq, total_q]
        .Ret<ffi::AnyBuffer>() // p: [hq, total_q, max_seqlen_k] (dropout mask)
        .Ret<ffi::AnyBuffer>() // rng_state: [2]
        .Attr<int>("max_seqlen_q")
        .Attr<int>("max_seqlen_k")
        .Attr<int>("min_seqlen_q")
        .Attr<float>("p_dropout")
        .Attr<float>("softmax_scale")
        .Attr<float>("logits_soft_cap")
        .Attr<bool>("zero_tensors")
        .Attr<bool>("is_causal")
        .Attr<int>("window_size_left")
        .Attr<int>("window_size_right")
        .Attr<bool>("return_softmax_lse")
        .Attr<bool>("return_dropout_randval"),
    {xla::ffi::Traits::kCmdBufferCompatible});

#pragma GCC visibility pop
