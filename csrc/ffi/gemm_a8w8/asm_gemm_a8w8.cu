#include <hip/hip_runtime.h>
#include <torch/all.h>

#include "asm_gemm_a8w8.h"
#include "xla/ffi/api/ffi.h"

#include "hip_utils.h"
#include "logging.h"
#include "torch_utils.h"

namespace ffi = xla::ffi;

namespace jax_aiter {
ffi::Error
GemmA8W8_Bridge(hipStream_t stream, ffi::Buffer<ffi::S8> A,
                ffi::Buffer<ffi::S8> B, ffi::Buffer<ffi::F32> A_scale,
                ffi::Buffer<ffi::F32> B_scale, ffi::ResultBuffer<ffi::BF16> out,
                ffi::Buffer<ffi::F32> bias, int32_t sub_m = 128,
                int32_t sub_n = 128, int32_t pad_a = 0, int32_t pad_b = 0,
                int32_t pad_c = 0, int32_t splitK = 0) {
  int64_t m = A.dimensions()[0];
  int64_t k = A.dimensions()[1];
  int64_t n = out->dimensions()[1];

  JA_LOG("GEMM (%ld*%ld) . (%ld*%ld) -> (%ld*%ld)  splitK=%d", m, k, k, n, m, n,
         splitK);

  // Be precise and use the "current_device".
  const int dev_idx = ::jax_aiter::device_from_ptr(A.untyped_data());

  // Wrap JAX buffers into torch Tensors.
  auto A_t = wrap_buffer<int8_t>(A.untyped_data(), {m, k}, {k, 1}, dev_idx);
  auto B_t = wrap_buffer<int8_t>(B.untyped_data(), {n, k}, {k, 1}, dev_idx);
  auto A_scale_t =
      wrap_buffer<float>(A_scale.untyped_data(), {m, 1}, {1, 1}, dev_idx);
  auto B_scale_t =
      wrap_buffer<float>(B_scale.untyped_data(), {1, n}, {n, 1}, dev_idx);
  auto out_t =
      wrap_buffer<at::BFloat16>(out->untyped_data(), {m, n}, {n, 1}, dev_idx);
  auto bias_t =
      wrap_buffer<float>(bias.untyped_data(), {1, n}, {n, 1}, dev_idx);

  gemm_a8w8_asm(A_t, B_t, A_scale_t, B_scale_t, out_t, bias_t, sub_m, sub_n,
                pad_a, pad_b, pad_c, splitK);

  return ffi::Error::Success();
}
} // namespace jax_aiter

#pragma GCC visibility push(default)
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    GemmA8W8, jax_aiter::GemmA8W8_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::Buffer<ffi::S8>>()   // A: [m, k]
        .Arg<ffi::Buffer<ffi::S8>>()   // B: [n, k]
        .Arg<ffi::Buffer<ffi::F32>>()  // A_scale: [m, 1]
        .Arg<ffi::Buffer<ffi::F32>>()  // B_scale: [1, n]
        .Ret<ffi::Buffer<ffi::BF16>>() // out: [m, n]
        .Arg<ffi::Buffer<ffi::F32>>()  // bias: [1, n]
        .Attr<int32_t>("sub_m")
        .Attr<int32_t>("sub_n")
        .Attr<int32_t>("pad_a")
        .Attr<int32_t>("pad_b")
        .Attr<int32_t>("pad_c")
        .Attr<int32_t>("splitK"),
    {xla::ffi::Traits::kCmdBufferCompatible});
#pragma GCC visibility pop
