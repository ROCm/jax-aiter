// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// M0 aliasing probe: the smallest op that mutates a donated buffer in place
// through XLA FFI. It exists only to prove the input_output_aliases contract on
// ROCm before any paged-KV work is built on top of it, and does not survive into
// production.
//
// The pool is bound as both .Arg and .Ret. Under input_output_aliases={0: 0} the
// two must resolve to the same device pointer, so the handler refuses to run if
// they differ rather than silently writing into a replacement buffer -- an
// unhonoured alias would otherwise present as a correctness bug much later.
//
// Shape contract, chosen to mirror what append_kv does at M1:
//   pool     [pool_rows, ...]  row_elems = product of trailing dims
//   row_idx  [n_rows]          int32, negative or >= pool_rows means "skip"
//   vals     [n_rows, ...]     same trailing dims as pool
// Effect: pool[row_idx[i]] += vals[i], skipping sentinel rows.

#include <hip/hip_runtime.h>

#include <cstdint>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace jax_aiter {
namespace {

// bf16 <-> f32 done by hand so the probe needs no ROCm math headers, whose
// spelling of the bf16 type has moved between releases.
__device__ inline float Bf16ToF32(uint16_t h) {
  uint32_t u = static_cast<uint32_t>(h) << 16;
  float f;
  __builtin_memcpy(&f, &u, sizeof(f));
  return f;
}

__device__ inline uint16_t F32ToBf16(float f) {
  uint32_t u;
  __builtin_memcpy(&u, &f, sizeof(u));
  // Round to nearest even on the truncated mantissa.
  uint32_t lsb = (u >> 16) & 1u;
  u += 0x7fffu + lsb;
  return static_cast<uint16_t>(u >> 16);
}

struct F32Access {
  using Storage = float;
  static __device__ inline float Load(const Storage *p, int64_t i) { return p[i]; }
  static __device__ inline void Store(Storage *p, int64_t i, float v) { p[i] = v; }
};

struct Bf16Access {
  using Storage = uint16_t;
  static __device__ inline float Load(const Storage *p, int64_t i) {
    return Bf16ToF32(p[i]);
  }
  static __device__ inline void Store(Storage *p, int64_t i, float v) {
    p[i] = F32ToBf16(v);
  }
};

template <typename Access>
__global__ void ScatterAddRowsKernel(typename Access::Storage *pool,
                                     const int32_t *row_idx,
                                     const typename Access::Storage *vals,
                                     int64_t n_rows, int64_t row_elems,
                                     int64_t pool_rows) {
  int64_t i = static_cast<int64_t>(blockIdx.x);
  if (i >= n_rows) return;

  int32_t dst = row_idx[i];
  if (dst < 0 || static_cast<int64_t>(dst) >= pool_rows) return;

  const typename Access::Storage *src = vals + i * row_elems;
  typename Access::Storage *out = pool + static_cast<int64_t>(dst) * row_elems;

  for (int64_t j = threadIdx.x; j < row_elems; j += blockDim.x) {
    Access::Store(out, j, Access::Load(out, j) + Access::Load(src, j));
  }
}

int64_t TrailingElems(ffi::Span<const int64_t> dims) {
  int64_t n = 1;
  for (size_t i = 1; i < dims.size(); ++i) n *= dims[i];
  return n;
}

} // namespace

ffi::Error KvAliasProbe_Bridge(hipStream_t stream, ffi::AnyBuffer pool_in,
                               ffi::AnyBuffer row_idx, ffi::AnyBuffer vals,
                               ffi::Result<ffi::AnyBuffer> pool_out) {
  void *in_ptr = pool_in.untyped_data();
  void *out_ptr = pool_out->untyped_data();

  if (!in_ptr || !out_ptr || !row_idx.untyped_data() || !vals.untyped_data()) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "KvAliasProbeJA: null buffer");
  }

  // The whole point of M0. If XLA did not honour the alias, the .Ret is a
  // different allocation and an in-place write would be lost, so fail loudly.
  if (in_ptr != out_ptr) {
    return ffi::Error(
        ffi::ErrorCode::kFailedPrecondition,
        "KvAliasProbeJA: input_output_aliases was not honoured -- the pool "
        "resolved to different device pointers as .Arg and .Ret, so in-place "
        "KV mutation is unsafe on this build");
  }

  auto pool_dims = pool_in.dimensions();
  auto vals_dims = vals.dimensions();
  if (pool_dims.size() < 1 || vals_dims.size() < 1) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "KvAliasProbeJA: pool and vals must be rank >= 1");
  }

  if (pool_in.element_type() != vals.element_type()) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "KvAliasProbeJA: pool and vals dtype must match");
  }
  if (row_idx.element_type() != ffi::DataType::S32) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "KvAliasProbeJA: row_idx must be int32");
  }

  const int64_t pool_rows = pool_dims[0];
  const int64_t row_elems = TrailingElems(pool_dims);
  const int64_t n_rows = vals_dims[0];

  if (TrailingElems(vals_dims) != row_elems) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "KvAliasProbeJA: vals trailing dims must match pool");
  }
  if (static_cast<int64_t>(row_idx.element_count()) != n_rows) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "KvAliasProbeJA: row_idx length must match vals rows");
  }
  if (n_rows == 0 || row_elems == 0) {
    return ffi::Error::Success();
  }

  const int threads = static_cast<int>(row_elems < 256 ? row_elems : 256);
  const dim3 grid(static_cast<unsigned>(n_rows));
  const dim3 block(static_cast<unsigned>(threads));
  const auto *idx = static_cast<const int32_t *>(row_idx.untyped_data());

  switch (pool_in.element_type()) {
  case ffi::DataType::F32:
    hipLaunchKernelGGL(
        (ScatterAddRowsKernel<F32Access>), grid, block, 0, stream,
        static_cast<float *>(out_ptr), idx,
        static_cast<const float *>(vals.untyped_data()), n_rows, row_elems,
        pool_rows);
    break;
  case ffi::DataType::BF16:
    hipLaunchKernelGGL(
        (ScatterAddRowsKernel<Bf16Access>), grid, block, 0, stream,
        static_cast<uint16_t *>(out_ptr), idx,
        static_cast<const uint16_t *>(vals.untyped_data()), n_rows, row_elems,
        pool_rows);
    break;
  default:
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "KvAliasProbeJA: only f32 and bf16 pools supported");
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      std::string("KvAliasProbeJA: kernel launch failed: ") +
                          hipGetErrorString(err));
  }

  return ffi::Error::Success();
}

} // namespace jax_aiter

#pragma GCC visibility push(default)

// No kCmdBufferCompatible trait here on purpose: HIP-graph capture would be one
// more variable in the M0 profile analysis, and the probe is throwaway.
XLA_FFI_DEFINE_HANDLER_SYMBOL(KvAliasProbeJA, jax_aiter::KvAliasProbe_Bridge,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<hipStream_t>>()
                                  .Arg<ffi::AnyBuffer>() // pool  (aliased)
                                  .Arg<ffi::AnyBuffer>() // row_idx int32
                                  .Arg<ffi::AnyBuffer>() // vals
                                  .Ret<ffi::AnyBuffer>() // pool' (aliases pool)
);

#pragma GCC visibility pop
