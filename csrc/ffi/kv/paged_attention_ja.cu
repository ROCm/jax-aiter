// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// M2b: paged decode. Binds the ahead-of-time compiled aiter paged-attention
// kernel through XLA FFI.
//
// Reads the same NHD pool append_kv writes -- [num_pages, tokens_per_page,
// num_kv_heads, head_dim] -- with no conversion between the two. The op is pure:
// it consumes the pools and returns attention output, so the data dependence on
// append_kv's aliased result orders the read after the write.
//
// Why this calls a generated symbol directly instead of aiter's C++ wrapper.
//
// aiter::paged_attention_ragged looks like the natural entry point, but it
// forwards to the generated kernel through an unchecked varargs call whose
// argument order has drifted from the template that generates it: it passes 23
// arguments where the generated entry declares 24, with the pointer block and
// the scalar block transposed, so `scale` lands where kv_indptr belongs. The
// first thing the kernel does with the resulting garbage is fail
// `num_heads % num_kv_heads == 0`. Only aiter's Python path matches the
// template, which is presumably why the drift went unnoticed.
//
// So the handler calls the generated `extern "C"` entry itself. That also means
// none of aiter's dispatch layer is linked, which removes its fmt and openssl
// dependencies.
//
// Those entries are compiled into this module ahead of time. scripts/
// gen_pa_ragged.py renders aiter's own pa_ragged.cpp.jinja once per
// configuration and emits pa_dispatch_generated.h, which Makefile.kv compiles
// alongside this file; JA_PA_KERNEL_LIST below expands into a static table.
//
// This used to dlopen $HOME/.aiter/build/<name>/lib.so, built by a separate
// scripts/prebuild_pa_ragged.py step driving aiter's Python JIT. That was
// aiter's packaging convention rather than anything the kernel required -- the
// generated source holds no kernel logic, only an extern "C" wrapper
// instantiating two templates from pa_ragged.cuh with literal constants. It
// cost a CI prebuild step, a jinja2 install, and a cache directory outside the
// wheel that made the module unshippable; and because aiter links the result
// with a bare `hipcc -shared`, the library carried no RUNPATH and would not
// load at all in a container that had not been hand-taught where ROCm lives.
// Compiling the kernels in makes a missing configuration a link-time fact
// rather than a run-time one.
//
// The signature below is transcribed from csrc/cpp_itfs/pa/pa_ragged.cpp.jinja.
// scripts/gen_pa_ragged.py guards it: it hashes that template's extern "C"
// block and refuses to generate if it no longer matches what this file expects.
// That guard matters more now that the aiter pin moves nightly -- a reordered
// argument would put scalars in pointer slots with nothing failing to compile.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <string>
#include <unordered_map>

#include "pa_dispatch_generated.h"

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace jax_aiter {
namespace {

// Generated entry point, per pa_ragged.cpp.jinja. head_size and block_size are
// baked into the compiled kernel rather than passed.
using PagedAttentionFn = void (*)(
    void *out_ptr, void *workspace_buffer, void *query_ptr, void *key_cache_ptr,
    void *value_cache_ptr, int *kv_indptr_ptr, int *kv_page_indices_ptr,
    int *kv_last_page_lens_ptr, const float *alibi_slopes_ptr,
    const float *q_scale_ptr, const float *k_scale_ptr, const float *v_scale_ptr,
    const float *fp8_out_scale_ptr, float scale, float logits_soft_cap,
    const int num_seqs, const int num_kv_heads, const int num_heads,
    const int max_num_partitions, const int q_stride, const int kv_block_stride,
    const int kv_head_stride, const int kv_seq_stride, void *stream);

// The ahead-of-time compiled entries, keyed by the same md5 configuration name
// jax_aiter.kv.pa_config.func_name computes and the FFI passes as an attribute.
// Built once on first use; the table is read-only afterwards.
const std::unordered_map<std::string, PagedAttentionFn> &KernelTable() {
#define JA_PA_ENTRY(name_str, sym) {name_str, &sym},
  static const std::unordered_map<std::string, PagedAttentionFn> table = {
      JA_PA_KERNEL_LIST(JA_PA_ENTRY)};
#undef JA_PA_ENTRY
  return table;
}

ffi::Error ResolveKernel(const std::string &name, PagedAttentionFn *out) {
  const auto &table = KernelTable();
  auto it = table.find(name);
  if (it == table.end()) {
    std::string known;
    for (const auto &kv : table) {
      known += "\n  " + kv.first;
    }
    return ffi::Error(
        ffi::ErrorCode::kFailedPrecondition,
        "PagedAttentionJA: kernel configuration '" + name +
            "' is not compiled into this module. Add it to default_configs() in "
            "scripts/gen_pa_ragged.py and rebuild "
            "(make -f Makefile.kv ja_kv). Compiled configurations:" + known);
  }

  *out = it->second;
  return ffi::Error::Success();
}

} // namespace

ffi::Error PagedAttention_Bridge(
    hipStream_t stream, ffi::AnyBuffer query, ffi::AnyBuffer k_pool,
    ffi::AnyBuffer v_pool, ffi::AnyBuffer kv_indptr,
    ffi::AnyBuffer kv_page_indices, ffi::AnyBuffer kv_last_page_lens,
    ffi::AnyBuffer k_scale, ffi::AnyBuffer v_scale, ffi::AnyBuffer workspace,
    ffi::Result<ffi::AnyBuffer> out, float scale, float logits_soft_cap,
    int64_t max_num_partitions, std::string_view func_name) {

  auto q_dims = query.dimensions();
  auto pool_dims = k_pool.dimensions();

  if (q_dims.size() != 3) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "PagedAttentionJA: query must be [num_seqs, num_heads, head_dim]");
  }
  if (pool_dims.size() != 4) {
    return ffi::Error(
        ffi::ErrorCode::kInvalidArgument,
        "PagedAttentionJA: pool must be [num_pages, tokens_per_page, num_kv_heads, head_dim]");
  }

  const int64_t num_seqs = q_dims[0];
  const int64_t num_heads = q_dims[1];
  const int64_t head_size = q_dims[2];
  const int64_t tokens_per_page = pool_dims[1];
  const int64_t num_kv_heads = pool_dims[2];

  if (pool_dims[3] != head_size) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "PagedAttentionJA: pool head_dim disagrees with query");
  }
  if (num_kv_heads == 0 || num_heads % num_kv_heads != 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "PagedAttentionJA: num_heads must be a multiple of num_kv_heads");
  }
  for (auto buf : {kv_indptr, kv_page_indices, kv_last_page_lens}) {
    if (buf.element_type() != ffi::DataType::S32) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                        "PagedAttentionJA: page metadata must be int32");
    }
  }
  if (static_cast<int64_t>(kv_indptr.element_count()) != num_seqs + 1) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "PagedAttentionJA: kv_indptr must have num_seqs + 1 entries");
  }

  const int64_t elems = num_seqs * num_heads * max_num_partitions;
  const int64_t need_bytes =
      2 * static_cast<int64_t>(sizeof(float)) * elems +
      static_cast<int64_t>(ffi::ByteWidth(query.element_type())) * elems * head_size;
  if (static_cast<int64_t>(workspace.size_bytes()) < need_bytes) {
    return ffi::Error(
        ffi::ErrorCode::kInvalidArgument,
        "PagedAttentionJA: workspace is " + std::to_string(workspace.size_bytes()) +
            " bytes, need " + std::to_string(need_bytes));
  }

  PagedAttentionFn kernel = nullptr;
  if (auto err = ResolveKernel(std::string(func_name), &kernel); err.failure()) {
    return err;
  }

  // NHD strides, in elements.
  const int q_stride = static_cast<int>(num_heads * head_size);
  const int kv_seq_stride = static_cast<int>(num_kv_heads * head_size);
  const int kv_block_stride = static_cast<int>(tokens_per_page * kv_seq_stride);
  const int kv_head_stride = static_cast<int>(head_size);

  kernel(out->untyped_data(), workspace.untyped_data(), query.untyped_data(),
         k_pool.untyped_data(), v_pool.untyped_data(),
         static_cast<int *>(kv_indptr.untyped_data()),
         static_cast<int *>(kv_page_indices.untyped_data()),
         static_cast<int *>(kv_last_page_lens.untyped_data()),
         /*alibi_slopes_ptr=*/nullptr, /*q_scale_ptr=*/nullptr,
         static_cast<const float *>(k_scale.untyped_data()),
         static_cast<const float *>(v_scale.untyped_data()),
         /*fp8_out_scale_ptr=*/nullptr, scale, logits_soft_cap,
         static_cast<int>(num_seqs), static_cast<int>(num_kv_heads),
         static_cast<int>(num_heads), static_cast<int>(max_num_partitions),
         q_stride, kv_block_stride, kv_head_stride, kv_seq_stride,
         reinterpret_cast<void *>(stream));

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      std::string("PagedAttentionJA: launch failed: ") +
                          hipGetErrorString(err));
  }
  return ffi::Error::Success();
}

} // namespace jax_aiter

#pragma GCC visibility push(default)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PagedAttentionJA, jax_aiter::PagedAttention_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>() // query              [num_seqs, num_heads, head_dim]
        .Arg<ffi::AnyBuffer>() // k_pool             [pages, tpp, kv_heads, head_dim]
        .Arg<ffi::AnyBuffer>() // v_pool
        .Arg<ffi::AnyBuffer>() // kv_indptr          [num_seqs + 1] int32
        .Arg<ffi::AnyBuffer>() // kv_page_indices    [total_pages] int32
        .Arg<ffi::AnyBuffer>() // kv_last_page_lens  [num_seqs] int32
        .Arg<ffi::AnyBuffer>() // k_scale            [1] f32
        .Arg<ffi::AnyBuffer>() // v_scale            [1] f32
        .Arg<ffi::AnyBuffer>() // workspace          scratch
        .Ret<ffi::AnyBuffer>() // out                [num_seqs, num_heads, head_dim]
        .Attr<float>("scale")
        .Attr<float>("logits_soft_cap")
        .Attr<int64_t>("max_num_partitions")
        .Attr<std::string_view>("func_name"));

#pragma GCC visibility pop
