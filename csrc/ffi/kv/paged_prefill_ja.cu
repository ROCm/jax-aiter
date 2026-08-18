// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// M2c: paged prefill. Binds aiter's batch-prefill attention through XLA FFI so a
// ragged batch of new tokens can attend over the same paged KV pool that
// append_kv writes and paged decode reads.
//
// Layout and metadata are deliberately identical to M2b. The pool stays NHD --
// [num_pages, tokens_per_page, num_kv_heads, head_dim] -- and the page table is
// the same (kv_indptr, kv_page_indices, kv_last_page_lens) triple, so prefill and
// decode share one control plane and one pool with no repacking between phases.
// In ck_tile's vocabulary that is LINEAR_LAYOUT plus SGLANG_PAGE_TABLE_1D; the
// alternatives (a swizzled 5D pool, or a 2D vLLM block table) would each force a
// conversion somewhere, so neither is used.
//
// Unlike paged decode this calls aiter's C++ entry point directly rather than
// dlopening a generated symbol. aiter::mha_batch_prefill is a real typed
// function whose signature the compiler checks, and it does not shell out: the
// kernels are generated and compiled into this module ahead of time by the
// batch_prefill codegen in Makefile.kv. The reason M2b had to avoid its
// equivalent -- a varargs call with drifted argument order -- does not apply
// here.
//
// Queries are ragged, described by cu_seqlens_q rather than padded to a
// rectangle, which is what "group mode" means to ck_tile. Sequence i owns
// q[cu_seqlens_q[i] : cu_seqlens_q[i+1]] and attends over its own pages. Because
// the query tokens for a step are usually already in the pool, the causal mask is
// bottom-right aligned: query token j of a sequence sees key positions up to
// (seqlen_k - seqlen_q + j), which is the alignment create_mask_info encodes with
// its "b:" prefix and the convention the rest of this repo's MHA shims use.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <string>
#include <utility>

#include "mask.hpp"
#include "mha_common_utils.h"
#include "mha_fwd.h"

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace jax_aiter {

ffi::Error PagedPrefill_Bridge(
    hipStream_t stream, ffi::AnyBuffer query, ffi::AnyBuffer k_pool,
    ffi::AnyBuffer v_pool, ffi::AnyBuffer cu_seqlens_q, ffi::AnyBuffer kv_indptr,
    ffi::AnyBuffer kv_page_indices, ffi::AnyBuffer kv_last_page_lens,
    ffi::Result<ffi::AnyBuffer> out, float scale, float logits_soft_cap,
    int64_t max_seqlen_q, int64_t max_seqlen_k, bool causal,
    int64_t window_size_left, int64_t window_size_right) {

  auto q_dims = query.dimensions();
  auto pool_dims = k_pool.dimensions();

  if (q_dims.size() != 3) {
    return ffi::Error(
        ffi::ErrorCode::kInvalidArgument,
        "PagedPrefillJA: query must be [total_q, num_heads, head_dim]");
  }
  if (pool_dims.size() != 4) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "PagedPrefillJA: pool must be [num_pages, "
                      "tokens_per_page, num_kv_heads, head_dim]");
  }

  const int64_t total_q = q_dims[0];
  const int64_t num_heads = q_dims[1];
  const int64_t head_dim = q_dims[2];
  const int64_t num_pages = pool_dims[0];
  const int64_t tokens_per_page = pool_dims[1];
  const int64_t num_kv_heads = pool_dims[2];

  if (pool_dims[3] != head_dim) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "PagedPrefillJA: pool head_dim disagrees with query");
  }
  if (num_kv_heads == 0 || num_heads % num_kv_heads != 0) {
    return ffi::Error(
        ffi::ErrorCode::kInvalidArgument,
        "PagedPrefillJA: num_heads must be a multiple of num_kv_heads");
  }

  const auto q_dtype = query.element_type();
  if (q_dtype != ffi::DataType::F16 && q_dtype != ffi::DataType::BF16) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "PagedPrefillJA: query must be float16 or bfloat16");
  }
  if (k_pool.element_type() != q_dtype || v_pool.element_type() != q_dtype ||
      out->element_type() != q_dtype) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "PagedPrefillJA: query, pools and output must share a dtype");
  }
  for (auto buf : {cu_seqlens_q, kv_indptr, kv_page_indices, kv_last_page_lens}) {
    if (buf.element_type() != ffi::DataType::S32) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                        "PagedPrefillJA: ragged and page metadata must be int32");
    }
  }

  const int64_t batch = static_cast<int64_t>(cu_seqlens_q.element_count()) - 1;
  if (batch <= 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "PagedPrefillJA: cu_seqlens_q must have batch + 1 entries");
  }
  if (static_cast<int64_t>(kv_indptr.element_count()) != batch + 1) {
    return ffi::Error(
        ffi::ErrorCode::kInvalidArgument,
        "PagedPrefillJA: kv_indptr must agree with cu_seqlens_q on the batch size");
  }
  if (static_cast<int64_t>(kv_last_page_lens.element_count()) != batch) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "PagedPrefillJA: kv_last_page_lens must have one entry per sequence");
  }

  // NHD strides in elements, matching the LINEAR_LAYOUT 4D contract:
  // K/V are [num_pages, tokens_per_page, num_kv_heads, head_dim], so the page
  // stride is tokens_per_page rows and the "token" stride is one row of heads.
  const auto stride_q = static_cast<ck_tile::index_t>(num_heads * head_dim);
  const auto nhead_stride_q = static_cast<ck_tile::index_t>(head_dim);
  const auto stride_kv = static_cast<ck_tile::index_t>(num_kv_heads * head_dim);
  const auto nhead_stride_kv = static_cast<ck_tile::index_t>(head_dim);
  const auto batch_stride_kv =
      static_cast<ck_tile::index_t>(tokens_per_page * num_kv_heads * head_dim);

  mask_info mask = mha_utils::create_mask_info(
      causal, static_cast<int>(window_size_left),
      static_cast<int>(window_size_right),
      static_cast<ck_tile::index_t>(max_seqlen_q),
      static_cast<ck_tile::index_t>(max_seqlen_k));

  fmha_batch_prefill_args args{
      .q_ptr = query.untyped_data(),
      .k_ptr = k_pool.untyped_data(),
      .v_ptr = v_pool.untyped_data(),
      .bias_ptr = nullptr,
      .q_descale_ptr = nullptr,
      .k_descale_ptr = nullptr,
      .v_descale_ptr = nullptr,
      .rand_val_ptr = nullptr,
      .lse_ptr = nullptr,
      .o_ptr = out->untyped_data(),

      .seqstart_q_ptr = cu_seqlens_q.untyped_data(),
      .sink_ptr = nullptr,

      .seqlen_q = static_cast<ck_tile::index_t>(total_q),
      .seqlen_k = static_cast<ck_tile::index_t>(num_pages * tokens_per_page),
      .batch = static_cast<ck_tile::index_t>(batch),
      .max_seqlen_q = static_cast<ck_tile::index_t>(max_seqlen_q),
      .hdim_q = static_cast<ck_tile::index_t>(head_dim),
      .hdim_v = static_cast<ck_tile::index_t>(head_dim),
      .nhead_q = static_cast<ck_tile::index_t>(num_heads),
      .nhead_k = static_cast<ck_tile::index_t>(num_kv_heads),

      .num_total_pages = static_cast<int32_t>(num_pages),
      .page_block_size = static_cast<ck_tile::index_t>(tokens_per_page),
      .kv_memory_layout =
          ck_tile::BlockAttentionKVCacheMemoryLayoutEnum::LINEAR_LAYOUT,
      .kv_lookup_table =
          ck_tile::BlockAttentionKVCacheLookupTableEnum::SGLANG_PAGE_TABLE_1D,
      .kv_indptr = kv_indptr.untyped_data(),
      .kv_page_indices = kv_page_indices.untyped_data(),
      .kv_last_page_lens = kv_last_page_lens.untyped_data(),
      .seqlen_k_ptr = nullptr,
      .batch_stride_block_table = 0,

      .scale_s = scale,
      .scale_p = 1.0f,
      .scale_o = 1.0f,

      .logits_soft_cap = logits_soft_cap,

      .stride_q = stride_q,
      .stride_k = stride_kv,
      .stride_v = stride_kv,
      .stride_bias = 0,
      .stride_randval = 0,
      .stride_o = stride_q,
      .nhead_stride_q = nhead_stride_q,
      .nhead_stride_k = nhead_stride_kv,
      .nhead_stride_v = nhead_stride_kv,
      .nhead_stride_bias = 0,
      .nhead_stride_randval = 0,
      .nhead_stride_lse = 0,
      .nhead_stride_o = nhead_stride_q,
      // Group mode indexes queries through seqstart_q rather than a batch
      // stride, so the query-side batch strides are zero while the pool keeps a
      // real page stride.
      .batch_stride_q = 0,
      .batch_stride_k = batch_stride_kv,
      .batch_stride_v = batch_stride_kv,
      .batch_stride_bias = 0,
      .batch_stride_randval = 0,
      .batch_stride_lse = 0,
      .batch_stride_o = 0,

      .window_size_left = mask.left,
      .window_size_right = mask.right,
      .sink_size = 0,
      .mask_type = static_cast<ck_tile::index_t>(mask.type),

      .p_drop = 0.0f,
      .s_randval = false,
      .drop_seed_offset = std::make_pair<uint64_t, uint64_t>(0, 0),
  };

  auto stream_config = mha_utils::create_stream_config(stream);
  const float elapsed = aiter::mha_batch_prefill(
      args, stream_config, mha_utils::dtype_to_string(q_dtype),
      /*is_group_mode=*/true, mask.type, bias_enum::no_bias, /*has_lse=*/false,
      quant_scale_enum::no_scale, /*use_ext_asm=*/false);

  // aiter reports "no kernel matches this configuration" as a negative time
  // rather than an error, and the codegen filter in Makefile.kv is what decides
  // which configurations exist, so say so.
  if (elapsed < 0) {
    return ffi::Error(
        ffi::ErrorCode::kFailedPrecondition,
        "PagedPrefillJA: no batch-prefill kernel matches this configuration "
        "(head_dim=" +
            std::to_string(head_dim) +
            ", tokens_per_page=" + std::to_string(tokens_per_page) +
            ", dtype=" + mha_utils::dtype_to_string(q_dtype) +
            ", causal=" + (causal ? "true" : "false") +
            "). The generated set is fixed at build time by BP_FILTER in "
            "Makefile.kv; widen it and rebuild paged_prefill_ja.so.");
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      std::string("PagedPrefillJA: launch failed: ") +
                          hipGetErrorString(err));
  }
  return ffi::Error::Success();
}

} // namespace jax_aiter

#pragma GCC visibility push(default)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    PagedPrefillJA, jax_aiter::PagedPrefill_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>() // query              [total_q, num_heads, head_dim]
        .Arg<ffi::AnyBuffer>() // k_pool             [pages, tpp, kv_heads, head_dim]
        .Arg<ffi::AnyBuffer>() // v_pool
        .Arg<ffi::AnyBuffer>() // cu_seqlens_q       [batch + 1] int32
        .Arg<ffi::AnyBuffer>() // kv_indptr          [batch + 1] int32
        .Arg<ffi::AnyBuffer>() // kv_page_indices    [total_pages] int32
        .Arg<ffi::AnyBuffer>() // kv_last_page_lens  [batch] int32
        .Ret<ffi::AnyBuffer>() // out                [total_q, num_heads, head_dim]
        .Attr<float>("scale")
        .Attr<float>("logits_soft_cap")
        .Attr<int64_t>("max_seqlen_q")
        .Attr<int64_t>("max_seqlen_k")
        .Attr<bool>("causal")
        .Attr<int64_t>("window_size_left")
        .Attr<int64_t>("window_size_right"));

#pragma GCC visibility pop
