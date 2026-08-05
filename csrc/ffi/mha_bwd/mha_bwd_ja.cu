// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Unified MHA backward FFI handler for both batch and varlen modes.
// Detects mode from tensor rank: 4D = batch [b,s,h,d], 3D = varlen [total,h,d].
// Calls aiter::mha_bwd(args, stream) which handles CK vs ASM v3 internally.
//
// Multi-GPU note: AITER's ASM kernel cache is a SynchronizedCache upstream as
// of v0.1.19 (cpp_itfs/mha_bwd.cu), so ASM v3 kernels can be used on all
// devices concurrently. This no longer depends on a local cherry-pick.

#include <hip/hip_runtime.h>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "hip_utils.h"
#include "mha_bwd.h"
#include "mha_common_utils.cu"
#include "mha_common_utils.h"

namespace ffi = xla::ffi;

namespace jax_aiter {

// Persistent workspace pool: reuses device memory across calls to avoid
// per-call hipMalloc/hipFree which synchronises the device.
// Device-aware: re-allocates when the current device changes.
struct WorkspacePool {
  void *ptr = nullptr;
  size_t cap = 0;
  int dev = -1;

  void *get(size_t bytes, hipStream_t stream, bool zero = true) {
    int cur_dev = -1;
    hipGetDevice(&cur_dev);
    if (cur_dev != dev || bytes > cap) {
      if (ptr) { hipSetDevice(dev); hipFree(ptr); hipSetDevice(cur_dev); }
      hipMalloc(&ptr, bytes);
      cap = bytes;
      dev = cur_dev;
    }
    if (zero) hipMemsetAsync(ptr, 0, bytes, stream);
    return ptr;
  }

  ~WorkspacePool() {
    if (ptr) {
      int cur; hipGetDevice(&cur);
      hipSetDevice(dev); hipFree(ptr); hipSetDevice(cur);
    }
  }
};

static thread_local WorkspacePool s_aiter_ws_pool;
static thread_local WorkspacePool s_dk_exp_pool;
static thread_local WorkspacePool s_dv_exp_pool;
static thread_local WorkspacePool s_dbias_pool;
static thread_local WorkspacePool s_rng_pool;

// Pinned host blocks handed to aiter::mha_bwd via mha_bwd_args::pinned_host_alloc,
// which is mandatory in group mode. aiter extends the shared_ptr lifetime to the
// stream tail, so the deleter firing is precisely the point at which no pending
// stream operation can still reference the block and reuse becomes safe.
class PinnedHostPool {
 public:
  std::shared_ptr<void> acquire(size_t bytes) {
    if (bytes == 0) bytes = 1;
    {
      std::lock_guard<std::mutex> lk(mu_);
      for (size_t i = 0; i < free_.size(); ++i) {
        if (free_[i].second >= bytes) {
          Block b = free_[i];
          free_.erase(free_.begin() + i);
          return wrap(b);
        }
      }
    }
    void *p = nullptr;
    if (hipHostMalloc(&p, bytes, hipHostMallocDefault) != hipSuccess) return {};
    return wrap(Block{p, bytes});
  }

 private:
  using Block = std::pair<void *, size_t>;

  std::shared_ptr<void> wrap(Block b) {
    return std::shared_ptr<void>(b.first, [this, b](void *) {
      std::lock_guard<std::mutex> lk(mu_);
      free_.push_back(b);
    });
  }

  std::mutex mu_;
  std::vector<Block> free_;
};

static PinnedHostPool s_pinned_host_pool;

ffi::Error MhaBwdUnified_Bridge(
    hipStream_t stream,
    ffi::AnyBuffer dout, ffi::AnyBuffer q, ffi::AnyBuffer k, ffi::AnyBuffer v,
    ffi::AnyBuffer out, ffi::AnyBuffer softmax_lse,
    std::optional<ffi::AnyBuffer> cu_seqlens_q_,
    std::optional<ffi::AnyBuffer> cu_seqlens_k_,
    std::optional<ffi::AnyBuffer> dq_, std::optional<ffi::AnyBuffer> dk_,
    std::optional<ffi::AnyBuffer> dv_,
    std::optional<ffi::AnyBuffer> bias_, std::optional<ffi::AnyBuffer> alibi_slopes_,
    std::optional<ffi::AnyBuffer> rng_state_, std::optional<ffi::AnyBuffer> gen_,
    std::optional<ffi::AnyBuffer> cu_seqlens_q_logical_,
    std::optional<ffi::AnyBuffer> cu_seqlens_k_logical_,
    ffi::Result<ffi::AnyBuffer> dq_ret, ffi::Result<ffi::AnyBuffer> dk_ret,
    ffi::Result<ffi::AnyBuffer> dv_ret, ffi::Result<ffi::AnyBuffer> softmax_d_ret,
    ffi::Result<ffi::AnyBuffer> dbias_ret,
    float dropout_p, float softmax_scale, bool is_causal,
    int window_size_left, int window_size_right, bool deterministic,
    bool use_asm_v3, bool is_v3_atomic_fp32, int how_v3_bf16_cvt,
    int max_seqlen_q_attr, int max_seqlen_k_attr, bool zero_tensors) {

  if (!q.untyped_data() || !k.untyped_data() || !v.untyped_data() ||
      !out.untyped_data() || !softmax_lse.untyped_data() || !dout.untyped_data()) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Required input buffer is null");
  }

  const int dev_idx = ::jax_aiter::device_from_ptr(q.untyped_data());
  if (dev_idx < 0)
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "bad device from q");

  auto q_dims = q.dimensions();
  auto k_dims = k.dimensions();
  auto v_dims = v.dimensions();
  auto dout_dims = dout.dimensions();
  auto out_dims = out.dimensions();
  auto lse_dims = softmax_lse.dimensions();

  const bool is_varlen = (q_dims.size() == 3);

  int64_t batch_size, seqlen_q, num_heads, head_size_q;
  int64_t seqlen_k, num_heads_k, head_size_v;
  int64_t max_sq, max_sk;

  if (is_varlen) {
    seqlen_q = q_dims[0]; // total_q
    num_heads = q_dims[1];
    head_size_q = q_dims[2];
    seqlen_k = k_dims[0]; // total_k
    num_heads_k = k_dims[1];
    head_size_v = v_dims[2];
    if (!cu_seqlens_q_.has_value() || !mha_utils::is_valid_buffer(*cu_seqlens_q_))
      return ffi::Error(ffi::ErrorCode::kInvalidArgument, "varlen requires cu_seqlens_q");
    batch_size = cu_seqlens_q_->dimensions()[0] - 1;
    max_sq = max_seqlen_q_attr;
    max_sk = max_seqlen_k_attr;
  } else {
    batch_size = q_dims[0];
    seqlen_q = q_dims[1];
    num_heads = q_dims[2];
    head_size_q = q_dims[3];
    seqlen_k = k_dims[1];
    num_heads_k = k_dims[2];
    head_size_v = v_dims[3];
    max_sq = seqlen_q;
    max_sk = seqlen_k;
  }

  if (max_sq == 0) {
    if (dq_ret->size_bytes() > 0)
      HIP_CHECK(hipMemsetAsync(dq_ret->untyped_data(), 0, dq_ret->size_bytes(), stream));
    if (dk_ret->size_bytes() > 0)
      HIP_CHECK(hipMemsetAsync(dk_ret->untyped_data(), 0, dk_ret->size_bytes(), stream));
    if (dv_ret->size_bytes() > 0)
      HIP_CHECK(hipMemsetAsync(dv_ret->untyped_data(), 0, dv_ret->size_bytes(), stream));
    if (softmax_d_ret->size_bytes() > 0)
      HIP_CHECK(hipMemsetAsync(softmax_d_ret->untyped_data(), 0, softmax_d_ret->size_bytes(), stream));
    if (dbias_ret->size_bytes() > 0)
      HIP_CHECK(hipMemsetAsync(dbias_ret->untyped_data(), 0, dbias_ret->size_bytes(), stream));
    return ffi::Error::Success();
  }

  if (num_heads % num_heads_k != 0)
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "num_heads_q must be divisible by num_heads_k");

  bool is_mqa_gqa = (num_heads != num_heads_k);
  std::string dtype_str = mha_utils::dtype_to_string(q.element_type());

  int ref_sk = is_varlen ? max_sk : seqlen_k;
  if (window_size_left >= ref_sk) window_size_left = -1;
  if (window_size_right >= ref_sk) window_size_right = -1;
  int ref_sq = is_varlen ? max_sq : seqlen_q;

  auto mask = mha_utils::create_mask_info(is_causal, window_size_left, window_size_right, ref_sq, ref_sk);

  // Bias handling
  const void *bias_ptr = nullptr;
  ck_tile::index_t stride_bias = 0;
  bool has_bias = bias_.has_value() && mha_utils::is_valid_buffer(*bias_);
  bool has_alibi = alibi_slopes_.has_value() && mha_utils::is_valid_buffer(*alibi_slopes_);

  if (has_bias) {
    bias_ptr = bias_->untyped_data();
    auto bd = bias_->dimensions();
    stride_bias = bd.size() >= 2 ? mha_utils::calculate_stride(bd, 0) : 0;
  } else if (has_alibi) {
    bias_ptr = alibi_slopes_->untyped_data();
    auto ad = alibi_slopes_->dimensions();
    stride_bias = ad.size() >= 2 ? mha_utils::calculate_stride(ad, 0) : 0;
  }
  bias_enum bias_type = mha_utils::get_bias_type(has_bias, has_alibi);

  bool has_dbias = has_bias && (dbias_ret->size_bytes() > 0) && !is_varlen;
  void *dbias_expanded_ptr = nullptr;
  ck_tile::index_t stride_dbias = 0, nhead_stride_dbias = 0, batch_stride_dbias = 0;

  if (has_dbias) {
    size_t dbias_sz = batch_size * seqlen_q * num_heads * seqlen_k * mha_utils::dtype_size(q.element_type());
    dbias_expanded_ptr = s_dbias_pool.get(dbias_sz, stream);
    stride_dbias = num_heads * seqlen_k;
    nhead_stride_dbias = seqlen_k;
    batch_stride_dbias = seqlen_q * num_heads * seqlen_k;
  }

  // RNG
  uint64_t *seed_ptr = nullptr, *offset_ptr = nullptr, *dummy_rng = nullptr;
  if (dropout_p > 0.0f && rng_state_.has_value() && mha_utils::is_valid_buffer(*rng_state_)) {
    try {
      auto [s, o] = mha_utils::get_rng_seed_offset_ptrs(rng_state_, dropout_p);
      seed_ptr = s; offset_ptr = o;
    } catch (...) { /* fallthrough to dummy */ }
  }
  if (!seed_ptr) {
    dummy_rng = (uint64_t *)s_rng_pool.get(2 * sizeof(uint64_t), stream, /*zero=*/false);
    seed_ptr = dummy_rng; offset_ptr = dummy_rng + 1;
  }

  // AITER v0.1.19 owns dq_acc sizing and layout internally; the caller supplies
  // only memory, through mha_bwd_args::workspace_alloc below.

  // MQA/GQA expansion
  auto dq_dims = dq_ret->dimensions();
  auto dk_dims = dk_ret->dimensions();
  auto dv_dims = dv_ret->dimensions();

  void *dk_expanded_ptr = nullptr, *dv_expanded_ptr = nullptr;
  void *dk_final = dk_ret->untyped_data(), *dv_final = dv_ret->untyped_data();

  if (is_mqa_gqa) {
    size_t dk_sz = (is_varlen ? seqlen_k : batch_size * seqlen_k) * num_heads * head_size_q * mha_utils::dtype_size(q.element_type());
    size_t dv_sz = (is_varlen ? seqlen_k : batch_size * seqlen_k) * num_heads * head_size_v * mha_utils::dtype_size(v.element_type());
    // The AITER backward overwrites every logical expanded dK/dV element.
    // Tight packing therefore needs no pre-clear. Padded packing sets
    // zero_tensors and is explicitly cleared below so unwritten padding rows
    // remain defined. Avoiding these unconditional clears removes 512 MiB of
    // writes per attention call at the llama3-8b production shape.
    dk_expanded_ptr = s_dk_exp_pool.get(dk_sz, stream, /*zero=*/false);
    dv_expanded_ptr = s_dv_exp_pool.get(dv_sz, stream, /*zero=*/false);
    dk_final = dk_expanded_ptr; dv_final = dv_expanded_ptr;
  }

  // Zero tensors (varlen)
  if (zero_tensors) {
    HIP_CHECK(hipMemsetAsync(dq_ret->untyped_data(), 0, dq_ret->size_bytes(), stream));
    HIP_CHECK(hipMemsetAsync(dk_final, 0, is_mqa_gqa ? ((is_varlen ? seqlen_k : batch_size * seqlen_k) * num_heads * head_size_q * mha_utils::dtype_size(q.element_type())) : dk_ret->size_bytes(), stream));
    HIP_CHECK(hipMemsetAsync(dv_final, 0, is_mqa_gqa ? ((is_varlen ? seqlen_k : batch_size * seqlen_k) * num_heads * head_size_v * mha_utils::dtype_size(v.element_type())) : dv_ret->size_bytes(), stream));
    HIP_CHECK(hipMemsetAsync(softmax_d_ret->untyped_data(), 0, softmax_d_ret->size_bytes(), stream));
  }

  float p_undrop = mha_utils::calculate_p_undrop(dropout_p);

  // Strides based on rank
  ck_tile::index_t stride_q, stride_k, stride_v, stride_o, stride_do, stride_dq, stride_dk, stride_dv;
  ck_tile::index_t nhs_q, nhs_k, nhs_v, nhs_o, nhs_do, nhs_lse, nhs_dq, nhs_dk, nhs_dv;
  ck_tile::index_t bs_q = 0, bs_k = 0, bs_v = 0, bs_o = 0, bs_do = 0, bs_lse = 0, bs_dq = 0, bs_dk = 0, bs_dv = 0;

  if (is_varlen) {
    stride_q = mha_utils::calculate_stride(q_dims, 0);
    stride_k = mha_utils::calculate_stride(k_dims, 0);
    stride_v = mha_utils::calculate_stride(v_dims, 0);
    stride_o = mha_utils::calculate_stride(out_dims, 0);
    stride_do = mha_utils::calculate_stride(dout_dims, 0);
    stride_dq = mha_utils::calculate_stride(dq_dims, 0);
    stride_dk = is_mqa_gqa ? (num_heads * head_size_q) : mha_utils::calculate_stride(dk_dims, 0);
    stride_dv = is_mqa_gqa ? (num_heads * head_size_v) : mha_utils::calculate_stride(dv_dims, 0);
    nhs_q = mha_utils::calculate_stride(q_dims, 1);
    nhs_k = mha_utils::calculate_stride(k_dims, 1);
    nhs_v = mha_utils::calculate_stride(v_dims, 1);
    nhs_o = mha_utils::calculate_stride(out_dims, 1);
    nhs_do = mha_utils::calculate_stride(dout_dims, 1);
    nhs_lse = mha_utils::calculate_stride(lse_dims, 0);
    nhs_dq = mha_utils::calculate_stride(dq_dims, 1);
    nhs_dk = is_mqa_gqa ? head_size_q : mha_utils::calculate_stride(dk_dims, 1);
    nhs_dv = is_mqa_gqa ? head_size_v : mha_utils::calculate_stride(dv_dims, 1);
  } else {
    stride_q = mha_utils::calculate_stride(q_dims, 1);
    stride_k = mha_utils::calculate_stride(k_dims, 1);
    stride_v = mha_utils::calculate_stride(v_dims, 1);
    stride_o = mha_utils::calculate_stride(out_dims, 1);
    stride_do = mha_utils::calculate_stride(dout_dims, 1);
    stride_dq = mha_utils::calculate_stride(dq_dims, 1);
    stride_dk = is_mqa_gqa ? (num_heads * head_size_q) : mha_utils::calculate_stride(dk_dims, 1);
    stride_dv = is_mqa_gqa ? (num_heads * head_size_v) : mha_utils::calculate_stride(dv_dims, 1);
    nhs_q = mha_utils::calculate_stride(q_dims, 2);
    nhs_k = mha_utils::calculate_stride(k_dims, 2);
    nhs_v = mha_utils::calculate_stride(v_dims, 2);
    nhs_o = mha_utils::calculate_stride(out_dims, 2);
    nhs_do = mha_utils::calculate_stride(dout_dims, 2);
    nhs_lse = mha_utils::calculate_stride(lse_dims, 1);
    nhs_dq = mha_utils::calculate_stride(dq_dims, 2);
    nhs_dk = is_mqa_gqa ? head_size_q : mha_utils::calculate_stride(dk_dims, 2);
    nhs_dv = is_mqa_gqa ? head_size_v : mha_utils::calculate_stride(dv_dims, 2);
    bs_q = mha_utils::calculate_stride(q_dims, 0);
    bs_k = mha_utils::calculate_stride(k_dims, 0);
    bs_v = mha_utils::calculate_stride(v_dims, 0);
    bs_o = mha_utils::calculate_stride(out_dims, 0);
    bs_do = mha_utils::calculate_stride(dout_dims, 0);
    bs_lse = mha_utils::calculate_stride(lse_dims, 0);
    bs_dq = mha_utils::calculate_stride(dq_dims, 0);
    bs_dk = is_mqa_gqa ? (seqlen_k * num_heads * head_size_q) : mha_utils::calculate_stride(dk_dims, 0);
    bs_dv = is_mqa_gqa ? (seqlen_k * num_heads * head_size_v) : mha_utils::calculate_stride(dv_dims, 0);
  }

  // Seqstart pointers. seqstart_* are cumulative PHYSICAL offsets; the optional
  // cu_seqlen_* are cumulative LOGICAL lengths that tell AITER how much of each
  // physical span is real (mha_bwd.h sequence-pointer notes). Both must match
  // what the forward passed, or the backward masks a different region.
  const void *seqstart_q_ptr = nullptr, *seqstart_k_ptr = nullptr;
  const void *cu_seqlen_q_ptr = nullptr, *cu_seqlen_k_ptr = nullptr;
  if (is_varlen) {
    seqstart_q_ptr = cu_seqlens_q_->untyped_data();
    if (cu_seqlens_k_.has_value() && mha_utils::is_valid_buffer(*cu_seqlens_k_))
      seqstart_k_ptr = cu_seqlens_k_->untyped_data();
    if (cu_seqlens_q_logical_.has_value() &&
        mha_utils::is_valid_buffer(*cu_seqlens_q_logical_))
      cu_seqlen_q_ptr = cu_seqlens_q_logical_->untyped_data();
    if (cu_seqlens_k_logical_.has_value() &&
        mha_utils::is_valid_buffer(*cu_seqlens_k_logical_))
      cu_seqlen_k_ptr = cu_seqlens_k_logical_->untyped_data();
  }

  auto args = aiter::mha_bwd_args{
      .use_asm_v3 = use_asm_v3,
      .v3_atomic_fp32 = is_v3_atomic_fp32,
      .v3_bf16_cvt = how_v3_bf16_cvt,
      .v3_api_check = false,
      .hdim_q = static_cast<int>(head_size_q),
      .hdim_v = static_cast<int>(head_size_v),
      .data_type = dtype_str,
      .is_group_mode = is_varlen,
      .mask_type = static_cast<int>(mask.type),
      .bias_type = static_cast<int>(bias_type),
      .has_dbias = has_dbias,
      .has_dropout = (dropout_p > 0.0f),
      .is_store_randval = false,
      .is_deterministic = deterministic,
      .q_ptr = q.untyped_data(), .k_ptr = k.untyped_data(),
      .v_ptr = v.untyped_data(), .bias_ptr = bias_ptr,
      .o_ptr = out.untyped_data(), .lse_ptr = softmax_lse.untyped_data(),
      .do_ptr = dout.untyped_data(), .d_ptr = softmax_d_ret->untyped_data(),
      .rand_val_ptr = nullptr,
      .dq_ptr = dq_ret->untyped_data(), .dk_ptr = dk_final, .dv_ptr = dv_final,
      .dbias_ptr = dbias_expanded_ptr,
      .seqstart_q_ptr = seqstart_q_ptr, .seqstart_k_ptr = seqstart_k_ptr,
      .cu_seqlen_q_ptr = cu_seqlen_q_ptr, .cu_seqlen_k_ptr = cu_seqlen_k_ptr,
      .seqlen_q = static_cast<int>(seqlen_q), .seqlen_k = static_cast<int>(seqlen_k),
      .batch = static_cast<int>(batch_size),
      .max_seqlen_q = static_cast<int>(max_sq), .max_seqlen_k = static_cast<int>(max_sk),
      .nhead_q = static_cast<int>(num_heads), .nhead_k = static_cast<int>(num_heads_k),
      .scale = softmax_scale,
      .stride_q = static_cast<int>(stride_q), .stride_k = static_cast<int>(stride_k),
      .stride_v = static_cast<int>(stride_v), .stride_bias = static_cast<int>(stride_bias),
      .stride_o = static_cast<int>(stride_o), .stride_randval = 0,
      .stride_do = static_cast<int>(stride_do),
      .stride_dq = static_cast<int>(stride_dq), .stride_dk = static_cast<int>(stride_dk),
      .stride_dv = static_cast<int>(stride_dv), .stride_dbias = static_cast<int>(stride_dbias),
      .nhead_stride_q = static_cast<int>(nhs_q), .nhead_stride_k = static_cast<int>(nhs_k),
      .nhead_stride_v = static_cast<int>(nhs_v), .nhead_stride_bias = 0,
      .nhead_stride_o = static_cast<int>(nhs_o), .nhead_stride_randval = 0,
      .nhead_stride_do = static_cast<int>(nhs_do),
      .nhead_stride_lsed = static_cast<int>(nhs_lse),
      .nhead_stride_dq = static_cast<int>(nhs_dq),
      .nhead_stride_dk = static_cast<int>(nhs_dk), .nhead_stride_dv = static_cast<int>(nhs_dv),
      .nhead_stride_dbias = static_cast<int>(nhead_stride_dbias),
      .batch_stride_q = static_cast<int>(bs_q), .batch_stride_k = static_cast<int>(bs_k),
      .batch_stride_v = static_cast<int>(bs_v), .batch_stride_bias = 0,
      .batch_stride_o = static_cast<int>(bs_o), .batch_stride_randval = 0,
      .batch_stride_do = static_cast<int>(bs_do),
      .batch_stride_lsed = static_cast<int>(bs_lse),
      .batch_stride_dq = static_cast<int>(bs_dq),
      .batch_stride_dk = static_cast<int>(bs_dk), .batch_stride_dv = static_cast<int>(bs_dv),
      .batch_stride_dbias = static_cast<int>(batch_stride_dbias),
      .window_size_left = static_cast<int>(mask.left),
      .window_size_right = static_cast<int>(mask.right),
      .p_drop = dropout_p, .p_undrop = p_undrop,
      .drop_seed_offset = std::make_pair(seed_ptr, offset_ptr),
      // zero_init is honoured rather than assumed, so the accumulator is cleared
      // only when the kernel aiter selects actually reads it.
      .workspace_alloc = [stream](size_t bytes, bool zero_init) -> void * {
        if (bytes == 0) return nullptr;
        return s_aiter_ws_pool.get(bytes, stream, zero_init);
      },
      .pinned_host_alloc = [](size_t bytes) -> std::shared_ptr<void> {
        return s_pinned_host_pool.acquire(bytes);
      }
  };

  // Ensure HIP device context matches the data device.  XLA usually sets
  // this, but being explicit prevents WorkspacePool and kernel loads from
  // targeting the wrong device.
  HIP_CHECK(hipSetDevice(dev_idx));

  args.use_asm_v3 = use_asm_v3;

  auto stream_config = mha_utils::create_stream_config(stream);
  float runtime = aiter::mha_bwd(args, stream_config);

  if (runtime < 0) {
    return ffi::Error(ffi::ErrorCode::kInternal, "aiter::mha_bwd failed");
  }

  // MQA/GQA reduction
  if (is_mqa_gqa) {
    int64_t groups = num_heads / num_heads_k;
    int64_t total_tokens = is_varlen ? seqlen_k : batch_size * seqlen_k;
    const char *fuse_env = std::getenv("JA_MHA_FUSE_GQA_REDUCE");
    const bool fuse_pair =
        head_size_q == head_size_v &&
        !(fuse_env != nullptr && std::string(fuse_env) == "0");

    if (fuse_pair) {
      mha_utils::launch_mqa_gqa_reduction_pair(
          dk_expanded_ptr, dv_expanded_ptr, dk_ret->untyped_data(),
          dv_ret->untyped_data(), is_varlen ? 1 : batch_size, seqlen_k,
          num_heads, num_heads_k, head_size_q, groups, q.element_type(),
          stream);
    } else {
      mha_utils::launch_mqa_gqa_reduction(
          dk_expanded_ptr, dk_ret->untyped_data(),
          is_varlen ? 1 : batch_size, seqlen_k, num_heads, num_heads_k,
          head_size_q, groups, q.element_type(), stream);
      mha_utils::launch_mqa_gqa_reduction(
          dv_expanded_ptr, dv_ret->untyped_data(),
          is_varlen ? 1 : batch_size, seqlen_k, num_heads, num_heads_k,
          head_size_v, groups, v.element_type(), stream);
    }

    // dk/dv expanded buffers managed by pool -- no free needed
  }

  if (has_dbias && dbias_expanded_ptr) {
    size_t dbias_sz = batch_size * seqlen_q * num_heads * seqlen_k * mha_utils::dtype_size(q.element_type());
    HIP_CHECK(hipMemcpyAsync(dbias_ret->untyped_data(), dbias_expanded_ptr,
                             dbias_sz, hipMemcpyDeviceToDevice, stream));
  }
  // All workspace buffers managed by pools -- no hipFree needed

  return ffi::Error::Success();
}

} // namespace jax_aiter

#pragma GCC visibility push(default)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MhaBwdUnifiedJA, jax_aiter::MhaBwdUnified_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>() // dout
        .Arg<ffi::AnyBuffer>() // q
        .Arg<ffi::AnyBuffer>() // k
        .Arg<ffi::AnyBuffer>() // v
        .Arg<ffi::AnyBuffer>() // out
        .Arg<ffi::AnyBuffer>() // softmax_lse
        .Arg<ffi::AnyBuffer>() // cu_seqlens_q (optional)
        .Arg<ffi::AnyBuffer>() // cu_seqlens_k (optional)
        .Arg<ffi::AnyBuffer>() // dq_ (optional)
        .Arg<ffi::AnyBuffer>() // dk_ (optional)
        .Arg<ffi::AnyBuffer>() // dv_ (optional)
        .Arg<ffi::AnyBuffer>() // bias_ (optional)
        .Arg<ffi::AnyBuffer>() // alibi_slopes_ (optional)
        .Arg<ffi::AnyBuffer>() // rng_state_ (optional)
        .Arg<ffi::AnyBuffer>() // gen_ (optional)
        .Arg<ffi::AnyBuffer>() // cu_seqlens_q_logical (optional)
        .Arg<ffi::AnyBuffer>() // cu_seqlens_k_logical (optional)
        .Ret<ffi::AnyBuffer>() // dq_ret
        .Ret<ffi::AnyBuffer>() // dk_ret
        .Ret<ffi::AnyBuffer>() // dv_ret
        .Ret<ffi::AnyBuffer>() // softmax_d_ret
        .Ret<ffi::AnyBuffer>() // dbias_ret
        .Attr<float>("dropout_p")
        .Attr<float>("softmax_scale")
        .Attr<bool>("is_causal")
        .Attr<int>("window_size_left")
        .Attr<int>("window_size_right")
        .Attr<bool>("deterministic")
        .Attr<bool>("use_asm_v3")
        .Attr<bool>("is_v3_atomic_fp32")
        .Attr<int>("how_v3_bf16_cvt")
        .Attr<int>("max_seqlen_q_attr")
        .Attr<int>("max_seqlen_k_attr")
        .Attr<bool>("zero_tensors"),
    {xla::ffi::Traits::kCmdBufferCompatible});

#pragma GCC visibility pop
