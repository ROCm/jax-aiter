// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Torch-free JAX FFI for the gfx950 fmha_v4_fwd ASM kernels.
//
// This launches the BF16 dense row (mode=0, 656-byte kernarg) and the BF16
// sorted-sparse row (mode=1, 752-byte kernarg). Code-object names, tile sizes,
// and the mangled kernel symbol come from hsa/<arch>/fmha_v4_fwd/fmha_v4_fwd.csv
// via asm_fmha_v4_fwd_configs.hpp. The MXFP4 dense row is launched from
// mha_v4_mxfp4_ja.cu, which also calls the aiter Q/K/V producers.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime.h>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "aiter_hip_common.h"
#include "asm_fmha_v4_fwd_configs.hpp"
#include "hip_utils.h"

namespace ffi = xla::ffi;

namespace jax_aiter {
namespace {

constexpr int64_t kHeadDim = 128;
constexpr int kMaxDevices = 16;
constexpr int32_t kWorkTableFusedMax = 8192;
constexpr int32_t kWorkTableSortThreads = 1024;
constexpr int32_t kWorkTableWave = 64;
constexpr int kBf16Format = 2;
constexpr int kScaleNone = 0;

struct PointerSlot {
  void* value;
  uint32_t padding[2];
};

struct ConstPointerSlot {
  const void* value;
  uint32_t padding[2];
};

struct ScalarSlot {
  uint32_t value;
  uint32_t padding[3];
};

struct __attribute__((packed)) FmhaV4Kernarg {
  PointerSlot ptr_o;
  ConstPointerSlot ptr_q;
  ConstPointerSlot ptr_k;
  ConstPointerSlot ptr_v;
  PointerSlot ptr_lse;
  ScalarSlot scalar;
  ScalarSlot s_seq_len;
  ScalarSlot s_Seqs;
  ScalarSlot s_Ts;
  ScalarSlot s_Hs;
  ScalarSlot s_Bs;
  ScalarSlot s_gqa;
  ScalarSlot s_k_Seqs;
  ScalarSlot s_k_Hs;
  ScalarSlot s_k_Bs;
  ScalarSlot s_opt;
  ScalarSlot s_lse;
  ScalarSlot s_kv_seq_len;
  ScalarSlot s_qk_head_dim;
  ScalarSlot s_v_head_dim;
  ScalarSlot s_q_head_num;
  ScalarSlot s_v_Seqs;
  ScalarSlot s_v_Hs;
  ScalarSlot s_v_Bs;
  ScalarSlot s_o_Seqs;
  ScalarSlot s_o_Hs;
  ScalarSlot s_o_Bs;
  ConstPointerSlot ptr_qseq;
  ConstPointerSlot ptr_kseq;
  ScalarSlot s_lse_Hs;
  ConstPointerSlot ptr_qseq_padding;
  ConstPointerSlot ptr_kseq_padding;
  ConstPointerSlot ptr_q_descale;
  ConstPointerSlot ptr_k_descale;
  ConstPointerSlot ptr_v_descale;
  ScalarSlot s_descale_q_Bs;
  ScalarSlot s_descale_q_Hs;
  ScalarSlot s_descale_k_Bs;
  ScalarSlot s_descale_k_Hs;
  ScalarSlot s_descale_v_Bs;
  ScalarSlot s_descale_v_Hs;
};

static_assert(sizeof(FmhaV4Kernarg) == 656, "MHA v4 dense kernarg ABI");
static_assert(offsetof(FmhaV4Kernarg, ptr_q_descale) == 0x200, "descale slot");

struct __attribute__((packed)) FmhaV4SparseSortedKernarg {
  FmhaV4Kernarg dense;
  ConstPointerSlot ptr_kv_block_indices;
  ConstPointerSlot ptr_lut_start;
  ConstPointerSlot ptr_lut_count;
  ConstPointerSlot ptr_lut_freeze;
  ConstPointerSlot ptr_work_table;
  uint32_t s_num_wgs;
  uint32_t s_total_tiles;
  uint64_t tail_pad;
};

static_assert(sizeof(FmhaV4SparseSortedKernarg) == 752, "MHA v4 sparse kernarg ABI");
static_assert(offsetof(FmhaV4SparseSortedKernarg, ptr_work_table) == 0x2D0,
              "work table slot");

struct Bshd {
  int64_t b, s, h, d;
  uint32_t stride_b;
  uint32_t stride_s;
  uint32_t stride_h;
  int elem;
};

bool fits_u32(int64_t value, uint32_t* out) {
  if (value < 0 || value > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
    return false;
  *out = static_cast<uint32_t>(value);
  return true;
}

ffi::Error contiguous_bshd(const ffi::AnyBuffer& buf, const char* name, Bshd* out) {
  auto dims = buf.dimensions();
  if (dims.size() != 4) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      std::string(name) + " must be BSHD rank 4");
  }
  if (buf.element_type() != ffi::DataType::BF16) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      std::string(name) + " must be bfloat16");
  }
  out->b = dims[0];
  out->s = dims[1];
  out->h = dims[2];
  out->d = dims[3];
  out->elem = 2;
  const int64_t numel = out->b * out->s * out->h * out->d;
  if (static_cast<int64_t>(buf.size_bytes()) != numel * out->elem) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      std::string(name) + " must be dense C-contiguous BSHD");
  }
  const int64_t stride_h = out->d * out->elem;
  const int64_t stride_s = out->h * stride_h;
  const int64_t stride_b = out->s * stride_s;
  if (!fits_u32(stride_b, &out->stride_b) || !fits_u32(stride_s, &out->stride_s) ||
      !fits_u32(stride_h, &out->stride_h)) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      std::string(name) + " stride does not fit the 32-bit kernarg");
  }
  return ffi::Error::Success();
}

const fmha_v4_fwdConfig* find_bf16(const std::string& arch, int mode) {
  for (const auto& entry : cfg_fmha_v4_fwd) {
    const auto& cfg = entry.second;
    if (cfg.arch == arch && cfg.q_format == kBf16Format && cfg.k_format == kBf16Format &&
        cfg.v_format == kBf16Format && cfg.v_pack == 0 && cfg.o_format == kBf16Format &&
        cfg.q_scale_mode == kScaleNone && cfg.k_scale_mode == kScaleNone &&
        cfg.v_scale_mode == kScaleNone && cfg.o_scale_mode == kScaleNone &&
        cfg.hdim_q == kHeadDim && cfg.hdim_v == kHeadDim && cfg.mask == 0 && cfg.mode == mode) {
      return &cfg;
    }
  }
  return nullptr;
}

struct KernelCache {
  std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> kernels;
};

KernelCache& cache_for(int dev) {
  static KernelCache caches[kMaxDevices];
  return caches[dev];
}

ffi::Error load_kernel(int dev, const fmha_v4_fwdConfig& cfg, AiterAsmKernel** out) {
  if (dev < 0 || dev >= kMaxDevices) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "device id out of range");
  }
  auto& cache = cache_for(dev);
  const std::string key = cfg.arch + "|" + cfg.knl_name + "|" + cfg.co_name;
  auto it = cache.kernels.find(key);
  if (it == cache.kernels.end()) {
    aiter_detail::AiterThrowGuard guard;
    try {
      AITER_LOG_INFO("jax-aiter fmha_v4_fwd load " << cfg.co_name << " on device " << dev);
      auto kernel = std::make_unique<AiterAsmKernel>(cfg.knl_name.c_str(), cfg.co_name.c_str());
      it = cache.kernels.emplace(key, std::move(kernel)).first;
    } catch (const std::exception& ex) {
      return ffi::Error(ffi::ErrorCode::kInternal, ex.what());
    }
  }
  *out = it->second.get();
  return ffi::Error::Success();
}

ffi::Error fill_dense(FmhaV4Kernarg* args, const Bshd& q, const Bshd& k, const Bshd& v,
                      const Bshd& o, const void* q_ptr, const void* k_ptr, const void* v_ptr,
                      void* o_ptr, const fmha_v4_fwdConfig& cfg, int64_t gqa, float softmax_scale) {
  std::memset(args, 0, sizeof(*args));
  args->ptr_o.value = o_ptr;
  args->ptr_q.value = q_ptr;
  args->ptr_k.value = k_ptr;
  args->ptr_v.value = v_ptr;
  // BF16 rows do not read descales. Point them at real device memory anyway.
  args->ptr_q_descale.value = q_ptr;
  args->ptr_k_descale.value = k_ptr;
  args->ptr_v_descale.value = v_ptr;
  std::memcpy(&args->scalar.value, &softmax_scale, sizeof(softmax_scale));
  uint32_t seq_q = 0, seq_k = 0, tile = 0, gqa_u = 0, heads = 0;
  if (!fits_u32(q.s, &seq_q) || !fits_u32(k.s, &seq_k) || !fits_u32(gqa, &gqa_u) ||
      !fits_u32(q.h, &heads) ||
      !fits_u32(static_cast<int64_t>(cfg.ts_qo) * q.stride_s, &tile)) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "MHA v4 shape does not fit the 32-bit kernarg");
  }
  args->s_seq_len.value = seq_q;
  args->s_Seqs.value = q.stride_s;
  args->s_Ts.value = tile;
  args->s_Hs.value = q.stride_h;
  args->s_Bs.value = q.stride_b;
  args->s_gqa.value = gqa_u;
  args->s_k_Seqs.value = k.stride_s;
  args->s_k_Hs.value = k.stride_h;
  args->s_k_Bs.value = k.stride_b;
  args->s_opt.value = 5;
  args->s_kv_seq_len.value = seq_k;
  args->s_qk_head_dim.value = static_cast<uint32_t>(kHeadDim);
  args->s_v_head_dim.value = static_cast<uint32_t>(kHeadDim);
  args->s_q_head_num.value = heads;
  args->s_v_Seqs.value = v.stride_s;
  args->s_v_Hs.value = v.stride_h;
  args->s_v_Bs.value = v.stride_b;
  args->s_o_Seqs.value = o.stride_s;
  args->s_o_Hs.value = o.stride_h;
  args->s_o_Bs.value = o.stride_b;
  return ffi::Error::Success();
}

ffi::Error check_bf16_shapes(const Bshd& q, const Bshd& k, const Bshd& v, const Bshd& o,
                             int64_t* gqa) {
  if (q.d != kHeadDim || k.d != kHeadDim || v.d != kHeadDim || o.d != kHeadDim) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "MHA v4 BF16 head dim must be 128");
  }
  if (q.b <= 0 || q.s <= 0 || k.s <= 0 || q.h <= 0 || k.h <= 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "MHA v4 requires non-empty inputs");
  }
  if (k.b != q.b || v.b != q.b || o.b != q.b || v.s != k.s || v.h != k.h) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "K/V/out batch layout does not match Q");
  }
  if (o.s != q.s || o.h != q.h) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "out must be [batch, seqlen_q, nhead_q, 128]");
  }
  if (q.h % k.h != 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "query heads must be divisible by KV heads");
  }
  *gqa = q.h / k.h;
  if (*gqa > 16 || (*gqa & (*gqa - 1)) != 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "MHA v4 GQA ratio must be a power of two up to 16");
  }
  return ffi::Error::Success();
}

__device__ __host__ int32_t pack_work_entry(int32_t linear, int32_t q_tiles, int32_t nhead) {
  const int32_t q_idx = linear % q_tiles;
  const int32_t h_idx = (linear / q_tiles) % nhead;
  const int32_t b_idx = linear / (q_tiles * nhead);
  return (q_idx & 0xFFFF) | ((h_idx & 0xFF) << 16) | ((b_idx & 0xFF) << 24);
}

__global__ void rank_and_pack_work_table_kernel(int32_t* __restrict__ table,
                                               const int32_t* __restrict__ lut_count,
                                               int32_t total, int32_t q_tiles, int32_t nhead) {
  __shared__ uint64_t keys[kWorkTableFusedMax];
  for (int32_t slot = threadIdx.x; slot < total; slot += blockDim.x) {
    keys[slot] = (static_cast<uint64_t>(~static_cast<uint32_t>(lut_count[slot])) << 32) |
                 static_cast<uint32_t>(slot);
  }
  __syncthreads();

  const int32_t lane = threadIdx.x % warpSize;
  const int32_t waves_per_block = blockDim.x / warpSize;
  const int32_t first_wave = blockIdx.x * waves_per_block + threadIdx.x / warpSize;
  for (int32_t slot = first_wave; slot < total; slot += gridDim.x * waves_per_block) {
    const uint64_t mine = keys[slot];
    int32_t rank = 0;
    for (int32_t other = lane; other < total; other += warpSize)
      rank += keys[other] < mine ? 1 : 0;
    for (int32_t offset = warpSize >> 1; offset > 0; offset >>= 1)
      rank += __shfl_xor(rank, offset);
    if (lane == 0)
      table[rank] = pack_work_entry(slot, q_tiles, nhead);
  }
}

__global__ void pack_work_table_kernel(int32_t* __restrict__ table,
                                      const int32_t* __restrict__ order, int32_t total,
                                      int32_t q_tiles, int32_t nhead) {
  const int32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= total)
    return;
  table[slot] = pack_work_entry(order[slot], q_tiles, nhead);
}

ffi::Error build_work_table(hipStream_t stream, const int32_t* lut_count, int32_t* table,
                            int64_t batch, int64_t nhead, int64_t q_tiles) {
  if (batch >= 256 || nhead >= 256 || q_tiles >= 65536 || batch <= 0 || nhead <= 0 ||
      q_tiles <= 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "work table requires 0 < batch,heads < 256 and 0 < query tiles < 65536");
  }
  const int64_t total64 = batch * nhead * q_tiles;
  if (total64 > std::numeric_limits<int32_t>::max()) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "work table is too large for int32");
  }
  const int32_t total = static_cast<int32_t>(total64);
  const int32_t tiles = static_cast<int32_t>(q_tiles);
  const int32_t heads = static_cast<int32_t>(nhead);

  if (total <= kWorkTableFusedMax) {
    const int32_t waves_per_block = kWorkTableSortThreads / kWorkTableWave;
    const dim3 grid(static_cast<uint32_t>((total + waves_per_block - 1) / waves_per_block));
    rank_and_pack_work_table_kernel<<<grid, dim3(kWorkTableSortThreads), 0, stream>>>(
        table, lut_count, total, tiles, heads);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
      return ffi::Error(ffi::ErrorCode::kInternal,
                        std::string("work table kernel: ") + hipGetErrorString(err));
    }
    return ffi::Error::Success();
  }

  // Past the LDS key array. Stable descending sort on the host, then pack.
  // This synchronizes the launch stream; the fused path above does not.
  std::vector<int32_t> counts(total);
  hipError_t err = hipMemcpyAsync(counts.data(), lut_count, total * sizeof(int32_t),
                                  hipMemcpyDeviceToHost, stream);
  if (err != hipSuccess) {
    return ffi::Error(ffi::ErrorCode::kInternal, hipGetErrorString(err));
  }
  err = hipStreamSynchronize(stream);
  if (err != hipSuccess) {
    return ffi::Error(ffi::ErrorCode::kInternal, hipGetErrorString(err));
  }
  std::vector<int32_t> order(total);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
    return counts[a] > counts[b];
  });

  int32_t* order_dev = nullptr;
  err = hipMalloc(&order_dev, total * sizeof(int32_t));
  if (err != hipSuccess) {
    return ffi::Error(ffi::ErrorCode::kInternal, hipGetErrorString(err));
  }
  err = hipMemcpyAsync(order_dev, order.data(), total * sizeof(int32_t), hipMemcpyHostToDevice,
                       stream);
  if (err == hipSuccess) {
    constexpr int32_t block = 256;
    const dim3 grid(static_cast<uint32_t>((total + block - 1) / block));
    pack_work_table_kernel<<<grid, dim3(block), 0, stream>>>(table, order_dev, total, tiles, heads);
    err = hipGetLastError();
  }
  if (err == hipSuccess)
    err = hipStreamSynchronize(stream);
  (void)hipFree(order_dev);
  if (err != hipSuccess) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      std::string("work table pack: ") + hipGetErrorString(err));
  }
  return ffi::Error::Success();
}

bool nonempty(const ffi::AnyBuffer& buf) {
  return buf.size_bytes() > 0 && buf.untyped_data() != nullptr;
}

}  // namespace

ffi::Error MhaV4Fwd_Bridge(hipStream_t stream, ffi::AnyBuffer q, ffi::AnyBuffer k, ffi::AnyBuffer v,
                           ffi::AnyBuffer seqlens_k, ffi::Result<ffi::AnyBuffer> out,
                           ffi::Result<ffi::AnyBuffer> lse, float softmax_scale) {
  const int dev = ::jax_aiter::device_from_ptr(q.untyped_data());
  hipError_t derr = hipSetDevice(dev);
  if (derr != hipSuccess) {
    return ffi::Error(ffi::ErrorCode::kInternal, hipGetErrorString(derr));
  }

  Bshd qs{}, ks{}, vs{}, os{};
  if (auto err = contiguous_bshd(q, "Q", &qs); !err.success()) return err;
  if (auto err = contiguous_bshd(k, "K", &ks); !err.success()) return err;
  if (auto err = contiguous_bshd(v, "V", &vs); !err.success()) return err;
  if (auto err = contiguous_bshd(*out, "out", &os); !err.success()) return err;
  int64_t gqa = 1;
  if (auto err = check_bf16_shapes(qs, ks, vs, os, &gqa); !err.success()) return err;

  const std::string arch = get_gpu_arch();
  const fmha_v4_fwdConfig* cfg = find_bf16(arch, /*mode=*/0);
  if (cfg == nullptr) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "no BF16 dense fmha_v4_fwd code object for " + arch);
  }

  FmhaV4Kernarg args{};
  if (auto err = fill_dense(&args, qs, ks, vs, os, q.untyped_data(), k.untyped_data(),
                            v.untyped_data(), out->untyped_data(), *cfg, gqa, softmax_scale);
      !err.success())
    return err;

  if (nonempty(seqlens_k)) {
    if (seqlens_k.element_type() != ffi::DataType::S32) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument, "seqlens_k must be int32");
    }
    if (seqlens_k.dimensions().size() != 1 || seqlens_k.dimensions()[0] < qs.b) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                        "seqlens_k must be int32 [batch]");
    }
    args.ptr_kseq.value = seqlens_k.untyped_data();
  }

  if (nonempty(*lse)) {
    auto dims = lse->dimensions();
    if (lse->element_type() != ffi::DataType::F32 || dims.size() != 3 || dims[0] != qs.b ||
        dims[1] != qs.h || dims[2] != qs.s) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                        "lse must be float32 [batch, nhead_q, seqlen_q]");
    }
    const int64_t lse_numel = dims[0] * dims[1] * dims[2];
    if (static_cast<int64_t>(lse->size_bytes()) != lse_numel * static_cast<int64_t>(sizeof(float))) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument, "lse must be dense C-contiguous");
    }
    uint32_t lse_hs = 0;
    if (!fits_u32(dims[2] * static_cast<int64_t>(sizeof(float)), &lse_hs)) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument, "lse stride does not fit");
    }
    args.ptr_lse.value = lse->untyped_data();
    args.s_lse.value = 1;
    args.s_lse_Hs.value = lse_hs;
  }

  AiterAsmKernel* kernel = nullptr;
  if (auto err = load_kernel(dev, *cfg, &kernel); !err.success()) return err;
  size_t arg_size = sizeof(args);
  const int gdx = static_cast<int>((qs.s + cfg->ts_qo - 1) / cfg->ts_qo);
  const int gdy = static_cast<int>(qs.h);
  const int gdz = static_cast<int>(qs.b);
  aiter_detail::AiterThrowGuard guard;
  try {
    kernel->launch_kernel({&args, &arg_size, gdx, gdy, gdz, 512, 1, 1, stream});
  } catch (const std::exception& ex) {
    return ffi::Error(ffi::ErrorCode::kInternal, ex.what());
  }
  return ffi::Error::Success();
}

ffi::Error MhaV4SparseFwd_Bridge(hipStream_t stream, ffi::AnyBuffer q, ffi::AnyBuffer k,
                                 ffi::AnyBuffer v, ffi::AnyBuffer kv_block_indices,
                                 ffi::AnyBuffer lut_start, ffi::AnyBuffer lut_count,
                                 ffi::Result<ffi::AnyBuffer> out,
                                 ffi::Result<ffi::AnyBuffer> work_table, float softmax_scale) {
  const int dev = ::jax_aiter::device_from_ptr(q.untyped_data());
  hipError_t derr = hipSetDevice(dev);
  if (derr != hipSuccess) {
    return ffi::Error(ffi::ErrorCode::kInternal, hipGetErrorString(derr));
  }

  Bshd qs{}, ks{}, vs{}, os{};
  if (auto err = contiguous_bshd(q, "Q", &qs); !err.success()) return err;
  if (auto err = contiguous_bshd(k, "K", &ks); !err.success()) return err;
  if (auto err = contiguous_bshd(v, "V", &vs); !err.success()) return err;
  if (auto err = contiguous_bshd(*out, "out", &os); !err.success()) return err;
  int64_t gqa = 1;
  if (auto err = check_bf16_shapes(qs, ks, vs, os, &gqa); !err.success()) return err;

  const std::string arch = get_gpu_arch();
  const fmha_v4_fwdConfig* cfg = find_bf16(arch, /*mode=*/1);
  if (cfg == nullptr) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "no BF16 sorted-sparse fmha_v4_fwd code object for " + arch);
  }
  if (ks.s % cfg->ts_kv != 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "sorted-sparse MHA v4 requires the key length to be a multiple of " +
                          std::to_string(cfg->ts_kv));
  }
  const int64_t q_tiles = (qs.s + cfg->ts_qo - 1) / cfg->ts_qo;
  const int64_t kv_tiles = ks.s / cfg->ts_kv;
  if (arch == "gfx950" && kv_tiles > 8192) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "gfx950 LDS-staged sparse MHA v4 supports at most 8192 KV tiles");
  }
  const int64_t lut_rows = qs.b * qs.h * q_tiles;
  auto i32_len = [](const ffi::AnyBuffer& buf) -> int64_t {
    if (buf.element_type() != ffi::DataType::S32 || buf.dimensions().size() != 1)
      return -1;
    return buf.dimensions()[0];
  };
  if (i32_len(lut_start) != lut_rows || i32_len(lut_count) != lut_rows) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "lut_start and lut_count must be int32 [batch * heads * query_tiles]");
  }
  if (i32_len(kv_block_indices) < 1) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "kv_block_indices must be a non-empty int32 vector");
  }
  if (work_table->element_type() != ffi::DataType::S32 ||
      work_table->dimensions().size() != 1 || work_table->dimensions()[0] != lut_rows) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "work table must be int32 [batch * heads * query_tiles]");
  }

  if (auto err = build_work_table(stream, static_cast<const int32_t*>(lut_count.untyped_data()),
                                  static_cast<int32_t*>(work_table->untyped_data()), qs.b, qs.h,
                                  q_tiles);
      !err.success())
    return err;

  FmhaV4SparseSortedKernarg args{};
  if (auto err = fill_dense(&args.dense, qs, ks, vs, os, q.untyped_data(), k.untyped_data(),
                            v.untyped_data(), out->untyped_data(), *cfg, gqa, softmax_scale);
      !err.success())
    return err;
  args.ptr_kv_block_indices.value = kv_block_indices.untyped_data();
  args.ptr_lut_start.value = lut_start.untyped_data();
  args.ptr_lut_count.value = lut_count.untyped_data();
  args.ptr_work_table.value = work_table->untyped_data();
  args.s_num_wgs = static_cast<uint32_t>(lut_rows);
  args.s_total_tiles = static_cast<uint32_t>(lut_rows);

  AiterAsmKernel* kernel = nullptr;
  if (auto err = load_kernel(dev, *cfg, &kernel); !err.success()) return err;
  size_t arg_size = sizeof(args);
  aiter_detail::AiterThrowGuard guard;
  try {
    kernel->launch_kernel(
        {&args, &arg_size, static_cast<int>(lut_rows), 1, 1, 512, 1, 1, stream});
  } catch (const std::exception& ex) {
    return ffi::Error(ffi::ErrorCode::kInternal, ex.what());
  }
  return ffi::Error::Success();
}

ffi::Error MhaV4SparseWorkTable_Bridge(hipStream_t stream, ffi::AnyBuffer lut_count,
                                       ffi::Result<ffi::AnyBuffer> table, int batch, int nhead,
                                       int q_tiles) {
  const int dev = ::jax_aiter::device_from_ptr(lut_count.untyped_data());
  hipError_t derr = hipSetDevice(dev);
  if (derr != hipSuccess) {
    return ffi::Error(ffi::ErrorCode::kInternal, hipGetErrorString(derr));
  }
  const int64_t total = static_cast<int64_t>(batch) * nhead * q_tiles;
  if (lut_count.element_type() != ffi::DataType::S32 || lut_count.dimensions().size() != 1 ||
      lut_count.dimensions()[0] != total) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "lut_count must be int32 [batch * heads * query_tiles]");
  }
  if (table->element_type() != ffi::DataType::S32 || table->dimensions().size() != 1 ||
      table->dimensions()[0] != total) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "work table shape mismatch");
  }
  return build_work_table(stream, static_cast<const int32_t*>(lut_count.untyped_data()),
                          static_cast<int32_t*>(table->untyped_data()), batch, nhead, q_tiles);
}

}  // namespace jax_aiter

#pragma GCC visibility push(default)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MhaV4FwdJA, jax_aiter::MhaV4Fwd_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()  // q [B, Sq, Hq, 128] bf16
        .Arg<ffi::AnyBuffer>()  // k [B, Sk, Hk, 128] bf16
        .Arg<ffi::AnyBuffer>()  // v [B, Sk, Hk, 128] bf16
        .Arg<ffi::AnyBuffer>()  // seqlens_k int32 [B], or empty
        .Ret<ffi::AnyBuffer>()  // out bf16
        .Ret<ffi::AnyBuffer>()  // lse f32 [B, Hq, Sq], or empty
        .Attr<float>("softmax_scale"),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MhaV4SparseFwdJA, jax_aiter::MhaV4SparseFwd_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()  // q
        .Arg<ffi::AnyBuffer>()  // k
        .Arg<ffi::AnyBuffer>()  // v
        .Arg<ffi::AnyBuffer>()  // kv_block_indices int32
        .Arg<ffi::AnyBuffer>()  // lut_start int32
        .Arg<ffi::AnyBuffer>()  // lut_count int32
        .Ret<ffi::AnyBuffer>()  // out
        .Ret<ffi::AnyBuffer>()  // work table
        .Attr<float>("softmax_scale"),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MhaV4SparseWorkTableJA, jax_aiter::MhaV4SparseWorkTable_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()  // lut_count
        .Ret<ffi::AnyBuffer>()  // packed work table
        .Attr<int>("batch")
        .Attr<int>("nhead")
        .Attr<int>("q_tiles"),
    {xla::ffi::Traits::kCmdBufferCompatible});

#pragma GCC visibility pop
