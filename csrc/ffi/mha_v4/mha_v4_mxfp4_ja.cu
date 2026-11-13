// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Torch-free JAX FFI for gfx950 FMHA v4 MXFP4.
//
// The producers are the aiter HIP kernels (Hadamard+MXFP4 Q, coalesced K, FP6-P
// V), which are packed for head dim 128. Head dim 256 reuses those kernels on a
// half-head view and permutes the buffers in JAX before this launch. The launch
// selects fwd_hd128_mxfp4.co or fwd_hd256_mxfp4.co from the manifest. Both
// objects read per-batch key lengths from kernarg 0x1C0 and can export LSE.
// Scale buffers keep the zeroed slack the K/V gathers read past the sequence.

#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>

#include <hip/hip_runtime.h>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "aiter_hip_common.h"
#include "aiter_stream.h"
#include "asm_fmha_v4_fwd_configs.hpp"
#include "hip_utils.h"
#include "mha_v4_quant.h"

namespace ffi = xla::ffi;

namespace jax_aiter {
namespace {

constexpr int64_t kHeadDim = 128;
constexpr int64_t kPackedWidth = 64;
constexpr int64_t kTileBytes = 8192;
constexpr int64_t kVScaleTile = 512;
constexpr int64_t kVSlack = 64;
constexpr int64_t kVScaleSlack = 2 * kVScaleTile;
constexpr int64_t kKScaleSlack = 4;
constexpr int64_t kQueryTile = 256;
constexpr int64_t kKvTile = 128;
constexpr int64_t kKvLookahead = 2 * kKvTile;
constexpr int kMxfp4Format = 9;
constexpr int kMxfp6Format = 7;
constexpr int kBf16Format = 2;
constexpr int kMxScale = 5;
constexpr int kScaleNone = 0;
constexpr int kMaxDevices = 16;
constexpr int64_t kMxfp6HeadDim = 256;
constexpr int64_t kMxfp6QRowBytes = 192;
constexpr int64_t kMxfp6QBlocks = 8;
constexpr int64_t kMxfp6KTileBytes = 13312;
constexpr int64_t kMxfp6KTileTokens = 128;
constexpr int64_t kMxfp6KRowBytes = 208;
constexpr int64_t kMxfp6KSlack = 256;
constexpr int64_t kMxfp6KScaleSlack = 64;
constexpr int64_t kMxfp6VTileBytes = 24576;
constexpr int64_t kMxfp6VScaleTileBytes = 1024;
constexpr int64_t kMxfp6VSlack = 256;

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

struct StreamGuard {
  hipStream_t previous;
  explicit StreamGuard(hipStream_t stream) : previous(aiter::getCurrentHIPStream()) {
    aiter::setCurrentHIPStream(stream);
  }
  ~StreamGuard() { aiter::setCurrentHIPStream(previous); }
};

int64_t ceil_div(int64_t n, int64_t d) { return (n + d - 1) / d; }

int64_t q_scale_bytes(int64_t b, int64_t s, int64_t h) {
  const int64_t logical = b * s * h * 4;
  const int64_t padded = ceil_div(s, kQueryTile) * kQueryTile;
  return logical + (padded - s) * h * 4;
}

int64_t k_raw_bytes(int64_t b, int64_t s, int64_t h) {
  return b * h * ceil_div(s, kKvTile) * kTileBytes;
}

int64_t k_scale_bytes(int64_t b, int64_t s, int64_t h) {
  const int64_t logical = b * s * h * 4;
  const int64_t padded = ceil_div(s, kKvTile) * kKvTile + kKvLookahead;
  return logical + (padded - s) * h * 4 + kKScaleSlack;
}

int64_t v_raw_bytes(int64_t b, int64_t s, int64_t h) {
  return b * h * ceil_div(s, kKvTile) * kTileBytes + kVSlack;
}

int64_t v_scale_logical(int64_t b, int64_t s, int64_t h) {
  return b * h * ceil_div(s, kKvTile) * kVScaleTile;
}

int64_t v_scale_bytes(int64_t b, int64_t s, int64_t h) {
  return v_scale_logical(b, s, h) + kVScaleSlack;
}

// Head-dim-aware sizes. At 128 these match the helpers above. At 256 the K/V
// producer still tiles 128 tokens; the row is twice as wide and V keeps one
// packed row of slack (128 bytes) plus the same 1024-byte scale lookahead.
int64_t mx_blocks(int64_t head_dim) { return head_dim / 32; }

int64_t mx_row_bytes(int64_t head_dim) { return head_dim / 2; }

int64_t q_scale_bytes_hd(int64_t b, int64_t s, int64_t h, int64_t head_dim) {
  const int64_t blocks = mx_blocks(head_dim);
  const int64_t logical = b * s * h * blocks;
  const int64_t padded = ceil_div(s, kQueryTile) * kQueryTile;
  return logical + (padded - s) * h * blocks;
}

int64_t k_raw_bytes_hd(int64_t b, int64_t s, int64_t h, int64_t head_dim) {
  return b * h * ceil_div(s, kKvTile) * kKvTile * mx_row_bytes(head_dim);
}

int64_t k_scale_bytes_hd(int64_t b, int64_t s, int64_t h, int64_t head_dim) {
  const int64_t blocks = mx_blocks(head_dim);
  const int64_t logical = b * s * h * blocks;
  const int64_t padded = ceil_div(s, kKvTile) * kKvTile + kKvLookahead;
  return logical + (padded - s) * h * blocks + kKScaleSlack;
}

int64_t v_raw_bytes_hd(int64_t b, int64_t s, int64_t h, int64_t head_dim) {
  return k_raw_bytes_hd(b, s, h, head_dim) + mx_row_bytes(head_dim);
}

int64_t v_scale_logical_hd(int64_t b, int64_t s, int64_t h, int64_t head_dim) {
  return b * h * (head_dim / kHeadDim) * ceil_div(s, kKvTile) * kVScaleTile;
}

int64_t v_scale_bytes_hd(int64_t b, int64_t s, int64_t h, int64_t head_dim) {
  return v_scale_logical_hd(b, s, h, head_dim) + kVScaleSlack;
}

bool fits_u32(int64_t value, uint32_t* out) {
  if (value < 0 || value > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
    return false;
  *out = static_cast<uint32_t>(value);
  return true;
}

bool nonempty(const ffi::AnyBuffer& buf) {
  return buf.size_bytes() > 0 && buf.untyped_data() != nullptr;
}

ffi::Error u8_bytes(const ffi::AnyBuffer& buf, const char* name, int64_t bytes) {
  if (buf.element_type() != ffi::DataType::U8 ||
      static_cast<int64_t>(buf.size_bytes()) != bytes) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      std::string(name) + " must be uint8[" + std::to_string(bytes) + "]");
  }
  return ffi::Error::Success();
}

ffi::Error bf16_bshd(const ffi::AnyBuffer& buf, const char* name, int64_t* b, int64_t* s,
                     int64_t* h) {
  auto dims = buf.dimensions();
  if (buf.element_type() != ffi::DataType::BF16 || dims.size() != 4 || dims[3] != kHeadDim) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      std::string(name) + " must be bfloat16 [B, S, H, 128]");
  }
  const int64_t numel = dims[0] * dims[1] * dims[2] * dims[3];
  if (static_cast<int64_t>(buf.size_bytes()) != numel * 2 || dims[0] <= 0 || dims[1] <= 0 ||
      dims[2] <= 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      std::string(name) + " must be a non-empty contiguous BSHD tensor");
  }
  *b = dims[0];
  *s = dims[1];
  *h = dims[2];
  return ffi::Error::Success();
}

ffi::Error bf16_bshd256(const ffi::AnyBuffer& buf, const char* name, int64_t* b, int64_t* s,
                        int64_t* h) {
  auto dims = buf.dimensions();
  if (buf.element_type() != ffi::DataType::BF16 || dims.size() != 4 || dims[3] != kMxfp6HeadDim) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      std::string(name) + " must be bfloat16 [B, S, H, 256]");
  }
  const int64_t numel = dims[0] * dims[1] * dims[2] * dims[3];
  if (static_cast<int64_t>(buf.size_bytes()) != numel * 2 || dims[0] <= 0 || dims[1] <= 0 ||
      dims[2] <= 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      std::string(name) + " must be a non-empty contiguous BSHD tensor");
  }
  *b = dims[0];
  *s = dims[1];
  *h = dims[2];
  return ffi::Error::Success();
}

aiter_tensor_t tensor_from(void* ptr, std::initializer_list<int64_t> dims, AiterDtype dtype,
                           int device) {
  aiter_tensor_t tensor{};
  tensor.ptr = ptr;
  tensor.dtype_ = dtype;
  tensor.device_id = device;
  tensor.ndim = static_cast<int>(dims.size());
  tensor.numel_ = 1;
  int axis = 0;
  for (int64_t dim : dims) {
    tensor.shape[axis++] = dim;
    tensor.numel_ *= static_cast<size_t>(dim < 0 ? 0 : dim);
  }
  if (tensor.numel_ == 0) {
    tensor.numel_ = 0;
  }
  int64_t stride = 1;
  for (int d = tensor.ndim - 1; d >= 0; --d) {
    tensor.strides[d] = stride;
    stride *= tensor.shape[d] == 0 ? 1 : tensor.shape[d];
  }
  return tensor;
}

ffi::Error zero_suffix(hipStream_t stream, void* ptr, int64_t logical, int64_t total) {
  if (total < logical) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "scale buffer is shorter than its logical size");
  }
  if (total == logical) return ffi::Error::Success();
  hipError_t err = hipMemsetAsync(static_cast<uint8_t*>(ptr) + logical, 0,
                                  static_cast<size_t>(total - logical), stream);
  if (err != hipSuccess) {
    return ffi::Error(ffi::ErrorCode::kInternal, hipGetErrorString(err));
  }
  return ffi::Error::Success();
}

ffi::Error call_quant(const char* what, const std::function<void()>& fn) {
  aiter_detail::AiterThrowGuard guard;
  try {
    fn();
  } catch (const std::exception& ex) {
    return ffi::Error(ffi::ErrorCode::kInternal, std::string(what) + ": " + ex.what());
  }
  return ffi::Error::Success();
}

const fmha_v4_fwdConfig* find_mxfp4(const std::string& arch, int64_t head_dim) {
  for (const auto& entry : cfg_fmha_v4_fwd) {
    const auto& cfg = entry.second;
    if (cfg.arch == arch && cfg.q_format == kMxfp4Format && cfg.k_format == kMxfp4Format &&
        cfg.v_format == kMxfp4Format && cfg.v_pack == 1 && cfg.o_format == kBf16Format &&
        cfg.q_scale_mode == kMxScale && cfg.k_scale_mode == kMxScale &&
        cfg.v_scale_mode == kMxScale && cfg.o_scale_mode == kScaleNone && cfg.hdim_q == head_dim &&
        cfg.hdim_v == head_dim && cfg.mask == 0 && cfg.mode == 0) {
      return &cfg;
    }
  }
  return nullptr;
}

const fmha_v4_fwdConfig* find_mxfp6(const std::string& arch) {
  for (const auto& entry : cfg_fmha_v4_fwd) {
    const auto& cfg = entry.second;
    if (cfg.arch == arch && cfg.q_format == kMxfp6Format && cfg.k_format == kMxfp6Format &&
        cfg.v_format == kMxfp6Format && cfg.v_pack == 1 && cfg.o_format == kBf16Format &&
        cfg.q_scale_mode == kMxScale && cfg.k_scale_mode == kMxScale &&
        cfg.v_scale_mode == kMxScale && cfg.o_scale_mode == kScaleNone &&
        cfg.hdim_q == kMxfp6HeadDim && cfg.hdim_v == kMxfp6HeadDim && cfg.mask == 0 &&
        cfg.mode == 0) {
      return &cfg;
    }
  }
  return nullptr;
}

struct KernelCache {
  std::unordered_map<std::string, std::unique_ptr<AiterAsmKernel>> kernels;
};

ffi::Error load_kernel(int dev, const fmha_v4_fwdConfig& cfg, AiterAsmKernel** out) {
  if (dev < 0 || dev >= kMaxDevices) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "device id out of range");
  }
  static KernelCache caches[kMaxDevices];
  auto& cache = caches[dev];
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

}  // namespace

ffi::Error MhaV4Mxfp4QuantQ_Bridge(hipStream_t stream, ffi::AnyBuffer q,
                                   ffi::Result<ffi::AnyBuffer> packed,
                                   ffi::Result<ffi::AnyBuffer> scale, float multiplier) {
  int64_t b = 0, s = 0, h = 0;
  if (auto err = bf16_bshd(q, "Q", &b, &s, &h); !err.success()) return err;
  const int64_t packed_n = b * s * h * kPackedWidth;
  const int64_t scale_n = q_scale_bytes(b, s, h);
  auto dims = packed->dimensions();
  if (packed->element_type() != ffi::DataType::U8 || dims.size() != 4 || dims[0] != b ||
      dims[1] != s || dims[2] != h || dims[3] != kPackedWidth) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "MXFP4 Q must be uint8 [B, S, H, 64]");
  }
  if (auto err = u8_bytes(*scale, "Q scale", scale_n); !err.success()) return err;

  const int dev = device_from_ptr(q.untyped_data());
  hipError_t derr = hipSetDevice(dev);
  if (derr != hipSuccess) return ffi::Error(ffi::ErrorCode::kInternal, hipGetErrorString(derr));
  if (auto err = zero_suffix(stream, scale->untyped_data(), b * s * h * 4, scale_n); !err.success())
    return err;

  auto input = tensor_from(const_cast<void*>(q.untyped_data()), {b, s, h, kHeadDim},
                           AITER_DTYPE_bf16, dev);
  auto out = tensor_from(packed->untyped_data(), {b, s, h, kPackedWidth}, AITER_DTYPE_u8, dev);
  auto scales = tensor_from(scale->untyped_data(), {b, s, h, 4}, AITER_DTYPE_u8, dev);
  StreamGuard guard(stream);
  return call_quant("MXFP4 Q", [&] {
    aiter::torch_itfs::rotate_activation_mxfp4_quant(out, scales, input, multiplier);
  });
}

ffi::Error MhaV4Mxfp4QuantK_Bridge(hipStream_t stream, ffi::AnyBuffer k, ffi::AnyBuffer mean,
                                   ffi::Result<ffi::AnyBuffer> raw,
                                   ffi::Result<ffi::AnyBuffer> scale) {
  int64_t b = 0, s = 0, h = 0;
  if (auto err = bf16_bshd(k, "K", &b, &s, &h); !err.success()) return err;
  const int64_t raw_n = k_raw_bytes(b, s, h);
  const int64_t scale_n = k_scale_bytes(b, s, h);
  if (auto err = u8_bytes(*raw, "K", raw_n); !err.success()) return err;
  if (auto err = u8_bytes(*scale, "K scale", scale_n); !err.success()) return err;

  const int dev = device_from_ptr(k.untyped_data());
  hipError_t derr = hipSetDevice(dev);
  if (derr != hipSuccess) return ffi::Error(ffi::ErrorCode::kInternal, hipGetErrorString(derr));
  if (auto err = zero_suffix(stream, scale->untyped_data(), b * s * h * 4, scale_n); !err.success())
    return err;

  aiter_tensor_t mean_tensor{};
  if (mean.size_bytes() == 0 || mean.untyped_data() == nullptr) {
    mean_tensor = tensor_from(nullptr, {0}, AITER_DTYPE_fp32, dev);
  } else {
    auto dims = mean.dimensions();
    if (mean.element_type() != ffi::DataType::F32 || dims.size() != 3 || dims[0] != b ||
        dims[1] != h || dims[2] != kHeadDim) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument, "K mean must be float32 [B, H, 128]");
    }
    mean_tensor = tensor_from(const_cast<void*>(mean.untyped_data()), {b, h, kHeadDim},
                              AITER_DTYPE_fp32, dev);
  }

  auto input = tensor_from(const_cast<void*>(k.untyped_data()), {b, s, h, kHeadDim},
                           AITER_DTYPE_bf16, dev);
  auto out = tensor_from(raw->untyped_data(), {raw_n}, AITER_DTYPE_u8, dev);
  auto scales = tensor_from(scale->untyped_data(), {b, s, h, 4}, AITER_DTYPE_u8, dev);
  StreamGuard guard(stream);
  return call_quant("MXFP4 K", [&] {
    aiter::torch_itfs::rotate_activation_mxfp4_quant_k(out, scales, input, mean_tensor);
  });
}

ffi::Error MhaV4Mxfp4QuantV_Bridge(hipStream_t stream, ffi::AnyBuffer v,
                                   ffi::Result<ffi::AnyBuffer> raw,
                                   ffi::Result<ffi::AnyBuffer> scale) {
  int64_t b = 0, s = 0, h = 0;
  if (auto err = bf16_bshd(v, "V", &b, &s, &h); !err.success()) return err;
  const int64_t raw_n = v_raw_bytes(b, s, h);
  const int64_t scale_n = v_scale_bytes(b, s, h);
  if (auto err = u8_bytes(*raw, "V", raw_n); !err.success()) return err;
  if (auto err = u8_bytes(*scale, "V scale", scale_n); !err.success()) return err;

  const int dev = device_from_ptr(v.untyped_data());
  hipError_t derr = hipSetDevice(dev);
  if (derr != hipSuccess) return ffi::Error(ffi::ErrorCode::kInternal, hipGetErrorString(derr));
  if (auto err = zero_suffix(stream, scale->untyped_data(), v_scale_logical(b, s, h), scale_n);
      !err.success())
    return err;

  auto input = tensor_from(const_cast<void*>(v.untyped_data()), {b, s, h, kHeadDim},
                           AITER_DTYPE_bf16, dev);
  auto out = tensor_from(raw->untyped_data(), {raw_n}, AITER_DTYPE_u8, dev);
  auto scales = tensor_from(scale->untyped_data(), {v_scale_logical(b, s, h)}, AITER_DTYPE_u8, dev);
  StreamGuard guard(stream);
  return call_quant("MXFP4 V", [&] {
    aiter::torch_itfs::quantize_v_mxfp4_fp6_p(out, scales, input);
  });
}

ffi::Error MhaV4Mxfp4Fwd_Bridge(hipStream_t stream, ffi::AnyBuffer q, ffi::AnyBuffer k,
                                ffi::AnyBuffer v, ffi::AnyBuffer q_scale, ffi::AnyBuffer k_scale,
                                ffi::AnyBuffer v_scale, ffi::AnyBuffer seqlens_k,
                                ffi::Result<ffi::AnyBuffer> out, ffi::Result<ffi::AnyBuffer> lse,
                                int seq_k, int heads_k, float softmax_scale, int head_dim) {
  if (head_dim != kHeadDim && head_dim != 256) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "MXFP4 MHA head dim must be 128 or 256");
  }
  const int64_t width = mx_row_bytes(head_dim);
  const int64_t blocks = mx_blocks(head_dim);
  auto q_dims = q.dimensions();
  if (q.element_type() != ffi::DataType::U8 || q_dims.size() != 4 || q_dims[3] != width) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "packed Q last dimension must be the MXFP4 row width");
  }
  const int64_t b = q_dims[0];
  const int64_t sq = q_dims[1];
  const int64_t hq = q_dims[2];
  const int64_t sk = seq_k;
  const int64_t hk = heads_k;
  if (b <= 0 || sq <= 0 || hq <= 0 || sk <= 0 || hk <= 0 || hq % hk != 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "MXFP4 MHA shape is empty or GQA is invalid");
  }
  const int64_t gqa = hq / hk;
  if (gqa > 16 || (gqa & (gqa - 1)) != 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "GQA ratio must be a power of two <= 16");
  }
  if (auto err = u8_bytes(k, "K", k_raw_bytes_hd(b, sk, hk, head_dim)); !err.success()) return err;
  if (auto err = u8_bytes(v, "V", v_raw_bytes_hd(b, sk, hk, head_dim)); !err.success()) return err;
  if (auto err = u8_bytes(q_scale, "Q scale", q_scale_bytes_hd(b, sq, hq, head_dim)); !err.success())
    return err;
  if (auto err = u8_bytes(k_scale, "K scale", k_scale_bytes_hd(b, sk, hk, head_dim)); !err.success())
    return err;
  if (auto err = u8_bytes(v_scale, "V scale", v_scale_bytes_hd(b, sk, hk, head_dim)); !err.success())
    return err;

  auto out_dims = out->dimensions();
  if (out->element_type() != ffi::DataType::BF16 || out_dims.size() != 4 || out_dims[0] != b ||
      out_dims[1] != sq || out_dims[2] != hq || out_dims[3] != head_dim) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "out must be bfloat16 [B, Sq, Hq, head_dim]");
  }

  const int dev = device_from_ptr(q.untyped_data());
  hipError_t derr = hipSetDevice(dev);
  if (derr != hipSuccess) return ffi::Error(ffi::ErrorCode::kInternal, hipGetErrorString(derr));
  const std::string arch = get_gpu_arch();
  const fmha_v4_fwdConfig* cfg = find_mxfp4(arch, head_dim);
  if (cfg == nullptr) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "no MXFP4 dense fmha_v4_fwd code object for head dim " +
                          std::to_string(head_dim) + " on " + arch);
  }

  // The MXFP4 K/V producers tile 128 tokens even when the kernel's KV tile is 64.
  const int64_t k_tiles = ceil_div(sk, kKvTile);
  const int64_t head_stride = k_tiles * kKvTile * width;
  const int64_t v_scale_head = (head_dim / kHeadDim) * k_tiles * kVScaleTile;
  FmhaV4Kernarg args{};
  std::memset(&args, 0, sizeof(args));
  args.ptr_o.value = out->untyped_data();
  args.ptr_q.value = q.untyped_data();
  args.ptr_k.value = k.untyped_data();
  args.ptr_v.value = v.untyped_data();
  args.ptr_q_descale.value = q_scale.untyped_data();
  args.ptr_k_descale.value = k_scale.untyped_data();
  args.ptr_v_descale.value = v_scale.untyped_data();
  std::memcpy(&args.scalar.value, &softmax_scale, sizeof(softmax_scale));

  uint32_t seq_q_u = 0, seq_k_u = 0, gqa_u = 0, heads_u = 0, tile_u = 0;
  const int64_t q_stride_s = hq * width;
  const int64_t q_stride_b = sq * q_stride_s;
  const int64_t k_stride_b = hk * head_stride;
  const int64_t o_stride_h = head_dim * 2;
  const int64_t o_stride_s = hq * o_stride_h;
  const int64_t o_stride_b = sq * o_stride_s;
  if (!fits_u32(sq, &seq_q_u) || !fits_u32(sk, &seq_k_u) || !fits_u32(gqa, &gqa_u) ||
      !fits_u32(hq, &heads_u) || !fits_u32(cfg->ts_qo * q_stride_s, &tile_u) ||
      !fits_u32(q_stride_b, &args.s_Bs.value) || !fits_u32(q_stride_s, &args.s_Seqs.value) ||
      !fits_u32(width, &args.s_Hs.value) || !fits_u32(k_stride_b, &args.s_k_Bs.value) ||
      !fits_u32(width, &args.s_k_Seqs.value) || !fits_u32(head_stride, &args.s_k_Hs.value) ||
      !fits_u32(k_stride_b, &args.s_v_Bs.value) || !fits_u32(width, &args.s_v_Seqs.value) ||
      !fits_u32(head_stride, &args.s_v_Hs.value) || !fits_u32(o_stride_b, &args.s_o_Bs.value) ||
      !fits_u32(o_stride_s, &args.s_o_Seqs.value) || !fits_u32(o_stride_h, &args.s_o_Hs.value) ||
      !fits_u32(sq * hq * blocks, &args.s_descale_q_Bs.value) ||
      !fits_u32(blocks, &args.s_descale_q_Hs.value) ||
      !fits_u32(sk * hk * blocks, &args.s_descale_k_Bs.value) ||
      !fits_u32(blocks, &args.s_descale_k_Hs.value) ||
      !fits_u32(hk * v_scale_head, &args.s_descale_v_Bs.value) ||
      !fits_u32(v_scale_head, &args.s_descale_v_Hs.value)) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "MXFP4 MHA shape does not fit the 32-bit kernarg");
  }
  args.s_seq_len.value = seq_q_u;
  args.s_Ts.value = tile_u;
  args.s_gqa.value = gqa_u;
  args.s_opt.value = 5;
  args.s_kv_seq_len.value = seq_k_u;
  args.s_qk_head_dim.value = static_cast<uint32_t>(head_dim);
  args.s_v_head_dim.value = static_cast<uint32_t>(head_dim);
  args.s_q_head_num.value = heads_u;

  // A null kernarg keeps the padded key length. The rebuilt hd128 and hd256
  // objects load this slot at 0x1C0 and clamp the KV loop per batch.
  if (nonempty(seqlens_k)) {
    if (seqlens_k.element_type() != ffi::DataType::S32 || seqlens_k.dimensions().size() != 1 ||
        seqlens_k.dimensions()[0] < b) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument, "seqlens_k must be int32 [batch]");
    }
    const int64_t n = seqlens_k.dimensions()[0];
    if (static_cast<int64_t>(seqlens_k.size_bytes()) != n * static_cast<int64_t>(sizeof(int32_t))) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument, "seqlens_k must be dense int32");
    }
    args.ptr_kseq.value = seqlens_k.untyped_data();
  }

  if (nonempty(*lse)) {
    auto dims = lse->dimensions();
    if (lse->element_type() != ffi::DataType::F32 || dims.size() != 3 || dims[0] != b ||
        dims[1] != hq || dims[2] != sq) {
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
  const int gdx = static_cast<int>(ceil_div(sq, cfg->ts_qo));
  aiter_detail::AiterThrowGuard guard;
  try {
    kernel->launch_kernel({&args, &arg_size, gdx, static_cast<int>(hq), static_cast<int>(b), 512,
                           1, 1, stream});
  } catch (const std::exception& ex) {
    return ffi::Error(ffi::ErrorCode::kInternal, ex.what());
  }
  return ffi::Error::Success();
}

ffi::Error MhaV4Mxfp6QuantQ_Bridge(hipStream_t stream, ffi::AnyBuffer q,
                                   ffi::Result<ffi::AnyBuffer> packed,
                                   ffi::Result<ffi::AnyBuffer> scale, float multiplier) {
  int64_t b = 0, s = 0, h = 0;
  if (auto err = bf16_bshd256(q, "Q", &b, &s, &h); !err.success()) return err;
  const int64_t packed_n = b * s * h * kMxfp6QRowBytes;
  const int64_t scale_n = b * s * h * kMxfp6QBlocks;
  auto dims = packed->dimensions();
  if (packed->element_type() != ffi::DataType::U8 || dims.size() != 4 || dims[0] != b ||
      dims[1] != s || dims[2] != h || dims[3] != kMxfp6QRowBytes) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "MXFP6 Q must be uint8 [B, S, H, 192]");
  }
  if (auto err = u8_bytes(*scale, "Q scale", scale_n); !err.success()) return err;
  const int dev = device_from_ptr(q.untyped_data());
  hipError_t derr = hipSetDevice(dev);
  if (derr != hipSuccess) return ffi::Error(ffi::ErrorCode::kInternal, hipGetErrorString(derr));
  auto input = tensor_from(const_cast<void*>(q.untyped_data()), {b, s, h * 2, kHeadDim},
                           AITER_DTYPE_bf16, dev);
  // The AITER Q producer is unchanged hd128 code. Its two logical heads are
  // contiguous in the JAX HD256 output, which is therefore a reshape only.
  auto out = tensor_from(packed->untyped_data(), {b, s, h * 2, 96}, AITER_DTYPE_u8, dev);
  auto scales = tensor_from(scale->untyped_data(), {b, s, h * 2, 4}, AITER_DTYPE_u8, dev);
  StreamGuard guard(stream);
  return call_quant("MXFP6 Q", [&] {
    aiter::torch_itfs::rotate_activation_mxfp6_quant(out, scales, input, multiplier);
  });
}

ffi::Error MhaV4Mxfp6QuantK_Bridge(hipStream_t stream, ffi::AnyBuffer k, ffi::AnyBuffer mean,
                                   ffi::Result<ffi::AnyBuffer> raw,
                                   ffi::Result<ffi::AnyBuffer> scale) {
  int64_t b = 0, s = 0, h = 0;
  if (auto err = bf16_bshd256(k, "K", &b, &s, &h); !err.success()) return err;
  const int64_t tiles = ceil_div(s, kMxfp6KTileTokens);
  const int64_t payload = b * h * 2 * tiles * kMxfp6KTileBytes;
  const int64_t raw_n = payload + kMxfp6KSlack;
  const int64_t scale_n = b * s * h * kMxfp6QBlocks + kMxfp6KScaleSlack;
  if (auto err = u8_bytes(*raw, "K", raw_n); !err.success()) return err;
  if (auto err = u8_bytes(*scale, "K scale", scale_n); !err.success()) return err;
  const int dev = device_from_ptr(k.untyped_data());
  hipError_t derr = hipSetDevice(dev);
  if (derr != hipSuccess) return ffi::Error(ffi::ErrorCode::kInternal, hipGetErrorString(derr));
  if (auto err = zero_suffix(stream, raw->untyped_data(), payload, raw_n); !err.success()) return err;
  if (auto err = zero_suffix(stream, scale->untyped_data(), b * s * h * kMxfp6QBlocks, scale_n);
      !err.success())
    return err;
  aiter_tensor_t mean_tensor{};
  if (mean.size_bytes() == 0 || mean.untyped_data() == nullptr) {
    mean_tensor = tensor_from(nullptr, {0}, AITER_DTYPE_fp32, dev);
  } else {
    auto dims = mean.dimensions();
    if (mean.element_type() != ffi::DataType::F32 || dims.size() != 3 || dims[0] != b ||
        dims[1] != h || dims[2] != kMxfp6HeadDim) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument, "K mean must be float32 [B, H, 256]");
    }
    mean_tensor = tensor_from(const_cast<void*>(mean.untyped_data()), {b, h * 2, kHeadDim},
                              AITER_DTYPE_fp32, dev);
  }
  auto input = tensor_from(const_cast<void*>(k.untyped_data()), {b, s, h * 2, kHeadDim},
                           AITER_DTYPE_bf16, dev);
  auto out = tensor_from(raw->untyped_data(), {raw_n}, AITER_DTYPE_u8, dev);
  auto scales = tensor_from(scale->untyped_data(), {scale_n}, AITER_DTYPE_u8, dev);
  StreamGuard guard(stream);
  return call_quant("MXFP6 K", [&] {
    aiter::torch_itfs::rotate_activation_mxfp6_quant_k(out, scales, input, mean_tensor, 2);
  });
}

ffi::Error MhaV4Mxfp6QuantV_Bridge(hipStream_t stream, ffi::AnyBuffer v,
                                   ffi::Result<ffi::AnyBuffer> raw,
                                   ffi::Result<ffi::AnyBuffer> scale) {
  int64_t b = 0, s = 0, h = 0;
  if (auto err = bf16_bshd256(v, "V", &b, &s, &h); !err.success()) return err;
  const int64_t tiles = ceil_div(s, kMxfp6KTileTokens);
  const int64_t raw_n = b * h * tiles * kMxfp6VTileBytes + kMxfp6VSlack;
  const int64_t scale_n = b * h * tiles * kMxfp6VScaleTileBytes;
  if (auto err = u8_bytes(*raw, "V", raw_n); !err.success()) return err;
  if (auto err = u8_bytes(*scale, "V scale", scale_n); !err.success()) return err;
  const int dev = device_from_ptr(v.untyped_data());
  hipError_t derr = hipSetDevice(dev);
  if (derr != hipSuccess) return ffi::Error(ffi::ErrorCode::kInternal, hipGetErrorString(derr));
  auto input = tensor_from(const_cast<void*>(v.untyped_data()), {b, s, h * 2, kHeadDim},
                           AITER_DTYPE_bf16, dev);
  auto out = tensor_from(raw->untyped_data(), {raw_n}, AITER_DTYPE_u8, dev);
  auto scales = tensor_from(scale->untyped_data(), {b, h, tiles * kMxfp6VScaleTileBytes},
                            AITER_DTYPE_u8, dev);
  StreamGuard guard(stream);
  return call_quant("MXFP6 V", [&] {
    aiter::torch_itfs::quantize_v_mxfp6_fp6_p(out, scales, input, 2);
  });
}

ffi::Error MhaV4Mxfp6Fwd_Bridge(hipStream_t stream, ffi::AnyBuffer q, ffi::AnyBuffer k,
                                ffi::AnyBuffer v, ffi::AnyBuffer q_scale, ffi::AnyBuffer k_scale,
                                ffi::AnyBuffer v_scale, ffi::AnyBuffer seqlens_k,
                                ffi::Result<ffi::AnyBuffer> out, ffi::Result<ffi::AnyBuffer> lse,
                                int seq_k, int heads_k, float softmax_scale) {
  auto q_dims = q.dimensions();
  if (q.element_type() != ffi::DataType::U8 || q_dims.size() != 4 ||
      q_dims[3] != kMxfp6QRowBytes) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "packed MXFP6 Q must be [B, Sq, Hq, 192]");
  }
  const int64_t b = q_dims[0], sq = q_dims[1], hq = q_dims[2], sk = seq_k, hk = heads_k;
  if (b <= 0 || sq <= 0 || hq <= 0 || sk <= 0 || hk <= 0 || hq % hk != 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "MXFP6 MHA shape is empty or GQA is invalid");
  }
  const int64_t gqa = hq / hk;
  if (gqa > 16 || (gqa & (gqa - 1)) != 0) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "GQA ratio must be a power of two <= 16");
  }
  const int64_t tiles = ceil_div(sk, kMxfp6KTileTokens);
  if (auto err = u8_bytes(k, "K", b * hk * 2 * tiles * kMxfp6KTileBytes + kMxfp6KSlack);
      !err.success())
    return err;
  if (auto err = u8_bytes(v, "V", b * hk * tiles * kMxfp6VTileBytes + kMxfp6VSlack);
      !err.success())
    return err;
  if (auto err = u8_bytes(q_scale, "Q scale", b * sq * hq * kMxfp6QBlocks); !err.success())
    return err;
  if (auto err = u8_bytes(k_scale, "K scale", b * sk * hk * kMxfp6QBlocks + kMxfp6KScaleSlack);
      !err.success())
    return err;
  if (auto err = u8_bytes(v_scale, "V scale", b * hk * tiles * kMxfp6VScaleTileBytes);
      !err.success())
    return err;
  auto out_dims = out->dimensions();
  if (out->element_type() != ffi::DataType::BF16 || out_dims.size() != 4 || out_dims[0] != b ||
      out_dims[1] != sq || out_dims[2] != hq || out_dims[3] != kMxfp6HeadDim) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "out must be bfloat16 [B, Sq, Hq, 256]");
  }
  const int dev = device_from_ptr(q.untyped_data());
  hipError_t derr = hipSetDevice(dev);
  if (derr != hipSuccess) return ffi::Error(ffi::ErrorCode::kInternal, hipGetErrorString(derr));
  const fmha_v4_fwdConfig* cfg = find_mxfp6(get_gpu_arch());
  if (cfg == nullptr) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "no HD256 MXFP6 fmha_v4_fwd code object");
  }
  FmhaV4Kernarg args{};
  std::memset(&args, 0, sizeof(args));
  args.ptr_o.value = out->untyped_data();
  args.ptr_q.value = q.untyped_data();
  args.ptr_k.value = k.untyped_data();
  args.ptr_v.value = v.untyped_data();
  args.ptr_q_descale.value = q_scale.untyped_data();
  args.ptr_k_descale.value = k_scale.untyped_data();
  args.ptr_v_descale.value = v_scale.untyped_data();
  std::memcpy(&args.scalar.value, &softmax_scale, sizeof(softmax_scale));
  const int64_t q_stride_s = hq * kMxfp6QRowBytes;
  const int64_t q_stride_b = sq * q_stride_s;
  const int64_t k_stride_h = 2 * tiles * kMxfp6KTileBytes;
  const int64_t k_stride_b = hk * k_stride_h;
  const int64_t v_stride_h = tiles * kMxfp6VTileBytes;
  const int64_t v_stride_b = hk * v_stride_h;
  const int64_t o_stride_h = kMxfp6HeadDim * 2;
  const int64_t o_stride_s = hq * o_stride_h;
  const int64_t o_stride_b = sq * o_stride_s;
  uint32_t sq_u = 0, sk_u = 0, gqa_u = 0, hq_u = 0, tile_u = 0;
  if (!fits_u32(sq, &sq_u) || !fits_u32(sk, &sk_u) || !fits_u32(gqa, &gqa_u) ||
      !fits_u32(hq, &hq_u) || !fits_u32(cfg->ts_qo * q_stride_s, &tile_u) ||
      !fits_u32(q_stride_b, &args.s_Bs.value) || !fits_u32(q_stride_s, &args.s_Seqs.value) ||
      !fits_u32(k_stride_b, &args.s_k_Bs.value) ||
      !fits_u32(kMxfp6KRowBytes, &args.s_k_Seqs.value) ||
      !fits_u32(k_stride_h, &args.s_k_Hs.value) || !fits_u32(v_stride_b, &args.s_v_Bs.value) ||
      !fits_u32(kMxfp6QRowBytes, &args.s_v_Seqs.value) ||
      !fits_u32(v_stride_h, &args.s_v_Hs.value) || !fits_u32(o_stride_b, &args.s_o_Bs.value) ||
      !fits_u32(o_stride_s, &args.s_o_Seqs.value) || !fits_u32(o_stride_h, &args.s_o_Hs.value) ||
      !fits_u32(sq * hq * kMxfp6QBlocks, &args.s_descale_q_Bs.value) ||
      !fits_u32(kMxfp6QBlocks, &args.s_descale_q_Hs.value) ||
      !fits_u32(sk * hk * kMxfp6QBlocks, &args.s_descale_k_Bs.value) ||
      !fits_u32(kMxfp6QBlocks, &args.s_descale_k_Hs.value) ||
      !fits_u32(hk * tiles * kMxfp6VScaleTileBytes, &args.s_descale_v_Bs.value) ||
      !fits_u32(tiles * kMxfp6VScaleTileBytes, &args.s_descale_v_Hs.value)) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "MXFP6 MHA shape does not fit kernarg");
  }
  args.s_seq_len.value = sq_u;
  args.s_Ts.value = tile_u;
  args.s_Hs.value = kMxfp6QRowBytes;
  args.s_gqa.value = gqa_u;
  args.s_kv_seq_len.value = sk_u;
  args.s_qk_head_dim.value = kMxfp6HeadDim;
  args.s_v_head_dim.value = kMxfp6HeadDim;
  args.s_q_head_num.value = hq_u;
  args.s_opt.value = 5;
  if (nonempty(seqlens_k)) {
    if (seqlens_k.element_type() != ffi::DataType::S32 || seqlens_k.dimensions().size() != 1 ||
        seqlens_k.dimensions()[0] < b) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument, "seqlens_k must be int32 [batch]");
    }
    args.ptr_kseq.value = seqlens_k.untyped_data();
  }
  if (nonempty(*lse)) {
    auto dims = lse->dimensions();
    if (lse->element_type() != ffi::DataType::F32 || dims.size() != 3 || dims[0] != b ||
        dims[1] != hq || dims[2] != sq) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument, "lse must be float32 [B, Hq, Sq]");
    }
    args.ptr_lse.value = lse->untyped_data();
    args.s_lse.value = 1;
    if (!fits_u32(sq * static_cast<int64_t>(sizeof(float)), &args.s_lse_Hs.value)) {
      return ffi::Error(ffi::ErrorCode::kInvalidArgument, "lse stride does not fit");
    }
  }
  AiterAsmKernel* kernel = nullptr;
  if (auto err = load_kernel(dev, *cfg, &kernel); !err.success()) return err;
  size_t arg_size = sizeof(args);
  try {
    kernel->launch_kernel({&args, &arg_size, static_cast<int>(ceil_div(sq, cfg->ts_qo)),
                           static_cast<int>(hq), static_cast<int>(b), 512, 1, 1, stream});
  } catch (const std::exception& ex) {
    return ffi::Error(ffi::ErrorCode::kInternal, ex.what());
  }
  return ffi::Error::Success();
}

}  // namespace jax_aiter

#pragma GCC visibility push(default)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MhaV4Mxfp4QuantQJA, jax_aiter::MhaV4Mxfp4QuantQ_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Attr<float>("multiplier"),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MhaV4Mxfp4QuantKJA, jax_aiter::MhaV4Mxfp4QuantK_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>(),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MhaV4Mxfp4QuantVJA, jax_aiter::MhaV4Mxfp4QuantV_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>(),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MhaV4Mxfp4FwdJA, jax_aiter::MhaV4Mxfp4Fwd_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()  // seqlens_k int32 [B], or empty
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()  // lse f32 [B, Hq, Sq], or empty
        .Attr<int>("seq_k")
        .Attr<int>("heads_k")
        .Attr<float>("softmax_scale")
        .Attr<int>("head_dim"),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MhaV4Mxfp6QuantQJA, jax_aiter::MhaV4Mxfp6QuantQ_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Attr<float>("multiplier"),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MhaV4Mxfp6QuantKJA, jax_aiter::MhaV4Mxfp6QuantK_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>(),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MhaV4Mxfp6QuantVJA, jax_aiter::MhaV4Mxfp6QuantV_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>(),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MhaV4Mxfp6FwdJA, jax_aiter::MhaV4Mxfp6Fwd_Bridge,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>()
        .Attr<int>("seq_k")
        .Attr<int>("heads_k")
        .Attr<float>("softmax_scale"),
    {xla::ffi::Traits::kCmdBufferCompatible});

#pragma GCC visibility pop
