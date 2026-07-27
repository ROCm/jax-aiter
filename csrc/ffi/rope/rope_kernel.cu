/*
 * NEOX-style RoPE forward kernel (self-contained, torch-free).
 * =============================================================
 *
 * Matches MaxText's RotaryEmbedding.apply_rotary
 * (maxtext/src/maxtext/layers/embeddings.py):
 *
 *     rotate_half(x): (x1, x2) -> (-x2, x1)   split on last dim
 *     out = x * cos + rotate_half(x) * sin
 *
 * where cos/sin are FULL-WIDTH [.., D] (MaxText builds them as
 * concat([cos_half, cos_half], -1)), broadcast over the head axis.
 *
 * Layout: x, out are [B, S, N, D] contiguous (BSHD); cos, sin are
 * [B, S, D] (shared across the N heads). Compute in fp32, store bf16.
 *
 * This follows the cast_mxfp4 precedent: a self-contained HIP kernel with
 * an extern "C" launcher, called by the JAX FFI shim. No torch / no ATen
 * (AITER's own rope_common.h is ATen-coupled, so it is not reused here).
 */

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

namespace ja_rope {

// One thread per (row, j) PAIR with j in [0, half): handles the two coupled
// elements d=j and d=j+half, reading each x element ONCE.
//   out[j]      = x[j]*cos[j]      - x[j+half]*sin[j]
//   out[j+half] = x[j+half]*cos[j] + x[j]*sin[j]      (cos/sin full-width => cos[j+half]==cos[j])
//   row = pair_row index -> (b*S + s) * N + n ; bs = row / n_heads
//   cos/sin offset = bs * D + j   (only the front half of cos/sin is read)
__global__ void rope_neox_fwd_bf16_kernel(
    const __hip_bfloat16* __restrict__ x,
    const __hip_bfloat16* __restrict__ cos,
    const __hip_bfloat16* __restrict__ sin,
    __hip_bfloat16* __restrict__ out,
    int64_t n_rows,
    int n_heads,
    int D) {
  const int half = D >> 1;
  const int64_t n_pairs = n_rows * static_cast<int64_t>(half);
  for (int64_t p = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
       p < n_pairs;
       p += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int j = static_cast<int>(p % half);
    const int64_t row = p / half;
    const int64_t base = row * D;            // start of this row's D elements
    const int64_t bs = row / n_heads;
    const int64_t cs = bs * D + j;           // front-half cos/sin index

    float x0 = __bfloat162float(x[base + j]);
    float x1 = __bfloat162float(x[base + j + half]);
    float c = __bfloat162float(cos[cs]);
    float s = __bfloat162float(sin[cs]);

    out[base + j]        = __float2bfloat16(x0 * c - x1 * s);
    out[base + j + half] = __float2bfloat16(x1 * c + x0 * s);
  }
}

extern "C" void launch_rope_neox_fwd_bf16(
    const void* x,
    const void* cos,
    const void* sin,
    void* out,
    int64_t n_rows,
    int n_heads,
    int D,
    hipStream_t stream) {
  if (n_rows <= 0 || D <= 0) return;
  const int threads = 256;
  const int64_t total = n_rows * static_cast<int64_t>(D >> 1);  // one thread per pair
  int64_t blocks64 = (total + threads - 1) / threads;
  // Cap grid; the grid-stride loop covers the remainder.
  const int max_blocks = 65535 * 8;
  int blocks = static_cast<int>(blocks64 > max_blocks ? max_blocks : blocks64);
  if (blocks < 1) blocks = 1;
  rope_neox_fwd_bf16_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __hip_bfloat16*>(x),
      reinterpret_cast<const __hip_bfloat16*>(cos),
      reinterpret_cast<const __hip_bfloat16*>(sin),
      reinterpret_cast<__hip_bfloat16*>(out),
      n_rows, n_heads, D);
}

}  // namespace ja_rope
