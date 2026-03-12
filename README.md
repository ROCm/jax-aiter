# jax-aiter

![Build Status](https://img.shields.io/github/actions/workflow/status/rocm/jax-aiter/ci.yml?branch=main&label=ci&logo=github)
![License](https://img.shields.io/github/license/ROCm/jax-aiter)

JAX-AITER integrates AMD's AITER operator library into JAX via XLA FFI, enabling high-performance attention kernels on ROCm GPUs for both inference and training.

**Status:** Experimental. Python 3.12 required.

## Features

- **JAX-native API:** `flash_attn_func` and `flash_attn_varlen` with `custom_vjp` for automatic differentiation
- **Unified kernel dispatch:** Single AITER entry point (`aiter::mha_fwd`/`aiter::mha_bwd`) handles CK vs ASM v3 kernel selection internally
- **Zero-copy FFI:** Raw GPU buffer pointers passed directly via JAX FFI -- no PyTorch dependency
- **Training-ready:** Gradients flow through AITER kernels for end-to-end differentiable pipelines
- **ROCm-native:** Targets MI300 (gfx942) and MI350 (gfx950) GPUs

## Quick start

### Option A: Install from wheel

```bash
pip install jax_aiter-<version>-*.whl
```

### Option B: Build from source

#### Prerequisites

- ROCm 7.2.0+
- Python 3.12
- JAX 0.9.0+ with ROCm support
- `cmake`, `ninja`, `pyyaml`, `psutil`, `pandas`

```bash
pip install cmake ninja pyyaml psutil pandas
```

#### 1. Clone and set up environment

```bash
git clone --recursive git@github.com:ROCm/jax-aiter.git
cd jax-aiter

export JA_ROOT_DIR="$PWD"
export AITER_SYMBOL_VISIBLE=1
export GPU_ARCHS=gfx950                          # or gfx942, or "gfx942;gfx950"
export AITER_ASM_DIR="$JA_ROOT_DIR/third_party/aiter/hsa/"
```

#### 2. Build

```bash
# Umbrella shared library (requires JAX installed for FFI headers)
make

# AITER JIT modules (libmha_fwd.so, libmha_bwd.so -- ~30-40 min first time)
python3 jax_aiter/jit/build_jit.py

# JAX-AITER FFI modules (mha_fwd_ja.so, mha_bwd_ja.so -- ~2 min)
make ja_mods
```

#### 3. Install

```bash
pip install --break-system-packages third_party/aiter
pip install --break-system-packages .
```

#### 4. Verify

```bash
python3 -c "from jax_aiter.mha import flash_attn_func, flash_attn_varlen; print('OK')"
```

## Usage

```python
import jax
import jax.numpy as jnp
from jax_aiter.mha import flash_attn_func, flash_attn_varlen

# Batch attention
q = jnp.ones((2, 128, 8, 64), dtype=jnp.bfloat16)  # [batch, seq, heads, dim]
k = jnp.ones((2, 128, 8, 64), dtype=jnp.bfloat16)
v = jnp.ones((2, 128, 8, 64), dtype=jnp.bfloat16)

out = flash_attn_func(q, k, v, causal=True)

# Gradients work naturally
grad_fn = jax.grad(lambda q, k, v: jnp.sum(flash_attn_func(q, k, v, causal=True)[0]))
dq, dk, dv = grad_fn(q, k, v)
```

### Supported features

| Feature | Status |
|---------|--------|
| MHA / MQA / GQA | Supported |
| Causal masking | Supported |
| Variable-length (packed sequences) | Supported |
| Sliding window attention (SWA) | Supported |
| Attention bias | Supported |
| ALiBi slopes | Supported |
| Dropout (forward) | Supported |
| Head dims 32-256 | Supported |
| float16 / bfloat16 | Supported |
| Deterministic backward | Supported |

## Testing

```bash
# Run full test suite (218 tests, ~45 seconds, 1 GPU)
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_nccl_comm_splitting=false --xla_gpu_enable_command_buffer="

pytest -v tests/test_mha_ja.py
```

## Architecture

```
Python: flash_attn_func() / flash_attn_varlen()
    |  @jax.custom_vjp, padding, flag-setting
    v
FFI: MhaFwdUnifiedJA / MhaBwdUnifiedJA  (2 .so files)
    |  4D batch vs 3D varlen via tensor rank detection
    |  Builds mha_fwd_args / mha_bwd_args (raw void* pointers)
    v
AITER: aiter::mha_fwd(args, stream) / aiter::mha_bwd(args, stream)
    |  Internal CK vs ASM v3 dispatch
    v
GPU Kernel (MI300 / MI350)
```

## Build outputs

| File | Description |
|------|-------------|
| `build/jax_aiter_build/libjax_aiter.so` | Umbrella library (shared utilities) |
| `build/aiter_build/libmha_fwd.so` | AITER forward kernels (~900MB) |
| `build/aiter_build/libmha_bwd.so` | AITER backward kernels (~1GB) |
| `build/jax_aiter_build/mha_fwd_ja.so` | FFI forward handler |
| `build/jax_aiter_build/mha_bwd_ja.so` | FFI backward handler |

## GPU architectures

| GPU | Architecture | `GPU_ARCHS` |
|-----|-------------|-------------|
| MI300 | CDNA3 | `gfx942` |
| MI350 | CDNA4 | `gfx950` |

## Troubleshooting

- **`ModuleNotFoundError: No module named 'jax'` during `make`:**
  JAX must be installed before building. The Makefile resolves FFI include paths at parse time.

- **Symbol not found errors:**
  Ensure `libmha_fwd.so` and `libmha_bwd.so` are built (`python3 jax_aiter/jit/build_jit.py`) before running.

- **JIT build takes 30+ minutes:**
  Normal for first build -- compiles ~19K kernel variants for forward, ~8K for backward. Cached on subsequent runs.

## Developer notes

- JIT module config: `jax_aiter/jit/optCompilerConfig.json`
  - `libmha_fwd` -- forward kernels (CK + ASM v3, `torch_exclude: True`)
  - `libmha_bwd` -- backward kernels (CK + ASM v3, `torch_exclude: True`)
- FFI handlers: `csrc/ffi/mha_fwd/mha_fwd_ja.cu`, `csrc/ffi/mha_bwd/mha_bwd_ja.cu`
- No PyTorch dependency at build or runtime
