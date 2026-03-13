# jax-aiter

![Nightly](https://img.shields.io/github/actions/workflow/status/rocm/jax-aiter/nightly-ci.yml?branch=main&label=nightly&logo=github)
![CI](https://img.shields.io/github/actions/workflow/status/rocm/jax-aiter/ci.yml?branch=main&label=ci&logo=github)
![License](https://img.shields.io/github/license/ROCm/jax-aiter)

<img width="256" height="391" alt="jax-aiter-github" src="https://github.com/user-attachments/assets/b30ac891-ce50-4cff-8074-8a42f46ee111" />

JAX-AITER integrates AMD's [AITER](https://github.com/ROCm/aiter) operator library into JAX via XLA FFI, bringing high-performance GPU kernels to JAX on ROCm. No PyTorch dependency at runtime.

Status: experimental. Python 3.12 required.

## What is AITER?

**AITER** (AI Tensor Engine for ROCm) is AMD's centralized library of AI operators optimized for ROCm GPUs (MI300, MI350). It provides hand-tuned CK (Composable Kernel) and ASM kernels for attention, normalization, activations, GEMM, and more.

**JAX-AITER** provides:

- **JAX-native API.** Operators exposed as JAX functions with `custom_vjp` gradient wiring.
- **Zero-copy FFI.** GPU buffers passed directly between JAX and AITER via XLA FFI.
- **Training-ready.** Gradients flow through AITER kernels for end-to-end training.
- **No torch dependency.** Pure JAX + AITER at runtime.

## Supported ops

| Op | API | Forward | Backward | Notes |
|----|-----|---------|----------|-------|
| Flash Attention | `flash_attn_func(q, k, v, ...)` | AITER CK/ASM v3 | AITER CK/ASM v3 | MHA/MQA/GQA, causal, SWA, bias, ALiBi, dropout. |
| Flash Attention (varlen) | `flash_attn_varlen(q, k, v, cu_sq, cu_sk, ...)` | AITER CK/ASM v3 | AITER CK/ASM v3 | Packed variable-length sequences. |
| RMSNorm | `rms_norm(x, gamma, epsilon)` | AITER CK | JAX | Fused square, mean, rsqrt, scale. |
| Fused Add+RMSNorm | `rms_norm_with_add(x, residual, gamma, epsilon)` | AITER CK | JAX | `y = rms_norm(x + residual) * gamma` in one kernel. |

## Quick start

```python
from jax_aiter.mha import flash_attn_func
from jax_aiter.rmsnorm import rms_norm, rms_norm_with_add

# Attention.
out = flash_attn_func(q, k, v, causal=True)

# RMSNorm.
y = rms_norm(x, gamma, epsilon=1e-6)

# Fused residual add + RMSNorm (one kernel, one memory pass).
y, residual_out = rms_norm_with_add(x, residual, gamma, epsilon=1e-6)
```

## Option A: Install from wheel

```bash
pip install path/to/jax_aiter-<version>-*.whl
```

## Option B: Build from source

Requires ROCm, `hipcc`, and JAX with ROCm support.

```bash
pip install cmake ninja pyyaml
```

### 1) Clone with submodules

```bash
git clone --recursive git@github.com:ROCm/jax-aiter.git
cd jax-aiter
```

### 2) Environment setup

```bash
export JA_ROOT_DIR="$PWD"
export AITER_SYMBOL_VISIBLE=1
export GPU_ARCHS=gfx950                                    # gfx942 for MI300, gfx950 for MI350.
export AITER_ASM_DIR="$JA_ROOT_DIR/third_party/aiter/hsa/" # Base path, no arch suffix.
```

### 3) Build umbrella shared library

```bash
make
```

### 4) Build AITER JIT modules

```bash
python3 jax_aiter/jit/build_jit.py
```

Build specific modules:

```bash
python3 jax_aiter/jit/build_jit.py --module libmha_fwd,libmha_bwd,librmsnorm_fwd
```

### 5) Build FFI modules

```bash
make ja_mods
```

### 6) Install and test

```bash
pip install .
```

Smoke test:

```bash
python3 -c "from jax_aiter.mha import flash_attn_func, flash_attn_varlen; from jax_aiter.rmsnorm import rms_norm; print('OK')"
```

Run tests:

```bash
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1"
pytest -v --reruns 2 tests/test_mha_ja.py tests/test_rmsnorm_ja.py
```

## Build wheel

```bash
pip wheel . --no-deps -w dist/
```

## GPU architectures

| GPU | Architecture | `GPU_ARCHS` |
|-----|-------------|-------------|
| MI300 series | CDNA3 | `gfx942` |
| MI350 series | CDNA4 | `gfx950` |

Multiple architectures: `GPU_ARCHS="gfx942;gfx950"`.

## Troubleshooting

- **Symbol not found errors.** Ensure `libmha_fwd.so`, `libmha_bwd.so`, `librmsnorm_fwd.so` are built (`ls build/aiter_build/*.so`). JIT libs must load before FFI modules.
- **Arch mismatch.** Set `GPU_ARCHS` to match your GPU, then rebuild all steps.
- **JIT build fails.** Run with `--verbose` for details: `python3 jax_aiter/jit/build_jit.py --verbose`.

## Developer notes

JIT module config: `jax_aiter/jit/optCompilerConfig.json`.

Available modules:
- `libmha_fwd` / `libmha_bwd` -- MHA forward/backward (CK + ASM v3).
- `librmsnorm_fwd` -- RMSNorm forward (CK).
