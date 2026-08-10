# jax-aiter

![Nightly](https://img.shields.io/github/actions/workflow/status/rocm/jax-aiter/nightly-ci.yml?branch=main&label=nightly&logo=github)
![CI](https://img.shields.io/github/actions/workflow/status/rocm/jax-aiter/ci.yml?branch=main&label=ci&logo=github)
![License](https://img.shields.io/github/license/ROCm/jax-aiter)

<img width="256" height="391" alt="jax-aiter-github" src="https://github.com/user-attachments/assets/b30ac891-ce50-4cff-8074-8a42f46ee111" />

JAX-AITER integrates AMD's [AITER](https://github.com/ROCm/aiter) operator library into JAX via XLA FFI, bringing high-performance GPU kernels to JAX on ROCm. No PyTorch dependency at runtime.

Python 3.12 required. ROCm 7.2+.

For the single-node Llama 3.1 8B MaxText reproduction recipe—MXFP4, TE FP8
current scaling, BF16, and MLPerf Training 6.0 convergence logging—see
[`docs/recipes/mxfp4_llama3_8b_maxtext.md`](docs/recipes/mxfp4_llama3_8b_maxtext.md).
The tracked runners support CPU-only recipe resolution before any GPU run.

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
| **FP4 GEMM (training)** | `gemm_fp4_bf16(a, b)` | AITER ASM (35 kernels) | AITER ASM dA + AITER ASM dB (FSDP-aware wgrad sharding) | BF16 in/out with `custom_vjp`. TE-parity MXFP4 recipe; beats native hipBLASLt FP8 by +14.7% at 8B and +6.0% at 70B (8x MI355X). |
| MXFP4 Quantizer / Workspace | `MXFP4Quantizer`, `WeightWorkspace` | -- | -- | TE-parity object API. `MXFP4Quantizer.for_weight/for_activation/for_grad` + `WeightWorkspace.get_or_quantize(w, q, cache_name=...)`. |
| MXFP4 Cast | `CastMxfp4JA` / `CastMxfp4DualJA` | Fused HIP kernel | -- | BF16 to MXFP4 (E2M1 + E8M0 block scales) with transpose + shuffle. |
| FP4 GEMM (low-level) | `gemm_fp4(a, b, a_scale, b_scale)` | AITER ASM | -- | Pre-quantized fp4x2 inputs with e8m0 block scales. |
| BF16 GEMM (training) | `gemm(a, b)` | AITER ASM | AITER ASM dX + hipBLASLt dW | A[M,K] @ B[N,K]^T with `custom_vjp`. 24 hand-tuned kernels. |
| Flash Attention | `flash_attn_func(q, k, v, ...)` | AITER CK/ASM v3 | AITER CK/ASM v3 | MHA/MQA/GQA, causal, SWA, bias, ALiBi, dropout. |
| Flash Attention (varlen) | `flash_attn_varlen(q, k, v, cu_sq, cu_sk, ...)` | AITER CK/ASM v3 | AITER CK/ASM v3 | Packed variable-length sequences. |
| RMSNorm | `rms_norm(x, gamma, epsilon)` | AITER CK | JAX | Fused square, mean, rsqrt, scale. |
| Fused Add+RMSNorm | `rms_norm_with_add(x, residual, gamma, epsilon)` | AITER CK | JAX | `y = rms_norm(x + residual) * gamma` in one kernel. |
| SiLU-and-Mul | `silu_and_mul(x)` | AITER HIP | -- | Fused `silu(x[:half]) * x[half:]` activation. |

## Quick start

```python
from jax_aiter.gemm_fp4 import gemm_fp4_bf16
from jax_aiter.gemm import gemm
from jax_aiter.mha import flash_attn_func
from jax_aiter.rmsnorm import rms_norm, rms_norm_with_add

# FP4 (MXFP4) GEMM: BF16 inputs, FP4 quantization + ASM GEMM, BF16 output.
# Has custom_vjp for training. TE-parity recipe: FP4 fwd + FP4 dA +
# FP4 dB (NT wgrad with FSDP-aware psum sharding). grad_out is cast with
# Hadamard transform for tighter convergence.
# Beats native hipBLASLt FP8 by +14.7% at 8B and +6.0% at 70B on 8x MI355X.
out = gemm_fp4_bf16(activations, weights)

# Object-oriented MXFP4 quantizer (TE-parity, opt-in).
from jax_aiter.gemm_fp4 import MXFP4Quantizer, WeightWorkspace
weight_q = MXFP4Quantizer.for_weight()
w_fp4 = weight_q.quantize(w_bf16)            # Mxfp4Tensor(row + col + scales)
ws = WeightWorkspace()
w_fp4_cached = ws.get_or_quantize(w_bf16, weight_q, cache_name="mlp_gate")

# BF16 GEMM: A[M,K] @ B[N,K]^T using AITER hand-tuned ASM kernels.
out = gemm(a, b)  # bf16 inputs, bf16 output, has custom_vjp for training.

# Attention.
out = flash_attn_func(q, k, v, causal=True)

# RMSNorm.
y = rms_norm(x, gamma, epsilon=1e-6)

# Fused residual add + RMSNorm (one kernel, one memory pass).
y, residual_out = rms_norm_with_add(x, residual, gamma, epsilon=1e-6)
```

## Option A: Install from wheel

Two wheel variants are published:

| Variant | Size | Contents | Use when |
|---------|------|----------|----------|
| **lite** (`+lite`) | ~43 MiB | MXFP4/FP4 GEMM, BF16 GEMM, MXFP4 cast, RMSNorm, SiLU-and-Mul. **No flash attention (MHA).** | MXFP4 training with attention routed through TransformerEngine. |
| **full** | ~513 MiB | Everything above **plus** AITER flash attention (`flash_attn_func` / `flash_attn_varlen`). | You need AITER MHA, including the direct-MHA publication recipe linked above. |

```bash
# Lite (MXFP4 path):
pip install jax_aiter-<version>+lite-*.whl

# Full (adds AITER flash attention):
pip install jax_aiter-<version>-*.whl
```

Importing `jax_aiter.mha` from the lite wheel raises a clear
`ModuleNotFoundError` directing you to the full variant. See
[`docs/RELEASE_NOTES_v0.1.0-alpha.md`](docs/RELEASE_NOTES_v0.1.0-alpha.md) for
the MXFP4 release scope, validation, and MaxText FP4 recipe.

**ROCm JAX runtime.** jax-aiter needs a ROCm-enabled JAX. `pip install jax
jax-rocm7-pjrt jax-rocm7-plugin` currently resolves to **0.9.1** (ABI-compatible
with the FP4 FFI; validated in the clean-container FP4 GEMM smoke). For the
exact build-matched **`0.9.0+rocm7.2.0`** stack, pull the `jax` / `jaxlib` /
`jax-rocm7-pjrt` / `jax-rocm7-plugin` wheels from the
[rocm-jax v0.9.0 release assets](https://github.com/ROCm/rocm-jax/releases/tag/rocm-jax-v0.9.0-rc3)
(published there, not on the pip indexes). The lite wheel's FP4 path is
validated on both 0.9.0 and 0.9.1.

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
python3 -c "from jax_aiter.mha import flash_attn_func; from jax_aiter.gemm_fp4 import gemm_fp4_bf16; from jax_aiter.gemm import gemm; print('OK')"
python3 tests/smoke_gemm_all_test.py
```

Run tests:

```bash
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_nccl_comm_splitting=false --xla_gpu_enable_command_buffer="
pytest -v --reruns 2 tests/test_mha_ja.py tests/test_rmsnorm_ja.py tests/test_gemm_ja.py \
    tests/test_gemm_fp4_ja.py tests/test_silu_and_mul_ja.py
```

## Build wheel

Use `scripts/build_wheel.sh` to stage a lite or full wheel. It **reuses** the
already-built AITER JIT libs in `build/aiter_build/` (it never rebuilds the
multi-GB MHA libs — a sha256 guard aborts the build if they change):

```bash
# Lite (drops MHA libs + shims; carries the +lite local-version tag):
bash scripts/build_wheel.sh --variant lite

# Full (ships everything):
bash scripts/build_wheel.sh --variant full
```

The wheel lands in `dist/`. Run inside the build container (e.g.
`docker exec rv_aiter bash -lc "cd /ruvaidya/aiter_proj/jax-aiter && bash scripts/build_wheel.sh --variant lite"`).

Validate the lite wheel standalone in a clean sibling container (no source
tree, GPU FP4 GEMM smoke):

```bash
bash scripts/validate_wheel.sh   # defaults to the newest dist/*+lite-*.whl
```

Low-level alternative (no variant filtering):

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

- **Symbol not found errors.** Ensure JIT libs are built (`ls build/aiter_build/*.so`). JIT libs must load before FFI modules.
- **Arch mismatch.** Set `GPU_ARCHS` to match your GPU, then rebuild all steps.
- **JIT build fails.** Run with `--verbose` for details: `python3 jax_aiter/jit/build_jit.py --verbose`.

## Developer notes

JIT module config: `jax_aiter/jit/optCompilerConfig.json`.

Available JIT modules:
- `libmha_fwd` / `libmha_bwd` -- MHA forward/backward (CK + ASM v3).
- `librmsnorm_fwd` -- RMSNorm forward (CK).

FFI modules (built by `make ja_mods`):
- `mha_fwd_ja.so` / `mha_bwd_ja.so` -- MHA FFI handlers.
- `rmsnorm_fwd_ja.so` -- RMSNorm FFI handler.
- `silu_and_mul_ja.so` -- SiLU activation FFI handler.
- `gemm_fwd_ja.so` -- BF16 GEMM FFI handler (24 ASM kernels, heuristic selection).
- `gemm_fp4_ja.so` -- FP4 GEMM (35 ASM kernels).
- `cast_mxfp4_ja.so` -- MXFP4 cast + transpose + shuffle (BF16 to E2M1+E8M0).

### GEMM architecture

All GEMM variants bypass AITER's PyTorch wrapper and call the ASM kernels directly via HIP:

```
JAX buffer → FFI handler → KernelArgs struct (void*) → AiterAsmKernel → hipModuleLaunchKernel → .co
```

No PyTorch code at any layer. Kernel configs are auto-generated from CSV by `hsa/codegen.py`.

### MXFP4 training architecture

The MXFP4 path (`gemm_fp4_bf16`) uses `custom_vjp` + `custom_partitioning` for FSDP-compatible training:

```
Forward:  CastMxfp4JA(act) + CastMxfp4DualJA(wt) + GemmFp4FwdJA    (3 FFI calls)
Backward: CastMxfp4JA(grad) + GemmFp4FwdJA(dA)                      (2 FFI calls)
          hipBLASLt FP8 dB via lax.dot_general                       (native XLA)
```

FP4 ASM kernels are 1.19-1.54x faster than hipBLASLt FP8 at MLP shapes. The dB backward
uses native FP8 `dot_general` so XLA can overlap it with FSDP communication.
