# JAX-AITER

![Nightly](https://img.shields.io/github/actions/workflow/status/ROCm/jax-aiter/nightly-ci.yml?branch=main&label=nightly&logo=github)
![CI](https://img.shields.io/github/actions/workflow/status/ROCm/jax-aiter/ci.yml?branch=main&label=ci&logo=github)
![License](https://img.shields.io/github/license/ROCm/jax-aiter)

JAX-AITER exposes selected
[AITER](https://github.com/ROCm/aiter) GPU kernels to JAX through XLA FFI.
The public APIs are JAX functions with `custom_vjp` and sharding rules for
training. PyTorch is not a runtime dependency.

Alpha2 targets one tested stack:

| Component | Supported version |
|---|---|
| GPU | AMD Instinct MI355X (`gfx950`) |
| ROCm | TheRock 7.14 GA |
| Python | 3.12 |
| JAX / `jaxlib` | 0.11.0 |
| ROCm plugin / PJRT | 0.11.0 |

`gfx942` is not included in alpha2 because no available CI runner can test it.

## Install

### 1. Start from TheRock 7.14 GA

The runtime-only image is enough for wheel users:

```bash
docker pull ghcr.io/rocm/jax-base-ubu24.therock-7.14:7.14
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --ipc=host --shm-size=16G --group-add video \
  --security-opt seccomp=unconfined \
  ghcr.io/rocm/jax-base-ubu24.therock-7.14:7.14 bash
```

TheRock packages ROCm through `rocm-sdk`; the image deliberately has no
`/opt/rocm`.

### 2. Install JAX from PyPI

JAX and `jaxlib` come from upstream PyPI. The ROCm plugin and PJRT wheels also
come from PyPI and use the TheRock `jax_plugins.xla_rocm7` backend:

```bash
python3 -m pip install \
  "jax==0.11.0" "jaxlib==0.11.0" \
  "jax-rocm7-plugin==0.11.0" "jax-rocm7-pjrt==0.11.0"

python3 -c "import jax; print(jax.devices())"
```

### 3. Install JAX-AITER

Download the alpha2 wheel from the
[GitHub release](https://github.com/ROCm/jax-aiter/releases/tag/v0.1.0-alpha2),
then install it:

```bash
python3 -m pip install \
  ./jax_aiter-0.1.0a2-cp312-cp312-manylinux_2_28_x86_64.whl
```

The default wheel is `gfx950`-only and contains all APIs, FFI shims, and
non-MHA runtime libraries. PyPI publication is intentionally not automated yet;
the release notes will state when `pip install jax-aiter==0.1.0a2` is available.

Verify the basic install:

```bash
python3 - <<'PY'
import jax
from jax_aiter.gemm import gemm
from jax_aiter.gemm_fp4 import gemm_fp4_bf16
from jax_aiter.rmsnorm import rms_norm

print("devices:", jax.devices())
print("JAX-AITER imports: OK")
PY
```

### 4. Add flash attention when needed

The two MHA JIT libraries are omitted from the default wheel because they
expand to multiple gigabytes. Download the matching, checksummed `gfx950`
artifacts instead of compiling them:

```bash
jax-aiter-fetch-mha
python3 -c "from jax_aiter.mha import flash_attn_func; print('MHA ready')"
```

The command installs into a versioned user cache
(`~/.cache/jax-aiter/0.1.0a2/`) and verifies the architecture plus compressed
and extracted SHA-256 checksums. It does not write into system
`site-packages`.

## Quick API examples

```python
import jax.numpy as jnp

from jax_aiter.activation import silu_and_mul
from jax_aiter.gemm import gemm
from jax_aiter.gemm_fp4 import gemm_fp4_bf16
from jax_aiter.rmsnorm import rms_norm, rms_norm_with_add

# A[M,K] @ B[N,K]^T. Inputs and output are BF16.
y_bf16 = gemm(a_bf16, b_bf16)

# BF16 inputs, MXFP4 casts and AITER FP4 GEMMs, BF16 output.
# custom_vjp supplies FP4 dA and FP4 dB/wgrad.
y_fp4 = gemm_fp4_bf16(a_bf16, b_bf16)

y_norm = rms_norm(x_bf16, gamma_bf16, epsilon=1e-6)
y_fused, residual_out = rms_norm_with_add(
    x_bf16, residual_bf16, gamma_bf16, epsilon=1e-6
)

y_silu = silu_and_mul(jnp.concatenate([gate_bf16, up_bf16], axis=-1))
```

After `jax-aiter-fetch-mha`:

```python
from jax_aiter.mha import flash_attn_func, flash_attn_varlen

out = flash_attn_func(q, k, v, causal=True)
```

## Supported operations

| Operation | Public API | Training behavior |
|---|---|---|
| BF16 GEMM | `gemm(a, b)` | AITER ASM forward and gradients |
| MXFP4 GEMM | `gemm_fp4_bf16(a, b)` | FP4 forward, dA, and FSDP-aware dB/wgrad |
| Pre-quantized FP4 GEMM | `gemm_fp4(...)` | Low-level forward |
| MXFP4 quantization | `MXFP4Quantizer`, `WeightWorkspace` | Row/column layouts and weight caching |
| Flash attention | `flash_attn_func`, `flash_attn_varlen` | Batch/varlen `custom_vjp`; optional download |
| RMSNorm | `rms_norm`, `rms_norm_with_add` | AITER forward, JAX backward |
| SiLU-and-Mul | `silu_and_mul` | Fused forward with `custom_vjp` |

The MXFP4 backward path uses `GemmFp4FwdJA` for both gradients. The dB/wgrad
partition contracts the FSDP-sharded batch axis and emits `jax.lax.psum`.

## MaxText recipes

- [Llama 3.1 8B MXFP4 training](docs/recipes/mxfp4_llama3_8b_maxtext.md)
- Direct JAX-AITER attention is selected with `attention=aiter_flash`.
- Set `aiter_attention=False` with an explicit alternate attention mode to
  roll back without changing the MXFP4 linear path.

The recipe documentation records commands and provenance. Performance results
are intentionally not claimed in repository documentation; publication results
belong in the accompanying ROCm blog.

## Build from source

Wheel installation is the supported fast path. For development:

```bash
git clone --recursive https://github.com/ROCm/jax-aiter.git
cd jax-aiter

export JA_ROOT_DIR="$PWD"
export GPU_ARCHS=gfx950
export AITER_SYMBOL_VISIBLE=1

make
python3 jax_aiter/jit/build_jit.py
make ja_mods
JA_WHEEL_VARIANT=full python3 -m pip install .
```

The full JIT build is intentionally not part of ordinary CI jobs. Prebuilt
libraries are generated only when the pinned AITER revision or JIT recipe
changes.

See:

- [Architecture](docs/architecture.md)
- [Development and testing](docs/development.md)
- [Release playbook](docs/release/RELEASE_PLAYBOOK.md)

## Troubleshooting

- **`jax-aiter-fetch-mha` is suggested on import:** run that command once for
  the installed package version.
- **Architecture mismatch:** alpha2 supports `gfx950` only.
- **No ROCm device in JAX:** verify the `jax-rocm7-plugin` and
  `jax-rocm7-pjrt` versions match JAX, then run `python3 -c "import jax;
  print(jax.devices())"`.
- **Source build cannot find `hipcc`:** use the TheRock dev image,
  `ghcr.io/rocm/jax-dev-ubu24.therock-7.14:7.14`. The Makefile resolves its
  compiler through `rocm-sdk`.
- **HIP process abort:** JAX-AITER prints the HIP error, source file, and line
  before aborting.

## License

MIT. See [LICENSE](LICENSE).
