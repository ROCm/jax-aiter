# jax-aiter

![Nightly Passing](https://img.shields.io/github/actions/workflow/status/rocm/jax-aiter/nightly-ci.yml?branch=main&label=nightly&logo=github) ![Build Status](https://img.shields.io/github/actions/workflow/status/rocm/jax-aiter/ci.yml?branch=main&label=ci&logo=github)
![License](https://img.shields.io/github/license/ROCm/jax-aiter)

JAX-AITER integrates AMD's AITER operator library into JAX, bringing AITER's high-performance attention kernels to JAX on ROCm via a stable FFI bridge and custom_vjp integration. It enables optimized MHA/FMHA (including variable-length attention) in JAX for both inference and training on AMD GPUs.

Status: experimental.

## AITER integration for JAX

**AITER** is AMD’s centralized library of AI operators optimized for ROCm GPUs.
It unifies multiple backends (C++, Python, CK, assembly, etc.) and exposes a consistent operator interface.

**JAX-AITER** builds on that foundation by providing:

- **A JAX-native API:** Operators are exposed as JAX functions with seamless `custom_vjp` (forward/backward) wiring.
- **Automatic operator dispatch:** Dynamically chooses optimized implementations based on tensor shapes, data type, and options (e.g. causal, windowed).
- **Zero-copy buffer exchange:** Uses JAX FFI to avoid unnecessary transfers between JAX and AITER.
- **Training-ready:** Gradients flow natively through AITER kernels, enabling end-to-end differentiable pipelines.
- **ROCm-native performance:** Fully integrated with AMD GPU runtime and compiler stack.

## Option A: Install from a released wheel

If a wheel is available (e.g. from project Releases):

```bash
pip install path/to/jax_aiter-<version>-*.whl
```

## Option B: Build from source (Docker optional)

Custom build requires cmake and ninja (both installable via pip):
```bash
pip install cmake ninja pyyaml
```

Environment setup (run from the top of the jax-aiter project tree):
```bash
export JA_ROOT_DIR="$PWD"                    # Set to the top of jax-aiter project tree
export AITER_SYMBOL_VISIBLE=1
export GPU_ARCHS=gfx950                      # Example for MI350; use your GPU arch (e.g., gfx942 for MI300)
export AITER_ASM_DIR=/aiter-hsa-path/gfx950/ # Example for MI350
```

You can build natively or inside a ROCm container. You can pull docker images from the latest release of ROCm jax.

https://hub.docker.com/r/rocm/jax/tags

We suggest to use latest jax docker images:

```bash
docker pull rocm/jax:rocm7.0.2-jax0.6.0-py3.10-ubu22
```

Inside the container (or on your host with ROCm installed), proceed:

### 1) Clone the repository and init submodules

```bash
git clone --recursive git@github.com:ROCm/jax-aiter.git
```

Submodules:
- third_party/aiter
- third_party/pytorch

### 2) Build static PyTorch for ROCm

Statically build minimal PyTorch libraries (c10, torch_cpu, torch_hip, caffe2_nvrtc) and headers for linking.

Apply the caffe2_nvrtc static/PIC patch:

```bash
cd third_party/pytorch
git apply ../../scripts/torch_caffe.patch
cd -

cd third_party/aiter
git apply ../../scripts/aiter_torch_remove.patch
cd -
```

Run the static build script:

```bash
bash ./scripts/build_static_pytorch.sh
```

Script details:
- Source: third_party/pytorch
- Build dir: third_party/pytorch/build_static
- Install prefix: third_party/pytorch/build_static/install
- Tunables: ROCM_ARCH (gpu arch), ROCM_PATH (/opt/rocm), JOBS (nproc), PYTHON (python3)

### 3) Build the umbrella shared library

Link the required static PyTorch objects and ROCm libs into a single .so:

```bash
make
```

Key paths (from Makefile):
- Output: build/aiter_build/libjax_aiter.so
- Static libs: third_party/pytorch/build_static/lib
- Include dirs: JAX FFI, PyTorch, and csrc/common are used


### 4) Build AITER JIT modules

Build specific AITER modules (example: varlen fwd+bwd):

```bash
python3 jax_aiter/jit/build_jit.py --module module_fmha_v3_varlen_fwd,module_fmha_v3_varlen_bwd
```

Build all configured AITER modules:

```bash
python3 jax_aiter/jit/build_jit.py
```

Outputs (.so) are placed under build/aiter_build/.

Notes:
- build_jit.py doesn't use any "jit" atm, but in future we may change that to do so.
- build_jit.py patches AITER's core to redirect user JIT dir to build/aiter_build and inject PyTorch/JAX-FFI include paths.
- Ensure static PyTorch build completed first; headers and libs expected under third_party/pytorch/build_static and build_static/install/include.

### 5) Build JAX-AITER modules

Build JAX-AITER frontend modules that bridge JAX FFI to AITER:

```bash
make ja_mods
```

Outputs (.so) are placed under build/jax_aiter_build/.

Notes:
- JA modules are thin host-only frontends that call into AITER implementations
- They must be built after the umbrella and AITER modules

### 6) Test / Verify

Smoke test:

```bash
python -c "from jax_aiter.mha import flash_attn_func, flash_attn_varlen; print('jax-aiter import OK')"
```

Run tests (requires JAX ROCm, Pytorch and GPU):

Pytorch is a test time dependency and requires to test with the baseline.

```Bash
pip install --find-links https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0/torch==2.8.0+rocm7.0.0.git64359f59
python3 setup.py # both aiter and jax aiter.
pip install pytest-xdist
```

```bash
pytest -q tests/test_mha_varlen_ja.py
pytest -q tests/test_mha_ck_ja.py
```

## Troubleshooting

- **Arch/driver mismatch:**
  Set both GPU_ARCHS (e.g., gfx950 for MI350) and ROCM_ARCH for the static PyTorch build, then rebuild:
  ```bash
  export GPU_ARCHS=<gfx*>
  env ROCM_ARCH=<gfx*> ./scripts/build_static_pytorch.sh
  make
  ```

  - **caffe2_nvrtc not found or not PIC:**
  Ensure the patch (scripts/torch_caffe.patch) was applied, then rerun build_static_pytorch.sh

- **JIT cannot find PyTorch headers:**
  Confirm third_party/pytorch/build_static/install/include exists. Re-run:
  ```bash
  python3 jax_aiter/jit/build_jit.py --verbose --module module_fmha_v3_varlen_fwd
  ```

- **Symbol not found errors while loading .sos for MHA kernels**
  Confitm that libmha_fwd and libmha_bwd are built and loaded before loading the respective modules.

## Developer notes

- Static PyTorch build targets:
  - c10, torch_cpu, torch_hip, caffe2_nvrtc (static, PIC)
- JIT module config: see jax_aiter/jit/optCompilerConfig.json for available modules:
  - module_fmha_v3_varlen_fwd, module_fmha_v3_varlen_bwd
  - module_mha_varlen_fwd, module_mha_varlen_bwd
  - module_mha_fwd, module_mha_bwd
  - libmha_fwd, libmha_bwd
  - module_custom
