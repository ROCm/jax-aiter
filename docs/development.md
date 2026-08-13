# Development and testing

## Environment

The supported alpha2 build environment is:

```text
ghcr.io/rocm/jax-dev-ubu24.therock-7.14:7.14
Python 3.12
JAX / jaxlib / ROCm plugin / PJRT 0.11.0
GPU_ARCHS=gfx950
```

TheRock has no `/opt/rocm`. The Makefile uses `/opt/rocm` in legacy images and
falls back to `rocm-sdk path --root` plus `hipcc` from `PATH` in TheRock.
Do not add an rpath to `_rocm_sdk_devel/lib`; TheRock resolves its runtime
libraries through `ldconfig` and package-relative RUNPATHs.

## Build targets

```bash
export JA_ROOT_DIR="$PWD"
export GPU_ARCHS=gfx950
export AITER_SYMBOL_VISIBLE=1

make
python3 jax_aiter/jit/build_jit.py
make ja_mods
```

Targets:

- `make`: umbrella `libjax_aiter.so`;
- `make ja_mods`: core and MHA FFI shims;
- `make ja_mods_nomha`: core shims only;
- `build_jit.py`: `librmsnorm_fwd.so`, `libmha_fwd.so`,
  `libmha_bwd.so`.

Do not run `make clean` casually. It removes multi-hour JIT outputs.
`make clean-stage` removes wheel staging without deleting
`build/aiter_build`.

## JIT-library producer

`.github/workflows/jit-libs.yml` is the only CI job allowed to build the AITER
JIT libraries. It runs on `build-only-jax-aiter`; offline gfx950 compilation
was verified with no `/dev/kfd`.

The manifest hard-gates reuse on:

- AITER submodule commit;
- GPU architecture list;
- hash of `jax_aiter/jit/build_jit.py`,
  `jax_aiter/jit/optCompilerConfig.json`, and any explicitly named
  `scripts/aiter_jit_*.patch`.

ROCm version is recorded as advisory provenance. Consumer jobs fetch the
release assets and fail on a cache miss instead of starting a hidden rebuild.

## Test tiers

`ci/test.sh` selects a tier through `JA_TEST_TIER`:

```bash
JA_TEST_TIER=pr bash ci/test.sh
JA_TEST_TIER=nightly bash ci/test.sh
JA_TEST_TIER=multigpu bash ci/test.sh
```

| Tier | Selection | Runner |
|---|---|---|
| `pr` | `not slow and not multigpu` | 1 × MI355 |
| `nightly` | full suite | 4 × MI355 |
| `multigpu` | FSDP/sharding cases | 4 × MI355 |

The `slow` marker is for exhaustive shape sweeps and production-size
projection shapes, not for ordinary correctness tests. `multigpu` tests retain
device-count skip guards as a safety net.

CI tooling tests are CPU-only and self-runnable:

```bash
python3 ci/test_jit_libs_manifest.py
python3 ci/perf/test_parse_perf_log.py
```

## Building wheels

The wheel script never invokes `build_jit.py` and verifies the JIT-library
hashes before and after staging:

```bash
# Default wheel: public version, MHA JIT libraries fetched later.
GPU_ARCHS=gfx950 JA_WHEEL_ARCH=gfx950 \
  bash scripts/build_wheel.sh --variant lite

# GitHub-only wheel containing the MHA JIT libraries.
GPU_ARCHS=gfx950 JA_WHEEL_ARCH=gfx950 \
  bash scripts/build_wheel.sh --variant full
```

Release wheels are built inside
`ghcr.io/rocm/jax-manylinux_2_28-therock-7.14:7.14` with:

```bash
export JA_WHEEL_PLAT_NAME=manylinux_2_39_x86_64
```

The tag is `2_39` even though the image is `2_28`, because the bundled AITER JIT
libraries come from `jit-libs.yml`, which builds them on Ubuntu 24.04; they need
`GLIBCXX_3.4.31` no matter which image packages the wheel.

The release workflow runs `auditwheel show`, asserts the platform tag is not
weaker than the measured symbol floor, checks wheel contents, then clean-room
validates both variants on one MI355 before upload.

## Performance and convergence

Repository README, recipes, and release notes do not publish throughput or
speedup claims. Raw measurements belong in run artifacts and CI output; public
claims are reserved for the ROCm blog.

A code change that affects numerics, optimizer state, quantization policy, or
the model recipe needs an explicit convergence decision. Refactors that cannot
execute, test-only changes, packaging changes, and error-message changes do not
require rerunning convergence.
