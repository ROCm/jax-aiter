# JAX-AITER v0.1.0-alpha2

Alpha2 makes the repository installable and testable on the public TheRock
7.14 GA stack. It targets Python 3.12 and AMD Instinct MI355X (`gfx950`).

This release intentionally contains no throughput or speedup claims. Public
performance results will be published separately in the ROCm blog.

## Runtime stack

- ROCm: TheRock 7.14 GA
- JAX / `jaxlib`: 0.11.0 from PyPI
- `jax-rocm7-plugin` / `jax-rocm7-pjrt`: 0.11.0 from PyPI
- AITER pin: `31350226161346314b3d8882c8085bd31dce6a34`
- GPU architecture: `gfx950`

## Wheels

Two wheels are attached to the GitHub release:

- `jax_aiter-0.1.0a2-...manylinux_2_39_x86_64.whl`
  - default installation;
  - all public APIs and thin FFI shims;
  - gfx950 HSA files only;
  - MHA JIT libraries downloaded on demand.
- `jax_aiter-0.1.0a2+full-...manylinux_2_39_x86_64.whl`
  - GitHub-only convenience artifact;
  - includes the MHA JIT libraries;
  - intended for environments that cannot download runtime assets later.

The `manylinux_2_39` tag reports what the wheels actually require: the bundled
AITER JIT libraries are built on Ubuntu 24.04 and need `GLIBCXX_3.4.31`. Broad
manylinux compatibility is not an alpha2 goal, and the honest tag makes `pip`
decline an unusable install rather than fail later at import. Alpha2 targets
ROCm 7.14 on Ubuntu 24.04, which satisfies it.

The release workflow builds in the TheRock manylinux image, verifies that the
tag matches the measured symbol floor, checks wheel contents, then validates
both variants on an MI355 before upload.

## Installation

Install the ROCm JAX stack:

```bash
python3 -m pip install \
  "jax==0.11.0" "jaxlib==0.11.0" \
  "jax-rocm7-plugin==0.11.0" "jax-rocm7-pjrt==0.11.0"
```

Install the downloaded default wheel:

```bash
python3 -m pip install \
  ./jax_aiter-0.1.0a2-cp312-cp312-manylinux_2_39_x86_64.whl
```

Add flash attention when needed:

```bash
jax-aiter-fetch-mha
python3 -c "from jax_aiter.mha import flash_attn_func; print('MHA ready')"
```

The fetch command validates the release manifest, architecture, compressed
checksum, and extracted checksum. It installs into a versioned user cache, not
system `site-packages`.

## Repository changes

- CI uses the real `build-only-jax-aiter` and 1/4-GPU MI355 runner labels.
- JIT libraries rebuild only when the AITER pin, architecture, or JIT recipe
  changes; unrelated integration patches do not invalidate them.
- PR and nightly tests are selected by `gpu`, `multigpu`, and `slow` markers.
- The default wheel is gfx950-only and omits only the large MHA JIT libraries.
- HIP failures now print the HIP error, source file, and line before aborting.
- Dead compatibility helpers, unbuildable registry entries, and the
  experimental unbuilt FP8 GEMM module were removed.
- The Makefile supports TheRock's `rocm-sdk` layout without `/opt/rocm`.

## MaxText recipe

The MXFP4 and direct-MHA recipe is documented at
[`docs/recipes/mxfp4_llama3_8b_maxtext.md`](recipes/mxfp4_llama3_8b_maxtext.md).
Its MaxText source is pinned to:

```text
ROCm/maxtext aiter-fp4-integration
ccd72e63e57193c6f1d51b06bd2e7f52ce895404
```

The repository also carries a patch against that exact revision, checked by
CPU CI.

## Limitations

- Alpha2 supports `gfx950` only.
- Python 3.12 is required.
- The default wheel needs `jax-aiter-fetch-mha` before importing
  `jax_aiter.mha`.
- PyPI publication is a separate, explicitly approved step. Until the release
  notes say otherwise, install the wheel from the GitHub release.
