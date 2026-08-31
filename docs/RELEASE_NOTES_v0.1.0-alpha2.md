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

Two wheels are attached to the GitHub release. **Most users want the `+full`
wheel**, which is complete on its own:

- `jax_aiter-0.1.0a2+full-...manylinux_2_39_x86_64.whl` — **recommended**
  - everything in one download, including the MHA JIT libraries;
  - nothing to fetch after installing;
  - roughly 433 MB, about 2.5 GB installed.
- `jax_aiter-0.1.0a2-...manylinux_2_39_x86_64.whl` — small alternative
  - all public APIs, thin FFI shims, and gfx950 HSA files;
  - omits the two multi-GB MHA JIT libraries, so it is roughly 30 MB;
  - `jax-aiter-fetch-mha` downloads those libraries on first use;
  - carries the plain version because it is the PyPI-shaped artifact, though
    alpha2 is not published to PyPI. Prefer it when download size or image
    size matters more than a self-contained install.

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
  "jax-rocm7-plugin==0.11.0.post1" "jax-rocm7-pjrt==0.11.0.post1"
```

Install the recommended `+full` wheel, which needs no follow-up download:

```bash
python3 -m pip install \
  ./jax_aiter-0.1.0a2+full-cp312-cp312-manylinux_2_39_x86_64.whl
python3 -c "from jax_aiter.mha import flash_attn_func; print('MHA ready')"
```

If you took the small wheel instead, add flash attention separately:

```bash
python3 -m pip install \
  ./jax_aiter-0.1.0a2-cp312-cp312-manylinux_2_39_x86_64.whl
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
- Both wheels are gfx950-only; the smaller one omits just the MHA JIT libraries.
- HIP failures now print the HIP error, source file, and line before aborting.
- Dead compatibility helpers, unbuildable registry entries, and the
  experimental unbuilt FP8 GEMM module were removed.
- The Makefile supports TheRock's `rocm-sdk` layout without `/opt/rocm`.

## MaxText recipe

The MXFP4 and direct-MHA recipe is documented at
[`docs/recipes/mxfp4_llama3_8b_maxtext.md`](recipes/mxfp4_llama3_8b_maxtext.md).
Its MaxText source is pinned to:

```text
ROCm/maxtext feature/jax-aiter-mxfp4-v26.6
b437942a5f33704f8438deb948488ad08164285c
```

The older standalone patch remains checked by CPU CI for its original
`aiter-fp4-integration` post-image; the blog recipe uses the public branch
commit above directly.

## Limitations

- Alpha2 supports `gfx950` only.
- Python 3.12 is required.
- The smaller plain wheel needs `jax-aiter-fetch-mha` before importing
  `jax_aiter.mha`. The recommended `+full` wheel does not.
- The paged-KV modules under `jax_aiter.kv` ship as source only. The release
  build runs `make ja_mods`, never `make -f Makefile.kv ja_kv`, so the wheel
  carries their Python but none of their FFI libraries. `import jax_aiter.kv`
  therefore succeeds and gives no warning; the first call that reaches a paged
  kernel raises `FileNotFoundError: Standalone module not found:
  append_kv_ja.so ... Run 'make -f Makefile.kv ja_kv' first.` They are not part
  of the validated alpha2 surface; build them from a checkout to use them.
- PyPI publication is a separate, explicitly approved step. Until the release
  notes say otherwise, install the wheel from the GitHub release.
