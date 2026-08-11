# JAX-AITER release playbook

This is the alpha2 release procedure. It is intentionally short: commands live
in workflows and scripts rather than being duplicated here.

## 1. Release contract

- Version: `0.1.0a2`
- Tag: `v0.1.0-alpha2`
- Python: 3.12
- ROCm: TheRock 7.14 GA
- GPU: gfx950 / MI355X only
- JAX / `jaxlib` / ROCm plugin / PJRT: 0.11.0
- Default wheel: public version; MHA JIT libraries fetched on demand
- Full wheel: `+full`; GitHub release only
- No performance or speedup claims in README, recipes, or release notes

Stop if any pin or validation result differs. Do not silently substitute a
ROCm image, JAX wheel, GPU architecture, or memory fraction.

## 2. Confirm source state

```bash
git status --short --branch
git submodule status third_party/aiter
python3 -c "from jax_aiter.__version__ import __version__; print(__version__)"
```

Expected:

```text
jax-aiter version: 0.1.0a2
AITER: 31350226161346314b3d8882c8085bd31dce6a34
```

The PyTorch submodule must not exist in `.gitmodules`; JIT builds have no
PyTorch header or runtime dependency.

## 3. Build the rolling JIT assets when needed

Run `.github/workflows/jit-libs.yml` only when one of these hard cache inputs
changes:

- AITER submodule commit;
- `GPU_ARCHS` (alpha2: `gfx950`);
- `jax_aiter/jit/build_jit.py`;
- `jax_aiter/jit/optCompilerConfig.json`;
- an explicitly named `scripts/aiter_jit_*.patch`.

Unrelated MaxText/XLA patch files do not invalidate JIT binaries.

Monitor:

```bash
gh run list --repo ROCm/jax-aiter --workflow=jit-libs.yml --limit 5
gh run watch RUN_ID --repo ROCm/jax-aiter
```

The workflow publishes `manifest.json` and three checksummed blobs to the
rolling `jit-libs` prerelease. Consumer workflows must fetch them; they are not
allowed to rebuild on a GPU runner.

## 4. Run tests

Required:

```bash
JA_TEST_TIER=pr bash ci/test.sh
JA_TEST_TIER=nightly bash ci/test.sh
python3 ci/test_jit_libs_manifest.py
python3 ci/perf/test_parse_perf_log.py
bash tests/test_publication_perf_runner.sh
```

The nightly workflow must run on the 4-GPU runner so `multigpu` tests execute
instead of skipping.

## 5. Build and validate wheels

Manually dispatch `.github/workflows/release-publish.yml` with
`publish=false`.

The workflow:

1. fetches the rolling JIT assets;
2. builds both wheels in
   `ghcr.io/rocm/jax-manylinux_2_28-therock-7.14:7.14`;
3. runs `auditwheel show`;
4. checks that only gfx950 HSA files are packaged;
5. checks default has MHA shims but not MHA JIT libraries;
6. checks `+full` has both;
7. clean-room validates default FP4 plus its MHA guard;
8. clean-room validates full FP4 and MHA forward/backward on MI355X.

Do not publish if either job is skipped or fails.

## 6. Create the GitHub prerelease

After validation:

```bash
gh release create v0.1.0-alpha2 \
  --repo ROCm/jax-aiter \
  --title "jax-aiter v0.1.0-alpha2" \
  --prerelease \
  --notes-file docs/RELEASE_NOTES_v0.1.0-alpha2.md
```

Then dispatch the release workflow with `publish=true` and tag
`v0.1.0-alpha2`. It re-runs all wheel gates before uploading.

## 7. PyPI is a separate approval

The workflow does not upload to PyPI. Do not create or upload the
`jax-aiter` project without explicit user approval and an AMD-owned publishing
identity/token decision.

When approved, upload only the default `0.1.0a2` manylinux wheel. The
`0.1.0a2+full` wheel stays on GitHub.

## 8. Final checks

```bash
gh release view v0.1.0-alpha2 --repo ROCm/jax-aiter
git ls-remote --tags origin v0.1.0-alpha2
```

From a clean TheRock 7.14 runtime container:

```bash
python3 -m pip install ./jax_aiter-0.1.0a2-*.whl
python3 -c "from jax_aiter.gemm_fp4 import gemm_fp4_bf16; print('FP4 ready')"
jax-aiter-fetch-mha
python3 -c "from jax_aiter.mha import flash_attn_func; print('MHA ready')"
```

Update the memory bank only with paths and final status; do not paste logs or
performance figures.
