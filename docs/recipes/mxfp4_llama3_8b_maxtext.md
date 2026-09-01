# Llama 3.1 8B MXFP4 training with JAX-AITER

This recipe runs the JAX-AITER MXFP4 linear path and direct JAX-AITER flash
attention in ROCm/MaxText. It records commands and provenance but makes no
performance claim; public results belong in the accompanying ROCm blog.

The convergence command is a seed-0, single-run reproduction path. It is not
MLPerf RCP validation.

## Pinned software

| Component | Revision |
|---|---|
| Container | `ghcr.io/rocm/jax-base-ubu24.therock-7.14:7.14` |
| Container digest | `sha256:a13556927770aa13c07bbb8bd1bd052d91cd3cdc254953d836158261d9a214a2` |
| JAX / `jaxlib` | `0.11.0` from PyPI |
| ROCm plugin / PJRT | `0.11.0.post1` from PyPI |
| JAX-AITER | `v0.1.0-alpha2` (`35b7175c763153ddb5da50c47d33dec436d5f191`) |
| AITER consumed by JAX-AITER | `31350226161346314b3d8882c8085bd31dce6a34` |
| ROCm/MaxText | `feature/jax-aiter-mxfp4-v26.6 @ b437942a5f33704f8438deb948488ad08164285c` |
| Flax | `0.12.8` |
| Transformer Engine packages | `2.17.0+rocm7.14.0.50a84ad` |
| MLPerf logging | `4.1.58` |

Record the dataset and tokenizer checksums for any result you retain. This
repository does not redistribute C4.

The alpha2 release commit and the blog's measured JAX-AITER commit
`3fcc5521afc968ab7fae4a2c0e06a59b16a4fa51` have identical Git trees. The
release commit is the stable public identity.

## Start the container

From a host directory that will hold both repositories and run outputs:

```bash
export ROCM_JAX_IMAGE='ghcr.io/rocm/jax-base-ubu24.therock-7.14:7.14@sha256:a13556927770aa13c07bbb8bd1bd052d91cd3cdc254953d836158261d9a214a2'
docker pull "$ROCM_JAX_IMAGE"
docker run --rm -it \
  --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --ipc=host --shm-size=64G --group-add video \
  --security-opt seccomp=unconfined \
  -v "$PWD":/workspace -w /workspace \
  "$ROCM_JAX_IMAGE" bash
```

TheRock packages ROCm through `rocm-sdk`; `/opt/rocm` is intentionally absent.

## Install JAX and JAX-AITER

```bash
python3 -m pip install \
  "jax==0.11.0" "jaxlib==0.11.0" \
  "jax-rocm7-plugin==0.11.0.post1" "jax-rocm7-pjrt==0.11.0.post1"

# The +full wheel is self-contained. The plain wheel is far smaller but then
# needs `jax-aiter-fetch-mha` before flash attention works.
# The '+' is percent-encoded as %2B in the download URL, not in the filename.
BASE=https://github.com/ROCm/jax-aiter/releases/download/v0.1.0-alpha2
WHEEL='jax_aiter-0.1.0a2+full-cp312-cp312-manylinux_2_39_x86_64.whl'
curl -fL -o "/tmp/$WHEEL" \
  "$BASE/jax_aiter-0.1.0a2%2Bfull-cp312-cp312-manylinux_2_39_x86_64.whl"
python3 -m pip install --no-deps "/tmp/$WHEEL"

python3 - <<'PY'
import jax
from jax_aiter.gemm_fp4 import gemm_fp4_bf16
from jax_aiter.mha import flash_attn_func

print("devices:", jax.devices())
print("MXFP4 and MHA imports: OK")
PY
```

The MHA libraries are checksummed `gfx950` assets stored in a versioned user
cache. No JAX-AITER source build is needed.

## Install the pinned MaxText integration

Clone JAX-AITER for the recipe launchers and MaxText at the exact tested commit:

```bash
cd /workspace
git clone https://github.com/ROCm/jax-aiter.git
git clone https://github.com/ROCm/maxtext.git
git -C maxtext checkout b437942a5f33704f8438deb948488ad08164285c
```

Install the decoupled MaxText dependencies without downgrading JAX to the
branch's older requirement:

```bash
cd /workspace/maxtext
python3 - <<'PY'
from pathlib import Path

source = Path(
    "src/dependencies/requirements/"
    "requirements_decoupled_rocm_jax_0_10_0.txt"
)
dest = Path("/tmp/maxtext-rocm-requirements.txt")
lines = [
    line for line in source.read_text().splitlines()
    if not line.strip().startswith(
        ("jax==", "jaxlib==", "jax-rocm7-plugin", "jax-rocm7-pjrt")
    )
]
dest.write_text("\n".join(lines) + "\n")
print(dest)
PY
python3 -m pip install -r /tmp/maxtext-rocm-requirements.txt
python3 -m pip install "flax==0.12.8" "mlperf-logging==4.1.58"
python3 -m pip install --no-deps -e .

python3 -c "import jax, maxtext; print('jax', jax.__version__, 'maxtext import OK')"
```

## Install the pinned Transformer Engine packages

The FP8 and BF16 reference legs use Transformer Engine MHA. Install the exact
public artifacts used by the reported cohort and verify their hashes:

```bash
TE_CORE=/tmp/transformer_engine_rocm7.whl
TE_JAX=/tmp/transformer_engine_rocm_jax.tar.gz
curl -fL -o "$TE_CORE" \
  'https://rocm.frameworks-devreleases.amd.com/whl-multi-arch-staging/transformer-engine-rocm7/transformer_engine_rocm7-2.17.0%2Brocm7.14.0.50a84ad-cp312-cp312-manylinux_2_28_x86_64.whl'
curl -fL -o "$TE_JAX" \
  'https://rocm.frameworks-devreleases.amd.com/whl-multi-arch-staging/transformer-engine-rocm-jax/transformer_engine_rocm_jax-2.17.0%2Brocm7.14.0.50a84ad.tar.gz'
printf '%s  %s\n' \
  1104fa964c91280235a6e9330a2f9ee6ed78e733be4efa14c23150f8c7c53b07 "$TE_CORE" \
  322c1b6d3fbca7a4be26fc8f5b8473449c77a747a62893fcc5ce62fbd10f85b8 "$TE_JAX" \
  | sha256sum --check
python3 -m pip install --no-deps "$TE_CORE" "$TE_JAX"
export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/lib:/usr/local/lib/python3.12/dist-packages/_rocm_sdk_libraries/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python3 -c "import transformer_engine.jax; print('Transformer Engine JAX import: OK')"
```

## Record the environment

```bash
python3 - <<'PY'
import importlib.metadata as metadata

for name in (
    "jax", "jaxlib", "jax-rocm7-plugin", "jax-rocm7-pjrt",
    "jax-aiter", "maxtext", "mlperf-logging",
    "transformer-engine-rocm7", "transformer-engine-rocm-jax",
):
    try:
        print(f"{name}=={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        print(f"{name}=NOT_INSTALLED")
PY

rocm-sdk version
git -C /workspace/jax-aiter rev-parse HEAD
git -C /workspace/maxtext rev-parse HEAD
sha256sum /workspace/jax-aiter/scripts/recipes/run_nvfp4_match_8b.sh
```

## Resolve the recipe without launching training

The dry run prints every MaxText argument and XLA flag without importing JAX or
using a GPU:

```bash
export PROJECT_ROOT=/workspace
export JAX_AITER_ROOT=/workspace/jax-aiter
export MAXTEXT_ROOT=/workspace/maxtext

cd "$JAX_AITER_ROOT"
RECIPE_DRY_RUN=1 bash scripts/recipes/run_nvfp4_match_8b.sh \
  mxfp4 /workspace/results/dry-run 50
RECIPE_DRY_RUN=1 bash scripts/recipes/run_nvfp4_match_8b.sh \
  fp8 /workspace/results/dry-run 50
RECIPE_DRY_RUN=1 bash scripts/recipes/run_nvfp4_match_8b.sh \
  bf16 /workspace/results/dry-run 50
bash tests/test_publication_perf_runner.sh
```

All three resolved banners must show:

- `model_name=llama3.1-8b`
- FSDP8 and Shardy
- `scan_layers=False`
- per-device batch 9 and global batch 72
- sequence length 8192
- BF16 weight and optimizer state
- memory fraction `.97`

The MXFP4 banner must additionally show `quantization=aiter_fp4`,
`attention=aiter_flash`, `minimal_flash_save_fp4_wtcol`, iota embedding off,
the 64 MiB pipelined scheduler, the explicit MLPerf-aligned Llama 3.1 model
controls, and JAX-AITER enabled. Plain FP8 must show `quantization=fp8`,
`cudnn_flash_te`, `minimal_flash`, iota embedding on, the UTD scheduler, and
MaxText's pinned model defaults. BF16 differs from plain FP8 only where the
article says it does: no quantization and `remat_policy=minimal`.

## Run the three performance processes

The runner requires exactly 50 steps, refuses to reuse an output directory,
records source/package provenance, and validates the complete finite step
sequence plus the mean of completed steps 40–49. Run each command separately
on an otherwise idle node:

```bash
cd "$JAX_AITER_ROOT"
export XLA_PYTHON_CLIENT_MEM_FRACTION=.97
export JAX_AITER_RUNTIME=installed
export PERF_ROOT=/workspace/results/llama3_8b_blog_50

bash scripts/recipes/run_nvfp4_match_8b.sh \
  mxfp4 "$PERF_ROOT" 50
bash scripts/recipes/run_nvfp4_match_8b.sh \
  fp8 "$PERF_ROOT" 50
bash scripts/recipes/run_nvfp4_match_8b.sh \
  bf16 "$PERF_ROOT" 50
```

The FP8 and BF16 modes require the two pinned ROCm Transformer Engine packages.
The output for each leg contains `train.log`, `provenance.txt`, and
`parsed_tail.json`. Do not combine outputs from retries or a different source
or package stack.

The repository's CI leg explicitly overrides batch, state dtype, and remat for
its smaller FSDP4 regression point. It is not a publication result.

## Run convergence

The convergence route uses the MLPerf Megatron-format C4 files:

```text
/workspace/datasets/mlperf_c4/c4-train.en_6_text_document.bin
/workspace/datasets/mlperf_c4/c4-train.en_6_text_document.idx
/workspace/datasets/mlperf_c4/c4-validation-91205-samples.en_text_document.bin
/workspace/datasets/mlperf_c4/c4-validation-91205-samples.en_text_document.idx
```

Run one seed-0 process:

```bash
cd "$JAX_AITER_ROOT"
export DATA_ROOT=/workspace/datasets
export XLA_PYTHON_CLIENT_MEM_FRACTION=.97
export TTC_ROOT=/workspace/results/llama3_8b_mxfp4_ttc

bash scripts/recipes/run_mlperf_ttc_8b.sh \
  mxfp4_ttc mxfp4 "$TTC_ROOT" 6000
```

Check the resulting MLPerf log from a temporary directory:

```bash
CHECK_DIR="$(mktemp -d)"
( cd "$CHECK_DIR" && \
  python3 -m mlperf_logging.compliance_checker \
    --usage training --ruleset 6.0.0 --werror \
    "$TTC_ROOT/mxfp4_ttc/mlperf.log" )
```

A successful compliance check is not RCP validation. RCP requires the official
multi-run protocol and `rcp_checker`.

## Failure policy

`.97` is the only canonical memory fraction. On OOM, `RESOURCE_EXHAUSTED`, or
an out-of-memory message:

1. stop;
2. preserve the resolved banner, command, and log;
3. use a fresh output directory for any approved retry;
4. do not lower the memory fraction and call the result canonical.
