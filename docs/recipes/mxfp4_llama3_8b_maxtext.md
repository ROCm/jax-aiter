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
| ROCm plugin / PJRT | `0.11.0` from PyPI |
| JAX-AITER | `v0.1.0-alpha2` |
| AITER consumed by JAX-AITER | `31350226161346314b3d8882c8085bd31dce6a34` |
| ROCm/MaxText | `aiter-fp4-integration @ ccd72e63e57193c6f1d51b06bd2e7f52ce895404` |
| MaxText patch base | `14aa40b3af3aae793c72bb886533b4e8790ee6fa` |

Record the dataset and tokenizer checksums for any result you retain. This
repository does not redistribute C4.

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
  "jax-rocm7-plugin==0.11.0" "jax-rocm7-pjrt==0.11.0"

WHEEL=jax_aiter-0.1.0a2-cp312-cp312-manylinux_2_39_x86_64.whl
curl -fL -o "/tmp/$WHEEL" \
  "https://github.com/ROCm/jax-aiter/releases/download/v0.1.0-alpha2/$WHEEL"
python3 -m pip install "/tmp/$WHEEL"
jax-aiter-fetch-mha

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
git clone --branch v0.1.0-alpha2 \
  https://github.com/ROCm/jax-aiter.git
git clone --branch aiter-fp4-integration \
  https://github.com/ROCm/maxtext.git
git -C maxtext checkout ccd72e63e57193c6f1d51b06bd2e7f52ce895404
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
python3 -m pip install --no-deps -e .

python3 -c "import jax, maxtext; print('jax', jax.__version__, 'maxtext import OK')"
```

### Patch route

If a branch checkout is unsuitable, the JAX-AITER repository carries a patch
whose post-image matches `ccd72e63` (whitespace-only blank lines are
normalized):

```bash
git clone https://github.com/ROCm/maxtext.git /workspace/maxtext-patched
git -C /workspace/maxtext-patched checkout \
  14aa40b3af3aae793c72bb886533b4e8790ee6fa
git -C /workspace/maxtext-patched apply \
  /workspace/jax-aiter/scripts/maxtext_aiter_fp4.patch
```

CPU CI checks both that the patch applies and that its post-image matches the
pinned branch commit while ignoring end-of-line whitespace.

## Record the environment

```bash
python3 - <<'PY'
import importlib.metadata as metadata

for name in (
    "jax", "jaxlib", "jax-rocm7-plugin", "jax-rocm7-pjrt",
    "jax-aiter", "maxtext", "mlperf-logging",
):
    try:
        print(f"{name}=={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        print(f"{name}=NOT_INSTALLED")
PY

rocm-sdk version
git -C /workspace/jax-aiter rev-parse HEAD
git -C /workspace/maxtext rev-parse HEAD
```

## Resolve the recipe without launching training

The dry run prints every MaxText argument and XLA flag without importing JAX or
using a GPU:

```bash
export PROJECT_ROOT=/workspace
export JAX_AITER_ROOT=/workspace/jax-aiter
export MAXTEXT_ROOT=/workspace/maxtext

cd "$JAX_AITER_ROOT"
RECIPE_DRY_RUN=1 MODEL_CONTROLS=llama31_mlperf \
  bash scripts/recipes/run_nvfp4_match_8b.sh \
  mxfp4 /workspace/results/dry-run 50
bash tests/test_publication_perf_runner.sh
```

The resolved banner must show:

- `model_name=llama3.1-8b`
- `quantization=aiter_fp4`
- `attention=aiter_flash`
- FSDP8 and Shardy
- `scan_layers=False`
- per-device batch 4 and global batch 32
- sequence length 8192
- FP32 weight and optimizer state
- memory fraction `.97`

`quantization=aiter_fp4` automatically routes supported dense 8B configurations
to direct JAX-AITER attention. An explicit alternate attention mode with
`aiter_attention=False` is the rollback path.

## Run one MXFP4 process

The runner requires exactly 50 steps and refuses to reuse an output directory:

```bash
cd "$JAX_AITER_ROOT"
export XLA_PYTHON_CLIENT_MEM_FRACTION=.97
export PERF_ROOT=/workspace/results/llama3_8b_mxfp4_50

MODEL_CONTROLS=llama31_mlperf \
  bash scripts/recipes/run_nvfp4_match_8b.sh \
  mxfp4 "$PERF_ROOT" 50
```

The repository's nightly regression leg uses the same recipe at FSDP4 because
the 8-GPU CI pool is heavily contended. It is a CI operating point, not the
8-GPU publication recipe above.

Optional comparison modes remain available to reproduce a matched study:

```bash
bash scripts/recipes/run_nvfp4_match_8b.sh \
  te_fp8_currentscaling "$PERF_ROOT" 50
bash scripts/recipes/run_nvfp4_match_8b.sh \
  bf16 "$PERF_ROOT" 50
```

Those two modes use TransformerEngine attention and require a compatible ROCm
TransformerEngine wheel. The pinned MaxText checkout contains
`.github/workflows/utils/install_te_rocm_wheel.py` for that optional setup.
Do not mix their outputs with the MXFP4 directory.

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
