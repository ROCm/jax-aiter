# Reproduce Llama 3.1 8B MXFP4 training on ROCm

This recipe compares **MXFP4**, **TE FP8 current scaling**, and **BF16** on one
8 × AMD Instinct MI355X (`gfx950`) node. Each command launches one explicit
leg; there is no precision loop. JAX-AITER has no PyTorch runtime dependency.

## Reproduction status and versions

The recipe controls are fixed, but release provenance still needs to be sealed:

- Official image: `rocm/jax-training:<TODO_VERIFY exact tag>` and
  `sha256:<TODO_VERIFY digest>`.
- ROCm, JAX, `jaxlib`, ROCm PJRT/plugin, TransformerEngine, and AITER versions:
  `TODO_VERIFY` from the selected image and built JAX-AITER tree.
- JAX-AITER release tag/commit containing this recipe: `TODO_VERIFY`.
  Development base was `perf/mxfp4-deterministic-sr @ 06d7adf`.
- ROCm/MaxText branch: `aiter-fp4-integration`; record the checked-out commit as
  `TODO_VERIFY` before publishing results.
- Dataset and tokenizer checksums: `TODO_VERIFY`.

Do not publish throughput from a run until these values and the resolved recipe
banner in each `train.log` have been recorded.

## Start the official ROCm JAX container

Use AMD's official `rocm/jax-training` image because this comparison needs the
JAX ROCm backend and a compatible JAX TransformerEngine build. Resolve and pin
the exact tag and digest above before running:

```bash
export ROCM_JAX_IMAGE='rocm/jax-training:<TODO_VERIFY exact tag>@sha256:<TODO_VERIFY digest>'
docker pull "$ROCM_JAX_IMAGE"
docker run --rm -it \
  --network=host \
  --device=/dev/kfd --device=/dev/dri \
  --ipc=host --shm-size=64G --group-add video \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v "$PWD":/workspace -w /workspace \
  "$ROCM_JAX_IMAGE" bash
```

The commands below assume the host workspace is mounted at `/workspace`.
The runners retain the project default `/ruvaidya/aiter_proj`; setting
`PROJECT_ROOT=/workspace` resolves all paths for this mount.

## Clone, build, and install

Clone both source trees on the host before starting the container, or inside
the mounted workspace:

```bash
cd /workspace
git clone --recursive https://github.com/ROCm/jax-aiter.git
git clone --branch aiter-fp4-integration https://github.com/ROCm/maxtext.git
git -C jax-aiter rev-parse HEAD
git -C maxtext rev-parse HEAD
```

Build the full JAX-AITER source distribution because MXFP4 uses direct
JAX-AITER MHA in this recipe:

```bash
export PROJECT_ROOT=/workspace
export JAX_AITER_ROOT="$PROJECT_ROOT/jax-aiter"
export MAXTEXT_ROOT="$PROJECT_ROOT/maxtext"
export JA_ROOT_DIR="$JAX_AITER_ROOT"
export AITER_SYMBOL_VISIBLE=1
export GPU_ARCHS=gfx950
export AITER_ASM_DIR="$JAX_AITER_ROOT/third_party/aiter/hsa/"

cd "$JAX_AITER_ROOT"
test -d third_party/pytorch/build_static/install/include \
  || bash scripts/build_static_pytorch.sh
make
python3 jax_aiter/jit/build_jit.py
make ja_mods
python3 -m pip install .

cd "$MAXTEXT_ROOT"
python3 -m pip install --no-deps -e .
python3 -m pip install 'mlperf-logging==4.1.58'
```

The static PyTorch tree supplies build-time headers only. Do not install,
import, or route execution through PyTorch at runtime. The selected official
image must already provide MaxText's compatible JAX/ROCm and TE dependencies;
`--no-deps` prevents an editable MaxText install from replacing that stack.

Record the environment before a run:

```bash
python3 - <<'PY'
import importlib.metadata as m
for name in ("jax", "jaxlib", "jax-rocm7-pjrt", "jax-rocm7-plugin",
             "transformer-engine", "jax-aiter", "maxtext", "mlperf-logging"):
    try:
        print(f"{name}=={m.version(name)}")
    except m.PackageNotFoundError:
        print(f"{name}=NOT_INSTALLED")
PY
```

## Dataset prerequisites

The convergence runner defaults to the MLPerf Megatron-format C4 data:

```text
/workspace/datasets/mlperf_c4/c4-train.en_6_text_document.bin
/workspace/datasets/mlperf_c4/c4-train.en_6_text_document.idx
/workspace/datasets/mlperf_c4/c4-validation-91205-samples.en_text_document.bin
/workspace/datasets/mlperf_c4/c4-validation-91205-samples.en_text_document.idx
```

Obtain and preprocess C4 under its applicable license and the MLPerf Training
6.0 Llama 3.1 8B data procedure. This repository does not redistribute the
dataset. The tokenizer defaults to MaxText's
`src/maxtext/assets/tokenizers/tokenizer_llama3.tiktoken`. Verify every data and
tokenizer checksum against the finalized publication manifest (`TODO_VERIFY`).

The runner also supports `DATASET_TYPE=grain` with
`DATA_ROOT/c4_en_parquet/{train,val}/*.parquet` and
`DATA_ROOT/llama31_tokenizer`, but that is a separately labeled data route.

## Resolve recipes without a GPU run

`RECIPE_DRY_RUN=1` prints the banner and every MaxText argument without
importing JAX or launching training:

```bash
cd "$JAX_AITER_ROOT"
RECIPE_DRY_RUN=1 PROJECT_ROOT=/workspace \
  bash scripts/recipes/run_nvfp4_match_8b.sh mxfp4 /workspace/results/dry-run 50
bash tests/test_publication_perf_runner.sh
```

Check that the banner says batch 4, global batch 32, sequence 8192, seed 0,
`.97`, FSDP8, Shardy, `scan_layers=False`, fp32 weight/mu state, iota false,
autotune 5, and the expected quantization and attention backend.

## Run the three 50-step performance legs

Each command is one foreground process. Run them sequentially, with no
profiler. Do not combine modes in a shell loop. The runner creates an atomic
`.launch_once` marker and refuses to reuse a leg directory; choose a fresh
`PERF_ROOT` for every new attempt.

```bash
cd "$JAX_AITER_ROOT"
export PROJECT_ROOT=/workspace
export XLA_PYTHON_CLIENT_MEM_FRACTION=.97
export PERF_ROOT=/workspace/results/llama3_8b_perf_50

bash scripts/recipes/run_nvfp4_match_8b.sh mxfp4 "$PERF_ROOT" 50
bash scripts/recipes/run_nvfp4_match_8b.sh te_fp8_currentscaling "$PERF_ROOT" 50
bash scripts/recipes/run_nvfp4_match_8b.sh bf16 "$PERF_ROOT" 50
```

The authoritative modes are:

- MXFP4: `quantization=aiter_fp4`, direct `attention=aiter_flash`.
- TE FP8 current scaling:
  `quantization=te_fp8_currentscaling`, `attention=cudnn_flash_te`.
- BF16: empty quantization, `attention=cudnn_flash_te`.

Parse the last ten completed steps. In a complete 50-step process these are
steps 40–49:

```bash
python3 ci/perf/parse_perf_log.py \
  --log "$PERF_ROOT/mxfp4/train.log" --tail-n 10 --label mxfp4 \
  --out-json "$PERF_ROOT/mxfp4/tail_40_49.json"
```

Repeat the parser command for `te_fp8_currentscaling` and `bf16`. Confirm
`tail_first_step=40` and `tail_last_step=49`; do not report a partial window.

## Run MXFP4 convergence

The canonical command uses Megatron C4, a 6,000-step horizon, eval every 384
steps for 32 eval steps, target loss 3.3, seed 0, and MLPerf logging:

```bash
cd "$JAX_AITER_ROOT"
export PROJECT_ROOT=/workspace
export DATA_ROOT=/workspace/datasets
export XLA_PYTHON_CLIENT_MEM_FRACTION=.97
export TTC_ROOT=/workspace/results/llama3_8b_mxfp4_ttc

bash scripts/recipes/run_mlperf_ttc_8b.sh \
  mxfp4_ttc mxfp4 "$TTC_ROOT" 6000
```

The convergence runner uses the same atomic one-launch guard. Keep a failed or
interrupted directory as evidence and choose a new `TTC_ROOT` for any retry.

The same runner accepts `te_fp8_currentscaling` and `bf16` as explicit
single-leg modes when matched convergence baselines are required.

## Outputs and MLPerf checks

Expected files:

- Performance: `$PERF_ROOT/<mode>/train.log` plus MaxText outputs below that
  leg directory.
- Convergence: `$TTC_ROOT/mxfp4_ttc/train.log`,
  `$TTC_ROOT/mxfp4_ttc/mlperf.log`, and MaxText outputs.
- Every `train.log` begins with `RESOLVED_RECIPE_BEGIN` /
  `RESOLVED_RECIPE_END` and records the full command and XLA flags.

Run the Training 6.0 compliance checker from a temporary directory because it
writes `compliance_checker.log` in its current working directory:

```bash
CHECK_DIR="$(mktemp -d)"
( cd "$CHECK_DIR" && \
  python3 -m mlperf_logging.compliance_checker \
    --usage training --ruleset 6.0.0 --werror \
    "$TTC_ROOT/mxfp4_ttc/mlperf.log" )
```

A single successful compliance check is not MLPerf RCP validation. RCP requires
the official multi-run protocol and `rcp_checker` over the complete result set;
do not present this one-run convergence recipe as an RCP score.

## Memory and failure policy

`.97` is the only canonical memory fraction. If any leg reports OOM,
`RESOURCE_EXHAUSTED`, or an out-of-memory error at `.97`, stop and preserve the
command, banner, and log. Do not lower the fraction, reduce the batch, retry in
the same output directory, or quote a fallback run as canonical.
