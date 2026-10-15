# syntax=docker/dockerfile:1.7
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Pinned source-build image for the JAX-AITER + MaxText MXFP4 stack.
#
# Build from the JAX-AITER repository root:
#
#   DOCKER_BUILDKIT=1 docker build --progress=plain \
#     -f docker/mxfp4/Dockerfile \
#     --build-context \
#       mxfp4-kernels=./jax_aiter/flydsl/kernels/mxfp4 \
#     --build-arg BUILD_JOBS=32 \
#     -t jax-aiter-mxfp4:local .
#
# The MXFP4 kernel source is license-gated and deliberately absent from Git.
# It must be supplied as an explicit BuildKit named context. The normal build
# context excludes it, and the build verifies every authorized source hash.
#
# JAX-AITER's MHA JIT libraries are compiled from the pinned AITER source.
# This is an offline gfx950 compilation and does not need /dev/kfd, but it can
# take hours and needs substantial host memory and disk.

ARG BASE_IMAGE=ghcr.io/rocm/jax-dev-ubu24.therock-10.0:10.0@sha256:8f9cbffd91f7923caf2ee720ba009a61cc1dd50870e77e315673ab9c667cc3f3
ARG JAX_AITER_REF=95a4323253d8151a6a279f1137808e95a4ffc3bd
ARG AITER_REPO=https://github.com/ROCm/aiter.git
ARG AITER_REF=e4ceb9dd95e12595dc9d87a06eaef79dfd6cdb8c
ARG MAXTEXT_REPO=https://github.com/ROCm/maxtext.git
ARG MAXTEXT_REF=eef2e2a4e93a168d5d2d207279b4861bf8c4a1f9

FROM ${BASE_IMAGE} AS system

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV HIPCC=/usr/local/bin/hipcc
ENV ROCM_VERSION=10.0.0

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      git \
      libnuma1 \
      lsof \
      ninja-build \
      numactl \
      pkg-config \
      procps \
      python3-dev \
      unzip \
      zstd \
 && rm -rf /var/lib/apt/lists/*

# This exact lock is the Python delta over the pinned TheRock 10.0 development
# base. --no-deps makes the transitive lock authoritative and prevents a later
# resolver run from silently changing it.
COPY docker/mxfp4/requirements.lock /tmp/mxfp4-requirements.lock
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --break-system-packages --ignore-installed --no-deps \
      -r /tmp/mxfp4-requirements.lock

# Install the validated ROCm 10 plugin/PJRT pair from the official JAX 0.11.2
# release wheelhouse. The attached archive is immutable and checksum-pinned;
# select only the Python 3.12 plugin wheel plus the architecture-neutral PJRT.
ARG JAX_ROCM_WHEELHOUSE_URL="https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.11.2/wheelhouse_theRock10.0.0.zip"
ARG JAX_ROCM_WHEELHOUSE_SHA256="bf209083dbaea661f1c37385f9199aed00293d720a00d644518f7472a7a69526"
RUN --mount=type=cache,target=/root/.cache/pip \
    curl --fail --location --retry 5 \
      --output /tmp/jax-rocm10-wheelhouse.zip \
      "${JAX_ROCM_WHEELHOUSE_URL}" \
 && echo "${JAX_ROCM_WHEELHOUSE_SHA256}  /tmp/jax-rocm10-wheelhouse.zip" \
      | sha256sum -c - \
 && mkdir -p /tmp/jax-rocm10-wheelhouse \
 && unzip -q -j /tmp/jax-rocm10-wheelhouse.zip \
      '*/jax_rocm10_plugin-0.11.2-cp312-cp312-*.whl' \
      '*/jax_rocm10_pjrt-0.11.2-*.whl' \
      -d /tmp/jax-rocm10-wheelhouse \
 && plugin=(/tmp/jax-rocm10-wheelhouse/jax_rocm10_plugin-0.11.2-cp312-cp312-*.whl) \
 && pjrt=(/tmp/jax-rocm10-wheelhouse/jax_rocm10_pjrt-0.11.2-*.whl) \
 && test "${#plugin[@]}" -eq 1 \
 && test "${#pjrt[@]}" -eq 1 \
 && python3 -m pip install --break-system-packages --no-deps \
      "${plugin[0]}" "${pjrt[0]}" \
 && rm -rf /tmp/jax-rocm10-wheelhouse /tmp/jax-rocm10-wheelhouse.zip

# FlyDSL 0.3.0 imports Torch even for framework-neutral compilation. Isolate
# its CPU-only Torch dependency so it cannot replace the ROCm/JAX packages.
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir -p /opt/flydsl030 \
 && python3 -m pip install --no-deps --target /opt/flydsl030 flydsl==0.3.0 \
 && python3 -m pip install --no-deps --target /opt/flydsl030 \
      filelock==3.32.3 \
      fsspec==2026.4.0 \
      Jinja2==3.1.6 \
      MarkupSafe==3.0.3 \
      mpmath==1.3.0 \
      networkx==3.6.1 \
      setuptools==78.1.0 \
      sympy==1.14.0 \
      typing_extensions==4.16.0 \
 && python3 -m pip install --no-deps --target /opt/flydsl030 \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.14.0+cpu

# The pinned AITER revision requires Triton 3.8.0. Use the ROCm 10.0 build and
# keep it isolated from MaxText and JAX's Python environment.
ARG TRITON_WHEEL_URL="https://stable.repo.amd.com/rocm/whl-next/triton/triton-3.8.0%2Bgit4cff872c.rocm10.0.0-cp312-cp312-linux_x86_64.whl#sha256=07f92a10d8c8286c205892c66cded7ea6df226ebff47e44af50805d643611c8a"
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir -p /opt/triton-rocm \
 && python3 -m pip install --no-deps --target /opt/triton-rocm \
      "${TRITON_WHEEL_URL}" \
 && PYTHONPATH=/opt/triton-rocm python3 -c \
      'import triton; assert triton.__version__.startswith("3.8.0"), triton.__version__'

RUN python3 -m pip check \
 && python3 - <<'PY'
from importlib import metadata

expected = {
    "drjax": "0.2.1",
    "flax": "0.12.10",
    "jax": "0.11.2",
    "jaxlib": "0.11.2",
    "jax-rocm10-plugin": "0.11.2",
    "jax-rocm10-pjrt": "0.11.2",
}
for distribution, version in expected.items():
    actual = metadata.version(distribution)
    if actual != version:
        raise SystemExit(f"{distribution}: expected {version}, got {actual}")

for distribution in (
    "transformer-engine",
    "transformer-engine-rocm7",
    "transformer-engine-rocm10",
    "transformer-engine-rocm-jax",
):
    try:
        actual = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        continue
    raise SystemExit(f"{distribution}: expected absent, got {actual}")
PY

FROM system AS sources

ARG AITER_REPO
ARG AITER_REF
ARG MAXTEXT_REPO
ARG MAXTEXT_REF

# Fetch exact immutable revisions. No dependency source is patched or edited.
RUN git init /opt/aiter \
 && git -C /opt/aiter remote add origin "${AITER_REPO}" \
 && git -C /opt/aiter fetch --depth=1 --filter=blob:none origin "${AITER_REF}" \
 && git -C /opt/aiter checkout --detach FETCH_HEAD \
 && test "$(git -C /opt/aiter rev-parse HEAD)" = "${AITER_REF}" \
 && git -C /opt/aiter submodule update --init --recursive --depth=1

RUN git init /opt/maxtext \
 && git -C /opt/maxtext remote add origin "${MAXTEXT_REPO}" \
 && git -C /opt/maxtext fetch --depth=1 --filter=blob:none origin "${MAXTEXT_REF}" \
 && git -C /opt/maxtext checkout --detach FETCH_HEAD \
 && test "$(git -C /opt/maxtext rev-parse HEAD)" = "${MAXTEXT_REF}"

# .dockerignore excludes local build products, Git metadata, third_party, and
# the rights-gated MXFP4 kernel tree.
COPY . /opt/jax-aiter

COPY --from=mxfp4-kernels \
  /ORIGIN.md \
  /__init__.py \
  /_buffer_ops.py \
  /_fwd_kernel_fp4.py \
  /_gemm_utils.py \
  /_rt.py \
  /_wgrad_kernel_fp4.py \
  /quant_dim0.py \
  /quant_dim1.py \
  /opt/jax-aiter/jax_aiter/flydsl/kernels/mxfp4/

COPY --from=mxfp4-kernels \
  /wholeloop/ORIGIN.md \
  /wholeloop/__init__.py \
  /wholeloop/buffer_ops.py \
  /wholeloop/gemm_helper.py \
  /wholeloop/gemm_mxfp4_kernel.py \
  /wholeloop/grouped_gemm_mxfp4_kernel.py \
  /opt/jax-aiter/jax_aiter/flydsl/kernels/mxfp4/wholeloop/

RUN mkdir -p /opt/jax-aiter/third_party \
 && ln -s /opt/aiter /opt/jax-aiter/third_party/aiter \
 && cd /opt/jax-aiter/jax_aiter/flydsl/kernels/mxfp4 \
 && sha256sum -c /opt/jax-aiter/docker/mxfp4/mxfp4-kernels.sha256 \
 && python3 - "${AITER_REF}" <<'PY'
import ast
from pathlib import Path
import sys

expected = sys.argv[1]
tree = ast.parse(Path("/opt/jax-aiter/jax_aiter/jit_assets.py").read_text())
values = {
    node.targets[0].id: ast.literal_eval(node.value)
    for node in tree.body
    if isinstance(node, ast.Assign)
    and len(node.targets) == 1
    and isinstance(node.targets[0], ast.Name)
    and node.targets[0].id in {"AITER_SHA", "GPU_ARCHS"}
}
if values.get("AITER_SHA") != expected:
    raise SystemExit(
        f"local JAX-AITER expects AITER {values.get('AITER_SHA')}, "
        f"but Docker build fetched {expected}"
    )
if values.get("GPU_ARCHS") != "gfx950":
    raise SystemExit(f"expected gfx950 JIT assets, got {values.get('GPU_ARCHS')}")
PY

FROM sources AS builder

ARG BUILD_JOBS=32

ENV JA_ROOT_DIR=/opt/jax-aiter
ENV GPU_ARCHS=gfx950
ENV AITER_GPU_ARCHS=gfx950
ENV AITER_SYMBOL_VISIBLE=1
ENV AITER_ASM_DIR=/opt/aiter/hsa/
ENV MAX_JOBS=${BUILD_JOBS}
ENV LD_LIBRARY_PATH=/opt/jax-aiter/build/jax_aiter_build:/opt/jax-aiter/build/aiter_build

WORKDIR /opt/jax-aiter

RUN case "${BUILD_JOBS}" in \
      ""|*[!0-9]*|0) echo "BUILD_JOBS must be a positive integer" >&2; exit 2 ;; \
    esac \
 && make -j"${BUILD_JOBS}"

# The expensive source build: librmsnorm_fwd.so, libmha_fwd.so, and
# libmha_bwd.so. build_jit.py streams periodic progress in non-interactive
# Docker builds. Separate layers preserve completed work if a later module
# fails.
RUN unset CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT \
 && python3 jax_aiter/jit/build_jit.py --module librmsnorm_fwd --verbose

RUN unset CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT \
 && python3 jax_aiter/jit/build_jit.py --module libmha_fwd --verbose

RUN unset CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT \
 && python3 jax_aiter/jit/build_jit.py --module libmha_bwd --verbose

RUN make -j"${BUILD_JOBS}" ja_mods \
 && make -f Makefile.flydsl -j"${BUILD_JOBS}" bridge \
 && make -f Makefile.triton -j"${BUILD_JOBS}" bridge

RUN make -f Makefile.kv -j"${BUILD_JOBS}" ja_kv

RUN set -eu; \
    required=" \
      build/aiter_build/librmsnorm_fwd.so \
      build/aiter_build/libmha_fwd.so \
      build/aiter_build/libmha_bwd.so \
      build/jax_aiter_build/libjax_aiter.so \
      build/jax_aiter_build/rmsnorm_fwd_ja.so \
      build/jax_aiter_build/silu_and_mul_ja.so \
      build/jax_aiter_build/gemm_fwd_ja.so \
      build/jax_aiter_build/gemm_fp4_ja.so \
      build/jax_aiter_build/cast_mxfp4_ja.so \
      build/jax_aiter_build/mha_fwd_ja.so \
      build/jax_aiter_build/mha_bwd_ja.so \
      build/jax_aiter_build/flydsl_bridge_ja.so \
      build/jax_aiter_build/triton_bridge_ja.so \
      build/jax_aiter_build/kv_alias_probe_ja.so \
      build/jax_aiter_build/append_kv_ja.so \
      build/jax_aiter_build/paged_attention_ja.so \
      build/jax_aiter_build/paged_prefill_ja.so"; \
    for library in ${required}; do \
      test -s "${library}"; \
      LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/lib" \
        ldd "${library}" | awk '/not found/ { bad=1 } END { exit bad }'; \
    done

# Preserve source and final libraries, but not multi-gigabyte compiler object
# trees or repository metadata. Never use `make clean`: it deletes final JIT
# libraries too.
RUN find /opt/jax-aiter/build/aiter_build \
      -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + \
 && find /opt/jax-aiter/build/jax_aiter_build \
      -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + \
 && rm -rf /opt/jax-aiter/build/hipified_aiter \
 && find /opt/aiter /opt/maxtext \
      -name .git -exec rm -rf {} + \
 && find /opt/aiter /opt/maxtext /opt/jax-aiter \
      -type d -name __pycache__ -prune -exec rm -rf {} +

FROM system AS runtime

ARG JAX_AITER_REF
ARG AITER_REF
ARG MAXTEXT_REF

COPY --from=builder /opt/aiter /opt/aiter
COPY --from=builder /opt/maxtext /opt/maxtext
COPY --from=builder /opt/jax-aiter /opt/jax-aiter

ENV JA_ROOT_DIR=/opt/jax-aiter
ENV JAX_AITER_ROOT=/opt/jax-aiter
ENV JA_AITER_PYTHON=/opt/aiter
ENV AITER_ASM_DIR=/opt/aiter/hsa/
ENV JA_TRITON_PATH=/opt/triton-rocm
ENV MAXTEXT_ROOT=/opt/maxtext
ENV XDG_CACHE_HOME=/var/cache/mxfp4
ENV FLYDSL_RUNTIME_CACHE_DIR=/var/cache/mxfp4/flydsl030
ENV TF_XLA_HSACO_CACHE_DIR=/var/cache/mxfp4/xla-hsaco
ENV JAX_AITER_FLYDSL_REPORT=1
ENV JAX_REMOVE_CUSTOM_PARTITIONING_PTR_FROM_CACHE_KEY=1
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV DECOUPLE_GCLOUD=TRUE
ENV DEBUG_HIP_DYNAMIC_QUEUES=0
ENV ROCPROFILER_QUEUE_INTERPOSITION=0
ENV ROCPROFILER_REGISTER_ENABLED=0
ENV PYTHONPATH=/opt/triton-rocm:/opt/flydsl030:/opt/jax-aiter:/opt/aiter:/opt/maxtext/src
ENV LD_LIBRARY_PATH=/opt/jax-aiter/build/jax_aiter_build:/opt/jax-aiter/build/aiter_build:/usr/local/lib/python3.12/dist-packages/_rocm_sdk_core/lib

LABEL org.opencontainers.image.title="jax-aiter-mxfp4-source-build"
LABEL org.opencontainers.image.jax-aiter="${JAX_AITER_REF}"
LABEL org.opencontainers.image.aiter="${AITER_REF}"
LABEL org.opencontainers.image.maxtext="${MAXTEXT_REF}"

RUN mkdir -p \
      "${XDG_CACHE_HOME}" \
      "${FLYDSL_RUNTIME_CACHE_DIR}" \
      "${TF_XLA_HSACO_CACHE_DIR}" \
 && python3 - <<'PY'
from importlib import metadata
from pathlib import Path

import flydsl
import jax
import jax_aiter
import jaxlib
import maxtext
import triton

assert jax.__version__ == "0.11.2", jax.__version__
assert jax_aiter.__version__ == "0.1.0a2", jax_aiter.__version__
assert jaxlib.__version__ == "0.11.2", jaxlib.__version__
assert flydsl.__version__ == "0.3.0", flydsl.__version__
assert triton.__version__.startswith("3.8.0"), triton.__version__
assert metadata.version("jax-rocm10-plugin") == "0.11.2"
assert metadata.version("jax-rocm10-pjrt") == "0.11.2"
for distribution in (
    "transformer-engine",
    "transformer-engine-rocm7",
    "transformer-engine-rocm10",
    "transformer-engine-rocm-jax",
):
    try:
        actual = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        continue
    raise AssertionError(f"{distribution}: expected absent, got {actual}")

root = Path("/opt/jax-aiter/build")
required = (
    root / "aiter_build/libmha_fwd.so",
    root / "aiter_build/libmha_bwd.so",
    root / "jax_aiter_build/libjax_aiter.so",
    root / "jax_aiter_build/flydsl_bridge_ja.so",
    root / "jax_aiter_build/triton_bridge_ja.so",
    root / "jax_aiter_build/paged_attention_ja.so",
    root / "jax_aiter_build/paged_prefill_ja.so",
)
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"missing JAX-AITER libraries: {missing}")

print(
    "source-build image ready:",
    f"jax={jax.__version__}",
    f"flydsl={flydsl.__version__}",
    f"triton={triton.__version__}",
)
PY

WORKDIR /opt/maxtext
CMD ["/bin/bash"]
