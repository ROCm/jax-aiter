# Local MXFP4 source-build image

From the JAX-AITER repository root:

```bash
DOCKER_BUILDKIT=1 docker build --progress=plain \
  -f docker/mxfp4/Dockerfile \
  --build-arg BUILD_JOBS=32 \
  --build-arg JAX_AITER_REF="$(git rev-parse HEAD)" \
  --build-arg AITER_JIT_LIBS_TAG=jit-libs-26a4ef9da0d91247 \
  -t jax-aiter-mxfp4:local .
```

The build starts from the checksum-pinned TheRock ROCm 10.0 development image,
installs JAX/JAXlib 0.11.2 with the official ROCm 10 plugin/PJRT wheelhouse,
fetches the exact pinned AITER, flash-linear-attention, and MaxText commits, and compiles all JAX-AITER
libraries for `gfx950`: the umbrella and core/MHA shims, FlyDSL and ROCm 10
Triton bridges, and paged-KV shims. Transformer Engine is not part of this
image; `Dockerfile.te` adds it. A GPU is not needed during the image build.

`AITER_JIT_LIBS_TAG` installs AITER's MHA JIT libraries from that ROCm/jax-aiter
release instead of compiling them. `fetch_aiter_jit_libs.py` stops if the
release was built from another AITER commit or GPU architecture and checks
every file's SHA-256. `jit-libs-26a4ef9da0d91247` holds the libraries of the
MXFP4 MoE blog runs. Without the argument, the build compiles the MHA JIT
libraries, which can take hours and use substantial host memory and disk.

The MXFP4 MoE casts and grouped GEMMs are the Primus-Turbo FlyDSL kernels
vendored in JAX-AITER; `jax_aiter/THIRD_PARTY_NOTICES.md` lists their sources
and licenses. A normal clone builds the image. The build fails if a
license-gated `jax_aiter/flydsl/kernels/mxfp4/` tree reaches it.

Run on an MI355X host:

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --ipc=host --network=host --shm-size=64G \
  --security-opt seccomp=unconfined \
  jax-aiter-mxfp4:local
```

JAX-AITER is under `/opt/jax-aiter`, pinned AITER under `/opt/aiter`,
flash-linear-attention under `/opt/flash-linear-attention`, and MaxText under
`/opt/maxtext`. The image contains the software environment, not
datasets, tokenizers, or model checkpoints.

## MaxText MoE layers

MaxText's Transformer Engine attention (`attention=cudnn_flash_te`) and its
BF16 grouped-GEMM experts need Transformer Engine; the MoE recipes also need
Primus-Turbo's intranode DeepEP dispatch. Clusters with AMD AI NIC (AINIC)
RoCE adapters need the matching `libionic` userspace. Build the layers on top
of the image above:

```bash
# ROCm Transformer Engine for JAX, compiled from the pinned commit.
docker build --progress=plain \
  -f docker/mxfp4/Dockerfile.te \
  --build-arg BASE_IMAGE=jax-aiter-mxfp4:local \
  --build-arg BUILD_JOBS=32 \
  -t jax-aiter-mxfp4:local-te .

# Only for AINIC RoCE adapters; the version must match the hosts' nicctl.
docker build --progress=plain \
  -f docker/mxfp4/Dockerfile.ainic \
  --build-arg BASE_IMAGE=jax-aiter-mxfp4:local-te \
  --build-arg AINIC_VERSION=1.117.5-a-77 \
  -t jax-aiter-mxfp4:local-te-ainic .

# Primus-Turbo DeepEP, compiled from the pinned Primus-Turbo commit.
docker build --progress=plain \
  -f docker/mxfp4/Dockerfile.deepep \
  --build-arg BASE_IMAGE=jax-aiter-mxfp4:local-te-ainic \
  -t jax-aiter-mxfp4:local-moe .
```

Without AINIC adapters, skip the second build and use
`BASE_IMAGE=jax-aiter-mxfp4:local-te` for the third. The images contain no
RDMA user-space libraries, so multi-node runs over other RoCE or InfiniBand
adapters need their libraries added at the same point. Transformer Engine is
under `/opt/transformer-engine` and the DeepEP module under
`/opt/primus-turbo-deepep`, both on `PYTHONPATH`.

## MaxText MoE training

`run_maxtext_moe.sh` trains DeepSeek-V3 671B, Qwen3.5-397B-A17B, or
GPT-OSS-120B with the BF16 baseline (Transformer Engine grouped GEMM) or
JAX-AITER's MXFP4 routed experts, one process per GPU, with expert
parallelism within a node and FSDP across nodes. Run it once on every node, as
root, inside the image with all layers:

```bash
MODEL=deepseek PRECISION=mxfp4 NNODES=8 NODE_RANK=<0..7> \
COORDINATOR=<node 0 address> OUTPUT_DIR=/work/out/deepseek_mxfp4_b4 \
bash /opt/jax-aiter/docker/mxfp4/run_maxtext_moe.sh
```

The script's header lists every setting. For multi-node runs, export the
fabric settings first: `NCCL_NET=IB`, `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`, and
`NCCL_SOCKET_IFNAME`.
