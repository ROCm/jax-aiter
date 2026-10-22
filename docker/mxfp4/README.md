# Local MXFP4 source-build image

From the JAX-AITER repository root:

```bash
DOCKER_BUILDKIT=1 docker build --progress=plain \
  -f docker/mxfp4/Dockerfile \
  --build-arg BUILD_JOBS=32 \
  -t jax-aiter-mxfp4:local .
```

The build starts from the checksum-pinned TheRock ROCm 10.0 development image,
installs JAX/JAXlib 0.11.2 with the official ROCm 10 plugin/PJRT wheelhouse,
fetches the exact pinned AITER and MaxText commits, and compiles all JAX-AITER
libraries for `gfx950`: the umbrella and core/MHA shims, MHA JIT libraries,
FlyDSL and ROCm 10 Triton bridges, and paged-KV shims. Transformer Engine is
deliberately absent; the MXFP4 training recipe uses `attention=jax_aiter_flash`.
A GPU is not needed during the image build. The MHA compile can take hours and
use substantial host memory and disk.

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

JAX-AITER is under `/opt/jax-aiter`, pinned AITER under `/opt/aiter`, and
MaxText under `/opt/maxtext`. The image contains the software environment, not
project datasets, tokenizers, model checkpoints, or benchmark launch configs.

## Multi-node MoE layers

The DeepSeek-V3 MXFP4 recipe also needs Primus-Turbo's intranode DeepEP
dispatch. Clusters with AMD Pollara (AINIC) RoCE adapters need the matching
`libionic` userspace first. Build both on top of the image above:

```bash
# Pollara RoCE userspace; the version must match the hosts' nicctl.
docker build --progress=plain \
  -f docker/mxfp4/Dockerfile.tensorwave \
  --build-arg BASE_IMAGE=jax-aiter-mxfp4:local \
  --build-arg AINIC_VERSION=1.117.5-a-77 \
  -t jax-aiter-mxfp4:local-ainic .

# Primus-Turbo DeepEP, compiled from the pinned Primus-Turbo commit.
docker build --progress=plain \
  -f docker/mxfp4/Dockerfile.deepep \
  --build-arg BASE_IMAGE=jax-aiter-mxfp4:local-ainic \
  -t jax-aiter-mxfp4:local-deepep .
```

The DeepEP module is under `/opt/primus-turbo-deepep`, ahead of the other
packages on `PYTHONPATH`.
