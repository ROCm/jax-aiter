#!/usr/bin/env bash
# third_party/build_static_pytorch.sh
#
# 1. HIP-ify the cloned PyTorch tree (idempotent – does nothing if already done)
# 2. Configure a static-only, PIC build that targets ROCm
# 3. Install c10, torch_cpu, and torch_hip as *.a archives + headers
#
# Environment overrides:
#   ROCM_ARCH   gfx arch (default gfx942)
#   ROCM_PATH   /opt/rocm by default
#   JOBS        parallel build jobs (default: $(nproc))
#   PYTHON      Python interpreter (default: python3)

set -euxo pipefail

ROCM_ARCH=${ROCM_ARCH:-gfx942}
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
PYTHON=${PYTHON:-python3}
JOBS=${JOBS:-$(nproc)}

SRC_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")/third_party/pytorch")"
BUILD_DIR="$SRC_DIR/build_static"
INSTALL_DIR="$BUILD_DIR/install"

echo "ROCm arch       : $ROCM_ARCH"
echo "ROCm path       : $ROCM_PATH"
echo "PyTorch source  : $SRC_DIR"
echo "Build directory : $BUILD_DIR"
echo "Install prefix  : $INSTALL_DIR"
echo "Jobs            : $JOBS"
echo

if [ ! -f "$SRC_DIR/.hipify_done" ]; then
  echo "HIPifying PyTorch sources (in-place)"
  "$PYTHON" "$SRC_DIR/tools/amd_build/build_amd.py" \
    --project-directory="$SRC_DIR"
  touch "$SRC_DIR/.hipify_done"
  echo "HIPify complete"
  echo
fi

cmake -S "$SRC_DIR" -B "$BUILD_DIR" -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DUSE_PYTHON=ON -DBUILD_PYTHON=OFF \
  -DUSE_ROCM=ON -DROCM_ARCH="$ROCM_ARCH" \
  -DUSE_CUDA=OFF -DUSE_NCCL=OFF \
  -DUSE_DISTRIBUTED=OFF \
  -DBUILD_JNI=OFF \
  -DUSE_OPENMP=OFF \
  -DUSE_MIMALLOC=OFF \
  -DUSE_OBSERVERS=OFF \
  -DUSE_NNPACK=OFF \
  -DUSE_ITT=OFF \
  -DUSE_KINETO=OFF \
  -DUSE_XNNPACK=OFF \
  -DBUILD_TEST=OFF -DBUILD_C10_HIP_TEST=OFF \
  -DBUILD_CAFFE2_OPS=OFF \
  -DUSE_FLASH_ATTENTION=OFF -DUSE_AOTRITON=OFF \
  -DUSE_MEM_EFF_ATTENTION=OFF \
  -DUSE_LITE_PROTO=ON \
  -DONNX_ML=OFF \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_CXX_FLAGS="-Wno-stringop-overflow"

cmake --build "$BUILD_DIR" --target c10 torch_cpu torch_hip -j"$JOBS"
cmake --install "$BUILD_DIR" --prefix "$INSTALL_DIR"

echo
echo "Static libraries ready:"
ls -1 "$INSTALL_DIR/lib/"lib{c10,torch_*}.a
echo "Headers installed under:"
echo "  $INSTALL_DIR/include"
