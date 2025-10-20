#!/usr/bin/env bash
# scripts/build_static_pytorch.sh
# Static-only, PIC PyTorch build for ROCm, minimized to c10/torch_cpu/torch_hip

set -euxo pipefail

ROCM_ARCH=${ROCM_ARCH:-gfx942}
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
PYTHON=${PYTHON:-python3}
JOBS=${JOBS:-$(nproc)}

SRC_DIR="$(realpath "./third_party/pytorch")"
BUILD_DIR="$SRC_DIR/build_static"
INSTALL_DIR="$BUILD_DIR/install"

echo "ROCm arch       : $ROCM_ARCH"
echo "ROCm path       : $ROCM_PATH"
echo "PyTorch source  : $SRC_DIR"
echo "Build directory : $BUILD_DIR"
echo "Install prefix  : $INSTALL_DIR"
echo "Jobs            : $JOBS"
echo

# 1) HIPify once (idempotent)
if [ ! -f "$SRC_DIR/.hipify_done" ]; then
  echo "HIPifying PyTorch sources (in-place)"
  "$PYTHON" "$SRC_DIR/tools/amd_build/build_amd.py" --project-directory="$SRC_DIR"
  touch "$SRC_DIR/.hipify_done"
  echo "HIPify complete"
  echo
fi

# 2) Configure a static, PIC, ROCm-only, minimal build
cmake -S "$SRC_DIR" -B "$BUILD_DIR" -GNinja \
  -DPYTORCH_ROCM_ARCH="gfx942;gfx950" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DBUILD_SHARED_LIBS=OFF \
  -DUSE_ROCM=ON -DROCM_ARCH="$ROCM_ARCH" \
  -DUSE_CUDA=OFF \
  -DUSE_XPU=OFF \
  -DUSE_CUDNN=OFF \
  -DUSE_CUSPARSELT=OFF \
  -DUSE_CUDSS=OFF \
  -DUSE_CUFILE=OFF \
  -DUSE_STATIC_CUDNN=OFF \
  \
  -DBUILD_PYTHON=OFF \
  -DUSE_PYTHON=OFF \
  -DUSE_NUMPY=OFF \
  \
  -DUSE_DISTRIBUTED=OFF \
  -DUSE_GLOO=OFF \
  -DUSE_MPI=OFF \
  -DUSE_TENSORPIPE=OFF \
  -DUSE_UCC=OFF \
  -DUSE_C10D_GLOO=OFF \
  -DUSE_C10D_NCCL=OFF \
  -DUSE_C10D_MPI=OFF \
  -DUSE_C10D_UCC=OFF \
  -DUSE_NCCL=OFF \
  -DUSE_RCCL=OFF \
  -DUSE_XCCL=OFF \
  -DUSE_STATIC_NCCL=OFF \
  -DUSE_SYSTEM_NCCL=OFF \
  -DUSE_NVSHMEM=OFF \
  \
  -DUSE_OPENMP=OFF \
  -DUSE_BLAS=OFF \
  -DUSE_MKLDNN=OFF \
  -DUSE_MKLDNN_ACL=OFF \
  -DUSE_MKLDNN_CBLAS=OFF \
  -DUSE_STATIC_MKL=OFF \
  -DUSE_MAGMA=OFF \
  -DUSE_NUMA=OFF \
  -DUSE_EIGEN_SPARSE=OFF \
  -DUSE_SYSTEM_EIGEN_INSTALL=OFF \
  -DUSE_VALGRIND=OFF \
  \
  -DUSE_XNNPACK=OFF \
  -DUSE_PYTORCH_QNNPACK=OFF \
  -DUSE_FBGEMM=OFF \
  -DUSE_NNPACK=OFF \
  \
  -DUSE_KINETO=OFF \
  -DUSE_PROF=OFF \
  -DUSE_ITT=OFF \
  -DUSE_OBSERVERS=OFF \
  -DUSE_MIMALLOC=OFF \
  -DUSE_CUPTI_SO=OFF \
  \
  -DUSE_VULKAN=OFF \
  -DUSE_OPENCL=OFF \
  -DUSE_PYTORCH_METAL=OFF \
  -DUSE_PYTORCH_METAL_EXPORT=OFF \
  -DUSE_MPS=OFF \
  -DUSE_COREML_DELEGATE=OFF \
  \
  -DONNX_ML=OFF \
  -DUSE_SYSTEM_ONNX=OFF \
  \
  -DBUILD_TEST=OFF \
  -DBUILD_C10_HIP_TEST=OFF \
  -DBUILD_BINARY=OFF \
  -DBUILD_JNI=OFF \
  -DBUILD_FUNCTORCH=OFF \
  -DBUILD_LAZY_TS_BACKEND=OFF \
  -DBUILD_LITE_INTERPRETER=OFF \
  -DUSE_LITE_PROTO=ON \
  \
  -DUSE_GLOG=OFF \
  -DUSE_GFLAGS=OFF \
  \
  -DUSE_ROCM_CK_GEMM=OFF \
  -DUSE_CK_FLASH_ATTENTION=OFF \
  -DCMAKE_DISABLE_FIND_PACKAGE_composable_kernel=ON \
  -DUSE_ROCM_CK_SDPA=OFF \
  \
  -DUSE_FLASH_ATTENTION=OFF \
  -DUSE_MEM_EFF_ATTENTION=OFF \
  -DUSE_AOTRITON=OFF \
  \
  -DFMT_INSTALL=OFF \
  -DFMT_TEST=OFF \
  -DFMT_DOC=OFF \
  -DFMT_HEADER_ONLY=ON \
  \
  -DCMAKE_C_FLAGS="-ffunction-sections -fdata-sections" \
  -DCMAKE_CXX_FLAGS="-Wno-stringop-overflow -ffunction-sections -fdata-sections -DUSE_DIRECT_NVRTC"

# 3) Build and install just what we need for linking
cmake --build "$BUILD_DIR" --target c10 torch_cpu torch_hip caffe2_nvrtc torch -j"$JOBS"
cmake --install "$BUILD_DIR" --prefix "$INSTALL_DIR"

echo
echo "Static libraries installed:"
ls -1 "$INSTALL_DIR/lib/"lib{c10,torch_*}.a || true
