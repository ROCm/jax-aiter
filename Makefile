# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# JAX-AITER build. No PyTorch dependency.
# Targets: all (umbrella lib), ja_mods (FFI modules), clean.

ROCM_ARCH    ?= gfx950
PYTHON3      ?= python3

# ROCm location. Legacy images install under /opt/rocm; TheRock images ship ROCm
# as the `rocm-sdk` pip package and have no /opt/rocm at all. Deliberately not
# named ROCM_PATH: some images export that pointing at a versioned directory,
# which would silently move the legacy rpath off /opt/rocm.
ifneq ($(wildcard /opt/rocm/bin/hipcc),)
JA_ROCM_PATH := /opt/rocm
HIP_RPATH    := -Wl,-rpath,$(JA_ROCM_PATH)/lib
else
JA_ROCM_PATH := $(shell rocm-sdk path --root 2>/dev/null)
# TheRock registers its libraries with ldconfig and resolves the rest through
# $ORIGIN-relative RUNPATHs inside the _rocm_sdk_* packages. An rpath into
# _rocm_sdk_devel/lib would shadow the per-arch device libraries, so omit it.
HIP_RPATH    :=
endif

HIPCC        ?= $(if $(wildcard $(JA_ROCM_PATH)/bin/hipcc),$(JA_ROCM_PATH)/bin/hipcc,$(shell command -v hipcc 2>/dev/null))
HIP_LIB      := $(JA_ROCM_PATH)/lib

AITER_SRC_DIR:= third_party/aiter
AITER_HIP_DIR:= build/hipified_aiter
AITER_INC    := $(AITER_HIP_DIR)/csrc/include

JAX_FFI_INC  := $(shell $(PYTHON3) -c 'from jax import ffi; print(ffi.include_dir())')
PYTHON_INC   := $(shell $(PYTHON3) -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
JAX_AITER_INC:= csrc/common

OUT_SO := build/jax_aiter_build/libjax_aiter.so

UMBRELLA_CXXFLAGS := -std=c++20 -fPIC -O3 -DUSE_ROCM -D__HIP_PLATFORM_AMD__ \
                     -I$(JAX_FFI_INC) -I$(PYTHON_INC) -I$(JAX_AITER_INC) -I$(AITER_INC) \
                     -fvisibility-inlines-hidden -fvisibility=hidden

UMBRELLA_LDFLAGS := -lamdhip64 -lhiprtc $(HIP_RPATH) -Wl,-soname,libjax_aiter.so

JA_BUILD_DIR := build/jax_aiter_build

GPU_ARCHS ?= $(if $(GFX),$(GFX),$(ROCM_ARCH))
GPU_ARCHS_LIST := $(subst ;, ,$(GPU_ARCHS))
AMDGPU_TARGET_FLAGS := $(foreach arch,$(GPU_ARCHS_LIST),--offload-arch=$(arch))

JA_CXXFLAGS := -std=c++20 -fPIC -O3 -DUSE_ROCM -D__HIP_PLATFORM_AMD__ -DENABLE_CK=1 \
               -fvisibility-inlines-hidden -fvisibility=hidden

JA_INCLUDES := -I$(AITER_SRC_DIR)/3rdparty/composable_kernel/include \
               -I$(AITER_SRC_DIR)/3rdparty/composable_kernel/library/include \
               -I$(AITER_SRC_DIR)/3rdparty/composable_kernel/example/ck_tile/01_fmha \
               -I$(JAX_FFI_INC) -I$(PYTHON_INC) -I$(JAX_AITER_INC) \
               -I$(AITER_INC) -I$(AITER_SRC_DIR)/csrc/include

RMSNORM_INCLUDES := $(JA_INCLUDES) \
                    -I$(AITER_SRC_DIR)/3rdparty/composable_kernel/example/ck_tile/10_rmsnorm2d

GEMM_CONFIG_DIR  := build/generated
GEMM_BF16_CFG    := $(GEMM_CONFIG_DIR)/asm_bf16gemm_configs.hpp
GEMM_FP4_CFG     := $(GEMM_CONFIG_DIR)/asm_f4gemm_configs.hpp
GEMM_INCLUDES    := $(JA_INCLUDES) -I$(GEMM_CONFIG_DIR)

# Core (non-MHA) FFI shims: shipped in both the lite and full wheels.
JA_CORE_MODULES := $(JA_BUILD_DIR)/rmsnorm_fwd_ja.so \
                   $(JA_BUILD_DIR)/silu_and_mul_ja.so \
                   $(JA_BUILD_DIR)/gemm_fwd_ja.so \
                   $(JA_BUILD_DIR)/gemm_fp4_ja.so \
                   $(JA_BUILD_DIR)/cast_mxfp4_ja.so

# MHA FFI shims: full wheel only (the heavy libmha_*.so JIT libs back these).
JA_MHA_MODULES := $(JA_BUILD_DIR)/mha_fwd_ja.so \
                  $(JA_BUILD_DIR)/mha_bwd_ja.so

# Full set (unchanged target for `make ja_mods`): core + MHA.
JA_MODULES := $(JA_CORE_MODULES) $(JA_MHA_MODULES)

.PHONY: all clean clean-stage ja_mods ja_mods_nomha

all: $(OUT_SO)

ja_mods: $(JA_MODULES)

# Lite wheel: build only the core (non-MHA) FFI shims.
ja_mods_nomha: $(JA_CORE_MODULES)

%/: 
	mkdir -p $@

$(OUT_SO): build/jax_aiter_build/ csrc/common/mha_common_utils.cu
	$(HIPCC) -shared $(UMBRELLA_CXXFLAGS) \
		-I$(AITER_SRC_DIR)/3rdparty/composable_kernel/include \
		-I$(AITER_SRC_DIR)/3rdparty/composable_kernel/example/ck_tile/01_fmha \
		csrc/common/mha_common_utils.cu \
		$(UMBRELLA_LDFLAGS) -o $@

$(JA_BUILD_DIR)/mha_fwd_ja.so: csrc/ffi/mha_fwd/mha_fwd_ja.cu | $(JA_BUILD_DIR)/
	$(HIPCC) -shared -fPIC $(JA_CXXFLAGS) $(AMDGPU_TARGET_FLAGS) $(JA_INCLUDES) $< -o $@

$(JA_BUILD_DIR)/mha_bwd_ja.so: csrc/ffi/mha_bwd/mha_bwd_ja.cu | $(JA_BUILD_DIR)/
	$(HIPCC) -shared -fPIC $(JA_CXXFLAGS) $(AMDGPU_TARGET_FLAGS) $(JA_INCLUDES) $< -o $@

$(JA_BUILD_DIR)/rmsnorm_fwd_ja.so: csrc/ffi/rmsnorm/rmsnorm_fwd_ja.cu | $(JA_BUILD_DIR)/
	$(HIPCC) -shared -fPIC $(JA_CXXFLAGS) $(AMDGPU_TARGET_FLAGS) $(RMSNORM_INCLUDES) $< -o $@

$(JA_BUILD_DIR)/silu_and_mul_ja.so: csrc/ffi/activation/silu_and_mul_ja.cu | $(JA_BUILD_DIR)/
	$(HIPCC) -shared -fPIC $(JA_CXXFLAGS) $(AMDGPU_TARGET_FLAGS) $(JA_INCLUDES) $< -o $@

$(GEMM_BF16_CFG): $(AITER_SRC_DIR)/hsa/codegen.py | $(GEMM_CONFIG_DIR)/
	cd $(AITER_SRC_DIR) && AITER_GPU_ARCHS="$(GPU_ARCHS)" $(PYTHON3) hsa/codegen.py -m bf16gemm -o $(CURDIR)/$(GEMM_CONFIG_DIR)

$(GEMM_FP4_CFG): $(AITER_SRC_DIR)/hsa/codegen.py | $(GEMM_CONFIG_DIR)/
	cd $(AITER_SRC_DIR) && AITER_GPU_ARCHS="$(GPU_ARCHS)" $(PYTHON3) hsa/codegen.py -m f4gemm -o $(CURDIR)/$(GEMM_CONFIG_DIR)

$(JA_BUILD_DIR)/gemm_fwd_ja.so: csrc/ffi/gemm_fwd/gemm_fwd_ja.cu $(GEMM_BF16_CFG) | $(JA_BUILD_DIR)/
	$(HIPCC) -shared -fPIC $(JA_CXXFLAGS) $(AMDGPU_TARGET_FLAGS) $(GEMM_INCLUDES) $< -o $@

$(JA_BUILD_DIR)/gemm_fp4_ja.so: csrc/ffi/gemm_fp4/gemm_fp4_ja.cu $(GEMM_FP4_CFG) | $(JA_BUILD_DIR)/
	$(HIPCC) -shared -fPIC $(JA_CXXFLAGS) $(AMDGPU_TARGET_FLAGS) $(GEMM_INCLUDES) $< -o $@

CAST_MXFP4_SRC := csrc/ffi/cast_mxfp4/cast_mxfp4_ja.cu
CAST_MXFP4_KERNEL := csrc/ffi/cast_mxfp4/cast_transpose_mxfp4_kernel_shuffled.cu
$(JA_BUILD_DIR)/cast_mxfp4_ja.so: $(CAST_MXFP4_SRC) $(CAST_MXFP4_KERNEL) | $(JA_BUILD_DIR)/
	$(HIPCC) -shared -fPIC $(JA_CXXFLAGS) $(AMDGPU_TARGET_FLAGS) $(JA_INCLUDES) \
		-Icsrc/ffi/cast_mxfp4 $(CAST_MXFP4_SRC) $(CAST_MXFP4_KERNEL) -o $@

# Stage-only clean: wipes the wheel staging dirs (pure copies of the
# source-of-truth libs/kernels) plus the stale moe_fwd_ja.so that has no
# source. Safe to run during release work -- it NEVER touches
# build/aiter_build/ (the multi-GB MHA JIT libs) nor the live *_ja.so /
# libjax_aiter.so in build/jax_aiter_build/.
clean-stage:
	rm -rf jax_aiter/_lib/ jax_aiter/_hsa/ build/lib/ build/jax_aiter_build/moe_fwd_ja.so

# WARNING: `make clean` deletes build/aiter_build/, which holds the
# multi-GB MHA JIT libs (libmha_fwd.so ~1.3 GB, libmha_bwd.so ~1.0 GB,
# librmsnorm_fwd.so ~167 MB) that cost hours to rebuild. DO NOT run this
# during lite/full wheel release work -- use `make clean-stage` instead.
clean:
	rm -rf build/jax_aiter_build build/aiter_build
