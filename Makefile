# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# JAX-AITER build. No PyTorch dependency.
# Targets: all (umbrella lib), ja_mods (FFI modules), clean.

HIPCC        ?= /opt/rocm/bin/hipcc
ROCM_ARCH    ?= gfx942
PYTHON3      ?= python3
HIP_LIB      := /opt/rocm/lib

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

UMBRELLA_LDFLAGS := -lamdhip64 -lhiprtc -Wl,-rpath,$(HIP_LIB) -Wl,-soname,libjax_aiter.so

JA_BUILD_DIR := build/jax_aiter_build

GPU_ARCHS ?= $(if $(GFX),$(GFX),$(ROCM_ARCH))
GPU_ARCHS_LIST := $(subst ;, ,$(GPU_ARCHS))
AMDGPU_TARGET_FLAGS := $(foreach arch,$(GPU_ARCHS_LIST),--offload-arch=$(arch))

JA_CXXFLAGS := -std=c++20 -fPIC -O3 -DUSE_ROCM -D__HIP_PLATFORM_AMD__ \
               -fvisibility-inlines-hidden -fvisibility=hidden

JA_INCLUDES := -I$(AITER_SRC_DIR)/3rdparty/composable_kernel/include \
               -I$(AITER_SRC_DIR)/3rdparty/composable_kernel/library/include \
               -I$(AITER_SRC_DIR)/3rdparty/composable_kernel/example/ck_tile/01_fmha \
               -I$(JAX_FFI_INC) -I$(PYTHON_INC) -I$(JAX_AITER_INC) \
               -I$(AITER_INC) -I$(AITER_SRC_DIR)/csrc/include

JA_MODULES := $(JA_BUILD_DIR)/mha_fwd_ja.so \
              $(JA_BUILD_DIR)/mha_bwd_ja.so

.PHONY: all clean ja_mods

all: $(OUT_SO)

ja_mods: $(JA_MODULES)

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

clean:
	rm -rf build/jax_aiter_build build/aiter_build
