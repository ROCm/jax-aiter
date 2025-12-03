# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# JAX-AITER umbrella build.
# Targets: all (umbrella lib), ja_mods (FFI modules), clean.

HIPCC        ?= /opt/rocm/bin/hipcc
ROCM_ARCH    ?= gfx942
PYTHON3      ?= python3
HIP_LIB      := /opt/rocm/lib

TORCH_SITE   := third_party/pytorch
TORCH_INC    := $(TORCH_SITE)/torch/csrc/api/include
TORCH_API_INC:= $(TORCH_SITE)/build_static

AITER_SRC_DIR:= third_party/aiter
AITER_HIP_DIR:= build/hipified_aiter
AITER_INC    := $(AITER_HIP_DIR)/csrc/include

JAX_FFI_INC  := $(shell $(PYTHON3) -c 'from jax import ffi; print(ffi.include_dir())')
PYTHON_INC   := $(shell $(PYTHON3) -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
JAX_AITER_INC:= csrc/common

# Umbrella library (mha_common_utils.cu) - no Torch dependency.
OUT_SO := build/jax_aiter_build/libjax_aiter.so

UMBRELLA_CXXFLAGS := -std=c++20 -fPIC -O3 -DUSE_ROCM -D__HIP_PLATFORM_AMD__ \
                     -I$(JAX_FFI_INC) -I$(PYTHON_INC) -I$(JAX_AITER_INC) -I$(AITER_INC) \
                     -fvisibility-inlines-hidden -fvisibility=hidden

UMBRELLA_LDFLAGS := -lamdhip64 -lhiprtc -Wl,-rpath,$(HIP_LIB) -Wl,-soname,libjax_aiter.so

# JA modules - link against AITER kernels.
JA_BUILD_DIR := build/jax_aiter_build

# Architecture selection: make ja_mods GPU_ARCHS=gfx942;gfx950 or GFX=gfx942;gfx950.
GPU_ARCHS ?= $(if $(GFX),$(GFX),$(ROCM_ARCH))
GPU_ARCHS_LIST := $(subst ;, ,$(GPU_ARCHS))
AMDGPU_TARGET_FLAGS := $(foreach arch,$(GPU_ARCHS_LIST),--offload-arch=$(arch))

JA_CXXFLAGS := -std=c++20 -fPIC -O3 -DUSE_ROCM -D__HIP_PLATFORM_AMD__ \
               -fvisibility-inlines-hidden -fvisibility=hidden

JA_INCLUDES := -I$(JAX_FFI_INC) -I$(PYTHON_INC) -I$(JAX_AITER_INC) \
               -I$(TORCH_SITE) -I$(TORCH_INC) -I$(TORCH_API_INC) \
               -I$(TORCH_API_INC)/install/include -I$(TORCH_SITE)/torch/csrc \
               -I$(TORCH_SITE)/aten -I$(TORCH_SITE)/aten/src -I$(TORCH_SITE)/aten/src/ATen \
               -I$(TORCH_API_INC)/aten/src -I$(TORCH_API_INC)/aten/src/ATen \
               -I$(AITER_INC) -I$(AITER_SRC_DIR)/csrc/include \
               -I$(AITER_SRC_DIR)/3rdparty/composable_kernel/example/ck_tile/01_fmha

# JA module targets.
JA_MODULES := $(JA_BUILD_DIR)/custom_ja.so \
              $(JA_BUILD_DIR)/asm_mha_fwd_ja.so \
              $(JA_BUILD_DIR)/asm_mha_bwd_ja.so \
              $(JA_BUILD_DIR)/asm_mha_varlen_fwd_ja.so \
              $(JA_BUILD_DIR)/asm_mha_varlen_bwd_ja.so \
              $(JA_BUILD_DIR)/ck_fused_attn_fwd_ja.so \
              $(JA_BUILD_DIR)/ck_fused_attn_bwd_ja.so \
              $(JA_BUILD_DIR)/ck_mha_varlen_fwd_ja.so \
              $(JA_BUILD_DIR)/ck_mha_varlen_bwd_ja.so \
              $(JA_BUILD_DIR)/ck_mha_batch_prefill_ja.so

.PHONY: all clean ja_mods

all: $(OUT_SO)

ja_mods: $(JA_MODULES)

%/: 
	mkdir -p $@

$(OUT_SO): build/jax_aiter_build/ csrc/common/mha_common_utils.cu
	$(HIPCC) -shared $(UMBRELLA_CXXFLAGS) \
		-I$(AITER_SRC_DIR)/3rdparty/composable_kernel/example/ck_tile/01_fmha \
		csrc/common/mha_common_utils.cu \
		$(UMBRELLA_LDFLAGS) -o $@

$(JA_BUILD_DIR)/custom_ja.so: csrc/ffi/custom/custom_ja.cu | $(JA_BUILD_DIR)/
	$(HIPCC) -shared -fPIC $(JA_CXXFLAGS) $(AMDGPU_TARGET_FLAGS) $(JA_INCLUDES) $< -o $@

$(JA_BUILD_DIR)/asm_mha_fwd_ja.so: csrc/ffi/asm_mha_fwd/asm_mha_fwd_ja.cu | $(JA_BUILD_DIR)/
	$(HIPCC) -shared -fPIC $(JA_CXXFLAGS) $(AMDGPU_TARGET_FLAGS) $(JA_INCLUDES) $< -o $@

$(JA_BUILD_DIR)/asm_mha_bwd_ja.so: csrc/ffi/asm_mha_bwd/asm_mha_bwd_ja.cu | $(JA_BUILD_DIR)/
	$(HIPCC) -shared -fPIC $(JA_CXXFLAGS) $(AMDGPU_TARGET_FLAGS) $(JA_INCLUDES) $< -o $@

$(JA_BUILD_DIR)/asm_mha_varlen_fwd_ja.so: csrc/ffi/asm_mha_varlen_fwd/asm_mha_varlen_fwd_ja.cu | $(JA_BUILD_DIR)/
	$(HIPCC) -shared -fPIC $(JA_CXXFLAGS) $(AMDGPU_TARGET_FLAGS) $(JA_INCLUDES) $< -o $@

$(JA_BUILD_DIR)/asm_mha_varlen_bwd_ja.so: csrc/ffi/asm_mha_varlen_bwd/asm_mha_varlen_bwd_ja.cu | $(JA_BUILD_DIR)/
	$(HIPCC) -shared -fPIC $(JA_CXXFLAGS) $(AMDGPU_TARGET_FLAGS) $(JA_INCLUDES) $< -o $@

$(JA_BUILD_DIR)/ck_fused_attn_fwd_ja.so: csrc/ffi/ck_fused_attn_fwd/ck_fused_attn_fwd_ja.cu | $(JA_BUILD_DIR)/
	$(HIPCC) -shared -fPIC $(JA_CXXFLAGS) $(AMDGPU_TARGET_FLAGS) $(JA_INCLUDES) $< -o $@

$(JA_BUILD_DIR)/ck_fused_attn_bwd_ja.so: csrc/ffi/ck_fused_attn_bwd/ck_fused_attn_bwd_ja.cu | $(JA_BUILD_DIR)/
	$(HIPCC) -shared -fPIC $(JA_CXXFLAGS) $(AMDGPU_TARGET_FLAGS) $(JA_INCLUDES) $< -o $@

$(JA_BUILD_DIR)/ck_mha_varlen_fwd_ja.so: csrc/ffi/ck_mha_varlen_fwd/ck_mha_varlen_fwd_ja.cu | $(JA_BUILD_DIR)/
	$(HIPCC) -shared -fPIC $(JA_CXXFLAGS) $(AMDGPU_TARGET_FLAGS) $(JA_INCLUDES) $< -o $@

$(JA_BUILD_DIR)/ck_mha_varlen_bwd_ja.so: csrc/ffi/ck_mha_varlen_bwd/ck_mha_varlen_bwd_ja.cu | $(JA_BUILD_DIR)/
	$(HIPCC) -shared -fPIC $(JA_CXXFLAGS) $(AMDGPU_TARGET_FLAGS) $(JA_INCLUDES) $< -o $@

$(JA_BUILD_DIR)/ck_mha_batch_prefill_ja.so: csrc/ffi/ck_mha_batch_prefill/ck_mha_batch_prefill_ja.cu | $(JA_BUILD_DIR)/
	$(HIPCC) -shared -fPIC $(JA_CXXFLAGS) $(AMDGPU_TARGET_FLAGS) $(JA_INCLUDES) $< -o $@

clean:
	rm -rf build/jax_aiter_build build/aiter_build
