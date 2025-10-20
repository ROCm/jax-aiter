# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# JAX-AITER Umbrella Makefile.
#
# This Makefile serves as the central build script for the JAX-AITER project.
# It orchestrates the compilation and linking of the main JAX-AITER shared library,
# sets up all necessary include paths for dependencies (PyTorch, Triton, AITER, JAX, etc.),
# and provides a single entry point.
#
# Usage:
#   make         - Build the JAX-AITER shared library
#   make clean   - Remove all build artifacts


HIPCC        ?= /opt/rocm/bin/hipcc
CC           ?= /opt/rocm/lib/llvm/bin/clang
ROCM_ARCH    ?= gfx942
PYTHON3      ?= python3
HIP_LIB      := /opt/rocm/lib

TORCH_SITE   := third_party/pytorch
TORCH_INC    := $(TORCH_SITE)/torch/csrc/api/include
TORCH_API_INC:= $(TORCH_SITE)/build_static
TORCH_LIBDIR := $(TORCH_SITE)/lib

AITER_SRC_DIR:= third_party/aiter
AITER_HIP_DIR:= build/hipified_aiter
AITER_INC    := $(AITER_HIP_DIR)/csrc/include

TRITON_INC   := $(shell $(PYTHON3) -c 'import importlib.util, os; m=importlib.util.find_spec("triton"); p=(f"{m.submodule_search_locations[0]}/backends/nvidia/include") if m else ""; print(p if p and os.path.isdir(p) else "")')
TRITON_INC_FLAG := $(if $(wildcard $(TRITON_INC)),-I$(TRITON_INC),)
JAX_FFI_INC  := $(shell $(PYTHON3) -c 'from jax import ffi; print(ffi.include_dir())')
PYTHON_INC   := $(shell $(PYTHON3) -c 'import sysconfig; print(sysconfig.get_paths()["include"])')

JAX_AITER_INC:= csrc/common

TORCH_STATIC_LIBDIR := ./third_party/pytorch/build_static/lib
SLEEF_LIBDIR        := third_party/pytorch/build_static/sleef/lib

CXXFLAGS := -std=c++20 -fPIC -O3 -DUSE_ROCM -D__HIP_PLATFORM_AMD__ \
            -I$(JAX_FFI_INC) -I$(PYTHON_INC) \
            -I$(TORCH_SITE) -I$(TORCH_INC) \
            -I$(TORCH_API_INC) -I$(TORCH_API_INC)/install/include \
            -I$(TORCH_SITE)/torch/csrc \
            $(TRITON_INC_FLAG) -I$(AITER_INC) \
            -I$(JAX_AITER_INC) \
            -fvisibility-inlines-hidden -fvisibility=hidden

LDFLAGS := -Wl,--whole-archive \
    $(TORCH_STATIC_LIBDIR)/libtorch_cpu.a \
    $(TORCH_STATIC_LIBDIR)/libtorch_hip.a \
    $(TORCH_STATIC_LIBDIR)/libc10.a \
    $(TORCH_STATIC_LIBDIR)/libc10_hip.a \
  -Wl,--no-whole-archive \
  -Wl,--start-group \
    $(TORCH_STATIC_LIBDIR)/libcaffe2_nvrtc.a \
    $(TORCH_STATIC_LIBDIR)/libcpuinfo.a \
    $(SLEEF_LIBDIR)/libsleef.a \
    $(TORCH_STATIC_LIBDIR)/libprotobuf-lite.a \
    $(TORCH_STATIC_LIBDIR)/libonnx_proto.a \
    $(TORCH_STATIC_LIBDIR)/libonnx.a \
  -Wl,--end-group \
  -Wl,--no-as-needed \
  -lhipsparse -lhipblaslt -lhipblas \
  -lhipfft -lhipsolver -lMIOpen \
  -lamdhip64 \
  -lhiprtc \
  -Wl,-rpath,$(HIP_LIB) \
  -ldl -lpthread -fopenmp \
  -Wl,--gc-sections \
  -Wl,--no-gnu-unique \
  -Wl,-soname,libjax_aiter.so

OUT_SO := build/jax_aiter_build/libjax_aiter.so

# JA modules directory and flags.
JA_BUILD_DIR := build/jax_aiter_build

# Architecture selection for JA modules
# Examples:
#   make ja_mods GPU_ARCHS=gfx942;gfx950
#   make ja_mods GFX=gfx942;gfx950
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
               -I$(AITER_INC) -I$(AITER_SRC_DIR)/csrc/include

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

.PHONY: all lib configure clean ja_mods

all: $(OUT_SO)

ja_mods: $(JA_MODULES)

configure:

%/: 
	mkdir -p $@

$(OUT_SO): build/jax_aiter_build/
	$(HIPCC) -shared -fPIC $(CXXFLAGS) $(LDFLAGS) -o $@

# JA module rules.
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
	@echo "Cleaned build directories"
