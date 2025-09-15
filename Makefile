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

TRITON_INC   := $(shell $(PYTHON3) -c 'import triton; print(f"{triton.__path__[0]}/backends/nvidia/include")')
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
            -I$(TRITON_INC) -I$(AITER_INC) \
            -I$(JAX_AITER_INC) \
            -fvisibility-inlines-hidden -fvisibility=hidden

LDFLAGS := -Wl,--whole-archive \
    $(TORCH_STATIC_LIBDIR)/libtorch_cpu.a \
    $(TORCH_STATIC_LIBDIR)/libtorch_hip.a \
    $(TORCH_STATIC_LIBDIR)/libc10.a \
    $(TORCH_STATIC_LIBDIR)/libc10_hip.a \
  -Wl,--no-whole-archive \
  -Wl,--start-group \
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
  -Wl,-rpath,$(HIP_LIB) \
  -ldl -lpthread -fopenmp \
  -Wl,--gc-sections \
  -Wl,--no-gnu-unique \
  -Wl,-soname,libjax_aiter.so

OUT_SO := build/bin/libjax_aiter.so

.PHONY: all lib configure clean

all: $(OUT_SO)

configure:

%/: 
	mkdir -p $@

$(OUT_SO): build/bin/
	$(HIPCC) -shared -fPIC $(CXXFLAGS) $(LDFLAGS) -o $@

clean:
	rm -rf build
	@echo "Cleaned build/ directory"
