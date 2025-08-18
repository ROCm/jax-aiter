HIPCC       ?= /opt/rocm/bin/hipcc
ROCM_ARCH   ?= gfx942
PYTHON3     ?= python3

HIP_LIB       := /opt/rocm/lib


TORCH_SITE    := /usr/local/lib/python3.10/dist-packages/torch
TORCH_INC     := $(TORCH_SITE)/include
TORCH_API_INC := $(TORCH_INC)/torch/csrc/api/include
TORCH_LIBDIR  := $(TORCH_SITE)/lib

AITER_SRC_DIR := third_party/aiter
AITER_HIP_DIR := build/hipified_aiter
AITER_INC     := $(AITER_HIP_DIR)/csrc/include

TRITON_INC    := /usr/local/lib/python3.10/dist-packages/triton/backends/nvidia

JAX_FFI_INC   := $(shell $(PYTHON3) -c 'from jax import ffi; print(ffi.include_dir())')
PYTHON_INC    := $(shell $(PYTHON3) -c 'import sysconfig; print(sysconfig.get_paths()["include"])')

HIPIFIED_MARKER := build/.hipify_done

JAX_AITER_INC := csrc/common


CXXFLAGS := -std=c++17 -fPIC -O3 -DUSE_ROCM -D__HIP_PLATFORM_AMD__ \
            -I$(JAX_FFI_INC) -I$(PYTHON_INC) \
            -I$(TORCH_INC) -I$(TORCH_API_INC) \
            -I$(TRITON_INC) -I$(AITER_INC) \
            -I$(JAX_AITER_INC) \
            --offload-arch=$(ROCM_ARCH) -v \
            -fvisibility-inlines-hidden \
            -fvisibility=hidden

TORCH_STATIC_LIBDIR := ./third_party/pytorch/build_static/lib
SLEEF_LIBDIR        := third_party/pytorch/build_static/install/lib

LDFLAGS := -Wl,--whole-archive \
    $(TORCH_STATIC_LIBDIR)/libtorch_cpu.a \
    $(TORCH_STATIC_LIBDIR)/libtorch_hip.a \
  -Wl,--no-whole-archive \
    $(TORCH_STATIC_LIBDIR)/libc10_hip.a \
    $(TORCH_STATIC_LIBDIR)/libc10.a \
  -Wl,--start-group \
    $(TORCH_STATIC_LIBDIR)/libfbgemm.a \
    $(TORCH_STATIC_LIBDIR)/libpthreadpool.a \
    $(TORCH_STATIC_LIBDIR)/libcpuinfo.a \
    $(TORCH_STATIC_LIBDIR)/libdnnl.a \
    -L$(SLEEF_LIBDIR) -lsleef \
    -lm \
    $(TORCH_STATIC_LIBDIR)/libfbgemm.a \
    $(TORCH_STATIC_LIBDIR)/libpytorch_qnnpack.a \
    $(TORCH_STATIC_LIBDIR)/libasmjit.a \
    $(TORCH_STATIC_LIBDIR)/libclog.a \
    $(TORCH_STATIC_LIBDIR)/libprotobuf-lite.a \
    $(TORCH_STATIC_LIBDIR)/libonnx_proto.a \
    $(TORCH_STATIC_LIBDIR)/libonnx.a \
  -Wl,--end-group \
  -Wl,--no-as-needed \
  -lhipsparse -lhipblaslt -lhipblas \
    -lhipfft -lhipsolver -lMIOpen \
    -lamdhip64 \
  -Wl,-rpath,$(HIP_LIB) \
  -ldl -lpthread -fopenmp\
  -Wl,--exclude-libs,ALL \
  -Wl,-Bsymbolic-functions \
  -Wl,--gc-sections

# Build targets
OUT_SO := build/bin/libjax_aiter.so
EXPORTS_MAP := build/exports.map

# Input files
MAIN_CU := jax_aiter/ffi/gemm_a8w8/asm_gemm_a8w8.cu
AITER_HIP_SRCS := build/hipified_aiter/csrc/py_itfs_cu/asm_gemm_a8w8_hip.cu

# Object files
OBJS := \
  build/obj/asm_gemm_a8w8.o \
  build/obj/py_itfs_cu/asm_gemm_a8w8_hip.o

.PHONY: all lib hipify configure clean

all: $(HIPIFIED_MARKER) $(OUT_SO)

configure: hipify

hipify: $(HIPIFIED_MARKER)

# Directory creation
%/:
	mkdir -p $@

# Hipify step.
$(HIPIFIED_MARKER): scripts/hipify_aiter.sh
	@echo "[hipify] Running scripts/hipify_aiter.sh"
	@bash scripts/hipify_aiter.sh
	@touch touch $@
	@echo "[hipify] Done"

$(EXPORTS_MAP): | build/
	@cp exports.map build/exports.map

# Compilation rules
build/obj/asm_gemm_a8w8.o: csrc/ffi/gemm_a8w8/asm_gemm_a8w8.cu \
    | build/obj/  $(HIPIFIED_MARKER)
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

build/obj/py_itfs_cu/asm_gemm_a8w8_hip.o: \
    $(AITER_HIP_DIR)/csrc/py_itfs_cu/asm_gemm_a8w8_hip.cu \
    | build/obj/py_itfs_cu/ $(HIPIFIED_MARKER)
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

$(OUT_SO): $(OBJS) $(EXPORTS_MAP) | build/bin/
	$(HIPCC) -shared -fPIC $(CXXFLAGS) $(LDFLAGS) \
	  -Wl,--version-script=$(EXPORTS_MAP) \
	  $(filter-out $(EXPORTS_MAP),$^) -o $@

clean:
	rm -rf build
	@echo "Cleaned build/ directory"
