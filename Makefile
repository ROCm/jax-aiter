HIPCC       ?= /opt/rocm/bin/hipcc
ROCM_ARCH   ?= gfx942
PYTHON3     ?= python3

HIP_LIB       := /opt/rocm/lib


TORCH_SITE    := third_party/pytorch
TORCH_INC     := $(TORCH_SITE)/torch/csrc/api/include
TORCH_API_INC := $(TORCH_SITE)/build_static
TORCH_LIBDIR  := $(TORCH_SITE)/lib

AITER_SRC_DIR := third_party/aiter
AITER_HIP_DIR := build/hipified_aiter
AITER_INC     := $(AITER_HIP_DIR)/csrc/include
CK_TILE_INC   := $(AITER_HIP_DIR)/3rdparty/composable_kernel/example/ck_tile/01_fmha
CK_TILE_INC2  := $(AITER_HIP_DIR)/aiter/jit/build/ck/example/ck_tile/01_fmha
CK_TILE_INC3  := $(AITER_HIP_DIR)/3rdparty/composable_kernel/include

TRITON_INC    := $(shell $(PYTHON3) -c \
  'import triton; print(f"{triton.__path__[0]}/backends/nvidia/include")')

JAX_FFI_INC   := $(shell $(PYTHON3) -c 'from jax import ffi; print(ffi.include_dir())')
PYTHON_INC    := $(shell $(PYTHON3) -c 'import sysconfig; print(sysconfig.get_paths()["include"])')

HIPIFIED_MARKER := build/.hipify_done

JAX_AITER_INC := csrc/common

CXXFLAGS := -std=c++20 -fPIC -O3 -DUSE_ROCM -D__HIP_PLATFORM_AMD__ \
            -I$(JAX_FFI_INC) -I$(PYTHON_INC) \
            -I$(TORCH_SITE) -I$(TORCH_INC) \
            -I$(TORCH_API_INC) -I$(TORCH_API_INC)/install/include \
            -I$(TORCH_SITE)/torch/csrc \
            -I$(TRITON_INC) -I$(AITER_INC) \
            -I$(CK_TILE_INC) -I$(CK_TILE_INC2) -I$(CK_TILE_INC3) \
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
#   csrc/ffi/gemm_a8w8/asm_gemm_a8w8.cu
JAX_FFI_SRCS := \
  csrc/ffi/custom/custom.cu \
  csrc/ffi/ck_fused_attn_fwd/ck_fused_attn_fwd.cu \
  csrc/ffi/ck_fused_attn_bwd/ck_fused_attn_bwd.cu \
  csrc/ffi/asm_mha_fwd/asm_mha_fwd.cu \
  csrc/ffi/asm_mha_bwd/asm_mha_bwd.cu \
  csrc/ffi/ck_mha_varlen_fwd/ck_mha_varlen_fwd.cu \
  csrc/ffi/ck_mha_varlen_bwd/ck_mha_varlen_bwd.cu \
  csrc/ffi/ck_mha_batch_prefill/ck_mha_batch_prefill.cu \
  csrc/ffi/asm_mha_varlen_bwd/asm_mha_varlen_bwd.cu

AITER_HIP_SRCS := \
  $(AITER_HIP_DIR)/csrc/py_itfs_cu/asm_gemm_a8w8.hip \
  $(AITER_HIP_DIR)/csrc/py_itfs_cu/custom.hip \
  $(AITER_HIP_DIR)/csrc/kernels/custom_kernels.hip \
  $(AITER_HIP_DIR)/csrc/py_itfs_ck/mha_fwd_kernels.hip \
  $(AITER_HIP_DIR)/csrc/py_itfs_ck/mha_bwd_kernels.hip \
  $(AITER_HIP_DIR)/csrc/py_itfs_ck/mha_varlen_fwd_kernels.hip \
  $(AITER_HIP_DIR)/csrc/py_itfs_ck/mha_varlen_bwd_kernels.hip \
  $(AITER_HIP_DIR)/csrc/py_itfs_ck/mha_batch_prefill_kernels.hip \
  $(AITER_HIP_DIR)/csrc/py_itfs_cu/asm_mha_varlen_bwd.hip \
  $(AITER_HIP_DIR)/csrc/py_itfs_cu/asm_mha_fwd.hip \
  $(AITER_HIP_DIR)/csrc/py_itfs_cu/asm_mha_bwd.hip

# JAX AITER Object files.
JA_OBJS := \
  build/obj/custom.o \
  build/obj/asm_gemm_a8w8.o \
  build/obj/ck_fused_attn_fwd.o \
  build/obj/ck_fused_attn_bwd.o \
  build/obj/asm_mha_fwd.o \
  build/obj/asm_mha_bwd.o \
  build/obj/ck_mha_varlen_fwd.o \
  build/obj/ck_mha_varlen_bwd.o \
  build/obj/ck_mha_batch_prefill.o \
  build/obj/asm_mha_varlen_bwd.o

# AITER Link Object files.
AITER_OBJS := \
  build/obj/aiter_asm_gemm_a8w8.o \
  build/obj/aiter_custom.o \
  build/obj/custom_kernels.o \
  build/obj/aiter_mha_fwd_kernels.o \
  build/obj/aiter_mha_bwd_kernels.o \
  build/obj/aiter_mha_varlen_fwd_kernels.o \
  build/obj/aiter_mha_varlen_bwd_kernels.o \
  build/obj/aiter_mha_batch_prefill_kernels.o \
  build/obj/aiter_asm_mha_varlen_bwd.o \
  build/obj/aiter_mha_common.o \
  build/obj/aiter_asm_mha_fwd.o \
  build/obj/aiter_asm_mha_bwd.o

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
	@touch $@
	@echo "[hipify] Done"

$(EXPORTS_MAP): | build/
	@cp exports.map build/exports.map

# Compilation rules
build/obj/asm_gemm_a8w8.o: csrc/ffi/gemm_a8w8/asm_gemm_a8w8.cu \
    | build/obj/  $(HIPIFIED_MARKER)
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

build/obj/custom.o: csrc/ffi/custom/custom.cu \
    | build/obj/  $(HIPIFIED_MARKER)
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

build/obj/ck_fused_attn_fwd.o: csrc/ffi/ck_fused_attn_fwd/ck_fused_attn_fwd.cu \
    | build/obj/  $(HIPIFIED_MARKER)
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

build/obj/ck_fused_attn_bwd.o: csrc/ffi/ck_fused_attn_bwd/ck_fused_attn_bwd.cu \
    | build/obj/  $(HIPIFIED_MARKER)
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

build/obj/asm_mha_fwd.o: csrc/ffi/asm_mha_fwd/asm_mha_fwd.cu \
    | build/obj/  $(HIPIFIED_MARKER)
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

build/obj/asm_mha_bwd.o: csrc/ffi/asm_mha_bwd/asm_mha_bwd.cu \
    | build/obj/  $(HIPIFIED_MARKER)
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

build/obj/ck_mha_varlen_fwd.o: csrc/ffi/ck_mha_varlen_fwd/ck_mha_varlen_fwd.cu \
    | build/obj/  $(HIPIFIED_MARKER)
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

build/obj/ck_mha_varlen_bwd.o: csrc/ffi/ck_mha_varlen_bwd/ck_mha_varlen_bwd.cu \
    | build/obj/  $(HIPIFIED_MARKER)
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

build/obj/ck_mha_batch_prefill.o: csrc/ffi/ck_mha_batch_prefill/ck_mha_batch_prefill.cu \
    | build/obj/  $(HIPIFIED_MARKER)
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

build/obj/asm_mha_varlen_bwd.o: csrc/ffi/asm_mha_varlen_bwd/asm_mha_varlen_bwd.cu \
    | build/obj/  $(HIPIFIED_MARKER)
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

# AITER imports.

build/obj/aiter_asm_gemm_a8w8.o: $(AITER_HIP_DIR)/csrc/py_itfs_cu/asm_gemm_a8w8.hip \
    | build/obj/  $(HIPIFIED_MARKER)
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

build/obj/aiter_custom.o: $(AITER_HIP_DIR)/csrc/py_itfs_cu/custom.hip \
    | build/obj/  $(HIPIFIED_MARKER)
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

build/obj/custom_kernels.o: $(AITER_HIP_DIR)/csrc/kernels/custom_kernels.hip \
    | build/obj/kernels/  $(HIPIFIED_MARKER)
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

build/obj/aiter_mha_fwd_kernels.o: $(AITER_HIP_DIR)/csrc/py_itfs_ck/mha_fwd_kernels.hip \
    | build/obj/  $(HIPIFIED_MARKER)
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

build/obj/aiter_mha_bwd_kernels.o: $(AITER_HIP_DIR)/csrc/py_itfs_ck/mha_bwd_kernels.hip \
    | build/obj/  $(HIPIFIED_MARKER)
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

build/obj/aiter_mha_varlen_fwd_kernels.o: $(AITER_HIP_DIR)/csrc/py_itfs_ck/mha_varlen_fwd_kernels.hip \
    | build/obj/  $(HIPIFIED_MARKER)
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

build/obj/aiter_mha_varlen_bwd_kernels.o: $(AITER_HIP_DIR)/csrc/py_itfs_ck/mha_varlen_bwd_kernels.hip \
    | build/obj/  $(HIPIFIED_MARKER)
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

build/obj/aiter_mha_batch_prefill_kernels.o: $(AITER_HIP_DIR)/csrc/py_itfs_ck/mha_batch_prefill_kernels.hip \
    | build/obj/  $(HIPIFIED_MARKER)
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

build/obj/aiter_asm_mha_varlen_bwd.o: $(AITER_HIP_DIR)/csrc/py_itfs_cu/asm_mha_varlen_bwd.hip \
    | build/obj/  $(HIPIFIED_MARKER)
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

build/obj/aiter_mha_common.o: $(AITER_HIP_DIR)/csrc/kernels/mha_common.hip \
    | build/obj/  $(HIPIFIED_MARKER)
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

build/obj/aiter_asm_mha_fwd.o: $(AITER_HIP_DIR)/csrc/py_itfs_cu/asm_mha_fwd.hip \
    | build/obj/  $(HIPIFIED_MARKER)
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

build/obj/aiter_asm_mha_bwd.o: $(AITER_HIP_DIR)/csrc/py_itfs_cu/asm_mha_bwd.hip \
    | build/obj/  $(HIPIFIED_MARKER)
	$(HIPCC) $(CXXFLAGS) -c $< -o $@

$(OUT_SO): $(JA_OBJS) $(AITER_OBJS) $(EXPORTS_MAP) | build/bin/
	$(HIPCC) -shared -fPIC $(CXXFLAGS) $(LDFLAGS) \
	  -Wl,--version-script=$(EXPORTS_MAP) \
	  $(filter-out $(EXPORTS_MAP),$^) -o $@

clean:
	rm -rf build
	@echo "Cleaned build/ directory"
