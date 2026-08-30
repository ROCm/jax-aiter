// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Torch-free stub. aiter_opus_plus.h includes <c10/util/BFloat16.h> only to
// name t2opus<c10::BFloat16>. cache_kernels.cu never instantiates that
// mapping, but the include is unconditional. This file lets
// `make -f Makefile.kv ja_kv` compile append_kv without PyTorch headers.
#pragma once

namespace c10 {
struct BFloat16 {};
}  // namespace c10
