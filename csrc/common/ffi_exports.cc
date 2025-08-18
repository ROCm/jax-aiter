// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ffi_exports.h"

#if !defined(__HIP_DEVICE_COMPILE__)

// Create a real reference so that the linker wouldn't eat it.
static void (*const keep_GemmA8W8)(void) __attribute__((used)) = &GemmA8W8;

#endif // !__HIP_DEVICE_COMPILE__
