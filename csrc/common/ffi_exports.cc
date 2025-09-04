// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ffi_exports.h"

#if !defined(__HIP_DEVICE_COMPILE__)

// Create a real reference so that the linker wouldn't eat it.
static void (*const keep_GemmA8W8)(void) __attribute__((used)) = &GemmA8W8;
static void (*const keep_MhaFwd)(void) __attribute__((used)) = &MhaFwd;
static void (*const keep_MhaBwd)(void) __attribute__((used)) = &MhaBwd;
static void (*const keep_FmhaV3Fwd)(void) __attribute__((used)) = &FmhaV3Fwd;
static void (*const keep_FmhaV3Bwd)(void) __attribute__((used)) = &FmhaV3Bwd;
static void (*const keep_MhaVarlenFwd)(void) __attribute__((used)) = &MhaVarlenFwd;
static void (*const keep_MhaVarlenBwd)(void) __attribute__((used)) = &MhaVarlenBwd;
static void (*const keep_MhaBatchPrefill)(void) __attribute__((used)) = &MhaBatchPrefill;
static void (*const keep_AsmMhaVarlenBwd)(void) __attribute__((used)) = &FmhaV3VarlenBwd;

#endif // !__HIP_DEVICE_COMPILE__
