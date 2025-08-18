// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

extern "C" void GemmA8W8(void);

static void (*const keep_GemmA8W8)(void) __attribute__((used)) = GemmA8W8;
