// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#ifdef __cplusplus
extern "C" {
#endif

void GemmA8W8(void);
void MhaFwd(void);
void MhaBwd(void);
void FmhaV3Fwd(void);
void FmhaV3Bwd(void);
void MhaVarlenFwd(void);
void MhaVarlenBwd(void);
void MhaBatchPrefill(void);
void FmhaV3VarlenBwd(void);

#ifdef __cplusplus
} // extern "C"
#endif
