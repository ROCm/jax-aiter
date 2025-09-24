// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#include <cstdio>
#include <cstdlib>

namespace jax_aiter {

// #define JAX_AITER_DEBUG 1

#ifdef JAX_AITER_DEBUG
#define JA_LOG(fmt, ...)                                                       \
  std::fprintf(stderr, "[JAX_AITER_CPP] " fmt "\n", ##__VA_ARGS__)
#else
#define JA_LOG(fmt, ...)                                                       \
  do {                                                                         \
  } while (0)
#endif

// A very small CHECK() aborts on failure in debug builds.
#ifdef JAX_AITER_DEBUG
#define JA_CHECK(cond, msg)                                                    \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "[JAX_AITER_CPP][FATAL] %s:%d: %s\n", __FILE__,         \
                   __LINE__, msg);                                             \
      std::abort();                                                            \
    }                                                                          \
  } while (0)
#else
#define JA_CHECK(cond, msg)                                                    \
  do {                                                                         \
  } while (0)
#endif

} // namespace jax_aiter
