// Force-included ahead of Primus-Turbo csrc/kernels/deep_ep/launch.cuh, whose
// `*kernel<<<...>>>(...)` parses as a dereference of the launch result under
// HIP 7.15 clang. Defining LAUNCH_KERNEL skips that block; the self-referential
// macro expands to the templates below.
#pragma once

#include "primus_turbo/macros.h"
#include <hip/hip_runtime.h>
#include <memory>
#include <utility>

namespace deepep_ab {
template <typename T> void fill_kernel_args(void **f, size_t idx, T &&arg) {
    f[idx] = static_cast<void *>(std::addressof(arg));
}

template <typename Head, typename... Tail>
void fill_kernel_args(void **f, size_t idx, Head &&head, Tail &&...tail) {
    f[idx] = static_cast<void *>(std::addressof(head));
    fill_kernel_args(f, idx + 1, std::forward<Tail>(tail)...);
}
} // namespace deepep_ab

template <typename T, typename Kern, typename... Args>
inline void LAUNCH_KERNEL(T &&config, Kern &&kernel, Args &&...args) {
    constexpr size_t k_num_kernel_args = sizeof...(args);
    void            *kernel_args[k_num_kernel_args];
    deepep_ab::fill_kernel_args(kernel_args, 0, std::forward<Args>(args)...);
    PRIMUS_TURBO_CHECK_HIP(hipLaunchCooperativeKernel(std::forward<Kern>(kernel), config->num_sms,
                                                      config->num_threads, kernel_args,
                                                      config->shared_mem_bytes, config->stream));
}

template <typename T, typename Kern, typename... Args>
inline void LAUNCH_KERNEL_NON_COOPERATIVE(T &&config, Kern &&kernel, Args &&...args) {
    (*kernel)<<<config->num_sms, config->num_threads, config->shared_mem_bytes, config->stream>>>(
        std::forward<Args>(args)...);
}

#define LAUNCH_KERNEL LAUNCH_KERNEL
