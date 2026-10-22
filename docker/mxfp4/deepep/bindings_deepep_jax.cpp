// DeepEP-only subset of Primus-Turbo csrc/jax/bindings_jax.cpp: the intranode
// DeepEP FFI handlers, Config, and the deep_ep buffer helpers, without the
// grouped-GEMM, normalization, and quantization registrations.

#include "jax/deep_ep/deep_ep.h"
#include "jax/extensions.h"
#include "jax/ffi.h"
#include "primus_turbo/deep_ep/config.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#define REGISTER_FFI_HANDLER(dict, name, fn) dict[#name] = ::primus_turbo::jax::EncapsulateFFI(fn);

namespace primus_turbo::jax {

template <typename T> pybind11::capsule EncapsulateFFI(T *fn) {
    static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                  "Encapsulated function must be an XLA FFI handler");
    return pybind11::capsule(reinterpret_cast<void *>(fn), "xla._CUSTOM_CALL_TARGET");
}

pybind11::dict Registrations() {
    pybind11::dict dict;
    REGISTER_FFI_HANDLER(dict, moe_dispatch_inproc, MoEDispatchHandler);
    REGISTER_FFI_HANDLER(dict, moe_cached_dispatch_inproc, MoECachedDispatchHandler);
    REGISTER_FFI_HANDLER(dict, moe_combine_inproc, MoECombineHandler);
    REGISTER_FFI_HANDLER(dict, moe_dispatch_per_process, MoEDispatchPerProcessHandler);
    REGISTER_FFI_HANDLER(dict, moe_cached_dispatch_per_process, MoECachedDispatchPerProcessHandler);
    REGISTER_FFI_HANDLER(dict, moe_combine_per_process, MoECombinePerProcessHandler);
    return dict;
}

PYBIND11_MODULE(_C, m) {
    m.def("registrations", &Registrations);

    pybind11::class_<primus_turbo::deep_ep::Config>(m, "Config")
        .def(pybind11::init<int, int, int, int, int>(), pybind11::arg("num_sms") = DEFAULT_NUM_CU,
             pybind11::arg("num_max_nvl_chunked_send_tokens") =
                 DEFAULT_NUM_MAX_XGMI_CHUNKED_SEND_TOKENS,
             pybind11::arg("num_max_nvl_chunked_recv_tokens") =
                 DEFAULT_NUM_MAX_XGMI_CHUNKED_RECV_TOKENS,
             pybind11::arg("num_max_rdma_chunked_send_tokens") =
                 DEFAULT_NUM_MAX_RDMA_CHUNKED_SEND_TOKENS,
             pybind11::arg("num_max_rdma_chunked_recv_tokens") =
                 DEFAULT_NUM_MAX_RDMA_CHUNKED_RECV_TOKENS)
        .def_readonly("num_sms", &primus_turbo::deep_ep::Config::num_sms)
        .def_readonly("num_max_nvl_chunked_send_tokens",
                      &primus_turbo::deep_ep::Config::num_max_nvl_chunked_send_tokens)
        .def_readonly("num_max_nvl_chunked_recv_tokens",
                      &primus_turbo::deep_ep::Config::num_max_nvl_chunked_recv_tokens)
        .def_readonly("num_max_rdma_chunked_send_tokens",
                      &primus_turbo::deep_ep::Config::num_max_rdma_chunked_send_tokens)
        .def_readonly("num_max_rdma_chunked_recv_tokens",
                      &primus_turbo::deep_ep::Config::num_max_rdma_chunked_recv_tokens)
        .def("get_nvl_buffer_size_hint", &primus_turbo::deep_ep::Config::get_nvl_buffer_size_hint)
        .def("get_rdma_buffer_size_hint",
             &primus_turbo::deep_ep::Config::get_rdma_buffer_size_hint);

    namespace dep = deep_ep;
    auto dep_m    = m.def_submodule("deep_ep");
    dep_m.def("create_per_process_buffer", &dep::create_per_process_buffer, py::arg("rank"),
              py::arg("num_ranks"), py::arg("num_nvl_bytes"), py::arg("num_rdma_bytes") = 0);
    dep_m.def("sync_per_process_buffer", &dep::sync_per_process_buffer, py::arg("all_handles"),
              py::arg("root_unique_id") = py::none());
    dep_m.def("destroy_per_process_buffer", &dep::destroy_per_process_buffer);
    dep_m.def("is_per_process_buffer_ready", &dep::is_per_process_buffer_ready);
    dep_m.def("per_process_buffer_nvl_bytes", &dep::per_process_buffer_nvl_bytes);
    dep_m.def("has_rocshmem", []() { return false; });
    dep_m.def(
        "get_nvl_buffer_size_hint",
        [](int64_t hidden_bytes, int num_ranks, int num_sms, int nvl_send, int nvl_recv,
           int rdma_send, int rdma_recv) -> int64_t {
            auto cfg =
                primus_turbo::deep_ep::Config(num_sms, nvl_send, nvl_recv, rdma_send, rdma_recv);
            return static_cast<int64_t>(
                cfg.get_nvl_buffer_size_hint(static_cast<size_t>(hidden_bytes), num_ranks));
        },
        py::arg("hidden_bytes"), py::arg("num_ranks"), py::arg("num_sms"), py::arg("nvl_send"),
        py::arg("nvl_recv"), py::arg("rdma_send"), py::arg("rdma_recv"));
}

} // namespace primus_turbo::jax
