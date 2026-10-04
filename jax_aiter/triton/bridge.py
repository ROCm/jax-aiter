# SPDX-License-Identifier: Apache-2.0
"""Load the process-local Triton HSACO trampoline."""

from __future__ import annotations

import ctypes
import threading

from ..ja_compat import config as ja_config

_TARGET_NAME = "TritonDispatchJA"
_BRIDGE_LIBRARY = "triton_bridge_ja.so"

_library: ctypes.CDLL | None = None
_library_lock = threading.Lock()
_target_registered = False
_target_lock = threading.Lock()
_hip: ctypes.CDLL | None = None
_compile_names: dict[int, str] = {}


def _load_library() -> ctypes.CDLL:
    global _library
    if _library is not None:
        return _library

    with _library_lock:
        if _library is not None:
            return _library

        path = ja_config.get_jax_aiter_lib_dir() / _BRIDGE_LIBRARY
        if not path.is_file():
            raise FileNotFoundError(
                f"Triton bridge not found at {path}. "
                "Run `make -f Makefile.triton bridge` in the jax-aiter checkout."
            )

        library = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        library.TritonBridgeRegister.restype = ctypes.c_int32
        library.TritonBridgeRegister.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_char_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
        ]
        for name in (
            "TritonBridgeMaxArgs",
            "TritonBridgeMaxDevices",
            "TritonBridgeMaxCompiles",
        ):
            function = getattr(library, name)
            function.restype = ctypes.c_int32
            function.argtypes = []
        library.TritonBridgeDispatchCount.restype = ctypes.c_uint64
        library.TritonBridgeDispatchCount.argtypes = [ctypes.c_int32]
        _library = library
        return library


def ensure_target_registered() -> str:
    global _target_registered
    if _target_registered:
        return _TARGET_NAME
    with _target_lock:
        if _target_registered:
            return _TARGET_NAME
        _load_library()
        from ..ffi.registry import register_ffi_target

        register_ffi_target(_TARGET_NAME, platform="ROCM")
        _target_registered = True
    return _TARGET_NAME


def max_args() -> int:
    return int(_load_library().TritonBridgeMaxArgs())


def max_devices() -> int:
    return int(_load_library().TritonBridgeMaxDevices())


def max_compiles() -> int:
    return int(_load_library().TritonBridgeMaxCompiles())


def register_kernel(
    hsaco: bytes,
    kernel_name: str,
    num_devices: int,
    num_warps: int,
    num_ctas: int,
    shared: int,
    warp_size: int,
    compile_key: int,
    abi_kinds: list[int],
    name: str | None = None,
) -> int:
    if not hsaco:
        raise ValueError("HSACO blob is empty")
    if num_devices < 1 or num_devices > max_devices():
        raise ValueError(f"Unsupported device count {num_devices}")
    kinds = (ctypes.c_int32 * len(abi_kinds))(*abi_kinds)
    blob = ctypes.create_string_buffer(hsaco, len(hsaco))
    index = int(
        _load_library().TritonBridgeRegister(
            ctypes.cast(blob, ctypes.c_void_p),
            len(hsaco),
            kernel_name.encode("utf-8"),
            int(num_devices),
            int(num_warps),
            int(num_ctas),
            int(shared),
            int(warp_size),
            int(compile_key),
            kinds,
            len(abi_kinds),
        )
    )
    if index == -2:
        raise RuntimeError(
            f"Triton bridge compile table is full ({max_compiles()} entries)"
        )
    if index == -4:
        raise RuntimeError("Triton bridge v1 requires num_ctas==1")
    if index < 0:
        raise RuntimeError(f"Triton bridge registration failed with code {index}")
    _compile_names[index] = name or kernel_name
    return index


def dispatch_count(compile_index: int) -> int:
    return int(_load_library().TritonBridgeDispatchCount(int(compile_index)))


def _load_hip() -> ctypes.CDLL:
    global _hip
    if _hip is not None:
        return _hip
    for name in ("libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"):
        try:
            library = ctypes.CDLL(name)
        except OSError:
            continue
        library.hipGetDevice.restype = ctypes.c_int
        library.hipGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        library.hipSetDevice.restype = ctypes.c_int
        library.hipSetDevice.argtypes = [ctypes.c_int]
        _hip = library
        return library
    raise RuntimeError("Unable to load libamdhip64")


def hip_get_device() -> int:
    current = ctypes.c_int(-1)
    error = _load_hip().hipGetDevice(ctypes.byref(current))
    if error != 0:
        raise RuntimeError(f"hipGetDevice failed with code {error}")
    return int(current.value)


def hip_set_device(device: int) -> None:
    error = _load_hip().hipSetDevice(int(device))
    if error != 0:
        raise RuntimeError(f"hipSetDevice({device}) failed with code {error}")
