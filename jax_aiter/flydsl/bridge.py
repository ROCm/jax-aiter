# SPDX-License-Identifier: Apache-2.0
"""Load and manage the process-local typed-FFI FlyDSL trampoline."""

from __future__ import annotations

import atexit
import ctypes
import os
import threading

from ..ja_compat import config as ja_config

_TARGET_NAME = "FlydslDispatchJA"
_BRIDGE_LIBRARY = "flydsl_bridge_ja.so"

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
                f"FlyDSL bridge not found at {path}. "
                "Run `make -f Makefile.flydsl bridge` in the jax-aiter checkout."
            )

        library = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        library.FlydslBridgeRegister.restype = ctypes.c_int32
        library.FlydslBridgeRegister.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int32,
            ctypes.c_int64,
        ]
        for name in (
            "FlydslBridgeMaxArgs",
            "FlydslBridgeMaxDevices",
            "FlydslBridgeMaxCompiles",
        ):
            function = getattr(library, name)
            function.restype = ctypes.c_int32
            function.argtypes = []
        library.FlydslBridgeDispatchCount.restype = ctypes.c_uint64
        library.FlydslBridgeDispatchCount.argtypes = [ctypes.c_int32]

        _library = library
        return library


def ensure_target_registered() -> str:
    """Register the typed FFI handler once and return its target name."""
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
    return int(_load_library().FlydslBridgeMaxArgs())


def max_devices() -> int:
    return int(_load_library().FlydslBridgeMaxDevices())


def max_compiles() -> int:
    return int(_load_library().FlydslBridgeMaxCompiles())


def register_compile(
    function_ptrs: list[int], compile_key: int, name: str | None = None
) -> int:
    """Register one compiled FlyDSL host wrapper per local GPU."""
    if not function_ptrs:
        raise ValueError("At least one FlyDSL function pointer is required")
    if len(function_ptrs) > max_devices():
        raise ValueError(
            f"Got {len(function_ptrs)} devices, bridge supports {max_devices()}"
        )
    pointers = (ctypes.c_void_p * len(function_ptrs))(*function_ptrs)
    index = int(
        _load_library().FlydslBridgeRegister(
            pointers, len(function_ptrs), int(compile_key)
        )
    )
    if index == -2:
        raise RuntimeError(
            f"FlyDSL bridge compile table is full ({max_compiles()} entries)"
        )
    if index < 0:
        raise RuntimeError(f"FlyDSL bridge registration failed with code {index}")
    _compile_names[index] = name or f"compile_{index}"
    return index


def dispatch_count(compile_index: int) -> int:
    return int(_load_library().FlydslBridgeDispatchCount(int(compile_index)))


@atexit.register
def _report_dispatch_counts() -> None:
    if os.environ.get("JAX_AITER_FLYDSL_REPORT", "0") != "1":
        return
    counts = {}
    for index, name in sorted(_compile_names.items()):
        count = dispatch_count(index)
        if count:
            counts[name] = count
    print(f"[jax-aiter] FlyDSL dispatch counts: {counts}", flush=True)


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
