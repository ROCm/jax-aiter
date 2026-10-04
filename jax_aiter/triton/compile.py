# SPDX-License-Identifier: Apache-2.0
"""Compile AITER Triton JIT kernels to HSACO without GPU PyTorch."""

from __future__ import annotations

import hashlib
import os
import sys
import threading
from pathlib import Path
from typing import Any

_IMPORT_LOCK = threading.Lock()
_AITER_READY = False
_TRITON_VERSION = "3.8.0"


def _project_deps_triton() -> Path | None:
    env = os.environ.get("JA_TRITON_PATH")
    if env:
        candidate = Path(env).expanduser().resolve()
        if (candidate / "triton").is_dir() or (candidate / "__init__.py").is_file():
            return candidate
    here = Path(__file__).resolve()
    checkout = here.parents[2]
    candidate = checkout.parent / ".deps" / "triton-rocm"
    if (candidate / "triton").is_dir():
        return candidate
    return None


def _find_aiter_python() -> Path:
    env = os.environ.get("JA_AITER_PYTHON")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        for candidate in (
            parent / "jax-aiter" / "third_party" / "aiter",
            parent / "third_party" / "aiter",
        ):
            if (candidate / "aiter" / "ops" / "triton").is_dir():
                return candidate
    raise FileNotFoundError(
        "AITER Python sources not found. Set JA_AITER_PYTHON to the "
        "directory that contains the `aiter` package (the third_party/aiter "
        "checkout)."
    )


def _import_triton() -> Any:
    # jax-aiter loads the AITER MHA libraries RTLD_GLOBAL, which exposes ROCm's
    # shared libLLVM. libtriton bundles a different LLVM and crashes in its
    # static initializers if those symbols interpose its own, so bind it deep.
    if "triton._C.libtriton" in sys.modules:
        import triton

        return triton
    flags = sys.getdlopenflags()
    sys.setdlopenflags(flags | os.RTLD_DEEPBIND)
    try:
        import triton
    finally:
        sys.setdlopenflags(flags)
    return triton


def _ensure_triton() -> Any:
    try:
        triton = _import_triton()
    except ImportError:
        root = _project_deps_triton()
        if root is None:
            raise
        sys.path.insert(0, str(root))
        triton = _import_triton()
    if not str(triton.__version__).startswith(_TRITON_VERSION):
        raise RuntimeError(
            f"jax-aiter Triton kernels require Triton {_TRITON_VERSION}, "
            f"got {triton.__version__}"
        )
    return triton


def prepare_aiter_triton() -> None:
    """Initialize JAX HIP, then import AITER Triton kernel modules.

    AITER's Triton files import Torch for dtype helpers. Initialize JAX first
    so a GPU Torch build cannot steal the HIP context. ``AITER_TRITON_ONLY``
    skips AITER's CK/HIP import tree.
    """
    global _AITER_READY
    if _AITER_READY:
        return
    with _IMPORT_LOCK:
        if _AITER_READY:
            return

        import jax

        devices = jax.devices("gpu")
        if not devices:
            raise RuntimeError("AITER Triton kernels require a ROCm GPU")

        os.environ.setdefault("AITER_TRITON_ONLY", "1")
        _ensure_triton()

        path = _find_aiter_python()
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

        import jaxlib.gpu_triton as gpu_triton

        sys.modules.setdefault("jax._src.lib.gpu_triton", gpu_triton)
        _AITER_READY = True


def compile_hsaco(
    fn,
    *,
    runtime_types: dict[str, str],
    constexprs: dict[str, Any],
    options: dict[str, Any] | None = None,
):
    """Compile one ``@triton.jit`` kernel to HSACO for gfx950."""
    prepare_aiter_triton()
    triton = _ensure_triton()
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource, compile

    constexpr_idx = set(fn.constexprs)
    signature: dict[str, str] = {}
    for index, name in enumerate(fn.arg_names):
        if name in constexprs or index in constexpr_idx:
            signature[name] = "constexpr"
            if name not in constexprs:
                raise ValueError(f"Missing constexpr {name} for {fn}")
        else:
            if name not in runtime_types:
                raise ValueError(f"Missing runtime type for {name} in {fn}")
            signature[name] = runtime_types[name]

    unknown = set(constexprs) - set(fn.arg_names)
    if unknown:
        raise ValueError(f"Unknown constexprs for {fn}: {sorted(unknown)}")

    source = ASTSource(fn, signature, constexprs)
    kernel = compile(
        source,
        target=GPUTarget("hip", "gfx950", 64),
        options=options or {},
    )
    if kernel.metadata.triton_version.split("+")[0] != _TRITON_VERSION:
        raise RuntimeError(
            "Compiled kernel Triton version "
            f"{kernel.metadata.triton_version} != {_TRITON_VERSION}"
        )
    if kernel.metadata.num_ctas != 1:
        raise RuntimeError("v1 Triton trampoline requires num_ctas==1")
    if kernel.metadata.global_scratch_size or kernel.metadata.profile_scratch_size:
        raise RuntimeError(
            "Kernel requested Triton scratch buffers; v1 trampoline passes null"
        )
    if kernel.metadata.launch_cooperative_grid:
        raise RuntimeError("Cooperative grid launches are not supported")
    hsaco = kernel.asm.get("hsaco") or kernel.kernel
    if not hsaco:
        raise RuntimeError(f"Compilation produced no HSACO for {fn}")
    return kernel


def compile_key(hsaco: bytes, name: str) -> int:
    digest = hashlib.sha256(name.encode("utf-8") + b"\0" + hsaco).digest()
    return int.from_bytes(digest[:8], "little", signed=False) & ((1 << 63) - 1)


_ABI = {
    "ptr": 0,
    "i32": 1,
    "i64": 2,
    "fp32": 3,
    "f32": 3,
    "fp64": 4,
    "f64": 4,
}


def abi_kind(triton_type: str) -> int:
    if triton_type.startswith("*"):
        return _ABI["ptr"]
    kind = _ABI.get(triton_type)
    if kind is None:
        raise ValueError(f"Unsupported Triton ABI type {triton_type}")
    return kind
