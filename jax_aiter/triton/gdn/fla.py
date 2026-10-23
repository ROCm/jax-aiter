# SPDX-License-Identifier: Apache-2.0
"""Load flash-linear-attention (FLA) Triton kernels without its model zoo."""

from __future__ import annotations

import os
import re
import sys
import threading
import types
from pathlib import Path

from ..compile import prepare_aiter_triton

_LOCK = threading.Lock()
_KERNELS: types.SimpleNamespace | None = None


def _find_fla_python() -> Path:
    env = os.environ.get("JA_FLA_PYTHON")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        for candidate in (
            parent / "jax-aiter" / "third_party" / "flash-linear-attention",
            parent / "third_party" / "flash-linear-attention",
        ):
            if (candidate / "fla" / "ops" / "gated_delta_rule").is_dir():
                return candidate
    raise FileNotFoundError(
        "flash-linear-attention sources not found. Set JA_FLA_PYTHON to the "
        "directory that contains the `fla` package (the "
        "third_party/flash-linear-attention checkout)."
    )


def _install_package_stub(root: Path) -> None:
    # fla/__init__.py imports the Transformers model zoo; only fla.ops is used.
    if "fla" in sys.modules:
        return
    init = (root / "fla" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", init)
    package = types.ModuleType("fla")
    package.__path__ = [str(root / "fla")]
    package.__version__ = match.group(1) if match else "0.0.0"
    sys.modules["fla"] = package


def fla_kernels() -> types.SimpleNamespace:
    """Return the FLA Gated DeltaNet Triton kernels used by JAX-AITER."""
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    with _LOCK:
        if _KERNELS is not None:
            return _KERNELS
        prepare_aiter_triton()
        _install_package_stub(_find_fla_python())

        from fla.ops.common import chunk_delta_h, chunk_o
        from fla.ops.gated_delta_rule import chunk_fwd, wy_fast

        _KERNELS = types.SimpleNamespace(
            kkt_solve=chunk_fwd.chunk_gated_delta_rule_fwd_kkt_solve_kernel,
            recompute_w_u=wy_fast.recompute_w_u_fwd_kernel,
            wy_bwd=wy_fast.prepare_wy_repr_bwd_kernel,
            fwd_h=chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
            bwd_dhu=chunk_delta_h.chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64,
            fwd_o=chunk_o.chunk_fwd_kernel_o,
            bwd_dv_local=chunk_o.chunk_bwd_kernel_dv_local,
            bwd_dqkwg=chunk_o.chunk_bwd_kernel_dqkwg,
        )
        return _KERNELS
