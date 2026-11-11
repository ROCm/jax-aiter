# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Raise TransformerEngine's per-stream hipBLASLt workspace when
MXFP4_PROJ_TE_GEMM_WORKSPACE_MIB is set; a no-op otherwise.

hipBLASLt on gfx950 can return a Stream-K solution whose partial-tile workspace
exceeds TE's fixed 64 MiB (DeepSeek-V3 down-projection dgrad, 2049-2560 rows per
expert) and then fails that matmul with status 6. Loaded by Python's site module
when this directory is on PYTHONPATH; TE files are not modified.
"""

import importlib.abc
import os
import sys

_TARGET = "transformer_engine.jax.cpp_extensions.gemm"
_MIB = int(os.environ.get("MXFP4_PROJ_TE_GEMM_WORKSPACE_MIB", "0") or 0)


def _patch(module):
  default = module.get_cublas_workspace_size_bytes()
  module.get_cublas_workspace_size_bytes = lambda: _MIB << 20
  print(f"[te_workspace_override] TE GEMM workspace {default >> 20} MiB -> {_MIB} MiB per stream",
        flush=True)


class _Finder(importlib.abc.MetaPathFinder):

  def find_spec(self, fullname, path, target=None):
    if fullname != _TARGET:
      return None
    sys.meta_path.remove(self)
    try:
      import importlib.util  # pylint: disable=import-outside-toplevel
      spec = importlib.util.find_spec(fullname)
    finally:
      sys.meta_path.insert(0, self)
    if spec is None or spec.loader is None:
      return spec
    exec_module = spec.loader.exec_module

    def exec_and_patch(module):
      exec_module(module)
      _patch(module)

    spec.loader.exec_module = exec_and_patch
    return spec


if _MIB > 0:
  if _TARGET in sys.modules:
    _patch(sys.modules[_TARGET])
  else:
    sys.meta_path.insert(0, _Finder())
