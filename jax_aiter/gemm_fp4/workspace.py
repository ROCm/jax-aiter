# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""WeightWorkspace — cache of MXFP4-quantized weights keyed by ``cache_name``.

Inspired by TE's ``TransformerEngineBaseModule._fp8_workspaces`` dict. The
workspace is **not safe to cache across optimizer steps** in JAX: parameters
are re-read from the optimizer state each step, so the BF16 master weight
changes even though the Python identity does not. The workspace is meant to
amortize FP4 quantization within a single logical step (e.g. across the
micro-batches of a gradient-accumulation loop, or across multiple call sites
that share a weight).

Usage pattern::

    workspace = WeightWorkspace()
    wq = MXFP4Quantizer.for_weight()
    mxfp4_w = workspace.get_or_quantize(w_bf16, wq, cache_name="mlp_gate/weight")
    ...  # use mxfp4_w.rowwise_data / columnwise_data

Cache invalidation strategies:

- Call ``workspace.reset()`` at the top of a new optimizer step.
- Pass ``skip_update_flag=True`` to reuse a previously-cached tensor even if
  the BF16 weight has changed (matches TE's ``skip_update_flag``).
- Pass ``force_update=True`` to ignore the cache and re-quantize.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .mxfp4_tensor import Mxfp4Tensor
from .quantizer import MXFP4Quantizer


@dataclass
class _Entry:
    tensor: Mxfp4Tensor
    # Cheap identity for BF16 source: id() of the source jax.Array object. We
    # use id() rather than value-equality because JAX arrays are immutable;
    # same id => same logical tensor. When ``skip_update_flag`` is True we
    # ignore identity mismatch.
    source_id: int


@dataclass
class WeightWorkspace:
    """In-memory cache of MXFP4 quantized weights.

    Not thread-safe. Typical lifetime is one forward+backward pass.
    """

    _entries: Dict[str, _Entry] = field(default_factory=dict)

    def get_or_quantize(
        self,
        weight_bf16,
        quantizer: MXFP4Quantizer,
        *,
        cache_name: Optional[str] = None,
        skip_update_flag: bool = False,
        force_update: bool = False,
    ) -> Mxfp4Tensor:
        """Return a cached ``Mxfp4Tensor`` or quantize fresh.

        Args:
            weight_bf16: BF16 source tensor.
            quantizer:   MXFP4Quantizer configured for the weight role.
            cache_name:  Identifier for the cache slot. If ``None``, the cache
                         is bypassed (equivalent to a direct ``quantizer(x)``).
            skip_update_flag: If True, reuse the cached tensor even when the
                         source BF16 array has changed.
            force_update: If True, ignore any cached entry and re-quantize.
        """
        if cache_name is None:
            return quantizer.quantize(weight_bf16)

        if force_update:
            self._entries.pop(cache_name, None)

        entry = self._entries.get(cache_name)
        if entry is not None:
            if skip_update_flag or entry.source_id == id(weight_bf16):
                return entry.tensor

        tensor = quantizer.quantize(weight_bf16)
        self._entries[cache_name] = _Entry(tensor=tensor, source_id=id(weight_bf16))
        return tensor

    def evict(self, cache_name: str) -> None:
        """Drop one cache entry."""
        self._entries.pop(cache_name, None)

    def reset(self) -> None:
        """Drop every cache entry (e.g. at the start of a new step)."""
        self._entries.clear()

    def __contains__(self, cache_name: str) -> bool:
        return cache_name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def keys(self):
        return self._entries.keys()


# ---------------------------------------------------------------------------
# Process-global default workspace.
# ---------------------------------------------------------------------------
_DEFAULT_WORKSPACE: Optional[WeightWorkspace] = None


def default_workspace() -> WeightWorkspace:
    """Return the process-global default workspace (created lazily)."""
    global _DEFAULT_WORKSPACE
    if _DEFAULT_WORKSPACE is None:
        _DEFAULT_WORKSPACE = WeightWorkspace()
    return _DEFAULT_WORKSPACE


def reset_default_workspace() -> None:
    """Drop the process-global workspace (e.g. between training phases)."""
    global _DEFAULT_WORKSPACE
    if _DEFAULT_WORKSPACE is not None:
        _DEFAULT_WORKSPACE.reset()
