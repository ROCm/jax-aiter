# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Unit tests for WeightWorkspace cache."""

import pytest
import jax
import jax.numpy as jnp

from jax_aiter.gemm_fp4 import (
    MXFP4Quantizer,
    WeightWorkspace,
    default_workspace,
    reset_default_workspace,
)
from jax_aiter.gemm_fp4.quantizer import _fused_quant_available


def _make_weight(M=128, K=256, seed=0):
    key = jax.random.PRNGKey(seed)
    return jax.random.normal(key, (M, K), dtype=jnp.bfloat16)


# ---------------------------------------------------------------------------
# JAX-fallback-compatible tests (rowwise-only, no fused kernel needed)
# ---------------------------------------------------------------------------

def _rowwise_quantizer():
    return MXFP4Quantizer(
        rowwise=True, columnwise=False,
        shuffle_B_matrix_for_aiter=True,
        use_fused_kernel=False,
    )


def test_bypass_cache_when_cache_name_none():
    ws = WeightWorkspace()
    q = _rowwise_quantizer()
    w = _make_weight()
    t = ws.get_or_quantize(w, q, cache_name=None)
    assert t.has_rowwise
    assert len(ws) == 0


def test_cache_hit_reuses_same_tensor():
    ws = WeightWorkspace()
    q = _rowwise_quantizer()
    w = _make_weight()
    t1 = ws.get_or_quantize(w, q, cache_name="mlp_gate/weight")
    t2 = ws.get_or_quantize(w, q, cache_name="mlp_gate/weight")
    # Same Mxfp4Tensor object (literally cached, not re-quantized).
    assert t1 is t2


def test_cache_miss_on_different_source_triggers_requantize():
    ws = WeightWorkspace()
    q = _rowwise_quantizer()
    w1 = _make_weight(seed=0)
    w2 = _make_weight(seed=1)
    t1 = ws.get_or_quantize(w1, q, cache_name="mlp/w")
    t2 = ws.get_or_quantize(w2, q, cache_name="mlp/w")
    assert t1 is not t2  # fresh tensor
    assert "mlp/w" in ws
    assert len(ws) == 1  # cache was overwritten


def test_skip_update_flag_reuses_even_when_source_changes():
    ws = WeightWorkspace()
    q = _rowwise_quantizer()
    w1 = _make_weight(seed=0)
    w2 = _make_weight(seed=1)
    t1 = ws.get_or_quantize(w1, q, cache_name="mlp/w")
    t2 = ws.get_or_quantize(w2, q, cache_name="mlp/w", skip_update_flag=True)
    assert t1 is t2  # stale cache reused
    # Cache still points at the old tensor.
    assert ws._entries["mlp/w"].tensor is t1


def test_force_update_bypasses_cache():
    ws = WeightWorkspace()
    q = _rowwise_quantizer()
    w = _make_weight()
    t1 = ws.get_or_quantize(w, q, cache_name="mlp/w")
    t2 = ws.get_or_quantize(w, q, cache_name="mlp/w", force_update=True)
    # New Mxfp4Tensor object.
    assert t1 is not t2


def test_reset_drops_all_entries():
    ws = WeightWorkspace()
    q = _rowwise_quantizer()
    w = _make_weight()
    ws.get_or_quantize(w, q, cache_name="a")
    ws.get_or_quantize(w, q, cache_name="b")
    assert len(ws) == 2
    ws.reset()
    assert len(ws) == 0


def test_evict_single_entry():
    ws = WeightWorkspace()
    q = _rowwise_quantizer()
    w = _make_weight()
    ws.get_or_quantize(w, q, cache_name="a")
    ws.get_or_quantize(w, q, cache_name="b")
    ws.evict("a")
    assert "a" not in ws
    assert "b" in ws


def test_evict_missing_key_is_noop():
    ws = WeightWorkspace()
    # Must not raise.
    ws.evict("nonexistent")


def test_default_workspace_is_singleton():
    ws1 = default_workspace()
    ws2 = default_workspace()
    assert ws1 is ws2


def test_reset_default_workspace():
    ws = default_workspace()
    q = _rowwise_quantizer()
    w = _make_weight()
    ws.get_or_quantize(w, q, cache_name="persistent")
    assert len(ws) >= 1
    reset_default_workspace()
    assert len(default_workspace()) == 0


# ---------------------------------------------------------------------------
# Fused-kernel dual cast
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _fused_quant_available(),
    reason="CastMxfp4DualJA FFI not available (build with 'make ja_mods')")
def test_workspace_caches_dual_tensor():
    ws = WeightWorkspace()
    q = MXFP4Quantizer.for_weight()
    w = _make_weight(M=128, K=256)
    t1 = ws.get_or_quantize(w, q, cache_name="weight_dual")
    t2 = ws.get_or_quantize(w, q, cache_name="weight_dual")
    assert t1 is t2
    assert t1.has_rowwise
    assert t1.has_columnwise
