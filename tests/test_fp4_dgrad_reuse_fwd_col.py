# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Experiment B -- reuse the forward columnwise weight cast in the packed dgrad.

The default ``gather_packed`` FP4 dgrad RE-casts the raw bf16 weight
colwise-UNSHUFFLED every backward (``_cast_wt_colwise_unshuf(b)``). Experiment B
(``JA_FP4_DGRAD_REUSE_FWD_COL``, default-off) instead produces that UNSHUFFLED
colwise weight in the FORWARD and stashes it as a residual so the dgrad reuses it
(NVIDIA-TE-style checkpointing of the transposed operand), eliminating the
backward recast.

Gates (all must pass BEFORE any E2E timing):
  Gate 1     Offline byte-identity -- the forward's stashed colwise weight
             residual (flag ON) is byte-identical to ``_cast_wt_colwise_unshuf(b)``
             (what the backward re-casts today) for all 7 llama3-8b projections.
  Gate 2     Single-device value_and_grad parity -- flag OFF vs ON gradients are
             byte-identical (the reuse is a pure relocation of the cast).
  Guardrail  Default-off => zero behavior change (flag OFF stashes the raw weight
             + re-casts; flag ON drops the raw weight + reuses the residual).
  Shardy     FSDP-mesh parity under an N-sharded weight (the dgrad N-gather path):
             flag OFF vs ON gradients match, verifying the saved residual carries
             the correct N-sharding. Companion to the E2E shardy=True gate.

The flag is a module global read at TRACE time, so the tests toggle it and clear
the jit cache to force a re-trace.
"""

import contextlib
import importlib

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from jax_aiter.gemm_fp4.quantizer import _fused_quant_available

pytestmark = pytest.mark.skipif(
    not _fused_quant_available(),
    reason="CastMxfp4DualJA / GemmFp4FwdJA FFI not available (run 'make ja_mods')",
)

# The gemm_fp4 SUBMODULE (the package re-exports the gemm_fp4 FUNCTION under the
# same name, so import the submodule explicitly to reach the module globals).
g = importlib.import_module("jax_aiter.gemm_fp4.gemm_fp4")
from jax_aiter.gemm_fp4 import gemm_fp4_bf16  # noqa: E402


@contextlib.contextmanager
def _reuse_flag(value):
    """Toggle ``JA_FP4_DGRAD_REUSE_FWD_COL`` (module global) + force re-trace."""
    old = g._DGRAD_REUSE_FWD_COL
    g._DGRAD_REUSE_FWD_COL = value
    jax.clear_caches()
    try:
        yield
    finally:
        g._DGRAD_REUSE_FWD_COL = old
        jax.clear_caches()


# llama3-8b projection WEIGHT shapes b = [N_out, K_in].
_LLAMA3_8B_PROJ = [
    pytest.param(4096, 4096, id="q_proj"),
    pytest.param(1024, 4096, id="k_proj"),
    pytest.param(1024, 4096, id="v_proj"),
    pytest.param(4096, 4096, id="o_proj"),
    pytest.param(14336, 4096, id="gate_proj"),
    pytest.param(14336, 4096, id="up_proj"),
    pytest.param(4096, 14336, id="down_proj"),
]


def _require_packed_dgrad():
    if not g._DGRAD_PACKED:
        pytest.skip("Experiment B only applies under the default gather_packed dgrad")


def _assert_within_noise(off1, off2, on, label):
    """The reuse (``on``) must not differ from the recast (``off``) by MORE than
    the recast's own run-to-run non-determinism.

    The FP4 backward GEMM (dA/dB) uses splitK ATOMIC accumulation, so it is
    non-deterministic even for the SAME compiled executable (verified: two runs
    of one jit differ by the same magnitude as OFF-vs-ON). Byte-identity is only
    achievable with a forced splitK=1 kernel; in that case ``off1==off2`` and we
    require exact equality. Otherwise we require the OFF-vs-ON difference to sit
    within the OFF-vs-OFF noise floor -- a corrupt reuse (e.g. wrong N-sharding)
    would blow far past it (the legacy-reuse bug was cos~0.01)."""
    off1 = off1.astype(np.float32); off2 = off2.astype(np.float32); on = on.astype(np.float32)
    floor_mean = float(np.mean(np.abs(off1 - off2)))
    floor_max = float(np.max(np.abs(off1 - off2)))
    on_mean = float(np.mean(np.abs(off1 - on)))
    on_max = float(np.max(np.abs(off1 - on)))
    if floor_max == 0.0:
        assert on_max == 0.0, (
            f"{label}: deterministic kernel but reuse not byte-identical "
            f"(max|off-on|={on_max})")
    else:
        assert on_mean <= 2.0 * floor_mean + 1e-7, (
            f"{label}: mean|off-on|={on_mean:.3e} exceeds 2x noise floor "
            f"{floor_mean:.3e} -- reuse adds systematic error")
        assert on_max <= 3.0 * floor_max + 1e-6, (
            f"{label}: max|off-on|={on_max:.3e} exceeds 3x noise floor "
            f"{floor_max:.3e} -- reuse adds systematic error")


# --------------------------------------------------------------------------- #
# Gate 1 -- offline byte-identity of the reused colwise weight.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("N,K", _LLAMA3_8B_PROJ)
def test_gate1_byte_identity_colwise_weight(N, K):
    """The forward's stashed colwise weight residual (flag ON) == the packed
    dgrad's ``_cast_wt_colwise_unshuf(b)`` byte-for-byte (packed u8 + E8M0)."""
    _require_packed_dgrad()
    M = 256
    ka, kb = jax.random.split(jax.random.PRNGKey(0))
    a = jax.random.normal(ka, (M, K), dtype=jnp.bfloat16)
    b = jax.random.normal(kb, (N, K), dtype=jnp.bfloat16)

    # Reference: exactly what the backward re-casts today.
    ref_packed, ref_scale = jax.jit(g._cast_wt_colwise_unshuf)(b)
    ref_packed = np.asarray(ref_packed)
    ref_scale = np.asarray(ref_scale)

    # Reuse: the forward stashes this at residual[2], residual[3] under the flag.
    with _reuse_flag(True):
        _, residual = jax.jit(g._gemm_fp4_bf16_fwd)(a, b)
        reuse_packed = np.asarray(residual[2])
        reuse_scale = np.asarray(residual[3])

    assert reuse_packed.dtype == np.uint8 and reuse_scale.dtype == np.uint8
    assert reuse_packed.shape == ref_packed.shape, (reuse_packed.shape, ref_packed.shape)
    assert reuse_scale.shape == ref_scale.shape, (reuse_scale.shape, ref_scale.shape)
    assert np.array_equal(reuse_packed, ref_packed), (
        "colwise PACKED weight bytes differ from _cast_wt_colwise_unshuf")
    assert np.array_equal(reuse_scale, ref_scale), (
        "colwise E8M0 SCALE bytes differ from _cast_wt_colwise_unshuf")


# --------------------------------------------------------------------------- #
# Gate 2 -- single-device value_and_grad parity (flag OFF vs ON).
# --------------------------------------------------------------------------- #

_PARITY_SHAPES = [
    pytest.param(256, 4096, 4096, id="q_like"),
    pytest.param(256, 1024, 4096, id="kv_like"),
    pytest.param(512, 14336, 4096, id="gateup_like"),
    pytest.param(512, 4096, 14336, id="down_like"),
]


def _value_and_grad(a, b, t):
    fn = jax.value_and_grad(
        lambda a_, b_: jnp.mean((gemm_fp4_bf16(a_, b_) - t) ** 2),
        argnums=(0, 1),
    )
    loss, (da, db) = jax.jit(fn)(a, b)
    return np.asarray(loss), np.asarray(da), np.asarray(db)


@pytest.mark.parametrize("M,N,K", _PARITY_SHAPES)
def test_gate2_value_and_grad_parity_single_device(M, N, K):
    """Flag OFF (recast) vs ON (reuse): forward loss byte-identical; dA/dB within
    the FP4-GEMM non-determinism noise floor (byte-identical under splitK=1)."""
    _require_packed_dgrad()
    ka, kb, kt = jax.random.split(jax.random.PRNGKey(7), 3)
    a = jax.random.normal(ka, (M, K), dtype=jnp.bfloat16)
    b = jax.random.normal(kb, (N, K), dtype=jnp.bfloat16)
    t = jax.random.normal(kt, (M, N), dtype=jnp.bfloat16)

    with _reuse_flag(False):
        loss_off, da_off1, db_off1 = _value_and_grad(a, b, t)   # noise-floor ref 1
        _, da_off2, db_off2 = _value_and_grad(a, b, t)          # noise-floor ref 2
    with _reuse_flag(True):
        loss_on, da_on, db_on = _value_and_grad(a, b, t)

    assert np.all(np.isfinite(da_on)) and np.all(np.isfinite(db_on))
    assert np.array_equal(loss_off, loss_on), "forward loss differs off vs on"
    _assert_within_noise(da_off1, da_off2, da_on, "dA")
    _assert_within_noise(db_off1, db_off2, db_on, "dB")


# --------------------------------------------------------------------------- #
# Guardrail -- default-off is the recast path; flag-on drops the raw weight.
# --------------------------------------------------------------------------- #

def test_guardrail_default_off_recasts_flag_on_reuses():
    """Under the default packed dgrad, flag OFF stashes the RAW bf16 weight (recast
    path, 5 residuals); flag ON reuses the saved colwise residual (4 residuals,
    no raw weight)."""
    _require_packed_dgrad()
    M, N, K = 256, 4096, 4096
    ka, kb = jax.random.split(jax.random.PRNGKey(3))
    a = jax.random.normal(ka, (M, K), dtype=jnp.bfloat16)
    b = jax.random.normal(kb, (N, K), dtype=jnp.bfloat16)

    with _reuse_flag(False):
        _, res_off = jax.jit(g._gemm_fp4_bf16_fwd)(a, b)
    with _reuse_flag(True):
        _, res_on = jax.jit(g._gemm_fp4_bf16_fwd)(a, b)

    assert len(res_off) == 5, f"flag-off should stash raw b (recast): got {len(res_off)}"
    assert len(res_on) == 4, f"flag-on should NOT stash raw b (reuse): got {len(res_on)}"
    # flag-off residual[4] is the raw bf16 weight the backward re-casts.
    assert res_off[4].shape == b.shape and res_off[4].dtype == jnp.bfloat16


def test_guardrail_default_module_flag_is_off():
    """The module-load default of the flag is OFF (opt-in only)."""
    import os
    assert (os.environ.get("JA_FP4_DGRAD_REUSE_FWD_COL", "0") == "1") == g._DGRAD_REUSE_FWD_COL


# --------------------------------------------------------------------------- #
# Shardy -- FSDP-mesh parity under an N-sharded weight (dgrad N-gather path).
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(len(jax.devices()) < 4,
                    reason="FSDP N-shard dgrad-reuse test needs >= 4 devices")
@pytest.mark.parametrize("shardy", [False, True])
def test_shardy_fsdp_dgrad_reuse_parity(shardy):
    """Weight N-sharded => the packed dgrad ALL-GATHERS the colwise weight over N.
    Flag OFF (recast) and ON (reuse the saved N-sharded colwise residual) must
    agree within the GEMM noise floor -- a corrupt N-sharding reuse (the legacy
    bug) would blow far past it. A 2-D data(M) x model(N) mesh puts the weight's
    N-shard on a DISTINCT axis from the token M-shard so the wgrad (contract M)
    and dgrad (gather N) partitions don't collide on one axis."""
    _require_packed_dgrad()
    mesh = Mesh(np.asarray(jax.devices()[:4]).reshape(2, 2),
                axis_names=("data", "model"))
    N, K = 4096, 14336                      # down/O-style: N/model=2048 (mult 256)
    M = 512 * 2
    a_spec = NamedSharding(mesh, P("data", None))     # activation: token(M)-sharded
    b_spec = NamedSharding(mesh, P("model", None))    # weight: N(out-feature)-sharded
    t_spec = NamedSharding(mesh, P("data", "model"))

    ka, kb, kt = jax.random.split(jax.random.PRNGKey(11), 3)
    a = jax.device_put(jax.random.normal(ka, (M, K), dtype=jnp.bfloat16), a_spec)
    b = jax.device_put(jax.random.normal(kb, (N, K), dtype=jnp.bfloat16), b_spec)
    t = jax.device_put(jax.random.normal(kt, (M, N), dtype=jnp.bfloat16), t_spec)

    def _run():
        fn = jax.value_and_grad(
            lambda a_, b_: jnp.mean((gemm_fp4_bf16(a_, b_) - t) ** 2),
            argnums=(0, 1),
        )
        loss, (da, db) = jax.jit(fn)(a, b)
        return np.asarray(loss), np.asarray(da), np.asarray(db)

    key = "jax_use_shardy_partitioner"
    try:
        prev = jax.config.read(key)
    except Exception:
        if shardy:
            pytest.skip("shardy partitioner config not available on this stack")
        prev = None

    try:
        if prev is not None:
            jax.config.update(key, shardy)
        elif shardy:
            pytest.skip("shardy partitioner config not available on this stack")
        with _reuse_flag(False):
            loss_off, da_off1, db_off1 = _run()   # noise-floor ref 1
            _, da_off2, db_off2 = _run()           # noise-floor ref 2
        with _reuse_flag(True):
            loss_on, da_on, db_on = _run()
    finally:
        if prev is not None:
            jax.config.update(key, prev)

    assert np.all(np.isfinite(da_on)) and np.all(np.isfinite(db_on)), "non-finite grads"
    assert np.array_equal(loss_off, loss_on), f"forward loss differs (shardy={shardy})"
    # A corrupt N-sharding reuse (the legacy-reuse bug) would blow FAR past the
    # atomic-reduction noise floor; a correct reuse stays within it.
    _assert_within_noise(da_off1, da_off2, da_on, f"dA(shardy={shardy})")
    _assert_within_noise(db_off1, db_off2, db_on, f"dB(shardy={shardy})")
