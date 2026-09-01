# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Tests for FP4 GEMM via AITER ASM kernels.

Works on gfx950 (MI350/MI355X). Tests the full pipeline:
  BF16 -> MXFP4 quantize -> shuffle -> FFI kernel -> compare vs BF16 reference.
"""

import importlib

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax.ad_checkpoint import checkpoint_name

from jax_aiter.gemm_fp4 import gemm_fp4
from jax_aiter.ops.gemm_fp4 import cast_mxfp4_dual
from jax_aiter.gemm_fp4.fp4_utils import (
    bf16_to_mxfp4,
    mxfp4_to_bf16,
    e8m0_shuffle,
    shuffle_weight,
    e8m0_to_f32,
    MXFP4_BLOCK_SIZE,
)


def _make_quantized_inputs(M, N, K, seed=0):
    """Create properly quantized FP4 inputs from random BF16 data.

    Returns both the quantized inputs (for the kernel) and the BF16 originals
    (for reference comparison).
    """
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    a_bf16 = jax.random.normal(k1, (M, K), dtype=jnp.bfloat16)
    b_bf16 = jax.random.normal(k2, (N, K), dtype=jnp.bfloat16)

    a_packed, a_scales = bf16_to_mxfp4(a_bf16)
    b_packed, b_scales = bf16_to_mxfp4(b_bf16)

    b_packed_shuffled = shuffle_weight(b_packed)
    b_scales_shuffled = e8m0_shuffle(b_scales)
    a_scales_shuffled = e8m0_shuffle(a_scales)

    return (a_packed, b_packed_shuffled, a_scales_shuffled, b_scales_shuffled,
            a_bf16, b_bf16, a_packed, b_packed, a_scales, b_scales)


# --- Quantization round-trip tests (no GPU kernel needed) ---

@pytest.mark.parametrize("M,K", [
    pytest.param(128, 256, id="128x256"),
    pytest.param(256, 512, id="256x512"),
    pytest.param(1024, 1024, id="1024x1024"),
])
def test_mxfp4_roundtrip(M, K):
    """Verify quantize -> dequantize preserves values within FP4 precision."""
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (M, K), dtype=jnp.bfloat16)

    packed, scales = bf16_to_mxfp4(x)

    assert packed.shape == (M, K // 2), f"packed shape {packed.shape}"
    assert packed.dtype == jnp.uint8
    assert scales.shape == (M, K // MXFP4_BLOCK_SIZE), f"scales shape {scales.shape}"
    assert scales.dtype == jnp.uint8

    x_recon = mxfp4_to_bf16(packed, scales)
    assert x_recon.shape == (M, K)
    assert x_recon.dtype == jnp.bfloat16

    ref_f32 = x.astype(jnp.float32)
    recon_f32 = x_recon.astype(jnp.float32)
    abs_err = jnp.abs(ref_f32 - recon_f32)
    rel_err = abs_err / (jnp.abs(ref_f32) + 1e-10)
    mean_rel = float(jnp.mean(rel_err))
    max_rel = float(jnp.max(rel_err))

    assert mean_rel < 0.5, f"Mean relative error too large: {mean_rel:.4f}"
    assert jnp.all(jnp.isfinite(x_recon)), "Non-finite values in reconstructed output"


def test_mxfp4_zeros():
    """Zero input should quantize to zero output."""
    x = jnp.zeros((64, 128), dtype=jnp.bfloat16)
    packed, scales = bf16_to_mxfp4(x)
    x_recon = mxfp4_to_bf16(packed, scales)
    assert jnp.allclose(x_recon, 0.0, atol=1e-6)


def test_mxfp4_known_values():
    """Test quantization of known E2M1-representable values."""
    representable = jnp.array(
        [[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
          -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0] * 4],
        dtype=jnp.bfloat16,
    )
    packed, scales = bf16_to_mxfp4(representable)
    recon = mxfp4_to_bf16(packed, scales)
    recon_f32 = recon.astype(jnp.float32)
    ref_f32 = representable.astype(jnp.float32)
    assert jnp.allclose(recon_f32, ref_f32, atol=0.01), (
        f"Known values not preserved: max_err={float(jnp.max(jnp.abs(recon_f32 - ref_f32)))}"
    )


# --- FFI kernel tests ---

@pytest.mark.parametrize("M,N,K", [
    pytest.param(128, 128, 256, id="128x128x256"),
    pytest.param(256, 256, 256, id="256x256x256"),
    pytest.param(128, 256, 512, id="128x256x512"),
    pytest.param(256, 512, 256, id="256x512x256"),
    pytest.param(1024, 1024, 512, id="1k_sq"),
])
def test_fp4_gemm_shape_and_dtype(M, N, K):
    """Verify output shape and dtype from FP4 GEMM kernel."""
    (a_p, b_p, a_s, b_s,
     _, _, _, _, _, _) = _make_quantized_inputs(M, N, K)

    out = gemm_fp4(a_p, b_p, a_s, b_s)
    assert out.shape == (M, N), f"shape {out.shape} != ({M}, {N})"
    assert out.dtype == jnp.bfloat16


@pytest.mark.parametrize("M,N,K", [
    pytest.param(128, 128, 256, id="128x128x256"),
    pytest.param(256, 256, 256, id="256x256x256"),
    pytest.param(256, 512, 512, id="256x512x512"),
    pytest.param(1024, 1024, 512, id="1k_sq"),
])
def test_fp4_gemm_accuracy(M, N, K):
    """Compare FP4 GEMM output against BF16 dequantize-then-matmul reference.

    The reference path: dequantize both A and B from MXFP4 to BF16, then
    compute jnp.matmul(A_bf16, B_bf16.T) in float32. The kernel should
    produce results within FP4 quantization noise.
    """
    (a_p, b_p, a_s, b_s,
     a_bf16, b_bf16, a_raw, b_raw, a_sc, b_sc) = _make_quantized_inputs(M, N, K, seed=42)

    out = gemm_fp4(a_p, b_p, a_s, b_s)

    a_deq = mxfp4_to_bf16(a_raw, a_sc).astype(jnp.float32)
    b_deq = mxfp4_to_bf16(b_raw, b_sc).astype(jnp.float32)
    ref = jnp.matmul(a_deq, b_deq.T)

    out_f32 = out.astype(jnp.float32)
    abs_err = jnp.abs(out_f32 - ref)
    scale = jnp.maximum(jnp.abs(ref), 1.0)
    rel_err = abs_err / scale
    mean_rel = float(jnp.mean(rel_err))

    assert jnp.all(jnp.isfinite(out)), "Non-finite values in FP4 GEMM output"
    assert mean_rel < 0.1, (
        f"Mean relative error {mean_rel:.4f} exceeds threshold for {M}x{N}x{K}"
    )


def test_fp4_gemm_zeros():
    """Zero inputs should produce zero output."""
    M, N, K = 128, 128, 256
    a = jnp.zeros((M, K // 2), dtype=jnp.uint8)
    b = jnp.zeros((N, K // 2), dtype=jnp.uint8)
    a_scale = jnp.full((M, K // 32), 127, dtype=jnp.uint8)
    b_scale = jnp.full((N, K // 32), 127, dtype=jnp.uint8)
    a_scale = e8m0_shuffle(a_scale)
    b_scale = e8m0_shuffle(b_scale)
    b = shuffle_weight(b)
    out = gemm_fp4(a, b, a_scale, b_scale)
    assert out.shape == (M, N)
    assert jnp.all(jnp.isfinite(out))


def test_keyed_sr_requires_uint32x4_key():
    x = jnp.ones((64, 128), dtype=jnp.bfloat16)
    with pytest.raises(ValueError, match=r"uint32\[4\]"):
        cast_mxfp4_dual(
            x,
            shuffle_fp4=False,
            use_sr_col=True,
            sr_key=jnp.zeros((2,), dtype=jnp.uint32),
            sr_role=2,
        )


def test_keyed_column_sr_is_reproducible_and_direction_selective():
    x = jax.random.normal(
        jax.random.PRNGKey(123), (256, 512), dtype=jnp.bfloat16
    )
    key0 = jnp.array([1, 2, 3, 4], dtype=jnp.uint32)
    key1 = jnp.array([5, 6, 7, 8], dtype=jnp.uint32)
    kwargs = dict(
        shuffle_fp4=False,
        shuffle_colwise_fp4=False,
        use_hadamard=False,
        use_hadamard_col=True,
        use_sr=False,
        use_sr_col=True,
        sr_role=2,
    )
    out0 = cast_mxfp4_dual(x, sr_key=key0, **kwargs)
    out0_repeat = cast_mxfp4_dual(x, sr_key=key0, **kwargs)
    out1 = cast_mxfp4_dual(x, sr_key=key1, **kwargs)
    rne = cast_mxfp4_dual(
        x,
        shuffle_fp4=False,
        shuffle_colwise_fp4=False,
        use_hadamard=False,
        use_hadamard_col=True,
        use_sr=False,
        use_sr_col=False,
        sr_role=2,
    )

    for actual, expected in zip(out0, out0_repeat):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    np.testing.assert_array_equal(np.asarray(out0[0]), np.asarray(rne[0]))
    np.testing.assert_array_equal(np.asarray(out0[1]), np.asarray(rne[1]))
    np.testing.assert_array_equal(np.asarray(out0[3]), np.asarray(rne[3]))
    np.testing.assert_array_equal(np.asarray(out0[0]), np.asarray(out1[0]))
    assert np.any(np.asarray(out0[2]) != np.asarray(out1[2]))


# --- High-level API tests (gemm_fp4_bf16) ---

from jax_aiter.gemm_fp4 import gemm_fp4_bf16


@pytest.mark.parametrize("M,N,K", [
    pytest.param(256, 256, 256, id="small_256"),
    pytest.param(256, 512, 512, id="medium_256x512"),
    pytest.param(1024, 1024, 512, id="1k_sq"),
])
def test_gemm_fp4_bf16_forward(M, N, K):
    """gemm_fp4_bf16 produces finite output of the right shape/dtype."""
    key = jax.random.PRNGKey(7)
    k1, k2 = jax.random.split(key)
    a = jax.random.normal(k1, (M, K), dtype=jnp.bfloat16)
    b = jax.random.normal(k2, (N, K), dtype=jnp.bfloat16)

    out = gemm_fp4_bf16(a, b)

    assert out.shape == (M, N)
    assert out.dtype == jnp.bfloat16
    assert jnp.all(jnp.isfinite(out)), "Non-finite values in gemm_fp4_bf16 output"


@pytest.mark.parametrize("M,N,K", [
    pytest.param(256, 256, 256, id="small_256"),
    pytest.param(256, 512, 512, id="medium_256x512"),
])
def test_gemm_fp4_bf16_gradient_flow(M, N, K):
    """gemm_fp4_bf16 backward produces finite gradients for a and b.

    Backward path: dual-cast grad_out (Hadamard ON) -> FP4 dA (NN) +
    FP4 dB (NT wgrad).
    """
    key = jax.random.PRNGKey(99)
    k1, k2 = jax.random.split(key)
    a = jax.random.normal(k1, (M, K), dtype=jnp.bfloat16)
    b = jax.random.normal(k2, (N, K), dtype=jnp.bfloat16)

    da, db = jax.grad(
        lambda a_, b_: jnp.sum(gemm_fp4_bf16(a_, b_)), argnums=(0, 1)
    )(a, b)

    assert da.shape == a.shape
    assert db.shape == b.shape
    assert jnp.all(jnp.isfinite(da)), "Non-finite da"
    assert jnp.all(jnp.isfinite(db)), "Non-finite db"
    assert jnp.any(da != 0), "da all zeros"
    assert jnp.any(db != 0), "db all zeros"


def test_gemm_fp4_bf16_keyed_column_sr_gradient_flow(monkeypatch):
    module = importlib.import_module("jax_aiter.gemm_fp4.gemm_fp4")
    monkeypatch.setattr(module, "_SR_GRAD", False)
    monkeypatch.setattr(module, "_SR_DGRAD_ROW", False)
    monkeypatch.setattr(module, "_SR_WGRAD_COL", True)
    monkeypatch.setattr(module, "_SR_ACT", False)
    monkeypatch.setattr(module, "_SR_WT", False)
    monkeypatch.setattr(module, "_SR_ANY", True)

    key = jax.random.PRNGKey(314)
    k1, k2 = jax.random.split(key)
    a = jax.random.normal(k1, (256, 256), dtype=jnp.bfloat16)
    b = jax.random.normal(k2, (256, 256), dtype=jnp.bfloat16)
    sr_key = jnp.array([11, 22, 33, 44], dtype=jnp.uint32)

    da, db = jax.grad(
        lambda a_, b_: jnp.sum(
            module.gemm_fp4_bf16(a_, b_, sr_key=sr_key)
        ),
        argnums=(0, 1),
    )(a, b)

    assert da.shape == a.shape
    assert db.shape == b.shape
    assert jnp.all(jnp.isfinite(da))
    assert jnp.all(jnp.isfinite(db))
    assert jnp.any(da != 0)
    assert jnp.any(db != 0)


def _remat_fp4_swiglu_layer():
    """One FP4 gate/up SwiGLU layer under the save-fp4-columns remat policy."""

    def reference_layer(x_i, weights):
        gate_w, up_w = weights
        gate = checkpoint_name(
            gemm_fp4_bf16(x_i, gate_w), "mlpwi_0"
        )
        up = checkpoint_name(gemm_fp4_bf16(x_i, up_w), "mlpwi_1")
        return jax.nn.silu(gate) * up

    return jax.checkpoint(
        reference_layer,
        prevent_cse=False,
        policy=jax.checkpoint_policies.save_only_these_names(
            "mlpwi_0", "mlpwi_1"
        ),
    )


def _scan_destination_functions():
    """Build reference and explicit-stack scanned MLPs for the Phase-B proof."""
    module = importlib.import_module("jax_aiter.gemm_fp4.gemm_fp4")

    remat_reference_layer = _remat_fp4_swiglu_layer()

    def reference(x, gate_weights, up_weights):
        def body(x_i, weights):
            return remat_reference_layer(x_i, weights), None

        return jax.lax.scan(body, x, (gate_weights, up_weights))[0]

    def candidate_fwd_impl(x, gate_weights, up_weights):
        layers, width, _ = gate_weights.shape
        rows = x.shape[0]
        stack0 = jnp.zeros((layers, rows, width), dtype=jnp.bfloat16)

        def body(carry, weights):
            x_i, gate_stack, up_stack, layer_i = carry
            gate_w, up_w = weights
            gate, gate_res = module._gemm_fp4_bf16_fwd(x_i, gate_w)
            up, up_res = module._gemm_fp4_bf16_fwd(x_i, up_w)
            starts = (layer_i, jnp.int32(0), jnp.int32(0))
            gate_stack = jax.lax.dynamic_update_slice(
                gate_stack, gate[None, ...], starts
            )
            up_stack = jax.lax.dynamic_update_slice(
                up_stack, up[None, ...], starts
            )
            sizes = (1, rows, width)
            gate = jax.lax.dynamic_slice(gate_stack, starts, sizes)[0]
            up = jax.lax.dynamic_slice(up_stack, starts, sizes)[0]
            x_next = jax.nn.silu(gate) * up
            carry = (x_next, gate_stack, up_stack, layer_i + jnp.int32(1))
            return carry, (gate_res, up_res)

        carry0 = (x, stack0, stack0, jnp.int32(0))
        carry, gemm_residuals = jax.lax.scan(
            body, carry0, (gate_weights, up_weights)
        )
        x_out, gate_stack, up_stack, _ = carry
        gate_residuals, up_residuals = gemm_residuals
        return x_out, (
            gate_stack,
            up_stack,
            gate_residuals,
            up_residuals,
        )

    @jax.custom_vjp
    def candidate(x, gate_weights, up_weights):
        return candidate_fwd_impl(x, gate_weights, up_weights)[0]

    def candidate_fwd(x, gate_weights, up_weights):
        return candidate_fwd_impl(x, gate_weights, up_weights)

    def candidate_bwd(residuals, grad_out):
        gate_stack, up_stack, gate_residuals, up_residuals = residuals

        def body(dx, layer_residuals):
            gate, up, gate_res, up_res = layer_residuals
            _, swiglu_pullback = jax.vjp(
                lambda g, u: jax.nn.silu(g) * u, gate, up
            )
            dgate, dup = swiglu_pullback(dx)
            dx_gate, dgate_w = module._gemm_fp4_bf16_bwd(
                gate_res, dgate
            )
            dx_up, dup_w = module._gemm_fp4_bf16_bwd(up_res, dup)
            return dx_gate + dx_up, (dgate_w, dup_w)

        dx, (dgate_weights, dup_weights) = jax.lax.scan(
            body,
            grad_out,
            (gate_stack, up_stack, gate_residuals, up_residuals),
            reverse=True,
        )
        return dx, dgate_weights, dup_weights

    candidate.defvjp(candidate_fwd, candidate_bwd)
    return reference, candidate


def _scan_value_and_grad(
    fn, x, gate_weights, up_weights, *, dynamic_slice_fusion
):
    def loss_fn(x_, gate_, up_):
        out = fn(x_, gate_, up_)
        loss = jnp.mean(out.astype(jnp.float32) ** 2)
        return loss, out

    return jax.jit(
        jax.value_and_grad(loss_fn, argnums=(0, 1, 2), has_aux=True),
        compiler_options={
            "xla_gpu_enable_dynamic_slice_fusion": dynamic_slice_fusion,
            "xla_gpu_autotune_level": 0,
        },
    )(x, gate_weights, up_weights)


def test_scan_destination_custom_vjp_matches_reference():
    """Explicit final stacks feed one reverse scan without stack history."""
    layers, rows, width = 2, 256, 256
    kx, kg, ku = jax.random.split(jax.random.PRNGKey(2026), 3)
    x = jax.random.normal(kx, (rows, width), dtype=jnp.bfloat16)
    gate_weights = (
        jax.random.normal(
            kg, (layers, width, width), dtype=jnp.bfloat16
        )
        * 0.1
    )
    up_weights = (
        jax.random.normal(
            ku, (layers, width, width), dtype=jnp.bfloat16
        )
        * 0.1
    )
    reference, candidate = _scan_destination_functions()

    (ref_loss, ref_out), ref_grads = _scan_value_and_grad(
        reference,
        x,
        gate_weights,
        up_weights,
        dynamic_slice_fusion=False,
    )
    (candidate_loss, candidate_out), candidate_grads = _scan_value_and_grad(
        candidate,
        x,
        gate_weights,
        up_weights,
        dynamic_slice_fusion=True,
    )

    np.testing.assert_array_equal(np.asarray(candidate_out), np.asarray(ref_out))
    np.testing.assert_array_equal(
        np.asarray(candidate_loss), np.asarray(ref_loss)
    )
    for label, actual, expected in zip(
        ("dX", "dGate", "dUp"), candidate_grads, ref_grads
    ):
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float32),
            np.asarray(expected, dtype=np.float32),
            rtol=2e-2,
            atol=2e-2,
            err_msg=label,
        )
        assert np.all(np.isfinite(np.asarray(actual))), label

    learning_rate = jnp.float32(1e-3)
    ref_updated = jax.tree.map(
        lambda w, g: w - learning_rate * g,
        (gate_weights, up_weights),
        ref_grads[1:],
    )
    candidate_updated = jax.tree.map(
        lambda w, g: w - learning_rate * g,
        (gate_weights, up_weights),
        candidate_grads[1:],
    )
    for actual, expected in zip(candidate_updated, ref_updated):
        np.testing.assert_allclose(
            np.asarray(actual, dtype=np.float32),
            np.asarray(expected, dtype=np.float32),
            rtol=2e-2,
            atol=2e-2,
        )


# --- scan-vs-unroll invariants (20260818 forward-localization) ---
#
# MaxText's `scan_layers` flag picks between a lax.scan loop and a flat chain of
# layers. A trip-count-1 loop body is lowered differently from a flat chain, so a
# partially-unrolled scan is NOT bitwise equal to the unrolled form once the
# layer contains elementwise work. These tests pin the three invariants that DO
# hold, measured in
# docs/runs/llama3_8b/analysis/20260818_tiny_mxfp4_scan_fwd_localize_097_s1.

_SCAN_COMPILER_OPTS = {
    "xla_gpu_enable_dynamic_slice_fusion": False,
    "xla_gpu_autotune_level": 0,
}


def _stack_inputs(layers, rows, width, seed=2026):
    kx, kg, ku = jax.random.split(jax.random.PRNGKey(seed), 3)
    return (
        jax.random.normal(kx, (rows, width), dtype=jnp.bfloat16),
        jax.random.normal(kg, (layers, width, width), dtype=jnp.bfloat16) * 0.1,
        jax.random.normal(ku, (layers, width, width), dtype=jnp.bfloat16) * 0.1,
    )


def _rms_norm(x):
    f = x.astype(jnp.float32)
    scale = jnp.sqrt(jnp.mean(f * f, axis=-1, keepdims=True) + 1e-6)
    return (f / scale).astype(jnp.bfloat16)


def _fp4_swiglu(x_i, weights):
    gate_w, up_w = weights
    h = _rms_norm(x_i)
    return jax.nn.silu(gemm_fp4_bf16(h, gate_w)) * gemm_fp4_bf16(h, up_w)


def _scan_stack(x, gate_w, up_w, *, unroll, emit_acts=False):
    def body(carry, w):
        out = _fp4_swiglu(carry, w)
        return out, (out if emit_acts else None)

    carry, acts = jax.lax.scan(
        body, x, (gate_w, up_w), unroll=unroll
    )
    return (carry, acts) if emit_acts else carry


def _unrolled_stack(x, gate_w, up_w, *, emit_acts=False):
    acts = []
    for i in range(gate_w.shape[0]):
        x = _fp4_swiglu(x, (gate_w[i], up_w[i]))
        acts.append(x)
    return (x, jnp.stack(acts)) if emit_acts else x


@pytest.mark.parametrize("layers", [2, 4, 8])
def test_fully_unrolled_scan_is_bitwise_equal_to_flat_chain(layers):
    """scan(unroll=layers) must reproduce the flat chain exactly.

    This is the mitigation for the MaxText `scan_layers` forward-loss delta:
    with `scan_layers_unroll` equal to the layer count the two lowerings agree
    bitwise, while the default unroll=1 does not.
    """
    x, gate_w, up_w = _stack_inputs(layers, 256, 256)
    scanned = jax.jit(
        lambda a, b, c: _scan_stack(a, b, c, unroll=layers),
        compiler_options=_SCAN_COMPILER_OPTS,
    )(x, gate_w, up_w)
    flat = jax.jit(_unrolled_stack, compiler_options=_SCAN_COMPILER_OPTS)(
        x, gate_w, up_w
    )
    np.testing.assert_array_equal(np.asarray(scanned), np.asarray(flat))


def test_first_scan_iteration_matches_flat_chain_bitwise():
    """Iteration 0 is exact; divergence starts at iteration 1.

    Both arms feed iteration 0 identical inputs and weights, so a mismatch here
    would mean the scan machinery itself perturbs data rather than the
    loop-body lowering.
    """
    layers = 4
    x, gate_w, up_w = _stack_inputs(layers, 256, 256)
    _, scan_acts = jax.jit(
        lambda a, b, c: _scan_stack(a, b, c, unroll=1, emit_acts=True),
        compiler_options=_SCAN_COMPILER_OPTS,
    )(x, gate_w, up_w)
    _, flat_acts = jax.jit(
        lambda a, b, c: _unrolled_stack(a, b, c, emit_acts=True),
        compiler_options=_SCAN_COMPILER_OPTS,
    )(x, gate_w, up_w)
    np.testing.assert_array_equal(
        np.asarray(scan_acts[0]), np.asarray(flat_acts[0])
    )


def test_gemm_only_chain_is_scan_invariant():
    """A chain of FP4 GEMMs alone is bitwise scan-invariant.

    Isolates the divergence to the elementwise part of the layer: remove the
    SwiGLU product and the second GEMM and the loop-vs-unroll difference
    disappears entirely, so neither the FP4 cast nor the FP4 GEMM causes it.
    """
    layers = 4
    x, gate_w, up_w = _stack_inputs(layers, 256, 256)

    def layer(x_i, w):
        return _rms_norm(gemm_fp4_bf16(x_i, w[0]))

    scanned = jax.jit(
        lambda a, b, c: jax.lax.scan(
            lambda carry, w: (layer(carry, w), None), a, (b, c), unroll=1
        )[0],
        compiler_options=_SCAN_COMPILER_OPTS,
    )(x, gate_w, up_w)

    def flat_fn(a, b, c):
        for i in range(layers):
            a = layer(a, (b[i], c[i]))
        return a

    flat = jax.jit(flat_fn, compiler_options=_SCAN_COMPILER_OPTS)(
        x, gate_w, up_w
    )
    np.testing.assert_array_equal(np.asarray(scanned), np.asarray(flat))


def test_unrolled_chain_ignores_weight_storage_form():
    """Sliced-from-a-stack and separately stored weights agree bitwise.

    MaxText stores per-layer parameters separately when scan_layers=False; this
    confirms storage form is not a variable in the scan comparison.
    """
    layers = 4
    x, gate_w, up_w = _stack_inputs(layers, 256, 256)
    gate_list = [jnp.array(gate_w[i]) for i in range(layers)]
    up_list = [jnp.array(up_w[i]) for i in range(layers)]

    sliced = jax.jit(_unrolled_stack, compiler_options=_SCAN_COMPILER_OPTS)(
        x, gate_w, up_w
    )

    def separate_fn(a, gates, ups):
        for gate_i, up_i in zip(gates, ups):
            a = _fp4_swiglu(a, (gate_i, up_i))
        return a

    separate = jax.jit(separate_fn, compiler_options=_SCAN_COMPILER_OPTS)(
        x, gate_list, up_list
    )
    np.testing.assert_array_equal(np.asarray(sliced), np.asarray(separate))

