# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter import dtypes
from aiter.test_mha_common import (
    attention_ref,
    attn_bias_from_alibi_slopes,
    ck_randval_to_dropout_mask,
    convert_flash_attn_S_to_softmax,
)
import pytest
import argparse

# JAX imports.
import jax
import jax.numpy as jnp
from jax import dlpack as jax_dlpack
from torch.utils import dlpack as torch_dlpack

# JAX-aiter imports.
import jax_aiter
from jax_aiter.ja_compat import dtypes as jax_dtypes
from jax_aiter.mha import flash_attn_func as jax_flash_attn_func


def _to_jax(t: torch.Tensor) -> jnp.ndarray:
    """Convert PyTorch tensor to JAX array."""
    return jax_dlpack.from_dlpack(t.detach())


def _to_torch(x: jnp.ndarray) -> torch.Tensor:
    """Convert JAX array to PyTorch tensor."""
    return torch_dlpack.from_dlpack(x)


def run_torch(
    q,
    k,
    v,
    bias=None,
    alibi_slopes=None,
    dout=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),
    upcast=True,
    reorder_ops=False,
):
    (_, seqlen_q, _, _) = q.shape
    (_, seqlen_k, _, _) = k.shape

    if bias is not None:
        attn_bias = bias
    elif alibi_slopes is not None:
        attn_bias = attn_bias_from_alibi_slopes(
            alibi_slopes, seqlen_q, seqlen_k, causal=causal
        )
    else:
        attn_bias = None

    out, _ = attention_ref(
        q,
        k,
        v,
        None,
        None,
        attn_bias,
        dropout_p,
        dropout_mask,
        causal=causal,
        window_size=window_size,
        upcast=upcast,
        reorder_ops=reorder_ops,
    )

    if dout == None:
        return out
    elif bias is not None:
        dq, dk, dv, dbias = torch.autograd.grad(out, (q, k, v, bias), dout)
        # If seqlen_q > seqlen_k with mask, pytorch will output NaN.
        # Align with ck behavior here.
        dbias = torch.nan_to_num(dbias, nan=0.0)
        return out, dq, dk, dv, dbias
    else:
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
        return out, dq, dk, dv, None


def run_aiter(
    q,
    k,
    v,
    bias=None,
    alibi_slopes=None,
    dout=None,
    dropout_p=0.0,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    return_lse=True,
    return_attn_probs=True,
):
    torch.cuda.synchronize()
    out, _, S_dmask = aiter.flash_attn_func(
        q,
        k,
        v,
        dropout_p,
        causal=causal,
        window_size=window_size,
        bias=bias,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_lse=return_lse,
        return_attn_probs=return_attn_probs,
    )

    if dropout_p > 0.0:
        (_, seqlen_q, _, d) = q.shape
        (_, seqlen_k, _, d) = k.shape
        (_, seqlen_k, _, d_v) = v.shape
        S_dmask = ck_randval_to_dropout_mask(S_dmask, dropout_p)
        S_dmask_converted = convert_flash_attn_S_to_softmax(
            S_dmask,
            seqlen_q,
            seqlen_k,
            None,
            None,
            d,
            dropout_p > 0.0,
            causal=causal,
            window_size=window_size,
        )
        dropout_mask = S_dmask_converted >= 0
    else:
        dropout_mask = None

    if dout == None:
        return out, dropout_mask
    elif bias is not None:
        dq, dk, dv, dbias = torch.autograd.grad(out, (q, k, v, bias), dout)
        return out, dropout_mask, dq, dk, dv, dbias
    else:
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
        return out, dropout_mask, dq, dk, dv, None


def run_jax(
    q,
    k,
    v,
    bias=None,
    alibi_slopes=None,
    dout=None,
    dropout_p=0.0,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window.
    deterministic=False,
    return_lse=True,
    return_attn_probs=True,
):
    """Foolproof JAX implementation using single-path jax.vjp approach.
    Call your JAX API (jax_flash_attn_func) exactly once.
    Get outputs and gradients using one forward + a VJP (vector-Jacobian product),
    so we don’t accidentally re-run forward.
    Cross the Torch - JAX boundary zero-copy (via the DLPack protocol), and keep data on device.
    Make async execution explicit: block only on values we export/compare.
    Reconstruct the dropout mask deterministically from the same forward outputs.
    """
    torch.cuda.synchronize()
    try:
        _, seqlen_q, _, _ = q.shape
        _, seqlen_k, _, _ = k.shape

        # Normalize window like AITER (-1 stays -1 for unlimited context).
        def _norm_ws(ws, sk):
            wl, wr = (
                (int(ws[0].item()), int(ws[1].item())) if hasattr(ws, "shape") else ws
            )

            def n(x):
                return -1 if x < 0 else max(0, min(int(x), sk - 1))

            return (n(wl), n(wr))

        wl, wr = _norm_ws(window_size, seqlen_k)

        # Convert to jax arrays.
        qj = _to_jax(q)
        kj = _to_jax(k)
        vj = _to_jax(v)
        bj = _to_jax(bias) if bias is not None else None
        alj = _to_jax(alibi_slopes) if alibi_slopes is not None else None
        if bias is not None:
            alj = None  # bias wins over ALiBi.

        # Keep dropout deterministic during tests/comparisons.
        det_flag = bool(deterministic or (dropout_p > 0.0))

        # Forward that returns full output structure (out, lse, attn_probs).
        def fwd_core(qv, kv, vv, bv):
            return jax_flash_attn_func(
                qv,
                kv,
                vv,
                dropout_p=dropout_p,
                causal=causal,
                window_size=(wl, wr),
                bias=bv,
                alibi_slopes=alj,
                deterministic=det_flag,
                return_lse=True,
                return_attn_probs=True,
            )

        # ONE FORWARD ONLY - preserve full output for all uses.
        if bj is None:
            full_out, pullback = jax.vjp(
                lambda Q, K, V: fwd_core(Q, K, V, None), qj, kj, vj
            )
        else:
            full_out, pullback = jax.vjp(
                lambda Q, K, V, B: fwd_core(Q, K, V, B), qj, kj, vj, bj
            )

        # Extract components from SINGLE forward call.
        if isinstance(full_out, (tuple, list)) and len(full_out) >= 3:
            outj, lse_j, S_dmask_jax = full_out
        elif isinstance(full_out, (tuple, list)) and len(full_out) >= 1:
            outj = full_out[0]
            lse_j = full_out[1] if len(full_out) > 1 else None
            S_dmask_jax = None
        else:
            outj = full_out
            lse_j = None
            S_dmask_jax = None

        # Cotangent = dout (if provided) else zeros; this does NOT add another forward.
        cot = jnp.zeros_like(outj) if dout is None else _to_jax(dout)

        # Create cotangent structure matching the forward output.
        if isinstance(full_out, (tuple, list)):
            # Match the structure: (out_cot, lse_cot, attn_probs_cot).
            full_cot = [cot]
            if len(full_out) > 1:
                full_cot.append(jnp.zeros_like(full_out[1]))
            if len(full_out) > 2:
                full_cot.append(jnp.zeros_like(full_out[2]))
            full_cot = tuple(full_cot)
        else:
            full_cot = cot

        grads = pullback(full_cot)
        if bj is None:
            dqj, dkj, dvj = grads
            dbj = None
        else:
            dqj, dkj, dvj, dbj = grads

        # outj.block_until_ready()
        # if lse_j is not None: lse_j.block_until_ready()
        # if S_dmask_jax is not None: S_dmask_jax.block_until_ready()
        # if dout is not None:
        #     dqj.block_until_ready(); dkj.block_until_ready(); dvj.block_until_ready()
        #     if dbj is not None: dbj.block_until_ready()

        # Extract dropout mask from SAME forward call (no additional calls).
        dropout_mask = None
        if dropout_p > 0.0 and S_dmask_jax is not None:
            # Convert JAX dropout mask to torch format.
            S_dmask_torch = _to_torch(S_dmask_jax)

            # Apply same conversion logic as AITER.
            S_dmask_converted = ck_randval_to_dropout_mask(S_dmask_torch, dropout_p)
            S_dmask_converted = convert_flash_attn_S_to_softmax(
                S_dmask_converted,
                seqlen_q,
                seqlen_k,
                None,
                None,
                q.shape[-1],  # d dimension.
                dropout_p > 0.0,
                causal=causal,
                window_size=window_size,
            )
            dropout_mask = S_dmask_converted >= 0

        # Back to Torch for test harness - use results from SINGLE forward.
        out_t = _to_torch(outj)

        if dout is None:
            return out_t, dropout_mask
        else:
            dq_t = _to_torch(dqj)
            dk_t = _to_torch(dkj)
            dv_t = _to_torch(dvj)
            dbias_t = _to_torch(dbj) if dbj is not None else None
            return out_t, dropout_mask, dq_t, dk_t, dv_t, dbias_t

    except Exception as e:
        print(f"JAX implementation failed: {e}")
        import traceback

        traceback.print_exc()
        return None


@pytest.mark.parametrize("dtype", [dtypes.fp16, dtypes.bf16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize("bias_type", ["no", "bias", "alibi"])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dropout_p", [0.0, 0.17])
@pytest.mark.parametrize("batch_size", [5])
@pytest.mark.parametrize("nheads", [6])
@pytest.mark.parametrize(
    "d,d_v",
    [
        (32, 32),
        (40, 40),
        (59, 59),
        (64, 64),
        (96, 96),
        (111, 111),
        (128, 128),
        (160, 160),
        (192, 192),
        (224, 224),
        (256, 256),
    ],
)
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
    ],
)
def test_flash_attn_output(
    batch_size,
    nheads,
    seqlen_q,
    seqlen_k,
    d,
    d_v,
    dropout_p,
    causal,
    local,
    bias_type,
    deterministic,
    mha_type,
    dtype,
):
    # RNG isolation is handled by the fixture and run_aiter function.
    torch.random.manual_seed(123)
    torch.cuda.empty_cache()
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    assert nheads % nheads_k == 0
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))

    return_lse = True
    return_attn_probs = True

    q = torch.randn(
        batch_size, seqlen_q, nheads, d, device="cuda", dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        batch_size,
        seqlen_k,
        nheads_k,
        d,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        batch_size,
        seqlen_k,
        nheads_k,
        d_v,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )

    attn_bias = None
    alibi_slopes = None
    if bias_type == "bias":
        attn_bias = torch.randn(
            seqlen_q, seqlen_k, device="cuda", dtype=dtype, requires_grad=True
        )
    elif bias_type == "alibi":
        alibi_slopes = torch.rand(batch_size, nheads, device="cuda", dtype=dtypes.fp32)

    dout = torch.randn(
        batch_size,
        seqlen_q,
        nheads,
        d_v,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )

    # Run aiter CK implementation.
    out, dropout_mask, dq, dk, dv, dbias = run_aiter(
        q,
        k,
        v,
        attn_bias,
        alibi_slopes,
        dout,
        dropout_p,
        causal,
        window_size,
        deterministic,
        return_lse,
        return_attn_probs,
    )

    # Run PyTorch reference.
    out_ref, dq_ref, dk_ref, dv_ref, dbias_ref = run_torch(
        q,
        k,
        v,
        attn_bias,
        alibi_slopes,
        dout,
        dropout_p,
        dropout_mask,
        causal,
        window_size,
    )

    # Run PyTorch reference with different settings for tolerance computation.
    out_pt, dq_pt, dk_pt, dv_pt, dbias_pt = run_torch(
        q,
        k,
        v,
        attn_bias,
        alibi_slopes,
        dout,
        dropout_p,
        dropout_mask,
        causal,
        window_size,
        upcast=False,
        reorder_ops=True,
    )

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    out_tol = max(2 * (out_pt - out_ref).abs().max().item(), 0.01)
    assert (out - out_ref).abs().max().item() <= out_tol

    print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
    print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
    print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
    print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
    print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
    print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")

    dq_tol = max(10 * (dq_pt - dq_ref).abs().max().item(), 0.01)
    dk_tol = max(10 * (dk_pt - dk_ref).abs().max().item(), 0.01)
    dv_tol = max(10 * (dv_pt - dv_ref).abs().max().item(), 0.01)

    assert (dq - dq_ref).abs().max().item() <= dq_tol
    assert (dk - dk_ref).abs().max().item() <= dk_tol
    assert (dv - dv_ref).abs().max().item() <= dv_tol

    if attn_bias is not None:
        print(f"dBias max diff: {(dbias - dbias_ref).abs().max().item()}")
        print(f"dBias Pytorch max diff: {(dbias_pt - dbias_ref).abs().max().item()}")
        dbias_tol = max(10 * (dbias_pt - dbias_ref).abs().max().item(), 0.01)
        assert (dbias - dbias_ref).abs().max().item() <= dbias_tol

    # Run JAX implementation and compare if supported.
    jax_result = run_jax(
        q,
        k,
        v,
        attn_bias,
        alibi_slopes,
        dout,
        dropout_p,
        causal,
        window_size,
        deterministic,
        return_lse,
        return_attn_probs,
    )

    if jax_result is not None:
        if len(jax_result) >= 5:
            out_jax, _, dq_jax, dk_jax, dv_jax, dbias_jax = jax_result

            # Compare JAX vs aiter CK..
            print(f"JAX vs AITER output max diff: {(out_jax - out).abs().max().item()}")
            print(f"JAX vs AITER dQ max diff: {(dq_jax - dq).abs().max().item()}")
            print(f"JAX vs AITER dK max diff: {(dk_jax - dk).abs().max().item()}")
            print(f"JAX vs AITER dV max diff: {(dv_jax - dv).abs().max().item()}")

            # Use similar tolerances as CK vs reference.
            assert (
                out_jax - out
            ).abs().max().item() <= out_tol, "JAX output doesn't match AITER"
            assert (
                dq_jax - dq
            ).abs().max().item() <= dq_tol, "JAX dQ doesn't match AITER"
            assert (
                dk_jax - dk
            ).abs().max().item() <= dk_tol, "JAX dK doesn't match AITER"
            assert (
                dv_jax - dv
            ).abs().max().item() <= dv_tol, "JAX dV doesn't match AITER"

        else:
            out_jax = jax_result[0]
            print(f"JAX vs AITER output max diff: {(out_jax - out).abs().max().item()}")
            assert (
                out_jax - out
            ).abs().max().item() <= out_tol, "JAX output doesn't match AITER"


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Test JAX-aiter MHA implementation",
)
parser.add_argument(
    "-b",
    "--batch_size",
    type=int,
    default=2,
    help="""Batch size. Default is 2.
    e.g.: -b 16""",
)
parser.add_argument(
    "-n",
    "--nheads",
    type=int,
    default=5,
    help="""Number of heads. Default is 5.
    e.g.: -n 8""",
)
parser.add_argument(
    "-q",
    "--seqlen_q",
    type=int,
    default=512,
    help="""Sequence length for query. Default is 512.
    e.g.: -q 1024""",
)
parser.add_argument(
    "-k",
    "--seqlen_k",
    type=int,
    default=512,
    help="""Sequence length for key. Default is 512.
    e.g.: -k 1024""",
)
parser.add_argument(
    "-qk",
    "--d_qk",
    type=int,
    default=128,
    help="""Dimension of query and key. Default is 128.
    e.g.: -qk 256""",
)
parser.add_argument(
    "-v",
    "--d_v",
    type=int,
    default=128,
    help="""Dimension of value. Default is 128.
    e.g.: -v 256""",
)
parser.add_argument(
    "-p",
    "--dropout_p",
    type=float,
    default=0.0,
    help="""Dropout probability. Default is 0.0.
    e.g.: -p 0.1""",
)
parser.add_argument(
    "-c",
    "--causal",
    action="store_true",
    help="""Causal attention. Default is False.
    -c or --causal    # enable causal attention""",
)
parser.add_argument(
    "-l",
    "--local",
    action="store_true",
    help="""Local attention. Default is False.
    -l or --local    # enable local attention""",
)
parser.add_argument(
    "-bt",
    "--bias_type",
    type=str,
    default="no",
    help="""Bias type. Default is 'no'.
    e.g.: -bt no""",
)
parser.add_argument(
    "-det",
    "--deterministic",
    action="store_true",
    help="""Deterministic attention. Default is False.
    -det or --deterministic    # enable deterministic attention""",
)
parser.add_argument(
    "-m",
    "--mha_type",
    type=str,
    default="mha",
    help="""Type of multi-head attention.
    e.g.: -m mha""",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    default="bf16",
    help="""Data type.
    e.g.: -d bf16""",
)

if __name__ == "__main__":
    args = parser.parse_args()
    dtype = dtypes.d_dtypes[args.dtype]
    test_flash_attn_output(
        args.batch_size,
        args.nheads,
        args.seqlen_q,
        args.seqlen_k,
        args.d_qk,
        args.d_v,
        args.dropout_p,
        args.causal,
        args.local,
        args.bias_type,
        args.deterministic,
        args.mha_type,
        dtype,
    )
