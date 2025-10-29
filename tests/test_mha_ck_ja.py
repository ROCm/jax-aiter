# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import os
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

    out, _, _ = attention_ref(
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


def run_jax(
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
    """Simplified JAX implementation with single VJP call."""
    # Convert PyTorch -> JAX (zero-copy via DLPack)
    qj = _to_jax(q)
    kj = _to_jax(k)
    vj = _to_jax(v)
    bj = _to_jax(bias) if bias is not None else None
    alj = _to_jax(alibi_slopes) if alibi_slopes is not None else None

    window_size = tuple(int(x) for x in window_size)

    (outj, lse_j, S_dmask_j), pullback = jax.vjp(
        lambda Q, K, V, B, ALJ: jax_flash_attn_func(
            Q,
            K,
            V,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            bias=B,
            alibi_slopes=ALJ,
            deterministic=deterministic,
            return_lse=return_lse,
            return_attn_probs=return_attn_probs,
        ),
        qj,
        kj,
        vj,
        bj,
        alj,
    )

    # Reconstruct dropout mask.
    dropout_mask = None
    if dropout_p > 0.0:
        (_, seqlen_q, _, d_q) = q.shape
        (_, seqlen_k, _, d_k) = k.shape
        (_, seqlen_k, _, d_v) = v.shape
        S_dmask_converted = ck_randval_to_dropout_mask(_to_torch(S_dmask_j), dropout_p)
        S_dmask_converted = convert_flash_attn_S_to_softmax(
            S_dmask_converted,
            seqlen_q,
            seqlen_k,
            None,
            None,
            d_q,
            dropout_p > 0.0,
            causal=causal,
            window_size=window_size,
        )
        dropout_mask = S_dmask_converted >= 0

    out_t = _to_torch(outj)

    if dout is None:
        return out_t, dropout_mask, _, _, _, _

    cot = (
        _to_jax(dout),
        jnp.zeros_like(lse_j),
        jnp.zeros_like(S_dmask_j),
    )

    if bj is None:
        dqj, dkj, dvj, _, _ = pullback(cot)
        dbj = None
    else:
        dqj, dkj, dvj, dbj, _ = pullback(cot)

    # Convert JAX -> PyTorch (zero-copy via DLPack)
    dq_t = _to_torch(dqj)
    dk_t = _to_torch(dkj)
    dv_t = _to_torch(dvj)
    dbias_t = _to_torch(dbj) if dbj is not None else None

    return out_t, dropout_mask, dq_t, dk_t, dv_t, dbias_t


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
    torch.random.manual_seed(0)
    torch.cuda.empty_cache()

    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    assert nheads % nheads_k == 0
    window_size = (-1, -1) if not local else tuple(torch.randint(0, seqlen_k, (2,)))

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

    out_jax, dropout_mask_jax, dq_jax, dk_jax, dv_jax, dbias_jax = run_jax(
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

    out_ref, dq_ref, dk_ref, dv_ref, dbias_ref = run_torch(
        q,
        k,
        v,
        attn_bias,
        alibi_slopes,
        dout,
        dropout_p,
        dropout_mask_jax,
        causal,
        window_size,
    )

    out_pt, dq_pt, dk_pt, dv_pt, dbias_pt = run_torch(
        q,
        k,
        v,
        attn_bias,
        alibi_slopes,
        dout,
        dropout_p,
        dropout_mask_jax,
        causal,
        window_size,
        upcast=False,
        reorder_ops=True,
    )

    # Compare outputs
    print(f"JAX vs REF Output max diff: {(out_jax - out_ref).abs().max().item()}")
    print(f"PT vs REF Output max diff: {(out_pt - out_ref).abs().max().item()}")

    out_tol = max(2 * (out_pt - out_ref).abs().max().item(), 0.01)
    assert (
        out_jax - out_ref
    ).abs().max().item() <= out_tol, "JAX output doesn't match REF"

    # Compare gradients
    if dq_jax is not None:
        print(f"JAX dQ max diff: {(dq_jax - dq_ref).abs().max().item()}")
        print(f"JAX dK max diff: {(dk_jax - dk_ref).abs().max().item()}")
        print(f"JAX dV max diff: {(dv_jax - dv_ref).abs().max().item()}")
        print(f"PT dQ max diff: {(dq_pt - dq_ref).abs().max().item()}")
        print(f"PT dK max diff: {(dk_pt - dk_ref).abs().max().item()}")
        print(f"PT dV max diff: {(dv_pt - dv_ref).abs().max().item()}")

        dq_tol = max(10 * (dq_pt - dq_ref).abs().max().item(), 0.01)
        dk_tol = max(10 * (dk_pt - dk_ref).abs().max().item(), 0.01)
        dv_tol = max(10 * (dv_pt - dv_ref).abs().max().item(), 0.01)

        assert (
            dq_jax - dq_ref
        ).abs().max().item() <= dq_tol, "JAX dQ doesn't match REF"
        assert (
            dk_jax - dk_ref
        ).abs().max().item() <= dk_tol, "JAX dK doesn't match REF"
        assert (
            dv_jax - dv_ref
        ).abs().max().item() <= dv_tol, "JAX dV doesn't match REF"

        if attn_bias is not None and dbias_jax is not None:
            print(f"JAX dBias max diff: {(dbias_jax - dbias_ref).abs().max().item()}")
            print(f"PT dBias max diff: {(dbias_pt - dbias_ref).abs().max().item()}")
            dbias_tol = max(10 * (dbias_pt - dbias_ref).abs().max().item(), 0.01)
            # assert (
            #     dbias_jax - dbias_ref
            # ).abs().max().item() <= dbias_tol, "JAX dBias doesn't match REF"


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
