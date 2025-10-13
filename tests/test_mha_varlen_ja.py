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
    generate_qkv,
    generate_random_padding_mask,
    pad_rearrange_dropout_mask_hts_to_bhss,
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
from jax_aiter.mha import flash_attn_varlen


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
    query_padding_mask,
    key_padding_mask,
    bias=None,
    alibi_slopes=None,
    dout=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    upcast=True,
    reorder_ops=False,
):
    (b, seqlen_q, _, _) = q.shape
    (_, seqlen_k, _, _) = k.shape

    if bias is not None:
        attn_bias = bias.reshape(b, 1, seqlen_q, seqlen_k)
    elif alibi_slopes is not None:
        attn_bias = attn_bias_from_alibi_slopes(
            alibi_slopes,
            seqlen_q,
            seqlen_k,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
        )
    else:
        attn_bias = None

    out, _ = attention_ref(
        q,
        k,
        v,
        query_padding_mask,
        key_padding_mask,
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
    else:
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
        return out, dq, dk, dv


def run_jax(
    q,
    k,
    v,
    query_padding_mask,
    key_padding_mask,
    min_seqlen_q=0,
    bias=None,
    alibi_slopes=None,
    dout=None,
    dropout_p=0.0,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    return_lse=True,
    return_attn_probs=False,
):
    """JAX implementation for varlen attention."""
    # Unpad inputs using generate_qkv.
    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q_padded,
        k_padded,
        v_padded,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)
    if bias is not None:
        # TODO - implement generate_bias() to unpad.
        total_q = q_unpad.shape[0]
        assert total_q == batch_size * max_seqlen_q
        assert q.shape[1] == max_seqlen_q
        assert k.shape[1] == max_seqlen_k
        bias_unpad = bias.reshape(batch_size * max_seqlen_q, max_seqlen_k)
    else:
        bias_unpad = None

    # Convert to JAX
    qj = _to_jax(q_unpad)
    kj = _to_jax(k_unpad)
    vj = _to_jax(v_unpad)
    cu_seqlens_qj = _to_jax(cu_seqlens_q)
    cu_seqlens_kj = _to_jax(cu_seqlens_k)
    bj = _to_jax(bias_unpad) if bias_unpad is not None else None
    alj = _to_jax(alibi_slopes) if alibi_slopes is not None else None

    window_size = tuple(int(x) for x in window_size)

    # JAX forward wrapper.
    def fwd_core(Q, K, V, B):
        return flash_attn_varlen(
            Q,
            K,
            V,
            cu_seqlens_qj,
            cu_seqlens_kj,
            max_seqlen_q,
            max_seqlen_k,
            min_seqlen_q=min_seqlen_q,
            dropout_p=dropout_p,
            softmax_scale=None,
            logits_soft_cap=0.0,
            causal=causal,
            window_size=window_size,
            bias=B,
            alibi_slopes=alj,
            block_table=None,
            deterministic=deterministic,
            return_lse=return_lse,
            return_attn_probs=return_attn_probs,
        )

    # One forward + vjp.
    (outj, lse_j, S_dmask_jax), pullback = jax.vjp(
        lambda Q, K, V, B: fwd_core(Q, K, V, B), qj, kj, vj, bj
    )

    # Convert and pad output.
    out_t = output_pad_fn(_to_torch(outj))

    # Convert dropout mask if present
    dropout_mask = None
    if dropout_p > 0.0 and return_attn_probs:
        (_, seqlen_q, _, d) = q.shape
        (_, seqlen_k, _, d) = k.shape
        S_dmask = _to_torch(S_dmask_jax)
        S_dmask = ck_randval_to_dropout_mask(S_dmask, dropout_p)
        S_dmask = pad_rearrange_dropout_mask_hts_to_bhss(
            S_dmask, cu_seqlens_q, seqlen_q, seqlen_k
        )
        S_dmask_converted = convert_flash_attn_S_to_softmax(
            S_dmask,
            seqlen_q,
            seqlen_k,
            query_padding_mask,
            key_padding_mask,
            d,
            True,  # dropout_p > 0.0
            causal=causal,
            window_size=window_size,
        )
        dropout_mask = S_dmask_converted >= 0

    # Handle gradients if dout provided.
    if dout is None or not return_lse:
        return out_t, dropout_mask, None, None, None

    # Unpad dout for JAX
    from aiter.bert_padding import unpad_input

    dout_unpad, _, _, _, _ = unpad_input(dout, query_padding_mask)
    cot = (
        _to_jax(dout_unpad),
        jnp.zeros_like(lse_j) if return_lse else None,
        jnp.zeros_like(S_dmask_jax) if return_attn_probs else None,
    )

    # Pullback.
    if bj is None:
        dqj, dkj, dvj, _ = pullback(cot)
    else:
        dqj, dkj, dvj, dbj = pullback(cot)

    # Convert grads back to torch and pad
    dq_t = _to_torch(dqj)
    dk_t = _to_torch(dkj)
    dv_t = _to_torch(dvj)

    dq = dq_pad_fn(dq_t)
    dk = dk_pad_fn(dk_t)
    dv = dk_pad_fn(dv_t)  # Note: CK test used dk_pad_fn for dv as well

    return out_t, dropout_mask, dq, dk, dv


@pytest.mark.parametrize("dtype", [dtypes.fp16, dtypes.bf16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize("bias_type", ["no", "alibi"])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("min_seqlen_q", [0])
@pytest.mark.parametrize("dropout_p", [0.0, 0.17])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("nheads", [9])
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
        (1, 147),
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
def test_flash_attn_varlen_func(
    batch_size,
    nheads,
    seqlen_q,
    seqlen_k,
    d,
    d_v,
    min_seqlen_q,
    dropout_p,
    causal,
    local,
    bias_type,
    deterministic,
    mha_type,
    dtype,
):
    return_lse = True
    torch.random.manual_seed(123)
    torch.cuda.empty_cache()
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    assert nheads % nheads_k == 0
    window_size = (-1, -1) if not local else tuple(torch.randint(0, seqlen_k, (2,)))

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

    if bias_type == "bias":
        # TODO - We need to implement unpad bias [batch_size, seqlen_q, seqlen_k] -> [total_q, max_seqlen_k]
        # Let total_q = batch_size * seqlen_q to pass the test for now
        query_padding_mask = generate_random_padding_mask(
            seqlen_q, batch_size, "cuda", mode="full"
        )
        key_padding_mask = generate_random_padding_mask(
            seqlen_k, batch_size, "cuda", mode="full"
        )
    else:
        query_padding_mask = generate_random_padding_mask(
            seqlen_q, batch_size, "cuda", mode="random"
        )
        key_padding_mask = generate_random_padding_mask(
            seqlen_k, batch_size, "cuda", mode="random"
        )

    attn_bias = None
    alibi_slopes = None
    if bias_type == "bias":
        attn_bias = torch.randn(
            batch_size,
            seqlen_q,
            seqlen_k,
            device="cuda",
            dtype=dtype,
            requires_grad=True,
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

    # return_attn_probs is just for host verification (to produce same dropout mask)
    # no need to use in actual case
    if dropout_p > 0:
        return_attn_probs = True
    else:
        return_attn_probs = False

    # Returns from the aiter:
    # out_aiter, dropout_mask_aiter, dq_aiter, dk_aiter, dv_aiter
    jax_result = run_jax(
        q,
        k,
        v,
        query_padding_mask,
        key_padding_mask,
        min_seqlen_q,
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

    out_jax, dropout_mask_jax, dq_jax, dk_jax, dv_jax = jax_result

    # Run torch reference
    out_ref, dq_ref, dk_ref, dv_ref = run_torch(
        q,
        k,
        v,
        query_padding_mask,
        key_padding_mask,
        attn_bias,
        alibi_slopes,
        dout,
        dropout_p,
        dropout_mask_jax,  # Use JAX dropout mask for comparison
        causal,
        window_size,
    )

    out_pt, dq_pt, dk_pt, dv_pt = run_torch(
        q,
        k,
        v,
        query_padding_mask,
        key_padding_mask,
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
    # print(f"AITER vs REF Output max diff: {(out_aiter - out_ref).abs().max().item()}")
    print(f"JAX vs REF Output max diff: {(out_jax - out_ref).abs().max().item()}")
    # print(f"JAX vs AITER Output max diff: {(out_jax - out_aiter).abs().max().item()}")
    print(f"PT vs REF Output max diff: {(out_pt - out_ref).abs().max().item()}")

    out_tol = max(4 * (out_pt - out_ref).abs().max().item(), 0.01)
    assert (
        out_jax - out_ref
    ).abs().max().item() <= out_tol, "JAX output doesn't match REF"

    # TODO: Support varlen bwd for bias
    if bias_type == "bias":
        pytest.skip("Does not support varlen bwd for bias")

    if dq_jax is not None:
        # print(f"AITER dQ max diff: {(dq_aiter - dq_ref).abs().max().item()}")
        # print(f"AITER dK max diff: {(dk_aiter - dk_ref).abs().max().item()}")
        # print(f"AITER dV max diff: {(dv_aiter - dv_ref).abs().max().item()}")
        print(f"JAX dQ max diff: {(dq_jax - dq_ref).abs().max().item()}")
        print(f"JAX dK max diff: {(dk_jax - dk_ref).abs().max().item()}")
        print(f"JAX dV max diff: {(dv_jax - dv_ref).abs().max().item()}")
        # print(f"JAX vs AITER dQ max diff: {(dq_jax - dq_aiter).abs().max().item()}")
        # print(f"JAX vs AITER dK max diff: {(dk_jax - dk_aiter).abs().max().item()}")
        # print(f"JAX vs AITER dV max diff: {(dv_jax - dv_aiter).abs().max().item()}")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Test JAX-aiter MHA varlen implementation",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=4,
        help="""Batch size. Default is 4.
    e.g.: -b 16""",
    )
    parser.add_argument(
        "-nh",
        "--nheads",
        type=int,
        default=4,
        help="""Number of attention heads. Default is 4.
    e.g. -nh 4""",
    )
    parser.add_argument(
        "-s",
        "--seqlen_q_k",
        type=dtypes.str2tuple,
        default=(4, 8),
        help="""Sequence length of query&key. Default is (4, 8).
    e.g. -s 4,8""",
    )
    parser.add_argument(
        "-d",
        type=int,
        default=128,
        help="""Dimension of query&key. Default is 128.
    e.g. -d 128""",
    )
    parser.add_argument(
        "-dv",
        type=int,
        default=128,
        help="""Dimension of value. Default is 128.
    e.g. -dv 128""",
    )
    parser.add_argument(
        "-dp",
        "--dropout_p",
        type=float,
        default=0.0,
        help="""Dropout probability. Default is 0.0.
    e.g. -dp 0.0""",
    )
    parser.add_argument(
        "-msq",
        "--min_seqlen_q",
        type=int,
        default=0,
        help="""Minimum sequence length of query. Default is 0.
    e.g. -msq 1""",
    )
    parser.add_argument(
        "-c",
        "--causal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="""Causal attention, default is True.
    -c or --causal    # enable causal attention
    --no-causal       # disable causal attention""",
    )
    parser.add_argument(
        "-l",
        "--local",
        action="store_true",
        default=False,
        help="""Local attention. default is False.
    e.g. -l or --local    # enable local attention""",
    )
    parser.add_argument(
        "-bt",
        "--bias_type",
        type=str,
        default="no",
        help="""Type of bias. Default is 'no'.
    e.g. -bt no/alibi""",
    )
    parser.add_argument(
        "-det",
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="""Deterministic attention, default is True.
    -det or --deterministic    # enable deterministic attention
    --no-deterministic         # disable deterministic attention""",
    )
    parser.add_argument(
        "-mha",
        "--mha_type",
        type=str,
        default="mha",
        help="""Type of multi-head attention. Default is 'mha'.
    e.g. -mha mha/mqa/gqa""",
    )
    parser.add_argument(
        "-dt",
        "--dtype",
        type=str,
        default="bf16",
        help="""Data type. Default is 'bf16'.
    e.g.: -dt bf16""",
    )

    args = parser.parse_args()
    dtype = dtypes.d_dtypes[args.dtype]
    (seqlen_q, seqlen_k) = args.seqlen_q_k

    test_flash_attn_varlen_func(
        args.batch_size,
        args.nheads,
        seqlen_q,
        seqlen_k,
        args.d,
        args.dv,
        args.min_seqlen_q,
        args.dropout_p,
        args.causal,
        args.local,
        args.bias_type,
        args.deterministic,
        args.mha_type,
        dtype,
    )
