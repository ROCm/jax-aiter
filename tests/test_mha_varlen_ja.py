# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import pytest
import argparse

# JAX imports.
import jax
import jax.numpy as jnp

# JAX-aiter imports.
import jax_aiter
from jax_aiter.ja_compat import dtypes
from jax_aiter.mha import flash_attn_varlen
from jax_aiter.ja_compat.test_common import benchmark, run_perftest
from jax_aiter.baseline.mha_attn import (
    attention_ref,
    generate_qkv,
    ck_randval_to_dropout_mask,
    convert_flash_attn_S_to_softmax,
    attn_bias_from_alibi_slopes,
    generate_random_padding_mask,
    pad_rearrange_dropout_mask_hts_to_bhss,
)


def run_jax(
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
    b, seqlen_q, _, _ = q.shape
    _, seqlen_k, _, _ = k.shape

    if bias is not None:
        attn_bias = bias.reshape(b, 1, seqlen_q, seqlen_k)
    elif alibi_slopes is not None:
        attn_bias = attn_bias_from_alibi_slopes(
            alibi_slopes,
            seqlen_q,
            seqlen_k,
            query_padding_mask=query_padding_mask,
            key_padding_mask=key_padding_mask,
            causal=causal,
            key_leftpad=None,
        )
    else:
        attn_bias = None

    out, _, _ = attention_ref(
        q,
        k,
        v,
        query_padding_mask=query_padding_mask,
        key_padding_mask=key_padding_mask,
        attn_bias=attn_bias,
        dropout_p=dropout_p,
        dropout_mask=dropout_mask,
        causal=causal,
        window_size=window_size,
        upcast=upcast,
        reorder_ops=reorder_ops,
        key_leftpad=None,
    )

    if dout == None:
        return out

    def f(q_, k_, v_):
        o_, _, _ = attention_ref(
            q_,
            k_,
            v_,
            query_padding_mask=query_padding_mask,
            key_padding_mask=key_padding_mask,
            attn_bias=attn_bias,
            dropout_p=dropout_p,
            dropout_mask=dropout_mask,
            causal=causal,
            window_size=window_size,
            upcast=upcast,
            reorder_ops=reorder_ops,
            key_leftpad=None,
        )
        return o_

    _, vjp_fn = jax.vjp(f, q, k, v)
    dq, dk, dv = vjp_fn(dout)

    return out, dq, dk, dv


def run_jax_aiter(
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
    cu_seqlens_q_padded=None,
    cu_seqlens_k_padded=None,
    input_layout="BSHD",
):
    window_size = tuple(int(x) for x in window_size)

    B, max_seqlen_q, nheads_q, d_q = q.shape
    _, max_seqlen_k, _, d_v = v.shape

    # lengths and cu_seqlens from masks
    seqlens_q = jnp.sum(query_padding_mask, axis=1).astype(dtypes.i32)
    seqlens_k = jnp.sum(key_padding_mask, axis=1).astype(dtypes.i32)

    if min_seqlen_q is not None:
        assert bool(jnp.all(seqlens_q >= min_seqlen_q))

    cu_seqlens_q = jnp.concatenate(
        [jnp.array([0], dtype=dtypes.i32), jnp.cumsum(seqlens_q)], axis=0
    )
    cu_seqlens_k = jnp.concatenate(
        [jnp.array([0], dtype=dtypes.i32), jnp.cumsum(seqlens_k)], axis=0
    )

    # unpad q, k, v
    q_chunks, k_chunks, v_chunks = [], [], []
    for b in range(B):
        q_len = int(seqlens_q[b])
        k_len = int(seqlens_k[b])
        q_chunks.append(q[b, :q_len, :, :])
        k_chunks.append(k[b, :k_len, :, :])
        v_chunks.append(v[b, :k_len, :, :])

    q_unpad = jnp.concatenate(q_chunks, axis=0)  # (total_q, nheads_q, d_q)
    k_unpad = jnp.concatenate(k_chunks, axis=0)  # (total_k, nheads_q, d_q)
    v_unpad = jnp.concatenate(v_chunks, axis=0)  # (total_k, nheads_q, d_v)

    # bias: forward-only (tests skip varlen bwd for bias)
    if bias is not None:
        bias_chunks = []
        for b in range(B):
            q_len = int(seqlens_q[b])
            bias_chunks.append(bias[b, :q_len, :max_seqlen_k])
        bias_unpad = jnp.concatenate(bias_chunks, axis=0)  # (total_q, max_seqlen_k)
    else:
        bias_unpad = None

    def _fwd(Q_unpad, K_unpad, V_unpad):
        return flash_attn_varlen(
            Q_unpad,
            K_unpad,
            V_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            int(max_seqlen_q),
            int(max_seqlen_k),
            min_seqlen_q=min_seqlen_q,
            dropout_p=dropout_p,
            softmax_scale=None,
            logits_soft_cap=0.0,
            causal=causal,
            window_size=window_size,
            bias=bias_unpad,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_lse=return_lse,
            return_attn_probs=return_attn_probs,
            block_table=None,
            out=None,
            cu_seqlens_q_padded=cu_seqlens_q_padded,
            cu_seqlens_k_padded=cu_seqlens_k_padded,
        )

    (out_unpad, softmax_lse, S_dmask), pullback = jax.vjp(
        _fwd, q_unpad, k_unpad, v_unpad
    )

    # pad output back to [B, max_seqlen_q, nheads_q, d_v]
    out = jnp.zeros((B, max_seqlen_q, nheads_q, d_v), dtype=out_unpad.dtype)
    start_q = 0
    for b in range(B):
        q_len = int(seqlens_q[b])
        out = out.at[b, :q_len, :, :].set(out_unpad[start_q : start_q + q_len])
        start_q += q_len

    # dropout mask reconstruction
    dropout_mask = None
    if dropout_p > 0.0 and return_attn_probs and S_dmask is not None:
        S_thresh = ck_randval_to_dropout_mask(S_dmask, dropout_p)
        S_bhss = pad_rearrange_dropout_mask_hts_to_bhss(
            S_thresh,
            cu_seqlens_q,
            seqlen_q_rounded=max_seqlen_q,
            seqlen_k_rounded=max_seqlen_k,
        )
        S_conv = convert_flash_attn_S_to_softmax(
            S_bhss,
            max_seqlen_q,
            max_seqlen_k,
            query_padding_mask=query_padding_mask,
            key_padding_mask=key_padding_mask,
            head_dim=d_q,
            is_dropout=True,
            causal=causal,
            window_size=window_size,
        )
        dropout_mask = S_conv >= 0

    if dout is None:
        return out, dropout_mask, None, None, None

    # unpad dout
    dout_chunks = []
    for b in range(B):
        q_len = int(seqlens_q[b])
        dout_chunks.append(dout[b, :q_len, :, :])
    dout_unpad = jnp.concatenate(dout_chunks, axis=0)

    # flash_attn_varlen always returns 3 outputs → 3 cotangents
    ct_lse = jnp.zeros_like(softmax_lse) if softmax_lse is not None else None
    ct_S = jnp.zeros_like(S_dmask) if S_dmask is not None else None
    cot = (dout_unpad, ct_lse, ct_S)

    dq_unpad, dk_unpad, dv_unpad = pullback(cot)

    # pad grads back
    dq = jnp.zeros_like(q)
    dk = jnp.zeros_like(k)
    dv = jnp.zeros_like(v)

    start_q = 0
    start_k = 0
    for b in range(B):
        q_len = int(seqlens_q[b])
        k_len = int(seqlens_k[b])

        dq = dq.at[b, :q_len, :, :].set(dq_unpad[start_q : start_q + q_len])
        dk = dk.at[b, :k_len, :, :].set(dk_unpad[start_k : start_k + k_len])
        dv = dv.at[b, :k_len, :, :].set(dv_unpad[start_k : start_k + k_len])

        start_q += q_len
        start_k += k_len

    return out, dropout_mask, dq, dk, dv


@pytest.mark.parametrize("input_layout", ["BSHD", "KVPACKED"])
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
    input_layout,
):
    return_lse = True
    key = jax.random.PRNGKey(0)
    dtype = dtypes.to_jax_dtype(dtype)
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    assert nheads % nheads_k == 0

    # window
    if not local:
        window_size = (-1, -1)
    else:
        key, sub = jax.random.split(key)
        # torch.randint(0, seqlen_k, (2,))
        window_size = tuple(
            jax.random.randint(sub, (2,), 0, seqlen_k, dtype=jnp.int32).tolist()
        )

    key, sub = jax.random.split(key)
    q = jax.random.normal(
        sub,
        (batch_size, seqlen_q, nheads, d),
        dtype=dtype,
    )
    key, sub = jax.random.split(key)
    k = jax.random.normal(
        sub,
        (batch_size, seqlen_k, nheads_k, d),
        dtype=dtype,
    )
    key, sub = jax.random.split(key)
    v = jax.random.normal(
        sub,
        (batch_size, seqlen_k, nheads_k, d_v),
        dtype=dtype,
    )

    if bias_type == "bias":
        # TODO - We need to implement unpad bias [batch_size, seqlen_q, seqlen_k] -> [total_q, max_seqlen_k]
        # Let total_q = batch_size * seqlen_q to pass the test for now
        query_padding_mask = generate_random_padding_mask(
            seqlen_q, batch_size, key, mode="full"
        )
        key_padding_mask = generate_random_padding_mask(
            seqlen_k, batch_size, key, mode="full"
        )
    else:
        query_padding_mask = generate_random_padding_mask(
            seqlen_q, batch_size, key, mode="random"
        )
        key_padding_mask = generate_random_padding_mask(
            seqlen_k, batch_size, key, mode="random"
        )

    attn_bias = None
    alibi_slopes = None
    if bias_type == "bias":
        key, sub = jax.random.split(key)
        attn_bias = jax.random.normal(
            sub,
            (batch_size, seqlen_q, seqlen_k),
            dtype=dtype,
        )
    elif bias_type == "alibi":
        key, sub = jax.random.split(key)
        alibi_slopes = jax.random.uniform(
            sub,
            (batch_size, nheads),
            minval=0.0,
            maxval=1.0,
            dtype=dtypes.to_jax_dtype(dtypes.fp32),
        )

    key, sub = jax.random.split(key)
    dout = jax.random.normal(
        sub,
        (batch_size, seqlen_q, nheads, d_v),
        dtype=dtype,
    )

    # return_attn_probs is just for host verification (to produce same dropout mask)
    # no need to use in actual case
    if dropout_p > 0:
        return_attn_probs = True
    else:
        return_attn_probs = False

    # Returns from the aiter:
    # out_aiter, dropout_mask_aiter, dq_aiter, dk_aiter, dv_aiter
    jax_result = run_jax_aiter(
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
        cu_seqlens_q_padded=None,
        cu_seqlens_k_padded=None,
        input_layout=input_layout,
    )

    out_jax, dropout_mask_jax, dq_jax, dk_jax, dv_jax = jax_result

    # Helper for max |.| like .abs().max().item() in torch
    def max_abs(x):
        return float(jnp.max(jnp.abs(x)))

    # Run torch reference
    out_ref, dq_ref, dk_ref, dv_ref = run_jax(
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

    out_pt, dq_pt, dk_pt, dv_pt = run_jax(
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
    out_diff = max_abs(out_jax - out_ref)
    ref_diff = max_abs(out_pt - out_ref)
    print(f"JAX vs REF Output max diff: {out_diff}")
    print(f"PT vs REF Output max diff: {ref_diff}")

    # (Ruturaj4): Note: Forward output assertion commented out to match third_party test behavior.
    # The CK kernels can have larger deviations in BF16 + dropout + local cases.
    out_tol = max(4 * ref_diff, 0.01)
    assert (
        out_diff <= out_tol
    ), f"JAX output doesn't match REF: diff {out_diff} > tol {out_tol}"

    # TODO: Support varlen bwd for bias
    if bias_type == "bias":
        pytest.skip("Does not support varlen bwd for bias")

    if dq_jax is not None:
        # print(f"AITER dQ max diff: {(dq_aiter - dq_ref).abs().max().item()}")
        # print(f"AITER dK max diff: {(dk_aiter - dk_ref).abs().max().item()}")
        # print(f"AITER dV max diff: {(dv_aiter - dv_ref).abs().max().item()}")
        print(f"JAX dQ max diff: {max_abs(dq_jax - dq_ref)}")
        print(f"JAX dK max diff: {max_abs(dk_jax - dk_ref)}")
        print(f"JAX dV max diff: {max_abs(dv_jax - dv_ref)}")
        # print(f"JAX vs AITER dQ max diff: {(dq_jax - dq_aiter).abs().max().item()}")
        # print(f"JAX vs AITER dK max diff: {(dk_jax - dk_aiter).abs().max().item()}")
        # print(f"JAX vs AITER dV max diff: {(dv_jax - dv_aiter).abs().max().item()}")
        print(f"PT dQ max diff: {max_abs(dq_pt - dq_ref)}")
        print(f"PT dK max diff: {max_abs(dk_pt - dk_ref)}")
        print(f"PT dV max diff: {max_abs(dv_pt - dv_ref)}")

        dq_tol = max(10 * max_abs(dq_pt - dq_ref), 0.01)
        dk_tol = max(10 * max_abs(dk_pt - dk_ref), 0.01)
        dv_tol = max(10 * max_abs(dv_pt - dv_ref), 0.01)

        assert max_abs(dq_jax - dq_ref) <= dq_tol, "JAX dQ doesn't match REF"
        assert max_abs(dk_jax - dk_ref) <= dk_tol, "JAX dK doesn't match REF"
        assert max_abs(dv_jax - dv_ref) <= dv_tol, "JAX dV doesn't match REF"


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
    parser.add_argument(
        "-i",
        "--input_layout",
        type=str,
        choices=["BSHD", "KVPACKED"],
        default="BSHD",
        help="""input_layout.
        e.g.: -i BSHD""",
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
        args.input_layout,
    )
